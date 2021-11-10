import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA 
import torch.nn.utils.prune as prune
from tools.utils import AverageMeter, compute_loss, accuracy, check_sum

import os
### QGS-complexity simple demo ###
def QGS_complexity_train(epoch, model, train_loader, optimizer, scheduler, criterion, loss_name, warmup, l, lam, T_list, num_classes, S=None, 
    print_frep=100, last_record=0):

    task_loss_meter = AverageMeter(name='Task')
    linear1_meter = AverageMeter(name='linear1')
    linear2_meter = AverageMeter(name='linear2')
    linear3_meter = AverageMeter(name='linear3')
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    warmup_stage = epoch < warmup or (epoch - last_record) < warmup
    num_objectives = 4
    maximize = False
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        task_loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)

        # complexity loss
        all_linear1_params = torch.cat([x.view(-1) for x in model.linear1.parameters()])
        all_linear2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])
        all_linear3_params = torch.cat([x.view(-1) for x in model.linear3.parameters()])

        linear1_loss = torch.norm(all_linear1_params, 1) * 1e-3
        linear2_loss = torch.norm(all_linear2_params, 1) * 1e-3
        linear3_loss = torch.norm(all_linear3_params, 1) * 1e-3
        obj_losses = [task_loss, linear1_loss, linear2_loss, linear3_loss]

        if warmup_stage:
            H = [obj_losses[j].item() - T_list[j] + torch.abs(S[j]) for j in range(num_objectives)]
            loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
            for j in range(1, num_objectives):
                loss += H[j].item() * obj_losses[j] + 0.5 * H[j] ** 2
            train_loss_meter.update(loss.item(), len(target))
            task_loss_meter.update(task_loss.item(), len(target))
            linear1_meter.update(linear1_loss.item(), len(target))
            linear2_meter.update(linear2_loss.item(), len(target))
            linear3_meter.update(linear3_loss.item(), len(target))
        else:
            if i % (l + 1) == 0:
                # maximize over lam (half of the minimization learning lrate)
                maximize = True
                lr = optimizer.param_groups[0]['lr'] * 0.1
                for j in range(num_objectives):
                    lam[j] += lr * F.relu(obj_losses[j] - T_list[j]).item()
            else:
                # minimize over w
                loss = obj_losses[0]
                for j in range(num_objectives):
                    loss += F.relu(obj_losses[j] - T_list[j]) * lam[j].item()
                maximize = False
                train_loss_meter.update(loss.item(), len(target))
                task_loss_meter.update(task_loss.item(), len(target))
                linear1_meter.update(linear1_loss.item(), len(target))
                linear2_meter.update(linear2_loss.item(), len(target))
                linear3_meter.update(linear3_loss.item(), len(target))

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))
        
        # backward
        if not maximize:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if i % print_frep == 0:
            print('step: {}, warmup: {}, {} loss: {:.4f} ({:.4f}), train loss: {:.4f} ({:.4f}), train accuracy: {:.3f}({:.3f}),' 
                'linear1: {:.4f}, linear2: {:.4f}, linear3: {:.4f}'.format(i, warmup_stage, loss_name, task_loss_meter.val, 
                task_loss_meter.avg, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg, 
                linear1_meter.avg, linear2_meter.avg, linear2_meter.avg))

    scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    log += '{} loss: {:.4f}, '.format(loss_name, task_loss_meter.avg)
    log += 'linear1 l1-norm: {:.4f}, '.format(linear1_meter.avg)
    log += 'linear2 l1-norm: {:.4f}, '.format(linear2_meter.avg)
    log += 'linear3 l1-norm: {:.4f}, '.format(linear3_meter.avg)
    log += 'accuracy: {:.3f}, '.format(accuracy_meter.avg)
    log += 'lr: {:.3f}'.format(optimizer.param_groups[0]['lr'])
    print(log)

    return [task_loss_meter, linear1_meter, linear2_meter, linear3_meter], task_loss_meter.avg, accuracy_meter.avg

### QGS-pruning CIFAR demo ###
def compute_layer_norm(model, name=None, order=1, b=1, a=1, c=0, special_norm_type=None):
    """
    Compute parameter-related loss. 
    sigle_mode norm: y = -ax^2 + b or y = -a|x| + b, e.g y=-x^2 + 1, y=-|x| + 1
    double_mode_norm: y = -ax^2 +b|x| + c or y = -a||x|-b| + c, e.g y=-x^2 + |x|, y=-||x| - 0.5| + 0.5
    w norm: y = |-|a|x|-b| + c|
    """
    norm = 0
    if not special_norm_type:
        for module_name, module in model.named_modules():
            if name is not None and name not in module_name:
                continue
            try:
                norm += torch.norm(module.weight.view(-1), order)
            except Exception as e:
                continue
    elif special_norm_type == 'single_mode':
        for module_name, module in model.named_modules():
            if name is not None and name not in module_name:
                continue
            try:
                if order == 2:
                    norm += torch.sum(-a * torch.square(module.weight.view(-1)) + b)
                else:
                    norm += torch.sum(-a * torch.abs(module.weight.view(-1)) + b)
            except Exception as e:
                continue
    elif special_norm_type == 'double_mode':
        for module_name, module in model.named_modules():
            if name is not None and name not in module_name:
                continue
            try:
                weight = module.weight.view(-1)
                if order == 2:
                    norm += torch.sum(-a * torch.square(weight) + b * torch.abs(weight) + c)
                else:
                    norm += torch.sum(-a * torch.abs(torch.abs(weight) - b) + c)
            except Exception as e:
                continue
    elif special_norm_type == 'w':
        # w norm: y = |-|a|x|-b| + c|
        for module_name, module in model.named_modules():
            if name is not None and name not in module_name:
                continue
            try:
                weight = module.weight.view(-1)
                norm += torch.sum(torch.abs(-torch.abs(a * torch.abs(weight) - b) + c))
            except Exception as e:
                continue
        
    return norm


def QGS_pruning_train(epoch, model, train_loader, optimizer, scheduler, criterion, loss_name, warmup, l, lam, T_list, num_classes, S=None, 
    print_frep=100, last_record=0, order=1, k=1e-3, blocks=['layer1', 'layer2', 'layer3'], QGS_warmup=True, special_norm_type=None, b=1, a=1, c=0):
    """
    One epoch for QGS-pruning

    T_list: target list whose first entry is the target of task-specific loss, while the others are targets of module complexity losses
    order: order of norm of model weight complexity
    k: constant for balancing the complexity and task loss
    """
    assert len(blocks) == len(T_list) - 1 # T_list include targets of task loss and all the architecture losses
    num_objectives = len(T_list)
    task_loss_meter = AverageMeter(name='Task')
    complexity_loss_meters = [AverageMeter(name=block) for block in blocks]
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    warmup_stage = epoch < warmup or (epoch - last_record) < warmup
    maximize = False
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        task_loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)
        obj_losses = [task_loss]
        # complexity loss
        for block_name in blocks:
            norm = compute_layer_norm(model, block_name, order=order, b=b, a=a, c=c, special_norm_type=special_norm_type)
            obj_losses.append(norm * k)

        if warmup_stage:
            if QGS_warmup:
                H = [obj_losses[j].item() - T_list[j] + torch.abs(S[j]) for j in range(num_objectives)]
                loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
                for j in range(1, num_objectives):
                    loss += H[j].item() * obj_losses[j] + 0.5 * H[j] ** 2
            else:
                # unconstrained warmup
                loss = obj_losses[0]
        else:
            if T_list[0] != 0:
                if i % (l + 1) == 0:
                    # maximize over lam (half of the minimization learning lrate)
                    maximize = True
                    lr = optimizer.param_groups[0]['lr'] * 0.1
                    for j in range(1, num_objectives):
                        lam[j] += lr * F.relu(obj_losses[j] - T_list[j]).item()
                else:
                    # minimize over w
                    loss = obj_losses[0]
                    for j in range(1, num_objectives):
                        loss += F.relu(obj_losses[j] - T_list[j]) * lam[j].item()
                    maximize = False
            else:
                # the network has already been pruned, finetune the model without constraints
                loss = obj_losses[0]
                maximize = False
        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))

        # backward
        if not maximize:
            train_loss_meter.update(loss.item(), len(target))
            task_loss_meter.update(task_loss.item(), len(target))
            for index in range(1, num_objectives):
                complexity_loss_meters[index - 1].update(obj_losses[index].item(), len(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if i % print_frep == 0:
            log = 'step: {}, warmup: {}, {} loss: {:.4f} ({:.4f}), train loss: {:.4f} ({:.4f}), train accuracy: {:.3f}({:.3f}), '.format(
                i, warmup_stage, loss_name, task_loss_meter.val, 
                task_loss_meter.avg, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg
            )
            for index, block_name in enumerate(blocks):
                log += '{}: {:.4f}, '.format(block_name, complexity_loss_meters[index].avg)
            print(log)

    scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    log += '{} loss: {:.4f}, '.format(loss_name, task_loss_meter.avg)
    for index, block_name in enumerate(blocks):
        log += '{}: {:.4f}, '.format(block_name, complexity_loss_meters[index].avg)
    log += 'accuracy: {:.3f}, '.format(accuracy_meter.avg)
    log += 'lr: {:.3f}'.format(optimizer.param_groups[0]['lr'])
    print(log)

    return task_loss_meter, complexity_loss_meters, train_loss_meter.avg, accuracy_meter.avg

def baseline_pruning_train(epoch, model, train_loader, optimizer, scheduler, criterion, loss_name, num_classes, finetune, prune_alpha,
    print_frep=100, order=1, k=1e-3, structured=False, harder_prune=0):
    """
    One epoch for baseline pruning, regular training while keeping track of network complexities
    k: constant for balancing the complexity and task loss
    """
    num_objectives = 2
    task_loss_meter = AverageMeter(name='Task')
    distance_meter = AverageMeter(name='Distance')
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        task_loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)
        loss = task_loss
        # complexity loss
        if not structured:
            prune_method = prune.L1Unstructured(amount=prune_alpha + harder_prune)
        else:
            prune_method = prune.LnStructured(amount=prune_alpha + harder_prune, n=order)
        distance = compute_pruned_distance(model, prune_method, structured, order=order, k=k)

        train_loss_meter.update(loss.item(), len(target))
        task_loss_meter.update(task_loss.item(), len(target))
        distance_meter.update(distance.item(), len(target))

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_frep == 0:
            log = 'step: {}, loss(original): {:.4f} ({:.4f}), distance: {:.4f}, train loss: {:.4f} ({:.4f}), train accuracy(original): {:.3f}({:.3f}), '.format(
                i, task_loss_meter.val, task_loss_meter.avg, distance_meter.avg, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg
            )
            print(log)

    scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    log += '{} loss: {:.4f}, distance: {:.4f}, '.format(loss_name, task_loss_meter.avg, distance_meter.avg)
    log += 'accuracy: {:.3f}, '.format(accuracy_meter.avg)
    log += 'finetune: {}, '.format(finetune)
    log += 'lr: {:.5f}'.format(optimizer.param_groups[0]['lr'])

    print(log)

    return task_loss_meter, distance_meter, train_loss_meter.avg, accuracy_meter.avg

def compute_pruned_distance(model, prune_method, structured, order=2, k=1e-3, anchor_path=None):
    """
    Compute the distance between the original and the pruned networks
    """
    distance = 0
    if not anchor_path or not os.path.isfile(anchor_path):
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                pruned = prune_method.prune(module.weight)
                distance += torch.norm((pruned - module.weight).view(-1), order)
    else:
        anchor_state_dict = torch.load(anchor_path)['state_dict']
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                key = name + '.weight'
                anchor_weight = anchor_state_dict[key]
                distance += torch.norm((anchor_weight - module.weight).view(-1), order)

    return distance * k

def importance_distance(model, pruned_method, k=1e-3):
    distance = 0:
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            gradient = module.weight.grad
            pruned = pruned_method.prune(module.weight)



def QGS_soft_pruning_train(epoch, model, train_loader, optimizer, scheduler, criterion, loss_name, T_list, num_classes, warmup=10, 
    l=0, lam=None, last_record=0, S=None, print_frep=100, prune_alpha=0.5, t=0, structured=False, order=1, add_window=False, finetune=False, 
    k=1e-3, pretrain_method='QGS', QGS_lr=False, QGS_lr_constant=0.05, QGS_lr_max=0.1):
    """
    One epoch for soft QGS-pruning: W is the original large network, \theta is a pruned netowrk.
    By default, \theta is a pruned version of W by l-1 norm. The objectives are task-specific loss of W, 
    task-specific loss of \theta and distance between W and \theta.

    add_window: when return to the original network whether to use a custom window as a mask
    finetune: if true, the input model is a pruned network with target sparsity
    """
    assert pretrain_method in ['QGS', 'QGS-H', 'QGS-L']
    original_loss_meter = AverageMeter(name='Task')
    # pruned_loss_meter = AverageMeter(name='Prune')
    distance_meter = AverageMeter(name='Distance')
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    # pruned_accuracy_meter = AverageMeter()
    warmup_stage = epoch < warmup or (epoch - last_record) < warmup
    maximize = False
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        bs = len(target)
        outputs = model(data)
        original_task_loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), bs)

        if not finetune:
            if not structured:
                prune_method = prune.L1Unstructured(name='weight', amount=prune_alpha)
            else:
                prune_method = prune.LnStructured(name='weight', amount=prune_alpha, n=order, dim=1)
            distance = compute_pruned_distance(model, prune_method, structured, order=order, k=k)
            obj_losses = [original_task_loss, distance]
            num_objectives = len(obj_losses)
            # QGS
            H = [obj_losses[j].item() - T_list[j] + torch.abs(S[j]) for j in range(num_objectives)]
            loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
            for j in range(1, num_objectives):
                loss += H[j].item() * obj_losses[j] + 0.5 * H[j] ** 2

            # QGS-H
            if pretrain_method == 'QGS-H':
                loss = (1 - t) * loss + t * obj_losses[1] # transfer from QGS to pruning
            # QGS Lagrangian
            elif pretrain_method == 'QGS-L' and not warmup_stage:
                if i % (l + 1) == 0:
                    # maximize over lam (half of the minimization learning lrate)
                    maximize = True
                    lr = optimizer.param_groups[0]['lr'] * 0.1
                    for j in range(1, num_objectives):
                        lam[j] += lr * F.relu(obj_losses[j] - T_list[j]).item()
                else:
                    # minimize over w
                    loss = obj_losses[0].clone()
                    # print('before: {}, original task loss: {}'.format(loss.item(), original_task_loss.item()))
                    # print(lam)
                    for j in range(1, num_objectives):
                        loss += F.relu(obj_losses[j] - T_list[j]) * lam[j].item()
                    # print('after: {}'.format(loss.item()))
                    maximize = False
        else:
            loss = original_task_loss
        # print(loss.item(), original_task_loss.item())
        if not maximize:
            train_loss_meter.update(loss.item(), bs)
            original_loss_meter.update(original_task_loss.item(), bs)
            if not finetune:
                distance_meter.update(distance.item(), bs)
            else:
                distance_meter.update(0, bs)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if i % print_frep == 0:
            # log = 'step: {}, loss(original): {:.4f} ({:.4f}), loss(pruned): {:.4f} ({:.4f}), distance: {:.4f}, train loss: {:.4f} ({:.4f}), train accuracy(original): {:.3f}({:.3f}), train accuracy(pruned): {:.3f}({:.3f}), '.format(
            #     i, original_loss_meter.val, original_loss_meter.avg, pruned_loss_meter.val, pruned_loss_meter.avg, distance_meter.avg, 
            #     train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg, pruned_accuracy_meter.val, pruned_accuracy_meter.avg
            # )
            log = 'step: {}, loss(original): {:.4f} ({:.4f}), distance: {:.4f}, train loss: {:.4f} ({:.4f}), train accuracy(original): {:.3f}({:.3f}), lr: {:.5f}'.format(
                i, original_loss_meter.val, original_loss_meter.avg, distance_meter.avg, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg,
                optimizer.param_groups[0]['lr']
            )
            print(log)
    if not QGS_lr:
        scheduler.step()
    else:
        # Update the learning rate based on QGS theory
        divider = 2 * (original_loss_meter.avg + distance_meter.avg) - sum(T_list) + torch.abs(S[0]).data + torch.abs(S[1]).data
        q_lr = QGS_lr_constant / divider
        q_lr = min(q_lr, QGS_lr_max)
        for g in optimizer.param_groups:
            g['lr'] = q_lr
    log = 'epoch: {}, '.format(epoch)
    # log += '{} loss (original): {:.4f}, {} loss (pruned): {:.4f}, '.format(loss_name, original_loss_meter.avg, loss_name, pruned_loss_meter.avg)
    log += '{} loss (original): {:.4f}, distance: {:.4f}, '.format(loss_name, original_loss_meter.avg, distance_meter.avg)
    log += 'train loss: {:.4f}, '.format(train_loss_meter.avg)
    log += 'accuracy: {:.3f}, '.format(accuracy_meter.avg)
    if pretrain_method == 'QGS-H':
        log += 't: {:.3f}, '.format(t)
    elif pretrain_method == 'QGS-L':
        log += 'lam: {}, '.format(lam)
    log += 'finetune: {}, '.format(finetune)
    log += 'lr: {:.5f}'.format(optimizer.param_groups[0]['lr'])
    print(log)

    # return [pruned_loss_meter, original_loss_meter, distance_meter], train_loss_meter.avg, accuracy_meter.avg, pruned_accuracy_meter.avg
    return [original_loss_meter, distance_meter], train_loss_meter.avg, accuracy_meter.avg
