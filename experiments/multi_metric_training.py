import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA 
from tools.utils import AverageMeter, compute_loss, accuracy, check_sum

### QGS-H ###
def QGS_H_train(epoch, model, train_loader, optimizer, scheduler, criterions, loss_name_list, T_list, S, t, num_classes, print_frep=100):
    """
    One epoch of QGS-H training process

    criterions: loss functions for training
    loss_name_list: loss function names
    T_list: target values for loss functions
    S: slack variables
    t:homotopy parameter
    num_classes: number of classes for classifiction

    Note: the first criterion is the dominant criterion
    """
    num_objectives = len(criterions)
    if num_objectives != len(loss_name_list):
        raise Exception('wrong loss_name_list!')
    obj_loss_meters = [AverageMeter(name=loss_name) for _, loss_name in enumerate(loss_name_list)]
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    energy_meter = AverageMeter()

    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        obj_losses = [compute_loss(loss_name_list[i], criterions[i], outputs, target, num_classes=num_classes) for i in range(num_objectives)]
        H = [obj_losses[i].item() - T_list[i] + torch.abs(S[i]) for i in range(num_objectives)]
        energy = 0.5 * (1 - t + t / H[0].item()) * H[0].item() ** 2
        for i in range(1, num_objectives):
            energy += 0.5 * (1 - t) * H[i].detach() ** 2

        # QGS loss function
        loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
        for i in range(1, num_objectives):
            loss += H[i].item() * obj_losses[i] + 0.5 * H[i] ** 2
        QGS_loss = (1 - t) * loss + t * obj_losses[0] # Apply Homotopy
        # backward
        optimizer.zero_grad()
        QGS_loss.backward()
        optimizer.step()

        # recording data
        for i in range(num_objectives):
            obj_loss_meters[i].update(obj_losses[i].item(), len(target))
        train_loss_meter.update(QGS_loss.item(), len(target))
        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))
        energy_meter.update(energy, len(target))

        if i % print_frep == 0:
            print('step: {}, train loss: {:.4f} ({:.4f}), train accuracy: {:.3f}({:.3f}), lr: {:.4f}'.format(
                i, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg, optimizer.param_groups[0]['lr']))

    scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    for i in range(num_objectives):
        log += '{}: {:.4f}, '.format(loss_name_list[i], obj_loss_meters[i].avg)
    log += 'loss: {:.4f}, '.format(train_loss_meter.avg)
    log += 'energy: {:.3f}, accuracy: {:.3f}'.format(energy_meter.avg, accuracy_meter.avg)
    print(log)

    return obj_loss_meters, train_loss_meter.avg, accuracy_meter.avg, energy_meter.avg

def validate(model, test_loader, criterions, loss_name_list, num_classes, maximum=float('inf')):
    """
    Validate trained model, return its test losses and test accuracy.
    """
    num_objectives = len(criterions)
    
    obj_loss_meters = [AverageMeter(name=loss_name) for _, loss_name in enumerate(loss_name_list)]
    accuracy_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            if batch > maximum:
                break
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            outputs = model(data)
            obj_losses = [compute_loss(loss_name_list[i], criterions[i], outputs, target, num_classes=num_classes) for i in range(num_objectives)]
            # print(obj_losses)
            for i in range(num_objectives):
                obj_loss_meters[i].update(obj_losses[i].item(), len(target))
        
            acc = accuracy(outputs.data, target)[0]
            accuracy_meter.update(acc.item(), len(target))

    return obj_loss_meters, accuracy_meter.avg

### Largrangian training ###
def Lagrangian_train(epoch, model, train_loader, optimizer, scheduler, criterions, loss_name_list, warmup, l, lam, T_list, num_classes,
    warmup_QGS=False, S=None, print_frep=100, warmup_T_list=None, last_record=0):
    """
    One epoch of Larangian training process (minmax problem: min over w and max over lambda)

    warmup: number of epochs of training without constraints
    l: number of steps of updating w per step of updating lambda
    lam: lambda variables for constraints
    T_list: list of targets of different metrics 
    """
    num_objectives = len(criterions)
    if num_objectives != len(loss_name_list):
        raise Exception('wrong loss_name_list!')
    obj_loss_meters = [AverageMeter(name=loss_name) for _, loss_name in enumerate(loss_name_list)]
    accuracy_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    model.train()
    warmup_stage = epoch < warmup or (epoch - last_record) < warmup
    maximize = False
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        obj_losses = [compute_loss(loss_name_list[i], criterions[i], outputs, target, num_classes=num_classes) for i in range(num_objectives)]

        for k in range(num_objectives):
            obj_loss_meters[k].update(obj_losses[k].item(), len(target))

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))

        if warmup_stage:
            if not warmup_QGS:
                loss = obj_losses[0]
                train_loss_meter.update(loss.item(), len(target))
            else:
                H = [obj_losses[j].item() - warmup_T_list[j] + torch.abs(S[j]) for j in range(num_objectives)]
                loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
                for j in range(1, num_objectives):
                    loss += H[j].item() * obj_losses[j] + 0.5 * H[j] ** 2
                train_loss_meter.update(loss.item(), len(target))
        else:
            if i % (l + 1) == 0:
                # maximize over lam (half of the minimization learning lrate)
                maximize = True
                lr = optimizer.param_groups[0]['lr'] * 0.5
                for j in range(num_objectives):
                    lam[j] += lr * F.relu(obj_losses[j] - T_list[j]).item()
            else:
                # minimize over w
                loss = obj_losses[0]
                for j in range(num_objectives):
                    loss += F.relu(obj_losses[j] - T_list[j]) * lam[j].item()
                train_loss_meter.update(loss.item(), len(target))
                maximize = False

        # backward
        if not maximize:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % print_frep == 0:
            print('step: {}, train loss: {:.4f} ({:.4f}), train accuracy: {:.3f}({:.3f}), lr: {:.4f}, warmup: {}'.format(
                i, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg, optimizer.param_groups[0]['lr'], warmup_stage))
    # change lr after each epoch
    scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    for i in range(num_objectives):
        log += '{}: {:.4f}, '.format(loss_name_list[i], obj_loss_meters[i].avg)
    log += 'loss: {:.4f}, '.format(train_loss_meter.avg)
    log += 'accuracy: {:.3f}'.format(accuracy_meter.avg)
    print(log)

    return obj_loss_meters, train_loss_meter.avg, accuracy_meter.avg

### Single-metric training ###
def regular_train(epoch, model, train_loader, optimizer, scheduler, criterion, loss_name, num_classes, print_frep=100):

    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        outputs = model(data)
        loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), len(target))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_meter.update(loss.item(), len(target))

        if i % print_frep == 0:
            print('step: {}, {} loss: {:.4f} ({:.4f}), train accuracy: {:.3f}({:.3f}), lr: {:.4f}'.format(
                i, loss_name, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg, optimizer.param_groups[0]['lr']))
    if scheduler:
        scheduler.step()
    log = 'epoch: {}, '.format(epoch)
    log += '{} loss: {:.4f}, '.format(loss_name, train_loss_meter.avg)
    log += 'accuracy: {:.3f}'.format(accuracy_meter.avg)
    print(log)

    return train_loss_meter.avg, accuracy_meter.avg
    




    