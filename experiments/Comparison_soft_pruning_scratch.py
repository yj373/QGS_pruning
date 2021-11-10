import os, sys, argparse, shutil
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from torchsummary import summary
from collections import OrderedDict, defaultdict
from data.prepare import get_dataset, get_loaders
from tools.utils import check_sum, setup_seed, get_criterions, get_lr_scheduler, get_model, check_sparsity, compute_loss, AverageMeter, accuracy
from tools.visualization import plot_multilines, plot_param_distribution
from multi_metric_training import validate, regular_train
from pruning.pruning_training import baseline_pruning_train, compute_pruned_distance
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser =  argparse.ArgumentParser(description='QGS_Lagrangian + pruning')
parser.add_argument('--model', default='resnet', type=str, help='network architecture')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--pretrain_lr', default=0.01, type=float, help='initial learning rate for pretraining stage')
parser.add_argument('--finetune_lr', default=0.01, type=float, help='initial learning rate for finetuning stage')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
parser.add_argument('--prune_alpha', default=0.5, type=float, help='once the targets are reached, prune the weights by prune_alpha')
parser.add_argument('--prune_order', default=1, type=int, help='order of pruning criterion')
parser.add_argument('--distance_order', default=2, type=int, help='order of distance loss')
parser.add_argument('--structured', type=bool, help='whether to use structured pruning')
parser.add_argument('--init_model', default='', type=str, help='path to the initial model')
parser.add_argument('--pretrain', default=160, type=int, help='epochs of training before hard pruning')
parser.add_argument('--finetune', default=60, type=int, help='epochs fo finetuning the pruned network')
parser.add_argument('--k', default=1e-3, type=float, help='constant for distance')
parser.add_argument('--pretrain_method', default='QGS', type=str, help='pretraining methods: QGS, soft_pruning')
parser.add_argument('--ce_target', default=0.0, type=float, help='target for Cross Entropy loss')
parser.add_argument('--distance_target', default=-1.0, type=float, help='target for distance')
parser.add_argument('--QGS_lr_constant', default=0.05, type=float, help='QGS lr constant')
parser.add_argument('--QGS_lr_max', default=0.1, type=float, help='maximum QGS_lr learning rate')
parser.add_argument('--finetune_best', type=bool, help='whether to finutune the model with the best soft-pruning training accuracy')

parser.add_argument('--soft_prune_cycle', default=10, type=int, help='soft pruning every n epochs in prune pretraining')
parser.add_argument('--soft_prune_start', default=0, type=int, help='soft pruning start after n epochs of pretraining')
parser.add_argument('--manual_pretrain_lr', type=bool, help='whether to use traditional learning rate')
parser.add_argument('--harder_prune', default=0.0, type=float, help='prune more parameters for the computation of distance')
parser.add_argument('--prune_dim',default=0)

parser.add_argument('--max_factor', default=0.1, type=float, help='maximum lr factor for manual lr tuning')
parser.add_argument('--min_factor', default=0.01, type=float, help='minimum lr factor for manual lr tuning')
parser.add_argument('--lr_warmup', default=50, type=int, help='number of epochs of warmup for manual lr tuning')
parser.add_argument('--k_warmup', default=-1, type=int, help='epochs for k warmup')
parser.add_argument('--framework', default='soft_pruning', type=str, help='soft pruning or DPF')
best_acc = 0
best_pretrain_acc = 0

def main():
    global args,  best_acc
    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    setup_seed(args.seed)
    train_set, test_set = get_dataset(name=args.dataset, target_type=torch.LongTensor, target_shape=(-1, 1), model=args.model)
    train_loader, _, test_loader = get_loaders(train_set, test_set, args.batch_size, args.num_workers)
        
    model = get_model(args.model, num_classes)
    model = torch.nn.DataParallel(model).cuda()
    reparam = False # record the state of reparameterization of the current model

    if len(args.init_model) != 0 and os.path.isfile(args.init_model):
        print('load the initial model from {}'.format(args.init_model))
        model_dict = torch.load(args.init_model)
        model.load_state_dict(model_dict['state_dict'])
        sparsity = check_sparsity(model, structured=args.structured)
    else:
        sparsity = check_sparsity(model, structured=args.structured)
        torch.save(
            {
                'epoch': 0,
                'state_dict': model.state_dict(),
                'sparsity': sparsity,
                'reparam': reparam,
                'structured': args.structured
            }, os.path.join(args.savedir, 'initial.pt')
        )
    print('model sum: ', check_sum(model))
    training_parameters = list(model.parameters())
    if args.pretrain_method == 'QGS':
        S = torch.ones(2, dtype=torch.float32, device='cuda') * 0.01
        S.requires_grad = True
        training_parameters += [S]
    optimizer = torch.optim.SGD(training_parameters, lr=args.pretrain_lr, weight_decay=args.weight_decay, momentum=0.9)
    print('initialize learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    k = args.k
    print('k: {:.5f}'.format(k))
    if args.pretrain_method == 'QGS':
        T_list = [args.ce_target, args.distance_target] # targets for L(W) and distance
        print('Targets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('original loss: {}, distance: {}'.format(T_list[0], T_list[1]))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    train_accuracy_data = []
    test_accracy_data = []

    print("pretraining stage with method: {} >>>>>>>>>>>>>>>>>>>>>>>".format(args.pretrain_method))
    if args.pretrain_method == 'QGS':
        # print('before get scheduler: {}'.format(optimizer.param_groups[0]['lr']))
        if args.manual_pretrain_lr:
            pretrain_scheduler = get_lr_scheduler(optimizer, mode='cosine', total_steps=args.pretrain, warmup_steps=args.lr_warmup, max_factor=1, min_factor=1e-2)
        else:
            pretrain_scheduler = None
        # print('after get scheduler: {}'.format(optimizer.param_groups[0]['lr']))
    else:
        pretrain_scheduler = get_lr_scheduler(optimizer, total_steps=args.pretrain, mode='cosine', warmup_steps=-1, 
            max_factor=1, min_factor=1e-2)

    for epoch in range(args.pretrain):
        if args.pretrain_method == 'QGS':
            if args.k_warmup > 0:
                k = min(args.k * epoch / args.k_warmup, args.k) if sparsity < args.prune_alpha else 0
            elif sparsity >= args.prune_alpha:
                k = 0
            _, _, train_accuracy = QGS_pruning_train(epoch, model, train_loader, optimizer, criterions[0], loss_name_list[0], T_list, num_classes, 
                S=S, print_frep=100, prune_alpha=args.prune_alpha, structured=args.structured, prune_order=args.prune_order, distance_order=args.distance_order, 
                k=k, QGS_lr_constant=args.QGS_lr_constant, QGS_lr_max=args.QGS_lr_max, lr_scheduler=pretrain_scheduler, harder_prune=args.harder_prune)
        else:
            _, _, _, train_accuracy = baseline_pruning_train(epoch, model, train_loader, optimizer, pretrain_scheduler, criterions[0], loss_name_list[0], 
            num_classes, finetune=True, prune_alpha= args.prune_alpha, print_frep=100, order=args.distance_order, k=k)

        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        test_accracy_data.append(test_accuracy)
        print('test accuracy: {:.3f}'.format(test_accuracy))

        sparsity = check_sparsity(model, structured=args.structured)
        train_accuracy_data.append(train_accuracy)

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'sparsity': sparsity,
                'reparam': reparam,
                'structured': args.structured
            }, os.path.join(args.savedir, 'pretrain_final_model.pt')
        )

        labels = ['train', 'test']
        data_source = [train_accuracy_data, test_accracy_data]
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')

        # soft pruning after each epoch
        if epoch >= args.soft_prune_start and (epoch - args.soft_prune_start) % args.soft_prune_cycle == 0:
            if args.framework == 'soft_pruning':
                soft_pruning(model, args.prune_alpha, train_loader, test_loader, criterions, loss_name_list, num_classes, epoch, 
                    savedir=args.savedir, structured=args.structured, path1=os.path.join(args.savedir, 'temp.pt'), recover=False, prune_order=args.prune_order, 
                    finetune_best=args.finetune_best)
                reparam = False
            else:
                dynamic_pruning(model, args.prune_alpha, train_loader, test_loader, criterions, loss_name_list, num_classes, epoch, structured=args.structured, 
                    prune_order=args.prune_order, finetune_best=args.finetune_best, savedir=args.savedir, 
                    prune_name='weight')
                reparam = True

    print('Hard pruning: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if args.finetune_best:
        model_path = os.path.join(args.savedir, 'best_sf_pretrain.pt') if args.framework == 'soft_pruning' else os.path.join(args.savedir, 'best_dpf_pretrain.pt')
        model_dict = torch.load(os.path.join(args.savedir, 'best_sf_pretrain.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('load best pretraining pruned model saved at {}'.format(model_dict['epoch']))
        reparam = model_dict['reparam']
        # print('load best pretrained model with model sum: ', check_sum(model))
    if not reparam:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if not args.structured:
                    prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                else:
                    prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=args.prune_order, dim=0)
     
        reparam = True
    sparsity = check_sparsity(model, structured=args.structured)
    test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
    print('pruned model test accuracy before finetuning: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
        test_accuracy, test_meter[0].avg, train_accuracy, train_meter[0].avg))

    print("finetuning stage (finetune best: {}) >>>>>>>>>>>>>>>>>>>>>>>".format(args.finetune_best))
    scheduler = get_lr_scheduler(optimizer, total_steps=args.finetune, mode='cosine', warmup_steps=-1, max_factor=1, min_factor=0.1)
    for g in optimizer.param_groups:
        g['lr'] = args.finetune_lr # reset the learning rate, by default 0.01
    for epoch in range(args.pretrain, args.pretrain + args.finetune):
        _, train_accuracy = regular_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], num_classes, print_frep=100)

        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        test_accracy_data.append(test_accuracy)
        print('test accuracy: {:.3f}'.format(test_accuracy))

        sparsity = check_sparsity(model, structured=args.structured)
        train_accuracy_data.append(train_accuracy)

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'sparsity': sparsity,
                'reparam': reparam,
                'structured': args.structured
            }, os.path.join(args.savedir, 'finetuned_pruned_final_model.pt')
        )
        is_best = test_accuracy > best_acc
        if is_best:
            best_acc = test_accuracy
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'sparsity': sparsity,
                    'reparam': reparam,
                    'structured': args.structured
                }, os.path.join(args.savedir, 'finetuned_pruned_best_model.pt')
            )

        labels = ['train', 'test']
        data_source = [train_accuracy_data, test_accracy_data]
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')
       
    # testing
    model_path = os.path.join(args.savedir, 'finetuned_pruned_final_model.pt')
    if os.path.isfile(model_path):
        print('start testing final pruned model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        model_dict_best = torch.load(model_path)
        sparsity = model_dict_best['sparsity']
        target_reparam = model_dict_best['reparam']
        structured = model_dict_best['structured']
        if not target_reparam and reparam:
            for _, module in model.named_modules():
                try:
                    prune.remove(module, name='weight')
                except:
                    continue
            reparam = False
        elif target_reparam and not reparam:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                    else:
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=args.prune_order, dim=0)
            reparam = True
        model.load_state_dict(model_dict_best['state_dict'])
        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        print('final pruned model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, 
            model_dict_best['epoch'], sparsity, structured))

    model_path = os.path.join(args.savedir, 'finetuned_pruned_best_model.pt')
    if os.path.isfile(model_path):
        print('start testing best pruned model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
        model_dict_best = torch.load(model_path)
        sparsity = model_dict_best['sparsity']
        target_reparam = model_dict_best['reparam']
        structured = model_dict_best['structured']
        if not target_reparam and reparam:
            for _, module in model.named_modules():
                try:
                    prune.remove(module, name='weight')
                except:
                    continue
            reparam = False
        elif target_reparam and not reparam:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                    else:
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=args.prune_order, dim=0)
            reparam = True
        model.load_state_dict(model_dict_best['state_dict'])
        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        print('best pruned model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, 
            model_dict_best['epoch'], sparsity, structured))


def QGS_pruning_train(epoch, model, train_loader, optimizer, criterion, loss_name, T_list, num_classes, 
    S=None, print_frep=100, prune_alpha=0.5, structured=False, prune_order=1, distance_order=2, k=1e-3, QGS_lr_constant=0.05, QGS_lr_max=0.1,
    lr_scheduler=None, harder_prune=0.0):
    original_loss_meter = AverageMeter(name='Task')
    distance_meter = AverageMeter(name='Distance')
    train_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        bs = len(target)
        outputs = model(data)
        original_task_loss = compute_loss(loss_name, criterion, outputs, target, num_classes=num_classes)

        acc = accuracy(outputs.data, target)[0]
        accuracy_meter.update(acc.item(), bs)

        if not structured:
            prune_method = prune.L1Unstructured(amount=prune_alpha + harder_prune)
        else:
            prune_method = prune.LnStructured(amount=prune_alpha + harder_prune, n=prune_order, dim=0)

        distance = compute_pruned_distance(model, prune_method, structured, order=distance_order, k=k)
        obj_losses = [original_task_loss, distance]
        num_objectives = len(obj_losses)
        H = [obj_losses[j].item() - T_list[j] + torch.abs(S[j]) for j in range(num_objectives)]
        loss = H[0].item() * obj_losses[0] + 0.5 * H[0] ** 2
        for j in range(1, num_objectives):
            loss += H[j].item() * obj_losses[j] + 0.5 * H[j] ** 2

        train_loss_meter.update(loss.item(), bs)
        original_loss_meter.update(original_task_loss.item(), bs)
        distance_meter.update(distance.item(), bs)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_frep == 0:
            log = 'step: {}, loss(original): {:.4f} ({:.4f}), distance: {:.4f}, train loss: {:.4f} ({:.4f}), train accuracy(original): {:.3f}({:.3f}), lr: {:.5f}'.format(
                i, original_loss_meter.val, original_loss_meter.avg, distance_meter.avg, train_loss_meter.val, train_loss_meter.avg, accuracy_meter.val, accuracy_meter.avg,
                optimizer.param_groups[0]['lr']
            )
            print(log)

    # Update the learning rate based on QGS theory
    if not lr_scheduler:
        divider = 2 * (original_loss_meter.avg + distance_meter.avg) - sum(T_list) + torch.abs(S[0]).data + torch.abs(S[1]).data
        q_lr = QGS_lr_constant / divider
        q_lr = min(q_lr, QGS_lr_max)
        for g in optimizer.param_groups:
            g['lr'] = q_lr
    else:
        lr_scheduler.step()

    log = 'epoch: {}, '.format(epoch)
    log += '{} loss (original): {:.4f}, distance: {:.4f}, '.format(loss_name, original_loss_meter.avg, distance_meter.avg)
    log += 'train loss: {:.4f}, '.format(train_loss_meter.avg)
    log += 'train accuracy: {:.3f}, '.format(accuracy_meter.avg)
    log += 'lr: {:.5f}, '.format(optimizer.param_groups[0]['lr'])
    log += 'k: {:.5f}'.format(k)
    print(log)

    return [original_loss_meter, distance_meter], train_loss_meter.avg, accuracy_meter.avg

def soft_pruning(model, prune_alpha, train_loader, test_loader, criterions, loss_name_list, num_classes, epoch, structured=False, path1='results/temp.pt',
    recover=False, prune_order=1, finetune_best=False, savedir='results'):
    global best_pretrain_acc
    torch.save(
        {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'reparam': False,
        }, path1
    )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not structured:
                prune.l1_unstructured(module, name='weight', amount=prune_alpha)
            else:
                prune.ln_structured(module, name='weight', amount=prune_alpha, n=prune_order, dim=0)
            prune.remove(module, name='weight')
    
    test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
    train_loss, test_loss = train_meter[0].avg, test_meter[0].avg
    print('pruned model test accuracy: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
        test_accuracy, test_loss, train_accuracy, train_loss))

    if finetune_best:
        is_best = train_accuracy > best_pretrain_acc
        if is_best:
            # Save the pre-pruned version of the model
            best_pretrain_acc = train_accuracy
            shutil.copy(path1, os.path.join(savedir, 'best_sf_pretrain.pt'))
            # print('save best model with model sum: ', check_sum(model))
    
    if recover:
        model_dict = torch.load(path1)
        model.load_state_dict(model_dict['state_dict'])

def dynamic_pruning(model, prune_alpha, train_loader, test_loader, criterions, loss_name_list, num_classes, epoch, structured=False,
    prune_order=1, finetune_best=False, savedir='results', prune_name='weight'):

    global best_pretrain_acc
    
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # remove the forward_pre_hooks of the last round of pruning
            # so that the current round of pruning will be disentangled with the last round
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod) and hook._tensor_name == prune_name:
                    del module._forward_pre_hooks[k]
            if not structured:
                prune.l1_unstructured(module, name=prune_name, amount=prune_alpha)
            else:
                prune.ln_structured(module, name=prune_name, amount=prune_alpha, n=prune_order, dim=0)
    
    test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
    train_loss, test_loss = train_meter[0].avg, test_meter[0].avg
    print('pruned model test accuracy: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
        test_accuracy, test_loss, train_accuracy, train_loss))

    if finetune_best:
        is_best = train_accuracy > best_pretrain_acc
        if is_best:
            best_pretrain_acc = train_accuracy
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'reparam': True,
                }, os.path.join(savedir, 'best_dpf_pretrain.pt')
            )
            # print('save best model with model sum: ', check_sum(model))

if __name__ == '__main__':
    main()
        
