import os, sys, argparse
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from torchsummary import summary
from collections import OrderedDict, defaultdict
from data.prepare import get_dataset, get_loaders
from tools.utils import check_sum, setup_seed, get_criterions, get_lr_scheduler, get_model, check_sparsity
from tools.visualization import plot_multilines, plot_param_distribution
from multi_metric_training import validate, regular_train
from pruning.pruning_training import QGS_pruning_train, baseline_pruning_train, compute_layer_norm

import matplotlib
matplotlib.use('Agg')

parser =  argparse.ArgumentParser(description='QGS_Lagrangian + pruning')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--model', default='resnet', type=str, help='network architecture')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
parser.add_argument('--validate', default=False, type=bool, help='save part of training dataset as validation dataset')
parser.add_argument('--valid_size', default=0.1, type=float, help='percentage of training set saved for validation')
parser.add_argument('--test_along', default=True, type=bool, help='record test performance along training')
parser.add_argument('--alpha', default=0.5, type=float, help='once the targets are reached, decrease the targets by alpha')
parser.add_argument('--prune_alpha', default=0.5, type=float, help='once the targets are reached, prune the weights by prune_alpha')
parser.add_argument('--warmup', default=10, type=int, help='epochs training without constraints')
parser.add_argument('--init_model', default='', type=str, help='path to the initial model')
parser.add_argument('--target_sparsity', default=0.5, type=float, help='target sparsity for the model')
parser.add_argument('--dist_interval', default=10, type=int, help='Every certain number of inspect the distribution of weight distribution')
parser.add_argument('--QGS_warmup', default='True', type=str, help='whether use QGS to warmup, if not, use unconstrained warmup')
# parser.add_argument('--max_pretrain', default=100, type=int, help='maximum pretrainin epochs')
parser.add_argument('--special_norm_type', default='', type=str, help='whether use special norm')
parser.add_argument('--order', default=1, type=int, help='order of complexity norm')
parser.add_argument('--structured', type=bool, help='whether use structured pruning')

# min-max parameterspercentage of parameters to be pruned
parser.add_argument('--l', default=10, type=int, help='number minimization steps per maximization step')
parser.add_argument('--d', default=10, type=int, help='rate of increment of l')

best_acc = 0
compress_rate = 1

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
    train_loader, val_loader, test_loader = get_loaders(train_set, test_set, args.batch_size, args.num_workers, validate=args.validate,
        valid_size=args.valid_size)
        
    model = get_model(args.model, num_classes)
    model = model.cuda()
    if len(args.init_model) != 0:
        print('load the initial model from {}'.format(args.init_model))
        model_dict = torch.load(args.init_model)
        model.load_state_dict(model_dict['state_dict'])
        target_model_path = args.init_model
        plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
                name=None, fig_name='weights_init.pdf', prune_model=False, prune_amount=0)
    else:
        torch.save(
            {
                'epoch': 0,
                'state_dict': model.state_dict(),
            }, os.path.join(args.savedir, 'initial.pt')
        )
        target_model_path =  os.path.join(args.savedir, 'initial.pt')
        plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
                name=None, fig_name='weights_init.pdf', prune_model=False, prune_amount=0)
    print('model sum: ', check_sum(model))
    summary(model, (3, 32, 32))

    training_parameters = list(model.parameters())
    lam = torch.zeros(4, dtype=torch.float32, device='cuda')
    lam.requires_grad = False
    S = torch.ones(4, dtype=torch.float32, device='cuda') * 0.01
    S.requires_grad = True
    training_parameters += [S]
    optimizer = torch.optim.SGD(training_parameters, lr=args.lr)

    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    order = args.order
    if args.special_norm_type == 'single_mode':
        # y = -0.5x^2+0.5 or y = -0.5|x|+0.5
        a = 0.1
        b = a
        c = 0
        special_norm_type = args.special_norm_type
    elif args.special_norm_type == 'double_mode':
        # y = -x^2 + |x| or y = -||x| - 0.5| + 0.5
        a = 1
        b = 1 if order == 2 else 0.5
        c = 0 if order == 2 else 0.5
        special_norm_type = args.special_norm_type
    elif args.special_norm_type == 'w':
        a = 3
        b = 0.3
        c = 0.5
        special_norm_type = args.special_norm_type
        order = 1
    else:
        special_norm_type = None
        a = 0
        b = 0
        c = 0

    k = 3e-4 
    blocks = ['layer1', 'layer2', 'layer3']
    T_list = [0.4] # t_task is 0.4
    target_name_list = ['Task', 'layer1', 'layer2', 'layer3']
    initial_complexity_losses = initial_model_analysis(model, blocks, special_norm_type=special_norm_type, b=b, a=a, c=c, k=k, order=order)
    for complexity_loss in initial_complexity_losses:
        T_list.append(complexity_loss * args.target_sparsity)
    print("Training targets: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for name, target in zip(target_name_list, T_list):
        print(name, target)
    print('special norm type: {}, order: {}, k: {:.5f}, a: {}, b: {}, c: {}'.format(
        special_norm_type if special_norm_type else 'regular', order, k, a, b, c))
    l, d = args.l, args.d
    # Initialize lr scheduler
    scheduler = get_lr_scheduler(optimizer, mode='step', step_size=50, gamma=0.5)
    
    train_accuracy_data = []
    valid_accuracy_data = []
    test_accracy_data = []
    obj_losses_data = defaultdict(list)

    last_record = 0
    verticals = [] # record the epochs when targets are achieved
    QGS_warmup = args.QGS_warmup == 'True' or args.QGS_warmup == 'true'
    for epoch in range(args.epochs):
        task_loss_meter, complexity_loss_meters, _, train_accuracy = QGS_pruning_train(epoch, model, train_loader, optimizer, scheduler, 
            criterions[0], loss_name_list[0], args.warmup, l, lam, T_list, num_classes, S=S, print_frep=100, last_record=last_record, order=order, 
            k=k, blocks=blocks, QGS_warmup=QGS_warmup, special_norm_type=special_norm_type, b=b, a=a, c=c)
        sparsity = check_sparsity(model, structured=args.structured)
        for meter in complexity_loss_meters:
            obj_losses_data[meter.name].append(meter.avg)
        obj_losses_data['Task'].append(task_loss_meter.avg)

        # if T_list is met, decrease T_list by alpha and prune half of the model parameters
        targets_met = True
        for i, loss_name in enumerate(target_name_list):
            if obj_losses_data[loss_name][-1] - T_list[i] > 1e-3:
                targets_met = False
                break
        if targets_met:
            
            last_record = epoch
            verticals.append(epoch)
            print("Targets are all achieved! Prune the parameter into half!")

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                    else:
                        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                            continue
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
            sparsity = check_sparsity(model, structured=args.structured)
            if sparsity >= args.target_sparsity:
                # required architecture sparsity is reached or reaches maximum pretraining epochs
                T_list[0] = 0
                last_record = 0 # disable QGS warming up
                # enter finetuning stage reinitialize the learning rate scheduler
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                scheduler = get_lr_scheduler(optimizer, mode='step', step_size=50, gamma=0.5)
            else:
                # Focus on architecture sparsity
                for index in range(1, len(T_list)):
                    T_list[index] *= args.alpha
            for name, target in zip(target_name_list, T_list):
                print(name, target)

        train_accuracy_data.append(train_accuracy)
        if args.validate:
            _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
            is_best = valid_accuracy > best_acc
            best_acc = valid_accuracy
            valid_accuracy_data.append(valid_accuracy)

        if args.test_along:
            _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            test_accracy_data.append(test_accuracy)
            print('test accuracy: {:.3f}'.format(test_accuracy))

        is_best = train_accuracy > best_acc if not args.validate else valid_accuracy > best_acc
        if is_best:
            best_acc = train_accuracy if not args.validate else valid_accuracy

        if is_best:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'sparsity': sparsity 
                }, os.path.join(args.savedir, 'best_model.pt')
            )

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'sparsity': sparsity
            }, os.path.join(args.savedir, 'final_model.pt')
        )

        labels = ['train']
        data_source = [train_accuracy_data]
        if args.validate:
            labels.append('validate')
            data_source.append(valid_accuracy_data)
        if args.test_along:
            labels.append('test')
            data_source.append(test_accracy_data)
        print('targets achieved: ', verticals)
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf', vertical=verticals)
        if (epoch + 1)% args.dist_interval == 0:
            print('Inspect weight distribution')
            target_model_path = os.path.join(args.savedir, 'final_model.pt')
            plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
                name=None, fig_name='weights_epoch{}.pdf'.format(epoch + 1), prune_model=(sparsity > 0), prune_amount=sparsity, structured=args.structured)
        # after each epoch increase l by d
        if epoch >= args.warmup:
            l += d

    # testing
    # by default no pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
            if (not isinstance(module, torch.nn.Conv2d)) and args.structured:
                continue
            prune.remove(module, name='weight')
                

    print('start testing best model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'best_model.pt')
    model_dict_best = torch.load(model_path)
    sparsity = model_dict_best['sparsity']
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
            if not args.structured:
                prune.l1_unstructured(module, name='weight', amount=sparsity)
            else:
                if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                    continue
                prune.ln_structured(module, name='weight', amount=sparsity, n=order, dim=1)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}, saved at: {}'.format(test_accuracy, model_dict_best['epoch']))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
            if (not isinstance(module, torch.nn.Conv2d)) and args.structured:
                continue
            prune.remove(module, name='weight')

    print('start testing final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'final_model.pt')
    model_dict_best = torch.load(model_path)
    sparsity = model_dict_best['sparsity']
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
            if not args.structured:
                prune.l1_unstructured(module, name='weight', amount=sparsity)
            else:
                if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                    continue
                prune.ln_structured(module, name='weight', amount=sparsity, n=order, dim=1)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('final model test accuracy: {:.3f}, saved at: {}'.format(test_accuracy, model_dict_best['epoch']))

def initial_model_analysis(model, blocks, special_norm_type=None, b=1, a=1, c=0, k=1e-3, order=1):
    complexity_losses = []
    for block_name in blocks:
        norm = compute_layer_norm(model, block_name, order=order, b=b, a=a, c=c, special_norm_type=special_norm_type)
        norm = norm.item()
        complexity_losses.append(norm * k)
        print("{}: {:.4f}".format(block_name, complexity_losses[-1]))

    return complexity_losses
    # for name, module in model.named_parameters():
    #     check_sparsity(model, param_name=name)


if __name__ == '__main__':
    main()