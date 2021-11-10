import os, sys, argparse
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision.models as models

from torchsummary import summary
from collections import OrderedDict, defaultdict
from data.prepare import get_dataset, get_loaders
from tools.utils import check_sum, setup_seed, get_criterions, get_lr_scheduler, get_model, check_sparsity, PresetLRScheduler
from tools.visualization import plot_multilines, plot_param_distribution
from multi_metric_training import validate, regular_train
from pruning.pruning_training import QGS_soft_pruning_train

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser =  argparse.ArgumentParser(description='QGS_Lagrangian + pruning')
parser.add_argument('--model', default='resnet', type=str, help='network architecture')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--lr', default=0.1, type=float, help='pretraining stage initial learning rate')
parser.add_argument('--prune_lr', default=0.01, type=float, help='pruning stage initial learning rate')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
parser.add_argument('--validate', default=False, type=bool, help='save part of training dataset as validation dataset')
parser.add_argument('--valid_size', default=0.1, type=float, help='percentage of training set saved for validation')
parser.add_argument('--test_along', default=True, type=bool, help='record test performance along training')
parser.add_argument('--prune_alpha', default=0.5, type=float, help='once the targets are reached, prune the weights by prune_alpha')
parser.add_argument('--order', default=1, type=int, help='order of complexity norm')
parser.add_argument('--structured', type=bool, help='whether use structured pruning')
parser.add_argument('--init_model', default='', type=str, help='path to the initial model')
parser.add_argument('--prune_warmup', default=10, type=int, help='epochs for warmup stage of QGS-L pretraining')
parser.add_argument('--prune_pretrain', default=50, type=int, help='epochs of training before hard pruning')
parser.add_argument('--soft_prune_cycle', default=10, type=int, help='soft pruning every n epochs in prune pretraining')
parser.add_argument('--prune_finetune', default=100, type=int, help='epochs for finetuning after hard pruning')
# parser.add_argument('--k', default=1e-3, type=float, help='constant for distance')
parser.add_argument('--dist_interval', default=10, type=int, help='Every certain number of inspect the distribution of weight distribution')
parser.add_argument('--pretrain_method', default='QGS', type=str, help='pretraining method: QGS, QGS-H, QGS-L')
parser.add_argument('--lr_reset', type=bool, help='whether reset the learning rate schedule after pretraining')
parser.add_argument('--QGS_lr', type=bool, help='whether to use QGS_lr')
parser.add_argument('--QGS_lr_constant', default=0.05, type=float, help='QGS lr constant')
parser.add_argument('--QGS_lr_max', default=0.1, type=float, help='maximum QGS_lr learning rate')
# min-max parameterspercentage of parameters to be pruned
parser.add_argument('--l', default=10, type=int, help='number minimization steps per maximization step')
parser.add_argument('--d', default=10, type=int, help='rate of increment of l')

parser.add_argument('--pretrain', default=100, type=int, help="num of epochs to pretrain a large network")
parser.add_argument('--pretrained_model', default='', type=str, help='path to the pretrained model')

best_acc = 0

def get_t(epoch, total_epochs):
    return epoch / total_epochs

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
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    reparam = False # record the state of reparameterization of the current model
    if len(args.init_model) != 0:
        print('load the initial model from {}'.format(args.init_model))
        model_dict = torch.load(args.init_model)
        model.load_state_dict(model_dict['state_dict'])
        target_model_path = args.init_model
    else:
        sparsity = check_sparsity(model, structured=args.structured)
        if not args.pretrained_model or not os.path.isfile(args.pretrained_model):
            torch.save(
                {
                    'epoch': 0,
                    'state_dict': model.state_dict(),
                    'sparsity': sparsity,
                    'reparam': reparam,
                    'structured': args.structured
                }, os.path.join(args.savedir, 'initial.pt')
            )
            target_model_path = os.path.join(args.savedir, 'initial.pt')
        else:
            target_model_path = args.pretrained_model
    print('init model path: {}'.format(target_model_path))
    # plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
    #     name=None, fig_name='weights_init.pdf')
    print('model sum: ', check_sum(model))
    summary(model, (3, 32, 32))

    training_parameters = list(model.parameters())
    lam = torch.zeros(2, dtype=torch.float32, device='cuda')
    lam.requires_grad = False
    S = torch.ones(2, dtype=torch.float32, device='cuda') * 0.01
    S.requires_grad = True
    training_parameters += [S]
    optimizer = torch.optim.SGD(training_parameters, lr=args.lr)

    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    order = args.order
    k = 5e-4
    print('k: {:.5f}'.format(k))

    T_list = [0.05, -1] # targets for L(W) and distance
    print('Targets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('original loss: {}, distance: {}'.format(T_list[0], T_list[1]))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    l, d = args.l, args.d
    

    train_accuracy_data = []
    valid_accuracy_data = []
    test_accracy_data = []
    obj_losses_data = defaultdict(list)

    # pretraining
    prune_start = 0
    last_record = 0
    if not args.pretrained_model or not os.path.isfile(args.pretrained_model):
        scheduler = get_lr_scheduler(optimizer, mode='multistep', milestones=[100, 150], gamma=0.1)
    
        for epoch in range(args.pretrain):
            _, train_accuracy = regular_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], num_classes, print_frep=100)
            train_accuracy_data.append(train_accuracy)
            sparsity = check_sparsity(model)
            if args.validate:
                _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
                valid_accuracy_data.append(valid_accuracy)

            if args.test_along:
                _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
                test_accracy_data.append(test_accuracy)
                print('test accuracy: {:.3f}'.format(test_accuracy))

            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'sparsity': sparsity,
                    'reparam': reparam,
                    'structured': args.structured
                }, os.path.join(args.savedir, 'pretrain_final_model.pt')
            )
            labels = ['train']
            data_source = [train_accuracy_data]
            if args.validate:
                labels.append('validate')
                data_source.append(valid_accuracy_data)
            if args.test_along:
                labels.append('test')
                data_source.append(test_accracy_data)
            plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')
        prune_start = args.pretrain
        last_record = args.pretrain
    else:
        print('load the pretrained model from {}'.format(args.pretrained_model))
        model_dict = torch.load(args.pretrained_model)
        model.load_state_dict(model_dict['state_dict'])
        sparsity = check_sparsity(model)
        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        print('pretraining final model test accuracy: {:.3f}'.format(test_accuracy))

    # soft pruning
    print('soft pruning >>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for g in optimizer.param_groups:
        g['lr'] = args.prune_lr # by default 0.01
    scheduler = get_lr_scheduler(optimizer, mode='multistep', milestones=[args.prune_finetune // 2], gamma=0.1)
    for epoch in range(prune_start, prune_start + args.prune_pretrain):
        t = get_t(epoch - last_record, args.prune_pretrain)

        obj_meters, _, train_accuracy = QGS_soft_pruning_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], 
            loss_name_list[0], T_list, num_classes, warmup=args.prune_warmup, l=l, lam=lam, 
            last_record=last_record, S=S, print_frep=100, prune_alpha=args.prune_alpha, t=t, structured=args.structured, order=order, 
            add_window=False, finetune=reparam, k=k, pretrain_method=args.pretrain_method, QGS_lr=args.QGS_lr, QGS_lr_constant=args.QGS_lr_constant, 
            QGS_lr_max=args.QGS_lr_max)

        if args.validate:
            _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
            valid_accuracy_data.append(valid_accuracy)

        if args.test_along:
            _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            test_accracy_data.append(test_accuracy)
            print('test accuracy: {:.3f}'.format(test_accuracy))

        sparsity = check_sparsity(model, structured=args.structured)
        for meter in obj_meters:
            obj_losses_data[meter.name].append(meter.avg)
        train_accuracy_data.append(train_accuracy)

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'sparsity': sparsity,
                'reparam': reparam,
                'structured': args.structured
            }, os.path.join(args.savedir, 'prune_final_model.pt')
        )

        labels = ['train']
        data_source = [train_accuracy_data]
        if args.validate:
            labels.append('validate')
            data_source.append(valid_accuracy_data)
        if args.test_along:
            labels.append('test')
            data_source.append(test_accracy_data)
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')

        if epoch != 0 and epoch % args.soft_prune_cycle == 0:
            # Soft pruning
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                        prune.remove(module, name='weight')
                    else:
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
                        prune.remove(module, name='weight')
            print('soft pruning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
            sparsity = check_sparsity(model)
            test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
            print('soft pruned model test accuracy: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
                test_accuracy, test_meter[0].avg, train_accuracy, train_meter[0].avg))

        # if (epoch + 1) % args.dist_interval == 0:
        #     print('Inspect weight distribution')
        #     target_model_path = os.path.join(args.savedir, 'prune_final_model.pt')
        #     plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
        #         name=None, fig_name='weights_epoch{}.pdf'.format(epoch + 1))
        if epoch - last_record > args.prune_warmup:
            l += d

    # Hard pruning
    print("Hard prune the model >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not args.structured:
                prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                # prune.remove(module, name='weight')
            else:
                prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
                # prune.remove(module, name='weight')
    reparam = True
    sparsity = check_sparsity(model)
    test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
    print('pruned model test accuracy before finetuning: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
        test_accuracy, test_meter[0].avg, train_accuracy, train_meter[0].avg))
    print("finish pruning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    torch.save(
        {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'sparsity': sparsity,
            'reparam': reparam,
            'structured': args.structured
        }, os.path.join(args.savedir, 'prune_final_model.pt')
    )
    # finetuning
    print('finetuning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    finetune_start = prune_start + args.prune_pretrain
    if args.QGS_lr:
        for g in optimizer.param_groups:
            g['lr'] = args.prune_lr # by default 0.01
    for epoch in range(finetune_start, finetune_start + args.prune_finetune):
        t = 1
        _, train_accuracy = regular_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], num_classes, print_frep=100)
        if args.validate:
            _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
            valid_accuracy_data.append(valid_accuracy)

        if args.test_along:
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
            }, os.path.join(args.savedir, 'prune_final_model.pt')
        )

        labels = ['train']
        data_source = [train_accuracy_data]
        if args.validate:
            labels.append('validate')
            data_source.append(valid_accuracy_data)
        if args.test_along:
            labels.append('test')
            data_source.append(test_accracy_data)
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')

        # if (epoch + 1) % args.dist_interval == 0:
        #     print('Inspect weight distribution')
        #     target_model_path = os.path.join(args.savedir, 'prune_final_model.pt')
        #     plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
        #         name=None, fig_name='weights_epoch{}.pdf'.format(epoch + 1))
    if not args.pretrained_model or not os.path.isfile(args.pretrained_model):
        print('start testing pretraining final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        model_path = os.path.join(args.savedir, 'pretrain_final_model.pt')
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
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
            reparam = True
        model.load_state_dict(model_dict_best['state_dict'])
        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        print('pretraining final model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))

    print('start testing finetune final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'prune_final_model.pt')
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
                    prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
        reparam = True
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('finetuning final model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))


if __name__ == '__main__':
    main()
    