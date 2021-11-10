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
from pruning.pruning_training import QGS_pruning_train, baseline_pruning_train

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser =  argparse.ArgumentParser(description='conventional pruning')
parser.add_argument('--epochs', default=100, type=int, help='pretraining training epochs')
parser.add_argument('--model', default='resnet')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
parser.add_argument('--init_model', default='', type=str, help='path to the initial model')
parser.add_argument('--prune_alpha', default=0.5, type=float, help='percentage of parameters to be pruned')
parser.add_argument('--dist_interval', default=10, type=int, help='Every certain number of inspect the distribution of weight distribution')
parser.add_argument('--order', default=1, type=int, help='order of complexity norm')
parser.add_argument('--structured', type=bool, help='whether use structured pruning')
parser.add_argument('--lr_reset', type=bool, help='whether reset the learning rate schedule after pretraining')
parser.add_argument('--pretrained_model', default='', type=str, help='path to the pretrained model')
parser.add_argument('--soft_pruning', type=bool, help='whether to use soft pruning')
parser.add_argument('--soft_pruning_epochs', default=20, type=int, help='number of epochs of soft pruning')

best_acc = 0

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
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    reparam = False # record the state of reparameterization of the current model
    # load the same initial model
    pretrained = True
    if os.path.isfile(args.pretrained_model):
        print('load the pretrained model from {}'.format(args.pretrained_model))
        model_dict = torch.load(args.pretrained_model)
        model.load_state_dict(model_dict['state_dict'])
        if args.pretrained_model.endswith('initial.pt'):
            pretrained = False
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
        pretrained = False
    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
    print('pretrained model: {}, test accuracy: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(pretrained,
        test_accuracy, test_meter[0].avg, train_accuracy, train_meter[0].avg))

    print('model sum: ', check_sum(model))
    summary(model, (3, 32, 32))

    training_parameters = list(model.parameters())
    optimizer = torch.optim.SGD(training_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    order = args.order

    scheduler = get_lr_scheduler(optimizer, mode='multistep', milestones=[100, 200], gamma=0.1)

    train_accuracy_data = []
    test_accracy_data = []
    k = 5e-4
    # model pruning
    if pretrained:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if not args.structured:
                    prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                else:
                    prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
        sparsity = check_sparsity(model, structured=args.structured)
        test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
        print('pruned model test accuracy before finetuning: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
            test_accuracy, test_meter[0].avg, train_accuracy, train_meter[0].avg))
        reparam = True

    best_acc = 0
    soft_pruning = args.soft_pruning
    for epoch in range(args.epochs):

        _, _, _, train_accuracy = baseline_pruning_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], 
            num_classes, finetune=True, prune_alpha= args.prune_alpha, print_frep=100, order=order, k=k)

        train_accuracy_data.append(train_accuracy)

        _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
        test_accracy_data.append(test_accuracy)
        print('test accuracy: {:.3f}'.format(test_accuracy))
        sparsity = check_sparsity(model, structured=args.structured)
        is_best = test_accuracy > best_acc and reparam == True
        if is_best:
            best_acc = test_accuracy
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'sparsity': sparsity,
                    'reparam': reparam,
                    'structured': args.structured 
                }, os.path.join(args.savedir, 'finetune_best_model.pt')
            )

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'sparsity': sparsity,
                'reparam': reparam,
                'structured': args.structured
            }, os.path.join(args.savedir, 'finetune_final_model.pt')
        )

        labels = ['train', 'test']
        data_source = [train_accuracy_data, test_accracy_data]
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')
        if soft_pruning:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                    else:
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)
                    prune.remove(module, name='weight')
            test_meter, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            train_meter, train_accuracy = validate(model, train_loader, criterions, loss_name_list, num_classes)
            train_loss, test_loss = train_meter[0].avg, test_meter[0].avg
            print('pruned model test accuracy: {:.3f}, test loss: {:.4f}, train accuracy: {:.3f}, train loss: {:.4f}'.format(
                test_accuracy, test_loss, train_accuracy, train_loss))

            soft_pruning = epoch < args.soft_pruning_epochs
        elif not reparam:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if not args.structured:
                        prune.l1_unstructured(module, name='weight', amount=args.prune_alpha)
                    else:
                        prune.ln_structured(module, name='weight', amount=args.prune_alpha, n=order, dim=1)

            reparam = True

 
        # if (epoch + 1)% args.dist_interval == 0:
        #     print('Inspect weight distribution')
        #     target_model_path = os.path.join(args.savedir, 'finetune_final_model.pt')
        #     plot_param_distribution(target_model_path, save_path=args.savedir, model_name=args.model, num_classes=num_classes, 
        #         name=None, fig_name='weights_epoch{}.pdf'.format(epoch + 1))

    # testing
    print('start testing best finetuning model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'finetune_best_model.pt')
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
    print('best finetuning model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))
    

    print('start testing final finetuning model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'finetune_final_model.pt')
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
    print('final finetuning model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))

    if not args.pretrained_model or not os.path.isfile(args.pretrained_model):
        print('start testing best pretraining model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        model_path = os.path.join(args.savedir, 'pretrain_best_model.pt')
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
        print('best pretraining model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))

        print('start testing final pretraining model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
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
        print('final finetuning model test accuracy: {:.3f}, saved at: {}, sparsity: {}, structured: {}'.format(test_accuracy, model_dict_best['epoch'], sparsity, structured))

if __name__ == '__main__':
    main()
