import os, sys, argparse
sys.path.append('./')
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_metric_training import QGS_H_train, validate
from tools.utils import check_sum, setup_seed, get_model, get_criterions, cosine_annealing
from tools.visualization import plot_multilines
from data.prepare import get_dataset, get_loaders

parser =  argparse.ArgumentParser(description='QGS-H multi-metric training')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--model', default='resnet164', type=str, help='network architecture')
parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--validate', default=True, type=bool, help='save part of training dataset as validation dataset')
parser.add_argument('--valid_size', default=0.1, type=float, help='percentage of training set saved for validation')
parser.add_argument('--test_along', default=True, type=bool, help='record test performance along training')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimization method')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

best_acc = 0

def main():
    global args,  best_acc
    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_flag = torch.cuda.is_available()
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    setup_seed(args.seed)
    train_set, test_set = get_dataset(name=args.dataset, target_type=torch.LongTensor, target_shape=(-1, 1), model=args.model)
    train_loader, val_loader, test_loader = get_loaders(train_set, test_set, args.batch_size, args.num_workers, validate=args.validate, valid_size=args.valid_size)

    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100

    model = get_model(args.model, num_classes)
    model = model.cuda()
    sum = check_sum(model)
    print('param checksum: ', sum)
    # QGS specific parameters
    loss_name_list = ['CrossEntropy', 'Hinge', '1-norm']
    T_list = [-4, -2, -2]
    print("objectives: ", loss_name_list)
    print("targets: ", T_list)
    criterions = get_criterions(loss_name_list)
    num_objectives = len(loss_name_list)
    # Slack variables
    S = torch.ones(num_objectives, dtype=torch.float32, device='cuda') * 0.01
    S.requires_grad = True
    # Homotopy parameter
    t = 0

    # Initialize optimizer
    if args.optimizer == 'adam':
        print('initialize Adam optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.Adam(list(model.parameters()) + [S])
    elif args.optimizer == 'adagrad':
        print('initialize Adagrad optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.Adagrad(list(model.parameters()) + [S])
    elif args.optimizer == 'rmsprop':
        print('initialize RMSprop optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.RMSprop(list(model.parameters()) + [S], momentum=0.9)
    else:
        print('initialize SGD optimizer>>>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.SGD(list(model.parameters()) + [S], lr=args.lr, momentum=0.9)

    # Initialize lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs,
            1,  # since lr_lambda computes multiplicative factor
            1e-6))

    # visualization data
    train_accuracy_data = []
    train_loss_data = []
    valid_accuracy_data = []
    test_accracy_data = []
    energy_data = []
    
    # training
    for epoch in range(args.epochs):
        t = (epoch + 1) / args.epochs
        # print("The learning rate is {:.4f}, t: {:.4f}".format(optimizer.param_groups[0]['lr'], t))
        _, train_loss, train_accuracy, energy = QGS_H_train(epoch, model, train_loader, 
            optimizer, scheduler, criterions, loss_name_list, T_list, S, t, num_classes)

        is_best = train_accuracy > best_acc # by default save the model with the highest training accuracy
        best_acc = train_accuracy
        train_accuracy_data.append(train_accuracy)
        train_loss_data.append(train_loss)
        energy_data.append(energy)
        if args.validate:
            _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
            is_best = valid_accuracy > best_acc
            best_acc = valid_accuracy
            valid_accuracy_data.append(valid_accuracy)

        if args.test_along:
            _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            test_accracy_data.append(test_accuracy)
            print('test accuracy: {:.3f}'.format(test_accuracy))

        if is_best:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc 
                }, os.path.join(args.savedir, 'best_model.pt')
            )

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc
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
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='accuracy', fig_name='accuracy.pdf')
        # sum = check_sum(model)
        # print('param checksum: ', sum)

        labels = ['energy']
        data_source = [energy_data]
        plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='energy', fig_name='energy.pdf')

    # testing
    print('start testing best model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'best_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}'.format(test_accuracy))

    print('start testing final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'final_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}'.format(test_accuracy))

if __name__ == '__main__':
    main()



        


