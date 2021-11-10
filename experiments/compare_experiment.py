import os, sys, argparse, inspect
sys.path.append('./')
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_metric_training import Lagrangian_train, validate, QGS_H_train, regular_train
from tools.utils import check_sum, setup_seed, get_model, get_criterions, get_lr_scheduler
from tools.visualization import plot_multilines
from data.prepare import get_dataset, get_loaders

from collections import defaultdict

best_acc = 0

def main(dataset='cifar10', model='resnet164', epochs=100, savedir='results/', gpu='2', batch_size=64, num_workers=8, val=False, valid_size=0.1, 
    l=10, d=10, lr=0.05, optimizer='sgd', warmup=20, warmup_QGS=True, seed=10, test_along=True, best_model_name = 'best_model.pt',
    final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='regular', alpha=0.5, T_list=[0.4, 0.4, 0.4], warmup_T_list=[0.4, 0.4, 0.4],
    multistage=True):
    
    global best_acc
    frame = inspect.currentframe()
    _,_,_, argvals = inspect.getargvalues(frame)
    print(argvals)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    setup_seed(seed)
    train_set, test_set = get_dataset(name=dataset, target_type=torch.LongTensor, target_shape=(-1, 1), model=model)
    train_loader, val_loader, test_loader = get_loaders(train_set, test_set, batch_size, num_workers, validate=validate, valid_size=valid_size)

    num_classes = 10
    if dataset == 'cifar100':
        num_classes = 100

    model = get_model(model, num_classes)
    model = model.cuda()
    sum = check_sum(model)
    print('param checksum: ', sum)

    # multi-metric parameters
    loss_name_list = ['CrossEntropy', 'Hinge', '1-norm']
    print("objectives: ", loss_name_list)
    print("targets: ", T_list)
    criterions = get_criterions(loss_name_list)
    num_objectives = len(loss_name_list)
    # constraint variables
    lam = torch.zeros(num_objectives, dtype=torch.float32, device='cuda')
    lam.requires_grad = False

    training_parameters = list(model.parameters())
    if warmup_QGS and train_mode != 'regular':
        S = torch.ones(num_objectives, dtype=torch.float32, device='cuda') * 0.01
        S.requires_grad = True
        training_parameters += [S]
        print('QGS warmup targets: ', warmup_T_list)
    else:
        S = None
    # Initialize optimizer
    if optimizer == 'adam':
        print('initialize Adam optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.Adam(training_parameters)
    elif optimizer == 'adagrad':
        print('initialize Adagrad optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.Adagrad(training_parameters)
    elif optimizer == 'rmsprop':
        print('initialize RMSprop optimizer>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.RMSprop(training_parameters, momentum=0.9)
    else:
        print('initialize SGD optimizer>>>>>>>>>>>>>>>>>>>>>>')
        optimizer = torch.optim.SGD(training_parameters, lr=lr, momentum=0.9)

    # Initialize lr scheduler
    scheduler = get_lr_scheduler(optimizer, mode='step', step_size=50, gamma=0.5)

    # visualization data
    train_accuracy_data = []
    valid_accuracy_data = []
    test_accracy_data = []
    obj_losses_data = defaultdict(list)

    last_record = 0
    verticals = [] # record the epochs when targets are achieved
    # training
    print('{} training >>>>>>>>>>>>>>>>>>'.format(train_mode))
    for epoch in range(epochs):
        # print("The learning rate is {:.4f}, t: {:.4f}".format(optimizer.param_groups[0]['lr'], t))
        if train_mode == 'Lagrangian':
            obj_loss_meters, _, train_accuracy = Lagrangian_train(epoch, model, train_loader, optimizer, scheduler, criterions, loss_name_list, 
                warmup, l, lam, T_list, num_classes, warmup_QGS=warmup_QGS, S=S, warmup_T_list=warmup_T_list, last_record=last_record)
        elif train_mode == 'QGS-H':
            t = (epoch + 1) / epochs
            _, _, train_accuracy, _ = QGS_H_train(epoch, model, train_loader, 
                optimizer, scheduler, criterions, loss_name_list, T_list, S, t, num_classes)
        else:
            _, train_accuracy = regular_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], num_classes)


         # if T_list is met, decrease T_list by alpha
        if train_mode == 'Lagrangian':
            for meter in obj_loss_meters:
                obj_losses_data[meter.name].append(meter.avg)
            targets_met = True
            for i, loss_name in enumerate(loss_name_list):
                if obj_losses_data[loss_name][-1] > T_list[i]:
                    targets_met = False
                    break
            if targets_met and multistage:
                T_list = [target * alpha for target in T_list]
                warmup_T_list = T_list
                last_record = epoch
                verticals.append(epoch)
                print("Targets are all achieved, new targets are: ", T_list)
                print("last record: ", last_record)

        is_best = train_accuracy > best_acc # by default save the model with the highest training accuracy
        best_acc = train_accuracy
        train_accuracy_data.append(train_accuracy)
        if val:
            _, valid_accuracy = validate(model, val_loader, criterions, loss_name_list, num_classes)
            is_best = valid_accuracy > best_acc
            best_acc = valid_accuracy
            valid_accuracy_data.append(valid_accuracy)

        if test_along:
            _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
            test_accracy_data.append(test_accuracy)
            print('test accuracy: {:.3f}'.format(test_accuracy))

        if is_best:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc 
                }, os.path.join(savedir, best_model_name)
            )

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc
            }, os.path.join(savedir, final_model_name)
        )

        labels = ['train']
        data_source = [train_accuracy_data]
        if val:
            labels.append('validate')
            data_source.append(valid_accuracy_data)
        if test_along:
            labels.append('test')
            data_source.append(test_accracy_data)
        plot_multilines(data_source, labels, savedir, xlabel='epoch', ylabel='accuracy', fig_name=fig_name, vertical=verticals)
        # after each epoch increase l by d
        if epoch >= warmup:
            l += d

    # testing
    print('start testing best model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(savedir, best_model_name)
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}'.format(test_accuracy))

    print('start testing final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(savedir, final_model_name)
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}'.format(test_accuracy))

if __name__ == '__main__':

    dataset='cifar100'
    model = 'resnet164'
    epochs = 200
    gpu = '3'
    lr = 0.04 # initial learning rate for the step learning rate schedule
    print('>>>>>>>>>>>>>>>>>>>>Pure QGS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    main(dataset=dataset, model=model, epochs=epochs, savedir='results/compare_experiment/QGS', gpu=gpu, batch_size=64, num_workers=8, 
        val=False, valid_size=0.1, l=10, d=10, lr=lr, optimizer='sgd', warmup=200, warmup_QGS=True, seed=10, test_along=True, 
        best_model_name = 'best_model.pt', final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='Lagrangian', T_list=[0, 0, 0], 
        warmup_T_list=[0, 0, 0], multistage=False)

    print('>>>>>>>>>>>>>>>>>>>>QGS-H>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    main(dataset=dataset, model=model, epochs=epochs, savedir='results/compare_experiment/QGS-H', gpu=gpu, batch_size=64, num_workers=8, 
        val=False, valid_size=0.1, l=10, d=10, lr=lr, optimizer='sgd', warmup=50, warmup_QGS=True, seed=10, test_along=True, 
        best_model_name = 'best_model.pt', final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='QGS-H', T_list=[-4, -2, -2], 
        warmup_T_list=[-4, -2, -2], multistage=False)
    
    print('>>>>>>>>>>>>>>>>>>>>Largrangian (QGS warmup, multi-stage)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    main(dataset=dataset, model=model, epochs=epochs, savedir='results/compare_experiment/QGS_Lagrangian', gpu=gpu, batch_size=64, num_workers=8, 
        val=False, valid_size=0.1, l=20, d=10, lr=lr, optimizer='sgd', warmup=25, warmup_QGS=True, seed=10, test_along=True, 
        best_model_name = 'best_model.pt', final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='Lagrangian', T_list=[0.4, 0.4, 0.4], 
        warmup_T_list=[0.4, 0.4, 0.4], multistage=True)

    print('>>>>>>>>>>>>>>>>>>>>Largrangian (std warmup, multi-stage)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    main(dataset=dataset, model=model, epochs=epochs, savedir='results/compare_experiment/std_Lagrangian', gpu=gpu, batch_size=64, num_workers=8, 
        val=False, valid_size=0.1, l=20, d=10, lr=lr, optimizer='sgd', warmup=25, warmup_QGS=False, seed=10, test_along=True, 
        best_model_name = 'best_model.pt', final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='Lagrangian', T_list=[0.4, 0.4, 0.4], 
        warmup_T_list=[0.4, 0.4, 0.4], multistage=True)

    print('>>>>>>>>>>>>>>>>>>>>regular training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    main(dataset=dataset, model=model, epochs=epochs, savedir='results/compare_experiment/regular', gpu=gpu, batch_size=64, num_workers=8, 
        val=False, valid_size=0.1, l=10, d=10, lr=lr, optimizer='sgd', warmup=50, warmup_QGS=False, seed=10, test_along=True, 
        best_model_name = 'best_model.pt', final_model_name='final_model.pt', fig_name='accuracy.pdf', train_mode='regular')




