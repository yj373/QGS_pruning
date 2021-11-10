import os, sys, argparse
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import linalg as LA 
from torchsummary import summary

from collections import OrderedDict
from data.prepare import get_dataset, get_loaders
from tools.utils import check_sum, setup_seed, get_criterions, get_lr_scheduler, count_zero_weights
from tools.visualization import plot_multilines
from multi_metric_training import validate
from pruning.pruning_training import QGS_complexity_train

import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

def initialize_MLP(num_input, num_neurons, num_classes):
    num_layers = len(num_neurons) # number of hidden layers
    layer_list = []
    layer_list.append(('linear0', nn.Linear(num_input, num_neurons[0])))
    layer_list.append(('relu0', nn.ReLU()))
    for i in range(1, num_layers - 1):
        layer_list.append(('linear{}'.format(i), nn.Linear(num_neurons[i-1], num_neurons[i])))
        layer_list.append(('relu{}'.format(i), nn.ReLU()))
    layer_list.append(('linear{}'.format(num_layers), nn.Linear(num_neurons[-1], num_classes)))
    layer_list.append(('relu{}'.format(num_layers), nn.ReLU()))

    return nn.Sequential(OrderedDict(num_layers))

class SimpleMLP2(nn.Module):
    def __init__(self, num_input, num_neurons1, num_neurons2, num_neurons3, num_classes):
        """
        Initialize am MLP with 2 hidden layers
        """
        super(SimpleMLP2, self).__init__()
        self.linear1 = nn.Linear(num_input, num_neurons1)
        self.linear2 = nn.Linear(num_neurons1, num_neurons2)
        self.linear3 = nn.Linear(num_neurons2, num_neurons3)
        self.linear4 = nn.Linear(num_neurons3, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        layer3_out = F.relu(self.linear3(layer2_out))
        out = self.linear4(layer3_out)
        return out

parser =  argparse.ArgumentParser(description='QGS_Lagrangian + pruning')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
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
parser.add_argument('--warmup', default=10, type=int, help='epochs training without constraints')

# min-max parameters
parser.add_argument('--l', default=10, type=int, help='number minimization steps per maximization step')
parser.add_argument('--d', default=10, type=int, help='rate of increment of l')

best_acc = 0

def main():
    global args,  best_acc
    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    setup_seed(args.seed)
    train_set, test_set = get_dataset(name='mnist', target_type=torch.LongTensor, target_shape=(-1, 1), model='mlp')
    train_loader, val_loader, test_loader = get_loaders(train_set, test_set, args.batch_size, args.num_workers, validate=args.validate,
        valid_size=args.valid_size)

    # initialize model and optimizer
    num_classes = 10
    model = SimpleMLP2(784, 256, 512, 256, 10)
    model = model.cuda()
    summary(model, (1, 28, 28))
    num_zero_weights = count_zero_weights(model)
    print('initial model num zero weights: ', num_zero_weights)
    training_parameters = list(model.parameters())
    lam = torch.zeros(4, dtype=torch.float32, device='cuda')
    lam.requires_grad = False
    S = torch.ones(4, dtype=torch.float32, device='cuda') * 0.01
    S.requires_grad = True
    training_parameters += [S]
    optimizer = torch.optim.SGD(training_parameters, lr=args.lr)

    loss_name_list = ['CrossEntropy']
    criterions = get_criterions(loss_name_list)
    T_list = [0.4, 0.8, 0.8, 0.8]
    target_name_list = ['Task', 'linear1', 'linear2', 'linear3']
    l, d = args.l, args.d
    # Initialize lr scheduler
    scheduler = get_lr_scheduler(optimizer, mode='step', step_size=50, gamma=0.5)

    train_accuracy_data = []
    valid_accuracy_data = []
    test_accracy_data = []
    obj_losses_data = defaultdict(list)

    last_record = 0
    verticals = [] # record the epochs when targets are achieved
    for epoch in range(args.epochs):
        # t = (epoch + 1) / args.epochs
        obj_loss_meters, _, train_accuracy = QGS_complexity_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], loss_name_list[0], 
            args.warmup, l, lam, T_list, num_classes, S, print_frep=100, last_record=last_record)

        for meter in obj_loss_meters:
            obj_losses_data[meter.name].append(meter.avg)
        # if T_list is met, decrease T_list by alpha
        targets_met = True
        for i, loss_name in enumerate(target_name_list):
            if obj_losses_data[loss_name][-1] - T_list[i] > 1e-3:
                targets_met = False
                break
        if targets_met:
            T_list = [target * args.alpha for target in T_list]
            last_record = epoch
            verticals.append(epoch)
            print("Targets are all achieved, new targets are: ", T_list)
            print("last record: ", last_record)

        is_best = train_accuracy > best_acc # by default save the model with the highest training accuracy
        best_acc = train_accuracy
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
        num_zero_weights = count_zero_weights(model)
        print('model num zero weights: ', num_zero_weights)

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
        # after each epoch increase l by d
        if epoch >= args.warmup:
            l += d

    # testing
    print('start testing best model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'best_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('best model test accuracy: {:.3f}'.format(test_accuracy))
    num_zero_weights = count_zero_weights(model)
    print('best model num zero weights: ', num_zero_weights)

    print('start testing final model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_path = os.path.join(args.savedir, 'final_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    _, test_accuracy = validate(model, test_loader, criterions, loss_name_list, num_classes)
    print('final model test accuracy: {:.3f}'.format(test_accuracy))
    num_zero_weights = count_zero_weights(model)
    print('final model num zero weights: ', num_zero_weights)

if __name__ == '__main__':
    main()
