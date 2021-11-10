import os, sys, argparse, shutil, time, random
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import _validate_pruning_amount_init, _validate_structured_pruning, _validate_pruning_dim, _compute_nparams_toprune

from data.prepare import get_dataset, get_loaders
from tools.utils import check_sum, setup_seed, get_model, get_criterions, get_lr_scheduler, check_sparsity
from tools.cca_helpers import pwcca_distance
from experiments.multi_metric_training import validate, regular_train
from tools.visualization import plot_multilines
from functools import partial
from torchsummary import summary

from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser =  argparse.ArgumentParser(description='pruning properties investigation')
parser.add_argument('--model', default='resnet', type=str, help='network architecture')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--gpu', default='2', type=str, help='gpu id for training')
parser.add_argument('--savedir', default='results/', type=str, help='directory to save training results')
parser.add_argument('--pretrained_model', default='', type=str, help='path to a pretrained model')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=3e-4, type=float)

parser.add_argument('--prune_alpha', default=0.5, type=float, help='once the targets are reached, prune the weights by prune_alpha')
parser.add_argument('--prune_order', default=1, type=int, help='order of pruning criterion')
parser.add_argument('--structured', type=bool, help='whether to use structured pruning')
parser.add_argument('--soft', type=bool, help='soft pruning or not')

parser.add_argument('--seed', default=10, type=int, help='random seed')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')

def _compute_cosine_distance(t, dim):
    flattened_t = t.view(t.size(dim), -1).cpu().detach().numpy()
    cosine_dist = pairwise_distances(flattened_t, metric="cosine")
    cosine_dist = np.mean(cosine_dist, axis=1)
    return torch.Tensor(cosine_dist)

def _compute_cca_distance(t, dim, num_seed=5):
    dims = list(range(t.size(dim)))
    distances = [0] * len(dims)
    # print(dims)
    seeds = random.sample(dims, num_seed)
    for seed_idx in seeds:
        for j in dims:
            if j != seed_idx:
                filter1 = t.select(dim, seed_idx)
                filter2 = t.select(dim, j)
                filter1 = filter1.view(-1, filter1.size(0))
                filter2 = filter2.view(-1, filter2.size(0))
                if filter1.size(0) < filter1.size(1):
                    filter1 = filter1.t()
                if filter2.size(0) < filter2.size(1):
                    filter2 = filter2.t()
                distance, _, _, _ = pwcca_distance(filter1, filter2, backend='qr')
                distances[seed_idx] += distance.item()
                distances[j] += distance.item()

    for i in dims:
        if i in seeds:
            distances[i] /= len(dims) - 1
        else:
            distances[i] /= num_seed
    return torch.Tensor(distances)

def _compute_norm(t, n, dim):
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    norm = torch.norm(t, p=n, dim=dims)
    return norm

def _compute_importance(t, dim):
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)
    gradient = t.grad
    importances = (gradient * t).pow(2)
    distances = torch.sum(importances, dims)
    return distances

class AllStructured(prune.BasePruningMethod):
    PRUNING_TYPE = "structured"

    def __init__(self, amount, dim=-1, mode='ln', order=1):
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim
        self.mode = mode # 'ln', 'cosine' or 'cca'
        self.order = order 
    
    def compute_mask(self, t, default_mask):
        """
        zeroing out the channels along the specified dim with the lowest averag PWCCA distance

        t(torch.Tensor): tensor to prune
        default_mask: base ask from previous pruning iterations.
        """
        _validate_structured_pruning(t)
        _validate_pruning_dim(t, self.dim)

        tensor_size = t.shape[self.dim]
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        if self.mode == 'cosine':
            distances = _compute_cosine_distance(t, self.dim)
        elif self.mode == 'cca':
            distances = _compute_cca_distance(t, self.dim)
        elif self.mode == 'importance':
            distances = _compute_importance(t, self.dim)
        else:
            distances = _compute_norm(t, self.order, self.dim)

        topk = torch.topk(distances, k=nparams_tokeep, largest=True)
        print(topk.indices)
        def make_mask(t, dim, indices):
            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = indices
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(cls, module, name, amount, dim, mode, order):
        return super(AllStructured, cls).apply(
            module,
            name,
            amount=amount,
            dim=dim,
            mode=mode,
            order=order
        )

def all_strctured(module, name, amount, dim, mode, order):
    AllStructured.apply(
        module, name, amount, dim, mode, order
    )
    return module

def pruning_recover_experiment(model, prune_mode, epochs, optimizer, lr, criterions, num_classes, train_loader, test_loader, pretrained_model_path, prune_order=1):
    print('{} structured pruning testing >>>>>>>>>>>>>>>>>>>>>>>>'.format(prune_mode))
    scheduler = get_lr_scheduler(optimizer, total_steps=args.epochs, mode='cosine', warmup_steps=-1, max_factor=1, min_factor=1e-3)
    for g in optimizer.param_groups:
        g['lr'] = lr # by default 0.01
    if prune_mode == 'ln':
        print('order: {}'.format(prune_order))
    model_dict = torch.load(pretrained_model_path)
    model.load_state_dict(model_dict['state_dict'])
    print('model sum: ', check_sum(model))
    train_loss, train_accuracy = validate(model, train_loader, criterions, ['CrossEntropy'], num_classes)
    _, test_accuracy = validate(model, test_loader, criterions, ['CrossEntropy'], num_classes)
    print('before pruning, train loss: {:.4f}, train acc: {:.4f}, test accuracy: {:.4f}'.format(train_loss[0].avg, train_accuracy, test_accuracy))
    if prune_mode == 'importance':
        print('warmup for one epoch')
        train_loss, train_accuracy = regular_train(-1, model, train_loader, optimizer, scheduler, criterions[0], 'CrossEntropy', num_classes, print_frep=100)
        _, test_accuracy = validate(model, test_loader, criterions, ['CrossEntropy'], num_classes)
        print('test acc: {:.4f}'.format(test_accuracy))
        # for name, p in model.named_parameters():
        #     print(name, p.grad)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print('start pruning {}, num of channels: {}'.format(name, module.weight.size(0)))
            start = time.perf_counter()
            all_strctured(module, name='weight', amount=args.prune_alpha, dim=0, mode=prune_mode, order=1)
            end = time.perf_counter()
            print('time: {:.4f}s'.format(end - start))

    check_sparsity(model, structured=args.structured)
    train_loss_data = []
    train_accuracy_data = []
    test_accuracy_data = []
    train_loss, train_accuracy = validate(model, train_loader, criterions, ['CrossEntropy'], num_classes)
    _, test_accuracy = validate(model, test_loader, criterions, ['CrossEntropy'], num_classes)
    print('epoch: 0, train loss: {:.4f}, train acc: {:.4f}, test accuracy: {:.4f}'.format(train_loss[0].avg, train_accuracy, test_accuracy))
    train_loss_data.append(train_loss[0].avg)
    train_accuracy_data.append(train_accuracy)
    test_accuracy_data.append(test_accuracy)
    for epoch in range(epochs):
        train_loss, train_accuracy = regular_train(epoch, model, train_loader, optimizer, scheduler, criterions[0], 'CrossEntropy', num_classes, print_frep=100)
        _, test_accuracy = validate(model, test_loader, criterions, ['CrossEntropy'], num_classes)
        print('test acc: {:.4f}'.format(test_accuracy))
        train_loss_data.append(train_loss)
        train_accuracy_data.append(train_accuracy)
        test_accuracy_data.append(test_accuracy)

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')

    return train_loss_data, train_accuracy_data, test_accuracy_data
    

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
    criterions = get_criterions(['CrossEntropy'])

    model = get_model(args.model, num_classes)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    # summary(model, (3, 32, 32))
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         print(name, module.weight.shape)
    imp_train_loss, imp_train_accuracy, imp_test_accuracy = pruning_recover_experiment(model, 'importance', args.epochs-1, optimizer, 
        args.lr, criterions, num_classes, train_loader, test_loader, args.pretrained_model, prune_order=2)

    cosine_train_loss, cosine_train_accuracy, cosine_test_accuracy = pruning_recover_experiment(model, 'cosine', args.epochs, optimizer, 
        args.lr, criterions, num_classes, train_loader, test_loader, args.pretrained_model, prune_order=1)
    
    l1_train_loss, l1_train_accuracy, l1_test_accuracy = pruning_recover_experiment(model, 'ln', args.epochs, optimizer, 
        args.lr, criterions, num_classes, train_loader, test_loader, args.pretrained_model, prune_order=1)
 
    l2_train_loss, l2_train_accuracy, l2_test_accuracy = pruning_recover_experiment(model, 'ln', args.epochs, optimizer, 
        args.lr, criterions, num_classes, train_loader, test_loader, args.pretrained_model, prune_order=2)
    

    data_source = [cosine_train_loss, l1_train_loss, l2_train_loss, imp_train_loss]
    # print(len(cosine_train_loss), len(l1_train_loss), len(l2_train_loss))
    
    labels = ['cosine', 'l1', 'l2', 'imp']
    plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='train_loss', fig_name='train_loss.pdf')

    data_source = [cosine_train_accuracy, l1_train_accuracy, l2_train_accuracy, imp_train_accuracy]
    plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='train_accuracy', fig_name='train_accuracy.pdf')

    data_source = [cosine_test_accuracy, l1_test_accuracy, l2_test_accuracy, imp_test_accuracy]
    plot_multilines(data_source, labels, args.savedir, xlabel='epoch', ylabel='test_accuracy', fig_name='test_accuracy.pdf')


    # unpruned_activations = {}
    # pruned_activations = {}
    # def save_activations(pruned, name, module, input, output):
    #     if not module.training:
    #         act = output.detach()
    #         act = act.view(-1, act.size(0))
    #         if pruned:
    #             if pruned_activations[name] is None:
    #                 pruned_activations[name] = act
    #             else:
    #                 pruned_activations[name] = torch.cat([pruned_activations[name], act], dim=1)
    #         else:
    #             if unpruned_activations[name] is None:
    #                 unpruned_activations[name] = act
    #             else:
    #                 unpruned_activations[name] = torch.cat([unpruned_activations[name], act], dim=1)
    # hook_handles = []
    # for name, m in model.named_modules():
    #     # print(name)
    #     if name == 'module.layer1.0.conv1':
    #         print('register forward hook for {}'.format(name))
    #         handle = m.register_forward_hook(partial(save_activations, False, name))
    #         hook_handles.append(handle)
    #         unpruned_activations[name] = None

    # # validate the pretrained unpruned model with training dataset and save its activation maps of each convolutional layer
    # _, accuracy = validate(model, test_loader, criterions, ['CrossEntropy'], num_classes, maximum=500)
    # print('pretrained model test accuracy: {:.4f}'.format(accuracy))
    # for name in unpruned_activations:
    #     print(unpruned_activations[name].shape)
    #     torch.save(unpruned_activations[name], os.path.join(args.savedir, 'unpruned_{}.pt'.format(name)))
    #     unpruned_activations[name] = None
    #     torch.cuda.empty_cache()

    # for hook in hook_handles:
    #     hook.remove()


    
    
if __name__ == '__main__':
    main()
