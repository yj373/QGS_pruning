import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms

import numpy as np
import random

### General helper classes ###
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name='meter'):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # val: values to sum up
        # n: add val to sum for n times
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# refer to: https://github.com/alecwangcq/EigenDamage-Pytorch/blob/master/utils/common_utils.py
class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = {}
        for k, v in decay_schedule.items(): # a dict, example: {"0":0.001, "30":0.00001, "45":0.000001}
            self.decay_schedule[int(k)] = v
        # print('Using a preset learning rate schedule:')
        # print(self.decay_schedule)

    def __call__(self, optimizer, e):
        epochs = list(self.decay_schedule.keys())
        epochs = sorted(epochs) # example: [0, 30, 45]
        lr = self.decay_schedule[epochs[-1]]
        for i in range(len(epochs) - 1):
            if epochs[i] <= e < epochs[i+1]:
                lr = self.decay_schedule[epochs[i]]
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
### General helper functions ###
def check_grad_norm(model):
    """
    Return the 2-norm of the gradient vector of the model
    """
    total_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def check_sum(model):
    """
    Return the sum of all the model parameters
    """
    checksum = 0.0
    for _, v in model.named_parameters():
        checksum += v.sum().item()
    return checksum

def transform_target(outputs, target, mode='one_hot'):
    # Transform target into tensorson with the same shape as output
    # Support mode: one_hot and padding
    if mode == 'one_hot':
        res = torch.FloatTensor(outputs.size()).cuda()
        res.zero_()
        new_labels = torch.unsqueeze(target, 1).cuda()
        res.scatter_(1, new_labels, 1)
    elif mode == 'padding':
        pad_dims = list(outputs.size())[1] - 1
        res = F.pad(input=torch.unsqueeze(target, 1), pad=(0, pad_dims, 0, 0), mode='constant', value=-1).cuda()
    return res

def compute_loss(loss_name, criterion, outputs, target, reduction='elementwise_mean', num_classes=10):
    # Compute loss value of: CrossEntropy loss, MSE loss, 1-norm loss, KLDivLoss and MultiLableMarginLoss
    # loss_name must match to criterion
    # reduction: 'elementwise_mean' return mse or 1-norm loss normally; 'elementwise_sum' return mse or 1-norm loss multiplied by number of classes
    if loss_name == 'CrossEntropy':
        return criterion(outputs, target)
    elif loss_name == 'Hinge':
        return criterion(outputs, target) / num_classes
    elif loss_name == 'mse' or loss_name == '1-norm' or loss_name == 'PoissonNLL':
        one_hot_target = transform_target(outputs, target, mode='one_hot')
        return num_classes * criterion(F.softmax(outputs, dim=1), one_hot_target)
        # return 10*criterion(F.softmax(outputs, dim=1), one_hot_target) if reduction == 'elementwise_sum' else criterion(F.softmax(outputs, dim=1), one_hot_target)
    elif loss_name == 'MultiLabelMargin':
        padding_target = transform_target(outputs, target, mode='padding')
        return criterion(outputs, padding_target)
    elif loss_name == 'KLDiv':
        one_hot_target = transform_target(outputs, target, mode='one_hot')
        return criterion(F.log_softmax(outputs, dim=1), one_hot_target)
    else:
        raise ValueError('Unsupported loss: {}'.format(loss_name))

def hinge(output, target, margin=1., reduction='mean'):
    # output: bs x num_classes
    # output = F.softmax(output, dim=1)
    temp = margin + output - output.gather(1, target.view(-1,1)) # bs x num_classes
    temp = torch.clamp(temp, min=0.) # bs x num_classes
    loss = torch.sum(temp, dim=1) - margin # bs x 1
    if reduction == 'mean':
        loss = loss.mean()
        # loss = loss.sum()/loss.shape[0]
    else:
        raise Exception('not implemented')
    return loss
    
def get_criterions(loss_name_list):
    res = []
    for name in loss_name_list:
        if name == 'CrossEntropy':
            print("initialize CrossEntropy")
            res.append(nn.CrossEntropyLoss())
        elif name == 'mse':
            print("initialize mse")
            res.append(nn.MSELoss())
        elif name == '1-norm':
            print('initialize 1-norm')
            res.append(nn.L1Loss())
        elif name == 'MultiLabelMargin':
            print('initialize MultiLabelMargin')
            res.append(nn.MultiLabelMarginLoss())
        elif name == 'KLDiv':
            print('initialize KLDiv')
            res.append(nn.KLDivLoss(reduction='batchmean'))
        elif name == 'Hinge':
            print('initialize Hinge')
            # res.append(nn.MultiMarginLoss())
            res.append(hinge)
        elif name == 'PoissonNLL':
            print('initialize PoissonNLL')
            res.append(nn.PoissonNLLLoss(log_input=False))
    return res

def accuracy(output, target, topk=(1,)):
    #print(output.shape)
    #print(target.shape)
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.detach().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

def get_model(model_name, num_classes):
    """
    Reture neural network by name
    """ 
    if model_name == 'densenet':
        from models.densenet import densenet100
        architecture = densenet100
        arch_args = (num_classes,)
        arch_kwargs = {}
    elif model_name == 'resnet164':
        from models.resnet import resnet164
        architecture = resnet164
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
    elif model_name == 'resnet110':
        from models.resnet import resnet110
        architecture = resnet110
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
    elif model_name == 'resnet56':
        # from models.resnet import resnet56
        import models.cifar as models
        # architecture = resnet56
        # arch_args = tuple()
        # arch_kwargs = {'num_classes':num_classes}
        model = models.__dict__['resnet'](num_classes=num_classes, depth=56)
        return model
    elif model_name == 'resnet20':
        from models.resnet import resnet20
        architecture = resnet20
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
    elif model_name == 'vgg16':
        from models.vgg import vgg16
        architecture = vgg16
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
    else:
        raise Exception('Model {} not supported'.format(model_name))
    model = architecture(*arch_args, **arch_kwargs)
    return model

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=-1):
    if warmup_steps > 0:
        if step < warmup_steps:
            lr = lr_min + (lr_max - lr_min) * step / warmup_steps
        else:
            lr = lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))
    return lr

def triangle(epoch, total_steps, lr_max, lr_min, warmup_steps=-1):
    if warmup_steps > 0:
        if epoch < warmup_steps:
            lr = (lr_min + (lr_max - lr_min) * epoch / warmup_steps) / lr_min
        else:
            lr = (lr_max - (lr_max - lr_min) * (epoch - warmup_steps) / (total_steps - warmup_steps)) / lr_min
    else:
        lr = (lr_max - (lr_max - lr_min) * (epoch - warmup_steps) / (total_steps - warmup_steps)) / lr_min
    # print(epoch, lr)
    return lr

def get_lr_scheduler(optimizer, total_steps=0, mode='cosine', warmup_steps=-1, step_size=20, gamma=0.5, milestones=[100, 150], max_factor=0.1,
    min_factor=0.01):
    # LambdaLR computes a multiplicative factor
    if mode == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                max_factor,  # since lr_lambda computes multiplicative factor
                min_factor,
                warmup_steps=warmup_steps))
    elif mode == 'triangle':
        # print('before initialize scheduler: {}'.format(optimizer.param_groups[0]['lr']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: triangle(
                epoch,
                total_steps,
                max_factor,  # since lr_lambda computes multiplicative factor
                min_factor,
                warmup_steps=warmup_steps))
        # print('after initialize scheduler: {}'.format(optimizer.param_groups[0]['lr']))
    elif mode =='step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size,
            gamma
        )
    elif mode == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma
        )
    else:
        raise Exception('Unexpected lr mode: {}'.format(mode))
    return scheduler

def count_zero_weights(model, thres=1e-7):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            param = param.cpu()
            temp = torch.zeros(param.shape)
            temp_param = torch.where(torch.abs(param) > thres, param, temp)
            zeros += temp_param.numel() - temp_param.nonzero().size(0)
    return zeros

def check_sparsity(model, module_name=None, structured=False):
    num_zero = 0.0
    num_weights = 0.0
    for name, param in model.named_modules():
        if module_name is not None and not name != module_name:
            continue
        if structured and not isinstance(param, torch.nn.Conv2d):
            continue
        try:
            num_zero += torch.sum(param.weight == 0).item()
            num_weights += param.weight.nelement()
        except Exception as e:
            continue

    sparsity = 100. * num_zero / num_weights
    if not module_name:
        print('Global sparsity: {:.2f}%'.format(sparsity))
    else:
        print('{} sparsity: {:.2f}%'.format(module_name, sparsity))

    return sparsity * 0.01



        