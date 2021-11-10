import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict

import curves_tt as curves

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    def forward(self, x, *args):
        out = x
        for key, m in self.named_children():
            if '_relu' in key or '_tanh' in key:
                out = m(out)
            else:
                out = m(out,*args)
        return out

class MLPRegressor(nn.Module):
    def __init__(self, num_neuron, fix_points=None):
        super().__init__()
        kwargs = dict()
        if fix_points is None:
            Conv2d = nn.Conv2d
            BatchNorm2d = nn.BatchNorm2d
            Linear = nn.Linear
        else:
            kwargs['fix_points'] = fix_points
            Conv2d = curves.Conv2d
            BatchNorm2d = curves.BatchNorm2d
            Linear = curves.Linear
        
        self.regressor = Sequential()
        for idx in range(1,len(num_neuron)-1):
            self.regressor.add_module('fc{}'.format(idx-1), Linear(num_neuron[idx-1], num_neuron[idx], **kwargs))
            self.regressor.add_module('fc{}_tanh'.format(idx), nn.Tanh())
            # self.regressor.add_module('fc{}_relu'.format(idx-1), nn.ReLU(inplace=True))
            # self.regressor.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        # self.regressor.add_module('fc{}'.format(len(num_neuron)), Linear(num_neuron[-1],1, **kwargs))
        if len(num_neuron)>2:
            self.regressor.add_module('fc{}'.format(len(num_neuron)-2), Linear(num_neuron[-2], num_neuron[-1], **kwargs))

        # # Initialization
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         n = param.numel()
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'bias' in name:
        #         param.data.fill_(0)
        for name, param in self.named_parameters():
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                    # nn.init.eye_(param)
                elif 'bias' in name:
                    # nn.init.kaiming_normal_(param)
                    # param.data.fill_(0)

                    nn.init.uniform_(param, -0.1, 0.1)
        
    
    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.regressor(x,*args)

    # def extra_repr(self):
    #     inplace_str = ', inplace' if self.inplace else ''
    #     return 'p={}{}'.format(self.p, inplace_str)


class MLPClassifier(nn.Module):
    def __init__(self, num_input, num_neuron, num_classes,fix_points=None):
        super().__init__()
        kwargs = dict()
        if fix_points is None:
            Conv2d = nn.Conv2d
            BatchNorm2d = nn.BatchNorm2d
            Linear = nn.Linear
        else:
            kwargs['fix_points'] = fix_points
            Conv2d = curves.Conv2d
            BatchNorm2d = curves.BatchNorm2d
            Linear = curves.Linear
        
        self.classifier = Sequential(OrderedDict([
            ('fc0', Linear(num_input, num_neuron[0], **kwargs)),
            # ('fc0_tanh', nn.Tanh())
            ('fc0_relu', nn.ReLU(inplace=True))
        ]))
        for idx in range(1,len(num_neuron)):
            self.classifier.add_module('fc{}'.format(idx), Linear(num_neuron[idx-1], num_neuron[idx], **kwargs))
            # self.classifier.add_module('fc{}_tanh'.format(idx), nn.Tanh())
            self.classifier.add_module('fc{}_relu'.format(idx), nn.ReLU(inplace=True))
            # self.classifier.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        self.classifier.add_module('fc_out', Linear(num_neuron[-1], num_classes, **kwargs))
        
        for name, param in self.named_parameters():
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    # nn.init.kaiming_normal_(param)
                    param.data.fill_(0)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    
    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.classifier(x,*args)

def mlp16(num_inputs=784, num_classes=10, fix_points=None):
    return MLPClassifier(num_input=num_inputs, num_neuron=[1000],num_classes=num_classes,fix_points=fix_points)
    

class ConvNN1d(nn.Module):
    def __init__(self, num_features, num_filters=[1], kernel_size=[1], stride=[1]):
        super().__init__()
        self.features = Sequential()
        self.features.add_module('conv0',  nn.Conv1d(num_features, num_filters[0], kernel_size=kernel_size[0], stride=stride[0]))
        
        self.features.add_module('tanh0', nn.Tanh())
        # self.features.add_module('relu0', nn.ReLU(inplace=False))
        # self.features.add_module('norm0', nn.BatchNorm1d(num_filters[0]))
        self.features.add_module('softmax0', nn.Softmax(dim=-1))
        # self.features.add_module('logsoftmax0', nn.LogSoftmax(dim=-1))
        for i, n_out in enumerate(num_filters[1:],1):
            self.features.add_module('conv{}'.format(i),  nn.Conv1d(num_filters[i-1], n_out, kernel_size=kernel_size[i], stride=stride[i]))
            if i!=len(num_filters)-1:
                # self.features.add_module('norm{}'.format(i), nn.BatchNorm1d(n_out))
                # self.features.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
                self.features.add_module('tanh{}'.format(i), nn.Tanh())
                self.features.add_module('softmax{}'.format(i), nn.Softmax(dim=-1))

    def forward(self, x):
        out = self.features(x)
        return out.view(out.shape[0],-1)
        # return F.log_softmax(out.view(out.shape[0],-1), dim=1)


class ConvNN1d_v2(nn.Module):
    def __init__(self, num_features, num_classes, num_filters=[1], kernel_size=[1], stride=[1]):
        super().__init__()
        self.features = Sequential()
        self.features.add_module('conv0',  nn.Conv1d(num_features, num_filters[0], kernel_size=kernel_size[0], stride=stride[0]))
        self.features.add_module('norm0', nn.BatchNorm1d(num_filters[0]))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        # self.features.add_module('tanh0', nn.Tanh())
        for i, n_out in enumerate(num_filters[1:],1):
            self.features.add_module('conv{}'.format(i),  nn.Conv1d(num_filters[i-1], n_out, kernel_size=kernel_size[i], stride=stride[i]))
            # if i!=len(num_filters)-1:
            self.features.add_module('norm{}'.format(i), nn.BatchNorm1d(n_out))
            self.features.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            # self.features.add_module('conv{}_tanh'.format(i), nn.Tanh())
        self.classifier = nn.Linear(num_filters[-1], num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool1d(features, (1,)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
        # return out.view(out.shape[0],-1)
        # return F.log_softmax(out.view(out.shape[0],-1), dim=1)


class LeNet5(nn.Module):
    def __init__(self, num_channels=1, image_size=28, num_classes=10):
        super().__init__()
        self.features = Sequential()
        init_padding = 2 if image_size==28 else 0
        self.features.add_module('conv1',  nn.Conv2d(num_channels, 6, kernel_size=5, stride=1, padding=init_padding))
        self.features.add_module('conv1_relu', nn.ReLU(inplace=True))
        self.features.add_module('conv1_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.features.add_module('conv2',  nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0))
        self.features.add_module('conv2_relu', nn.ReLU(inplace=True))
        self.features.add_module('conv2_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.classifier = Sequential()
        self.classifier.add_module('fc1', nn.Linear(16*5*5, 120))
        self.classifier.add_module('fc1_relu', nn.ReLU(inplace=True))
        self.classifier.add_module('fc2', nn.Linear(120, 84))
        self.classifier.add_module('fc2_relu', nn.ReLU(inplace=True))
        
        # last fc layer
        self.classifier.add_module('fc3', nn.Linear(84,num_classes))

    def forward(self, x):
        features = self.features(x)
        # features = features.view(features.size(0), -1)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)

        return F.log_softmax(out, dim=1)
        # return out


class AlexNet(nn.Module):

    def __init__(self, num_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



class bLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(bLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # self.weight = Parameter(torch.Tensor(out_features, in_features, dtype=torch.uint8))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, codebook=None):
        return F.linear(input, self.weight if codebook is None else codebook[self.weight], self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )