import torch
from torchvision import datasets, transforms
import numpy as np
import os
def get_dataset(name='cifar100', target_type=torch.FloatTensor, target_shape=None, model='densenet', test_size=0.2):
    """
    return the training datset and test dataset according to the arguments
    """
    root = '/data'
    train_set, test_set = None, None
    if 'cifar' in name:
        # for CIFAR dataset
        if 'densenet' in model:
            mean = [0.5071, 0.4867, 0.4408]
            stdv = [0.2675, 0.2565, 0.2761]
        elif 'vgg' in model:
            mean=[0.485, 0.456, 0.406]
            stdv=[0.229, 0.224, 0.225]
        else:
            mean=[0.4914, 0.4822, 0.4465]
            stdv=[0.2023, 0.1994, 0.2010]
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        if name == 'cifar100':
            train_set = datasets.CIFAR100(root=root, train=True, transform=trans_train, download=True)
            test_set = datasets.CIFAR100(root=root, train=False, transform=trans_test, download=False)
        elif name == 'cifar10':
            train_set = datasets.CIFAR10(root=root, train=True, transform=trans_train, download=True)
            test_set = datasets.CIFAR10(root=root, train=False, transform=trans_test, download=False)
        else:
            raise Exception('dataset not recognized.')
    elif name=='mnist':
        root = '../QGS_training/data'
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        # if not exist, download mnist dataset
        train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

    elif name=='svhn':
        trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        # if not exist, download mnist dataset
        temp_root = '../data/'
        if not os.path.isdir(temp_root):
            os.mkdir(temp_root)
        train_set = datasets.SVHN(root=temp_root, split='train', transform=trans, download=True)
        test_set = datasets.SVHN(root=temp_root, split='test', transform=trans, download=True)
            
    return train_set, test_set

def get_loaders(train_set, test_set, batch_size, num_workers, validate=False, valid_size=0.1):
    if validate:
        indices = list(range(len(train_set)))
        split = int(np.floor(valid_size*len(train_set)))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = torch.utils.data.Subset(train_set, train_idx)
        valid_sampler = torch.utils.data.Subset(train_set, valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_sampler,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            valid_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = None

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)


    return train_loader, val_loader, test_loader