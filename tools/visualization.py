import os, sys, shutil, torch
sys.path.append('./')
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.utils.prune as prune
from collections import defaultdict
from tools.utils import get_model, check_sparsity
def plot_multilines(data_source, labels, save_path, xlabel='epoch', ylabel='accuracy',title=None, fig_name='fig.pdf', horizon=[], vertical=[]):
    """
    Generate figure with multiple lines
    """
    if len(data_source) != len(labels):
        raise Exception('len(data_source) != len(labels)')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    figure = plt.figure(figsize=(6, 8))
    for i in range(len(data_source)):
        plt.plot(data_source[i], label=labels[i])
    plt.legend()
    if len(horizon) > 0:
        for h in horizon:
            plt.axhline(y=h, color='r', linestyle='dashed')
    if len(vertical) > 0:
        for v in vertical:
            plt.axvline(x=v, color='r', linestyle='dashed')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    figure.savefig(os.path.join(save_path, fig_name))
    plt.close()

def plot_with_uncertainty(data_source, labels, save_path, xlabel='epoch', ylabel='accuracy',title=None, fig_name='fig.pdf'):
    """
    Generate figure with multiple lines and uncertatinty

    data_source: a 3-d list
    label: legend names
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    mpl.rcParams['xtick.major.size'] = 12
    colors = sns.color_palette('husl', len(data_source))
    epochs = list(range(data_source[0].shape[1]))
    for i in range(len(data_source)):
            mean_data = np.mean(data_source[i], axis=0, dtype=np.float64)
            std_data = np.std(data_source[i], axis=0, dtype=np.float64)
            print('{}: data shape: {}, std norm: {}, mean_data norm: {}'.format(labels[i], data_source[i].shape, np.linalg.norm(std_data), np.linalg.norm(mean_data)))
            ax.plot(epochs, mean_data, label=labels[i], color=colors[i])
            ax.fill_between(epochs, mean_data-std_data, mean_data+std_data, alpha=0.3, facecolor=colors[i])
    ax.legend()
    ax.grid(linestyle='--', linewidth=0.5)
    name = '{}.pdf'.format(fig_name)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.savefig(os.path.join(save_path, name))
    plt.close()

def plot_param_distribution(target_model_path, save_path='results/', model_name='resnet', num_classes=10, name=None, fig_name='weights.pdf'):
    model = get_model(model_name, num_classes)
    save_states = torch.load(target_model_path)
    model_state_dict = save_states['state_dict']
    prune_reparam = save_states['reparam'] if 'remaram' in save_states else False
    structured = save_states['structured'] if 'structured' in save_states else False
    if prune_reparam:
        for _, module in model.named_modules():
            if not structured:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                    prune.l1_unstructured(module, name='weight', amount=0.5)
            else:
                if not isinstance(module, torch.nn.Conv2d):
                    continue
                prune.ln_structured(module, name='weight', amount=0.5, n=1, dim=1)
    model.load_state_dict(model_state_dict)

    parameters = []
    for module_name, module in model.named_modules():
        if name is not None and not module_name.startswith(name):
            continue
        try:
            parameters.extend(module.weight.view(-1).tolist())
        except Exception:
            continue
    if name is not None:
        x_name = "weights of {}".format(name)
    else:
        x_name = "weights"
    print('number of parameters: ', len(parameters))
    plot = sns.displot(parameters, kind='kde')
    plot.axes[0, 0].set_xlabel(x_name)
    # fig = plt.get_figure()
    plot.savefig(os.path.join(save_path, fig_name))
    plt.close()

def log_split(target_log, sign='{', savedir='logs/', names=[]):
    """
    split big log file into target small ones

    target_log: path to the target log file
    sign: sign for new new log file
    savedir: path to save the log files
    """
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    small_file = None
    file_idx = 0
    with open(target_log, 'r') as big_file:
        lines = big_file.readlines()
        for line in lines:
            if line.startswith(sign):
                if small_file:
                    small_file.close()
                    file_idx += 1
                if file_idx < len(names):
                    name = '{}.txt'.format(names[file_idx])
                else:
                    name = '{}.txt'.format(str(file_idx))
                small_file_path = os.path.join(savedir, name)
                small_file = open(small_file_path, 'w')

            small_file.write(line)
        if small_file:
            small_file.close()


class log_data_extractor:
    def __init__(self, log_paths):
        """
        extract data from one or multiple log files

        log_paths: paths of all the target logs
        """
        for path in log_paths:
            if not os.path.isfile(path):
                raise ValueError('{} does not exist!'.format(path))

        self.log_paths = log_paths

    @staticmethod
    def process_data(path, target_key, line_split=', ', key_split=': ', search_sign=None):
        """
        Given a path to a log process the data in a standard way.
        Extract data based on a target key
        
        path: path to the log
        target_keys: key of target data
        line_split: split to generate (key, value) pairs
        key_split: split to generate key and value
        search_sign: if none search for all the lines
        """
        data = []
        with open(path, 'r') as log:
            lines = log.readlines()
            lines = [l.strip().replace('\n', '') for l in lines]

            for line in lines:
                if search_sign is not None and not line.startswith(search_sign):
                    continue
                info_line = line.split(line_split)
                for pair in info_line:
                    split_pair = pair.split(key_split)
                    key = split_pair[0]
                    if key == target_key and len(split_pair) > 1:
                        value = split_pair[1]
                        data.append(float(value))
        return data

    def extract_data_by_key(self, target_keys, line_split=', ', key_split=': ', search_sign=None):
        """
        extract data from all the log files, according to target keys
        """
        target_data = []
        for key in target_keys:
            data_dict = {} 
            for path in self.log_paths:
                name = path.split('/')[-1].split('.')[0]
                data = self.process_data(path, key, line_split=line_split, key_split=key_split, search_sign=search_sign)
                data_dict[name] = data
            target_data.append(data_dict)
        return target_data


# testing function
def main():
    target_model_path = "results/QGS_pruning/initial.pt"
    plot_param_distribution(target_model_path, save_path='results/', model_name='resnet20', num_classes=10, 
        name=None, fig_name='weights.pdf', prune_model=False, prune_amount=0.5)

if __name__ == '__main__':
    main()


