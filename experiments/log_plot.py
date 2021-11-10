import os, sys
sys.path.append('./')

from tools.visualization import log_data_extractor, log_split, plot_multilines

def main():
    target_log = './compare_experiment.txt'
    log_savedir = 'logs'
    plot_savedir = 'results/compare_experiment/'
    log_split(target_log, sign='>>>', savedir=log_savedir, names=['pure_QGS', 'QGS-H', 'Lagrangian(QGS warmup)', 'Lagrangian(std warmup)', 'regular'])
    lde = log_data_extractor([os.path.join(log_savedir, f) for f in os.listdir(log_savedir)])

    target_keys = ['accuracy']
    data = lde.extract_data_by_key(target_keys, line_split='\t', key_split=': ', search_sign='epoch')
    for i, key in enumerate(target_keys):
        labels = list(data[i].keys())
        data_source = [data[i][k] for k in labels]
        plot_multilines(data_source, labels, plot_savedir, xlabel='epoch', ylabel='train_accuracy', title=None, fig_name='train_{}.pdf'.format(key), horizon=None)

    target_keys = ['test accuracy']
    data = lde.extract_data_by_key(target_keys, line_split='\t', key_split=': ', search_sign='test accuracy')
    for i, key in enumerate(target_keys):
        labels = list(data[i].keys())
        data_source = [data[i][k] for k in labels]
        plot_multilines(data_source, labels, plot_savedir, xlabel='epoch', ylabel='test_accuracy', title=None, fig_name='test_{}.pdf'.format(key), horizon=None)

if __name__ == '__main__':
    main()