import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp
import copy
import tqdm
from scipy import signal
from solver import Solver
from model import FullyConnectedNet

try:
    plt.style.use('gadfly')
except:
    print('fail to use gadfly')
    pass

np.random.seed(0)
# plt.style.use('ggplot')
OPTIMIZER = 'sgd'
NUM_EPOCH = 400


def hyperpara_combination(hp_range, cur_set=[]):
    if cur_set == []:
        hp, r = list(hp_range.items())[0]
        if not isinstance(r, list):
            cur_set = [{hp: r}]
        else:
            cur_set = list(map(lambda x: {hp: x}, r))
        del hp_range[hp]
    if hp_range == {}:
        return cur_set
    else:
        next_set = []
        hp, r = list(hp_range.items())[0]
        # print(hp, r)
        if not (isinstance(r, list)):
            tmp_set = copy.deepcopy(cur_set)
            for i in tmp_set:
                i.update({hp: r})
            _ = map(lambda x: x.update({hp: r}), tmp_set)
            # print('->', tmp_set)
            next_set = next_set + tmp_set
        else:
            for j in r:
                tmp_set = copy.deepcopy(cur_set)
                for i in tmp_set:
                    i.update({hp: j})
                next_set = next_set + tmp_set
                # print('->', tmp_set)
        del hp_range[hp], cur_set
        # print('->', next_set)
        return hyperpara_combination(hp_range, next_set)


def grid_search(data, **kwargs):
    hp_range = {
        'in_dim': kwargs.get('in_dim', 6),
        'bs': kwargs.get('batch_size', 8),
        'ws': kwargs.get('weight_scale', 1e-1),
        'reg': kwargs.get('reg', 1e-3),
        'lr_dec': kwargs.get('lr_dec', 0.99),
        'lr': kwargs.get('lr', 1e-3)
    }
    # print(hp_range)
    hyper_set = hyperpara_combination(hp_range)
    # print(hyper_set)
    acc = []
    for s in tqdm.tqdm(hyper_set):
        model = FullyConnectedNet([3, 3],
                                  input_dim=s['in_dim'],
                                  num_classes=2,
                                  weight_scale=s['ws'],
                                  reg=s['reg'])
        solver = Solver(
            model,
            data,
            update_rule=OPTIMIZER,
            optim_config={
                'learning_rate': s['lr'],
            },
            lr_decay=s['lr_dec'],
            num_epochs=NUM_EPOCH,
            batch_size=s['bs'],
            print_every=100,
            verbose=False)
        solver.train()
        acc.append(solver.check_accuracy(data['X_val'], data['y_val']))
    print(max(acc))
    print(hyper_set[np.argmax(acc)])
    return


def plot(solver, filename, a=1, m='-o'):
    plt.subplot(3, 1, 1)
    plt.title('Training Loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', markersize=4, alpha=a)
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, m, markersize=4, alpha=a)
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, m, markersize=4, alpha=a)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        # plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return


def plot_smooth(solver, filename, a=1, m='-o'):
    plt.subplot(3, 1, 1)
    plt.title('Training Loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    plt.plot(
        sp.signal.savgol_filter(solver.loss_history, 401, 2),
        m,
        markersize=4,
        alpha=a)
    plt.subplot(3, 1, 2)
    plt.plot(
        sp.signal.savgol_filter(solver.train_acc_history, 9, 2),
        m,
        markersize=4,
        alpha=a)
    plt.subplot(3, 1, 3)
    plt.plot(
        sp.signal.savgol_filter(solver.val_acc_history, 9, 2),
        m,
        markersize=4,
        alpha=a)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        # plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return


def plot_solvers(solvers, filename, alpha=1, m='-o'):
    plt.subplot(3, 1, 1)
    plt.title('Training Loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    for s in solvers:
        plt.plot(s.loss_history, 'o', markersize=4, label=s.name, alpha=alpha)

    plt.subplot(3, 1, 2)
    for s in solvers:
        plt.plot(
            s.train_acc_history, m, markersize=4, label=s.name, alpha=alpha)

    plt.subplot(3, 1, 3)
    for s in solvers:
        plt.plot(s.val_acc_history, m, markersize=4, label=s.name, alpha=alpha)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return


def plot_solvers_smooth(solvers, filename, alpha=1, m='-o'):
    plt.subplot(3, 1, 1)
    plt.title('Training Loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    for s in solvers:
        plt.plot(
            sp.signal.savgol_filter(s.loss_history, 201, 3),
            # 'o',
            markersize=4,
            label=s.name,
            alpha=alpha)

    plt.subplot(3, 1, 2)
    for s in solvers:
        plt.plot(
            sp.signal.savgol_filter(s.train_acc_history, 15, 2),
            m,
            markersize=4,
            label=s.name,
            alpha=alpha)

    plt.subplot(3, 1, 3)
    for s in solvers:
        plt.plot(
            sp.signal.savgol_filter(s.val_acc_history, 15, 2),
            m,
            markersize=4,
            label=s.name,
            alpha=alpha)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return

def smooth(value):
    return sp.signal.savgol_filter(value, 5, 2)

