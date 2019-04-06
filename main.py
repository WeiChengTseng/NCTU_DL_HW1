import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from solver import Solver
from model import FullyConnectedNet

np.random.seed(0)
plt.style.use('ggplot')
OPTIMIZER = 'adam'
NUM_EPOCH = 100


def plot(solver, filename):
    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', markersize=4)
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', markersize=4)
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', markersize=4)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        # plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return

def plot_solvers(solvers, filename):
    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    for s in solvers:
        plt.plot(s.loss_history, 'o', markersize=4, label=s.name)

    plt.subplot(3, 1, 2)
    for s in solvers:
        plt.plot(s.train_acc_history, '-o', markersize=4, label=s.name)
        
    plt.subplot(3, 1, 3)
    for s in solvers:
        plt.plot(s.val_acc_history, '-o', markersize=4, label=s.name)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(9, 12)
    plt.tight_layout()
    plt.savefig(os.path.join('./result/', filename), dpi=250)
    plt.close()

    return


def problem1():
    return


def problem2():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.as_matrix()

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]
    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_2 = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-4)
    solver_2 = Solver(
        model_2,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.01,
        },
        lr_decay=0.95,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        verbose=False)
    solver_2.train()
    plot(solver_2, 'prob2.png')
    return


def problem3():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.as_matrix()

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]

    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_3 = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-4)
    solver_3 = Solver(
        model_3,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.01,
        },
        lr_decay=0.95,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='original',
        verbose=False)
    solver_3.train()
    plot(solver_3, 'prob3.png')

    # -------------------------------------------------------------------------

    data = data_df.as_matrix()
    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]

    fare_mean, fare_std = np.mean(x_train[:, -1]), np.std(x_train[:, -1])
    age_mean, age_std = np.mean(x_train[:, 2]), np.std(x_train[:, 2])
    x_train[:, -1] = (x_train[:, -1] - fare_mean) / fare_std
    x_test[:, -1] = (x_test[:, -1] - fare_mean) / fare_std
    x_train[:, 2] = (x_train[:, 2] - age_mean) / age_std
    x_test[:, 2] = (x_test[:, 2] - age_mean) / age_std

    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_3_nor = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-4)
    solver_3_nor = Solver(
        model_3_nor,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.01,
        },
        lr_decay=0.95,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='normalized',
        verbose=False)
    solver_3_nor.train()
    plot(solver_3_nor, 'prob3_nor.png')
    plot_solvers([solver_3, solver_3_nor], 'prob3_compared.png')
    return


def problem4():
    return


def problem5():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.as_matrix()

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]

    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_5 = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-4)
    solver_5 = Solver(
        model_5,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.01,
        },
        lr_decay=0.98,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='original',
        verbose=False)
    solver_5.train()
    plot(solver_5, 'prob5.png')


    data_df['Pclass'] = pd.Categorical(data_df['Pclass'])
    df_dummies = pd.get_dummies(data_df['Pclass'], prefix='category')
    data_df = pd.concat([data_df, df_dummies], axis=1)
    data_df = data_df.drop(['Pclass'], axis=1)
    # print(data_df.head())
    data = data_df.as_matrix()

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]

    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_5_cat = FullyConnectedNet([3, 3],
                                input_dim=8,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-4)
    solver_5_cat = Solver(
        model_5_cat,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.01,
        },
        lr_decay=0.98,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='categorical',
        verbose=False)
    solver_5_cat.train()
    plot(solver_5_cat, 'prob5_cat.png')
    plot_solvers([solver_5, solver_5_cat], 'prob5_compared.png')
    return


def problem6():
    return


if __name__ == '__main__':
    problem2()
    problem3()
    problem5()
    pass