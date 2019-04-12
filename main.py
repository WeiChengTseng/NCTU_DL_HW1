import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp
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
NUM_EPOCH = 600


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


def plot_smooth(solver, filename, a=1, m='-o'):
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


def problem1():
    data_df = pd.read_csv('titanic.csv')
    mask = np.random.choice(800, 800, replace=False)
    data = data_df.values
    data[:800] = data[:800][mask]
    train_acc, val_acc = [], []
    for i in range(1, 17, 1):
        x_train, y_train = data[:int(i * 50), 1:], data[:int(i * 50), 0]
        x_test, y_test = data[800:, 1:], data[800:, 0]
        data_dict = {
            'X_train': x_train,
            'y_train': y_train.astype(int),
            'X_val': x_test,
            'y_val': y_test.astype(int),
        }

        model = FullyConnectedNet([50, 50, 10],
                                  input_dim=6,
                                  num_classes=2,
                                  weight_scale=5e-2,
                                  reg=1e-4)
        solver = Solver(
            model,
            data_dict,
            update_rule=OPTIMIZER,
            optim_config={
                'learning_rate': 0.05,
            },
            lr_decay=0.95,
            num_epochs=800,
            batch_size=10,
            print_every=100,
            verbose=False)
        solver.train()
        train_acc.append(solver.check_accuracy(x_train, y_train))
        val_acc.append(solver.check_accuracy(x_test, y_test))

    index = np.array((range(1, 17, 1))) / 16
    plt.plot(index, train_acc, '-o', label='training accuracy')
    plt.plot(index, val_acc, '-o', label='validation accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Size of Dataset')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('./result/', 'prob1.png'), dpi=250)
    plt.close()

    plt.plot(range(len(solver.loss_history)), solver.loss_history, '-')
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('./result/', 'prob1_loss.png'), dpi=250)
    plt.close()

    plt.plot(
        range(len(solver.train_acc_history)),
        1 - np.array(solver.train_acc_history), '-', label='Training')
    plt.plot(
        range(len(solver.val_acc_history)),
        1 - np.array(solver.val_acc_history), '-', label='Testing')
    plt.title('Error Rate Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Error Rate')
    plt.savefig(os.path.join('./result/', 'prob1_error.png'), dpi=250)
    plt.close()

    return


def problem2():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.values

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]
    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    # model_2 = FullyConnectedNet([3, 3],
    #                             input_dim=6,
    #                             num_classes=2,
    #                             weight_scale=5e-2,
    #                             reg=1e-5)

    model_2 = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=0.01,
                                reg=0)
    solver_2 = Solver(
        model_2,
        data,
        update_rule='adam',
        optim_config={
            'learning_rate': 1e-3,
        },
        lr_decay=0.999,
        num_epochs=1000,
        batch_size=8,
        print_every=100,
        verbose=False)

    # solver_2 = Solver(
    #     model_2,
    #     data,
    #     update_rule=OPTIMIZER,
    #     optim_config={
    #         'learning_rate': 0.1,
    #     },
    #     lr_decay=0.98,
    #     num_epochs=NUM_EPOCH,
    #     batch_size=40,
    #     print_every=100,
    #     verbose=False)
    solver_2.train()
    plot_smooth(solver_2, 'prob2.png', a=0.8, m='-')
    return


def problem3():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.values

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
            'learning_rate': 0.1,
        },
        lr_decay=0.98,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='original',
        verbose=False)
    solver_3.train()
    plot(solver_3, 'prob3.png')

    # -------------------------------------------------------------------------

    data = data_df.values
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
        lr_decay=0.98,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        name='normalized',
        verbose=False)
    solver_3_nor.train()
    plot(solver_3_nor, 'prob3_nor.png')
    plot_solvers_smooth([solver_3, solver_3_nor],
                        'prob3_compared.png',
                        alpha=0.8,
                        m='-')
    return


def problem4():
    data_df = pd.read_csv('titanic.csv')
    col = list(data_df.columns)[1:]
    solvers = []
    for i in col:
        data = data_df.drop(i, axis=1).values
        # print(data_df.drop(i, axis=1).head())
        x_train, y_train = data[:800, 1:], data[:800, 0]
        x_test, y_test = data[800:, 1:], data[800:, 0]
        data = {
            'X_train': x_train,
            'y_train': y_train.astype(int),
            'X_val': x_test,
            'y_val': y_test.astype(int),
        }

        model_4 = FullyConnectedNet([3, 3],
                                    input_dim=5,
                                    num_classes=2,
                                    weight_scale=5e-2,
                                    reg=1e-4)
        solver_4 = Solver(
            model_4,
            data,
            update_rule=OPTIMIZER,
            optim_config={
                'learning_rate': 0.1,
            },
            lr_decay=0.98,
            num_epochs=400,
            batch_size=20,
            print_every=100,
            verbose=False,
            name='without ' + i)
        solver_4.train()
        solvers.append(solver_4)

    data = data_df.values

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]

    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    model_ori = FullyConnectedNet([3, 3],
                                  input_dim=6,
                                  num_classes=2,
                                  weight_scale=5e-2,
                                  reg=1e-4)
    solver_ori = Solver(
        model_ori,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.1,
        },
        lr_decay=0.98,
        num_epochs=400,
        batch_size=20,
        print_every=100,
        name='original',
        verbose=False)
    solver_ori.train()
    solvers.append(solver_ori)

    plot_solvers_smooth(solvers, 'prob4.png', 0.8, '-')

    return


def problem5():
    data_df = pd.read_csv('titanic.csv')
    data = data_df.values

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
    data = data_df.values

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
    data_df = pd.read_csv('titanic.csv')
    data = data_df.values
    column = list(data_df.columns)[1:]

    x_train, y_train = data[:800, 1:], data[:800, 0]
    x_test, y_test = data[800:, 1:], data[800:, 0]
    data = {
        'X_train': x_train,
        'y_train': y_train.astype(int),
        'X_val': x_test,
        'y_val': y_test.astype(int),
    }

    num_survivor, num_victim = np.count_nonzero(
        y_train), 800 - np.count_nonzero(y_train)

    uni = [np.unique(x_train[:, i]) for i in range(6)]
    feat_analysis = {}
    for j in range(6):
        feat_dict = {}
        for i in uni[j]:
            person_in_class = (x_train[:, j] == i)

            num_sur = np.count_nonzero(y_train[person_in_class])
            # print(num_sur, np.count_nonzero(person_in_class)-num_sur)
            feat_dict[i] = (num_sur,
                            np.count_nonzero(person_in_class) - num_sur)
            pass
        feat_analysis[column[j]] = feat_dict

    his('Age', feat_analysis, 10, 70)
    his('Fare', feat_analysis, 10, 100)
    his('Sex', feat_analysis, 10)
    his('Pclass', feat_analysis, 10)
    # FEAT = 'Age'
    # fares = sorted(feat_analysis[FEAT].keys())
    # his = []
    # for i in fares:
    #     his += [i] * feat_analysis[FEAT][i][1]
    # plt.gcf().set_size_inches(16, 9)
    # plt.tight_layout()
    # # plt.hist(fares, [feat_analysis[FEAT][i][1] for i in fares])
    # plt.hist(his, len(fares), align='mid')
    # plt.hist(his, len(fares), align='mid')
    # # plt.hist(fares, [feat_analysis[FEAT][i][0] for i in fares])
    # plt.show()

    return
    model_6 = FullyConnectedNet([3, 3],
                                input_dim=6,
                                num_classes=2,
                                weight_scale=5e-2,
                                reg=1e-5)
    solver_6 = Solver(
        model_6,
        data,
        update_rule=OPTIMIZER,
        optim_config={
            'learning_rate': 0.1,
        },
        lr_decay=0.98,
        num_epochs=NUM_EPOCH,
        batch_size=40,
        print_every=100,
        verbose=False)
    solver_6.train()
    # plot_smooth(solver_5, 'prob2.png', a=0.8, m='-')

    survivor = np.array([1, 0, 22, 0, 0, 70.0])
    victim = np.array([3, 1, 42, 5, 0, 5.0])
    fake_data = np.vstack((survivor, victim))
    print(solver_6.check_accuracy(fake_data, np.array([1, 0])))
    return


def his(FEAT, feat_analysis, nbin, upper_bound=1e5):
    x = sorted(feat_analysis[FEAT].keys())
    his_s, his_v = [], []
    for i in x:
        if i > upper_bound:
            his_s += [upper_bound] * feat_analysis[FEAT][i][0]
        else:
            his_s += [i] * feat_analysis[FEAT][i][0]
    for i in x:
        if i > upper_bound:
            his_s += [upper_bound] * feat_analysis[FEAT][i][0]
        else:
            his_v += [i] * feat_analysis[FEAT][i][1]
    plt.gcf().set_size_inches(8, 4.5)
    plt.tight_layout()

    his_s, his_v = np.array(his_s), np.array(his_v)
    # plt.hist(his_s, len(x), align='mid', label='Survivors')
    # plt.hist(his_v, len(x), align='mid', label='Victims')
    plt.hist([his_s, his_v],
             nbin,
             align='mid',
             label=['Survivors', 'Victims'],
             density=True,
             log=False)
    plt.xlabel(FEAT)
    plt.ylabel('Normalized Number of Pensangers')
    plt.legend()
    plt.savefig('./result/prob6_{}.png'.format(FEAT), dpi=250)
    plt.close()
    # plt.show()
    return


if __name__ == '__main__':
    problem1()
    # problem2()
    # problem3()
    # problem4()
    # problem5()
    # problem6()
    pass