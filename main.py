import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from solver import Solver
from model import FullyConnectedNet

plt.style.use('ggplot')  

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
    plt.plot(solver.loss_history, 'o')
    
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o')

    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o')

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        # plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 20)
    plt.savefig(os.path.join('./result/', filename), dpi=300)
    # plt.show()
    return


# read data
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
                          reg=1e-3)
solver_2 = Solver(
    model_2,
    data,
    update_rule='sgd',
    optim_config={
        'learning_rate': 0.01,
    },
    lr_decay=0.95,
    num_epochs=60,
    batch_size=40,
    print_every=100)
solver_2.train()
plot(solver_2, 'prob2.png')