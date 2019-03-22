import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solver import Solver
from model import FullyConnectedNet, TwoLayerNet

plt.style.use('ggplot')  

def plot(solver):
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
    plt.savefig('./result/2.png', dpi=300)
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

model = FullyConnectedNet([3],
                          input_dim=6,
                          num_classes=2,
                          weight_scale=5e-2,
                          reg=1e-2)
solver_2 = Solver(
    model,
    data,
    update_rule='sgd',
    optim_config={
        'learning_rate': 1e-3,
    },
    lr_decay=0.95,
    num_epochs=50,
    batch_size=40,
    print_every=100)
solver.train()
plot(solver)