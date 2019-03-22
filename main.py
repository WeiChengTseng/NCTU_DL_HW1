import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('titanic.csv')
print(data.shape)
print(data.columns.values)
data = data.as_matrix()
print(data.shape)

x_train, y_train = data[: 800, 1: ], data[: 800, 0] 
x_test, y_test = data[800: , 1: ], data[800: , 0] 

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,},
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()