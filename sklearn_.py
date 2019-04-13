import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

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

params = {
    'learning_rate_init': list(np.logspace(-1, -4, 8)),
    'alpha': np.logspace(-1, -5, 10),
    'batch_size': np.linspace(5, 100, 10, dtype=int)
}
solver = MLPClassifier([3, 3], max_iter=8000, solver='sgd')

gs = GridSearchCV(
    estimator=solver, param_grid=params, n_jobs=-1, scoring='accuracy')
gs.fit(data['X_train'], data['y_train'])
print(gs.best_params_)
