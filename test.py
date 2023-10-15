from HROCH.regressor import SymbolicRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 'init_predefined_const_set': [0.99, 1.99],
params = [{'num_threads': 1, 'time_limit': 0.0, 'iter_limit': 1000, 'random_state': 42, 'verbose': 2},
          {'num_threads': 2, 'time_limit': 0.0, 'iter_limit': 100000,
              'random_state': 42, 'verbose': True, 'save_model': True},
          {'num_threads': 8, 'time_limit': 0.0, 'iter_limit': 100000, 'random_state': 42, 'verbose': True, 'save_model': True}]

reg1 = SymbolicRegressor(**params[0])
reg1.fit(X_train, y_train)
y_pred1 = reg1.predict(X_test)

reg2 = SymbolicRegressor(**params[0])
reg2.fit(X_train, y_train)
y_pred2 = reg2.predict(X_test)
