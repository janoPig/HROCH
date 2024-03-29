from HROCH import SymbolicRegressor
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class TestFeatures(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.params = [{'num_threads': 1, 'time_limit': 0.0, 'iter_limit': 1000, 'random_state': 42},
                       {'num_threads': 2, 'time_limit': 0.0, 'iter_limit': 100000,'random_state': 42},
                       {'num_threads': 8, 'time_limit': 0.0, 'iter_limit': 100000, 'random_state': 42}]
        super(TestFeatures, self).__init__(*args, **kwargs)

    def test_random_state(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)

        reg1 = SymbolicRegressor(**self.params[0])
        reg1.fit(X_train, y_train)
        y_pred1 = reg1.predict(X_test)

        reg2 = SymbolicRegressor(**self.params[0])
        reg2.fit(X_train, y_train)
        y_pred2 = reg2.predict(X_test)

        np.testing.assert_array_almost_equal(y_pred1, y_pred2, decimal=6)

    def test_random_state_mt(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)

        reg1 = SymbolicRegressor(**self.params[1])
        reg1.fit(X_train, y_train)
        y_pred1 = reg1.predict(X_test)

        reg2 = SymbolicRegressor(**self.params[1])
        reg2.fit(X_train, y_train)
        y_pred2 = reg2.predict(X_test)

        np.testing.assert_array_almost_equal(y_pred1, y_pred2, decimal=6)
        
    def test_random_state_mt2(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)

        reg1 = SymbolicRegressor(**self.params[2])
        reg1.fit(X_train, y_train)
        y_pred1 = reg1.predict(X_test)

        reg2 = SymbolicRegressor(**self.params[2])
        reg2.fit(X_train, y_train)
        y_pred2 = reg2.predict(X_test)

        np.testing.assert_array_almost_equal(y_pred1, y_pred2, decimal=6)
