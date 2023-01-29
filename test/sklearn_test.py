import HROCH
from HROCH import PHCRegressor
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class TestSklearn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.params = {'num_threads': 1, 'time_limit': 0.0,
                       'iter_limit': 1000, 'random_state': 42}
        super(TestSklearn, self).__init__(*args, **kwargs)

    def test_fit_predict(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.params['random_state'])

        reg = PHCRegressor(**self.params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        self.assertEqual(y_pred.shape[0], y_test.shape[0])

        # ResourceWarning: Implicitly cleaning up TemporaryDirectory
        reg.dir.cleanup()

    def test_fit_predict_dataframe(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.params['random_state'])

        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)

        reg = PHCRegressor(**self.params)
        reg.fit(X_train_df, y_train)
        y_pred = reg.predict(X_test_df)

        self.assertEqual(y_pred.shape[0], y_test.shape[0])

        # ResourceWarning: Implicitly cleaning up TemporaryDirectory
        reg.dir.cleanup()

    def test_score(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.params['random_state'])

        reg = PHCRegressor(**self.params)
        reg.fit(X_train, y_train)
        score = reg.score(X_test, y_test)
        expected_score = r2_score(y_test, reg.predict(X_test))
        self.assertAlmostEqual(score, expected_score, delta=1e-6)

        # ResourceWarning: Implicitly cleaning up TemporaryDirectory
        reg.dir.cleanup()
