import HROCH
from HROCH import SymbolicRegressor, NLLRegressor
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class TestSklearn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSklearn, self).__init__(*args, **kwargs)
        self.params = {'num_threads': 1, 'time_limit': 0.0,
                       'iter_limit': 1000, 'random_state': 42, 'verbose':True, 'cv':False}
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.params['random_state'])

    def test_fit_predict(self):
        reg = SymbolicRegressor(**self.params)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)

        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_fit_predict_dataframe(self):
        X_train_df = pd.DataFrame(self.X_train)
        X_test_df = pd.DataFrame(self.X_test)

        reg = SymbolicRegressor(**self.params)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)

        reg = SymbolicRegressor(**self.params)
        reg.fit(X_train_df, self.y_train)
        y_pred_df = reg.predict(X_test_df)

        self.assertEqual(y_pred_df.shape[0], self.y_test.shape[0])

        np.testing.assert_array_almost_equal(y_pred_df, y_pred, decimal=6)

    def test_score(self):
        reg = SymbolicRegressor(**self.params)
        reg.fit(self.X_train, self.y_train)
        score = reg.score(self.X_test, self.y_test)
        expected_score = r2_score(self.y_test, reg.predict(self.X_test))
        self.assertAlmostEqual(score, expected_score, delta=1e-6)

    def test_weights(self):
        reg = NLLRegressor(**self.params)
        reg.fit(self.X_train, self.y_train)
        y = reg.predict_proba(self.X_test)

        reg_cw = NLLRegressor(**self.params, class_weight=[1.0, 2.0])
        reg_cw.fit(self.X_train, self.y_train)
        y_cw = reg_cw.predict_proba(self.X_test)

        sample_weight_dummy = np.array([1.0, 1.0])[self.y_train]
        reg_swd = NLLRegressor(**self.params)
        reg_swd.fit(self.X_train, self.y_train, sample_weight=sample_weight_dummy)
        y_swd = reg_swd.predict_proba(self.X_test)

        sample_weight = np.array([1.0, 2.0])[self.y_train]
        reg_sw = NLLRegressor(**self.params)
        reg_sw.fit(self.X_train, self.y_train, sample_weight=sample_weight)
        y_sw = reg_sw.predict_proba(self.X_test)

        np.testing.assert_array_almost_equal(y_swd, y, decimal=6)
        self.assertGreater(np.sum(np.abs(y_cw - y)), 0.001)
        np.testing.assert_array_almost_equal(y_cw, y_sw, decimal=6)
