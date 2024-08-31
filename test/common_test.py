from HROCH import SymbolicRegressor, NonlinearLogisticRegressor, SymbolicClassifier, FuzzyRegressor, FuzzyClassifier, Xicor
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, log_loss, make_scorer


class TestCommon(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCommon, self).__init__(*args, **kwargs)
        self.params = {'num_threads': 1, 'time_limit': 0.0,'iter_limit': 1000, 'random_state': 42}
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
        reg = NonlinearLogisticRegressor(**self.params)
        reg.fit(self.X_train, self.y_train)
        y = reg.predict_proba(self.X_test)

        reg_cw = NonlinearLogisticRegressor(**self.params, class_weight={0:1.0, 1:2.0})
        reg_cw.fit(self.X_train, self.y_train)
        y_cw = reg_cw.predict_proba(self.X_test)

        sample_weight_dummy = np.array([1.0, 1.0])[self.y_train]
        reg_swd = NonlinearLogisticRegressor(**self.params)
        reg_swd.fit(self.X_train, self.y_train, sample_weight=sample_weight_dummy)
        y_swd = reg_swd.predict_proba(self.X_test)

        sample_weight = np.array([1.0, 2.0])[self.y_train]
        reg_sw = NonlinearLogisticRegressor(**self.params)
        reg_sw.fit(self.X_train, self.y_train, sample_weight=sample_weight)
        y_sw = reg_sw.predict_proba(self.X_test)

        np.testing.assert_array_almost_equal(y_swd, y, decimal=6)
        self.assertGreater(np.sum(np.abs(y_cw - y)), 0.001)
        np.testing.assert_array_almost_equal(y_cw, y_sw, decimal=6)
        
    def test_classifier(self):
        params = [{'num_threads': 1, 'time_limit': 0.0,'iter_limit': 1000, 'random_state': 42},
                  {'num_threads': 2, 'time_limit': 0.0,'iter_limit': 1000, 'random_state': 42},]
        for p in params:
            classifiers = [
            NonlinearLogisticRegressor(**p), 
            SymbolicClassifier(estimator=NonlinearLogisticRegressor(**p)),
            FuzzyRegressor(**p),
            FuzzyClassifier(estimator=FuzzyRegressor(**p)),
            ]
            for model in classifiers:
                model.fit(self.X_train, self.y_train)
                y = model.predict(self.X_test)
                yp = model.predict_proba(self.X_test)
                np.testing.assert_equal(y.shape, self.y_test.shape)
                np.testing.assert_equal(yp.shape, (self.y_test.shape[0], 2))
                
    def test_classifier_cv(self):
        cv = {'n':2, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'L-BFGS-B'}, 'opt_metric':make_scorer(log_loss, greater_is_better=False, needs_proba=True)}
        params = [{'num_threads': 1, 'time_limit': 0.0,'iter_limit': 1000, 'random_state': 42, 'cv_params' : cv},
                  {'num_threads': 2, 'time_limit': 0.0,'iter_limit': 1000, 'random_state': 42, 'cv_params' : cv},]
        for p in params:
            classifiers = [
            NonlinearLogisticRegressor(**p), 
            SymbolicClassifier(estimator=NonlinearLogisticRegressor(**p)),
            FuzzyRegressor(**p),
            FuzzyClassifier(estimator=FuzzyRegressor(**p)),
            ]
            for model in classifiers:
                model.fit(self.X_train, self.y_train)
                y = model.predict(self.X_test)
                yp = model.predict_proba(self.X_test)
                np.testing.assert_equal(y.shape, self.y_test.shape)
                np.testing.assert_equal(yp.shape, (self.y_test.shape[0], 2))
                if model.__class__ in [SymbolicClassifier, FuzzyClassifier]:
                    self.assertEqual(len(model.estimators_), 1)
                    est = model.estimators_[0]
                    equations = est.get_models()[:2]
                    for eq in equations:
                        self.assertTrue(hasattr(eq, 'cv_score'))
                        eq.fit(self.X_train, self.y_train)
                        y = eq.predict(self.X_test)
                        yp = eq.predict_proba(self.X_test)
                        np.testing.assert_equal(y.shape, self.y_test.shape)
                        np.testing.assert_equal(yp.shape, (self.y_test.shape[0], 2))

    def test_xicor(self):
        # https://towardsdatascience.com/a-new-coefficient-of-correlation-64ae4f260310
        def xicor_orig(X, Y, ties=True):
            np.random.seed(42)
            n = len(X)
            order = np.array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
            if ties:
                l = np.array([sum(y >= Y[order]) for y in Y[order]])
                r = l.copy()
                for j in range(n):
                    if sum([r[j] == r[i] for i in range(n)]) > 1:
                        tie_index = np.array([r[j] == r[i] for i in range(n)])
                        r[tie_index] = np.random.choice(r[tie_index] - np.arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
                return 1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l)))
            else:
                r = np.array([sum(y >= Y[order]) for y in Y[order]])
                return 1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1)
        np.random.seed(seed=42)
        n = 100
        f = 5
        X = np.random.random((n, f))
        y = X[:,0]**3 + 2*X[:,1]**2
        noise = np.random.normal(0,0.1,n)
        for i in range(f):
            orig1 = xicor_orig(X[:,i], y)
            orig2 = xicor_orig(y, X[:,i])
            orig = max(orig1, orig2)
            xi = Xicor(X[:,i], y)
            self.assertAlmostEqual(xi, orig)

