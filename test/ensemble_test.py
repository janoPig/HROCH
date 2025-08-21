from HROCH import SymbolicRegressor, NonlinearLogisticRegressor, FuzzyRegressor
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier, AdaBoostRegressor, AdaBoostClassifier


class TestEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        self.params = {"num_threads": 1, "time_limit": 0.0, "iter_limit": 1000, "random_state": 42}
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.params["random_state"])

    def test_bagging_regressor(self):
        ensemble_model = BaggingRegressor(estimator=SymbolicRegressor(**self.params))
        ensemble_model.fit(self.X_train, self.y_train)
        y_pred = ensemble_model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_adaboost_regressor(self):
        ensemble_model = AdaBoostRegressor(estimator=SymbolicRegressor(**self.params))
        ensemble_model.fit(self.X_train, self.y_train)
        y_pred = ensemble_model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_stacking_regressor(self):
        base_model = SymbolicRegressor(**self.params)
        base_model.fit(self.X_train, self.y_train)
        math_models = base_model.get_models()[:5]
        estimators = [(str(m), m) for m in math_models]
        ensemble_model = StackingRegressor(estimators=estimators)
        ensemble_model.fit(self.X_train, self.y_train)
        y_pred = ensemble_model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_voting_regressor(self):
        base_model = SymbolicRegressor(**self.params)
        base_model.fit(self.X_train, self.y_train)
        math_models = base_model.get_models()[:5]
        estimators = [(str(m), m) for m in math_models]
        ensemble_model = VotingRegressor(estimators=estimators)
        ensemble_model.fit(self.X_train, self.y_train)
        y_pred = ensemble_model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_bagging_classifier(self):
        for c in [NonlinearLogisticRegressor, FuzzyRegressor]:
            ensemble_model = BaggingClassifier(estimator=c(**self.params))
            ensemble_model.fit(self.X_train, self.y_train)
            y_pred = ensemble_model.predict(self.X_test)
            self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_adaboost_classifier(self):
        ensemble_model = AdaBoostClassifier(estimator=NonlinearLogisticRegressor(**self.params))
        ensemble_model.fit(self.X_train, self.y_train)
        y_pred = ensemble_model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_stacking_classifier(self):
        for c in [NonlinearLogisticRegressor, FuzzyRegressor]:
            base_model = c(**self.params)
            base_model.fit(self.X_train, self.y_train)
            math_models = base_model.get_models()[:5]
            estimators = [(str(m), m) for m in math_models]
            ensemble_model = StackingClassifier(estimators=estimators)
            ensemble_model.fit(self.X_train, self.y_train)
            y_pred = ensemble_model.predict(self.X_test)
            self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_voting_classifier(self):
        for c in [NonlinearLogisticRegressor, FuzzyRegressor]:
            base_model = c(**self.params)
            base_model.fit(self.X_train, self.y_train)
            math_models = base_model.get_models()[:5]
            estimators = [(str(m), m) for m in math_models]
            ensemble_model = VotingClassifier(estimators=estimators)
            ensemble_model.fit(self.X_train, self.y_train)
            y_pred = ensemble_model.predict(self.X_test)
            self.assertEqual(y_pred.shape[0], self.y_test.shape[0])
