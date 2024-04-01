import unittest
from HROCH import SymbolicRegressor, NonlinearLogisticRegressor, SymbolicClassifier, FuzzyRegressor, FuzzyClassifier, PseudoClassifier, RegressorMathModel, ClassifierMathModel, PseudoClassifierMathModel
from sklearn.utils.estimator_checks import check_estimator


skipped_tests = {
    'check_sample_weights_invariance': [{'kind': 'zeros'}], # mixing samples in this test leads to inconsistent results for small iter_limit
}

common_params = {
    'iter_limit':1000,
    'time_limit':0.0,
    'random_state':42,
    'num_threads':1,
    'problem':{'add':1.0, 'mul':1.0, 'sub':0.1}, # avoid dangerous div or sqrt
    }

class TestSklearnCheck(unittest.TestCase):
    def __test_estimator(self, estimator):
        print(estimator.__class__.__name__ )
        generator = check_estimator(estimator=estimator, generate_only=True)
        for est, check in generator:
            if check.func.__name__ in skipped_tests and check.keywords in skipped_tests.get(check.func.__name__):
                print('skip ', check.func.__name__, check.keywords)   
                continue
            print(check.func.__name__, check.keywords)
            check(est)
        
    def test_symbolic_regressor(self):
        self.__test_estimator(SymbolicRegressor(**common_params))
        
    def test_nonlinear_logistic_regressor(self):
        self.__test_estimator(NonlinearLogisticRegressor(**common_params))
        
    def test_fuzzy_regressor(self):
        self.__test_estimator(FuzzyRegressor(**common_params))
                
    def test_symbolic_classifier(self):
        self.__test_estimator(SymbolicClassifier(NonlinearLogisticRegressor(**common_params)))
            
    def test_fuzzy_classifier(self):
        self.__test_estimator(FuzzyClassifier(FuzzyRegressor(**common_params)))
        
    def test_pseudo_classifier(self):
        self.__test_estimator(PseudoClassifier(t=3.0, n=4, regressor_params=common_params))