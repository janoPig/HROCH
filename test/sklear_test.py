import unittest
from HROCH import SymbolicRegressor, NonlinearLogisticRegressor, SymbolicClassifier, FuzzyRegressor, FuzzyClassifier, RegressorMathModel, ClassifierMathModel
from sklearn.utils.estimator_checks import check_estimator, check_parameters_default_constructible

skipped_tests = {
    'check_sample_weights_invariance': [{'kind': 'zeros'}], # currently not possible
    'check_estimators_pickle' : [{}, {'readonly_memmap': True}], # cant pickle handle
}

common_params = {'iter_limit':1000, 'time_limit':0.0, 'random_state':42}
classes = [SymbolicRegressor, NonlinearLogisticRegressor, SymbolicClassifier, FuzzyRegressor, FuzzyClassifier, RegressorMathModel, ClassifierMathModel]

class TestSklearnCheck(unittest.TestCase):
    def test_all(self):
        for c in classes:
            print(c.__name__ )
            generator = check_estimator(estimator=c(**common_params), generate_only=True)
            for estimator, check in generator:
                if check.func.__name__ in skipped_tests and check.keywords in skipped_tests.get(check.func.__name__):
                    print('skip ', check.func.__name__, check.keywords)   
                    continue
                
                print(check.func.__name__, check.keywords)
                #if check.func.__name__ == 'check_parameters_default_constructible':
                #    check(estimator)
                check(estimator)