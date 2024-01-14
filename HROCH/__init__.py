"""
.. include:: ../README.md
"""
from HROCH.hroch import *
from HROCH.regressor import SymbolicRegressor
from HROCH.fuzzy import FuzzyRegressor, FuzzyClassifier
from HROCH.classifier import NonlinearLogisticRegressor, SymbolicClassifier

__all__ = ['SymbolicRegressor','NonlinearLogisticRegressor','SymbolicClassifier','FuzzyRegressor', 'FuzzyClassifier','RegressorMathModel','ClassifierMathModel']
