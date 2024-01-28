"""
.. include:: ../README.md
"""
from .hroch import RegressorMathModel, ClassifierMathModel
from .regressor import SymbolicRegressor
from .fuzzy import FuzzyRegressor, FuzzyClassifier
from .classifier import NonlinearLogisticRegressor, SymbolicClassifier
from .version import __version__

__all__ = ['SymbolicRegressor','NonlinearLogisticRegressor','SymbolicClassifier','FuzzyRegressor', 'FuzzyClassifier','RegressorMathModel','ClassifierMathModel', '__version__']
