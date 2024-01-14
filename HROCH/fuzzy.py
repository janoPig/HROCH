from sklearn.multiclass import OneVsRestClassifier
from .hroch import SymbolicSolver
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, make_scorer
import numpy as numpy
from typing import Iterable

class FuzzyRegressor(SymbolicSolver, ClassifierMixin):
    """
    Fuzzy Regressor

    Parameters
    ----------
    num_threads : int, default=1
        Number of used threads.

    time_limit : float, default=5.0
        Timeout in seconds. If is set to 0 there is no limit and the algorithm runs until iter_limit is met.

    iter_limit : int, default=0
        Iterations limit. If is set to 0 there is no limit and the algorithm runs until time_limit is met.

    precision : str, default='f32'
        'f64' or 'f32'. Internal floating number representation.

    problem : str or dict, default='fuzzy'
        Predefined instructions sets 'fuzzy' or custom defines set of instructions with mutation probability.
        ```python
        problem={'f_and':10.0, 'f_or':10.0, 'f_xor':1.0, 'f_not':1.0, 'nop':1.0}
        ```

        |**supported instructions**||
        |-|-|
        |**other**|nop|
        |**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

    feature_probs : array of shape (n_features,), default=None
        The probability that a mutation will select a feature.

    random_state : int, default=0
        Random generator seed. If 0 then random generator will be initialized by system time.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    metric : str, default='LogLoss'
        Metric used for evaluating error. Choose from {'MSE', 'MAE', 'MSLE', 'LogLoss'}

    transformation : str, default='LOGISTIC'
        Final transformation for computed value. Choose from { None, 'LOGISTIC', 'ORDINAL'}
        
    algo_settings : dict,  default = SymbolicSolver.ALGO_SETTINGS
        ```python
        algo_settings = {'neighbours_count':15, 'alpha':0.15, 'beta':0.5}
        ```
        - 'neighbours_count' : (int) Number tested neighbours in each iteration
        - 'alpha' : (float) Score worsening limit for a iteration
        - 'beta' : (float) Tree breadth-wise expanding factor in a range from 0 to 1

    code_settings : dict, default SymbolicSolver.CODE_SETTINGS
        ```python
        code_settings = {'min_size': 32, 'max_size':32, 'const_size':8}
        ```
        - 'const_size' : (int) Maximum alloved constants in symbolic model, accept also 0.
        - 'min_size': (int) Minimum allowed equation size(as a linear program).
        - 'max_size' : (int) Maximum allowed equation size(as a linear program).
        
    population_settings : dict, default SymbolicSolver.POPULATION_SETTINGS
        ```python
        population_settings = {'size': 64, 'tournament':4}
        ```
        - 'size' : (int) Number of individuals in the population.
        - 'tournament' : (int) Tournament selection.

    init_const_settings : dict, default FuzzyRegressor.INIT_CONST_SETTINGS
        ```python
        init_const_settings = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for initializing constants.
        - 'const_max' : (float) Upper range for initializing constants.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during initialization.
        - 'predefined_const_set' : (array of floats) Predefined constants used during initialization.

    const_settings : dict, default FuzzyRegressor.CONST_SETTINGS
        ```python
        const_settings = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for constants used in equations.
        - 'const_max' : (float) Upper range for constants used in equations.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during search process(mutation).
        - 'predefined_const_set' : (array of floats) Predefined constants used during search process(mutation).

    target_clip : array, default SymbolicSolver.CLASSIFICATION_TARGET_CLIP
        Array of two float values clip_min and clip_max, default None
        ```python
        target_clip=[3e-7, 1.0-3e-7]
        ```
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    cv_params : dict, default SymbolicSolver.CLASSIFICATION_CV_PARAMS
        ```python
        cv_params = {'n':0, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'Nelder-Mead'}, 'opt_metric':make_scorer(log_loss, greater_is_better=False, needs_proba=True)}
        ```
        - 'n' : (int) Crossvalidate n top models
        - 'cv_params' : (dict) Parameters passed to scikit-learn cross_validate method
        - select : (str) Best model selection method choose from 'mean'or 'median'
        - opt_params : (dict) Parameters passed to scipy.optimize.minimize method
        - opt_metric : (make_scorer) Scoring method
    """
    
    FUZZY_INIT_CONST_SETTINGS = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
    FUZZY_CONST_SETTINGS = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}

    def __init__(self,
                 num_threads: int = 1,
                 time_limit: float = 5.0,
                 iter_limit: int = 0,
                 precision: str = 'f32',
                 problem: any = 'fuzzy',
                 feature_probs: any = None,
                 random_state: int = 0,
                 verbose: int = 0,
                 metric: str = 'LogLoss',
                 transformation: str = 'LOGISTIC',
                 algo_settings : dict = SymbolicSolver.ALGO_SETTINGS,
                 code_settings : dict = SymbolicSolver.CODE_SETTINGS,
                 population_settings: dict = SymbolicSolver.POPULATION_SETTINGS,
                 init_const_settings : dict = FUZZY_INIT_CONST_SETTINGS,
                 const_settings : dict = FUZZY_CONST_SETTINGS,
                 target_clip: Iterable = SymbolicSolver.CLASSIFICATION_TARGET_CLIP,
                 class_weight = None,
                 cv_params=SymbolicSolver.CLASSIFICATION_CV_PARAMS
                 ):
        super(FuzzyRegressor, self).__init__(
            num_threads=num_threads,
            time_limit=time_limit,
            iter_limit=iter_limit,
            precision=precision,
            problem=problem,
            feature_probs=feature_probs,
            random_state=random_state,
            verbose=verbose,
            metric=metric,
            algo_settings=algo_settings,
            transformation=transformation,
            code_settings=code_settings,
            population_settings=population_settings,
            init_const_settings=init_const_settings,
            const_settings=const_settings,
            target_clip=target_clip,
            class_weight=class_weight,
            cv_params=cv_params
        )

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, sample_weight=None, check_input=True):
        """
        Fit the symbolic models according to the given training data. 

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. Should be in the range [0, 1].

        y : numpy.ndarray of shape (n_samples,)
            Target vector relative to X. Needs samples of 2 classes.

        sample_weight : numpy.ndarray of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        self
            Fitted estimator.
        """

        self.classes_ = numpy.unique(y)
        self.n_classes_ = len(self.classes_)

        super(FuzzyRegressor, self).fit(X, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X: numpy.ndarray, id=None, check_input=True):
        """
        Predict class for X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input samples.
            
        id : int
            Model id, default=None. id can be obtained from get_models method. If its None prediction use the best model.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : numpy.ndarray of shape (n_samples,)
            The predicted classes.
        """
        preds = super(FuzzyRegressor, self).predict(X, id, check_input=check_input)
        return (preds > 0.5)*1.0

    def predict_proba(self, X: numpy.ndarray, id=None, check_input=True):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        preds = super(FuzzyRegressor, self).predict(X, id, check_input=check_input)
        proba = numpy.vstack([1 - preds, preds]).T
        return proba


class FuzzyClassifier(OneVsRestClassifier):
    """
    Fuzzy multiclass symbolic classificator
    
    Parameters
    ----------
    kwargs : Any
        Parameters passed to [FuzzyRegressor](https://janopig.github.io/HROCH/HROCH.html#FuzzyRegressor) estimator
        
    verbose : int
        Verbosity level for OneVsRestClassifier

    """
    def __init__(self, verbose=0, **kwargs):
        super().__init__(estimator=FuzzyRegressor(**kwargs), verbose=verbose)
        
    
    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """
        Fit the symbolic models according to the given training data. 

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. Should be in the range [0, 1].

        y : numpy.ndarray of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """

        super(OneVsRestClassifier, self).fit(X, y)
        return self

    def predict(self, X: numpy.ndarray):
        """
        Predict class for X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_samples,)
            The predicted classes.
        """
        return super(OneVsRestClassifier, self).predict(X)

    def predict_proba(self, X: numpy.ndarray):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return super(OneVsRestClassifier, self).predict_proba(X)
