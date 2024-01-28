from .hroch import SymbolicSolver
from sklearn.base import RegressorMixin
import numpy as numpy
from typing import Iterable


class SymbolicRegressor(SymbolicSolver, RegressorMixin):
    """
    SymbolicRegressor class

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

    problem : str or dict, default='math'
        Predefined instructions sets 'math' or 'simple' or 'fuzzy' or custom defines set of instructions with mutation probability.
        ```python
        problem={'add':10.0, 'mul':10.0, 'gt':1.0, 'lt':1.0, 'nop':1.0}
        ```

        |**supported instructions**||
        |-|-|
        |**math**|add, sub, mul, div, pdiv, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
        |**goniometric**|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
        |**other**|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
        |**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

        *nop - no operation*

        *pdiv - protected division*

        *inv - inverse* $(-x)$

        *minv - multiplicative inverse* $(1/x)$

        *lt, gt, lte, gte -* $<, >, <=, >=$

    feature_probs : array of shape (n_features,), default=None
        The probability that a mutation will select a feature.

    random_state : int, default=0
        Random generator seed. If 0 then random generator will be initialized by system time.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    metric : str, default='MSE'
            Metric used for evaluating error. Choose from {'MSE', 'MAE', 'MSLE', 'LogLoss'}

    transformation : str, default=None
        Final transformation for computed value. Choose from { None, 'LOGISTIC', 'ORDINAL'}

    algo_settings : dict, default = None
        If not defined SymbolicSolver.ALGO_SETTINGS is used.
        ```python
        algo_settings = {'neighbours_count':15, 'alpha':0.15, 'beta':0.5, 'pretest_size':1, 'sample_size':16}
        ```
        - 'neighbours_count' : (int) Number tested neighbours in each iteration
        - 'alpha' : (float) Score worsening limit for a iteration
        - 'beta' : (float) Tree breadth-wise expanding factor in a range from 0 to 1
        - 'pretest_size' : (int) Batch count(batch is 64 rows sample) for fast fitness preevaluating
        - 'sample_size : (int) Number of batches of sample used to calculate the score during training

    code_settings : dict, default = None
        If not defined SymbolicSolver.CODE_SETTINGS is used.
        ```python
        code_settings = {'min_size': 32, 'max_size':32, 'const_size':8}
        ```
        - 'const_size' : (int) Maximum alloved constants in symbolic model, accept also 0.
        - 'min_size': (int) Minimum allowed equation size(as a linear program).
        - 'max_size' : (int) Maximum allowed equation size(as a linear program).
        
    population_settings : dict, default = None
        If not defined SymbolicSolver.POPULATION_SETTINGS is used.
        ```python
        population_settings = {'size': 64, 'tournament':4}
        ```
        - 'size' : (int) Number of individuals in the population.
        - 'tournament' : (int) Tournament selection.

    init_const_settings : dict, default = None
        If not defined SymbolicSolver.INIT_CONST_SETTINGS is used.
        ```python
        init_const_settings = {'const_min':-1.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for initializing constants.
        - 'const_max' : (float) Upper range for initializing constants.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during initialization.
        - 'predefined_const_set' : (array of floats) Predefined constants used during initialization.

    const_settings : dict, default = None
        If not defined SymbolicSolver.CONST_SETTINGS is used.
        ```python
        const_settings = {'const_min':-LARGE_FLOAT, 'const_max':LARGE_FLOAT, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for constants used in equations.
        - 'const_max' : (float) Upper range for constants used in equations.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during search process(mutation).
        - 'predefined_const_set' : (array of floats) Predefined constants used during search process(mutation).

    target_clip : array of two float values clip_min and clip_max, default None
        ```python
        target_clip=[-1, 1]
        ```

    cv_params : dict, default = None
        If not defined SymbolicSolver.REGRESSION_CV_PARAMS is used
        ```python
        cv_params = {'n':0, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'Nelder-Mead'}, 'opt_metric':make_scorer(mean_squared_error, greater_is_better=False)}
        ```
        - 'n' : (int) Crossvalidate n top models
        - 'cv_params' : (dict) Parameters passed to scikit-learn cross_validate method
        - select : (str) Best model selection method choose from 'mean'or 'median'
        - opt_params : (dict) Parameters passed to scipy.optimize.minimize method
        - opt_metric : (make_scorer) Scoring method
    """

    def __init__(self,
                 num_threads: int = 1,
                 time_limit: float = 5.0,
                 iter_limit: int = 0,
                 precision: str = 'f32',
                 problem = 'math',
                 feature_probs = None,
                 random_state: int = 0,
                 verbose: int = 0,
                 metric: str = 'MSE',
                 transformation: str = None,
                 algo_settings = None,
                 code_settings = None,
                 population_settings = None,
                 init_const_settings = None,
                 const_settings = None,
                 target_clip: Iterable = None,
                 cv_params = None,
                 warm_start : bool = False
                 ):
        super(SymbolicRegressor, self).__init__(
            num_threads=num_threads,
            time_limit=time_limit,
            iter_limit=iter_limit,
            precision=precision,
            problem=problem,
            feature_probs=feature_probs,
            random_state=random_state,
            verbose=verbose,
            metric=metric,
            transformation=transformation,
            algo_settings=algo_settings,
            code_settings=code_settings,
            population_settings=population_settings,
            init_const_settings=init_const_settings,
            const_settings=const_settings,
            target_clip=target_clip,
            class_weight=None,
            cv_params=cv_params,
            warm_start=warm_start
        )

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the symbolic models according to the given training data. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
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

        super(SymbolicRegressor, self).fit(X, y, sample_weight=sample_weight, check_input=check_input)
        return self
    
    def predict(self, X, id=None, check_input=True, use_parsed_model=True):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        id : int
            Model id, default=None. id can be obtained from get_models method. If its None prediction use the best model.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        return super(SymbolicRegressor, self).predict(X, id=id, check_input=check_input, use_parsed_model=use_parsed_model)
