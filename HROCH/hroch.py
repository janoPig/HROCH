import os
import numpy as numpy
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate
import scipy.optimize as opt
import ctypes
import platform
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Windows":
    lib_path = os.path.join(current_dir, "hroch.dll")
else:
    lib_path = os.path.join(current_dir, "hroch.bin")

lib = ctypes.cdll.LoadLibrary(lib_path)

FloatPointer = numpy.ctypeslib.ndpointer(
    dtype=numpy.float32, ndim=1, flags='C')
FloatDPointer = numpy.ctypeslib.ndpointer(
    dtype=numpy.float32, ndim=2, flags='C')
DoublePointer = numpy.ctypeslib.ndpointer(
    dtype=numpy.float64, ndim=1, flags='C')
DoubleDPointer = numpy.ctypeslib.ndpointer(
    dtype=numpy.float64, ndim=2, flags='C')


class Params(ctypes.Structure):
    _fields_ = [("random_state", ctypes.c_ulonglong),
                ("num_threads", ctypes.c_uint),
                ("precision", ctypes.c_uint),
                ("pop_size", ctypes.c_uint),
                ("transformation", ctypes.c_uint),
                ("clip_min", ctypes.c_double),
                ("clip_max", ctypes.c_double),
                ("input_size", ctypes.c_uint),
                ("const_size", ctypes.c_uint),
                ("code_min_size", ctypes.c_uint),
                ("code_max_size", ctypes.c_uint),
                ("init_const_min", ctypes.c_double),
                ("init_const_max", ctypes.c_double),
                ("init_predefined_const_prob", ctypes.c_double),
                ("init_predefined_const_count", ctypes.c_uint),
                ("init_predefined_const_set", DoublePointer)]


class FitParams(ctypes.Structure):
    _fields_ = [("time_limit", ctypes.c_uint),
                ("verbose", ctypes.c_uint),
                ("pop_sel", ctypes.c_uint),
                ("metric", ctypes.c_uint),
                ("pretest_size", ctypes.c_uint),
                ("sample_size", ctypes.c_uint),
                ("neighbours_count", ctypes.c_uint),
                ("alpha", ctypes.c_double),
                ("beta", ctypes.c_double),
                ("iter_limit", ctypes.c_ulonglong),
                ("const_min", ctypes.c_double),
                ("const_max", ctypes.c_double),
                ("predefined_const_prob", ctypes.c_double),
                ("predefined_const_count", ctypes.c_uint),
                ("predefined_const_set", DoublePointer),
                ('problem', ctypes.c_char_p),
                ('feature_probs', ctypes.c_char_p),
                ("cw0", ctypes.c_double),
                ("cw1", ctypes.c_double),
                ]


class PredictParams(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64),
                ("verbose", ctypes.c_uint32),
                ]


class MathModel(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64),
                ("score", ctypes.c_double),
                ("partial_score", ctypes.c_double),
                ('str_representation', ctypes.c_char_p),
                ('str_code_representation', ctypes.c_char_p),
                ("used_constants_count", ctypes.c_uint32),
                ("used_constants", ctypes.POINTER(ctypes.c_double)),
                ]

def apply_const(s, c):
    pattern = r'\b(c\d+)\b'

    def replace_coef(match):
        symbol = match.group(1)
        index = int(symbol[1:])
        value = c[index]
        if value < 0:
            return f'({value})'
        return str(value)
    
    return re.sub(pattern, replace_coef, s)

class ParsedMathModel:
    def __init__(self, m: MathModel) -> None:
        self.id = m.id
        self.score = m.score
        self.partial_score = m.partial_score
        self.str_representation = m.str_representation.decode('ascii')
        self.str_code_representation = m.str_code_representation.decode(
            'ascii')
        self.coeffs = numpy.zeros(m.used_constants_count)
        for i in range(m.used_constants_count):
            self.coeffs[i] = m.used_constants[i]

        self.method_name = self.str_code_representation[4:self.str_code_representation.find(
            '(')]
        exec(self.str_code_representation)
        self.method = locals()[self.method_name]
        
    def __getstate__(self):
        all_attributes = self.__dict__.copy()
        all_attributes.pop('method', None)
        return all_attributes
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        exec(self.str_code_representation)
        self.method = locals()[self.method_name]


class MathModelBase(BaseEstimator):
    """
    Base class for RegressorMathModel and ClassifierMathModel
    """
    def __init__(self, m: ParsedMathModel, parent_params) -> None:
        self.m = m
        self.parent_params = parent_params
        cv_params = parent_params.get('cv_params')
        self.opt_metric = cv_params.get('opt_metric')
        self.opt_params = cv_params.get('opt_params')
        self.transformation = None if parent_params['transformation'] is None else parent_params['transformation'].upper()
        self.target_clip = self.parent_params['target_clip']
        if self.target_clip is not None and (len(self.target_clip) != 2 or self.target_clip[0] >= self.target_clip[1]):
            self.target_clip = None
        self.class_weight_ = parent_params.get('class_weight_')
        self.classes_ = parent_params.get('classes_')
        self.is_fitted_ = True

    def _predict(self, X, c=None, transform=True, check_input=True):
        check_is_fitted(self)
        if check_input:
            X = check_array(X, accept_sparse=False)

        preds = self.m.method(X, self.m.coeffs if c is None else c)

        if type(preds) is not numpy.ndarray:
            preds = numpy.full(len(X), preds)

        return self.__transform(preds) if transform else preds

    def __transform(self, y):
        if self.transformation is not None:
            if self.transformation == 'LOGISTIC':
                y = 1.0/(1.0+numpy.exp(-numpy.clip(y,a_min=-20.0, a_max=20.0)))
            elif self.transformation == 'ORDINAL':
                y = numpy.round(y)

        if self.target_clip is not None:
            y = numpy.clip(y, self.target_clip[0], self.target_clip[1])

        return y
    
    @property
    def equation(self):
        return apply_const(self.m.str_representation, self.m.coeffs)


class RegressorMathModel(MathModelBase, RegressorMixin):
    """
    A regressor class for the symbolic model.
    """
    def __init__(self, m: ParsedMathModel, parent_params) -> None:
        super().__init__(m, parent_params)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the model according to the given training data. 
        
        That means find a optimal values for constants in a symbolic equation.

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

        def objective(c):
            return self.__eval(X, y, metric=self.opt_metric, c=c, sample_weight=sample_weight)

        if len(self.m.coeffs) > 0:
            result = opt.minimize(objective, self.m.coeffs, **self.opt_params)

            for i in range(len(self.m.coeffs)):
                self.m.coeffs[i] = result.x[i]

        self.is_fitted_ = True
        return self

    def predict(self, X, check_input=True):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        return self._predict(X)
    
    def __eval(self, X, y, metric, c=None, sample_weight=None):
        if c is not None:
            self.m.coeffs = c
        return -metric(self, X, y, sample_weight=sample_weight)
    
    def __str__(self):
        return f"RegressorMathModel({self.m.str_representation})"
    
    def __repr__(self):
        return f"RegressorMathModel({self.m.str_representation})"


class ClassifierMathModel(MathModelBase, ClassifierMixin):
    """
    A classifier class for the symbolic model.
    """
    def __init__(self, m: ParsedMathModel, parent_params) -> None:
        super().__init__(m, parent_params)

    def eval(self, X, y, metric, c=None, sample_weight=None):
        if c is not None:
            self.m.coeffs = c
        return -metric(self, X, y, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the model according to the given training data. 
        
        That means find a optimal values for constants in a symbolic equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X. Needs samples of 2 classes.

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

        check_classification_targets(y)
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ != 2:
            raise ValueError(
                "This solver needs samples of 2 classes"
                " in the data, but the data contains"
                " %r classes"
                % self.n_classes_
            )
        
        cw = self.class_weight_
        cw_sample_weight = numpy.array(cw)[y_ind] if len(cw) == 2 and cw[0] != cw[1] else None
        if sample_weight is None:
            sample_weight = cw_sample_weight
        elif cw_sample_weight is not None:
            sample_weight = sample_weight*cw_sample_weight

        def objective(c):
            return self.eval(X, y, metric=self.opt_metric, c=c, sample_weight=sample_weight)

        if len(self.m.coeffs) > 0:
            result = opt.minimize(objective, self.m.coeffs,
                                  **self.opt_params)

            for i in range(len(self.m.coeffs)):
                self.m.coeffs[i] = result.x[i]

        self.is_fitted_ = True
        return self

    def predict(self, X, check_input=True):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        preds = self._predict(X, check_input=check_input)
        return self.classes_[(preds > 0.5).astype(int)]

    def predict_proba(self, X, check_input=True):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        preds = self._predict(X, check_input=check_input)
        proba = numpy.vstack([1 - preds, preds]).T
        return proba
    
    def __str__(self):
        return f"ClassifierMathModel({self.m.str_representation})"
    
    def __repr__(self):
        return f"ClassifierMathModel({self.m.str_representation})"


# void * CreateSolver(solver_params * params)
CreateSolver = lib.CreateSolver
CreateSolver.argtypes = [ctypes.POINTER(Params)]
CreateSolver.restype = ctypes.c_void_p

# void DeleteSolver(void *hsolver)
DeleteSolver = lib.DeleteSolver
DeleteSolver.argtypes = [ctypes.c_void_p]
DeleteSolver.restype = None

# int FitData[32/64](void *hsolver, const [float/double] *X, const [float/double] *y, unsigned int rows, unsigned int xcols, fit_params *params, const [float/double] *sw)
FitData32 = lib.FitData32
FitData32.argtypes = [ctypes.c_void_p, FloatDPointer, FloatPointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(FitParams), FloatPointer, ctypes.c_uint]
FitData32.restype = ctypes.c_int

FitData64 = lib.FitData64
FitData64.argtypes = [ctypes.c_void_p, DoubleDPointer, DoublePointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(FitParams), DoublePointer, ctypes.c_uint]
FitData64.restype = ctypes.c_int

# int Predict[32/64](void * hsolver, const [float/double] *X, [float/double] *y, unsigned int rows, unsigned int xcols, predict_params * params)
Predict32 = lib.Predict32
Predict32.argtypes = [ctypes.c_void_p, FloatDPointer, FloatPointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(PredictParams)]
Predict32.restype = ctypes.c_int

Predict64 = lib.Predict64
Predict64.argtypes = [ctypes.c_void_p, DoubleDPointer, DoublePointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(PredictParams)]
Predict64.restype = ctypes.c_int

# int GetBestModel(void *hsolver, math_model *model);
GetBestModel = lib.GetBestModel
GetBestModel.argtypes = [ctypes.c_void_p, ctypes.POINTER(MathModel)]
GetBestModel.restype = ctypes.c_int

# int GetModel(void *hsolver, unsigned long long id, math_model *model)
GetModel = lib.GetModel
GetModel.argtypes = [ctypes.c_void_p,
                     ctypes.c_uint64, ctypes.POINTER(MathModel)]
GetModel.restype = ctypes.c_int

# void FreeModel(math_model *model)
FreeModel = lib.FreeModel
FreeModel.argtypes = [ctypes.POINTER(MathModel)]
FreeModel.restype = None

class SymbolicSolver(BaseEstimator):
    """
    Symbolic regression base for SymbolicRegressor, NonlinearLogisticRegressor and FuzzyRegressor

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
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

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

    LARGE_FLOAT = 1e30

    SIMPLE = {'nop': 0.01, 'add': 1.0, 'sub': 1.0,
          'mul': 1.0, 'div': 0.1, 'sq2': 0.05}

    MATH = {'nop': 0.01, 'add': 1.0, 'sub': 1.0, 'mul': 1.0,
            'div': 0.1, 'sq2': 0.05, 'pow': 0.001, 'exp': 0.001,
            'log': 0.001, 'sqrt': 0.1, 'sin': 0.005, 'cos': 0.005,
            'tan': 0.001, 'asin': 0.001, 'acos': 0.001, 'atan': 0.001,
            'sinh': 0.001, 'cosh': 0.001, 'tanh': 0.001}

    FUZZY = {'nop': 0.01, 'f_and': 1.0, 'f_or': 1.0, 'f_xor': 1.0, 'f_not': 1.0}
    
    ALGO_SETTINGS = {'neighbours_count':15, 'alpha':0.15, 'beta':0.5, 'pretest_size':1, 'sample_size':16}
    CODE_SETTINGS = {'min_size': 32, 'max_size':32, 'const_size':8}
    POPULATION_SETTINGS = {'size': 64, 'tournament':4}
    INIT_CONST_SETTINGS = {'const_min':-1.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
    CONST_SETTINGS = {'const_min':-LARGE_FLOAT, 'const_max':LARGE_FLOAT, 'predefined_const_prob':0.0, 'predefined_const_set': []}
    REGRESSION_CV_PARAMS = {'n':0, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'Nelder-Mead'}, 'opt_metric':make_scorer(mean_squared_error, greater_is_better=False)}
    CLASSIFICATION_CV_PARAMS = {'n':0, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'Nelder-Mead'}, 'opt_metric':make_scorer(log_loss, greater_is_better=False, needs_proba=True)}
    REGRESSION_TARGET_CLIP = [0.0, 0.0]
    CLASSIFICATION_TARGET_CLIP = [3e-7, 1.0-3e-7]

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
                 target_clip = None,
                 class_weight = None,
                 cv_params = None,
                 warm_start : bool = False
                 ):

        if precision not in ['f32', 'f64']:
            raise ValueError("precision parameter must be 'f32' or 'f64'")

        if num_threads <= 0:
            raise ValueError(
                "num_threads parameter must be greather that zero")

        self.num_threads = num_threads
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.verbose = verbose
        self.precision = precision
        self.problem = problem
        self.feature_probs = feature_probs
        self.random_state = random_state
        self.code_settings = code_settings
        self.population_settings = population_settings
        self.metric = metric
        self.transformation = transformation
        self.algo_settings = algo_settings
        self.init_const_settings = init_const_settings
        self.const_settings = const_settings
        self.target_clip = target_clip
        self.class_weight = class_weight
        self.cv_params = cv_params
        self.warm_start = warm_start

    def __del__(self):
        if hasattr(self, "handle_") and self.handle_ is not None:
            DeleteSolver(self.handle_)

    @property
    def equation(self):
        return self.sexpr_

    def fit(self, X, y, sample_weight = None, check_input=True):
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
        
        if not self.warm_start and hasattr(self,'handle_'):
            if self.handle_ is not None:
                DeleteSolver(self.handle_)
                self.handle_ = None
            self.is_fitted_ = False

        def val(d, key, v):
            if d is not None and key in d:
                return d[key]
            return v
        
        if check_input:
            X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=True, multi_output=False)
            
        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype, only_non_negative=True
            )
            
        algo_settings = self.algo_settings if self.algo_settings is not None else self.ALGO_SETTINGS,
        code_settings = self.code_settings if self.code_settings is not None else self.CODE_SETTINGS
        population_settings = self.population_settings if self.population_settings is not None else self.POPULATION_SETTINGS
        init_const_settings = self.init_const_settings if self.init_const_settings is not None else self.INIT_CONST_SETTINGS
        const_settings = self.const_settings if self.const_settings is not None else self.CONST_SETTINGS
        target_clip_ = self.REGRESSION_TARGET_CLIP if self._estimator_type == 'regressor' else self.CLASSIFICATION_TARGET_CLIP
        target_clip = self.target_clip if self.target_clip is not None else target_clip_
        cv_params_ = self.REGRESSION_CV_PARAMS if self._estimator_type == 'regressor' else self.CLASSIFICATION_CV_PARAMS
        cv_params = self.cv_params if self.cv_params is not None else cv_params_

        if not hasattr(self, "handle_") or self.handle_ is None:
            tmp = val(init_const_settings, 'predefined_const_set', [])
            init_predefined_const_set = numpy.ascontiguousarray(tmp).astype('float64').ctypes.data_as(DoublePointer) if len(tmp) > 0 else None
            params = Params(random_state=self.random_state,
                            num_threads=self.num_threads,
                            precision=1 if self.precision == 'f32' else 2,
                            pop_size=val(population_settings,'size', 64),
                            transformation=self.__parse_transformation(
                                self.transformation),
                            clip_min=target_clip[0],
                            clip_max=target_clip[1],
                            input_size=X.shape[1],
                            const_size=val(code_settings, 'const_size', 8),
                            code_min_size=val(code_settings, 'min_size', 32),
                            code_max_size=val(code_settings, 'max_size', 32),
                            init_const_min=val(init_const_settings, 'const_min', -1.0),
                            init_const_max=val(init_const_settings, 'const_max', 1.0),
                            init_predefined_const_prob=val(init_const_settings, 'predefined_const_prob', 0.0),
                            init_predefined_const_count=0 if init_predefined_const_set is None else len(init_predefined_const_set),
                            init_predefined_const_set=init_predefined_const_set,
                            )
            self.handle_ = CreateSolver(ctypes.pointer(params))

        _x = numpy.ascontiguousarray(X.T).astype(
            'float32' if self.precision == 'f32' else 'float64')
        _y = numpy.ascontiguousarray(
            y.astype('float32' if self.precision == 'f32' else 'float64'))
        
        _sw = numpy.ndarray(shape=(0,), dtype=numpy.float32 if self.precision == 'f32' else numpy.float64)
        _sw_len = 0 if sample_weight is None else len(sample_weight)
        if sample_weight is not None:
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight len incorrect")
            _sw = numpy.ascontiguousarray(sample_weight.astype('float32' if self.precision == 'f32' else 'float64'))

        tmp = val(const_settings, 'predefined_const_set', [])
        predefined_const_set = numpy.ascontiguousarray(tmp).astype('float64').ctypes.data_as(DoublePointer) if len(tmp) > 0 else None

        fit_params = FitParams(
            time_limit=round(self.time_limit*1000),
            verbose=self.verbose,
            pop_sel=val(population_settings, 'tournament', 4),
            metric=self.__parse_metric(self.metric),
            pretest_size=val(algo_settings, 'pretest_size', 1),
            sample_size=val(algo_settings, 'sample_size', 16),
            neighbours_count=val(algo_settings, 'neighbours_count', 15),
            alpha=val(algo_settings, 'alpha', 0.15),
            beta=val(algo_settings, 'beta', 0.5),
            iter_limit=self.iter_limit,
            const_min=val(const_settings, 'const_min', -self.LARGE_FLOAT),
            const_max=val(const_settings, 'const_max', self.LARGE_FLOAT),
            predefined_const_prob=val(const_settings, 'predefined_const_prob', 0.0),
            predefined_const_count=0 if predefined_const_set is None else len(predefined_const_set),
            predefined_const_set=predefined_const_set,
            problem=self.__problem_to_string(self.problem).encode('utf-8'),
            feature_probs=self.__feature_probs_to_string(
                self.feature_probs).encode('utf-8'),
            cw0=self.class_weight_[0] if hasattr(self, 'class_weight_') else 1.0,
            cw1=self.class_weight_[1] if hasattr(self, 'class_weight_') else 1.0,
        )

        if self.precision == 'f32':
            ret = FitData32(self.handle_, _x, _y,
                            X.shape[0], X.shape[1], ctypes.pointer(fit_params), _sw, _sw_len)
        elif self.precision == 'f64':
            ret = FitData64(self.handle_, _x, _y,
                            X.shape[0], X.shape[1], ctypes.pointer(fit_params), _sw, _sw_len)

        self.is_fitted_ = True if ret == 0 else False

        if not self.is_fitted_:
            return
        
        n = cv_params['n']
        cv_select = cv_params['select']
        opt_metric = cv_params['opt_metric']
        self.models_ = self.__get_models()
        if n > 0:
            invalid_score = self.LARGE_FLOAT*(-opt_metric._sign)
            i = 0
            for m in self.models_:
                i = i + 1
                if i > n:
                    m.cv_score = invalid_score
                    continue
                try:
                    m.cv_results = cross_validate(
                        estimator=m, X=X, y=y, n_jobs=None, error_score=invalid_score, scoring=cv_params['opt_metric'], **cv_params['cv_params'])
                    if cv_select == 'mean':
                        m.cv_score = numpy.mean(m.cv_results['test_score'])
                    elif cv_select == 'median':
                        m.cv_score = numpy.median(m.cv_results['test_score'])
                except Exception:
                    m.cv_score = invalid_score
                
                if numpy.isnan(m.cv_score):
                    m.cv_score = invalid_score

                m.cv_score *= opt_metric._sign

                # fit final coeffs from whole data
                m.fit(X, y, check_input=False)

            self.models_.sort(
                key=lambda x: x.cv_score*(-opt_metric._sign))

        self.sexpr_ = self.models_[0].equation

    def predict(self, X, id=None, check_input=True, use_parsed_model=True):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        id : int
            Model id, default=None. id can be obtained from get_models method. If its none prediction use best model.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)

        if use_parsed_model:
            m = self.models_[0] if id is None else next(
                (x for x in self.models_ if x.m.id == id), None)
            return m._predict(X, check_input=check_input)
        else:
            return self._predict(X, id, check_input=check_input)
        
    def get_models(self):
        """
        Get population of symbolic models.

        Returns
        -------
        models : array of RegressorMathModel or ClassifierMathModel
        """
        check_is_fitted(self)
        if not hasattr(self, 'models_'):
            self.models_ = self.__get_models()
        return self.models_

    def _predict(self, X, id=None, check_input=True):
        check_is_fitted(self)

        if check_input:
            X = check_array(X, accept_sparse=False)

        _x = numpy.ascontiguousarray(X.T).astype(
            'float32' if self.precision == 'f32' else 'float64')
        _y = numpy.ascontiguousarray(
            numpy.zeros(X.shape[0], dtype=numpy.float32 if self.precision == 'f32' else numpy.float64))

        params = PredictParams(
            id if id is not None else 0xffffffff, self.verbose)

        if self.precision == 'f32':
            Predict32(self.handle_, _x, _y,
                      X.shape[0], X.shape[1], ctypes.pointer(params))
        elif self.precision == 'f64':
            Predict64(self.handle_, _x, _y,
                      X.shape[0], X.shape[1], ctypes.pointer(params))
        return _y

    def __get_models(self):
        models = []
        population_settings = self.population_settings if self.population_settings is not None else self.POPULATION_SETTINGS
        for i in range(self.num_threads*population_settings['size']):
            model = MathModel()
            GetModel(self.handle_, i, model)
            models.append(self.__create_model(model))
            FreeModel(model)
        return sorted(models, key=lambda x: x.m.score)

    def __problem_to_string(self, problem):
        if isinstance(problem, str):
            if problem == 'simple' or problem == 'math' or problem == 'fuzzy':
                return problem
            raise ValueError('Invalid problem type')
        if not isinstance(problem, dict):
            raise TypeError('Invalid problem type')
        result = ""
        for instr, prob in problem.items():
            if not isinstance(instr, str) or not isinstance(prob, float):
                raise TypeError('Invalid instruction type')
            if len(instr) == 0 or prob < 0.0:
                raise ValueError('Invalid instruction value')
            result = result + f'{instr} {prob};'
        return result

    def __feature_probs_to_string(self, feat):
        result = ""
        if feat is None:
            return result
        for prob in feat:
            result = result + f'{prob};'
        return result

    def __parse_metric(self, metric: str):
        if metric is None:
            return 0

        metric = metric.upper()
        if metric == 'MSE':
            return 0
        elif metric == 'MAE':
            return 1
        elif metric == 'MSLE':
            return 2
        elif metric == 'LOGLOSS':
            return 4
        elif metric == 'LOGITAPPROX':
            return 20
        return 0

    def __parse_transformation(self, transformation: str):
        if transformation is None:
            return 0

        transformation = transformation.upper()
        if transformation == 'NONE':
            return 0
        elif transformation == 'LOGISTIC':
            return 1
        elif transformation == 'PSEUDOLOG':
            return 2
        elif transformation == 'ORDINAL':
            return 3
        return 0

    def __create_model(self, m: MathModel):
        attrib = self.__dict__.copy()
        attrib['algo_settings'] = self.algo_settings if self.algo_settings is not None else self.ALGO_SETTINGS,
        attrib['code_settings'] = self.code_settings if self.code_settings is not None else self.CODE_SETTINGS
        attrib['population_settings'] = self.population_settings if self.population_settings is not None else self.POPULATION_SETTINGS
        attrib['init_const_settings'] = self.init_const_settings if self.init_const_settings is not None else self.INIT_CONST_SETTINGS
        attrib['const_settings'] = self.const_settings if self.const_settings is not None else self.CONST_SETTINGS
        target_clip_ = self.REGRESSION_TARGET_CLIP if self._estimator_type == 'regressor' else self.CLASSIFICATION_TARGET_CLIP
        attrib['target_clip'] = self.target_clip if self.target_clip is not None else target_clip_
        cv_params_ = self.REGRESSION_CV_PARAMS if self._estimator_type == 'regressor' else self.CLASSIFICATION_CV_PARAMS
        attrib['cv_params'] = self.cv_params if self.cv_params is not None else cv_params_
        if self._estimator_type == 'regressor':
            return RegressorMathModel(ParsedMathModel(m), attrib)
        else:
            return ClassifierMathModel(ParsedMathModel(m), attrib)
        
    def __getstate__(self):
        all_attributes = self.__dict__.copy()
        # disable warm_start because handle is not valid
        all_attributes['warm_start'] = False
        # store models
        if hasattr(self,'is_fitted_') and self.is_fitted_:
            if not hasattr(self, 'models_'):
                all_attributes['models_'] = self.get_models()
        all_attributes.pop('handle_', None)
        return all_attributes
