
import os
import numpy as numpy
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import log_loss, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate
import scipy.optimize as opt
import ctypes
import platform
import re

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


class MathModelBase(BaseEstimator):
    def __init__(self, m: ParsedMathModel, parent_params, opt_metric) -> None:
        self.m = m
        self.parent_params = parent_params
        self.opt_metric = opt_metric
        self.is_fitted_ = True

    def _predict(self, X: numpy.ndarray, c=None, transform=True):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        preds = self.m.method(X, self.m.coeffs if c is None else c)

        if type(preds) is not numpy.ndarray:
            preds = numpy.full(len(X), preds)

        return self.__transform(preds) if transform else preds

    def __transform(self, y):
        transformation = self.parent_params['transformation']
        if transformation == 'LOGISTIC':
            y = 1.0/(1.0+numpy.exp(-y))
        elif transformation == 'ORDINAL':
            y = numpy.round(y)

        clip_min = self.parent_params['clip_min']
        clip_max = self.parent_params['clip_max']

        if clip_max > clip_min:
            y = numpy.clip(y, clip_min, clip_max)

        return y
    
    @property
    def equation(self):
        return apply_const(self.m.str_representation, self.m.coeffs)


class RegressorMathModel(MathModelBase, RegressorMixin):
    def __init__(self, m: ParsedMathModel, parent_params, opt_metric) -> None:
        super().__init__(m, parent_params, opt_metric)

    def eval(self, X: numpy.ndarray, y: numpy.ndarray, c=None, metric=mean_squared_error, sample_weight=None):
        preds = self._predict(X, c=self.m.coeffs if c is None else c)

        val = 1e30
        if not numpy.isnan(numpy.sum(preds)):
            val = metric(y, preds, sample_weight=sample_weight)
        return -val if self.parent_params['opt_greater_is_better'] else val

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, sample_weight=None):
        """Fit constants in symbolic model.

        Args:
            - X (numpy.ndarray): Training data.
            - y (numpy.ndarray): Target values.
            - sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
        """

        def objective(c):
            return self.eval(X, y, c=c, metric=self.opt_metric, sample_weight=sample_weight)

        if len(self.m.coeffs) > 0:
            result = opt.minimize(objective, self.m.coeffs,
                                  **self.parent_params['opt_params'])

            for i in range(len(self.m.coeffs)):
                self.m.coeffs[i] = result.x[i]

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray):
        return self._predict(X)


class ClassifierMathModel(MathModelBase, ClassifierMixin):
    def __init__(self, m: ParsedMathModel, parent_params, opt_metric) -> None:
        super().__init__(m, parent_params, opt_metric)

    def eval(self, X: numpy.ndarray, y: numpy.ndarray, c=None, metric=log_loss, sample_weight=None):
        preds = self._predict(X, c=self.m.coeffs if c is None else c)

        val = 1e30
        if not numpy.isnan(numpy.sum(preds)):
            val = metric(y, preds, sample_weight=sample_weight)
        return -val if self.parent_params['opt_greater_is_better'] else val

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, sample_weight=None):
        """Fit constants in symbolic model.

        Args:
            - X (numpy.ndarray): Training data.
            - y (numpy.ndarray): Target values.
            - sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
        """

        if sample_weight is None:
            cw = self.parent_params['cw']
            if len(cw) == 2 and cw[0] != cw[1]:
                sample_weight = numpy.array(cw)[y]

        def objective(c):
            return self.eval(X, y, c=c, metric=self.opt_metric)

        if len(self.m.coeffs) > 0:
            result = opt.minimize(objective, self.m.coeffs,
                                  **self.parent_params['opt_params'])

            for i in range(len(self.m.coeffs)):
                self.m.coeffs[i] = result.x[i]

        self.is_fitted_ = True
        return self

    def predict(self, X: numpy.ndarray):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        preds = self._predict(X)
        return (preds > 0.5)*1.0

    def predict_proba(self, X: numpy.ndarray, id=None):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.
            - id (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

        Returns:
            numpy.ndarray, shape = [n_samples, n_classes]: The class probabilities of the input samples.
        """
        preds = self._predict(X)
        proba = numpy.vstack([1 - preds, preds]).T
        return proba


class ProbaClf(ClassifierMathModel):
    """
    Wrapper for cross validation
    """

    def __init__(self, clf: ClassifierMathModel) -> None:
        super().__init__(clf.m, clf.parent_params, clf.opt_metric)
        self.clf = clf

    def predict(self, X):
        return ClassifierMathModel.predict_proba(self, X)[:, 1]


# void * CreateSolver(solver_params * params)
CreateSolver = lib.CreateSolver
CreateSolver.argtypes = [ctypes.POINTER(Params)]
CreateSolver.restype = ctypes.c_void_p

# void DeleteSolver(void *hsolver)
DeleteSolver = lib.DeleteSolver
DeleteSolver.argtypes = [ctypes.c_void_p]
DeleteSolver.restype = None

# int FitData[32/64](void *hsolver, const [float/double] *X, const [float/double] *y, unsigned int rows, unsigned int xcols, fit_params *params)
FitData32 = lib.FitData32
FitData32.argtypes = [ctypes.c_void_p, FloatDPointer, FloatPointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(FitParams)]
FitData32.restype = ctypes.c_int

FitData64 = lib.FitData64
FitData64.argtypes = [ctypes.c_void_p, DoubleDPointer, DoublePointer,
                      ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(FitParams)]
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

simple = {'nop': 0.01, 'add': 1.0, 'sub': 1.0,
          'mul': 1.0, 'div': 0.1, 'sq2': 0.05}

math = {'nop': 0.01, 'add': 1.0, 'sub': 1.0, 'mul': 1.0,
        'div': 0.1, 'sq2': 0.05, 'pow': 0.001, 'exp': 0.001,
        'log': 0.001, 'sqrt': 0.1, 'sin': 0.005, 'cos': 0.005,
        'tan': 0.001, 'asin': 0.001, 'acos': 0.001, 'atan': 0.001,
        'sinh': 0.001, 'cosh': 0.001, 'tanh': 0.001}

fuzzy = {'nop': 0.01, 'f_and': 1.0, 'f_or': 1.0, 'f_xor': 1.0, 'f_not': 1.0}


class PHCRegressor(BaseEstimator):
    """
    Parallel Hill Climbing symbolic Regression

    Parameters:
        - num_threads (int, optional), default=8:

            Number of used threads.

        - time_limit (float, optional), default=5.0:

            Timeout in seconds. If is set to 0 there is no limit and the algorithm runs until some other condition is met.

        - iter_limit (int, optional), default=0:

            Iterations limit. If is set to 0 there is no limit and the algorithm runs until some other condition is met.

        - precision (str, optional), default='f32':

            'f64' or 'f32'. Internal floating number representation. 32bit AVX2 instructions are 2x faster as 64bit.

        - problem (any, optional), default='math':

            Predefined instructions sets 'mat' or 'simple' or 'fuzzy' or custom defines set of instructions with mutation probability.
            `reg = PHCRegressor(problem={'add':10.0, 'mul':10.0, 'gt':1.0, 'lt':1.0, 'nop':1.0})`

        - feature_probs (any, optional), default=None:

            `reg = PHCRegressor(feature_probs=[1.0,1.0, 0.01])`

        - random_state (int, optional), default=0:

            Random generator seed. If 0 then random generator will be initialized by system time.

        - verbose (int, optional), default=0:

            Controls the verbosity when fitting and predicting.

        - pop_size (int, optional), default=64:

            Number of individuals in the population.

        - pop_sel (int, optional), default=4:

            Tournament selection.

        - const_size (int, optional), default=8:

            Maximum alloved constants in symbolic model, accept also 0.

        - code_min_size (int, optional), default=32:

            Minimum allowed equation size.

        - code_max_size (int, optional), default=32:

            Maximum allowed equation size.

        - metric (str, optional), default='MSE':
            Metric used for evaluating error. Choose from {'MSE', 'MAE', 'MSLE', 'LogLoss'}

        - transformation (str, optional), default=None:
            Final transformation for computed value. Choose from { None, 'LOGISTIC', 'PSEUDOLOG', 'ORDINAL'}

        - init_const_min (float, optional), default=-1.0:
            Lower range for initializing constants.

        - init_const_max (float, optional), default=1.0:
            Upper range for initializing constants.

        - init_predefined_const_prob (float, optional), default=0.0:
            Probability of selecting one of the predefined constants during initialization.

        - init_predefined_const_set (list of floats, optional) default=[]:
            Predefined constants used during initialization.

        - clip_min (float, optional) default=0.0:
            Lower limit for calculated values. If both values (clip_min and clip_max) are the same, then no clip is performed.

        - clip_max (float, optional) default=0.0:
            Upper limit for calculated values. If both values (clip_min and clip_max) are the same, then no clip is performed.

        - const_min (float, optional) default=-1e30:
            Lower bound for constants used in generated equations.

        - const_max (float, optional) default=1e30:
            Upper bound for constants used in generated equations.

        - predefined_const_prob (float, optional), default=0.0:
            Probability of selecting one of the predefined constants during equations search.

        - predefined_const_set (list of floats, optional) default=[]:
            Predefined constants used during equations search.
    """

    def __init__(self,
                 num_threads: int = 8,
                 time_limit: float = 5.0,
                 iter_limit: int = 0,
                 precision: str = 'f32',
                 problem: any = math,
                 feature_probs: any = None,
                 save_model: bool = False,
                 random_state: int = 0,
                 verbose: int = 0,
                 pop_size: int = 64,
                 pop_sel: int = 4,
                 const_size: int = 8,
                 code_min_size: int = 32,
                 code_max_size: int = 32,
                 metric: str = 'MSE',
                 transformation: str = None,
                 init_const_min: float = -1.0,
                 init_const_max: float = 1.0,
                 init_predefined_const_prob: float = 0.0,
                 init_predefined_const_set: list = [],
                 clip_min: float = 0.0,
                 clip_max: float = 0.0,
                 const_min: float = -1e30,
                 const_max: float = 1e30,
                 predefined_const_prob: float = 0.0,
                 predefined_const_set: list = [],
                 cw: list = [1.0, 1.0],
                 opt_metric=mean_squared_error,
                 opt_greater_is_better=False,
                 opt_params={'method': 'Nelder-Mead'},
                 cv: bool = True,
                 cv_params={},
                 cv_select: str = 'mean',
                 ):

        if not precision in ['f32', 'f64']:
            raise ValueError("precision parameter must be 'f32' or 'f64'")

        if num_threads <= 0:
            raise ValueError(
                "num_threads parameter must be greather that zero")

        if len(cw) != 2:
            raise ValueError(
                "cw parameter incorrect")

        self.num_threads = num_threads
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.verbose = verbose
        self.precision = precision
        self.problem = problem
        self.feature_probs = feature_probs
        self.save_model = save_model
        self.random_state = random_state
        self.pop_size = pop_size
        self.pop_sel = pop_sel
        self.const_size = const_size
        self.code_min_size = code_min_size
        self.code_max_size = code_max_size
        self.metric = metric
        self.transformation = transformation
        self.init_const_min = init_const_min
        self.init_const_max = init_const_max
        self.init_predefined_const_prob = init_predefined_const_prob
        self.init_predefined_const_set = init_predefined_const_set
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.const_min = const_min
        self.const_max = const_max
        self.predefined_const_prob = predefined_const_prob
        self.predefined_const_set = predefined_const_set
        self.cw = cw
        self.opt_metric = opt_metric
        self.opt_greater_is_better = opt_greater_is_better
        self.opt_params = opt_params
        self.cv = cv
        self.cv_params = cv_params
        self.cv_select = cv_select

        self.handle = None

    def __del__(self):
        if self.handle is not None:
            DeleteSolver(self.handle)

    @property
    def equation(self):
        return self.sexpr

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """Fit symbolic model.

        Args:
            - X (numpy.ndarray): Training data.
            - y (numpy.ndarray): Target values.
        """

        if self.handle is None:
            init_predefined_const_set = numpy.ascontiguousarray(self.init_predefined_const_set).astype('float64').ctypes.data_as(DoublePointer) if len(self.init_predefined_const_set) > 0 else None
            params = Params(random_state=self.random_state,
                            num_threads=self.num_threads,
                            precision=1 if self.precision == 'f32' else 2,
                            pop_size=self.pop_size,
                            transformation=self.__parse_transformation(
                                self.transformation),
                            clip_min=self.clip_min,
                            clip_max=self.clip_max,
                            input_size=X.shape[1],
                            const_size=self.const_size,
                            code_min_size=self.code_min_size,
                            code_max_size=self.code_max_size,
                            init_const_min=self.init_const_min,
                            init_const_max=self.init_const_max,
                            init_predefined_const_prob=self.init_predefined_const_prob,
                            init_predefined_const_count=len(self.init_predefined_const_set),
                            init_predefined_const_set=init_predefined_const_set,
                            )
            self.handle = CreateSolver(ctypes.pointer(params))

        X, y = check_X_y(X, y, accept_sparse=False)

        _x = numpy.ascontiguousarray(X.T).astype(
            'float32' if self.precision == 'f32' else 'float64')
        _y = numpy.ascontiguousarray(
            y.astype('float32' if self.precision == 'f32' else 'float64'))

        predefined_const_set = numpy.ascontiguousarray(self.predefined_const_set).astype('float64').ctypes.data_as(DoublePointer) if len(self.predefined_const_set) > 0 else None

        fit_params = FitParams(
            time_limit=round(self.time_limit*1000),
            verbose=self.verbose,
            pop_sel=self.pop_sel,
            metric=self.__parse_metric(self.metric),
            iter_limit=self.iter_limit,
            const_min=self.const_min,
            const_max=self.const_max,
            predefined_const_prob=self.predefined_const_prob,
            predefined_const_count=len(self.predefined_const_set),
            predefined_const_set=predefined_const_set,
            problem=self.__problem_to_string(self.problem).encode('utf-8'),
            feature_probs=self.__feature_probs_to_string(
                self.feature_probs).encode('utf-8'),
            cw0=self.cw[0],
            cw1=self.cw[1],
        )

        if self.precision == 'f32':
            ret = FitData32(self.handle, _x, _y,
                            X.shape[0], X.shape[1], ctypes.pointer(fit_params))
        elif self.precision == 'f64':
            ret = FitData64(self.handle, _x, _y,
                            X.shape[0], X.shape[1], ctypes.pointer(fit_params))

        self.is_fitted_ = True if ret == 0 else False

        if not self.is_fitted_:
            return

        if self.cv:
            self.models = self.__get_models()
            for m in self.models:
                est = m if self._estimator_type == 'regressor' else ProbaClf(
                    m)

                m.cv_results = cross_validate(
                    estimator=est, X=X, y=y, n_jobs=self.num_threads, scoring=make_scorer(self.opt_metric), **self.cv_params)
                if self.cv_select == 'mean':
                    m.cv_score = numpy.mean(m.cv_results['test_score'])
                elif self.cv_select == 'median':
                    m.cv_score = numpy.median(m.cv_results['test_score'])

                if numpy.isnan(m.cv_score):
                    m.cv_score = -1e30 if self.opt_greater_is_better else 1e30

                # fit final coeffs from whole data
                m.fit(X, y)

            self.models.sort(
                key=lambda x: -x.cv_score if self.opt_greater_is_better else x.cv_score)
            self.sexpr = self.models[0].equation
        else:
            m = MathModel()
            GetBestModel(self.handle, m)
            model = self.__create_model(m)
            self.sexpr = apply_const(model.m.str_representation, model.m.coeffs)
            FreeModel(m)

    def predict(self, X: numpy.ndarray, id=None):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.
            - id (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        check_is_fitted(self)

        if self.cv:
            m = self.models[0] if id is None else next(
                (x for x in self.models if x.m.id == id), None)
            return m._predict(X)
        else:
            return self._predict(X, id)

    def _predict(self, X: numpy.ndarray, id=None):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.
            - id (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        _x = numpy.ascontiguousarray(X.T).astype(
            'float32' if self.precision == 'f32' else 'float64')
        _y = numpy.ascontiguousarray(
            numpy.zeros(X.shape[0], dtype=numpy.float32 if self.precision == 'f32' else numpy.float64))

        params = PredictParams(
            id if id is not None else 0xffffffff, self.verbose)

        if self.precision == 'f32':
            Predict32(self.handle, _x, _y,
                      X.shape[0], X.shape[1], ctypes.pointer(params))
        elif self.precision == 'f64':
            Predict64(self.handle, _x, _y,
                      X.shape[0], X.shape[1], ctypes.pointer(params))
        return _y

    def __get_models(self):
        models = []
        for i in range(self.num_threads*self.pop_size):
            model = MathModel()
            GetModel(self.handle, i, model)
            models.append(self.__create_model(model))
            FreeModel(model)
        return models

    def get_models(self):
        check_is_fitted(self)
        if not hasattr(self, 'models'):
            self.models = self.__get_models()
        return self.models

    def __problem_to_string(self, problem: any):
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

    def __feature_probs_to_string(self, feat: any):
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
        elif metric == 'LOGLOSS1':
            return 11
        elif metric == 'LOGLOSS2':
            return 12
        elif metric == 'LOGLOSS3':
            return 13
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
            return 4
        return 0

    def __create_model(self, m: MathModel):
        if self._estimator_type == 'regressor':
            return RegressorMathModel(ParsedMathModel(m), self.__dict__.copy(), self.opt_metric)
        else:
            return ClassifierMathModel(ParsedMathModel(m), self.__dict__.copy(), self.opt_metric)
