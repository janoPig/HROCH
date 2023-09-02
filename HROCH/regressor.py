from .hroch import PHCRegressor, math
from sklearn.base import RegressorMixin
import numpy as numpy


class SymbolicRegressor(PHCRegressor, RegressorMixin):
    """
    Symbolic Regressor

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
                 predefined_const_set: list = []
                 ):
        super(SymbolicRegressor, self).__init__(
            num_threads=num_threads,
            time_limit=time_limit,
            iter_limit=iter_limit,
            precision=precision,
            problem=problem,
            feature_probs=feature_probs,
            save_model=save_model,
            random_state=random_state,
            verbose=verbose,
            pop_size=pop_size,
            pop_sel=pop_sel,
            const_size=const_size,
            code_min_size=code_min_size,
            code_max_size=code_max_size,
            metric=metric,
            transformation=transformation,
            init_const_min=init_const_min,
            init_const_max=init_const_max,
            init_predefined_const_prob=init_predefined_const_prob,
            init_predefined_const_set=init_predefined_const_set,
            clip_min=clip_min,
            clip_max=clip_max,
            const_min=const_min,
            const_max=const_max,
            predefined_const_prob=predefined_const_prob,
            predefined_const_set=predefined_const_set
        )

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, sample_weight=None):
        """Fit symbolic model.

        Args:
            - X (numpy.ndarray): Training data.
            - y (numpy.ndarray): Target values.

            !!!In the current version, the sample_weight parameter is ignored!!!
            - sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
        """

        super(SymbolicRegressor, self).fit(X, y)
        return self
