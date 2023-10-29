from .hroch import PHCRegressor, fuzzy
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
import numpy as numpy


class FuzzyRegressor(PHCRegressor, ClassifierMixin):
    """
    Fuzzy Regressor

    Parameters:
        - num_threads (int, optional), default=8:

            Number of used threads.

        - time_limit (float, optional), default=5.0:

            Timeout in seconds. If is set to 0 there is no limit and the algorithm runs until some other condition is met.

        - iter_limit (int, optional), default=0:

            Iterations limit. If is set to 0 there is no limit and the algorithm runs until some other condition is met.

        - precision (str, optional), default='f32':

            'f64' or 'f32'. Internal floating number representation. 32bit AVX2 instructions are 2x faster as 64bit.

        - problem (any, optional), default=fuzzy:

            Predefined instructions sets fuzzy or custom defines set of instructions with mutation probability.
            `reg = FuzzyRegressor(problem={'nop': 0.01, 'f_and': 1.0, 'f_or': 1.0, 'f_xor': 1.0, 'f_not': 1.0})`

        - feature_probs (any, optional), default=None:

            `reg = FuzzyRegressor(feature_probs=[1.0,1.0, 0.01])`

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

        - metric (str, optional), default='LogLoss':

            Metric used for evaluating error. Choose from {'MSE', 'MAE', 'MSLE', 'LogLoss'}

        - init_const_min (float, optional), default=0.0:

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

        - const_min (float, optional) default=0.0:

            Lower bound for constants used in generated equations.

        - const_max (float, optional) default=1.0:

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
                 problem: any = fuzzy,
                 feature_probs: any = None,
                 save_model: bool = False,
                 random_state: int = 0,
                 verbose: int = 0,
                 pop_size: int = 64,
                 pop_sel: int = 4,
                 const_size: int = 8,
                 code_min_size: int = 32,
                 code_max_size: int = 32,
                 metric: str = 'LogLoss',
                 init_predefined_const_prob: float = 0.0,
                 init_predefined_const_set: list = [],
                 predefined_const_prob: float = 0.0,
                 predefined_const_set: list = [],
                 cw: list = [1.0, 1.0],
                 opt_metric=log_loss,
                 opt_greater_is_better=False,
                 opt_params={'method': 'Nelder-Mead'},
                 cv: bool = True,
                 cv_params={},
                 cv_select: str = 'mean',
                 ):
        super(FuzzyRegressor, self).__init__(
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
            transformation=None,
            init_const_min=0.0,
            init_const_max=1.0,
            init_predefined_const_prob=init_predefined_const_prob,
            init_predefined_const_set=init_predefined_const_set,
            clip_min=3e-7,
            clip_max=1.0-3e-7,
            const_min=0.0,
            const_max=1.0,
            predefined_const_prob=predefined_const_prob,
            predefined_const_set=predefined_const_set,
            cw=cw,
            opt_metric=opt_metric,
            opt_greater_is_better=opt_greater_is_better,
            opt_params=opt_params,
            cv=cv,
            cv_params=cv_params,
            cv_select=cv_select,
        )

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, sample_weight=None):
        """Fit symbolic model.

        Args:
            - X (numpy.ndarray): Training data. Values must be in the range 0.0 to 1.0.
            - y (numpy.ndarray): Target values.

            !!!In the current version, the sample_weight parameter is ignored!!!
            - sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
        """

        super(FuzzyRegressor, self).fit(X, y)
        return self

    def predict(self, X: numpy.ndarray, id=None):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.
            - id (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        preds = super(FuzzyRegressor, self).predict(X, id)
        return (preds > 0.5)*1.0

    def predict_proba(self, X: numpy.ndarray, id=None):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.
            - id (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

        Returns:
            numpy.ndarray, shape = [n_samples, n_classes]: The class probabilities of the input samples.
        """
        preds = super(FuzzyRegressor, self).predict(X, id)
        proba = numpy.vstack([1 - preds, preds]).T
        return proba
