from sklearn.multiclass import OneVsRestClassifier
from .hroch import SymbolicSolver
from sklearn.base import ClassifierMixin
from sklearn.utils import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
import numpy as numpy

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
        If not defined FuzzyRegressor.INIT_CONST_SETTINGS is used.
        ```python
        init_const_settings = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for initializing constants.
        - 'const_max' : (float) Upper range for initializing constants.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during initialization.
        - 'predefined_const_set' : (array of floats) Predefined constants used during initialization.

    const_settings : dict, default = None
        If not defined FuzzyRegressor.CONST_SETTINGS is used.
        ```python
        const_settings = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
        ```
        - 'const_min' : (float) Lower range for constants used in equations.
        - 'const_max' : (float) Upper range for constants used in equations.
        - 'predefined_const_prob': (float) Probability of selecting one of the predefined constants during search process(mutation).
        - 'predefined_const_set' : (array of floats) Predefined constants used during search process(mutation).

    target_clip : array, default = None
        Array of two float values clip_min and clip_max.
        If not defined SymbolicSolver.CLASSIFICATION_TARGET_CLIP is used.
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

    cv_params : dict, default = None
        If not defined SymbolicSolver.CLASSIFICATION_CV_PARAMS is used.
        ```python
        cv_params = {'n':0, 'cv_params':{}, 'select':'mean', 'opt_params':{'method': 'Nelder-Mead'}, 'opt_metric':make_scorer(log_loss, greater_is_better=False, needs_proba=True)}
        ```
        - 'n' : (int) Crossvalidate n top models
        - 'cv_params' : (dict) Parameters passed to scikit-learn cross_validate method
        - select : (str) Best model selection method choose from 'mean'or 'median'
        - opt_params : (dict) Parameters passed to scipy.optimize.minimize method
        - opt_metric : (make_scorer) Scoring method
    """
    
    INIT_CONST_SETTINGS = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}
    CONST_SETTINGS = {'const_min':0.0, 'const_max':1.0, 'predefined_const_prob':0.0, 'predefined_const_set': []}

    def __init__(self,
                 num_threads: int = 1,
                 time_limit: float = 5.0,
                 iter_limit: int = 0,
                 precision: str = 'f32',
                 problem = 'fuzzy',
                 feature_probs = None,
                 random_state: int = 0,
                 verbose: int = 0,
                 metric: str = 'LogLoss',
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
            `n_features` is the number of features. Should be in the range [0, 1].

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
        if check_input:
            X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=False, multi_output=False)
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

        self.class_weight_ = compute_class_weight(self.class_weight, classes=self.classes_, y=y)

        super(FuzzyRegressor, self).fit(X, y_ind, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, id=None, check_input=True, use_parsed_model=True):
        """
        Predict class for X.

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
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        preds = super(FuzzyRegressor, self).predict(X, id, check_input=check_input, use_parsed_model=use_parsed_model)
        return self.classes_[(preds > 0.5).astype(int)]

    def predict_proba(self, X, id=None, check_input=True):
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
        preds = super(FuzzyRegressor, self).predict(X, id, check_input=check_input)
        proba = numpy.vstack([1 - preds, preds]).T
        return proba
    
    def _more_tags(self):
        return {
            'binary_only': True,
            'poor_score':True, # tests from check_estimator dont have fuzzy number type
            }


class FuzzyClassifier(OneVsRestClassifier):
    """
    Fuzzy multiclass symbolic classificator
    
    Parameters
    ----------
    estimator : FuzzyRegressor
        Instance of FuzzyRegressor class.
    """
    def __init__(self, estimator=FuzzyRegressor()):
        super().__init__(estimator=estimator)
    
    def fit(self, X, y):
        """
        Fit the symbolic models according to the given training data. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. Should be in the range [0, 1].

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """

        super().fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        return super().predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return super().predict_proba(X)
    
    def _more_tags(self):
        return {
            'poor_score':True, # tests from check_estimator dont have fuzzy number type
            }