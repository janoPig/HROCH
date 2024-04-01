from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from .regressor import SymbolicRegressor
from .hroch import RegressorMathModel
from sklearn.metrics import log_loss
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import minimize
import copy

class PseudoClassifierMathModel(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor_model : RegressorMathModel, opt_params = None, verbose = False):
        self.regressor_model = regressor_model
        self.opt_params = opt_params
        self.verbose = verbose
        
    def fit(self, X, y):
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
            
        self.optimized_model_ = copy.deepcopy(self.regressor_model)
        def objective(c):
            self.optimized_model_.m.coeffs = c
            preds = self.optimized_model_.predict(X)
            preds = np.nan_to_num(preds,nan=0)
            preds = np.clip(preds, -20, 20)
            proba = 1.0/(1.0+np.exp(-preds))
            return log_loss(y_ind, proba)
        
        self.opt_score_ = objective(self.optimized_model_.m.coeffs)
        self.opt_score0_ = self.opt_score_

        if len(self.optimized_model_.m.coeffs) > 0:
            result = minimize(objective, self.optimized_model_.m.coeffs, **self.opt_params)

            for i in range(len(self.optimized_model_.m.coeffs)):
                self.optimized_model_.m.coeffs[i] = result.x[i]
            self.opt_score_ = result.fun
        
        self.is_fitted_ = True
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
        check_is_fitted(self)
        preds = self.optimized_model_.predict(X)
        preds = np.nan_to_num(preds,nan=0)
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
        check_is_fitted(self)
        preds = self.optimized_model_.predict(X)
        preds = np.nan_to_num(preds,nan=0)
        preds = np.clip(preds, -20, 20)
        proba = 1.0/(1.0+np.exp(-preds))
        return np.vstack([1 - proba, proba]).T
    
    def __str__(self):
        return f"PseudoClassifierMathModel({self.optimized_model_.m.str_representation})"
    
    def __repr__(self):
        return f"PseudoClassifierMathModel({self.optimized_model_.m.str_representation})"
    
class PseudoClassifier(BaseEstimator, ClassifierMixin):
    OPT_PARMS = {'method':'Nelder-Mead', 'options':{'maxiter': 50}}
    
    def __init__(self, t : float = 1.0, n : int = 16, regressor_params = None, opt_params = None, verbose : int = 0):
        """
        PseudoClassifier class. Perform binary classification using symbolic regression.
        Transform the classification problem into a regression problem by transforming {negative_class, positive_class} -> {-t, t}

        Parameters
        ----------
        t : float, default=1.0
            Transformation parameter for transformation {negative_class, positive_class} -> {-t, t}.

        n : int, default=16
            Number of returned models

        regressor_params : dict, default=None
            parameters passed to SymbolicRegressor
            
        opt_params : dict, default=None
            parameters passed to scipy.optimize.minimize
            if None default {'method':'Nelder-Mead', 'options':{'maxiter': 50}} used
            
        verbose : int, default=0
            verbosity level
        """
        self.t = t
        self.n = n
        self.regressor_params = regressor_params
        self.opt_params = opt_params
        self.verbose = verbose
    
    def fit(self, X, y, check_input=True):
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
        # transform classes to {0, 1}
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
            
        # transform data -t for negative class, t for positive class
        y_ = (y_ind-0.5)*2.0
        y_transformed = y_*self.t
        regressor_params = self.regressor_params if self.regressor_params is not None else {}
        reg = SymbolicRegressor(**regressor_params)
        reg.fit(X, y_transformed)
        
        # basic score
        preds0 = reg.predict(X)
        proba0 = 1.0/(1.0+np.exp(-preds0))
        score0 = log_loss(y_ind, proba0)
        if self.verbose > 0:
            print(f'Train log_loss score for basic regression: {score0}')
        
        models = reg.get_models()
        models_count = min(self.n, len(models))
        self.models_ = models[:models_count]
        
        # train models to log_loss score
        opt_params = self.opt_params if self.opt_params is not None else self.OPT_PARMS
        self.optimized_models_ = []
        for m in self.models_:
            opt_model = PseudoClassifierMathModel(m, opt_params, self.verbose > 1)
            opt_model.fit(X, y_ind)
            self.optimized_models_.append(opt_model)
        self.optimized_models_.sort(key=lambda x: x.opt_score_)
        self.reg_ = reg
        self.is_fitted_ = True
        return self
    
    def predict(self, X, check_input=True):
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
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        
        return self.optimized_models_[0].predict(X)
    
    def predict_proba(self, X, check_input=True):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : narray-like of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        
        return self.optimized_models_[0].predict_proba(X)
    
    def get_models(self):
        """
        Get population of symbolic models.

        Returns
        -------
        models : array of RegressorMathModel or ClassifierMathModel
        """
        return self.optimized_models_
    
    def _more_tags(self):
        return {'binary_only': True}
