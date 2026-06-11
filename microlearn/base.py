"""
microlearn.base
===============
BaseEstimator — the common interface every estimator inherits from.
Provides fit, predict, predict_proba, transform, fit_transform,
score, get_params, set_params, and __repr__.
"""

import numpy as np


class BaseEstimator:
    """
    Base class for all microlearn estimators.

    Convention (mirrors sklearn):
      - Hyperparameters are set in __init__ and stored as plain attributes.
      - Fitted attributes always end with a trailing underscore (e.g. coef_).
      - fit() always returns self so calls can be chained.
    """

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fit()"
        )

    # --------------------------------------------------------------- predict
    def predict(self, X):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict()"
        )

    # --------------------------------------------------------- predict_proba
    def predict_proba(self, X):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba()"
        )

    # ------------------------------------------------------------- transform
    def transform(self, X):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement transform()"
        )

    # --------------------------------------------------------- fit_transform
    def fit_transform(self, X, y=None):
        """Fit and transform in one step (default: fit then transform)."""
        return self.fit(X, y).transform(X)

    # ----------------------------------------------------------------- score
    def score(self, X, y):
        """
        Default scoring: accuracy for classifiers.
        Regression subclasses override this with R².
        """
        preds = self.predict(X)
        return float(np.mean(np.asarray(preds) == np.asarray(y)))

    # -------------------------------------------------------------- get/set params
    def get_params(self):
        """Return a dict of hyperparameters (attributes NOT ending in '_')."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.endswith("_")
        }

    def set_params(self, **params):
        """Set hyperparameters by name. Returns self."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ----------------------------------------------------------------- repr
    def __repr__(self):
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
