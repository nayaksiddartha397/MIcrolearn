"""
microlearn.linear
=================
Linear models for regression.

Classes
-------
LinearRegression  – OLS via the normal equation
RidgeRegression   – L2-regularised OLS
"""

import numpy as np
from .base import BaseEstimator


class LinearRegression(BaseEstimator):
    """Ordinary least-squares regression solved by the normal equation.

    w = (Xᵀ X)⁻¹ Xᵀ y

    Parameters
    ----------
    fit_intercept : bool, default True

    Attributes
    ----------
    coef_ : ndarray
        If fit_intercept=True, coef_[0] is the bias and coef_[1:] are
        the feature weights.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def _augment(self, X):
        """Prepend a column of ones for the intercept term."""
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb = self._augment(X)
        # Normal equation — pinv handles ill-conditioned matrices safely
        self.coef_ = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        return self

    def predict(self, X):
        return self._augment(np.asarray(X, float)) @ self.coef_

    def score(self, X, y):
        """Return R² coefficient of determination."""
        y = np.asarray(y, float)
        y_pred  = self.predict(X)
        ss_res  = np.sum((y - y_pred) ** 2)
        ss_tot  = np.sum((y - y.mean())  ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-10))


class RidgeRegression(LinearRegression):
    """L2-regularised linear regression (Ridge).

    w = (Xᵀ X + α I)⁻¹ Xᵀ y

    The bias term is NOT regularised.

    Parameters
    ----------
    alpha : float, default 1.0
        Regularisation strength — larger = more shrinkage.
    fit_intercept : bool, default True
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha         = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb   = self._augment(X)
        d    = Xb.shape[1]
        I    = np.eye(d)
        if self.fit_intercept:
            I[0, 0] = 0.0   # don't shrink the bias
        self.coef_ = np.linalg.pinv(Xb.T @ Xb + self.alpha * I) @ Xb.T @ y
        return self
