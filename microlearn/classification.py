"""
microlearn.classification
=========================
Classification algorithms — all trained with gradient descent or
distance-based rules, no sklearn under the hood.

Classes
-------
LogisticRegression      – binary + multi-class OvR, gradient descent
KNeighborsClassifier    – k-NN (Euclidean or Manhattan)
SVC                     – linear SVM via subgradient SGD, OvR multi-class
KNearestNeighbors       – alias for KNeighborsClassifier
"""

import numpy as np
from .base import BaseEstimator


# ============================================================= LOGISTIC ===

class LogisticRegression(BaseEstimator):
    """Logistic regression trained with gradient descent.

    Binary classification is handled directly; multi-class uses
    One-vs-Rest (OvR) — one binary classifier per class.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate.
    n_iters : int, default 1000
        Gradient-descent iterations.
    C : float, default 1.0
        Inverse regularisation strength. Larger = less regularisation.
    multi_class : str, default 'ovr'
        Only 'ovr' is supported.

    Attributes
    ----------
    classes_  : ndarray
    weights_  : ndarray — (n_features,) for binary; (n_classes, n_features) for multi
    bias_     : float or ndarray
    """

    def __init__(self, lr=0.01, n_iters=1000, C=1.0, multi_class="ovr"):
        self.lr          = lr
        self.n_iters     = n_iters
        self.C           = C
        self.multi_class = multi_class

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ---- inner binary fitter -------------------------------------------
    def _fit_binary(self, X, y_01):
        n, m = X.shape
        w    = np.zeros(m)
        b    = 0.0
        reg  = 1.0 / (self.C * n + 1e-10)   # L2 penalty coefficient

        for _ in range(self.n_iters):
            y_hat = self._sigmoid(X @ w + b)
            diff  = y_hat - y_01
            dw    = X.T @ diff / n + reg * w
            db    = diff.mean()
            w    -= self.lr * dw
            b    -= self.lr * db

        return w, b

    # ---- public API -------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            y_01 = (y == self.classes_[1]).astype(float)
            self.weights_, self.bias_ = self._fit_binary(X, y_01)
        else:
            ws, bs = [], []
            for c in self.classes_:
                w, b = self._fit_binary(X, (y == c).astype(float))
                ws.append(w); bs.append(b)
            self.weights_ = np.array(ws)
            self.bias_    = np.array(bs)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if len(self.classes_) == 2:
            p1 = self._sigmoid(X @ self.weights_ + self.bias_)
            return np.column_stack([1 - p1, p1])
        # OvR: raw sigmoid scores, row-normalise
        scores = np.column_stack([
            self._sigmoid(X @ w + b)
            for w, b in zip(self.weights_, self.bias_)
        ])
        row_sum = scores.sum(axis=1, keepdims=True)
        return scores / np.where(row_sum == 0, 1, row_sum)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ================================================================= KNN ===

class KNeighborsClassifier(BaseEstimator):
    """k-Nearest Neighbours classifier.

    Parameters
    ----------
    n_neighbors : int, default 5
    metric : {'euclidean', 'manhattan'}, default 'euclidean'

    Attributes
    ----------
    classes_ : ndarray
    """

    def __init__(self, n_neighbors=5, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric      = metric

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def _dist(self, x):
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train_ - x), axis=1)
        diff = self.X_train_ - x
        return np.sqrt((diff * diff).sum(axis=1))

    def predict_proba(self, X):
        X  = np.asarray(X, float)
        k  = self.n_neighbors
        nc = len(self.classes_)
        out = np.zeros((len(X), nc))
        for i, x in enumerate(X):
            idx = np.argsort(self._dist(x))[:k]
            votes = self.y_train_[idx]
            for j, c in enumerate(self.classes_):
                out[i, j] = np.sum(votes == c) / k
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# Alias used in the companion document
KNearestNeighbors = KNeighborsClassifier


# ================================================================= SVC ===

class SVC(BaseEstimator):
    """Linear Support Vector Classifier trained with subgradient SGD.

    Uses a vectorised primal SVM objective (hinge loss + L2 penalty).
    Multi-class is handled via One-vs-Rest.

    Parameters
    ----------
    C : float, default 1.0
        Regularisation parameter. Larger C = softer margin.
    lr : float, default 0.001
        Learning rate.
    n_iters : int, default 1000

    Attributes
    ----------
    classes_  : ndarray
    weights_  : ndarray
    bias_     : float or ndarray
    """

    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C       = C
        self.lr      = lr
        self.n_iters = n_iters

    def _fit_binary(self, X, y_pm):
        """y_pm must be ±1."""
        n, m = X.shape
        w, b = np.zeros(m), 0.0

        for _ in range(self.n_iters):
            margins = y_pm * (X @ w + b)
            sv      = margins < 1          # support-vector mask

            if sv.any():
                grad_w = w - self.C * (y_pm[sv, None] * X[sv]).mean(axis=0)
                grad_b = -(self.C * y_pm[sv].mean())
            else:
                grad_w, grad_b = w.copy(), 0.0

            w -= self.lr * grad_w
            b -= self.lr * grad_b

        return w, b

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            self._multi  = False
            y_pm         = np.where(y == self.classes_[0], -1.0, 1.0)
            self.weights_, self.bias_ = self._fit_binary(X, y_pm)
        else:
            self._multi = True
            ws, bs = [], []
            for c in self.classes_:
                w, b = self._fit_binary(X, np.where(y == c, 1.0, -1.0))
                ws.append(w); bs.append(b)
            self.weights_ = np.array(ws)
            self.bias_    = np.array(bs)

        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        if not self._multi:
            return X @ self.weights_ + self.bias_
        return np.column_stack([X @ w + b
                                 for w, b in zip(self.weights_, self.bias_)])

    def predict(self, X):
        scores = self.decision_function(X)
        if not self._multi:
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Platt-scaling approximation via sigmoid / softmax of decision scores."""
        scores = self.decision_function(X)
        if not self._multi:
            p = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1 - p, p])
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
