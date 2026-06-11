"""
microlearn.preprocessing
========================
Data preprocessing utilities.

Classes
-------
StandardScaler      – zero mean, unit variance
MinMaxScaler        – scale to a given range
OneHotEncoder       – categorical → binary columns
LabelEncoder        – class labels → integers
SimpleImputer       – fill NaN values
Pipeline            – chain transformers + final estimator
ColumnTransformer   – different transformers on different columns
"""

import numpy as np
from .base import BaseEstimator


# ============================================================== SCALERS ===

class StandardScaler(BaseEstimator):
    """Standardise features: z = (x − mean) / std.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
    std_  : ndarray (n_features,)
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8        # guard against zero std
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.std_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """Scale features to *feature_range* (default 0–1).

    Parameters
    ----------
    feature_range : (min, max) tuple, default (0, 1)

    Attributes
    ----------
    min_ : ndarray (n_features,)
    max_ : ndarray (n_features,)
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        return X_std * (hi - lo) + lo

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - lo) / (hi - lo + 1e-8)
        return X_std * (self.max_ - self.min_) + self.min_


# ============================================================= ENCODERS ===

class OneHotEncoder(BaseEstimator):
    """Encode categorical features as one-hot binary arrays.

    Each unique category in a column becomes its own 0/1 column.

    Attributes
    ----------
    categories_ : list of ndarray — unique values per input feature
    """

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        parts = []
        for i, cats in enumerate(self.categories_):
            col = X[:, i].reshape(-1, 1)
            parts.append((col == cats[np.newaxis, :]).astype(float))
        return np.hstack(parts)

    def get_feature_names_out(self):
        names = []
        for i, cats in enumerate(self.categories_):
            for c in cats:
                names.append(f"x{i}_{c}")
        return np.array(names)


class LabelEncoder(BaseEstimator):
    """Encode class labels as integers 0 … n_classes−1.

    Attributes
    ----------
    classes_ : ndarray — sorted unique classes
    """

    def fit(self, y, _ignored=None):
        self.classes_       = np.unique(y)
        self._cls_to_idx    = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self._cls_to_idx[yi] for yi in y])

    def inverse_transform(self, y):
        return np.array([self.classes_[int(i)] for i in y])

    def fit_transform(self, y, _ignored=None):
        return self.fit(y).transform(y)


# ============================================================== IMPUTER ===

class SimpleImputer(BaseEstimator):
    """Fill NaN values with a column-level statistic.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default 'mean'
    fill_value : scalar — used when strategy='constant'

    Attributes
    ----------
    fill_values_ : ndarray (n_features,)
    """

    def __init__(self, strategy="mean", fill_value=0):
        self.strategy   = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            vals = []
            for i in range(X.shape[1]):
                col = X[~np.isnan(X[:, i]), i]
                uniq, counts = np.unique(col, return_counts=True)
                vals.append(uniq[counts.argmax()])
            self.fill_values_ = np.array(vals)
        elif self.strategy == "constant":
            self.fill_values_ = np.full(X.shape[1], float(self.fill_value))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.fill_values_[i]
        return X


# =========================================================== PIPELINE ===

class Pipeline(BaseEstimator):
    """Chain transformers and a final estimator into a single object.

    Parameters
    ----------
    steps : list of (name, estimator) tuples
        All but the last step must implement transform().
        The last step is the final estimator (classifier / regressor).

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf',    LogisticRegression(lr=0.1, n_iters=500)),
    ... ])
    >>> pipe.fit(X_train, y_train).score(X_test, y_test)
    """

    def __init__(self, steps):
        self.steps = steps

    # ---- internal helpers -----------------------------------------------
    def _transformers(self):
        """All steps except the last."""
        return self.steps[:-1]

    def _final(self):
        return self.steps[-1][1]

    def _apply_transforms(self, X):
        Xt = np.asarray(X)
        for _, t in self._transformers():
            Xt = t.transform(Xt)
        return Xt

    # ---- public API -------------------------------------------------------
    def fit(self, X, y=None):
        Xt = np.asarray(X)
        for _, t in self._transformers():
            Xt = t.fit_transform(Xt, y)
        self._final().fit(Xt, y)
        return self

    def predict(self, X):
        return self._final().predict(self._apply_transforms(X))

    def predict_proba(self, X):
        return self._final().predict_proba(self._apply_transforms(X))

    def transform(self, X):
        """Apply ALL steps (useful when all steps are transformers)."""
        Xt = np.asarray(X)
        for _, t in self.steps:
            Xt = t.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        Xt = np.asarray(X)
        for _, t in self.steps:
            Xt = t.fit_transform(Xt, y)
        return Xt

    def score(self, X, y):
        return self._final().score(self._apply_transforms(X), y)

    def set_params(self, **params):
        """Supports step__param notation, e.g. clf__lr=0.1."""
        for key, val in params.items():
            if "__" in key:
                step_name, param = key.split("__", 1)
                for name, step in self.steps:
                    if name == step_name:
                        step.set_params(**{param: val})
            else:
                setattr(self, key, val)
        return self


# ====================================================== COLUMN TRANSFORMER ===

class ColumnTransformer(BaseEstimator):
    """Apply different transformers to different subsets of columns.

    Parameters
    ----------
    transformers : list of (name, transformer, columns)
        *columns* can be a list of int indices or a ``slice``.
    remainder : {'drop', 'passthrough'}, default 'drop'

    Examples
    --------
    >>> ct = ColumnTransformer([
    ...     ('num', StandardScaler(),  [0, 1, 2]),
    ...     ('cat', OneHotEncoder(),   [3, 4]),
    ... ])
    >>> ct.fit_transform(X)
    """

    def __init__(self, transformers, remainder="drop"):
        self.transformers = transformers
        self.remainder    = remainder

    def _col_idx(self, X, cols):
        if isinstance(cols, slice):
            return list(range(*cols.indices(X.shape[1])))
        return list(cols)

    def fit(self, X, y=None):
        X = np.asarray(X)
        self._fitted = []
        self._used   = set()
        for name, transformer, cols in self.transformers:
            idx = self._col_idx(X, cols)
            self._used.update(idx)
            transformer.fit(X[:, idx], y)
            self._fitted.append((name, transformer, idx))
        return self

    def transform(self, X):
        X     = np.asarray(X)
        parts = []
        for _, transformer, idx in self._fitted:
            parts.append(transformer.transform(X[:, idx]))
        if self.remainder == "passthrough":
            rest = sorted(set(range(X.shape[1])) - self._used)
            if rest:
                parts.append(X[:, rest])
        return np.hstack(parts)
