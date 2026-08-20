"""
microlearn.model_selection
==========================
Utilities for splitting data, cross-validation, and hyper-parameter search.

Functions / Classes
-------------------
train_test_split        – random or stratified train/test split
KFold                   – k consecutive folds
StratifiedKFold         – class-proportionate folds
cross_val_score         – k-fold CV score
GridSearchCV            – exhaustive grid search
RandomizedSearchCV      – randomised hyperparameter search
"""

import copy
import itertools
import numpy as np


# ======================================================= TRAIN/TEST SPLIT ===

def train_test_split(*arrays, test_size=0.2, random_state=None, stratify=None):
    """Split arrays into train and test subsets.

    Parameters
    ----------
    *arrays      : one or more array-likes with the same first dimension.
    test_size    : float in (0, 1), default 0.2
    random_state : int or None
    stratify     : array-like or None — preserve class proportions when set.

    Returns
    -------
    List of split arrays: [A_train, A_test, B_train, B_test, ...]

    Examples
    --------
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
    ...                                             random_state=42)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(arrays[0])

    if stratify is not None:
        stratify = np.asarray(stratify)
        train_idx, test_idx = [], []
        for c in np.unique(stratify):
            c_idx = np.where(stratify == c)[0]
            np.random.shuffle(c_idx)
            n_test = max(1, int(len(c_idx) * test_size))
            test_idx.extend(c_idx[:n_test])
            train_idx.extend(c_idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx  = np.array(test_idx)
        # Do not leave the returned arrays grouped by class.  Besides being
        # surprising to callers, class blocks make a subsequent ordinary
        # KFold split produce single-class validation folds.
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
    else:
        perm      = np.random.permutation(n)
        split     = int(n * (1.0 - test_size))
        train_idx = perm[:split]
        test_idx  = perm[split:]

    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


# ======================================================= K-FOLD SPLITTERS ===

class KFold:
    """K-Folds cross-validator.

    Splits the dataset into *n_splits* consecutive folds.

    Parameters
    ----------
    n_splits     : int, default 5
    shuffle      : bool, default False
    random_state : int or None

    Examples
    --------
    >>> for train_idx, val_idx in KFold(n_splits=5).split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits    = n_splits
        self.shuffle     = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n       = len(X)
        indices = np.arange(n)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        sizes[:n % self.n_splits] += 1   # distribute remainder

        cur = 0
        for size in sizes:
            val_idx   = indices[cur:cur + size]
            train_idx = np.concatenate([indices[:cur], indices[cur + size:]])
            yield train_idx, val_idx
            cur += size

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


class StratifiedKFold:
    """Stratified K-Folds — each fold preserves the class distribution.

    Parameters
    ----------
    n_splits     : int, default 5
    shuffle      : bool, default False
    random_state : int or None

    Examples
    --------
    >>> for train_idx, val_idx in StratifiedKFold(5).split(X, y):
    ...     ...
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits    = n_splits
        self.shuffle     = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        y       = np.asarray(y)
        classes = np.unique(y)

        # Per-class index lists
        cls_idx = {}
        for c in classes:
            idx = np.where(y == c)[0]
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(idx)
            cls_idx[c] = idx

        # Distribute each class's indices evenly across folds
        folds = [[] for _ in range(self.n_splits)]
        for c in classes:
            idx    = cls_idx[c]
            nc     = len(idx)
            sizes  = np.full(self.n_splits, nc // self.n_splits, dtype=int)
            sizes[:nc % self.n_splits] += 1
            cur = 0
            for i, sz in enumerate(sizes):
                folds[i].extend(idx[cur:cur + sz])
                cur += sz

        for i in range(self.n_splits):
            val_idx   = np.array(folds[i])
            train_idx = np.concatenate([
                np.array(folds[j]) for j in range(self.n_splits) if j != i
            ])
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


# ======================================================= CROSS-VAL SCORE ===

def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """Evaluate an estimator with k-fold cross-validation.

    The estimator is deep-copied for each fold so the original is untouched.

    Parameters
    ----------
    estimator : BaseEstimator
    X, y      : array-like
    cv        : int or splitter object, default 5
    scoring   : None → estimator.score(); or 'accuracy' / 'r2' string; or callable

    Returns
    -------
    scores : ndarray (cv,)

    Examples
    --------
    >>> cv = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
    >>> print(f"{cv.mean():.3f} ± {cv.std():.3f}")
    """
    X = np.asarray(X)
    y = np.asarray(y)

    splitter = KFold(n_splits=cv) if isinstance(cv, int) else cv
    scores   = []

    for train_idx, val_idx in splitter.split(X, y):
        est = copy.deepcopy(estimator)
        est.fit(X[train_idx], y[train_idx])

        if scoring is None:
            scores.append(est.score(X[val_idx], y[val_idx]))
        elif scoring == "accuracy":
            from microlearn.metrics import accuracy_score
            scores.append(accuracy_score(y[val_idx], est.predict(X[val_idx])))
        elif scoring == "r2":
            from microlearn.metrics import r2_score
            scores.append(r2_score(y[val_idx], est.predict(X[val_idx])))
        elif callable(scoring):
            scores.append(scoring(est, X[val_idx], y[val_idx]))
        else:
            raise ValueError(f"Unknown scoring: {scoring!r}")

    return np.array(scores)


# ======================================================= GRID SEARCH CV ===

class GridSearchCV:
    """Exhaustive search over a hyperparameter grid using cross-validation.

    Parameters
    ----------
    estimator  : BaseEstimator
    param_grid : dict — {param_name: [val1, val2, ...], ...}
    cv         : int, default 5
    scoring    : None / 'accuracy' / 'r2' / callable
    verbose    : int, default 0

    Attributes
    ----------
    best_params_    : dict
    best_score_     : float
    best_estimator_ : fitted estimator (refitted on full training data)
    cv_results_     : list of {'params', 'mean_score', 'std_score'} dicts

    Examples
    --------
    >>> grid = GridSearchCV(
    ...     LogisticRegression(),
    ...     param_grid={'lr': [0.01, 0.1], 'n_iters': [500, 1000]},
    ...     cv=5
    ... )
    >>> grid.fit(X_train, y_train)
    >>> print(grid.best_params_, grid.best_score_)
    """

    def __init__(self, estimator, param_grid, cv=5, scoring=None, verbose=0):
        self.estimator  = estimator
        self.param_grid = param_grid
        self.cv         = cv
        self.scoring    = scoring
        self.verbose    = verbose

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        keys   = list(self.param_grid.keys())
        combos = list(itertools.product(*[self.param_grid[k] for k in keys]))

        self.cv_results_ = []
        best_score       = -np.inf

        for combo in combos:
            params = dict(zip(keys, combo))
            est    = copy.deepcopy(self.estimator)
            est.set_params(**params)

            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            ms, ss = float(scores.mean()), float(scores.std())

            self.cv_results_.append({"params": params,
                                     "mean_score": ms,
                                     "std_score":  ss})

            if self.verbose > 0:
                print(f"  {params}  →  {ms:.4f} ± {ss:.4f}")

            if ms > best_score:
                best_score        = ms
                self.best_params_ = params
                self.best_score_  = ms

        # Refit best model on all training data
        self.best_estimator_ = copy.deepcopy(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):       return self.best_estimator_.predict(X)
    def predict_proba(self, X): return self.best_estimator_.predict_proba(X)
    def score(self, X, y):      return self.best_estimator_.score(X, y)


# ================================================= RANDOMISED SEARCH CV ===

class RandomizedSearchCV:
    """Randomised search over hyperparameter distributions.

    Faster than GridSearchCV — samples *n_iter* random combinations.

    Parameters
    ----------
    estimator           : BaseEstimator
    param_distributions : dict
        {param: list_of_values}  or  {param: scipy_dist_with_rvs()}
    n_iter       : int, default 10
    cv           : int, default 5
    scoring      : None / 'accuracy' / 'r2' / callable
    random_state : int or None
    verbose      : int, default 0

    Attributes
    ----------
    best_params_, best_score_, best_estimator_, cv_results_  (same as GridSearchCV)

    Examples
    --------
    >>> rs = RandomizedSearchCV(
    ...     LogisticRegression(),
    ...     param_distributions={'lr': [0.001, 0.01, 0.1], 'n_iters': [200, 500, 1000]},
    ...     n_iter=6, cv=5, random_state=42
    ... )
    >>> rs.fit(X_train, y_train)
    """

    def __init__(self, estimator, param_distributions, n_iter=10, cv=5,
                 scoring=None, random_state=None, verbose=0):
        self.estimator           = estimator
        self.param_distributions = param_distributions
        self.n_iter              = n_iter
        self.cv                  = cv
        self.scoring             = scoring
        self.random_state        = random_state
        self.verbose             = verbose

    def _sample(self):
        params = {}
        for k, dist in self.param_distributions.items():
            if hasattr(dist, "rvs"):                      # scipy distribution
                params[k] = dist.rvs()
            elif isinstance(dist, (list, np.ndarray)):    # explicit list
                params[k] = np.random.choice(dist)
            else:
                params[k] = dist
        return params

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.cv_results_ = []
        best_score       = -np.inf

        for _ in range(self.n_iter):
            params = self._sample()
            est    = copy.deepcopy(self.estimator)
            est.set_params(**params)

            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            ms, ss = float(scores.mean()), float(scores.std())

            self.cv_results_.append({"params": params,
                                     "mean_score": ms,
                                     "std_score":  ss})

            if self.verbose > 0:
                print(f"  {params}  →  {ms:.4f} ± {ss:.4f}")

            if ms > best_score:
                best_score        = ms
                self.best_params_ = params
                self.best_score_  = ms

        self.best_estimator_ = copy.deepcopy(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):       return self.best_estimator_.predict(X)
    def predict_proba(self, X): return self.best_estimator_.predict_proba(X)
    def score(self, X, y):      return self.best_estimator_.score(X, y)
