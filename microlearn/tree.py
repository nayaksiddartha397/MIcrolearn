"""
microlearn.tree
===============
Tree-based classifiers built from scratch.

Classes
-------
DecisionTreeClassifier   – CART tree (Gini / entropy, depth-prunable)
RandomForestClassifier   – bagged CART trees with random feature selection
"""

import numpy as np
from .base import BaseEstimator


# ========================================================= TREE NODE ===

class _Node:
    """A single node in a decision tree."""
    __slots__ = (
        "feature", "threshold", "left", "right",
        "value", "proba", "n_samples", "impurity", "impurity_decrease",
    )

    def __init__(self, *, feature=None, threshold=None,
                 left=None, right=None,
                 value=None, proba=None,
                 n_samples=0, impurity=0.0, impurity_decrease=0.0):
        self.feature           = feature
        self.threshold         = threshold
        self.left              = left
        self.right             = right
        self.value             = value          # majority class (leaf only)
        self.proba             = proba          # class distribution (leaf only)
        self.n_samples         = n_samples
        self.impurity          = impurity
        self.impurity_decrease = impurity_decrease

    @property
    def is_leaf(self):
        return self.value is not None


# ============================================ DECISION TREE CLASSIFIER ===

class DecisionTreeClassifier(BaseEstimator):
    """CART decision-tree classifier.

    Recursively splits data by maximising information gain
    (Gini impurity or entropy criterion).

    Parameters
    ----------
    max_depth : int or None, default None
    min_samples_split : int, default 2
        Minimum samples required to split a node.
    min_samples_leaf : int, default 1
        Minimum samples required at a leaf node.
    criterion : {'gini', 'entropy'}, default 'gini'
    max_features : int, 'sqrt', 'log2', or None, default None
        Features to consider at each split (used by RandomForest).

    Attributes
    ----------
    tree_                 : _Node  – root of the fitted tree
    classes_              : ndarray
    n_features_           : int
    feature_importances_  : ndarray (n_features,)
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini", max_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.criterion         = criterion
        self.max_features      = max_features

    # ---- impurity --------------------------------------------------------
    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1.0 - np.dot(p, p))

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        p = p[p > 0]
        return float(-np.dot(p, np.log2(p)))

    def _impurity(self, y):
        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    # ---- feature sampling ------------------------------------------------
    def _feat_idx(self, n_features):
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = int(mf)
        return np.random.choice(n_features, k, replace=False)

    # ---- best split search -----------------------------------------------
    def _best_split(self, X, y, parent_imp):
        best_gain, best_feat, best_thr = -np.inf, None, None
        n = len(y)

        for feat in self._feat_idx(X.shape[1]):
            vals = np.unique(X[:, feat])
            # mid-points between consecutive unique values
            thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals

            for thr in thresholds:
                l_mask = X[:, feat] <= thr
                nl, nr  = l_mask.sum(), (~l_mask).sum()

                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                gain = (parent_imp
                        - (nl * self._impurity(y[l_mask])
                           + nr * self._impurity(y[~l_mask])) / n)

                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr

        return best_feat, best_thr, max(best_gain, 0.0)

    # ---- leaf helpers ----------------------------------------------------
    def _leaf_value(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[counts.argmax()]

    def _leaf_proba(self, y):
        p = np.zeros(len(self.classes_))
        if len(y) > 0:
            for i, c in enumerate(self.classes_):
                p[i] = np.sum(y == c) / len(y)
        return p

    # ---- recursive tree builder ------------------------------------------
    def _grow(self, X, y, depth):
        n = len(y)
        cur_imp = self._impurity(y)

        stop = (
            len(np.unique(y)) == 1
            or n < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return _Node(value=self._leaf_value(y),
                         proba=self._leaf_proba(y),
                         n_samples=n, impurity=cur_imp)

        feat, thr, gain = self._best_split(X, y, cur_imp)
        if feat is None:
            return _Node(value=self._leaf_value(y),
                         proba=self._leaf_proba(y),
                         n_samples=n, impurity=cur_imp)

        lm = X[:, feat] <= thr
        left_node  = self._grow(X[lm],  y[lm],  depth + 1)
        right_node = self._grow(X[~lm], y[~lm], depth + 1)

        imp_dec = (cur_imp
                   - (lm.sum()  * left_node.impurity
                      + (~lm).sum() * right_node.impurity) / n)

        return _Node(feature=feat, threshold=thr,
                     left=left_node, right=right_node,
                     n_samples=n, impurity=cur_imp,
                     impurity_decrease=imp_dec)

    # ---- public API -------------------------------------------------------
    def fit(self, X, y):
        X, y             = np.asarray(X, float), np.asarray(y)
        self.classes_    = np.unique(y)
        self.n_features_ = X.shape[1]
        self.tree_       = self._grow(X, y, depth=0)
        return self

    def _walk(self, x, node):
        """Traverse the tree and return the leaf node for sample x."""
        if node.is_leaf:
            return node
        return self._walk(x, node.left if x[node.feature] <= node.threshold
                          else node.right)

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._walk(x, self.tree_).value for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return np.array([self._walk(x, self.tree_).proba for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ---- feature importances ---------------------------------------------
    @property
    def feature_importances_(self):
        imp = np.zeros(self.n_features_)
        self._accum(self.tree_, imp)
        total = imp.sum()
        return imp / (total + 1e-10)

    def _accum(self, node, imp):
        if node.is_leaf:
            return
        imp[node.feature] += node.n_samples * node.impurity_decrease
        self._accum(node.left,  imp)
        self._accum(node.right, imp)


# ============================================= RANDOM FOREST CLASSIFIER ===

class RandomForestClassifier(BaseEstimator):
    """Ensemble of CART trees trained on bootstrap samples.

    Each tree sees a bootstrap resample of the data and considers only
    a random subset of features at each split (max_features='sqrt').

    Parameters
    ----------
    n_estimators : int, default 100
    max_depth : int or None, default None
    min_samples_split : int, default 2
    min_samples_leaf : int, default 1
    max_features : 'sqrt', 'log2', int, or None, default 'sqrt'
    bootstrap : bool, default True
    criterion : {'gini', 'entropy'}, default 'gini'
    random_state : int or None, default None

    Attributes
    ----------
    estimators_          : list of DecisionTreeClassifier
    classes_             : ndarray
    feature_importances_ : ndarray (n_features,)
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True,
                 criterion="gini", random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.bootstrap         = bootstrap
        self.criterion         = criterion
        self.random_state      = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_    = np.unique(y)
        self.n_features_ = X.shape[1]
        self.estimators_ = []
        n = len(X)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
                tree.fit(X[idx], y[idx])
            else:
                tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        X  = np.asarray(X, float)
        nc = len(self.classes_)
        # Average probabilities; handle trees that saw only a class subset
        agg = np.zeros((len(X), nc))
        for tree in self.estimators_:
            p = tree.predict_proba(X)             # shape (n, tree_n_classes)
            full = np.zeros((len(X), nc))
            for i, c in enumerate(tree.classes_):
                j = np.where(self.classes_ == c)[0][0]
                full[:, j] = p[:, i]
            agg += full
        return agg / self.n_estimators

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def feature_importances_(self):
        imp = np.zeros(self.n_features_)
        for tree in self.estimators_:
            imp += tree.feature_importances_
        return imp / self.n_estimators
