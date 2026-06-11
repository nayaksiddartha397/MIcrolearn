"""
microlearn.unsupervised
=======================
Unsupervised learning algorithms.

Classes
-------
KMeans  – Lloyd's algorithm with k-means++ initialisation option
PCA     – eigendecomposition of the covariance matrix
"""

import numpy as np
from .base import BaseEstimator


# ================================================================ KMEANS ===

class KMeans(BaseEstimator):
    """K-Means clustering via Lloyd's algorithm.

    Parameters
    ----------
    n_clusters : int, default 3
    max_iters : int, default 300
    tol : float, default 1e-4
        Convergence tolerance (centroid shift norm).
    init : {'random', 'k-means++'}, default 'random'
        Centroid initialisation strategy.
    random_state : int or None, default None

    Attributes
    ----------
    centroids_ : ndarray (n_clusters, n_features)
    labels_    : ndarray (n_samples,)  – cluster index per sample
    inertia_   : float – total within-cluster sum of squares
    """

    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4,
                 init="random", random_state=None):
        self.n_clusters   = n_clusters
        self.max_iters    = max_iters
        self.tol          = tol
        self.init         = init
        self.random_state = random_state

    # ---- initialisation --------------------------------------------------
    def _init_centroids(self, X):
        if self.init == "k-means++":
            return self._kmeanspp(X)
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _kmeanspp(self, X):
        centres = [X[np.random.randint(len(X))]]
        for _ in range(self.n_clusters - 1):
            d2 = np.array([min(np.sum((x - c) ** 2) for c in centres) for x in X])
            probs = d2 / d2.sum()
            centres.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centres)

    # ---- assignment step -------------------------------------------------
    def _assign(self, X):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids_])
        return np.argmin(dists, axis=0)

    # ---- public API -------------------------------------------------------
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.centroids_ = self._init_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_c  = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any()
                else self.centroids_[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_c - self.centroids_)
            self.centroids_ = new_c
            if shift < self.tol:
                break

        self.labels_  = self._assign(X)
        self.inertia_ = float(sum(
            np.sum((X[self.labels_ == k] - self.centroids_[k]) ** 2)
            for k in range(self.n_clusters)
        ))
        return self

    def predict(self, X):
        return self._assign(np.asarray(X, float))

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def score(self, X, y=None):
        """Return negative inertia (higher is better, mirrors sklearn)."""
        labels = self._assign(np.asarray(X, float))
        inertia = sum(
            np.sum((X[labels == k] - self.centroids_[k]) ** 2)
            for k in range(self.n_clusters)
        )
        return -float(inertia)


# ==================================================================== PCA ===

class PCA(BaseEstimator):
    """Principal Component Analysis via eigendecomposition.

    Centers the data, computes the covariance matrix, then takes the
    top *n_components* eigenvectors as principal axes.

    Parameters
    ----------
    n_components : int, default 2

    Attributes
    ----------
    components_              : ndarray (n_components, n_features)
        Principal axes (each row is an eigenvector).
    explained_variance_      : ndarray (n_components,)
    explained_variance_ratio_: ndarray (n_components,)
    mean_                    : ndarray (n_features,)
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Covariance matrix (uses 1/(n-1) Bessel correction)
        cov = np.cov(Xc.T)

        # eigh is numerically stable for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by descending eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        k = self.n_components
        self.components_               = eigenvectors[:, :k].T
        self.explained_variance_       = eigenvalues[:k]
        self.explained_variance_ratio_ = eigenvalues[:k] / (eigenvalues.sum() + 1e-10)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return np.asarray(X, float) @ self.components_ + self.mean_
