"""
microlearn
==========
A from-scratch machine-learning library with a scikit-learn-compatible API.
Built on NumPy only — no sklearn under the hood.

Quick start
-----------
>>> from microlearn.preprocessing import StandardScaler, Pipeline
>>> from microlearn.classification import LogisticRegression
>>> from microlearn.tree import RandomForestClassifier
>>> from microlearn.model_selection import train_test_split, cross_val_score
>>> from microlearn import metrics
>>>
>>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
>>> pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
>>> pipe.fit(X_tr, y_tr)
>>> print(pipe.score(X_te, y_te))
"""

from .linear import LinearRegression, RidgeRegression
from .classification import (
    LogisticRegression,
    KNeighborsClassifier,
    KNearestNeighbors,
    SVC,
)
from .tree import DecisionTreeClassifier, RandomForestClassifier
from .unsupervised import KMeans, PCA
from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    SimpleImputer,
    Pipeline,
    ColumnTransformer,
)
from . import metrics
from . import model_selection

__version__ = "0.1.0"
__author__  = "Your Name"

__all__ = [
    # Linear models
    "LinearRegression", "RidgeRegression",
    # Classifiers
    "LogisticRegression", "KNeighborsClassifier", "KNearestNeighbors", "SVC",
    # Trees
    "DecisionTreeClassifier", "RandomForestClassifier",
    # Unsupervised
    "KMeans", "PCA",
    # Preprocessing
    "StandardScaler", "MinMaxScaler", "OneHotEncoder",
    "LabelEncoder", "SimpleImputer", "Pipeline", "ColumnTransformer",
    # Sub-modules
    "metrics", "model_selection",
]
