"""
microlearn.metrics
==================
Evaluation metrics — pure NumPy, no sklearn.

Classification
--------------
accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report, log_loss, roc_auc_score

Regression
----------
mean_squared_error, mean_absolute_error, r2_score
"""

import numpy as np


# ========================================================== REGRESSION ===

def mean_squared_error(y_true, y_pred, squared=True):
    """Mean squared error (or RMSE when squared=False)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse if squared else np.sqrt(mse))


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """R² coefficient of determination."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ====================================================== CLASSIFICATION ===

def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """Confusion matrix — rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    idx     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    mat     = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[idx[t], idx[p]] += 1
    return mat


# ---- shared helper: per-class TP / FP / FN counts ----------------------

def _class_stats(y_true, y_pred):
    """Return arrays (precision, recall, f1, support) over sorted classes."""
    classes  = np.unique(y_true)
    ps, rs, fs, ss = [], [], [], []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        s  = int(np.sum(y_true == c))
        p  = tp / (tp + fp + 1e-10)
        r  = tp / (tp + fn + 1e-10)
        f  = 2 * p * r / (p + r + 1e-10)
        ps.append(p); rs.append(r); fs.append(f); ss.append(s)
    return (np.array(ps), np.array(rs),
            np.array(fs), np.array(ss))


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    """Precision.

    Parameters
    ----------
    average : {'binary','macro','weighted','micro', None}
    pos_label : used only when average='binary'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return float(tp / (tp + fp + 1e-10))

    ps, _, _, ss = _class_stats(y_true, y_pred)
    if average is None:      return ps
    if average == "macro":   return float(ps.mean())
    if average == "weighted":return float(np.average(ps, weights=ss))
    if average == "micro":
        tp_all = sum(np.sum((y_pred == c) & (y_true == c)) for c in np.unique(y_true))
        fp_all = sum(np.sum((y_pred == c) & (y_true != c)) for c in np.unique(y_true))
        return float(tp_all / (tp_all + fp_all + 1e-10))
    raise ValueError(f"Unknown average: {average!r}")


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    """Recall (sensitivity).

    Parameters
    ----------
    average : {'binary','macro','weighted','micro', None}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(tp / (tp + fn + 1e-10))

    _, rs, _, ss = _class_stats(y_true, y_pred)
    if average is None:      return rs
    if average == "macro":   return float(rs.mean())
    if average == "weighted":return float(np.average(rs, weights=ss))
    if average == "micro":
        tp_all = sum(np.sum((y_pred == c) & (y_true == c)) for c in np.unique(y_true))
        fn_all = sum(np.sum((y_pred != c) & (y_true == c)) for c in np.unique(y_true))
        return float(tp_all / (tp_all + fn_all + 1e-10))
    raise ValueError(f"Unknown average: {average!r}")


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    """F1 score — harmonic mean of precision and recall.

    Parameters
    ----------
    average : {'binary','macro','weighted','micro', None}
    """
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true,   y_pred, average=average, pos_label=pos_label)
    if isinstance(p, np.ndarray):
        return 2 * p * r / (p + r + 1e-10)
    return float(2 * p * r / (p + r + 1e-10))


def log_loss(y_true, y_pred_proba, eps=1e-15):
    """Cross-entropy / log loss.

    Parameters
    ----------
    y_true : array-like (n_samples,) — integer class labels
    y_pred_proba : array-like
        Shape (n_samples,) for binary, or (n_samples, n_classes) for multi.
    """
    y_true       = np.asarray(y_true)
    y_pred_proba = np.clip(np.asarray(y_pred_proba, float), eps, 1 - eps)

    if y_pred_proba.ndim == 1:
        # Binary: labels are 0/1
        return float(-np.mean(
            y_true * np.log(y_pred_proba) +
            (1 - y_true) * np.log(1 - y_pred_proba)
        ))

    # Multi-class: one-hot encode y_true
    classes  = np.unique(y_true)
    y_onehot = np.zeros((len(y_true), len(classes)))
    for i, c in enumerate(classes):
        y_onehot[y_true == c, i] = 1.0
    return float(-np.mean(np.sum(y_onehot * np.log(y_pred_proba), axis=1)))


def roc_auc_score(y_true, y_score):
    """Area Under the ROC Curve (binary classification only).

    Parameters
    ----------
    y_true  : array-like — binary labels (0 / 1)
    y_score : array-like — continuous decision scores or probabilities
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score, float)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires both positive and negative samples.")

    # Mann-Whitney U statistic with average ranks for tied scores.  Walking
    # tied samples one-by-one makes AUC depend on their arbitrary sort order.
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; every tied observation receives the mean rank.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# -------------------------------------------------------- report ---------

def classification_report(y_true, y_pred, target_names=None):
    """Text report of precision, recall, F1, and support per class.

    Parameters
    ----------
    y_true       : array-like
    y_pred       : array-like
    target_names : list of str, optional — display names for classes

    Returns
    -------
    report : str
    """
    y_true   = np.asarray(y_true)
    y_pred   = np.asarray(y_pred)
    classes  = np.unique(y_true)

    if target_names is None:
        target_names = [str(c) for c in classes]

    ps, rs, fs, ss = _class_stats(y_true, y_pred)
    total = len(y_true)

    W = max(max(len(n) for n in target_names), 12)   # column width

    header = (f"\n{'':>{W}}  {'precision':>9}  {'recall':>9}"
              f"  {'f1-score':>9}  {'support':>7}\n\n")

    rows = []
    for name, p, r, f, s in zip(target_names, ps, rs, fs, ss):
        rows.append(f"{name:>{W}}  {p:9.2f}  {r:9.2f}  {f:9.2f}  {s:7d}")

    acc = accuracy_score(y_true, y_pred)
    mp  = ps.mean();                    mr  = rs.mean();  mf  = fs.mean()
    wp  = np.average(ps, weights=ss);  wr  = np.average(rs, weights=ss)
    wf  = np.average(fs, weights=ss)

    footer = (
        f"\n\n{'accuracy':>{W}}  {'':>9}  {'':>9}  {acc:9.2f}  {total:7d}"
        f"\n{'macro avg':>{W}}  {mp:9.2f}  {mr:9.2f}  {mf:9.2f}  {total:7d}"
        f"\n{'weighted avg':>{W}}  {wp:9.2f}  {wr:9.2f}  {wf:9.2f}  {total:7d}"
    )

    return header + "\n".join(rows) + footer
