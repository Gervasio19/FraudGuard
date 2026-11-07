"""Class-sensitive evaluation metrics for binary fraud classifiers.

Threshold-free ranking metrics (average precision, ROC-AUC) use predicted
positive-class probabilities. Recall, precision and the confusion matrix use
hard labels and therefore depend on a decision rule; in Phase 4b that rule is
each estimator's default ``predict`` output, which is *not* a selected
operating threshold.

Undefined values are reported as ``NaN`` rather than silently replaced:
precision is undefined when nothing is predicted positive, recall and average
precision when there are no positives, and ROC-AUC when only one class occurs.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_score, y_pred, positive_label: int = 1) -> dict:
    """Return evaluation metrics for one model on one partition.

    Parameters
    ----------
    y_true:
        True labels.
    y_score:
        Predicted probability of ``positive_label`` for each row.
    y_pred:
        Hard predictions from the estimator's default decision rule.
    positive_label:
        Label value treated as the positive (fraud) class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred)
    if not (len(y_true) == len(y_score) == len(y_pred)):
        raise ValueError("y_true, y_score and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty partition.")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains missing or infinite values.")

    is_positive = y_true == positive_label
    predicted_positive = y_pred == positive_label
    n_positive = int(is_positive.sum())
    n_predicted_positive = int(predicted_positive.sum())
    both_classes = 0 < n_positive < len(y_true)

    tn, fp, fn, tp = confusion_matrix(is_positive, predicted_positive, labels=[False, True]).ravel()

    return {
        "n_rows": int(len(y_true)),
        "n_positive": n_positive,
        "prevalence": n_positive / len(y_true),
        "average_precision": (
            float(average_precision_score(is_positive, y_score)) if n_positive else math.nan
        ),
        "roc_auc": float(roc_auc_score(is_positive, y_score)) if both_classes else math.nan,
        "precision": (
            float(precision_score(is_positive, predicted_positive, zero_division=np.nan))
            if n_predicted_positive else math.nan
        ),
        "recall": (
            float(recall_score(is_positive, predicted_positive, zero_division=np.nan))
            if n_positive else math.nan
        ),
        "n_predicted_positive": n_predicted_positive,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
