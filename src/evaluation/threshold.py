"""Threshold analysis for probability-scoring binary classifiers (Phase 4c).

Conventions (``docs/TECHNICAL_DOCUMENTATION.md``):

* **Candidates** are the unique predicted probabilities of the evaluated rows
  (``numpy.unique``, ascending). No fixed grid is used.
* **Decision rule** is inclusive: a row is predicted positive when
  ``y_score >= threshold``.
* **Metrics** per threshold: TP, FP, TN, FN, precision, recall, F1, predicted
  positives and alert rate. F1 is computed as ``2·TP / (2·TP + FP + FN)``,
  which equals ``2·P·R / (P + R)`` whenever both are defined. Computing it from
  integer counts with one division means equal F1 values compare exactly
  equal, so ties are detected reliably.
* **Undefined values** are ``NaN`` (serialized as JSON ``null``): precision
  when nothing is predicted positive, recall when there are no positives, and
  F1 when either precision or recall is undefined. When both are defined but
  zero (TP = 0), F1 is 0.
* **Selection** maximizes F1. Ties are broken by choosing the *lowest*
  threshold. This is an evaluation convention, not a business optimum.
"""

from __future__ import annotations

import math

import numpy as np

CURVE_COLUMNS = (
    "threshold", "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "n_predicted_positive", "alert_rate",
)


def _validate_scores(y_score) -> np.ndarray:
    """Return scores as a float array after checking they are finite probabilities."""
    y_score = np.asarray(y_score)
    if y_score.ndim != 1:
        raise ValueError("y_score must be one-dimensional.")
    if len(y_score) == 0:
        raise ValueError("Cannot analyse thresholds on empty inputs.")
    if y_score.dtype == bool or not np.issubdtype(y_score.dtype, np.number):
        raise TypeError("y_score must be numeric probabilities.")
    y_score = y_score.astype(float)
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains missing or infinite values.")
    if (y_score < 0).any() or (y_score > 1).any():
        raise ValueError("y_score must lie in [0, 1].")
    return y_score


def _validate(y_true, y_score, positive_label) -> tuple[np.ndarray, np.ndarray]:
    """Return (boolean positives, float scores) after validating the inputs."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if len(y_true) != len(np.asarray(y_score)):
        raise ValueError(f"Length mismatch: {len(y_true)} labels vs {len(np.asarray(y_score))} scores.")
    scores = _validate_scores(y_score)
    if _has_missing(y_true):
        raise ValueError("y_true contains missing values.")
    return y_true == positive_label, scores


def _has_missing(values: np.ndarray) -> bool:
    if values.dtype.kind == "f":
        return bool(np.isnan(values).any())
    if values.dtype == object:
        return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)
    return False


def threshold_candidates(y_score) -> np.ndarray:
    """Unique predicted probabilities, ascending."""
    return np.unique(_validate_scores(y_score))


def _metrics_from_counts(tp, fp, tn, fn) -> dict[str, np.ndarray]:
    """Vectorized metrics from count arrays, with NaN for undefined values."""
    tp, fp, tn, fn = (np.asarray(a, dtype=np.int64) for a in (tp, fp, tn, fn))
    predicted = tp + fp
    positives = tp + fn
    total = tp + fp + tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.where(predicted > 0, predicted, 1), np.nan)
        recall = np.where(positives > 0, tp / np.where(positives > 0, positives, 1), np.nan)
        denominator = 2 * tp + fp + fn
        f1 = np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1), np.nan)
    f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_predicted_positive": predicted, "alert_rate": predicted / total,
    }


def threshold_curve(y_true, y_score, positive_label=1) -> dict[str, list]:
    """Metrics at every unique-score threshold, as columns of equal-length lists.

    Counts are exact: for each candidate ``t``, TP is the number of positives
    with ``score >= t`` (found by binary search over the sorted scores).
    """
    is_positive, scores = _validate(y_true, y_score, positive_label)
    thresholds = np.unique(scores)
    positive_scores = np.sort(scores[is_positive])
    negative_scores = np.sort(scores[~is_positive])
    tp = len(positive_scores) - np.searchsorted(positive_scores, thresholds, side="left")
    fp = len(negative_scores) - np.searchsorted(negative_scores, thresholds, side="left")
    fn = len(positive_scores) - tp
    tn = len(negative_scores) - fp
    metrics = _metrics_from_counts(tp, fp, tn, fn)
    curve = {"threshold": thresholds.tolist()}
    for column in CURVE_COLUMNS[1:]:
        values = metrics[column]
        curve[column] = values.tolist() if values.dtype.kind == "f" else [int(v) for v in values]
    return curve


def evaluate_threshold(y_true, y_score, threshold: float, positive_label=1) -> dict:
    """Metrics for one fixed threshold (``y_score >= threshold`` is positive)."""
    is_positive, scores = _validate(y_true, y_score, positive_label)
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    predicted = scores >= threshold
    tp = int((predicted & is_positive).sum())
    fp = int((predicted & ~is_positive).sum())
    fn = int((~predicted & is_positive).sum())
    tn = int((~predicted & ~is_positive).sum())
    metrics = _metrics_from_counts([tp], [fp], [tn], [fn])
    row = {"threshold": threshold}
    for column in CURVE_COLUMNS[1:]:
        value = metrics[column][0]
        row[column] = float(value) if metrics[column].dtype.kind == "f" else int(value)
    return row


def select_max_f1_threshold(curve: dict[str, list]) -> dict:
    """Return the curve row with maximal F1, breaking ties by the lowest threshold.

    Raises ``ValueError`` when no threshold has a defined F1 (for example,
    when the evaluated rows contain no positives).
    """
    thresholds = np.asarray(curve["threshold"], dtype=float)
    f1 = np.asarray(curve["f1"], dtype=float)
    if len(thresholds) == 0 or len(thresholds) != len(f1):
        raise ValueError("Curve must contain matching, non-empty threshold and F1 columns.")
    defined = ~np.isnan(f1)
    if not defined.any():
        raise ValueError("F1 is undefined at every threshold; no threshold can be selected.")
    best = f1[defined].max()
    tied = np.flatnonzero(defined & (f1 == best))
    index = tied[np.argmin(thresholds[tied])]
    return {column: curve[column][index] for column in CURVE_COLUMNS}
