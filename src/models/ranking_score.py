"""Common positive-class ranking score for Phase 5 (Addendum C, item C5).

Both Phase 5 model families are ranked, evaluated (AP, ROC-AUC, Precision@K,
queue) and calibrated on the same score: the estimator's binary
``decision_function``. Verified for scikit-learn 1.9.1 (Phase 5B-0 report,
Section 6):

* ``LogisticRegression.decision_function`` returns ``X @ coef_ + intercept_``,
  the log-odds of ``classes_[1]``. Its ``predict_proba`` is ``expit`` of it.
* ``HistGradientBoostingClassifier.decision_function`` returns the raw sum of
  tree leaves (float64), the log-odds of ``classes_[1]``. Its ``predict_proba``
  is ``expit`` of it.
* ``CalibratedClassifierCV`` feeds ``decision_function`` (preferred over
  ``predict_proba``) to the sigmoid calibrator.

``expit`` is strictly increasing in exact arithmetic but not in float64: every
logit at or above about 36.74 maps to exactly 1.0, and logits above about 30
lose resolution, creating artificial ties. ``decision_function`` has no such
saturation, so it is the score used everywhere.

This module only transforms scores that a caller already has. It fits nothing.
"""

from __future__ import annotations

import numpy as np

from src.models.phase5_settings import (
    CALIBRATION_CONSTANT_LOGIT_SD,
    RANKING_SCORE_METHOD,
    RELIABILITY_BIN_EDGES,
)


def ranking_scores(estimator, X, positive_label=1) -> np.ndarray:
    """Return the float64 positive-class ranking score of a *fitted* binary estimator.

    Larger means more likely ``positive_label``. Uses ``decision_function`` only;
    there is deliberately no fallback to ``predict_proba``. Raises if the
    estimator lacks ``decision_function``, is not binary, does not know
    ``positive_label``, or returns a non-finite or mis-shaped score.
    """
    if not hasattr(estimator, RANKING_SCORE_METHOD):
        raise TypeError(
            f"{type(estimator).__name__} has no {RANKING_SCORE_METHOD}; Phase 5 ranking requires it."
        )
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,):
        raise ValueError(f"A binary estimator is required; got classes {classes.tolist()}.")
    if positive_label not in classes.tolist():
        raise ValueError(f"positive_label {positive_label!r} is not one of {classes.tolist()}.")

    scores = np.asarray(getattr(estimator, RANKING_SCORE_METHOD)(X), dtype=np.float64)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 1:
        raise ValueError(f"Expected a one-dimensional score, got shape {scores.shape}.")
    if not np.isfinite(scores).all():
        raise ValueError("The ranking score contains non-finite values.")
    # scikit-learn's binary decision_function scores classes_[1]. Negation is
    # exact in floating point, so it reverses the order without creating ties.
    if positive_label == classes[0]:
        scores = -scores
    return np.array(scores, dtype=np.float64, copy=True)


def ranking_order(scores, times, positions) -> np.ndarray:
    """Indices in review order under the protocol Section 9.2 key.

    Key: score descending, then ``Time`` ascending, then file row position
    ascending. Comparisons are exact (no tolerance); ``-0.0`` equals ``0.0``.
    The key never reads a label.
    """
    scores = np.asarray(scores, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions)
    if not (scores.ndim == times.ndim == positions.ndim == 1):
        raise ValueError("scores, times and positions must be one-dimensional.")
    if not (len(scores) == len(times) == len(positions)):
        raise ValueError("scores, times and positions must have the same length.")
    if not (np.isfinite(scores).all() and np.isfinite(times).all()):
        raise ValueError("scores and times must be finite.")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("File row positions must be unique.")
    # lexsort sorts by the last key first.
    return np.lexsort((positions, times, -scores))


def protocol_platt_parameters(sklearn_a: float, sklearn_b: float) -> tuple[float, float]:
    """Convert scikit-learn's sigmoid parameters to the protocol's ``(A, B)``.

    scikit-learn: ``p = expit(-(a_ * s + b_))``. Protocol: ``z = A * s + B``,
    ``p = expit(z)``. Hence ``A = -a_`` and ``B = -b_``; the protocol's
    positive-slope requirement (``A > 0``) is ``a_ < 0`` in scikit-learn terms.
    """
    a, b = float(sklearn_a), float(sklearn_b)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Sigmoid parameters must be finite.")
    return -a, -b


def calibrated_logits(scores, A: float, B: float) -> np.ndarray:
    """Return ``z = A * s + B`` in float64, computed directly from the scores."""
    return float(A) * np.asarray(scores, dtype=np.float64) + float(B)


def is_numerically_constant(logits, tolerance: float = CALIBRATION_CONSTANT_LOGIT_SD) -> bool:
    """Protocol numerical-constancy guard: SD (ddof=0) below ``tolerance`` or any non-finite logit."""
    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one logit is required.")
    if not np.isfinite(values).all():
        return True
    return bool(np.std(values) < tolerance)


def reliability_bin_index(probabilities, edges=RELIABILITY_BIN_EDGES) -> np.ndarray:
    """Reliability-bin index of each calibrated probability (Addendum C, item C8).

    Bin ``k`` holds ``p`` iff ``edges[k] <= p < edges[k+1]``; the last bin is
    closed, so ``p == edges[-1]`` (1.0) falls in it. Comparisons are exact in
    float64. Raises on non-finite values or values outside ``[edges[0], edges[-1]]``.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Bin edges must be a strictly increasing sequence of at least two values.")
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")
    if p.size and (p.min() < edges[0] or p.max() > edges[-1]):
        raise ValueError(f"Probabilities must lie in [{edges[0]}, {edges[-1]}].")
    index = np.searchsorted(edges, p, side="right") - 1
    return np.minimum(index, edges.size - 2)
