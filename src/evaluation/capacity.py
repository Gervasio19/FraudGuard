"""Fixed-K alert-capacity metrics (protocol Section 9.2), Phase 5B-2.

Review exactly ``K`` transactions: the top ``K`` under the frozen ranking key
(score descending, ``Time`` ascending, file row position ascending;
:func:`src.models.ranking_score.ranking_order`). The key never reads a label.

Reported per K:

* ``tp`` / ``fp``: frauds and legitimate transactions inside the reviewed set
  (``fp`` = legitimate transactions that would enter the alert queue).
* ``precision_at_k = tp / K``; ``recall_at_k = tp / n_positive`` (the fraud
  capture rate; NaN when the period holds no fraud, as in
  :mod:`src.evaluation.metrics`).
* The oracle ceiling ``min(K, n_positive) / n_positive``.
* The boundary-tie report: rows whose score equals the K-th score, how many of
  them are inside the top K, the frauds among them, the TP range over every
  ordering of that tied group ``[tp_min, tp_max]``, and the expected TP under a
  uniformly random tie order (hypergeometric mean). ``tie_sensitive`` is True
  iff ``tp_min != tp_max``.

AP and ROC-AUC (scikit-learn) treat tied scores as one step and need no
tie-break; fixed-K does. The two therefore need not agree on tied boundaries.
"""

from __future__ import annotations

import math

import numpy as np

from src.models.ranking_score import ranking_order


def fixed_k_metrics(y_true, scores, times, positions, k: int, positive_label=1) -> dict:
    """Precision@K, Recall@K and the boundary-tie report for one score vector."""
    y = np.asarray(y_true) == positive_label
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if y.shape != scores.shape:
        raise ValueError("y_true and scores must have the same length.")
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("K must be a positive integer.")
    k = int(k)
    if k > n:
        raise ValueError(f"K = {k} exceeds the number of rows ({n}).")

    order = ranking_order(scores, times, positions)
    reviewed = order[:k]
    tp = int(y[reviewed].sum())
    n_positive = int(y.sum())

    kth_score = scores[order[k - 1]]
    above = scores > kth_score
    tied = scores == kth_score
    n_above, n_tied = int(above.sum()), int(tied.sum())
    k_in = k - n_above
    frauds_above, frauds_tied = int(y[above].sum()), int(y[tied].sum())
    tp_min = frauds_above + max(0, k_in - (n_tied - frauds_tied))
    tp_max = frauds_above + min(k_in, frauds_tied)
    if not tp_min <= tp <= tp_max:
        raise AssertionError("Deterministic TP lies outside its tie range.")

    return {
        "k": k,
        "n_rows": n,
        "n_positive": n_positive,
        "alert_share_of_rows": k / n,
        "tp": tp,
        "fp": k - tp,
        "precision_at_k": tp / k,
        "recall_at_k": tp / n_positive if n_positive else math.nan,
        "oracle_recall_ceiling": min(k, n_positive) / n_positive if n_positive else math.nan,
        "ties": {
            "n_tied_at_boundary": n_tied,
            "n_tied_inside_top_k": k_in,
            "n_tied_outside_top_k": n_tied - k_in,
            "frauds_in_tied_group": frauds_tied,
            "tp_min": tp_min,
            "tp_max": tp_max,
            "expected_tp_random_tie_order": frauds_above + k_in * frauds_tied / n_tied,
            "tie_sensitive": tp_min != tp_max,
        },
    }


def fixed_k_table(y_true, scores, times, positions, ks, positive_label=1) -> list[dict]:
    """:func:`fixed_k_metrics` for every K in ``ks`` (same scores, same order)."""
    return [fixed_k_metrics(y_true, scores, times, positions, k, positive_label) for k in ks]
