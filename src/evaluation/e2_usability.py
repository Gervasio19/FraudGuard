"""E2 usable-split count (Phase 5B-1A; protocol Section 7.5, Addendum C3/C4).

For each of the ``E2_N_SPLITS`` (2,000) sequential selection-block draws of
``default_rng(E2_SPLIT_SEED)`` (42), the buffered held-out half is formed with
the frozen half-open block and 300-second buffer convention
(``src/data/validation_blocks.py``). A split is **usable** iff its buffered
held-out half holds at least ``E2_MIN_HELDOUT_FRAUDS`` (10) frauds (Addendum
C4.6). E2 is **estimable** iff at least ``E2_MIN_USABLE_SPLITS`` (1,000) of the
attempted splits are usable (Section 7.5 step 8, decision #3b).

Only validation timestamps and validation labels are read. No score, model or
metric is involved, and no threshold is chosen here: every constant comes from
``src/models/phase5_settings.py``. Timestamps outside the validation period
(train or test rows) are refused by the block assignment, so test labels can
never enter the count.

This module counts usability only. The E2 statistics themselves (re-selection,
held-out delta-AP, split-stability frequency) need the 28 cached configuration
scores and are a later 5B stage.
"""

from __future__ import annotations

import numpy as np

from src.data.validation_blocks import assign_blocks, block_edges, draw_selection_blocks, heldout_keep_mask
from src.models.phase5_settings import (
    E2_BLOCK_SECONDS,
    E2_BUFFER_SECONDS,
    E2_MIN_HELDOUT_FRAUDS,
    E2_MIN_USABLE_SPLITS,
    E2_N_BLOCKS,
    E2_N_SELECTION_BLOCKS,
    E2_N_SPLITS,
    E2_SPLIT_SEED,
    VALIDATION_START_TIME,
)
from src.data.split import VALIDATION_END_TIME


def _summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"min": float(v.min()), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "max": float(v.max()), "mean": float(v.mean())}


def split_counts(times, y_true, edges, selection_draws, buffer_seconds=E2_BUFFER_SECONDS,
                 positive_label=1) -> dict[str, np.ndarray]:
    """Per-split row and fraud counts of the selection half and the buffered held-out half."""
    times = np.asarray(times)
    positive = np.asarray(y_true) == positive_label
    if times.shape != positive.shape or times.ndim != 1:
        raise ValueError("times and y_true must be one-dimensional and of equal length.")
    blocks = assign_blocks(times, edges)  # raises on any timestamp outside the edges
    draws = np.asarray(selection_draws, dtype=np.int64)
    n = len(draws)
    out = {k: np.empty(n, dtype=np.int64) for k in (
        "selection_rows", "selection_frauds", "heldout_rows", "heldout_frauds",
        "buffer_dropped_rows", "buffer_dropped_frauds")}
    for i, selection in enumerate(draws):
        in_selection = np.isin(blocks, selection)
        keep = heldout_keep_mask(times, edges, selection, buffer_seconds)
        if (keep & in_selection).any():
            raise AssertionError("A row was assigned to both halves.")
        dropped = ~in_selection & ~keep
        out["selection_rows"][i] = in_selection.sum()
        out["selection_frauds"][i] = positive[in_selection].sum()
        out["heldout_rows"][i] = keep.sum()
        out["heldout_frauds"][i] = positive[keep].sum()
        out["buffer_dropped_rows"][i] = dropped.sum()
        out["buffer_dropped_frauds"][i] = positive[dropped].sum()
    return out


def usability_decision(heldout_frauds, min_heldout_frauds=E2_MIN_HELDOUT_FRAUDS,
                       min_usable_splits=E2_MIN_USABLE_SPLITS) -> dict:
    """Apply the pre-registered usability and minimum-usable-split rules."""
    heldout_frauds = np.asarray(heldout_frauds)
    usable = heldout_frauds >= min_heldout_frauds
    n_usable = int(usable.sum())
    estimable = n_usable >= min_usable_splits
    return {
        "n_attempted": int(heldout_frauds.size),
        "n_usable": n_usable,
        "n_skipped": int(heldout_frauds.size - n_usable),
        "usable_share": n_usable / heldout_frauds.size,
        "min_heldout_frauds_rule": f"usable iff buffered held-out frauds >= {min_heldout_frauds}",
        "min_usable_splits_rule": f"E2 estimable iff n_usable >= {min_usable_splits}",
        "e2_estimable": bool(estimable),
        "consequence": (
            "E2 outputs may be computed in the E2 stage (usable splits only)" if estimable else
            "E2 not estimable under the pre-registered rule: no split-stability frequency or median "
            "held-out delta-AP is computed, E2 does not pass, and logistic regression is retained "
            "(Section 4.3). No retry with other sizes, buffers, seeds, B or thresholds."
        ),
    }


def e2_usability(times, y_true, positive_label=1) -> dict:
    """Run the full usability count on validation timestamps and labels. JSON-ready."""
    edges = block_edges(VALIDATION_START_TIME, VALIDATION_END_TIME, E2_BLOCK_SECONDS)
    if len(edges) - 1 != E2_N_BLOCKS:
        raise ValueError(f"Expected {E2_N_BLOCKS} blocks, got {len(edges) - 1}.")
    draws = draw_selection_blocks(E2_N_SPLITS, E2_N_BLOCKS, E2_N_SELECTION_BLOCKS, E2_SPLIT_SEED)
    counts = split_counts(times, y_true, edges, draws, E2_BUFFER_SECONDS, positive_label)
    decision = usability_decision(counts["heldout_frauds"])

    positive = np.asarray(y_true) == positive_label
    blocks = assign_blocks(times, edges)
    n_rows, n_frauds = int(len(blocks)), int(positive.sum())
    values, freq = np.unique(counts["heldout_frauds"], return_counts=True)
    usable = counts["heldout_frauds"] >= E2_MIN_HELDOUT_FRAUDS
    return {
        "settings": {
            "seed": E2_SPLIT_SEED, "generator": "numpy.random.default_rng, sequential draws",
            "n_splits_attempted": E2_N_SPLITS, "n_blocks": E2_N_BLOCKS,
            "n_selection_blocks": E2_N_SELECTION_BLOCKS, "block_seconds": E2_BLOCK_SECONDS,
            "buffer_seconds": E2_BUFFER_SECONDS, "intervals": "half-open [a, b); buffer [a-300, a) U [b, b+300)",
            "block_edges": [int(e) for e in edges],
        },
        "validation": {"n_rows": n_rows, "n_frauds": n_frauds,
                       "per_block": [{"block": j, "start": int(edges[j]), "end_exclusive": int(edges[j + 1]),
                                      "n_rows": int((blocks == j).sum()),
                                      "n_frauds": int(positive[blocks == j].sum())}
                                     for j in range(len(edges) - 1)]},
        "decision": decision,
        "diagnostics": {
            "n_distinct_splits": int(len(np.unique(draws, axis=0))),
            "first_three_splits": draws[:3].tolist(),
            "heldout_frauds": _summary(counts["heldout_frauds"]),
            "heldout_frauds_histogram": {str(int(v)): int(c) for v, c in zip(values, freq)},
            "heldout_frauds_usable_splits": _summary(counts["heldout_frauds"][usable]) if usable.any() else None,
            "heldout_rows": _summary(counts["heldout_rows"]),
            "heldout_row_share_of_validation": _summary(counts["heldout_rows"] / n_rows),
            "selection_frauds": _summary(counts["selection_frauds"]),
            "selection_rows": _summary(counts["selection_rows"]),
            "buffer_dropped_rows": _summary(counts["buffer_dropped_rows"]),
            "buffer_dropped_frauds": _summary(counts["buffer_dropped_frauds"]),
            "selection_frauds_plus_heldout_plus_dropped_equals_total": bool(np.all(
                counts["selection_frauds"] + counts["heldout_frauds"] + counts["buffer_dropped_frauds"]
                == n_frauds)),
        },
    }
