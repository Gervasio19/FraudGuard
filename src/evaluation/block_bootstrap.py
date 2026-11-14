"""Conditional paired block bootstrap (protocol Sections 7.2-7.4; Addendum C3), Phase 5B-3.

Blocks are **time intervals** in whole seconds, not row counts. The evaluated
period is ``[t0, t0 + T)`` with ``T = t1 - t0 + 1`` seconds. A replicate
concatenates ``m = ceil(T / L)`` blocks and truncates the last one, so the
resampled duration is exactly ``T``; the resampled row and fraud counts vary.

Schemes (block start ``s`` is an offset in seconds from ``t0``):

* ``cbb`` (primary): circular block bootstrap. ``s`` uniform on ``{0, ..., T-1}``;
  a block that passes the end continues at the start. Every second has the same
  inclusion weight.
* ``mbb`` (sensitivity): moving blocks, no wrap. ``s`` uniform on
  ``{0, ..., T-L}``; the truncated last block takes the first seconds of its draw.
* ``stationary`` (sensitivity; Politis-Romano): circular, with geometric block
  lengths of mean ``L`` seconds (``Generator.geometric(1 / L)``), drawn until the
  duration reaches ``T``; the last block is truncated at ``T`` (mentor decision D7).

A replicate is represented by an integer **weight** per original row (how many
times the row was drawn). Every statistic is then a weighted statistic of the
original rows, which equals the statistic of the expanded resample. Both models
always use the same weights (pairing).

Random numbers: one fresh ``numpy.random.default_rng(seed)`` per analysis
(partition, scheme, block length). Draw order, fixed and tested:

* ``cbb`` / ``mbb``: a single ``rng.integers(0, high, size=(B, m))`` call;
* ``stationary``: per replicate, alternately ``rng.integers(0, T)`` (start) and
  ``rng.geometric(1 / L)`` (length) until the duration reaches ``T``.

Nothing here fits a model. Labels are used only to evaluate.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from src.data.validation_blocks import assign_blocks, block_edges
from src.models.ranking_score import ranking_order

CBB, MBB, STATIONARY = "cbb", "mbb", "stationary"
SCHEMES = (CBB, MBB, STATIONARY)


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------

def period_offsets(times, start_time: int, end_time_inclusive: int) -> tuple[np.ndarray, int]:
    """Whole-second offsets of each row from ``start_time`` and the period length ``T``."""
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values != np.floor(values)).any():
        raise ValueError("Timestamps must be one-dimensional whole seconds.")
    T = int(end_time_inclusive) - int(start_time) + 1
    if T <= 0:
        raise ValueError("The period must contain at least one second.")
    offsets = values.astype(np.int64) - int(start_time)
    if offsets.size and (offsets.min() < 0 or offsets.max() >= T):
        raise ValueError("Timestamps fall outside the period.")
    return offsets, T


def fixed_block_lengths(T: int, block_seconds: int) -> np.ndarray:
    """``m = ceil(T / L)`` block lengths; the last is truncated so that they sum to ``T``."""
    if block_seconds <= 0 or int(block_seconds) != block_seconds:
        raise ValueError("block_seconds must be a positive whole number.")
    m = math.ceil(T / block_seconds)
    lengths = np.full(m, int(block_seconds), dtype=np.int64)
    lengths[-1] = T - (m - 1) * int(block_seconds)
    return lengths


def second_counts(starts, lengths, T: int) -> np.ndarray:
    """How many blocks cover each second offset ``0..T-1`` (blocks wrap circularly; length <= T)."""
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if starts.shape != lengths.shape:
        raise ValueError("starts and lengths must have the same shape.")
    if starts.size and (starts.min() < 0 or starts.max() >= T or lengths.min() < 1 or lengths.max() > T):
        raise ValueError("Block starts must lie in [0, T) and lengths in [1, T].")
    diff = np.zeros(T + 1, dtype=np.int64)
    ends = starts + lengths
    wraps = ends > T
    np.add.at(diff, starts, 1)
    np.add.at(diff, np.where(wraps, T, ends), -1)
    if wraps.any():
        diff[0] += int(wraps.sum())
        np.add.at(diff, ends[wraps] - T, -1)
    return np.cumsum(diff[:-1])


def iter_replicate_weights(offsets: np.ndarray, T: int, scheme: str, block_seconds: int,
                           n_replicates: int, seed: int) -> Iterator[np.ndarray]:
    """Yield the integer row weights of each replicate (see the module docstring for the draw order)."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; use one of {SCHEMES}.")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    L = int(block_seconds)
    rng = np.random.default_rng(seed)
    if scheme in (CBB, MBB):
        lengths = fixed_block_lengths(T, L)
        if scheme == MBB and L > T:
            raise ValueError("The moving-block scheme needs block_seconds <= T.")
        high = T if scheme == CBB else T - L + 1
        starts = rng.integers(0, high, size=(n_replicates, len(lengths)))
        for row in starts:
            yield second_counts(row, lengths, T)[offsets]
        return
    if L <= 0:
        raise ValueError("block_seconds must be positive.")
    for _ in range(n_replicates):
        starts, lengths, total = [], [], 0
        while total < T:
            start = int(rng.integers(0, T))
            length = min(int(rng.geometric(1.0 / L)), T - total)
            starts.append(start)
            lengths.append(length)
            total += length
        yield second_counts(starts, lengths, T)[offsets]


# ---------------------------------------------------------------------------
# Weighted ranking statistics
# ---------------------------------------------------------------------------

class RankedScores:
    """One model's scores in the frozen review order, for fast weighted AP and TP@K.

    The order is the Section 9.2 key (score descending, ``Time`` ascending, file
    position ascending). AP treats each group of equal scores as one threshold
    step, exactly as ``sklearn.metrics.average_precision_score`` does.
    """

    def __init__(self, scores, y_true, times, positions, positive_label=1):
        scores = np.asarray(scores, dtype=np.float64)
        self.order = ranking_order(scores, times, positions)
        self.positive_sorted = (np.asarray(y_true) == positive_label)[self.order]
        if self.positive_sorted.shape != scores.shape:
            raise ValueError("y_true and scores must have the same length.")
        ordered = scores[self.order]
        self.group_end = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
        self.fraud_rank = np.flatnonzero(self.positive_sorted)
        self.n = len(scores)

    def average_precision(self, weights=None) -> float:
        """Weighted AP; NaN when the weighted sample holds no fraud."""
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=np.float64)[self.order]
        tp = np.cumsum(w * self.positive_sorted)[self.group_end]
        total = np.cumsum(w)[self.group_end]
        if tp[-1] <= 0:
            return math.nan
        seen = total > 0
        tp, total = tp[seen], total[seen]
        return float(np.sum(np.diff(tp, prepend=0.0) * (tp / total)) / tp[-1])

    def tp_at_k(self, ks, weights=None) -> np.ndarray:
        """Frauds among exactly ``K`` reviewed rows of the weighted sample, for each K.

        Copies of a row are adjacent in the review order (same score, ``Time``
        and position), so the boundary row may be reviewed only in part.
        """
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=np.float64)[self.order]
        before = np.cumsum(w) - w
        if float(before[-1] + w[-1]) < max(ks):
            raise ValueError("K exceeds the number of rows in the weighted sample.")
        before, w = before[self.fraud_rank], w[self.fraud_rank]
        return np.array([np.clip(k - before, 0.0, w).sum() for k in ks], dtype=np.float64)


# ---------------------------------------------------------------------------
# Paired bootstrap of the fixed selected pair
# ---------------------------------------------------------------------------

def paired_block_bootstrap(ranked: dict[str, RankedScores], y_true, times, start_time: int,
                           end_time_inclusive: int, scheme: str, block_seconds: int, n_replicates: int,
                           seed: int, ks=(), positive_label=1) -> dict:
    """Per-replicate AP and TP@K of every model in ``ranked`` on identical replicate weights.

    Returns in-memory arrays (never written to a result file): ``n_rows``,
    ``n_frauds``, ``ap[name]`` and ``tp_at_k[name]`` (shape ``(B, len(ks))``).
    Replicates with zero frauds keep NaN AP; callers drop and count them.
    """
    offsets, T = period_offsets(times, start_time, end_time_inclusive)
    positive = (np.asarray(y_true) == positive_label).astype(np.float64)
    ks = tuple(int(k) for k in ks)
    out = {"n_rows": np.empty(n_replicates), "n_frauds": np.empty(n_replicates),
           "ap": {name: np.empty(n_replicates) for name in ranked},
           "tp_at_k": {name: np.empty((n_replicates, len(ks))) for name in ranked}}
    for b, weights in enumerate(iter_replicate_weights(offsets, T, scheme, block_seconds, n_replicates, seed)):
        w = weights.astype(np.float64)
        out["n_rows"][b] = w.sum()
        out["n_frauds"][b] = w @ positive
        for name, model in ranked.items():
            out["ap"][name][b] = model.average_precision(w)
            if ks:
                out["tp_at_k"][name][b] = model.tp_at_k(ks, w)
    return out


def stratified_iid_bootstrap(ranked: dict[str, RankedScores], y_true, n_replicates: int, seed: int,
                             positive_label=1) -> dict:
    """Class-stratified i.i.d. paired bootstrap (the Phase 4 audit scheme; continuity only).

    Frauds and legitimate rows are resampled separately with replacement, so
    every replicate keeps the observed class counts. It ignores temporal
    dependence and is reported for continuity, never as decision evidence.
    """
    positive = np.asarray(y_true) == positive_label
    pos_idx, neg_idx = np.flatnonzero(positive), np.flatnonzero(~positive)
    if not pos_idx.size or not neg_idx.size:
        raise ValueError("Both classes are required.")
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_replicates) for name in ranked}
    for b in range(n_replicates):
        w = np.zeros(len(positive))
        w[pos_idx] = np.bincount(rng.integers(0, pos_idx.size, pos_idx.size), minlength=pos_idx.size)
        w[neg_idx] = np.bincount(rng.integers(0, neg_idx.size, neg_idx.size), minlength=neg_idx.size)
        for name, model in ranked.items():
            out[name][b] = model.average_precision(w)
    return {"ap": out}


def interval_summary(values, observed: float | None = None, level: float = 0.95) -> dict:
    """Percentile interval and descriptive summaries of finite replicate values (NaN dropped, counted)."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    out = {"n_replicates": int(values.size), "n_valid": int(finite.size),
           "n_dropped": int(values.size - finite.size), "interval": "percentile", "level": level}
    if finite.size < 2:
        return {**out, "status": "not estimable (fewer than two valid replicates)"}
    alpha = (1.0 - level) / 2.0
    lower, upper = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    out.update({"lower": float(lower), "upper": float(upper), "mean": float(finite.mean()),
                "median": float(np.median(finite)), "standard_error": float(finite.std(ddof=1)),
                "share_at_or_below_zero": float((finite <= 0).mean())})
    if observed is not None:
        out["observed"] = float(observed)
        out["centring_bias_mean_minus_observed"] = float(finite.mean() - observed)
    return out


def block_jackknife(ranked: dict[str, RankedScores], pair: tuple[str, str], times, start_time: int,
                    end_time_inclusive: int, block_seconds: int) -> dict:
    """Delete-one-block jackknife standard error of ``AP(pair[0]) - AP(pair[1])`` (cross-check, Section 7.2)."""
    edges = block_edges(start_time, end_time_inclusive, block_seconds)
    blocks = assign_blocks(times, edges)
    n_blocks = len(edges) - 1
    deltas = np.empty(n_blocks)
    for j in range(n_blocks):
        w = (blocks != j).astype(np.float64)
        deltas[j] = ranked[pair[0]].average_precision(w) - ranked[pair[1]].average_precision(w)
    finite = deltas[np.isfinite(deltas)]
    se = (math.sqrt((finite.size - 1) / finite.size * np.sum((finite - finite.mean()) ** 2))
          if finite.size > 1 else math.nan)
    return {"n_blocks": n_blocks, "n_valid": int(finite.size), "block_seconds": int(block_seconds),
            "standard_error": se, "delta_min": float(finite.min()), "delta_max": float(finite.max())}


# ---------------------------------------------------------------------------
# Pre-registered dependence diagnostic (Section 7.4; descriptive, selects nothing)
# ---------------------------------------------------------------------------

def _dispersion_and_lag1(counts: np.ndarray) -> tuple[float, float]:
    mean = counts.mean()
    dispersion = float(counts.var(ddof=1) / mean) if mean > 0 and counts.size > 1 else math.nan
    centred = counts - mean
    denominator = float(centred @ centred)
    lag1 = float(centred[:-1] @ centred[1:] / denominator) if denominator > 0 else math.nan
    return dispersion, lag1


def status_dependence(fraud_times, status, start_time: int, end_time_inclusive: int, bin_minutes,
                      n_permutations: int, seed: int) -> dict:
    """Dispersion of a fraud status (for example "missed at K") over time bins, against a permutation null.

    ``status`` is a boolean per fraud. Frauds with the status are counted in
    consecutive bins aligned to the period start. The null shuffles the status
    among the frauds and keeps the fraud times. Reported per bin width:
    dispersion index (variance / mean, ``ddof=1``), the null 95th percentile,
    the permutation p-value ``(1 + #null >= observed) / (1 + n)`` and the lag-1
    autocorrelation of the bin counts.
    """
    fraud_times = np.asarray(fraud_times, dtype=np.int64)
    status = np.asarray(status, dtype=bool)
    if fraud_times.shape != status.shape:
        raise ValueError("fraud_times and status must have the same length.")
    n_with = int(status.sum())
    out = {"n_frauds": int(status.size), "n_with_status": n_with, "n_permutations": int(n_permutations),
           "by_bin_minutes": {}}
    if n_with == 0 or n_with == status.size:
        out["status"] = "not computable (the status is constant across frauds)"
        return out
    for minutes in bin_minutes:
        edges = block_edges(start_time, end_time_inclusive, int(minutes) * 60)
        bins = assign_blocks(fraud_times, edges)
        n_bins = len(edges) - 1
        observed, lag1 = _dispersion_and_lag1(np.bincount(bins[status], minlength=n_bins).astype(float))
        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            null[i] = _dispersion_and_lag1(
                np.bincount(bins[rng.permutation(status)], minlength=n_bins).astype(float))[0]
        out["by_bin_minutes"][str(int(minutes))] = {
            "n_bins": n_bins, "dispersion_index": observed,
            "null_95th_percentile": float(np.percentile(null, 95)),
            "permutation_p": float((1 + (null >= observed).sum()) / (1 + n_permutations)),
            "lag1_autocorrelation": lag1,
        }
    return out
