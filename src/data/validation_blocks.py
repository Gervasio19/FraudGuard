"""Validation time blocks and the E2 held-out buffer (Phase 5B-0).

Implements the deterministic convention of protocol Addendum C, item C4, for
the repeated held-out block splits of protocol Section 7.5:

* **Blocks** are half-open time intervals ``[e_j, e_{j+1})`` in whole seconds,
  aligned to the first validation timestamp. The last block ends one second
  after the last validation timestamp, so it may be shorter than the others
  (400 s on the FraudGuard validation partition). It is an ordinary block.
* **Buffer**: for each selection-half block ``[a, b)``, the buffer zone is
  ``[a - buffer, a) U [b, b + buffer)``. A held-out row is *kept* iff it lies
  in a held-out block and in no buffer zone. Selection rows are never dropped,
  and no buffer is applied beyond the outer edges of the period.
* Consequence (integer seconds): every kept held-out row is at least
  ``buffer + 1`` seconds (301 s) from every second of every selection block.

Every function depends only on timestamps and block indices. None reads a
label or a score, fits a model or computes a metric.
"""

from __future__ import annotations

import numpy as np

from src.models.phase5_settings import E2_BLOCK_SECONDS, E2_BUFFER_SECONDS


def _as_integer_times(times) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError("Timestamps contain missing or infinite values.")
    if (values != np.floor(values)).any():
        raise ValueError("Timestamps must be whole seconds.")
    return values.astype(np.int64)


def block_edges(start_time: int, end_time_inclusive: int, block_seconds: int = E2_BLOCK_SECONDS) -> np.ndarray:
    """Return block edges ``[e_0, ..., e_n]``; block ``j`` is ``[e_j, e_{j+1})``.

    ``e_0 = start_time`` and ``e_n = end_time_inclusive + 1``. Interior edges are
    spaced ``block_seconds`` apart; the final block holds the remainder.
    """
    for name, value in (("start_time", start_time), ("end_time_inclusive", end_time_inclusive),
                        ("block_seconds", block_seconds)):
        if int(value) != value:
            raise ValueError(f"{name} must be a whole number of seconds.")
    start, stop, width = int(start_time), int(end_time_inclusive) + 1, int(block_seconds)
    if width <= 0:
        raise ValueError("block_seconds must be positive.")
    if stop <= start:
        raise ValueError("The period must contain at least one second.")
    interior = np.arange(start + width, stop, width, dtype=np.int64)
    return np.concatenate([[start], interior, [stop]]).astype(np.int64)


def assign_blocks(times, edges: np.ndarray) -> np.ndarray:
    """Return the block index of each timestamp. Raises if any lies outside the edges."""
    values = _as_integer_times(times)
    edges = np.asarray(edges, dtype=np.int64)
    if values.size and (values.min() < edges[0] or values.max() >= edges[-1]):
        raise ValueError("Timestamps fall outside the block edges.")
    return np.searchsorted(edges, values, side="right") - 1


def heldout_keep_mask(times, edges: np.ndarray, selection_blocks,
                      buffer_seconds: int = E2_BUFFER_SECONDS) -> np.ndarray:
    """Boolean mask of rows that belong to the buffered held-out half.

    A row is kept iff its block is not a selection block and its timestamp lies
    in no buffer zone ``[a - buffer, a) U [b, b + buffer)`` of any selection
    block ``[a, b)``.
    """
    if int(buffer_seconds) != buffer_seconds or buffer_seconds < 0:
        raise ValueError("buffer_seconds must be a non-negative whole number.")
    edges = np.asarray(edges, dtype=np.int64)
    n_blocks = len(edges) - 1
    selection = np.unique(np.asarray(selection_blocks, dtype=np.int64))
    if selection.size and (selection.min() < 0 or selection.max() >= n_blocks):
        raise ValueError("Selection block index out of range.")
    values = _as_integer_times(times)
    blocks = assign_blocks(values, edges)
    keep = ~np.isin(blocks, selection)
    buffer = int(buffer_seconds)
    for j in selection:
        a, b = edges[j], edges[j + 1]
        in_zone = ((values >= a - buffer) & (values < a)) | ((values >= b) & (values < b + buffer))
        keep &= ~in_zone
    return keep


def draw_selection_blocks(n_splits: int, n_blocks: int, n_selection: int, seed: int) -> np.ndarray:
    """Draw the selection-half block indices of each split.

    One ``numpy.random.default_rng(seed)`` generator is created and used
    sequentially; split ``i`` is ``sorted(rng.choice(n_blocks, n_selection,
    replace=False))``. Repeated splits are kept (not de-duplicated), so every
    one of ``n_splits`` counts as attempted. Returns shape ``(n_splits, n_selection)``.
    """
    if not (0 < n_selection < n_blocks):
        raise ValueError("n_selection must be between 1 and n_blocks - 1.")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_splits, n_selection), dtype=np.int64)
    for i in range(n_splits):
        draws[i] = np.sort(rng.choice(n_blocks, size=n_selection, replace=False))
    return draws
