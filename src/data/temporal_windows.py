"""Split a chronological partition into two consecutive time windows (Phase 4c).

Algorithm (``docs/TECHNICAL_DOCUMENTATION.md``):

1. Take the unique timestamps in ascending order and count rows per timestamp.
2. Consider every boundary between consecutive unique timestamps; window 1 is
   all rows with ``time <= t_k`` and window 2 all rows with ``time > t_k``.
3. Choose the boundary minimizing ``|n_window_1 - n_window_2|``; on ties,
   choose the earlier boundary (smaller window 1).

Assignment depends only on the timestamp, so rows sharing a timestamp always
share a window. No row is dropped, reordered or modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_1 = "window_1"
WINDOW_2 = "window_2"
WINDOW_DTYPE = pd.CategoricalDtype(categories=[WINDOW_1, WINDOW_2], ordered=True)


@dataclass(frozen=True)
class TimeWindows:
    """Window labels aligned to the input index, plus the chosen boundary."""

    labels: pd.Series
    window_1_end_time: float  # last timestamp in window 1 (inclusive)
    window_2_start_time: float  # first timestamp in window 2 (inclusive)
    n_window_1: int
    n_window_2: int


def split_two_time_windows(times: pd.Series) -> TimeWindows:
    """Assign each row to ``window_1`` or ``window_2`` by its timestamp.

    Raises if ``times`` is not a numeric Series, contains missing or infinite
    values, or has fewer than two distinct timestamps.
    """
    if not isinstance(times, pd.Series):
        raise TypeError(f"Expected a pandas Series, got {type(times).__name__}.")
    if pd.api.types.is_bool_dtype(times) or not pd.api.types.is_numeric_dtype(times):
        raise TypeError(f"Timestamps must be numeric, got dtype {times.dtype}.")
    values = times.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Timestamps contain missing or infinite values.")

    unique_times, counts = np.unique(values, return_counts=True)
    if len(unique_times) < 2:
        raise ValueError("At least two distinct timestamps are required to form two windows.")

    n_rows = len(values)
    window_1_sizes = np.cumsum(counts)[:-1]  # boundary after each unique time except the last
    imbalance = np.abs(2 * window_1_sizes - n_rows)
    k = int(np.argmin(imbalance))  # argmin returns the first, i.e. earliest, boundary on ties
    end_time = unique_times[k]

    codes = np.where(values <= end_time, 0, 1)
    labels = pd.Series(
        pd.Categorical.from_codes(codes, dtype=WINDOW_DTYPE), index=times.index, name="window"
    )
    return TimeWindows(
        labels=labels,
        window_1_end_time=float(end_time),
        window_2_start_time=float(unique_times[k + 1]),
        n_window_1=int(window_1_sizes[k]),
        n_window_2=int(n_rows - window_1_sizes[k]),
    )
