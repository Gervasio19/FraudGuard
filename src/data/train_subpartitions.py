"""Chronological ``T_fit`` / ``T_cal`` sub-partitions of the training partition (Phase 5B-1).

Implements protocol Section 3.1 and Section 6.2 (option C-A), with the frozen
boundaries of ``src/models/phase5_settings.py``:

* ``T_fit``: ``Time <= T_FIT_END_TIME`` (112,553): earlier observations; fits
  preprocessing and models.
* ``T_cal``: ``T_CAL_START_TIME <= Time <= TRAIN_END_TIME`` (112,555-132,928):
  later observations; fits the calibrator only.

Assignment is a pure function of ``Time``. It never reads a label, never
shuffles, and never reorders, drops or modifies a row, so rows that share a
timestamp always share a sub-partition. Rows outside the training partition
(validation or test) are refused, which keeps the test partition out of this
helper by construction. A timestamp strictly between the two boundaries would
mean a different dataset and is refused rather than assigned by an
undocumented rule.

The label-dependent part (:func:`summarize_subpartitions`) only *counts*
training labels, which the protocol permits (Section 6.2, note on size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split import TRAIN_END_TIME
from src.models.phase5_settings import T_CAL_START_TIME, T_FIT_END_TIME

T_FIT = "t_fit"
T_CAL = "t_cal"
SUBPARTITION_NAMES: tuple[str, ...] = (T_FIT, T_CAL)
SUBPARTITION_DTYPE = pd.CategoricalDtype(categories=list(SUBPARTITION_NAMES), ordered=True)


def _validated_times(times) -> np.ndarray:
    if isinstance(times, pd.Series):
        if pd.api.types.is_bool_dtype(times) or not pd.api.types.is_numeric_dtype(times):
            raise TypeError(f"Timestamps must be numeric, got dtype {times.dtype}.")
        values = times.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError("Timestamps contain missing or infinite values.")
    if (values < 0).any():
        raise ValueError("Timestamps must be non-negative.")
    if (values != np.floor(values)).any():
        raise ValueError("Timestamps must be whole seconds.")
    if (values > TRAIN_END_TIME).any():
        raise ValueError(
            f"Rows after Time {TRAIN_END_TIME} are not training rows; T_fit/T_cal are "
            "sub-partitions of train only (validation and test rows are refused)."
        )
    in_gap = (values > T_FIT_END_TIME) & (values < T_CAL_START_TIME)
    if in_gap.any():
        raise ValueError(
            f"Timestamps strictly between {T_FIT_END_TIME} and {T_CAL_START_TIME} are not assigned "
            "by the frozen boundary; this is not the dataset the boundary was defined on."
        )
    return values


def assign_train_subpartitions(times) -> pd.Series:
    """Label each training row ``t_fit`` or ``t_cal`` by its timestamp.

    ``times`` is a Series (index kept) or a 1-D array of training timestamps.
    Returns an ordered categorical Series in the input order.
    """
    values = _validated_times(times)
    codes = np.where(values <= T_FIT_END_TIME, 0, 1)
    index = times.index if isinstance(times, pd.Series) else None
    return pd.Series(pd.Categorical.from_codes(codes, dtype=SUBPARTITION_DTYPE), index=index,
                     name="train_subpartition")


def check_temporal_integrity(times, subpartitions: pd.Series) -> dict:
    """Verify strict time order and that no timestamp spans both sub-partitions.

    Label-free. Raises ``ValueError`` on any violation and returns a JSON-ready
    record of what was checked.
    """
    values = _validated_times(times)
    labels = np.asarray(subpartitions.astype(object))
    if len(labels) != len(values):
        raise ValueError("times and subpartitions must have the same length.")
    fit_times, cal_times = values[labels == T_FIT], values[labels == T_CAL]
    if fit_times.size == 0 or cal_times.size == 0:
        raise ValueError("Both T_fit and T_cal must be non-empty.")
    if not fit_times.max() < cal_times.min():
        raise ValueError("T_fit must lie strictly before T_cal.")
    shared = np.intersect1d(fit_times, cal_times)
    if shared.size:
        raise ValueError(f"{shared.size} timestamp(s) span T_fit and T_cal.")
    return {
        "t_fit_time_range": [int(fit_times.min()), int(fit_times.max())],
        "t_cal_time_range": [int(cal_times.min()), int(cal_times.max())],
        "gap_seconds_between_last_t_fit_and_first_t_cal": int(cal_times.min() - fit_times.max()),
        "max_t_fit_time_lt_min_t_cal_time": True,
        "n_timestamps_spanning_both": 0,
    }


def summarize_subpartitions(subpartitions: pd.Series, y_true, positive_label=1) -> dict:
    """Row, fraud and prevalence counts per sub-partition, with the estimability check.

    *Estimable* here means only what fitting requires: each sub-partition holds
    at least one positive and one negative row (a classifier cannot be fitted
    on ``T_fit`` otherwise, and a sigmoid calibrator cannot be fitted on
    ``T_cal``). The protocol pre-registers no minimum fraud count for
    ``T_fit`` or ``T_cal``, so none is applied; the counts are reported for
    comparison with the protocol's [V] values (Section 3.1).
    """
    labels = np.asarray(subpartitions.astype(object))
    positive = np.asarray(y_true) == positive_label
    if len(labels) != len(positive):
        raise ValueError("subpartitions and y_true must have the same length.")
    summary: dict = {}
    for name in SUBPARTITION_NAMES:
        mask = labels == name
        n_rows, n_pos = int(mask.sum()), int(positive[mask].sum())
        summary[name] = {
            "n_rows": n_rows,
            "n_positive": n_pos,
            "n_negative": n_rows - n_pos,
            "prevalence": n_pos / n_rows if n_rows else None,
            "estimable": bool(n_pos > 0 and n_rows - n_pos > 0),
        }
    summary["all_estimable"] = all(summary[n]["estimable"] for n in SUBPARTITION_NAMES)
    return summary


def require_estimable(summary: dict) -> None:
    """Raise if either sub-partition lacks a class (see :func:`summarize_subpartitions`)."""
    failing = [n for n in SUBPARTITION_NAMES if not summary[n]["estimable"]]
    if failing:
        raise ValueError(f"Sub-partition(s) without both classes: {failing}.")
