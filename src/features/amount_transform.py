"""Stateless ``log1p(Amount)`` feature for the Phase 5B-2 experiments E1 and E3.

``log1p(Amount) = log(1 + Amount)`` is a fixed, row-wise transformation: it has
no fitted parameters, so computing it on any partition cannot carry
information between rows or partitions. Any *fitted* step (the StandardScaler
of the LR pipelines) still runs inside the model pipeline, which is fitted on
``T_fit`` rows only.

The original ``Amount`` column is kept unchanged and the new column is
*appended* ("existing features + log1p(Amount)"). The input frame is never
modified; the raw CSV is never written.

Safety: ``log1p`` is defined for values > -1. Amounts are monetary values and
must be finite and non-negative (the dataset has 1,825 zeros and no negatives,
Phase 2 report); anything else is refused rather than silently mapped to NaN.
``log1p(0) = 0``, so zero amounts are handled exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AMOUNT_COLUMN = "Amount"
LOG1P_AMOUNT_COLUMN = "log1p_Amount"


def log1p_amount(values) -> np.ndarray:
    """Return ``log1p`` of finite, non-negative amounts as float64. Raises otherwise."""
    amounts = np.asarray(values, dtype=np.float64)
    if not np.isfinite(amounts).all():
        raise ValueError("Amount contains missing or infinite values.")
    if (amounts < 0).any():
        raise ValueError("Amount contains negative values; log1p(Amount) is defined here for Amount >= 0 only.")
    return np.log1p(amounts)


def add_log1p_amount(df: pd.DataFrame, amount_column: str = AMOUNT_COLUMN,
                     new_column: str = LOG1P_AMOUNT_COLUMN) -> pd.DataFrame:
    """Return a copy of ``df`` with ``new_column = log1p(amount_column)`` appended.

    Index, row order and every existing column are preserved; ``df`` itself is
    not modified. Refuses a frame that already holds ``new_column``.
    """
    if amount_column not in df.columns:
        raise KeyError(f"Column {amount_column!r} is missing.")
    if new_column in df.columns:
        raise ValueError(f"Column {new_column!r} already exists.")
    if pd.api.types.is_bool_dtype(df[amount_column]) or not pd.api.types.is_numeric_dtype(df[amount_column]):
        raise TypeError(f"Column {amount_column!r} must be numeric.")
    out = df.copy()
    out[new_column] = log1p_amount(df[amount_column].to_numpy(dtype=float, na_value=np.nan))
    return out
