"""Chronological train/validation/test partitioning for FraudGuard.

Implements the Phase 3 validation design documented in
``docs/validation_strategy.md``:

* Train:      ``Time <= 132928``
* Validation: ``132929 <= Time <= 151328``
* Test:       ``Time >= 151329``

Assignment depends only on the ``Time`` column, never on the ``Class`` label,
so it is deterministic and label-independent. All records, including
duplicates, are kept; no rows are dropped, reordered or modified. Because
assignment is a pure function of ``Time``, rows with equal timestamps always
share a partition, and duplicate feature vectors (which include ``Time``)
always share a partition too.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TIME_COLUMN = "Time"
LABEL_COLUMN = "Class"

#: Last ``Time`` value (inclusive) assigned to the training partition.
TRAIN_END_TIME = 132_928
#: Last ``Time`` value (inclusive) assigned to the validation partition.
VALIDATION_END_TIME = 151_328

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
PARTITION_NAMES: tuple[str, ...] = (TRAIN, VALIDATION, TEST)

PARTITION_DTYPE = pd.CategoricalDtype(categories=list(PARTITION_NAMES), ordered=True)


def _validate_time(df: pd.DataFrame, time_column: str) -> np.ndarray:
    """Return the time column as a float array after validating it."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if time_column not in df.columns:
        raise KeyError(f"Required column {time_column!r} is missing.")

    time = df[time_column]
    if isinstance(time, pd.DataFrame):
        raise ValueError(f"Column {time_column!r} appears more than once.")
    if pd.api.types.is_bool_dtype(time) or not pd.api.types.is_numeric_dtype(time):
        raise TypeError(f"Column {time_column!r} must be numeric, got dtype {time.dtype}.")

    values = time.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"Column {time_column!r} contains missing or infinite values.")
    if (values < 0).any():
        raise ValueError(
            f"Column {time_column!r} contains negative values; "
            "Time is seconds elapsed since the first transaction."
        )
    if (values != np.floor(values)).any():
        # The documented boundaries are inclusive integer-second ranges. A
        # fractional value (e.g. 132928.5) would fall between them, so it is
        # rejected rather than assigned by an undocumented rule.
        raise ValueError(
            f"Column {time_column!r} contains non-integer values; "
            "the documented boundaries are defined on whole seconds."
        )
    return values


def assign_partitions(df: pd.DataFrame, time_column: str = TIME_COLUMN) -> pd.Series:
    """Assign each row to ``train``, ``validation`` or ``test`` by its timestamp.

    Parameters
    ----------
    df:
        Transactions. Only ``time_column`` is read; the frame is not modified.
    time_column:
        Name of the column holding seconds since the first transaction.

    Returns
    -------
    pandas.Series
        Ordered categorical partition labels, aligned to ``df.index`` and in
        the same row order as ``df``.
    """
    values = _validate_time(df, time_column)
    codes = np.select(
        [values <= TRAIN_END_TIME, values <= VALIDATION_END_TIME],
        [0, 1],
        default=2,
    )
    labels = pd.Categorical.from_codes(codes, dtype=PARTITION_DTYPE)
    return pd.Series(labels, index=df.index, name="partition")


def split_partitions(
    df: pd.DataFrame, time_column: str = TIME_COLUMN
) -> dict[str, pd.DataFrame]:
    """Split ``df`` into train, validation and test frames.

    Rows keep their original index labels, values and relative order. The
    returned frames together contain every row of ``df`` exactly once.
    """
    partitions = assign_partitions(df, time_column).to_numpy()
    return {name: df.loc[partitions == name] for name in PARTITION_NAMES}


def count_cross_partition_duplicate_groups(
    df: pd.DataFrame,
    partitions: pd.Series,
    feature_columns: Iterable[str] | None = None,
) -> int:
    """Count identical-feature groups whose rows span more than one partition.

    A non-zero result means copies of the same record would appear in more than
    one partition (duplicate leakage). ``feature_columns`` defaults to every
    column except ``Class``, matching the Phase 3 duplicate-group definition.
    """
    if not partitions.index.equals(df.index):
        raise ValueError("`partitions` must be aligned to `df.index`.")
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != LABEL_COLUMN]
    feature_columns = list(feature_columns)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from df: {missing}.")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    group_ids = df.groupby(feature_columns, sort=False, dropna=False).ngroup().to_numpy()
    pairs = pd.DataFrame({"group": group_ids, "partition": partitions.cat.codes.to_numpy()})
    per_group = pairs.drop_duplicates().groupby("group").size()
    return int((per_group > 1).sum())
