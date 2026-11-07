"""Duplicate-group-aware stratified random split (secondary benchmark).

Implements the secondary benchmark described in ``docs/validation_strategy.md``:
a seeded random train/validation/test split that preserves class proportions
and never places copies of an identical record in different partitions.

Procedure
---------
1. Rows are grouped by identical values of ``group_columns`` (by default every
   column except the target, matching the Phase 3 duplicate definition). With
   no group columns, every row is its own group.
2. Every group must have a single label; otherwise a ``ValueError`` is raised.
3. Within each class separately, groups (ordered by their sorted feature
   values, so the result does not depend on input row order) are shuffled with
   ``numpy.random.default_rng(random_state)`` and allocated in that order by
   cumulative row count: a group goes to train while the running row total is
   at most ``round(fractions[0] * n_class_rows)``, then to validation while it
   is at most ``round((fractions[0] + fractions[1]) * n_class_rows)``, then to
   test.

The output uses the same ordered categorical labels as the chronological split
(:data:`src.data.split.PARTITION_DTYPE`), so both splits plug into the same
training and evaluation code. This split does not simulate future scoring and
must only be reported as a secondary benchmark.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.split import PARTITION_DTYPE

DEFAULT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_RANDOM_STATE = 42


def _validate_fractions(fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,):
        raise ValueError("`fractions` must contain exactly three values (train, validation, test).")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("`fractions` must be finite and positive.")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError(f"`fractions` must sum to 1, got {values.sum()}.")
    return values


def assign_stratified_group_partitions(
    df: pd.DataFrame,
    target_column: str,
    group_columns: Sequence[str] | None = None,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.Series:
    """Assign rows to train/validation/test, stratified by label, keeping groups whole.

    Parameters
    ----------
    df:
        Records to split. The frame is not modified.
    target_column:
        Label column used for stratification.
    group_columns:
        Columns defining identical-record groups. ``None`` uses every column
        except ``target_column``; an empty sequence treats each row separately.
    fractions:
        Target (train, validation, test) row fractions within each class.
    random_state:
        Seed for the shuffle; the same inputs and seed always give the same split.

    Returns
    -------
    pandas.Series
        Ordered categorical partition labels aligned to ``df.index``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} is missing.")
    if df[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")
    fraction_values = _validate_fractions(fractions)

    if group_columns is None:
        group_columns = [c for c in df.columns if c != target_column]
    group_columns = list(group_columns)
    if target_column in group_columns:
        raise ValueError("The target column cannot define duplicate groups.")
    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Group columns missing from df: {missing}.")

    if group_columns:
        # sort=True numbers groups by their values, independent of row order.
        group_ids = df.groupby(group_columns, sort=True, dropna=False).ngroup().to_numpy()
    else:
        group_ids = np.arange(len(df))

    labels = df[target_column].to_numpy()
    groups = pd.DataFrame({"group": group_ids, "label": labels})
    label_counts = groups.groupby("group")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError(
            f"{int((label_counts > 1).sum())} duplicate groups contain conflicting labels; "
            "they cannot be stratified as a single unit."
        )
    group_table = groups.groupby("group").agg(label=("label", "first"), size=("label", "size"))

    rng = np.random.default_rng(random_state)
    group_codes = pd.Series(2, index=group_table.index, dtype=np.int64)
    for label in sorted(group_table["label"].unique()):
        class_groups = group_table[group_table["label"] == label]
        order = rng.permutation(len(class_groups))
        sizes = class_groups["size"].to_numpy()[order]
        ids = class_groups.index.to_numpy()[order]
        cumulative = np.cumsum(sizes)
        n_rows = cumulative[-1]
        train_end = round(fraction_values[0] * n_rows)
        validation_end = round((fraction_values[0] + fraction_values[1]) * n_rows)
        codes = np.select([cumulative <= train_end, cumulative <= validation_end], [0, 1], default=2)
        group_codes.loc[ids] = codes

    row_codes = group_codes.loc[group_ids].to_numpy()
    partitions = pd.Categorical.from_codes(row_codes, dtype=PARTITION_DTYPE)
    return pd.Series(partitions, index=df.index, name="partition")
