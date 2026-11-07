"""Dataset-specific configuration for FraudGuard modeling code.

Modeling, preprocessing and evaluation code reads column roles from a
:class:`DatasetConfig` instead of hardcoding names such as ``Class`` or
``Time``. Supporting a new dataset means adding a new config instance (and,
if it needs a chronological split, its own documented boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    """Column roles and location for one tabular binary-classification dataset.

    Attributes
    ----------
    name:
        Short identifier used in experiment outputs.
    path:
        Location of the raw CSV file.
    target_column:
        Binary label column.
    positive_label:
        Value of ``target_column`` that denotes the positive (fraud) class.
    feature_columns:
        Explicit model features. ``None`` means every column except the target
        and ``excluded_feature_columns``.
    excluded_feature_columns:
        Columns never used as model features (they may still be used for
        splitting or grouping).
    time_column:
        Optional timestamp column used for chronological splitting.
    duplicate_group_columns:
        Columns that define identical-record groups, which must not cross
        partitions. ``None`` means every column except the target (the Phase 3
        definition); an empty tuple disables duplicate grouping.
    """

    name: str
    path: Path
    target_column: str
    positive_label: int = 1
    feature_columns: tuple[str, ...] | None = None
    excluded_feature_columns: tuple[str, ...] = ()
    time_column: str | None = None
    duplicate_group_columns: tuple[str, ...] | None = None

    def resolve_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the model feature columns for ``df`` in a stable order."""
        if self.feature_columns is not None:
            columns = list(self.feature_columns)
            if self.target_column in columns:
                raise ValueError("The target column cannot be a model feature.")
            overlap = sorted(set(columns) & set(self.excluded_feature_columns))
            if overlap:
                raise ValueError(f"Columns are both selected and excluded: {overlap}.")
        else:
            excluded = {self.target_column, *self.excluded_feature_columns}
            columns = [c for c in df.columns if c not in excluded]
        _require_columns(df, columns, "Feature")
        if not columns:
            raise ValueError("No feature columns remain after exclusions.")
        return columns

    def resolve_duplicate_group_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the columns that identify duplicate-record groups (may be empty)."""
        if self.duplicate_group_columns is None:
            return [c for c in df.columns if c != self.target_column]
        columns = list(self.duplicate_group_columns)
        _require_columns(df, columns, "Duplicate-group")
        return columns


def _require_columns(df: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{role} columns missing from data: {missing}.")


#: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud).
#: ``Time`` is used for the chronological split and duplicate grouping but is
#: excluded from model features: it counts seconds since the first transaction
#: in this 48-hour extract, so chronological validation values lie entirely
#: outside the training range (see docs/PROJECT_CONTEXT.md, Phase 4b).
CREDITCARD = DatasetConfig(
    name="creditcard",
    path=PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
    target_column="Class",
    positive_label=1,
    excluded_feature_columns=("Time",),
    time_column="Time",
)
