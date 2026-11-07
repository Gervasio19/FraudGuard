"""Leakage-safe preprocessing for FraudGuard baseline models.

Preprocessing is expressed as an unfitted scikit-learn transformer that is
placed inside the model :class:`~sklearn.pipeline.Pipeline`. Calling
``pipeline.fit`` on the training partition fits the scaler on training rows
only; ``predict``/``predict_proba`` on validation rows reuse the fitted
statistics without refitting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def validate_feature_frame(X: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    """Raise if any feature column is non-numeric or contains missing/infinite values.

    The Phase 4b baselines implement no imputation or categorical encoding, so
    such data must be handled by an explicit, dataset-specific step instead of
    failing silently.
    """
    non_numeric = [
        c for c in feature_columns
        if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns are not supported: {non_numeric}.")
    values = X[list(feature_columns)].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Feature columns contain missing or infinite values.")


def build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    """Return an unfitted transformer that standardizes ``feature_columns``.

    Columns not listed are dropped, so the target, timestamps or identifiers
    can never leak into the model through the preprocessing step.
    """
    columns = list(feature_columns)
    if not columns:
        raise ValueError("At least one feature column is required.")
    return ColumnTransformer(
        [("standardize", StandardScaler(), columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
