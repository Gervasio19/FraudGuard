"""Train-on-train, evaluate-on-validation experiment runner.

The runner is split-agnostic: it accepts any partition Series using the labels
in :data:`src.data.split.PARTITION_NAMES` (chronological or random), fits each
model on the ``train`` rows only, and evaluates on ``validation`` rows. Test
rows are never passed to any model; evaluating on the test partition is
refused, because the test set is reserved for final evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.config import DatasetConfig
from src.data.split import PARTITION_NAMES, TEST, TRAIN, VALIDATION, count_cross_partition_duplicate_groups
from src.evaluation.metrics import compute_binary_metrics
from src.features.preprocessing import validate_feature_frame
from src.models.baselines import ModelBuilder


def summarize_partitions(df: pd.DataFrame, partitions: pd.Series, config: DatasetConfig) -> dict:
    """Describe row and positive counts per partition plus the duplicate-leakage check.

    Counting labels is descriptive only; no model sees the test rows.
    """
    _check_partitions(df, partitions)
    is_positive = df[config.target_column].to_numpy() == config.positive_label
    labels = partitions.to_numpy()
    summary: dict = {"partitions": {}}
    for name in PARTITION_NAMES:
        mask = labels == name
        n_rows = int(mask.sum())
        n_positive = int(is_positive[mask].sum())
        summary["partitions"][name] = {
            "n_rows": n_rows,
            "n_positive": n_positive,
            "prevalence": n_positive / n_rows if n_rows else float("nan"),
        }
    group_columns = config.resolve_duplicate_group_columns(df)
    summary["cross_partition_duplicate_groups"] = (
        count_cross_partition_duplicate_groups(df, partitions, group_columns) if group_columns else None
    )
    return summary


@dataclass(frozen=True)
class ScoredModel:
    """A pipeline fitted on the training partition plus its evaluation-partition scores.

    ``y_true`` and ``y_score`` are aligned to the evaluation rows' original
    index and keep their original row order.
    """

    pipeline: Pipeline
    evaluation_rows: pd.DataFrame
    y_true: pd.Series
    y_score: pd.Series
    convergence_warnings: list[str]
    n_iter: int | None


def fit_and_score(
    df: pd.DataFrame,
    partitions: pd.Series,
    config: DatasetConfig,
    builder: ModelBuilder,
    random_state: int,
    evaluation_partition: str = VALIDATION,
) -> ScoredModel:
    """Fit one model on the training partition and score ``evaluation_partition`` once.

    Only training rows reach ``fit`` and only evaluation rows reach
    ``predict_proba``. Test rows are never used; ``evaluation_partition="test"``
    is refused because the test partition is reserved for final evaluation.
    """
    train, evaluation, feature_columns = _prepare(df, partitions, config, evaluation_partition)
    pipeline = builder(feature_columns, random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(train[feature_columns], train[config.target_column])
    positive_index = list(pipeline.classes_).index(config.positive_label)
    y_score = pipeline.predict_proba(evaluation[feature_columns])[:, positive_index]
    model = pipeline.named_steps["model"]
    return ScoredModel(
        pipeline=pipeline,
        evaluation_rows=evaluation,
        y_true=evaluation[config.target_column],
        y_score=pd.Series(y_score, index=evaluation.index, name="y_score"),
        convergence_warnings=[
            str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
        ],
        n_iter=int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else None,
    )


def fit_and_evaluate(
    df: pd.DataFrame,
    partitions: pd.Series,
    config: DatasetConfig,
    builders: Mapping[str, ModelBuilder],
    random_state: int,
    evaluation_partition: str = VALIDATION,
) -> dict[str, dict]:
    """Fit each model on the training partition and evaluate it on ``evaluation_partition``.

    Returns a mapping of model name to a result dict with ``metrics``,
    ``convergence_warnings`` and (for linear models) ``n_iter``.
    """
    _prepare(df, partitions, config, evaluation_partition)  # validate before fitting anything
    results: dict[str, dict] = {}
    for name, builder in builders.items():
        scored = fit_and_score(df, partitions, config, builder, random_state, evaluation_partition)
        feature_columns = config.resolve_feature_columns(df)
        y_pred = scored.pipeline.predict(scored.evaluation_rows[feature_columns])
        results[name] = {
            "metrics": compute_binary_metrics(
                scored.y_true, scored.y_score.to_numpy(), y_pred, config.positive_label
            ),
            "convergence_warnings": scored.convergence_warnings,
            "n_iter": scored.n_iter,
        }
    return results


def _prepare(
    df: pd.DataFrame, partitions: pd.Series, config: DatasetConfig, evaluation_partition: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate inputs and return (train rows, evaluation rows, feature columns)."""
    if evaluation_partition == TEST:
        raise ValueError("The test partition is reserved for final evaluation and cannot be used here.")
    if evaluation_partition not in PARTITION_NAMES or evaluation_partition == TRAIN:
        raise ValueError(f"Invalid evaluation partition {evaluation_partition!r}.")
    _check_partitions(df, partitions)

    feature_columns = config.resolve_feature_columns(df)
    labels = partitions.to_numpy()
    train = df.loc[labels == TRAIN]
    evaluation = df.loc[labels == evaluation_partition]
    if train.empty or evaluation.empty:
        raise ValueError("Training and evaluation partitions must both be non-empty.")
    validate_feature_frame(train[feature_columns], feature_columns)
    validate_feature_frame(evaluation[feature_columns], feature_columns)
    if config.positive_label not in set(train[config.target_column].unique()):
        raise ValueError("The training partition contains no positive examples.")
    return train, evaluation, feature_columns


def _check_partitions(df: pd.DataFrame, partitions: pd.Series) -> None:
    if not partitions.index.equals(df.index):
        raise ValueError("`partitions` must be aligned to `df.index`.")
    unknown = set(pd.unique(partitions.astype(object))) - set(PARTITION_NAMES)
    if unknown:
        raise ValueError(f"Unknown partition labels: {sorted(map(str, unknown))}.")
