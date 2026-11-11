"""Protocol Section 4.2 step 2: the frozen 28-configuration grid (Phase 5B-2).

Every configuration is built from ``pre_registered_configuration()["models"]``
only (protocol Sections 5.2 and 5.3); nothing here adds or changes a value.

* 12 LR = ``C`` x ``class_weight``; StandardScaler + LogisticRegression (lbfgs,
  ``max_iter`` 1,000, refit once at 5,000 on a convergence warning,
  ``random_state`` 42). The L2 penalty is scikit-learn's default and is not
  passed (``penalty=`` is deprecated in scikit-learn >= 1.8).
* 16 HGB = ``learning_rate`` x ``max_leaf_nodes`` x ``min_samples_leaf`` x
  ``class_weight``; no scaler; ``max_iter`` 300, ``early_stopping=False``,
  ``l2_regularization`` 1.0, ``max_bins`` 255, ``random_state`` 42.
* Features: the frozen 29 model features (``V1``-``V28`` + ``Amount``); never
  ``Time``. The exploratory ``log1p(Amount)`` experiments (E1, E3) are not part
  of the grid.

Each configuration is fitted **once** on ``T_fit`` and scores ``T_cal`` and the
validation rows **once** each (``decision_function``, track B; and
``predict_proba`` of the same fit, track C). The scores are returned as an
in-memory cache for the later stages (protocol Section 12.2: scores are cached
in memory and recomputed deterministically from code; the storage policy forbids
per-row score vectors in result files).

:func:`select_within_family` implements the pre-registered step-3 rule
(Section 5.4) so that the selected candidates can be recorded; it reads
validation AP only.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import DatasetConfig
from src.data.split import TRAIN, VALIDATION, VALIDATION_END_TIME
from src.features.preprocessing import build_preprocessor
from src.models.experiment import fit_and_score
from src.models.phase5_settings import (
    MODEL_RANDOM_STATE, T_CAL_END_TIME, T_CAL_START_TIME, T_FIT_END_TIME, VALIDATION_START_TIME,
    pre_registered_configuration,
)
from src.models.ranking_score import ranking_scores

LR = "logistic_regression"
HGB = "hist_gradient_boosting"
FAMILIES = (LR, HGB)
_LR_FIXED_PASSED = ("solver",)  # max_iter and random_state are passed separately; penalty is the default
_HGB_FIXED_PASSED = ("max_iter", "early_stopping", "l2_regularization", "max_bins")


def model_features() -> tuple[str, ...]:
    return tuple(pre_registered_configuration()["features"]["model_features"])


def grid_configurations() -> list[dict]:
    """The 28 frozen configurations, in grid order (Cartesian product in the frozen key order)."""
    models = pre_registered_configuration()["models"]
    configs = []
    for family, prefix in ((LR, "LR"), (HGB, "HGB")):
        spec = models[family]
        keys = list(spec["grid"])
        for i, values in enumerate(itertools.product(*(spec["grid"][k] for k in keys)), start=1):
            configs.append({"config_id": f"{prefix}-{i:02d}", "family": family,
                            "params": dict(zip(keys, values))})
        if sum(c["family"] == family for c in configs) != spec["n_configurations"]:
            raise AssertionError(f"{family}: grid size differs from the frozen n_configurations.")
    return configs


def build_pipeline(family: str, params: dict, feature_columns: Sequence[str], random_state: int,
                   max_iter: int | None = None) -> Pipeline:
    """Unfitted pipeline for one grid configuration (fixed values from the frozen settings)."""
    fixed = pre_registered_configuration()["models"][family]["fixed"]
    columns = list(feature_columns)
    if family == LR:
        if fixed["penalty"] != "l2":
            raise ValueError("Only the frozen L2 penalty (scikit-learn default) is supported.")
        model = LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                   max_iter=fixed["max_iter"] if max_iter is None else max_iter,
                                   random_state=random_state, **{k: fixed[k] for k in _LR_FIXED_PASSED})
        return Pipeline([("preprocess", build_preprocessor(columns)), ("model", model)])
    if family == HGB:
        model = HistGradientBoostingClassifier(random_state=random_state, **params,
                                               **{k: fixed[k] for k in _HGB_FIXED_PASSED})
        return Pipeline([("preprocess", ColumnTransformer([("select", "passthrough", columns)],
                                                          remainder="drop", verbose_feature_names_out=False)),
                         ("model", model)])
    raise ValueError(f"Unknown family {family!r}.")


def _builder(family: str, params: dict, max_iter: int | None = None):
    def builder(feature_columns, random_state):
        return build_pipeline(family, params, feature_columns, random_state, max_iter)
    return builder


def check_row_roles(fit_rows: pd.DataFrame, cal_rows: pd.DataFrame, validation_rows: pd.DataFrame,
                    time_column: str) -> None:
    """Refuse any row outside its role: T_fit, T_cal, validation (test is never accepted)."""
    for name, rows in (("fit_rows", fit_rows), ("cal_rows", cal_rows), ("validation_rows", validation_rows)):
        if rows.empty:
            raise ValueError(f"{name} must be non-empty.")
    if (fit_rows[time_column] > T_FIT_END_TIME).any():
        raise ValueError(f"fit_rows must be T_fit rows (Time <= {T_FIT_END_TIME}).")
    ct = cal_rows[time_column]
    if (ct < T_CAL_START_TIME).any() or (ct > T_CAL_END_TIME).any():
        raise ValueError("cal_rows must be T_cal rows.")
    vt = validation_rows[time_column]
    if (vt < VALIDATION_START_TIME).any() or (vt > VALIDATION_END_TIME).any():
        raise ValueError("validation_rows must lie in the validation period; test rows are refused.")


def fit_configuration(configuration: dict, fit_rows: pd.DataFrame, cal_rows: pd.DataFrame,
                      validation_rows: pd.DataFrame, config: DatasetConfig) -> dict:
    """Fit one configuration on T_fit; score T_cal and validation once each (both representations)."""
    check_row_roles(fit_rows, cal_rows, validation_rows, config.time_column)
    features = model_features()
    if config.time_column in features:
        raise AssertionError("Time must never be a model feature.")
    family, params = configuration["family"], configuration["params"]
    fixed = pre_registered_configuration()["models"][family]["fixed"]
    exp_config = _with_features(config, features)
    df = pd.concat([fit_rows, validation_rows])  # T_cal is never passed to the runner
    if df.index.has_duplicates:
        raise ValueError("fit_rows and validation_rows must not share index labels.")
    partitions = pd.Series(pd.Categorical([TRAIN] * len(fit_rows) + [VALIDATION] * len(validation_rows),
                                          categories=[TRAIN, VALIDATION, "test"], ordered=True),
                           index=df.index, name="partition")

    start = time.perf_counter()
    max_iter = fixed["max_iter"] if family == LR else None
    scored = fit_and_score(df, partitions, exp_config, _builder(family, params, max_iter), MODEL_RANDOM_STATE)
    first_warnings = scored.convergence_warnings
    refit = False
    if family == LR and first_warnings:  # pre-registered: refit once with max_iter 5000
        refit, max_iter = True, fixed["max_iter_on_convergence_warning"]
        scored = fit_and_score(df, partitions, exp_config, _builder(family, params, max_iter), MODEL_RANDOM_STATE)
    fit_seconds = time.perf_counter() - start

    pipeline = scored.pipeline
    X_val, X_cal = scored.evaluation_rows[list(features)], cal_rows[list(features)]
    start = time.perf_counter()
    val_decision = ranking_scores(pipeline, X_val, config.positive_label)
    val_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cal_decision = ranking_scores(pipeline, X_cal, config.positive_label)
    positive_index = list(pipeline.classes_).index(config.positive_label)
    cal_proba = pipeline.predict_proba(X_cal)[:, positive_index].astype(np.float64)
    cal_seconds = time.perf_counter() - start
    model = pipeline.named_steps["model"]

    effective = {**params, **({"solver": fixed["solver"], "penalty": "l2 (scikit-learn default)",
                               "max_iter": max_iter} if family == LR else
                              {k: fixed[k] for k in _HGB_FIXED_PASSED}), "random_state": MODEL_RANDOM_STATE}
    return {
        **configuration,
        "pipeline": pipeline,
        "validation_index": scored.evaluation_rows.index.to_numpy(),
        "t_cal_index": cal_rows.index.to_numpy(),
        "scores": {
            "validation": {"decision_function": val_decision,
                           "predict_proba": scored.y_score.to_numpy(dtype=np.float64)},
            "t_cal": {"decision_function": cal_decision, "predict_proba": cal_proba},
        },
        "fit_info": {
            "effective_params": effective,
            "n_fit_rows": int(len(fit_rows)),
            "n_fit_positive": int((fit_rows[config.target_column] == config.positive_label).sum()),
            "n_iter": scored.n_iter if family == LR else int(model.n_iter_),
            "convergence_warnings_first_fit": first_warnings,
            "convergence_warnings_final_fit": scored.convergence_warnings,
            "refit_with_max_iter_5000": refit if family == LR else None,
        },
        "runtime_seconds": {"fit_and_validation_predict_proba": fit_seconds,
                            "validation_decision_function": val_seconds,
                            "t_cal_scoring": cal_seconds},
    }


def _with_features(config: DatasetConfig, features: Sequence[str]) -> DatasetConfig:
    return dataclasses.replace(config, feature_columns=tuple(features))


def simplicity_key(configuration: dict) -> tuple:
    """Section 5.4 tie-break order (smaller = simpler, preferred)."""
    p = configuration["params"]
    weight = 0 if p["class_weight"] is None else 1
    if configuration["family"] == LR:
        return (p["C"], weight)
    return (p["max_leaf_nodes"], -p["min_samples_leaf"], p["learning_rate"], weight)


def select_within_family(rows: list[dict], family: str, tolerance: float | None = None) -> dict:
    """Pre-registered step-3 rule (Section 5.4): max validation AP; ties within 1e-12 go to the simpler model.

    ``rows`` hold ``config_id``, ``family``, ``params`` and ``average_precision``
    (track B, validation). Returns the selected ID with the tie group and spread.
    """
    if tolerance is None:
        tolerance = pre_registered_configuration()["models"]["selection"]["tie_tolerance"]
    candidates = [r for r in rows if r["family"] == family]
    if not candidates:
        raise ValueError(f"No configuration of family {family!r}.")
    aps = np.array([r["average_precision"] for r in candidates], dtype=float)
    if not np.isfinite(aps).all():
        raise ValueError("Every configuration needs a finite validation AP.")
    best = float(aps.max())
    tied = [r for r in candidates if r["average_precision"] >= best - tolerance]
    selected = min(tied, key=simplicity_key)
    return {
        "selected_config_id": selected["config_id"],
        "selected_params": selected["params"],
        "selected_average_precision": selected["average_precision"],
        "max_average_precision": best,
        "median_average_precision": float(np.median(aps)),
        "spread_max_minus_median": best - float(np.median(aps)),
        "n_within_tie_tolerance": len(tied),
        "tie_group": [r["config_id"] for r in tied],
        "rule": ("max validation AP on track B (decision_function); ties within 1e-12 -> simpler model: "
                 + ("smaller C, then class_weight None before balanced" if family == LR else
                    "smaller max_leaf_nodes, larger min_samples_leaf, lower learning_rate, then class_weight "
                    "None before balanced")),
    }
