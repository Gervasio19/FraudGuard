"""Phase 5B-2 controlled model experiments E0-E3 (owner-scoped; see the report).

| ID | Family | Features |
|----|--------|----------|
| E0 | Logistic Regression (Phase 4 configuration) | ``V1``-``V28`` + ``Amount`` |
| E1 | Logistic Regression (Phase 4 configuration) | E0 features + ``log1p(Amount)`` |
| E2 | HistGradientBoosting (one pre-registered grid point) | ``V1``-``V28`` + ``Amount`` |
| E3 | HistGradientBoosting (same configuration as E2) | E2 features + ``log1p(Amount)`` |

Naming: the experiment IDs E1/E2 are unrelated to the protocol's champion
criteria E1/E2 (Section 4.3). Documents say "experiment E2" vs "criterion E2".

Design (fixed before any Phase 5B-2 score existed):

* Every experiment is fitted **once on ``T_fit``** through the Phase 4 runner
  :func:`src.models.experiment.fit_and_score` (which refuses the test
  partition) and scores the **same 42,722 validation rows** once. Each pair
  E0/E1 and E2/E3 differs only in the appended ``log1p(Amount)`` column.
* LR: the Phase 4 configuration (C = 1, balanced, lbfgs, ``max_iter`` 1,000,
  refit at 5,000 on a convergence warning), StandardScaler fitted on ``T_fit``
  inside the pipeline. Identical to the Phase 5B-1 primary track.
* HGB: one grid point of the frozen protocol grid (Section 5.3), chosen a
  priori and not tuned: scikit-learn's defaults for the tuned parameters
  (``learning_rate`` 0.1, ``max_leaf_nodes`` 31, ``min_samples_leaf`` 20),
  ``class_weight="balanced"`` (the incumbent's imbalance convention) and the
  protocol's fixed values (``max_iter`` 300, ``early_stopping=False``,
  ``l2_regularization`` 1.0, ``max_bins`` 255, ``random_state`` 42). No scaler.
* Score tracks: **A** historical (E0 only; quoted from Phase 4b, full train,
  ``predict_proba``), **B** primary (``T_fit``, ``decision_function``),
  **C** diagnostic (same fit as B, ``predict_proba``). Every metric block names
  its ``score_representation`` and :func:`validate_experiment_tracks` enforces it.

Nothing here selects a model, a threshold or a hyperparameter.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import DatasetConfig
from src.data.split import TRAIN, VALIDATION, VALIDATION_END_TIME
from src.data.validation_blocks import block_edges, draw_selection_blocks, heldout_keep_mask
from src.evaluation.capacity import fixed_k_table
from src.features.amount_transform import LOG1P_AMOUNT_COLUMN, add_log1p_amount
from src.models.experiment import fit_and_score
from src.models.phase4_continuity import (
    CONTINUITY_LR_PARAMS, MAX_ITER, MAX_ITER_ON_WARNING, build_continuity_lr, score_diagnostics,
)
from src.models.phase5_settings import (
    E2_BLOCK_SECONDS, E2_BUFFER_SECONDS, E2_MIN_HELDOUT_FRAUDS, E2_N_BLOCKS, E2_N_SELECTION_BLOCKS,
    E2_N_SPLITS, E2_SPLIT_SEED, MODEL_RANDOM_STATE, T_FIT_END_TIME, VALIDATION_START_TIME,
    pre_registered_configuration,
)
from src.models.ranking_score import ranking_scores

LOGISTIC_REGRESSION = "logistic_regression"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"

BASELINE_FEATURES: tuple[str, ...] = tuple(pre_registered_configuration()["features"]["model_features"])
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_FEATURES,
    "baseline_plus_log1p_amount": BASELINE_FEATURES + (LOG1P_AMOUNT_COLUMN,),
}
EXPERIMENTS: dict[str, dict] = {
    "E0": {"family": LOGISTIC_REGRESSION, "feature_set": "baseline"},
    "E1": {"family": LOGISTIC_REGRESSION, "feature_set": "baseline_plus_log1p_amount"},
    "E2": {"family": HIST_GRADIENT_BOOSTING, "feature_set": "baseline"},
    "E3": {"family": HIST_GRADIENT_BOOSTING, "feature_set": "baseline_plus_log1p_amount"},
}
#: Pairwise contrasts. Each isolates one factor; the others are held fixed.
CONTRASTS: dict[str, tuple[str, str, str]] = {
    "E1_minus_E0": ("E1", "E0", "log1p(Amount) effect within LR"),
    "E3_minus_E2": ("E3", "E2", "log1p(Amount) effect within HGB"),
    "E2_minus_E0": ("E2", "E0", "model-family effect (HGB vs LR), baseline features"),
    "E3_minus_E1": ("E3", "E1", "model-family effect (HGB vs LR), features with log1p(Amount)"),
}

#: One grid point of the frozen HGB grid (protocol Section 5.3), chosen a priori.
HGB_PARAMS: dict = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "max_iter": 300,
    "early_stopping": False,
    "l2_regularization": 1.0,
    "max_bins": 255,
}

TRACK_A, TRACK_B, TRACK_C = "track_a_historical", "track_b_primary", "track_c_diagnostic"
TRACK_REPRESENTATION = {TRACK_A: "predict_proba", TRACK_B: "decision_function", TRACK_C: "predict_proba"}
TRACK_ROLE = {
    TRACK_A: "historical benchmark (Phase 4b; full train; quoted, never recomputed)",
    TRACK_B: "Phase 5 primary (T_fit; decision_function)",
    TRACK_C: ("diagnostic only (same T_fit fit as track B; predict_proba); never used for selection, "
              "thresholds, calibration or primary evaluation"),
}
REQUIRED_TRACK_KEYS = ("score_representation", "role", "fit_rows")
NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Builders and fitting
# ---------------------------------------------------------------------------

def build_hgb(feature_columns: Sequence[str], random_state: int) -> Pipeline:
    """Unfitted HGB pipeline. The first step only selects ``feature_columns`` (no scaling)."""
    columns = list(feature_columns)
    if not columns:
        raise ValueError("At least one feature column is required.")
    return Pipeline([
        ("preprocess", ColumnTransformer([("select", "passthrough", columns)], remainder="drop",
                                         verbose_feature_names_out=False)),
        ("model", HistGradientBoostingClassifier(random_state=random_state, **HGB_PARAMS)),
    ])


def experiment_features(experiment_id: str) -> tuple[str, ...]:
    return FEATURE_SETS[EXPERIMENTS[experiment_id]["feature_set"]]


def prepare_frames(experiment_id: str, fit_rows: pd.DataFrame, validation_rows: pd.DataFrame,
                   time_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check the row sets and add ``log1p(Amount)`` when the experiment uses it.

    ``fit_rows`` must lie in ``T_fit`` and ``validation_rows`` in the validation
    period, so neither ``T_cal`` nor test rows can enter. The transform is
    stateless and is applied to each frame separately (inputs are not modified).
    """
    if experiment_id not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {experiment_id!r}.")
    if fit_rows.empty or validation_rows.empty:
        raise ValueError("fit_rows and validation_rows must be non-empty.")
    if (fit_rows[time_column] > T_FIT_END_TIME).any():
        raise ValueError(f"fit_rows must be T_fit rows (Time <= {T_FIT_END_TIME}).")
    vt = validation_rows[time_column]
    if (vt < VALIDATION_START_TIME).any() or (vt > VALIDATION_END_TIME).any():
        raise ValueError("validation_rows must lie in the validation period; test rows are refused.")
    if LOG1P_AMOUNT_COLUMN in experiment_features(experiment_id):
        return add_log1p_amount(fit_rows), add_log1p_amount(validation_rows)
    return fit_rows, validation_rows


def fit_experiment(experiment_id: str, fit_rows: pd.DataFrame, validation_rows: pd.DataFrame,
                   config: DatasetConfig) -> dict:
    """Fit one experiment on ``T_fit`` and score the validation rows with both representations."""
    fit_rows, validation_rows = prepare_frames(experiment_id, fit_rows, validation_rows, config.time_column)
    features = experiment_features(experiment_id)
    if config.time_column in features:
        raise AssertionError("Time must never be a model feature.")
    exp_config = dataclasses.replace(config, feature_columns=features)
    df = pd.concat([fit_rows, validation_rows])
    if df.index.has_duplicates:
        raise ValueError("fit_rows and validation_rows must not share index labels.")
    partitions = pd.Series(pd.Categorical([TRAIN] * len(fit_rows) + [VALIDATION] * len(validation_rows),
                                          categories=[TRAIN, VALIDATION, "test"], ordered=True),
                           index=df.index, name="partition")

    family = EXPERIMENTS[experiment_id]["family"]
    refit = False
    if family == LOGISTIC_REGRESSION:
        max_iter = MAX_ITER
        start = time.perf_counter()
        scored = fit_and_score(df, partitions, exp_config, build_continuity_lr(max_iter), MODEL_RANDOM_STATE)
        if scored.convergence_warnings:  # pre-registered rule: refit once with max_iter 5000
            refit, max_iter = True, MAX_ITER_ON_WARNING
            scored = fit_and_score(df, partitions, exp_config, build_continuity_lr(max_iter), MODEL_RANDOM_STATE)
        params = {**CONTINUITY_LR_PARAMS, "penalty": "l2 (scikit-learn default)", "max_iter": max_iter,
                  "random_state": MODEL_RANDOM_STATE}
        estimator = "LogisticRegression + StandardScaler (fitted on T_fit rows only)"
    else:
        start = time.perf_counter()
        scored = fit_and_score(df, partitions, exp_config, build_hgb, MODEL_RANDOM_STATE)
        params = {**HGB_PARAMS, "random_state": MODEL_RANDOM_STATE}
        estimator = "HistGradientBoostingClassifier (no scaler; fitted on T_fit rows only)"
    fit_seconds = time.perf_counter() - start

    X_val = scored.evaluation_rows[list(features)]
    start = time.perf_counter()
    decision = ranking_scores(scored.pipeline, X_val, config.positive_label)
    score_seconds = time.perf_counter() - start
    proba = scored.y_score.to_numpy(dtype=np.float64)
    y_pred = scored.pipeline.predict(X_val)
    model = scored.pipeline.named_steps["model"]
    return {
        "experiment_id": experiment_id,
        "family": family,
        "feature_set": EXPERIMENTS[experiment_id]["feature_set"],
        "features": list(features),
        "pipeline": scored.pipeline,
        "validation_index": scored.evaluation_rows.index.to_numpy(),
        "y_true": scored.y_true.to_numpy(),
        "decision_function": decision,
        "predict_proba": proba,
        "default_predict": np.asarray(y_pred),
        "fit_info": {
            "estimator": estimator,
            "params": params,
            "n_features": len(features),
            "n_fit_rows": int(len(fit_rows)),
            "n_fit_positive": int((fit_rows[config.target_column] == config.positive_label).sum()),
            "n_iter": scored.n_iter,
            "convergence_warnings": scored.convergence_warnings,
            "refit_with_max_iter_5000": refit if family == LOGISTIC_REGRESSION else None,
            "n_trees": int(model.n_iter_) if family == HIST_GRADIENT_BOOSTING else None,
        },
        "runtime_seconds": {"fit_and_predict_proba": fit_seconds, "decision_function": score_seconds},
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def ranking_metrics(y_true, scores, positive_label=1) -> dict:
    """AP and ROC-AUC with counts; NaN where undefined (no positive / one class)."""
    y = np.asarray(y_true) == positive_label
    n_pos = int(y.sum())
    return {
        "n_rows": int(y.size),
        "n_positive": n_pos,
        "prevalence": n_pos / y.size,
        "average_precision": float(average_precision_score(y, scores)) if n_pos else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < n_pos < y.size else float("nan"),
    }


def evaluate_scores(y_true, scores, times, positions, ks, positive_label=1) -> dict:
    """Ranking metrics plus the fixed-K table for one score vector on one row set."""
    return {**ranking_metrics(y_true, scores, positive_label),
            "fixed_k": fixed_k_table(y_true, scores, times, positions, ks, positive_label)}


def default_rule_diagnostic(y_true, y_pred, positive_label=1) -> dict:
    """Alerts flagged by the estimator's default ``predict()`` (decision_function > 0). Diagnostic only."""
    y = np.asarray(y_true) == positive_label
    flagged = np.asarray(y_pred) == positive_label
    tp, n_flag, n_pos = int((y & flagged).sum()), int(flagged.sum()), int(y.sum())
    return {
        "rule": "estimator default predict() (decision_function > 0); not a selected or operational threshold",
        "n_alerts": n_flag, "alert_share_of_rows": n_flag / y.size, "tp": tp, "fp": n_flag - tp,
        "precision": tp / n_flag if n_flag else float("nan"),
        "recall": tp / n_pos if n_pos else float("nan"),
    }


def evaluate_track(scores, y_true, times, positions, window_labels, ks_full, ks_window,
                   positive_label=1) -> dict:
    """Full-validation and per-window evaluation of one score vector."""
    out = {"validation": evaluate_scores(y_true, scores, times, positions, ks_full, positive_label),
           "windows": {}}
    for window in pd.unique(window_labels):
        mask = np.asarray(window_labels) == window
        out["windows"][str(window)] = evaluate_scores(np.asarray(y_true)[mask], np.asarray(scores)[mask],
                                                      np.asarray(times)[mask], np.asarray(positions)[mask],
                                                      ks_window, positive_label)
    out["score_diagnostics"] = score_diagnostics(scores)
    return out


def validate_experiment_tracks(experiments: dict) -> None:
    """Every experiment needs tracks A/B/C; each non-N/A track must name the frozen representation."""
    for exp_id in EXPERIMENTS:
        if exp_id not in experiments:
            raise ValueError(f"Missing experiment {exp_id!r}.")
        tracks = experiments[exp_id]["tracks"]
        for track in (TRACK_A, TRACK_B, TRACK_C):
            if track not in tracks:
                raise ValueError(f"{exp_id}: missing {track!r}.")
            block = tracks[track]
            if block.get("status") == NOT_APPLICABLE:
                if track != TRACK_A or exp_id == "E0":
                    raise ValueError(f"{exp_id}: {track!r} cannot be marked not applicable.")
                continue
            missing = [k for k in REQUIRED_TRACK_KEYS if k not in block]
            if missing:
                raise ValueError(f"{exp_id}/{track} lacks required field(s) {missing}.")
            if block["score_representation"] != TRACK_REPRESENTATION[track]:
                raise ValueError(f"{exp_id}/{track} must use {TRACK_REPRESENTATION[track]}.")


# ---------------------------------------------------------------------------
# Temporal stability over the E2 held-out halves (descriptive)
# ---------------------------------------------------------------------------

def _summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "min": float(v.min()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "max": float(v.max()),
            "mean": float(v.mean())}


def heldout_half_stability(times, y_true, scores_by_experiment: dict, contrasts=CONTRASTS,
                           positive_label=1) -> dict:
    """AP of each fixed experiment on the buffered held-out half of every usable E2 split.

    Reuses the frozen E2 split machinery (21 blocks, 2,000 draws of
    ``default_rng(42)``, 300-s buffers, usable iff >= 10 held-out frauds). The
    configurations are **fixed**: there is no re-selection on the selection
    half, so this is *not* the protocol's criterion E2 and is no gate.
    """
    times = np.asarray(times)
    y = np.asarray(y_true) == positive_label
    edges = block_edges(VALIDATION_START_TIME, VALIDATION_END_TIME, E2_BLOCK_SECONDS)
    if len(edges) - 1 != E2_N_BLOCKS:
        raise ValueError(f"Expected {E2_N_BLOCKS} blocks, got {len(edges) - 1}.")
    draws = draw_selection_blocks(E2_N_SPLITS, E2_N_BLOCKS, E2_N_SELECTION_BLOCKS, E2_SPLIT_SEED)
    ids = list(scores_by_experiment)
    aps: dict[str, list[float]] = {e: [] for e in ids}
    n_heldout_frauds = []
    for selection in draws:
        keep = heldout_keep_mask(times, edges, selection, E2_BUFFER_SECONDS)
        if y[keep].sum() < E2_MIN_HELDOUT_FRAUDS:
            continue
        n_heldout_frauds.append(int(y[keep].sum()))
        for e in ids:
            aps[e].append(float(average_precision_score(y[keep], np.asarray(scores_by_experiment[e])[keep])))
    arr = {e: np.asarray(v) for e, v in aps.items()}
    n_usable = len(n_heldout_frauds)
    contrast_out = {}
    for name, (a, b, meaning) in contrasts.items():
        if a not in arr or b not in arr:
            continue
        d = arr[a] - arr[b]
        contrast_out[name] = {"meaning": meaning, "delta_ap": _summary(d),
                              "share_positive": float((d > 0).mean()) if n_usable else None,
                              "share_negative": float((d < 0).mean()) if n_usable else None}
    best_counts = {e: 0 for e in ids}
    if n_usable:
        stacked = np.vstack([arr[e] for e in ids])
        for j in np.argmax(stacked, axis=0):  # first experiment wins exact ties
            best_counts[ids[j]] += 1
    return {
        "scope": ("descriptive temporal-stability check of FIXED configurations on the buffered held-out "
                  "halves of the frozen E2 splits; NOT the protocol criterion E2 (no re-selection, "
                  "no 28-configuration grid) and not a probability, confidence level or test"),
        "n_splits_attempted": int(len(draws)),
        "n_usable": n_usable,
        "heldout_frauds": _summary(n_heldout_frauds),
        "heldout_ap": {e: _summary(arr[e]) for e in ids},
        "contrasts": contrast_out,
        "share_of_splits_where_best": {e: (c / n_usable if n_usable else None) for e, c in best_counts.items()},
        "caveat": ("all splits reuse the same 21 blocks and 56 validation frauds; they are not independent "
                   "samples, and the spread understates uncertainty about other periods"),
    }
