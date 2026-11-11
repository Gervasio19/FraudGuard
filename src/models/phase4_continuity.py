"""Phase 4 continuity comparison: three separate score tracks (Phase 5B-1C).

Implements protocol Section 4.4 with the three-column design frozen in
``pre_registered_configuration()["reporting"]["phase4_continuity_comparison"]``
(Addendum C5):

1. ``phase4_historical``: the Phase 4b balanced LR (C=1) fitted on the full
   train partition, ``predict_proba``. **Quoted** from
   ``reports/phase4b_baseline_results.json``; never recomputed or rewritten.
2. ``phase5_primary``: the same configuration fitted on ``T_fit`` only,
   ``decision_function``. The only track that may feed later Phase 5 work.
3. ``phase5_diagnostic``: the *same* ``T_fit`` fit as track 2, scored with
   ``predict_proba``. Diagnostic only: it never feeds hyperparameter selection,
   the champion decision, thresholds, calibration or any primary evaluation.

Tracks 1 vs 2 mix the data change (``T_fit`` and the 5.66 h gap) with the
representation change; tracks 2 vs 3 isolate the representation (same fit).

Every metric block carries a required ``score_representation`` field
(Phase 5B-0 report, Section 8 recommendation), enforced by
:func:`validate_score_tracks`. Fitting reuses the Phase 4 runner
(:func:`src.models.experiment.fit_and_score`), which refuses the test partition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import DatasetConfig
from src.data.split import TRAIN, VALIDATION
from src.features.preprocessing import build_preprocessor
from src.models.experiment import fit_and_score
from src.models.phase5_settings import MODEL_RANDOM_STATE, pre_registered_configuration
from src.models.ranking_score import ranking_scores

PHASE4_HISTORICAL = "phase4_historical"
PHASE5_PRIMARY = "phase5_primary"
PHASE5_DIAGNOSTIC = "phase5_diagnostic"
TRACK_NAMES = (PHASE4_HISTORICAL, PHASE5_PRIMARY, PHASE5_DIAGNOSTIC)
SCORE_REPRESENTATIONS = ("predict_proba", "decision_function")
EXPECTED_REPRESENTATION = {PHASE4_HISTORICAL: "predict_proba", PHASE5_PRIMARY: "decision_function",
                           PHASE5_DIAGNOSTIC: "predict_proba"}
REQUIRED_TRACK_KEYS = ("score_representation", "role", "fit_rows", "source", "metrics")

#: Phase 4 configuration, as frozen in the Phase 5 LR grid (protocol Sections 4.4, 5.2).
#: The L2 penalty is scikit-learn's default; it is not passed explicitly because
#: ``penalty=`` is deprecated in scikit-learn 1.8+ (FutureWarning). The Phase 4
#: builder (``src/models/baselines.py``) relies on the same default.
CONTINUITY_LR_PARAMS = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs"}
MAX_ITER, MAX_ITER_ON_WARNING = 1000, 5000


def build_continuity_lr(max_iter: int):
    """Builder for the C=1, balanced LR with StandardScaler (Phase 4 configuration)."""
    def builder(feature_columns: Sequence[str], random_state: int) -> Pipeline:
        return Pipeline([
            ("preprocess", build_preprocessor(feature_columns)),
            ("model", LogisticRegression(random_state=random_state, max_iter=max_iter, **CONTINUITY_LR_PARAMS)),
        ])
    return builder


def score_diagnostics(scores) -> dict:
    """Tie and saturation counts of a score vector (no labels)."""
    s = np.asarray(scores, dtype=np.float64)
    _, counts = np.unique(s, return_counts=True)
    return {"n_scores": int(s.size), "n_unique": int(counts.size),
            "n_rows_in_tied_groups": int(counts[counts > 1].sum()),
            "n_equal_to_1": int((s == 1.0).sum()), "max": float(s.max()), "min": float(s.min())}


def ranking_metrics(y_true, scores, positive_label=1) -> dict:
    y = (np.asarray(y_true) == positive_label).astype(int)
    return {"n_rows": int(y.size), "n_positive": int(y.sum()),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores))}


def quote_phase4_historical(path: Path) -> dict:
    """Read the historical track from the Phase 4b result file (read only)."""
    with open(path) as handle:
        data = json.load(handle)
    experiment = data["experiments"]["A_chronological"]
    lr = experiment["results"]["logistic_regression_balanced"]["metrics"]
    params = data["models"]["logistic_regression_balanced"]
    if params.get("C") != 1.0 or params.get("class_weight") != "balanced":
        raise ValueError("The Phase 4b file does not describe the C=1 balanced LR.")
    return {
        "score_representation": "predict_proba",
        "role": "historical reference (quoted, never recomputed)",
        "fit_rows": "full train partition (Time 0-132,928)",
        "source": f"{Path(path).parent.name}/{Path(path).name}",
        "metrics": {k: lr[k] for k in ("n_rows", "n_positive", "average_precision", "roc_auc")},
    }


def fit_phase5_tracks(fit_rows: pd.DataFrame, validation_rows: pd.DataFrame, config: DatasetConfig) -> dict:
    """Fit the Phase 4 configuration on ``T_fit`` once; score validation with both representations.

    ``fit_rows`` must be ``T_fit`` and ``validation_rows`` the validation
    partition. Test rows are never passed in; the Phase 4 runner additionally
    refuses a test evaluation partition.
    """
    df = pd.concat([fit_rows, validation_rows])
    if df.index.has_duplicates:
        raise ValueError("fit_rows and validation_rows must not share index labels.")
    partitions = pd.Series(pd.Categorical([TRAIN] * len(fit_rows) + [VALIDATION] * len(validation_rows),
                                          categories=[TRAIN, VALIDATION, "test"], ordered=True),
                           index=df.index, name="partition")

    max_iter = MAX_ITER
    scored = fit_and_score(df, partitions, config, build_continuity_lr(max_iter), MODEL_RANDOM_STATE)
    refit = bool(scored.convergence_warnings)
    if refit:  # pre-registered rule: refit once with max_iter 5000 on a convergence warning
        max_iter = MAX_ITER_ON_WARNING
        scored = fit_and_score(df, partitions, config, build_continuity_lr(max_iter), MODEL_RANDOM_STATE)

    features = config.resolve_feature_columns(df)
    X_val = scored.evaluation_rows[features]
    decision = ranking_scores(scored.pipeline, X_val, config.positive_label)
    proba = scored.y_score.to_numpy(dtype=np.float64)
    y_val = scored.y_true.to_numpy()
    fit_info = {
        "estimator": "LogisticRegression + StandardScaler (fitted on T_fit rows only)",
        "params": {**CONTINUITY_LR_PARAMS, "penalty": "l2 (scikit-learn default)", "max_iter": max_iter,
                   "random_state": MODEL_RANDOM_STATE},
        "n_fit_rows": int(len(fit_rows)),
        "n_fit_positive": int((fit_rows[config.target_column] == config.positive_label).sum()),
        "n_iter": scored.n_iter,
        "convergence_warnings": scored.convergence_warnings,
        "refit_with_max_iter_5000": refit,
        "proba_equals_expit_decision_max_abs_diff": float(np.max(np.abs(proba - expit(decision)))),
    }
    common = {"fit_rows": "T_fit (Time 0-112,553)", "source": "computed in Phase 5B-1 (src/models/run_phase5b1.py)"}
    return {
        "fit": fit_info,
        PHASE5_PRIMARY: {**common, "score_representation": "decision_function", "role": "primary",
                         "metrics": ranking_metrics(y_val, decision, config.positive_label),
                         "score_diagnostics": score_diagnostics(decision)},
        PHASE5_DIAGNOSTIC: {**common, "score_representation": "predict_proba",
                            "role": ("diagnostic only; never used for hyperparameter selection, the champion "
                                     "decision, thresholds, calibration or any primary evaluation"),
                            "metrics": ranking_metrics(y_val, proba, config.positive_label),
                            "score_diagnostics": score_diagnostics(proba)},
    }


def validate_score_tracks(tracks: dict) -> None:
    """Require the three tracks, each with a ``score_representation`` matching the frozen design."""
    frozen = pre_registered_configuration()["reporting"]["phase4_continuity_comparison"]["columns"]
    frozen_reps = [c["score_representation"] for c in frozen]
    if frozen_reps != [EXPECTED_REPRESENTATION[t] for t in TRACK_NAMES]:
        raise ValueError("Track representations no longer match the frozen configuration.")
    for name in TRACK_NAMES:
        if name not in tracks:
            raise ValueError(f"Missing score track {name!r}.")
        track = tracks[name]
        missing = [k for k in REQUIRED_TRACK_KEYS if k not in track]
        if missing:
            raise ValueError(f"Track {name!r} lacks required field(s) {missing}.")
        if track["score_representation"] not in SCORE_REPRESENTATIONS:
            raise ValueError(f"Track {name!r} has unknown score_representation {track['score_representation']!r}.")
        if track["score_representation"] != EXPECTED_REPRESENTATION[name]:
            raise ValueError(f"Track {name!r} must use {EXPECTED_REPRESENTATION[name]}.")


def continuity_comparison(historical: dict, phase5: dict) -> dict:
    """Assemble the three tracks and their pairwise differences (descriptive)."""
    tracks = {PHASE4_HISTORICAL: historical, PHASE5_PRIMARY: phase5[PHASE5_PRIMARY],
              PHASE5_DIAGNOSTIC: phase5[PHASE5_DIAGNOSTIC]}
    validate_score_tracks(tracks)

    def diff(a, b, key):
        return tracks[a]["metrics"][key] - tracks[b]["metrics"][key]

    if tracks[PHASE5_PRIMARY]["metrics"]["n_rows"] != historical["metrics"]["n_rows"]:
        raise ValueError("Phase 5 and Phase 4 tracks were evaluated on different validation rows.")
    return {
        "fit": phase5["fit"],
        "tracks": tracks,
        "differences": {
            "primary_minus_historical": {
                "average_precision": diff(PHASE5_PRIMARY, PHASE4_HISTORICAL, "average_precision"),
                "roc_auc": diff(PHASE5_PRIMARY, PHASE4_HISTORICAL, "roc_auc"),
                "interpretation": "joint effect of T_fit (removing T_cal), the 5.66 h gap and the representation change",
            },
            "diagnostic_minus_historical": {
                "average_precision": diff(PHASE5_DIAGNOSTIC, PHASE4_HISTORICAL, "average_precision"),
                "roc_auc": diff(PHASE5_DIAGNOSTIC, PHASE4_HISTORICAL, "roc_auc"),
                "interpretation": "data change only (T_fit and the gap; same representation); diagnostic",
            },
            "primary_minus_diagnostic": {
                "average_precision": diff(PHASE5_PRIMARY, PHASE5_DIAGNOSTIC, "average_precision"),
                "roc_auc": diff(PHASE5_PRIMARY, PHASE5_DIAGNOSTIC, "roc_auc"),
                "interpretation": "representation only (same fit): predict_proba saturation ties",
            },
        },
        "notes": [
            "Development (validation) figures; no selection is made from them.",
            "Phase 4 results are quoted from reports/phase4b_baseline_results.json and never rewritten.",
        ],
    }
