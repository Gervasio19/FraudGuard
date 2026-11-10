"""Frozen, pre-registered Phase 5 configuration (Phase 5B-0).

Every value here is taken from the approved experimental protocol
(``docs/phase_5a_experimental_protocol.md``, Revision 4) or from its
implementation clarification addendum (Addendum C, Phase 5B-0). The section
that fixes each value is cited next to it. Nothing here is a result: no model
has been fitted or scored to produce any of these values.

This module only *declares* settings. It fits nothing, reads no data and draws
no random numbers. Changing any value after a Phase 5 score exists is a
protocol deviation and must be logged as such (protocol Section 12.2).
"""

from __future__ import annotations

import copy

from src.data.split import TRAIN_END_TIME, VALIDATION_END_TIME

# ---------------------------------------------------------------------------
# Seeds (protocol Sections 4.1, 7.2, 12.2; Addendum C, item C3)
# ---------------------------------------------------------------------------

#: ``random_state`` of every model (LR and HGB) (protocol Sections 4.1, 12.2).
MODEL_RANDOM_STATE = 42
#: Seed of the repeated held-out block splits behind criterion E2
#: (protocol Section 7.5; frozen by Addendum C, item C3).
E2_SPLIT_SEED = 42
#: Seed of the T_cal calibrator-parameter block bootstrap
#: (protocol Section 6.5 (A); frozen by Addendum C, item C3).
TCAL_BOOTSTRAP_SEED = 42
#: Seed of the conditional paired block bootstrap behind criterion E1, and of
#: the calibration-evaluation bootstrap that reuses its replicates
#: (protocol Section 7.2: ``numpy.random.default_rng(0)``).
CONDITIONAL_BOOTSTRAP_SEED = 0
#: Seed of every other bootstrap (i.i.d. continuity CI, W1 replay bootstraps)
#: under the general rule of protocol Section 12.2 ("bootstrap default_rng(0)").
OTHER_BOOTSTRAP_SEED = 0
# Generator handling (Addendum C, item C3): each analysis creates its own fresh
# generator from its seed. This makes every result reproducible regardless of
# the order in which analyses run. It does NOT make different analyses'
# random streams statistically independent: generators created from the same
# seed produce the same stream. Within one analysis, both models share each
# replicate's resampled rows (paired bootstrap).

# ---------------------------------------------------------------------------
# Ranking score representation (Addendum C, item C5)
# ---------------------------------------------------------------------------

#: The uncalibrated score used for AP, ROC-AUC, every ranking and as the
#: calibrator input, for both model families.
RANKING_SCORE_METHOD = "decision_function"

# ---------------------------------------------------------------------------
# Partitions (protocol Sections 3.1, 6.2, 10.1)
# ---------------------------------------------------------------------------

#: Last ``Time`` (inclusive) of T_fit. T_cal starts at the next timestamp present.
T_FIT_END_TIME = 112_553
#: First ``Time`` (inclusive) of T_cal (option C-A, last ~20% of train rows).
T_CAL_START_TIME = 112_555
T_CAL_END_TIME = TRAIN_END_TIME
VALIDATION_START_TIME = TRAIN_END_TIME + 1
#: Last ``Time`` (inclusive) of Window 1; Window 2 starts at 142,281.
WINDOW_1_END_TIME = 142_280

# ---------------------------------------------------------------------------
# E2: repeated held-out block splits (protocol Sections 4.3, 7.5; Addendum C4)
# ---------------------------------------------------------------------------

E2_BLOCK_SECONDS = 15 * 60
E2_N_BLOCKS = 21
E2_N_SELECTION_BLOCKS = 10
E2_BUFFER_SECONDS = 5 * 60
E2_N_SPLITS = 2_000
E2_MIN_HELDOUT_FRAUDS = 10
E2_MIN_USABLE_SPLITS = 1_000
E2_STABILITY_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Calibration (protocol Sections 6.3-6.5)
# ---------------------------------------------------------------------------

#: Numerical-constancy guard: SD of calibrated logits below this is "constant".
CALIBRATION_CONSTANT_LOGIT_SD = 1e-8
CALIBRATION_MIN_VALID_SHARE = 0.90
#: Reliability decade-bin edges (protocol Section 6.5; lower-tail bin added by
#: Addendum C, item C8). Bins are [e_k, e_{k+1}) except the last, [0.1, 1.0],
#: which is closed so that p = 1.0 is included. Together they cover [0, 1].
RELIABILITY_BIN_EDGES: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


_PRE_REGISTERED_CONFIGURATION: dict = {
    "seeds": {
        "model_random_state": MODEL_RANDOM_STATE,
        "e2_split_seed": E2_SPLIT_SEED,
        "tcal_bootstrap_seed": TCAL_BOOTSTRAP_SEED,
        "conditional_bootstrap_seed": CONDITIONAL_BOOTSTRAP_SEED,
        "other_bootstrap_seed": OTHER_BOOTSTRAP_SEED,
        "generator": "numpy.random.default_rng",
        "generator_handling": (
            "fresh generator per analysis: reproducible independent of execution order; "
            "streams from the same seed are not statistically independent; "
            "replicates are paired between models within an analysis"
        ),
    },
    "splits": {
        "chronological": {
            "train_time": [0, TRAIN_END_TIME],
            "validation_time": [VALIDATION_START_TIME, VALIDATION_END_TIME],
            "test_time_start": VALIDATION_END_TIME + 1,
            "source": "src/data/split.py (Phase 4a, unchanged)",
        },
        "train_subpartitions": {
            "t_fit_time": [0, T_FIT_END_TIME],
            "t_cal_time": [T_CAL_START_TIME, T_CAL_END_TIME],
            "calibration_holdout_option": "C-A",
            "sensitivity_holdouts_time_start": {"C-B_10pct": 124_075, "C-B_15pct": 119_715},
        },
        "validation_windows": {
            "window_1_time": [VALIDATION_START_TIME, WINDOW_1_END_TIME],
            "window_2_time": [WINDOW_1_END_TIME + 1, VALIDATION_END_TIME],
            "source": "src/data/temporal_windows.py (Phase 4c, unchanged)",
        },
    },
    "features": {
        "model_features": [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "excluded_from_features": ["Time"],
        "target": "Class",
    },
    "models": {
        "logistic_regression": {
            "grid": {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "fixed": {
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "max_iter_on_convergence_warning": 5000,
                "random_state": MODEL_RANDOM_STATE,
                "preprocessing": "StandardScaler fitted on T_fit inside the Pipeline",
            },
            "n_configurations": 12,
        },
        "hist_gradient_boosting": {
            "grid": {
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "min_samples_leaf": [20, 200],
                "class_weight": [None, "balanced"],
            },
            "fixed": {
                "max_iter": 300,
                "early_stopping": False,
                "l2_regularization": 1.0,
                "max_bins": 255,
                "random_state": MODEL_RANDOM_STATE,
                "preprocessing": "none (no scaler)",
            },
            "n_configurations": 16,
        },
        "reference": {"estimator": "DummyClassifier", "strategy": "prior", "role": "reported only"},
        "selection": {
            "objective": "max validation average precision on uncalibrated ranking scores",
            "tie_tolerance": 1e-12,
            "tie_break": "simpler model first (protocol Section 5.4)",
        },
    },
    "ranking_score": {
        "method": RANKING_SCORE_METHOD,
        "dtype": "float64",
        "positive_class": 1,
        "tie_break_key": ["score descending", "Time ascending", "file row position ascending"],
        "equality": "exact float64 comparison, no tolerance",
    },
    "calibration": {
        "method": "sigmoid (Platt)",
        "holdout": "T_cal (C-A)",
        "input": RANKING_SCORE_METHOD,
        "protocol_logit": "z = A * s + B; calibrated p = expit(z)",
        "sklearn_mapping": "A = -a_, B = -b_ (sklearn: p = expit(-(a_ * s + b_)))",
        "technical_failure": [
            "fit error, non-convergence or non-finite parameters",
            "slope A <= 0",
            "SD of calibrated logits on T_cal < 1e-8 (population SD, ddof=0)",
        ],
        "constant_logit_sd": CALIBRATION_CONSTANT_LOGIT_SD,
        "safe_alternative": "uncalibrated scores for the affected model only",
        "status_per_model": "independent for LR and HGB",
        "sensitivities": ["isotonic", "C-B 10%", "C-B 15%"],
        "c_f": "retrospective sensitivity only, on explicit request; never a fallback",
        "amount_tercile_calibration": "not required in Phase 5B (Addendum C, item C2)",
        "reliability_bins": {
            "edges": list(RELIABILITY_BIN_EDGES),
            "rule": "[e_k, e_k+1) for every bin except the last, which is closed: [0.1, 1.0]",
            "covers": "[0, 1]; every probability falls in exactly one bin",
            "source": "protocol Section 6.5 decade edges; lower-tail bin [0, 1e-6) added by Addendum C, item C8",
        },
        "bootstrap_min_valid_share": CALIBRATION_MIN_VALID_SHARE,
        "tcal_bootstrap": {"scheme": "CBB", "block_minutes": 15, "n_replicates": 2000},
    },
    "uncertainty": {
        "conditional_bootstrap": {
            "primary_scheme": "circular block bootstrap (CBB)",
            "sensitivity_schemes": ["moving block (MBB)", "stationary"],
            "block_minutes_reported": [5, 10, 15, 30],
            "e1_envelope_block_minutes": [10, 15, 30],
            "n_replicates_primary": 5000,
            "n_replicates_sensitivity": 2000,
            "interval": "percentile 95%",
        },
        "e2_heldout_splits": {
            "block_seconds": E2_BLOCK_SECONDS,
            "n_blocks": E2_N_BLOCKS,
            "n_selection_blocks": E2_N_SELECTION_BLOCKS,
            "buffer_seconds": E2_BUFFER_SECONDS,
            "n_splits_attempted": E2_N_SPLITS,
            "min_heldout_frauds": E2_MIN_HELDOUT_FRAUDS,
            "min_usable_splits": E2_MIN_USABLE_SPLITS,
            "stability_threshold": E2_STABILITY_THRESHOLD,
        },
    },
    "operations": {
        "fixed_k": [25, 50, 100, 250, 500],
        "fixed_k_windows": [13, 25, 50, 125, 250],
        "queue_capacity_per_hour": [5, 10, 20, 50],
        "sla_hours": [1, 2, 4],
        "review_cost_ratio_R": [5, 10, 20, 50, 100, 200, 500],
        "amount_aware_cost_accounting": {"c_r": [1, 5, 10], "alpha": [0, 20], "lambda": 1},
        "amount_aware_decision_policy": "excluded (decision #9)",
        "review_minutes": [5, 10, 15],
    },
    "reporting": {
        "phase4_continuity_comparison": {
            "source": "mentor review of Phase 5B-0 (2026-11-11); protocol Section 4.4, Addendum C5",
            "columns": [
                {
                    "label": "Phase 4 historical",
                    "model": "LogisticRegression C=1.0, class_weight=balanced, fitted on full train",
                    "score_representation": "predict_proba",
                    "source": "reports/phase4b_baseline_results.json (quoted, never recomputed)",
                    "role": "historical reference",
                },
                {
                    "label": "Phase 5 primary",
                    "model": "LogisticRegression C=1.0, class_weight=balanced, fitted on T_fit",
                    "score_representation": "decision_function",
                    "role": "primary",
                },
                {
                    "label": "Phase 5 diagnostic",
                    "model": "the same Phase 5 T_fit fit as the primary column",
                    "score_representation": "predict_proba",
                    "role": "diagnostic only; never used for model selection or primary evaluation",
                },
            ],
        },
    },
    "champion_rule": {
        "default": "logistic_regression",
        "promote_hist_gradient_boosting_only_if": ["E1 passes", "E2 passes"],
        "e2_not_estimable": "logistic_regression retained",
    },
}


def pre_registered_configuration() -> dict:
    """Return a deep copy of the frozen configuration (JSON-serializable)."""
    return copy.deepcopy(_PRE_REGISTERED_CONFIGURATION)
