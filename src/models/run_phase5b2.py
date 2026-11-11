"""Phase 5B-2 runner: controlled experiments E0-E3 on T_fit, evaluated on validation.

Test isolation: the test partition is removed immediately after the label-free
chronological assignment (reusing the Phase 5B-1 helper), before any label is
read. Each experiment is fitted once on ``T_fit`` and scores the same
validation rows once; every analysis reuses those cached scores. ``T_cal`` is
neither fitted nor scored. Nothing is selected, tuned, thresholded or
calibrated here.

Usage, from the repository root::

    python -m src.models.run_phase5b2 [--output reports/phase5b2_results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.split import TRAIN, VALIDATION
from src.data.temporal_windows import split_two_time_windows
from src.data.train_subpartitions import T_FIT, assign_train_subpartitions
from src.models import phase5b2_experiments as px
from src.models.phase4_continuity import quote_phase4_historical
from src.models.phase5_settings import WINDOW_1_END_TIME, pre_registered_configuration
from src.models.run_phase5b1 import PHASE4B_RESULTS, drop_test_rows
from src.reproducibility.dataset import verify_dataset
from src.reproducibility.manifest import canonical_sha256, git_state, installed_version, write_result_json

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase5b2_results.json"
PHASE5B1_RESULTS = PROJECT_ROOT / "reports" / "phase5b1_results.json"
RUNTIME_KEY = "runtime_seconds"  # wall-clock; the only non-reproducible key in the result file


def _contrast_table(tracks_b: dict, ks_full, window_names) -> dict:
    """Track-B differences for each contrast: full validation and per window (same rows)."""
    out = {}
    for name, (a, b, meaning) in px.CONTRASTS.items():
        va, vb = tracks_b[a]["validation"], tracks_b[b]["validation"]
        out[name] = {
            "meaning": meaning,
            "score_representation": "decision_function",
            "validation": {
                "delta_average_precision": va["average_precision"] - vb["average_precision"],
                "delta_roc_auc": va["roc_auc"] - vb["roc_auc"],
                "delta_tp_at_k": {str(k): ra["tp"] - rb["tp"]
                                  for k, ra, rb in zip(ks_full, va["fixed_k"], vb["fixed_k"])},
            },
            "windows": {w: {"delta_average_precision": tracks_b[a]["windows"][w]["average_precision"]
                            - tracks_b[b]["windows"][w]["average_precision"],
                            "delta_roc_auc": tracks_b[a]["windows"][w]["roc_auc"]
                            - tracks_b[b]["windows"][w]["roc_auc"]}
                        for w in window_names},
        }
    return out


def _ap_ranking(tracks_b: dict, key: str | None) -> list[str]:
    """Experiment IDs ordered by track-B AP (descending; ties keep E0..E3 order)."""
    def ap(e):
        block = tracks_b[e]["validation"] if key is None else tracks_b[e]["windows"][key]
        return block["average_precision"]
    return sorted(tracks_b, key=lambda e: -ap(e))


def compute(df: pd.DataFrame, config: DatasetConfig, phase4b_path: Path = PHASE4B_RESULTS,
            phase5b1_path: Path | None = PHASE5B1_RESULTS, check_hgb_determinism: bool = True) -> dict:
    """All Phase 5B-2 results from the raw frame. Never reads a test-row label."""
    dev, partitions, n_test_removed = drop_test_rows(df, config)
    times_col, target = config.time_column, config.target_column
    train = dev.loc[(partitions == TRAIN).to_numpy()]
    validation = dev.loc[(partitions == VALIDATION).to_numpy()]
    fit_rows = train.loc[(assign_train_subpartitions(train[times_col]) == T_FIT).to_numpy()]

    settings = pre_registered_configuration()
    ks_full, ks_window = settings["operations"]["fixed_k"], settings["operations"]["fixed_k_windows"]
    windows = split_two_time_windows(validation[times_col])
    window_labels = windows.labels.astype(str).to_numpy()
    window_names = list(pd.unique(window_labels))
    v_times = validation[times_col].to_numpy()
    v_positions = validation.index.to_numpy()  # file row positions (RangeIndex of the raw CSV)

    fitted = {e: px.fit_experiment(e, fit_rows, validation, config) for e in px.EXPERIMENTS}
    for e, f in fitted.items():
        if not np.array_equal(f["validation_index"], v_positions):
            raise AssertionError(f"{e} was evaluated on different validation rows.")

    determinism = {}
    runtime = {e: f["runtime_seconds"] for e, f in fitted.items()}
    if check_hgb_determinism:
        for e, f in fitted.items():
            if f["family"] == px.HIST_GRADIENT_BOOSTING:
                again = px.fit_experiment(e, fit_rows, validation, config)
                determinism[e] = {
                    "second_fit_identical_decision_function": bool(
                        np.array_equal(f["decision_function"], again["decision_function"])),
                    "max_abs_difference": float(np.max(np.abs(f["decision_function"] - again["decision_function"]))),
                }
                runtime[e] = {**runtime[e], "second_fit_and_predict_proba": again["runtime_seconds"]["fit_and_predict_proba"]}

    y_val = validation[target].to_numpy()
    historical = quote_phase4_historical(phase4b_path)
    experiments = {}
    for e, f in fitted.items():
        common = {"fit_rows": "T_fit (Time 0-112,553)"}
        track_b = {**common, "score_representation": "decision_function", "role": px.TRACK_ROLE[px.TRACK_B],
                   **px.evaluate_track(f["decision_function"], y_val, v_times, v_positions, window_labels,
                                       ks_full, ks_window, config.positive_label),
                   "default_rule_diagnostic": px.default_rule_diagnostic(y_val, f["default_predict"],
                                                                        config.positive_label)}
        track_c = {**common, "score_representation": "predict_proba", "role": px.TRACK_ROLE[px.TRACK_C],
                   **px.evaluate_track(f["predict_proba"], y_val, v_times, v_positions, window_labels,
                                       ks_full, ks_window, config.positive_label)}
        if e == "E0":
            track_a = {**historical, "role": px.TRACK_ROLE[px.TRACK_A],
                       "note": "AP and ROC-AUC only; fixed-K and windows would require refitting the historical "
                               "model, which the frozen design forbids (quoted, never recomputed)"}
        else:
            track_a = {"status": px.NOT_APPLICABLE,
                       "reason": "track A is the existing Phase 4 benchmark, which exists for E0 only"}
        experiments[e] = {
            "family": f["family"], "feature_set": f["feature_set"], "features": f["features"],
            "fit": f["fit_info"],
            "tracks": {px.TRACK_A: track_a, px.TRACK_B: track_b, px.TRACK_C: track_c},
            "track_b_minus_track_c_average_precision": (track_b["validation"]["average_precision"]
                                                         - track_c["validation"]["average_precision"]),
            "predict_proba_saturation": {"n_equal_to_1": track_c["score_diagnostics"]["n_equal_to_1"],
                                         "max_logit": track_b["score_diagnostics"]["max"]},
        }
    px.validate_experiment_tracks(experiments)

    tracks_b = {e: experiments[e]["tracks"][px.TRACK_B] for e in experiments}
    continuity = None
    if phase5b1_path is not None and Path(phase5b1_path).is_file():
        with open(phase5b1_path) as handle:
            ref = json.load(handle)["phase4_continuity"]["tracks"]["phase5_primary"]["metrics"]["average_precision"]
        continuity = {"phase5b1_primary_average_precision": ref,
                      "e0_track_b_average_precision": tracks_b["E0"]["validation"]["average_precision"],
                      "abs_difference": abs(ref - tracks_b["E0"]["validation"]["average_precision"])}

    return {
        "test_isolation": {
            "test_rows_removed_before_label_access": n_test_removed,
            "max_time_in_processed_rows": int(dev[times_col].max()),
            "partitions_processed": ["t_fit (fit)", "validation (score)"],
            "t_cal": "neither fitted nor scored",
            "statement": "No test row reached any fit, score or metric in this run.",
        },
        "data": {
            "t_fit": {"n_rows": int(len(fit_rows)), "n_positive": int((fit_rows[target] == config.positive_label).sum())},
            "validation": {"n_rows": int(len(validation)), "n_positive": int((y_val == config.positive_label).sum())},
            "windows": {"window_1_end_time": windows.window_1_end_time,
                        "window_2_start_time": windows.window_2_start_time,
                        "n_window_1": windows.n_window_1, "n_window_2": windows.n_window_2,
                        "matches_frozen_window_boundary": windows.window_1_end_time == WINDOW_1_END_TIME},
            "fixed_k_validation": ks_full, "fixed_k_windows": ks_window,
            "k_source": "protocol Section 9.2 (frozen in src/models/phase5_settings.py); illustrative capacities, "
                        "not business operating limits",
        },
        "experiments": experiments,
        "comparisons": {
            "contrasts_track_b": _contrast_table(tracks_b, ks_full, window_names),
            "ap_ranking_track_b": {"validation": _ap_ranking(tracks_b, None),
                                   **{w: _ap_ranking(tracks_b, w) for w in window_names}},
        },
        "temporal_stability_heldout_halves": px.heldout_half_stability(
            v_times, y_val, {e: fitted[e]["decision_function"] for e in fitted}, positive_label=config.positive_label),
        "hgb_determinism": determinism,
        "continuity_with_phase5b1": continuity,
        RUNTIME_KEY: runtime,
    }


def run(config: DatasetConfig = CREDITCARD) -> dict:
    dataset = verify_dataset(config.path)
    if dataset["status"] != "ok":
        raise SystemExit(f"Dataset check failed: {dataset['status']} ({dataset['path']}).")
    df = pd.read_csv(config.path)
    results = compute(df, config)
    settings = pre_registered_configuration()
    return {
        "phase": "5B-2 (controlled experiments E0-E3: LR / HGB x baseline / + log1p(Amount))",
        "result_type": ("development (validation); exploratory, owner-scoped; no selection, tuning, "
                        "calibration, threshold or test evaluation"),
        "protocol_deviation": ("log1p(Amount) (experiments E1, E3) is excluded by protocol Section 5.5; it is run "
                               "on the project owner's explicit Phase 5B-2 instruction as an exploratory analysis "
                               "outside the champion pathway (logged in WORK_LOG)"),
        "provenance": {
            "dataset": dataset,
            "git": git_state(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "scipy": installed_version("scipy")},
            "seeds": settings["seeds"],
            "pre_registered_configuration_sha256": canonical_sha256(settings),
        },
        **results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5B-2 experiments E0-E3.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = run()
    record = write_result_json(output, args.output)
    print(f"wrote {record['path']} ({record['bytes']} bytes, {record['status']})")
    for e, block in output["experiments"].items():
        b = block["tracks"][px.TRACK_B]["validation"]
        c = block["tracks"][px.TRACK_C]["validation"]
        print(f"{e}: AP B={b['average_precision']:.4f} C={c['average_precision']:.4f} "
              f"ROC-AUC B={b['roc_auc']:.4f} fit={output[RUNTIME_KEY][e]['fit_and_predict_proba']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
