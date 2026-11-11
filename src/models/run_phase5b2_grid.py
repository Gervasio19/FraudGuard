"""Runner for protocol Section 4.2 step 2: fit the 28 frozen configurations (Phase 5B-2).

Test rows are removed immediately after the label-free chronological
assignment (``run_phase5b1.drop_test_rows``). Each configuration is fitted once
on ``T_fit`` and scores ``T_cal`` and validation once each; every HGB
configuration is fitted a second time to verify determinism (Sections 5.3,
12.2). Validation metrics are computed from the cached scores. ``T_cal`` labels
are not read: ``T_cal`` scores are kept for the calibration stage and only
label-free diagnostics are reported. The step-3 rule (Section 5.4) is applied
to record the selected candidate of each family; nothing else is decided.

Usage, from the repository root::

    python -m src.models.run_phase5b2_grid [--output reports/phase5b2_grid_results.json]

Use :func:`compute_grid` (in-memory scores) for later stages.
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
from src.data.train_subpartitions import T_CAL, T_FIT, assign_train_subpartitions
from src.models import phase5_grid as grid
from src.models import phase5b2_experiments as px
from src.models.phase4_continuity import quote_phase4_historical, score_diagnostics
from src.models.phase5_settings import WINDOW_1_END_TIME, pre_registered_configuration
from src.models.run_phase5b1 import PHASE4B_RESULTS, drop_test_rows
from src.reproducibility.dataset import verify_dataset
from src.reproducibility.manifest import canonical_sha256, git_state, installed_version, write_result_json

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase5b2_grid_results.json"
PHASE5B2_RESULTS = PROJECT_ROOT / "reports" / "phase5b2_results.json"
RUNTIME_KEY = "runtime_seconds"
PHASE4_CONFIG = {"family": grid.LR, "params": {"C": 1.0, "class_weight": "balanced"}}
#: Grid points equal to the 5B-2 protocol-aligned experiments (continuity check only).
EXPERIMENT_GRID_POINTS = {
    "E0": PHASE4_CONFIG,
    "E2": {"family": grid.HGB, "params": {k: px.HGB_PARAMS[k] for k in
                                         ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "class_weight")}},
}


def split_rows(df: pd.DataFrame, config: DatasetConfig):
    """(T_fit rows, T_cal rows, validation rows, number of test rows removed). Label-free."""
    dev, partitions, n_test = drop_test_rows(df, config)
    train = dev.loc[(partitions == TRAIN).to_numpy()]
    validation = dev.loc[(partitions == VALIDATION).to_numpy()]
    sub = assign_train_subpartitions(train[config.time_column])
    return (train.loc[(sub == T_FIT).to_numpy()], train.loc[(sub == T_CAL).to_numpy()], validation, n_test,
            int(dev[config.time_column].max()))


def compute_grid(df: pd.DataFrame, config: DatasetConfig, check_hgb_determinism: bool = True) -> dict:
    """Fit all 28 configurations; return the in-memory score cache and fit records."""
    fit_rows, cal_rows, validation, n_test, max_time = split_rows(df, config)
    fitted, determinism = {}, {}
    for configuration in grid.grid_configurations():
        f = grid.fit_configuration(configuration, fit_rows, cal_rows, validation, config)
        if check_hgb_determinism and f["family"] == grid.HGB:
            again = grid.fit_configuration(configuration, fit_rows, cal_rows, validation, config)
            same = all(np.array_equal(f["scores"][p]["decision_function"], again["scores"][p]["decision_function"])
                       for p in ("validation", "t_cal"))
            determinism[f["config_id"]] = {"second_fit_identical_scores": same,
                                           "second_fit_seconds": again["runtime_seconds"]["fit_and_validation_predict_proba"]}
            if not same:
                raise AssertionError(f"{f['config_id']}: HGB fit is not deterministic.")
        f.pop("pipeline")  # scores are the cache; fitted models are not kept
        fitted[f["config_id"]] = f
    for f in fitted.values():
        if not np.array_equal(f["validation_index"], validation.index.to_numpy()):
            raise AssertionError(f"{f['config_id']} was evaluated on different validation rows.")
    return {"fitted": fitted, "determinism": determinism, "fit_rows": fit_rows, "cal_rows": cal_rows,
            "validation": validation, "n_test_removed": n_test, "max_time": max_time}


def summarize(run: dict, config: DatasetConfig, phase4b_path: Path = PHASE4B_RESULTS,
              phase5b2_path: Path | None = PHASE5B2_RESULTS) -> dict:
    """Validation metrics (tracks B, C; A for the Phase 4 configuration), selection record, diagnostics."""
    settings = pre_registered_configuration()
    ks_full, ks_window = settings["operations"]["fixed_k"], settings["operations"]["fixed_k_windows"]
    validation, fit_rows, cal_rows = run["validation"], run["fit_rows"], run["cal_rows"]
    target, times_col = config.target_column, config.time_column
    y_val = validation[target].to_numpy()
    v_times, v_pos = validation[times_col].to_numpy(), validation.index.to_numpy()
    windows = split_two_time_windows(validation[times_col])
    window_labels = windows.labels.astype(str).to_numpy()
    historical = quote_phase4_historical(phase4b_path)

    configurations, table = {}, []
    for cid, f in run["fitted"].items():
        val, cal = f["scores"]["validation"], f["scores"]["t_cal"]
        track_b = {"fit_rows": "T_fit (Time 0-112,553)", "score_representation": "decision_function",
                   "role": px.TRACK_ROLE[px.TRACK_B],
                   **px.evaluate_track(val["decision_function"], y_val, v_times, v_pos, window_labels,
                                       ks_full, ks_window, config.positive_label)}
        track_c = {"fit_rows": "T_fit (Time 0-112,553)", "score_representation": "predict_proba",
                   "role": px.TRACK_ROLE[px.TRACK_C],
                   **px.evaluate_track(val["predict_proba"], y_val, v_times, v_pos, window_labels,
                                       ks_full, ks_window, config.positive_label)}
        is_phase4 = f["family"] == PHASE4_CONFIG["family"] and f["params"] == PHASE4_CONFIG["params"]
        track_a = ({**historical, "role": px.TRACK_ROLE[px.TRACK_A]} if is_phase4 else
                   {"status": px.NOT_APPLICABLE, "reason": "the Phase 4 benchmark exists for LR C=1, balanced only"})
        configurations[cid] = {
            "family": f["family"], "params": f["params"], "fit": f["fit_info"],
            "tracks": {px.TRACK_A: track_a, px.TRACK_B: track_b, px.TRACK_C: track_c},
            "track_b_minus_track_c_average_precision": (track_b["validation"]["average_precision"]
                                                         - track_c["validation"]["average_precision"]),
            "t_cal_scores": {"note": "scored once and cached for the calibration stage; label-free diagnostics only",
                             "decision_function": score_diagnostics(cal["decision_function"]),
                             "predict_proba": score_diagnostics(cal["predict_proba"])},
        }
        recall_100 = next(r for r in track_b["validation"]["fixed_k"] if r["k"] == 100)
        table.append({"config_id": cid, "family": f["family"], "params": f["params"],
                      "average_precision": track_b["validation"]["average_precision"],
                      "roc_auc": track_b["validation"]["roc_auc"],
                      "recall_at_100": recall_100["recall_at_k"], "tp_at_100": recall_100["tp"],
                      "average_precision_track_c": track_c["validation"]["average_precision"]})

    # every configuration block must carry the frozen representations
    for cid, block in configurations.items():
        for track, rep in px.TRACK_REPRESENTATION.items():
            t = block["tracks"][track]
            if t.get("status") != px.NOT_APPLICABLE and t["score_representation"] != rep:
                raise ValueError(f"{cid}/{track} must use {rep}.")

    selection = {fam: grid.select_within_family(table, fam) for fam in grid.FAMILIES}

    continuity = {}
    if phase5b2_path is not None and Path(phase5b2_path).is_file():
        recorded = json.loads(Path(phase5b2_path).read_text())["experiments"]
        for exp_id, point in EXPERIMENT_GRID_POINTS.items():
            row = next(r for r in table if r["family"] == point["family"] and r["params"] == point["params"])
            ref = recorded[exp_id]["tracks"][px.TRACK_B]["validation"]["average_precision"]
            continuity[exp_id] = {"grid_config_id": row["config_id"], "phase5b2_average_precision": ref,
                                  "grid_average_precision": row["average_precision"],
                                  "abs_difference": abs(ref - row["average_precision"])}

    return {
        "test_isolation": {
            "test_rows_removed_before_label_access": run["n_test_removed"],
            "max_time_in_processed_rows": run["max_time"],
            "partitions_processed": ["t_fit (fit)", "t_cal (scored; labels not read)", "validation (scored)"],
            "statement": "No test row reached any fit, score or metric in this run.",
        },
        "data": {
            "t_fit": {"n_rows": int(len(fit_rows)), "n_positive": int((fit_rows[target] == config.positive_label).sum())},
            "t_cal": {"n_rows": int(len(cal_rows))},
            "validation": {"n_rows": int(len(validation)), "n_positive": int((y_val == config.positive_label).sum())},
            "windows": {"window_1_end_time": windows.window_1_end_time, "n_window_1": windows.n_window_1,
                        "n_window_2": windows.n_window_2,
                        "matches_frozen_window_boundary": windows.window_1_end_time == WINDOW_1_END_TIME},
            "features": list(grid.model_features()),
            "fixed_k_validation": ks_full, "fixed_k_windows": ks_window,
        },
        "grid_table_track_b": table,
        "configurations": configurations,
        "step3_selection_record": {
            "status": ("pre-registered Section 5.4 rule applied to record the candidates for the later "
                       "stages; development (selection) result, optimistic by selection; not a champion decision"),
            **selection,
        },
        "hgb_determinism": run["determinism"],
        "continuity_with_phase5b2_experiments": continuity,
        RUNTIME_KEY: {cid: f["runtime_seconds"] for cid, f in run["fitted"].items()},
    }


def run(config: DatasetConfig = CREDITCARD) -> dict:
    dataset = verify_dataset(config.path)
    if dataset["status"] != "ok":
        raise SystemExit(f"Dataset check failed: {dataset['status']} ({dataset['path']}).")
    df = pd.read_csv(config.path)
    results = summarize(compute_grid(df, config), config)
    settings = pre_registered_configuration()
    return {
        "phase": "5B-2, protocol Section 4.2 step 2 (28-configuration grid on T_fit)",
        "result_type": ("development (validation); grid fit and scoring; step-3 candidates recorded; "
                        "no calibration, champion decision or test evaluation"),
        "scope_note": "log1p(Amount) experiments E1/E3 are exploratory deviations and are not part of this grid",
        "provenance": {
            "dataset": dataset, "git": git_state(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__, "scipy": installed_version("scipy")},
            "seeds": settings["seeds"],
            "pre_registered_configuration_sha256": canonical_sha256(settings),
        },
        **results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the 28-configuration Phase 5 grid (protocol 4.2 step 2).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = run()
    record = write_result_json(output, args.output)
    print(f"wrote {record['path']} ({record['bytes']} bytes, {record['status']})")
    for row in output["grid_table_track_b"]:
        print(f"{row['config_id']} {row['params']} AP={row['average_precision']:.6f} "
              f"ROC={row['roc_auc']:.4f} R@100={row['recall_at_100']:.3f}")
    for fam, sel in output["step3_selection_record"].items():
        if isinstance(sel, dict):
            print(f"{fam}: selected {sel['selected_config_id']} AP={sel['selected_average_precision']:.6f} "
                  f"spread={sel['spread_max_minus_median']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
