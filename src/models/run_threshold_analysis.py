"""Run the Phase 4c threshold analysis and temporal stability experiment.

Protocol: fit the balanced Logistic Regression on the chronological training
partition once, score the chronological validation partition once with
``predict_proba``, and reuse those scores for every threshold analysis:

1. Full-validation threshold curve and max-F1 threshold (retrospective).
2. Validation split into two consecutive time windows; a curve and max-F1
   threshold for each window.
3. Transfer: the Window 1 threshold, frozen, applied unchanged to Window 2,
   compared with Window 2's retrospective optimum.
4. Common-reference comparison (descriptive, separate from 3): the fixed 0.5
   threshold and the full-validation max-F1 threshold (retrospective) applied
   independently to each window.

The test partition is never scored. The secondary random split is not used.

Usage, from the repository root::

    python -m src.models.run_threshold_analysis [--output reports/phase4c_threshold_results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.split import TRAIN_END_TIME, VALIDATION, VALIDATION_END_TIME, assign_partitions
from src.data.temporal_windows import WINDOW_1, WINDOW_2, split_two_time_windows
from src.evaluation.threshold import (
    evaluate_threshold,
    select_max_f1_threshold,
    threshold_curve,
)
from src.models.baselines import LOGISTIC_REGRESSION, LOGISTIC_REGRESSION_PARAMS, ModelBuilder, build_logistic_regression
from src.models.experiment import fit_and_score
from src.models.run_baselines import _json_safe, _sha256

DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase4c_threshold_results.json"
SELECTION = {"selection_objective": "maximize_f1", "tie_breaking": "lowest_threshold"}


def _summary(y_true: pd.Series, times: pd.Series, positive_label) -> dict:
    n_positive = int((y_true.to_numpy() == positive_label).sum())
    return {
        "n_rows": int(len(y_true)),
        "n_positive": n_positive,
        "prevalence": n_positive / len(y_true),
        "min_time": float(times.min()),
        "max_time": float(times.max()),
    }


def _analyse(y_true: pd.Series, y_score: pd.Series, times: pd.Series, positive_label) -> dict:
    """Curve and max-F1 selection on one set of (already computed) scores."""
    curve = threshold_curve(y_true.to_numpy(), y_score.to_numpy(), positive_label)
    selected = select_max_f1_threshold(curve)
    return {
        "partition_summary": _summary(y_true, times, positive_label),
        "selected_threshold": selected["threshold"],
        **SELECTION,
        "metrics": selected,
        "n_thresholds": len(curve["threshold"]),
        "threshold_curve": curve,
    }


def analyse(
    df: pd.DataFrame,
    config: DatasetConfig,
    random_state: int = DEFAULT_RANDOM_STATE,
    builder: ModelBuilder = build_logistic_regression,
) -> dict:
    """Run the full Phase 4c protocol on an in-memory dataset (JSON-ready, NaN not yet converted)."""
    if config.time_column is None:
        raise ValueError("Phase 4c requires a configured time column.")
    partitions = assign_partitions(df, config.time_column)
    scored = fit_and_score(df, partitions, config, builder, random_state, VALIDATION)
    y_true, y_score = scored.y_true, scored.y_score
    times = scored.evaluation_rows[config.time_column]
    positive = config.positive_label

    full = _analyse(y_true, y_score, times, positive)

    windows = split_two_time_windows(times)
    labels = windows.labels.to_numpy()
    per_window = {}
    for name in (WINDOW_1, WINDOW_2):
        mask = labels == name
        per_window[name] = _analyse(y_true[mask], y_score[mask], times[mask], positive)

    # Transfer: only the Window 1 selection is used; Window 2 labels are used
    # solely to *measure* the frozen threshold, never to choose it.
    transferred_threshold = per_window[WINDOW_1]["selected_threshold"]
    w2 = labels == WINDOW_2
    transferred = evaluate_threshold(y_true[w2].to_numpy(), y_score[w2].to_numpy(), transferred_threshold, positive)
    retrospective = per_window[WINDOW_2]["metrics"]

    # Descriptive common-reference comparison (separate from the transfer above):
    # the same reference thresholds are applied to each window's existing scores.
    reference_thresholds = {
        "fixed_0_5": (0.5, "fixed reference; diagnostic only, not an operational threshold"),
        "full_validation_max_f1": (
            full["selected_threshold"],
            "retrospective: selected on full validation, which includes the labels of both windows",
        ),
    }
    common_reference = {
        "role": "descriptive comparison; not a transfer test",
        "note": (
            "Separate from threshold_transfer (Window 1 max-F1 frozen and applied to Window 2). "
            "Here each reference threshold is applied independently to both windows using the "
            "scores already computed; no refitting or rescoring."
        ),
        "decision_rule": "y_score >= threshold",
        "window_prevalence": {
            name: per_window[name]["partition_summary"]["prevalence"] for name in (WINDOW_1, WINDOW_2)
        },
        "thresholds": {},
    }
    for key, (threshold, description) in reference_thresholds.items():
        entry = {"threshold": threshold, "description": description}
        for name in (WINDOW_1, WINDOW_2):
            mask = labels == name
            entry[name] = evaluate_threshold(
                y_true[mask].to_numpy(), y_score[mask].to_numpy(), threshold, positive
            )
        common_reference["thresholds"][key] = entry

    return {
        "phase": "4c threshold analysis",
        "dataset": {
            "name": config.name,
            "path": str(config.path.relative_to(PROJECT_ROOT)) if config.path.is_relative_to(PROJECT_ROOT) else str(config.path),
            "sha256": _sha256(config.path) if config.path.exists() else None,
            "n_rows": int(len(df)),
            "target_column": config.target_column,
            "feature_columns": config.resolve_feature_columns(df),
        },
        "model": {
            "name": LOGISTIC_REGRESSION,
            "estimator": "LogisticRegression",
            "preprocessing": "StandardScaler fitted on training partition",
            **LOGISTIC_REGRESSION_PARAMS,
            "random_state": random_state,
            "n_iter": scored.n_iter,
            "convergence_warnings": scored.convergence_warnings,
            "excluded_from_threshold_analysis": {
                "majority_class": "constant probabilities; no ranking ability"
            },
        },
        "evaluation_protocol": {
            "split": "chronological",
            "train_end_time_inclusive": TRAIN_END_TIME,
            "validation_end_time_inclusive": VALIDATION_END_TIME,
            "fit_partition": "train",
            "scored_partition": "validation",
            "n_validation_rows_scored": int(len(y_score)),
            "test_partition": "not scored",
            "scores": "predict_proba positive-class probability; computed once and reused",
            "threshold_candidates": "unique validation scores of the evaluated rows",
            "decision_rule": "y_score >= threshold",
            **SELECTION,
            "undefined_values": "null",
            "f1_definition": "2*TP/(2*TP+FP+FN); null when precision or recall is undefined",
            "interpretation": (
                "Full-validation and Window 2 optima are retrospective diagnostics, not "
                "independently validated or production thresholds."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "full_validation": full,
        "temporal_windows": {
            "boundary_time": windows.window_1_end_time,
            "boundary_rule": (
                "window_1: time <= boundary_time; window_2: time > boundary_time; boundary "
                "minimizes |n_window_1 - n_window_2|, earlier boundary on ties"
            ),
            "window_2_start_time": windows.window_2_start_time,
            "n_window_1": windows.n_window_1,
            "n_window_2": windows.n_window_2,
            **per_window,
        },
        "threshold_transfer": {
            "source": WINDOW_1,
            "target": WINDOW_2,
            "transferred_threshold": transferred_threshold,
            "source_selection_f1": per_window[WINDOW_1]["metrics"]["f1"],
            "metrics": transferred,
            "retrospective_optimum": retrospective,
            "f1_difference": transferred["f1"] - retrospective["f1"],
            "f1_difference_definition": "transferred F1 - retrospective Window 2 optimal F1",
        },
        "common_reference_thresholds": common_reference,
    }


def run(config: DatasetConfig, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Load ``config.path`` and run :func:`analyse`."""
    return analyse(pd.read_csv(config.path), config, random_state)


def _fmt(value) -> str:
    return "null" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:<34} t={m['threshold']:.6f}  P={_fmt(m['precision'])}  R={_fmt(m['recall'])}  "
        f"F1={_fmt(m['f1'])}  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}  "
        f"alerts={m['n_predicted_positive']} ({100 * m['alert_rate']:.3f}%)"
    )


def print_summary(output: dict) -> None:
    fv, tw, tt = output["full_validation"], output["temporal_windows"], output["threshold_transfer"]
    s = fv["partition_summary"]
    print(f"Validation: {s['n_rows']} rows, {s['n_positive']} frauds, {fv['n_thresholds']} candidate thresholds")
    print(_row("full validation max-F1 (retro)", fv["metrics"]))
    print(f"Windows: boundary Time <= {tw['boundary_time']:.0f} | "
          f"W1 {tw['n_window_1']} rows / {tw[WINDOW_1]['partition_summary']['n_positive']} frauds | "
          f"W2 {tw['n_window_2']} rows / {tw[WINDOW_2]['partition_summary']['n_positive']} frauds")
    print(_row("window 1 max-F1 (selection)", tw[WINDOW_1]["metrics"]))
    print(_row("window 2 with frozen W1 threshold", tt["metrics"]))
    print(_row("window 2 max-F1 (retro)", tt["retrospective_optimum"]))
    print(f"  F1 difference (transferred - retrospective): {_fmt(tt['f1_difference'])}")
    print("Common-reference thresholds (descriptive, not a transfer test):")
    for key, entry in output["common_reference_thresholds"]["thresholds"].items():
        for name in (WINDOW_1, WINDOW_2):
            print(_row(f"{key} on {name}", entry[name]))


def dumps(output: dict) -> str:
    """Strict JSON (NaN -> null). Each threshold-curve column is written on one
    line to keep the file navigable; values keep full float precision."""
    safe = _json_safe(output)
    compact: dict[str, str] = {}

    def collapse(node):
        if isinstance(node, dict):
            if "threshold_curve" in node:
                curve = node["threshold_curve"]
                for column, values in curve.items():
                    key = f"@@curve{len(compact)}@@"
                    compact[key] = json.dumps(values, separators=(",", ":"), allow_nan=False)
                    curve[column] = key
            for value in node.values():
                collapse(value)

    collapse(safe)
    text = json.dumps(safe, indent=2, allow_nan=False)
    for key, values in compact.items():
        text = text.replace(f'"{key}"', values, 1)
    return text + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=CREDITCARD.path, help="Path to the raw CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON results file.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    output = run(replace(CREDITCARD, path=args.data.resolve()), args.random_state)
    print_summary(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(output))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
