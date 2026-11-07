"""Run the Phase 4b baseline experiments.

Experiment A (primary):   chronological 70/15/15 split (``src/data/split.py``).
Experiment B (secondary): duplicate-group-aware stratified random 70/15/15
                          split (``src/data/random_split.py``).

Both baselines are fitted on each experiment's training partition and
evaluated on its validation partition only. The experiments are reported
separately and never pooled. No threshold is selected and no test rows are
scored.

Usage, from the repository root::

    python -m src.models.run_baselines [--output reports/phase4b_baseline_results.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import replace
from pathlib import Path

import pandas as pd
import sklearn

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.random_split import DEFAULT_FRACTIONS, assign_stratified_group_partitions
from src.data.split import TRAIN_END_TIME, VALIDATION_END_TIME, assign_partitions
from src.models.baselines import BASELINE_BUILDERS, LOGISTIC_REGRESSION_PARAMS
from src.models.experiment import fit_and_evaluate, summarize_partitions

DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase4b_baseline_results.json"


def run(config: DatasetConfig, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Run both experiments on ``config``'s dataset and return a JSON-ready dict."""
    if config.time_column is None:
        raise ValueError("The chronological experiment requires a configured time column.")
    df = pd.read_csv(config.path)
    group_columns = config.resolve_duplicate_group_columns(df)

    experiments = {
        "A_chronological": {
            "role": "primary",
            "split": {
                "method": "chronological",
                "time_column": config.time_column,
                "train_end_time_inclusive": TRAIN_END_TIME,
                "validation_end_time_inclusive": VALIDATION_END_TIME,
            },
            "partitions": assign_partitions(df, config.time_column),
        },
        "B_stratified_random": {
            "role": "secondary benchmark",
            "split": {
                "method": "stratified random, duplicate groups kept together",
                "fractions": list(DEFAULT_FRACTIONS),
                "random_state": random_state,
                "duplicate_group_columns": "all columns except target"
                if config.duplicate_group_columns is None else group_columns,
            },
            "partitions": assign_stratified_group_partitions(
                df, config.target_column, group_columns, DEFAULT_FRACTIONS, random_state
            ),
        },
    }

    output = {
        "phase": "4b baseline modeling",
        "dataset": {
            "name": config.name,
            "path": str(config.path.relative_to(PROJECT_ROOT)) if config.path.is_relative_to(PROJECT_ROOT) else str(config.path),
            "sha256": _sha256(config.path),
            "n_rows": int(len(df)),
            "target_column": config.target_column,
            "feature_columns": config.resolve_feature_columns(df),
        },
        "models": {
            "majority_class": {"estimator": "DummyClassifier", "strategy": "prior"},
            "logistic_regression_balanced": {
                "estimator": "LogisticRegression",
                "preprocessing": "StandardScaler fitted on training partition",
                **LOGISTIC_REGRESSION_PARAMS,
                "random_state": random_state,
            },
        },
        "evaluation_partition": "validation",
        "decision_rule_note": (
            "average_precision and roc_auc are ranking metrics computed from predicted "
            "probabilities. precision, recall and confusion_matrix are diagnostic metrics at "
            "each estimator's default predict() rule (probability >= 0.5 for Logistic "
            "Regression; always the majority class for the baseline). 0.5 is not an optimal "
            "or production threshold."
        ),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "experiments": {},
    }
    for name, spec in experiments.items():
        partitions = spec["partitions"]
        output["experiments"][name] = {
            "role": spec["role"],
            "split": spec["split"],
            **summarize_partitions(df, partitions, config),
            "results": fit_and_evaluate(df, partitions, config, BASELINE_BUILDERS, random_state),
        }
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """Replace NaN with None so the output is strict JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _format(value) -> str:
    return "undefined" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def print_summary(output: dict) -> None:
    for name, experiment in output["experiments"].items():
        print(f"\n=== {name} ({experiment['role']}) — evaluated on validation ===")
        for part, stats in experiment["partitions"].items():
            print(f"  {part:<10} rows={stats['n_rows']:>7}  positives={stats['n_positive']:>4}")
        print(f"  cross-partition duplicate groups: {experiment['cross_partition_duplicate_groups']}")
        for model, result in experiment["results"].items():
            m, cm = result["metrics"], result["metrics"]["confusion_matrix"]
            print(
                f"  {model:<30} AP={_format(m['average_precision'])}  ROC-AUC={_format(m['roc_auc'])}  "
                f"precision={_format(m['precision'])}  recall={_format(m['recall'])}  "
                f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"
            )
            if result["convergence_warnings"]:
                print(f"    convergence warnings: {result['convergence_warnings']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=CREDITCARD.path, help="Path to the raw CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON results file.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    output = run(replace(CREDITCARD, path=args.data.resolve()), args.random_state)
    print_summary(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(output), indent=2) + "\n")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
