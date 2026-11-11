"""Phase 5B-1 runner: T_fit/T_cal partition, E2 usability, Phase 4 continuity.

Protocol Section 4.2 step 1 (sub-partitions, timestamp integrity, duplicate
crossing), Section 7.5 / Addendum C4 (E2 usable-split count) and Section 4.4 /
Addendum C5 (three-track Phase 4 continuity).

Test isolation: the test partition is removed immediately after the
label-free chronological assignment, before any label is accessed. Every later
step sees train and validation rows only. Nothing is selected, tuned or
calibrated here.

Usage, from the repository root::

    python -m src.models.run_phase5b1 [--output reports/phase5b1_results.json]
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.split import TEST, TRAIN, VALIDATION, VALIDATION_END_TIME, assign_partitions, count_cross_partition_duplicate_groups
from src.data.train_subpartitions import (
    T_CAL, T_FIT, assign_train_subpartitions, check_temporal_integrity, require_estimable, summarize_subpartitions,
)
from src.evaluation.e2_usability import e2_usability
from src.models.phase4_continuity import continuity_comparison, fit_phase5_tracks, quote_phase4_historical
from src.models.phase5_settings import T_CAL_START_TIME, T_FIT_END_TIME, pre_registered_configuration
from src.reproducibility.dataset import verify_dataset
from src.reproducibility.manifest import canonical_sha256, git_state, installed_version, write_result_json

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase5b1_results.json"
PHASE4B_RESULTS = PROJECT_ROOT / "reports" / "phase4b_baseline_results.json"
#: Protocol Section 3.1 / Appendix A [V] counts, used as a regression check only.
PROTOCOL_COUNTS = {T_FIT: {"n_rows": 159_491, "n_positive": 356}, T_CAL: {"n_rows": 39_873, "n_positive": 28}}
SUBSET = "subpartition"


def drop_test_rows(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series, int]:
    """Return (train+validation rows, their partition labels, number of test rows removed). Label-free."""
    partitions = assign_partitions(df, config.time_column)
    keep = (partitions != TEST).to_numpy()
    dev = df.loc[keep]
    if (dev[config.time_column] > VALIDATION_END_TIME).any():
        raise AssertionError("A test-period row survived the test-row removal.")
    return dev, partitions.loc[keep], int((~keep).sum())


def model_input_twins(dev: pd.DataFrame, labels: pd.Series, config: DatasetConfig) -> dict:
    """Descriptive cross-sub-partition duplicate counts (protocol Section 12.1)."""
    features = config.resolve_feature_columns(dev)
    cats = pd.Series(pd.Categorical(labels, categories=[T_FIT, T_CAL, VALIDATION], ordered=True), index=dev.index)
    groups = dev.groupby(features, sort=False, dropna=False).ngroup().to_numpy()
    frame = pd.DataFrame({"g": groups, "p": cats.cat.codes.to_numpy(),
                          "y": (dev[config.target_column] == config.positive_label).to_numpy()})
    n_parts = frame.groupby("g")["p"].nunique()
    crossing = frame["g"].isin(n_parts.index[n_parts > 1]).to_numpy()
    rows_by_part = {name: int((crossing & (frame["p"].to_numpy() == code)).sum())
                    for code, name in enumerate([T_FIT, T_CAL, VALIDATION])}
    return {
        "key_model_inputs": "V1-V28 + Amount",
        "n_groups_crossing_t_fit_t_cal_validation": int((n_parts > 1).sum()),
        "rows_in_crossing_groups": rows_by_part,
        "fraud_rows_in_crossing_groups": int(frame["y"].to_numpy()[crossing].sum()),
        "key_all_columns_except_target_crossing_groups": count_cross_partition_duplicate_groups(
            dev, cats, config.resolve_duplicate_group_columns(dev)),
        "note": "Descriptive only (as in Phases 3-4); twins are potential leakage of unmeasured effect, "
                "not proven leakage. Test rows are excluded from this count.",
    }


def compute(df: pd.DataFrame, config: DatasetConfig, phase4b_path: Path = PHASE4B_RESULTS) -> dict:
    """All Phase 5B-1 results from the raw frame. Never reads a test-row label."""
    dev, partitions, n_test_removed = drop_test_rows(df, config)
    train = dev.loc[(partitions == TRAIN).to_numpy()]
    validation = dev.loc[(partitions == VALIDATION).to_numpy()]
    times, target = config.time_column, config.target_column

    # 5B-1B: T_fit / T_cal
    sub = assign_train_subpartitions(train[times])
    integrity = check_temporal_integrity(train[times], sub)
    summary = summarize_subpartitions(sub, train[target], config.positive_label)
    require_estimable(summary)
    fit_rows = train.loc[(sub == T_FIT).to_numpy()]
    cal_rows = train.loc[(sub == T_CAL).to_numpy()]
    order_preserved = bool(np.all(np.diff(fit_rows.index.to_numpy()) > 0) and np.all(np.diff(cal_rows.index.to_numpy()) > 0))
    labels = pd.Series(np.where(partitions.to_numpy() == VALIDATION, VALIDATION, "pending"), index=dev.index, dtype=object)
    labels.loc[fit_rows.index], labels.loc[cal_rows.index] = T_FIT, T_CAL
    partition_record = {
        "boundaries": {"t_fit_time": [0, T_FIT_END_TIME], "t_cal_time": [T_CAL_START_TIME, int(train[times].max())],
                       "rule": "T_fit = Time <= 112,553; T_cal = 112,555 <= Time <= 132,928; no shuffling"},
        "temporal_integrity": integrity,
        "counts": summary,
        "matches_protocol_section_3_1": all(
            summary[n][k] == v for n, d in PROTOCOL_COUNTS.items() for k, v in d.items()),
        "rows_cover_train_exactly_once": bool(len(fit_rows) + len(cal_rows) == len(train)
                                              and fit_rows.index.intersection(cal_rows.index).empty),
        "file_order_preserved_within_subpartitions": order_preserved,
        "model_input_twins": model_input_twins(dev, labels, config),
        "estimability_rule": ("each sub-partition must hold >= 1 positive and >= 1 negative row; the protocol "
                              "pre-registers no other minimum fraud count for T_fit or T_cal"),
    }

    # 5B-1A: E2 usability (validation timestamps and labels only)
    e2 = e2_usability(validation[times].to_numpy(), validation[target].to_numpy(), config.positive_label)

    # 5B-1C: three-track continuity
    phase5 = fit_phase5_tracks(fit_rows, validation, config)
    continuity = continuity_comparison(quote_phase4_historical(phase4b_path), phase5)

    return {
        "test_isolation": {
            "test_rows_removed_before_label_access": n_test_removed,
            "max_time_in_processed_rows": int(dev[times].max()),
            "partitions_processed": [TRAIN, VALIDATION],
            "statement": "No test row reached any count, fit, score or metric in this run.",
        },
        "t_fit_t_cal_partition": partition_record,
        "e2_usability": e2,
        "phase4_continuity": continuity,
    }


def run(config: DatasetConfig = CREDITCARD) -> dict:
    dataset = verify_dataset(config.path)
    if dataset["status"] != "ok":
        raise SystemExit(f"Dataset check failed: {dataset['status']} ({dataset['path']}).")
    df = pd.read_csv(config.path)
    results = compute(df, config)
    settings = pre_registered_configuration()
    return {
        "phase": "5B-1 (E2 usability, T_fit/T_cal partition, Phase 4 continuity)",
        "result_type": "development (validation); no selection, tuning, calibration or test evaluation",
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
    parser = argparse.ArgumentParser(description="Run Phase 5B-1.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = run()
    record = write_result_json(output, args.output)
    e2 = output["e2_usability"]["decision"]
    tracks = output["phase4_continuity"]["tracks"]
    print(f"wrote {record['path']} ({record['bytes']} bytes, {record['status']})")
    print(f"E2: {e2['n_usable']}/{e2['n_attempted']} usable; estimable={e2['e2_estimable']}")
    for name, track in tracks.items():
        print(f"{name}: AP={track['metrics']['average_precision']:.4f} ({track['score_representation']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
