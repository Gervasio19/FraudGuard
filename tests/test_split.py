"""Tests for the chronological partitioning module (``src/data/split.py``).

Run from the repository root:

    python -m unittest discover -s tests -v

Tests marked as real-data tests read ``data/raw/creditcard.csv`` (read-only)
and are skipped when the file is not present, since raw data is not tracked.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import (
    PARTITION_NAMES,
    TEST,
    TRAIN,
    TRAIN_END_TIME,
    VALIDATION,
    VALIDATION_END_TIME,
    assign_partitions,
    count_cross_partition_duplicate_groups,
    split_partitions,
)

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"

# Documented in docs/validation_strategy.md (Phase 3).
EXPECTED = {
    TRAIN: {"rows": 199_364, "frauds": 384, "min_time": 0, "max_time": 132_928},
    VALIDATION: {"rows": 42_722, "frauds": 56, "min_time": 132_929, "max_time": 151_328},
    TEST: {"rows": 42_721, "frauds": 52, "min_time": 151_329, "max_time": 172_792},
}
EXPECTED_TOTAL_ROWS = 284_807
EXPECTED_TOTAL_FRAUDS = 492
EXPECTED_DUPLICATE_GROUPS = 773
EXPECTED_EXCESS_DUPLICATE_ROWS = 1_081


def _frame(times, classes=None, index=None) -> pd.DataFrame:
    times = list(times)
    if classes is None:
        classes = [0] * len(times)
    return pd.DataFrame(
        {"Time": times, "V1": np.arange(len(times), dtype=float), "Class": classes},
        index=index,
    )


class TestBoundaries(unittest.TestCase):
    def test_boundary_values_are_inclusive_as_documented(self):
        df = _frame([0, TRAIN_END_TIME, TRAIN_END_TIME + 1, VALIDATION_END_TIME,
                     VALIDATION_END_TIME + 1, 172_792])
        result = assign_partitions(df).tolist()
        self.assertEqual(result, [TRAIN, TRAIN, VALIDATION, VALIDATION, TEST, TEST])

    def test_boundary_constants_match_documentation(self):
        self.assertEqual(TRAIN_END_TIME, 132_928)
        self.assertEqual(VALIDATION_END_TIME, 151_328)

    def test_integer_valued_float_times_are_accepted(self):
        df = _frame([132928.0, 132929.0])
        self.assertEqual(assign_partitions(df).tolist(), [TRAIN, VALIDATION])

    def test_times_beyond_dataset_range_go_to_test(self):
        self.assertEqual(assign_partitions(_frame([10_000_000])).tolist(), [TEST])

    def test_output_is_ordered_categorical(self):
        result = assign_partitions(_frame([0, 200_000]))
        self.assertIsInstance(result.dtype, pd.CategoricalDtype)
        self.assertEqual(list(result.cat.categories), list(PARTITION_NAMES))
        self.assertTrue(result.cat.ordered)


class TestGroupingInvariants(unittest.TestCase):
    def test_identical_timestamps_share_a_partition(self):
        times = [TRAIN_END_TIME] * 3 + [TRAIN_END_TIME + 1] * 3 + [VALIDATION_END_TIME] * 2
        df = _frame(times).sample(frac=1, random_state=0)  # scramble row order
        parts = assign_partitions(df)
        per_time = pd.DataFrame({"t": df["Time"], "p": parts}).groupby("t")["p"].nunique()
        self.assertTrue((per_time == 1).all())

    def test_duplicate_records_share_a_partition_and_are_kept(self):
        row_a = {"Time": TRAIN_END_TIME, "V1": 1.5, "Class": 0}
        row_b = {"Time": VALIDATION_END_TIME + 1, "V1": -2.0, "Class": 1}
        df = pd.DataFrame([row_a, row_b, row_a, row_b, row_a])
        parts = assign_partitions(df)
        self.assertEqual(len(parts), 5)
        self.assertEqual(count_cross_partition_duplicate_groups(df, parts), 0)
        split = split_partitions(df)
        self.assertEqual(len(split[TRAIN]), 3)
        self.assertEqual(len(split[TEST]), 2)

    def test_cross_partition_duplicate_detector_flags_leakage(self):
        df = pd.DataFrame([{"Time": 5, "V1": 1.0, "Class": 0}] * 2)
        leaky = pd.Series(pd.Categorical([TRAIN, TEST], categories=PARTITION_NAMES),
                          index=df.index)
        self.assertEqual(count_cross_partition_duplicate_groups(df, leaky), 1)

    def test_duplicate_detector_rejects_misaligned_partitions(self):
        df = _frame([0, 1])
        parts = assign_partitions(df)
        parts.index = [10, 11]
        with self.assertRaises(ValueError):
            count_cross_partition_duplicate_groups(df, parts)


class TestPreservation(unittest.TestCase):
    def setUp(self):
        self.df = _frame(
            [151_400, 0, 140_000, TRAIN_END_TIME, 160_000, 132_929],
            classes=[1, 0, 0, 1, 0, 1],
            index=["f", "a", "d", "b", "e", "c"],
        )

    def test_input_frame_is_not_modified(self):
        before = self.df.copy(deep=True)
        assign_partitions(self.df)
        split_partitions(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_result_aligned_to_original_index_and_order(self):
        parts = assign_partitions(self.df)
        self.assertTrue(parts.index.equals(self.df.index))
        self.assertEqual(parts.tolist(),
                         [TEST, TRAIN, VALIDATION, TRAIN, TEST, VALIDATION])

    def test_split_preserves_every_record_exactly_once(self):
        split = split_partitions(self.df)
        self.assertEqual(set(split), set(PARTITION_NAMES))
        recombined = pd.concat(split.values()).loc[self.df.index]
        pd.testing.assert_frame_equal(recombined, self.df)
        indices = [i for part in split.values() for i in part.index]
        self.assertEqual(sorted(indices), sorted(self.df.index))

    def test_split_keeps_original_relative_order(self):
        split = split_partitions(self.df)
        self.assertEqual(split[TRAIN].index.tolist(), ["a", "b"])
        self.assertEqual(split[VALIDATION].index.tolist(), ["d", "c"])
        self.assertEqual(split[TEST].index.tolist(), ["f", "e"])

    def test_empty_partition_returns_empty_frame_with_columns(self):
        split = split_partitions(_frame([0, 1]))
        self.assertEqual(len(split[TEST]), 0)
        self.assertEqual(list(split[TEST].columns), ["Time", "V1", "Class"])


class TestDeterminismAndLabelIndependence(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.df = _frame(rng.integers(0, 172_793, size=2_000),
                         classes=rng.integers(0, 2, size=2_000))

    def test_repeated_calls_are_identical(self):
        pd.testing.assert_series_equal(assign_partitions(self.df), assign_partitions(self.df))

    def test_assignment_does_not_depend_on_row_order(self):
        shuffled = self.df.sample(frac=1, random_state=7)
        pd.testing.assert_series_equal(
            assign_partitions(shuffled).loc[self.df.index], assign_partitions(self.df)
        )

    def test_assignment_does_not_depend_on_labels(self):
        flipped = self.df.assign(Class=1 - self.df["Class"])
        pd.testing.assert_series_equal(assign_partitions(flipped), assign_partitions(self.df))

    def test_class_column_is_not_required(self):
        result = assign_partitions(self.df.drop(columns="Class"))
        pd.testing.assert_series_equal(result, assign_partitions(self.df))


class TestInvalidInputs(unittest.TestCase):
    def test_non_dataframe_rejected(self):
        for bad in (None, [1, 2], pd.Series([1, 2]), np.array([1, 2])):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(TypeError):
                assign_partitions(bad)

    def test_missing_time_column_rejected(self):
        with self.assertRaises(KeyError):
            assign_partitions(pd.DataFrame({"V1": [1.0], "Class": [0]}))

    def test_duplicate_time_columns_rejected(self):
        df = pd.DataFrame([[0, 1]], columns=["Time", "Time"])
        with self.assertRaises(ValueError):
            assign_partitions(df)

    def test_non_numeric_time_rejected(self):
        for values in (["0", "1"], [True, False], pd.to_datetime(["2013-09-01", "2013-09-02"])):
            with self.subTest(values=values), self.assertRaises(TypeError):
                assign_partitions(pd.DataFrame({"Time": values}))

    def test_missing_or_infinite_time_rejected(self):
        for bad in (np.nan, np.inf, -np.inf, pd.NA):
            df = pd.DataFrame({"Time": pd.array([0, 1], dtype="Float64")})
            df.loc[1, "Time"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                assign_partitions(df)

    def test_negative_time_rejected(self):
        with self.assertRaises(ValueError):
            assign_partitions(_frame([0, -1]))

    def test_fractional_time_rejected(self):
        with self.assertRaises(ValueError):
            assign_partitions(_frame([TRAIN_END_TIME + 0.5]))

    def test_custom_time_column_name(self):
        df = pd.DataFrame({"seconds": [0, VALIDATION_END_TIME + 1]})
        self.assertEqual(assign_partitions(df, time_column="seconds").tolist(), [TRAIN, TEST])
        with self.assertRaises(KeyError):
            assign_partitions(df)

    def test_empty_frame_returns_empty_assignment(self):
        result = assign_partitions(pd.DataFrame({"Time": pd.Series([], dtype=float)}))
        self.assertEqual(len(result), 0)

    def test_duplicate_detector_rejects_unknown_feature_columns(self):
        df = _frame([0])
        with self.assertRaises(KeyError):
            count_cross_partition_duplicate_groups(df, assign_partitions(df), ["nope"])


@unittest.skipUnless(RAW_DATA_PATH.exists(), f"raw dataset not found at {RAW_DATA_PATH}")
class TestDocumentedSplitOnRealData(unittest.TestCase):
    """Verify the frozen Phase 3 partitions against the local raw dataset."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RAW_DATA_PATH)
        cls.partitions = assign_partitions(cls.df)
        cls.split = split_partitions(cls.df)

    def test_all_records_retained(self):
        self.assertEqual(len(self.df), EXPECTED_TOTAL_ROWS)
        self.assertEqual(int(self.df["Class"].sum()), EXPECTED_TOTAL_FRAUDS)
        self.assertEqual(sum(len(p) for p in self.split.values()), EXPECTED_TOTAL_ROWS)

    def test_exact_row_counts(self):
        for name, expected in EXPECTED.items():
            with self.subTest(partition=name):
                self.assertEqual(len(self.split[name]), expected["rows"])

    def test_exact_fraud_counts(self):
        for name, expected in EXPECTED.items():
            with self.subTest(partition=name):
                self.assertEqual(int(self.split[name]["Class"].sum()), expected["frauds"])

    def test_documented_time_ranges(self):
        for name, expected in EXPECTED.items():
            with self.subTest(partition=name):
                times = self.split[name]["Time"]
                self.assertEqual(times.min(), expected["min_time"])
                self.assertEqual(times.max(), expected["max_time"])

    def test_identical_timestamps_never_cross_partitions(self):
        per_time = (pd.DataFrame({"t": self.df["Time"], "p": self.partitions})
                    .groupby("t")["p"].nunique())
        self.assertTrue((per_time == 1).all())

    def test_duplicate_groups_retained_and_kept_together(self):
        features = [c for c in self.df.columns if c != "Class"]
        sizes = self.df.groupby(features, sort=False).size()
        self.assertEqual(int((sizes > 1).sum()), EXPECTED_DUPLICATE_GROUPS)
        self.assertEqual(int((sizes - 1).sum()), EXPECTED_EXCESS_DUPLICATE_ROWS)
        self.assertEqual(count_cross_partition_duplicate_groups(self.df, self.partitions), 0)

    def test_original_index_preserved(self):
        self.assertTrue(self.partitions.index.equals(self.df.index))
        recombined_index = np.sort(np.concatenate([p.index.to_numpy()
                                                   for p in self.split.values()]))
        np.testing.assert_array_equal(recombined_index, self.df.index.to_numpy())

    def test_deterministic_on_real_data(self):
        pd.testing.assert_series_equal(assign_partitions(self.df), self.partitions)


if __name__ == "__main__":
    unittest.main()
