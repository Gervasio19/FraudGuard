"""Tests for the duplicate-group-aware stratified random split (``src/data/random_split.py``).

Run from the repository root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.random_split import assign_stratified_group_partitions
from src.data.split import PARTITION_DTYPE, PARTITION_NAMES, TEST, TRAIN, VALIDATION, count_cross_partition_duplicate_groups

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"


def _frame(n_negative=900, n_positive=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_negative + n_positive
    return pd.DataFrame(
        {
            "Time": np.arange(n),
            "V1": rng.normal(size=n),
            "Class": [0] * n_negative + [1] * n_positive,
        }
    )


def _split(df, **kwargs):
    return assign_stratified_group_partitions(df, "Class", **kwargs)


class TestProportionsAndStratification(unittest.TestCase):
    def test_row_fractions_close_to_70_15_15(self):
        counts = _split(_frame()).value_counts()
        self.assertEqual(counts[TRAIN], 700)
        self.assertEqual(counts[VALIDATION], 150)
        self.assertEqual(counts[TEST], 150)

    def test_class_proportions_preserved_in_each_partition(self):
        df = _frame()
        partitions = _split(df)
        for name, expected_positives in [(TRAIN, 70), (VALIDATION, 15), (TEST, 15)]:
            self.assertEqual(int(df.loc[partitions == name, "Class"].sum()), expected_positives)

    def test_custom_fractions(self):
        counts = _split(_frame(), fractions=(0.5, 0.25, 0.25)).value_counts()
        self.assertEqual((counts[TRAIN], counts[VALIDATION], counts[TEST]), (500, 250, 250))

    def test_output_format_matches_chronological_split(self):
        df = _frame()
        partitions = _split(df)
        self.assertEqual(partitions.dtype, PARTITION_DTYPE)
        self.assertEqual(partitions.name, "partition")
        self.assertTrue(partitions.index.equals(df.index))
        self.assertTrue(set(partitions.unique()) <= set(PARTITION_NAMES))


class TestDuplicateGroups(unittest.TestCase):
    def _with_duplicates(self) -> pd.DataFrame:
        df = _frame()
        copies = pd.concat([df.iloc[[0, 5, 950]]] * 4)  # 3 groups, 5 copies each in total
        return pd.concat([df, copies], ignore_index=True)

    def test_duplicate_groups_never_cross_partitions(self):
        df = self._with_duplicates()
        for seed in range(20):
            partitions = _split(df, random_state=seed)
            feature_cols = [c for c in df.columns if c != "Class"]
            self.assertEqual(count_cross_partition_duplicate_groups(df, partitions, feature_cols), 0)

    def test_all_records_retained(self):
        df = self._with_duplicates()
        partitions = _split(df)
        self.assertEqual(len(partitions), len(df))
        self.assertFalse(partitions.isna().any())

    def test_conflicting_labels_in_a_group_rejected(self):
        df = _frame()
        conflict = df.iloc[[0]].assign(Class=1)
        with self.assertRaises(ValueError):
            _split(pd.concat([df, conflict], ignore_index=True))

    def test_empty_group_columns_split_individual_rows(self):
        df = self._with_duplicates()
        partitions = _split(df, group_columns=[])
        self.assertEqual(len(partitions), len(df))

    def test_custom_group_columns(self):
        df = _frame()
        df["account"] = np.arange(len(df)) // 10  # groups of 10 rows with a single label
        partitions = _split(df, group_columns=["account"])
        self.assertTrue((partitions.groupby(df["account"], observed=True).nunique() == 1).all())


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_split(self):
        df = _frame()
        pd.testing.assert_series_equal(_split(df, random_state=7), _split(df, random_state=7))

    def test_different_seed_different_split(self):
        df = _frame()
        self.assertFalse(_split(df, random_state=1).equals(_split(df, random_state=2)))

    def test_independent_of_input_row_order(self):
        df = _frame()
        shuffled = df.sample(frac=1.0, random_state=3)
        original = _split(df)
        reordered = _split(shuffled).loc[df.index]
        pd.testing.assert_series_equal(original, reordered)

    def test_input_not_modified(self):
        df = _frame()
        before = df.copy()
        _split(df)
        pd.testing.assert_frame_equal(df, before)


class TestInvalidInputs(unittest.TestCase):
    def test_non_dataframe_rejected(self):
        with self.assertRaises(TypeError):
            assign_stratified_group_partitions([1, 2, 3], "Class")

    def test_missing_target_rejected(self):
        with self.assertRaises(KeyError):
            assign_stratified_group_partitions(_frame(), "label")

    def test_missing_target_values_rejected(self):
        df = _frame().astype({"Class": float})
        df.loc[0, "Class"] = np.nan
        with self.assertRaises(ValueError):
            _split(df)

    def test_bad_fractions_rejected(self):
        for fractions in [(0.7, 0.3), (0.7, 0.2, 0.2), (0.8, 0.2, 0.0), (1.2, -0.1, -0.1)]:
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                _split(_frame(), fractions=fractions)

    def test_target_as_group_column_rejected(self):
        with self.assertRaises(ValueError):
            _split(_frame(), group_columns=["V1", "Class"])

    def test_unknown_group_column_rejected(self):
        with self.assertRaises(KeyError):
            _split(_frame(), group_columns=["missing"])


@unittest.skipUnless(RAW_DATA_PATH.exists(), f"raw dataset not found at {RAW_DATA_PATH}")
class TestRandomSplitOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RAW_DATA_PATH)
        cls.partitions = _split(cls.df, random_state=42)

    def test_all_records_retained(self):
        self.assertEqual(len(self.partitions), 284_807)

    def test_fraud_rate_preserved(self):
        overall = self.df["Class"].mean()
        for name in PARTITION_NAMES:
            rate = self.df.loc[self.partitions == name, "Class"].mean()
            self.assertAlmostEqual(rate, overall, delta=0.0001, msg=name)

    def test_no_duplicate_group_crosses_partitions(self):
        feature_cols = [c for c in self.df.columns if c != "Class"]
        self.assertEqual(count_cross_partition_duplicate_groups(self.df, self.partitions, feature_cols), 0)

    def test_documented_seed_42_partition_counts(self):
        # Recorded in docs/WORK_LOG.md (Phase 4b).
        expected = {TRAIN: (199_364, 344), VALIDATION: (42_722, 74), TEST: (42_721, 74)}
        for name, (rows, frauds) in expected.items():
            mask = self.partitions == name
            self.assertEqual((int(mask.sum()), int(self.df.loc[mask, "Class"].sum())), (rows, frauds))


if __name__ == "__main__":
    unittest.main()
