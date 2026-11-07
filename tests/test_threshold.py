"""Unit tests for threshold analysis (``src/evaluation/threshold.py``) and
temporal windows (``src/data/temporal_windows.py``).

Run from the repository root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.data.temporal_windows import WINDOW_1, WINDOW_2, split_two_time_windows
from src.evaluation.threshold import (
    CURVE_COLUMNS,
    evaluate_threshold,
    select_max_f1_threshold,
    threshold_candidates,
    threshold_curve,
)

# Hand-calculated example (see test_hand_calculated_curve):
Y_TRUE = [0, 0, 1, 1, 0, 1]
Y_SCORE = [0.1, 0.4, 0.35, 0.8, 0.8, 0.9]


def _row(curve, i):
    return {c: curve[c][i] for c in CURVE_COLUMNS}


class TestCandidates(unittest.TestCase):
    def test_unique_sorted_scores(self):
        np.testing.assert_array_equal(threshold_candidates(Y_SCORE), [0.1, 0.35, 0.4, 0.8, 0.9])

    def test_curve_uses_one_row_per_unique_score(self):
        curve = threshold_curve(Y_TRUE, Y_SCORE)
        self.assertEqual(curve["threshold"], [0.1, 0.35, 0.4, 0.8, 0.9])
        self.assertEqual(set(curve), set(CURVE_COLUMNS))
        self.assertTrue(all(len(curve[c]) == 5 for c in CURVE_COLUMNS))


class TestMetrics(unittest.TestCase):
    def test_hand_calculated_curve(self):
        curve = threshold_curve(Y_TRUE, Y_SCORE)
        # threshold: (tp, fp, tn, fn, precision, recall, f1)
        expected = {
            0.1: (3, 3, 0, 0, 0.5, 1.0, 6 / 9),
            0.35: (3, 2, 1, 0, 0.6, 1.0, 6 / 8),
            0.4: (2, 2, 1, 1, 0.5, 2 / 3, 4 / 7),
            0.8: (2, 1, 2, 1, 2 / 3, 2 / 3, 4 / 6),
            0.9: (1, 0, 3, 2, 1.0, 1 / 3, 2 / 4),
        }
        for i, t in enumerate(curve["threshold"]):
            tp, fp, tn, fn, p, r, f1 = expected[t]
            row = _row(curve, i)
            self.assertEqual((row["tp"], row["fp"], row["tn"], row["fn"]), (tp, fp, tn, fn), t)
            self.assertAlmostEqual(row["precision"], p)
            self.assertAlmostEqual(row["recall"], r)
            self.assertAlmostEqual(row["f1"], f1)
            self.assertAlmostEqual(row["f1"], 2 * p * r / (p + r))  # matches the P/R formula
            self.assertEqual(row["n_predicted_positive"], tp + fp)
            self.assertAlmostEqual(row["alert_rate"], (tp + fp) / 6)

    def test_rule_is_inclusive(self):
        # At t=0.8 both rows scored exactly 0.8 (one fraud, one not) are flagged.
        m = evaluate_threshold(Y_TRUE, Y_SCORE, 0.8)
        self.assertEqual((m["tp"], m["fp"]), (2, 1))
        m = evaluate_threshold([1], [0.5], 0.5)
        self.assertEqual(m["tp"], 1)

    def test_evaluate_threshold_matches_curve_rows(self):
        curve = threshold_curve(Y_TRUE, Y_SCORE)
        for i, t in enumerate(curve["threshold"]):
            self.assertEqual(evaluate_threshold(Y_TRUE, Y_SCORE, t), _row(curve, i))

    def test_threshold_between_candidates(self):
        m = evaluate_threshold(Y_TRUE, Y_SCORE, 0.5)  # same rows as t=0.8
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (2, 1, 2, 1))
        self.assertEqual(m["threshold"], 0.5)

    def test_confusion_counts_sum_to_rows(self):
        rng = np.random.default_rng(0)
        y, s = rng.integers(0, 2, 500), rng.random(500)
        curve = threshold_curve(y, s)
        totals = np.add.reduce([np.asarray(curve[c]) for c in ("tp", "fp", "tn", "fn")])
        self.assertTrue((totals == 500).all())
        # Brute-force comparison for a sample of thresholds.
        for i in range(0, len(curve["threshold"]), 37):
            t = curve["threshold"][i]
            pred = s >= t
            self.assertEqual(curve["tp"][i], int((pred & (y == 1)).sum()))
            self.assertEqual(curve["fp"][i], int((pred & (y == 0)).sum()))

    def test_custom_positive_label(self):
        m = evaluate_threshold(["ok", "fraud", "fraud"], [0.2, 0.7, 0.9], 0.7, positive_label="fraud")
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 0, 0))


class TestUndefinedValues(unittest.TestCase):
    def test_no_predicted_positives(self):
        m = evaluate_threshold([0, 1], [0.2, 0.3], 0.9)
        self.assertTrue(math.isnan(m["precision"]))
        self.assertEqual(m["recall"], 0.0)
        self.assertTrue(math.isnan(m["f1"]))
        self.assertEqual(m["alert_rate"], 0.0)

    def test_no_positives(self):
        curve = threshold_curve([0, 0, 0], [0.1, 0.5, 0.9])
        self.assertTrue(all(math.isnan(v) for v in curve["recall"]))
        self.assertTrue(all(math.isnan(v) for v in curve["f1"]))
        self.assertEqual(curve["precision"], [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            select_max_f1_threshold(curve)

    def test_zero_tp_gives_zero_f1(self):
        m = evaluate_threshold([1, 0], [0.1, 0.9], 0.5)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))


class TestSelection(unittest.TestCase):
    def test_hand_calculated_selection(self):
        selected = select_max_f1_threshold(threshold_curve(Y_TRUE, Y_SCORE))
        self.assertEqual(selected["threshold"], 0.35)
        self.assertAlmostEqual(selected["f1"], 0.75)

    def test_ties_broken_by_lowest_threshold(self):
        # t=0.4 → TP2 FP2 FN0 (F1=4/6); t=0.9 → TP1 FP0 FN1 (F1=2/3): equal F1.
        curve = threshold_curve([1, 0, 0, 1], [0.9, 0.6, 0.5, 0.4])
        f1 = dict(zip(curve["threshold"], curve["f1"]))
        self.assertEqual(f1[0.4], f1[0.9])
        self.assertEqual(select_max_f1_threshold(curve)["threshold"], 0.4)

    def test_tie_break_independent_of_curve_order(self):
        curve = {c: [] for c in CURVE_COLUMNS}
        for t, f1 in [(0.9, 0.5), (0.2, 0.5), (0.5, 0.4), (0.7, float("nan"))]:
            for c in CURVE_COLUMNS:
                curve[c].append({"threshold": t, "f1": f1}.get(c, 0))
        self.assertEqual(select_max_f1_threshold(curve)["threshold"], 0.2)

    def test_empty_curve_rejected(self):
        with self.assertRaises(ValueError):
            select_max_f1_threshold({c: [] for c in CURVE_COLUMNS})


class TestInputValidation(unittest.TestCase):
    FUNCTIONS = (
        lambda y, s: threshold_curve(y, s),
        lambda y, s: evaluate_threshold(y, s, 0.5),
    )

    def _assert_all_raise(self, y, s, error):
        for fn in self.FUNCTIONS:
            with self.assertRaises(error):
                fn(y, s)

    def test_empty_inputs_rejected(self):
        self._assert_all_raise([], [], ValueError)
        with self.assertRaises(ValueError):
            threshold_candidates([])

    def test_non_finite_scores_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            self._assert_all_raise([0, 1], [0.5, bad], ValueError)

    def test_scores_outside_unit_interval_rejected(self):
        self._assert_all_raise([0, 1], [0.5, 1.0000001], ValueError)
        self._assert_all_raise([0, 1], [-0.1, 0.5], ValueError)

    def test_length_mismatch_rejected(self):
        self._assert_all_raise([0, 1, 1], [0.5, 0.6], ValueError)

    def test_non_numeric_scores_rejected(self):
        self._assert_all_raise([0, 1], ["a", "b"], TypeError)
        self._assert_all_raise([0, 1], [True, False], TypeError)

    def test_missing_labels_rejected(self):
        self._assert_all_raise([0.0, np.nan], [0.1, 0.2], ValueError)

    def test_non_finite_threshold_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_threshold([0, 1], [0.1, 0.2], np.nan)


class TestTemporalWindows(unittest.TestCase):
    def test_balanced_boundary_and_chronological_order(self):
        times = pd.Series([0, 0, 1, 1, 1, 2, 3, 3])
        w = split_two_time_windows(times)
        # Boundaries after 0/1/2 give window-1 sizes 2/5/6 → imbalance 4/2/4.
        self.assertEqual((w.window_1_end_time, w.window_2_start_time), (1.0, 2.0))
        self.assertEqual((w.n_window_1, w.n_window_2), (5, 3))
        self.assertLess(times[w.labels == WINDOW_1].max(), times[w.labels == WINDOW_2].min())

    def test_imbalance_is_minimal_among_all_boundaries(self):
        rng = np.random.default_rng(1)
        times = pd.Series(np.sort(rng.integers(0, 50, 400)))
        w = split_two_time_windows(times)
        best = min(abs(2 * int((times <= t).sum()) - len(times)) for t in times.unique()[:-1])
        self.assertEqual(abs(w.n_window_1 - w.n_window_2), best)

    def test_ties_choose_earlier_boundary(self):
        w = split_two_time_windows(pd.Series([0, 1, 2]))  # boundaries give 1|2 and 2|1
        self.assertEqual((w.n_window_1, w.n_window_2), (1, 2))

    def test_equal_timestamps_never_cross_windows(self):
        times = pd.Series([5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 8])
        w = split_two_time_windows(times)
        self.assertTrue((w.labels.groupby(times, observed=True).nunique() == 1).all())

    def test_every_row_in_exactly_one_window_with_original_alignment(self):
        times = pd.Series([30, 10, 20, 10, 40], index=[7, 3, 9, 1, 5])
        w = split_two_time_windows(times)
        self.assertTrue(w.labels.index.equals(times.index))
        self.assertFalse(w.labels.isna().any())
        self.assertEqual(w.n_window_1 + w.n_window_2, len(times))
        # Boundaries after 10 and after 20 both leave a 1-row imbalance; the earlier wins.
        self.assertEqual(w.labels.tolist(), [WINDOW_2, WINDOW_1, WINDOW_2, WINDOW_1, WINDOW_2])

    def test_deterministic_and_input_unchanged(self):
        times = pd.Series([3, 1, 2, 2, 5, 8])
        before = times.copy()
        a, b = split_two_time_windows(times), split_two_time_windows(times)
        pd.testing.assert_series_equal(a.labels, b.labels)
        pd.testing.assert_series_equal(times, before)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(TypeError):
            split_two_time_windows([1, 2, 3])
        with self.assertRaises(TypeError):
            split_two_time_windows(pd.Series(["a", "b"]))
        with self.assertRaises(TypeError):
            split_two_time_windows(pd.Series([True, False]))
        with self.assertRaises(ValueError):
            split_two_time_windows(pd.Series([1.0, np.nan]))
        with self.assertRaises(ValueError):
            split_two_time_windows(pd.Series([4, 4, 4]))  # one distinct timestamp
        with self.assertRaises(ValueError):
            split_two_time_windows(pd.Series([], dtype=float))


if __name__ == "__main__":
    unittest.main()
