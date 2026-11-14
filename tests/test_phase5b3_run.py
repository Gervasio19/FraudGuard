"""Tests for the Phase 5B-3 runner and the practical-relevance module.

Synthetic tests use generated data only. Real-data tests read the recorded
result files and are skipped when a file is absent; they never fit a model.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import CREDITCARD, PROJECT_ROOT
from src.data.split import VALIDATION_END_TIME
from src.evaluation import calibration as cal
from src.evaluation import evidence_labels as ev
from src.evaluation.practical import fixed_k_comparison, scenario_inputs, scenario_outcomes
from src.models import phase5_grid as grid
from src.models.phase5_settings import pre_registered_configuration
from src.models.phase5b3_champion import champion_decision
from src.models.run_phase5b2_grid import compute_grid
from src.models.run_phase5b3 import (
    C_A, EVIDENCE_SECTIONS, RUNTIME_KEY, Replicates, analyse, frozen_replicates,
)
from src.reproducibility.manifest import canonical_sha256, dumps_strict
from tests.test_phase5b2_grid import synthetic_frame

RESULTS_FILE = PROJECT_ROOT / "reports" / "phase5b3_results.json"
GRID_FILE = PROJECT_ROOT / "reports" / "phase5b2_grid_results.json"
SMALL = Replicates(conditional_primary=40, conditional_sensitivity=20, t_cal=30, calibration_evaluation=30,
                   iid_continuity=20, permutations=20, e2_splits=40, e2_min_usable=10)


def without_runtime(results: dict) -> dict:
    out = copy.deepcopy(results)
    out.pop(RUNTIME_KEY)
    return out


class TestFrozenReplicates(unittest.TestCase):
    def test_real_runs_use_the_frozen_counts(self):
        r = frozen_replicates()
        self.assertEqual((r.conditional_primary, r.conditional_sensitivity, r.t_cal, r.calibration_evaluation),
                         (5_000, 2_000, 2_000, 5_000))
        self.assertIsNone(r.e2_splits)  # E2 falls back to the frozen 2,000 / 1,000


class TestSyntheticEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_frame(n=9_000, seed=1)
        cls.grid_run = compute_grid(cls.df, CREDITCARD, check_hgb_determinism=False)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = analyse(cls.grid_run, CREDITCARD, SMALL, figures_dir=Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_section_has_an_evidence_class_and_scenarios_are_disclosed(self):
        self.assertEqual(ev.missing_evidence_classes(self.results, EVIDENCE_SECTIONS), [])
        self.assertEqual(ev.scenario_violations(self.results), [])
        for name in ("scenario_inputs", "scenario_outcomes", "calibration_threshold_propagation"):
            self.assertTrue(self.results[name]["hypothetical"])
            self.assertEqual(self.results[name]["disclosure"], ev.HYPOTHETICAL_DISCLOSURE)
        for name in ("calibration", "e1_conditional_bootstrap", "e2_selection_aware", "champion_decision",
                     "fixed_k_comparison"):
            self.assertEqual(self.results[name][ev.EVIDENCE_KEY], ev.MODEL_GENERATED)
        dumps_strict(self.results)  # strict JSON, no arrays left in the result

    def test_champion_follows_the_two_gates_only(self):
        decision = self.results["champion_decision"]
        e1 = self.results["e1_conditional_bootstrap"]["gate"]
        e2 = self.results["e2_selection_aware"]["gate"]
        expected = champion_decision(e1, e2)
        self.assertEqual(decision["champion"], expected["champion"])
        self.assertEqual(decision["champion"], grid.HGB if (e1["passed"] and e2["passed"]) else grid.LR)
        self.assertEqual(decision["inputs_used"], ["E1 gate record", "E2 gate record"])

    def test_e1_structure(self):
        e1 = self.results["e1_conditional_bootstrap"]
        self.assertEqual(set(e1["schemes"]), {"cbb", "mbb", "stationary"})
        for scheme in e1["schemes"].values():
            self.assertEqual(set(scheme["by_block_minutes"]), {"5", "10", "15", "30"})
        self.assertEqual(e1["schemes"]["cbb"]["role"], "primary")
        self.assertEqual(e1["gate"]["envelope_block_minutes"], [10, 15, 30])
        row = e1["schemes"]["cbb"]["by_block_minutes"]["15"]
        self.assertEqual([r["k"] for r in row["delta_at_k_hgb_minus_lr"]], [25, 50, 100, 250, 500])

    def test_calibration_preserves_the_ranking_and_uses_t_cal_only(self):
        calibration = self.results["calibration"]
        for label in ("LR", "HGB"):
            calibrator = calibration["calibrators"][label][C_A]
            self.assertTrue(calibrator["calibrated"])
            self.assertGreater(calibrator["A"], 0)
            preserved = calibration["evaluation"][label][C_A]["ranking_preserved"]
            self.assertAlmostEqual(preserved["average_precision_uncalibrated_score"],
                                   preserved["average_precision_calibrated_logit"], places=12)
            equivalence = calibration["public_api_equivalence"][label]
            self.assertTrue(equivalence["equivalent_within_1e-6_logit"])
            self.assertTrue(equivalence["own_objective_not_worse"])
            self.assertEqual(calibrator["n_rows"], len(self.grid_run["cal_rows"]))
        self.assertIn("NOT a champion gate", calibration["role"])

    def test_calibrators_and_k_edges_are_functions_of_label_free_or_t_cal_inputs_only(self):
        # Recomputed here from T_cal scores + T_cal labels (calibrator) and from validation scores
        # alone (K-derived edges): no validation label is an input of either.
        run, calibration = self.grid_run, self.results["calibration"]
        y_cal = (run["cal_rows"]["Class"] == 1).to_numpy()
        validation = run["validation"]
        ks = pre_registered_configuration()["operations"]["fixed_k"]
        for family, label in ((grid.LR, "LR"), (grid.HGB, "HGB")):
            fitted = run["fitted"][self.results["selection_continuity"]["selected"][family]]
            expected = cal.primary_calibrator(fitted["scores"]["t_cal"]["decision_function"], y_cal, True)
            self.assertEqual(calibration["calibrators"][label][C_A], expected)
            edges = cal.k_derived_score_edges(fitted["scores"]["validation"]["decision_function"],
                                              validation["Time"].to_numpy(), validation.index.to_numpy(), ks)
            self.assertEqual(calibration["k_derived_edges"]["edges"][label], edges)
        self.assertIn("before validation labels", calibration["k_derived_edges"]["timing"])

    def test_test_rows_cannot_change_any_result(self):
        tampered = self.df.copy()
        is_test = tampered["Time"] > VALIDATION_END_TIME
        self.assertGreater(int(is_test.sum()), 0)
        tampered.loc[is_test, "Class"] = 1 - tampered.loc[is_test, "Class"]
        tampered.loc[is_test, [f"V{i}" for i in range(1, 29)] + ["Amount"]] = 1e6
        other = analyse(compute_grid(tampered, CREDITCARD, check_hgb_determinism=False), CREDITCARD, SMALL)
        self.assertEqual(dumps_strict(without_runtime(other)),
                         dumps_strict(without_runtime({**self.results, "figures": []})))
        self.assertLessEqual(self.results["test_isolation"]["max_time_in_processed_rows"], VALIDATION_END_TIME)

    def test_rerun_is_identical_except_runtime(self):
        again = analyse(self.grid_run, CREDITCARD, SMALL)
        self.assertEqual(dumps_strict(without_runtime(again)),
                         dumps_strict(without_runtime({**self.results, "figures": []})))

    def test_figures_are_written_and_labelled(self):
        files = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.assertEqual(files, ["phase5b3_fixed_k_scenarios.png", "phase5b3_reliability_hgb.png",
                                 "phase5b3_reliability_lr.png"])
        scenario = next(f for f in self.results["figures"] if "scenarios" in f["file"])
        self.assertTrue(scenario["hypothetical"])
        self.assertEqual(scenario[ev.EVIDENCE_KEY], ev.SIMULATED_OUTCOME)

    def test_a_mismatch_with_the_recorded_grid_stops_the_run(self):
        recorded = {"grid_table_track_b": [{"config_id": cid, "average_precision": 0.5}
                                           for cid in self.grid_run["fitted"]],
                    "step3_selection_record": {f: {"selected_config_id": "X"} for f in grid.FAMILIES}}
        with self.assertRaises(AssertionError):
            analyse(self.grid_run, CREDITCARD, SMALL, recorded_grid=recorded)


class TestPracticalRelevance(unittest.TestCase):
    def setUp(self):
        # 12 rows, 4 frauds. Baseline ranks frauds at positions 1, 2, 5, 9; candidate at 1, 2, 3, 12.
        y = np.zeros(12, int)
        baseline, candidate = np.arange(12, 0, -1.0), np.arange(12, 0, -1.0)
        frauds_baseline, frauds_candidate = [0, 1, 4, 8], [0, 1, 2, 11]
        y[frauds_baseline] = 1
        candidate_scores = np.empty(12)
        legit = [i for i in range(12) if i not in frauds_baseline]
        order = [None] * 12
        for rank, row in zip(frauds_candidate, frauds_baseline):
            order[rank] = row
        rest = iter(legit)
        order = [row if row is not None else next(rest) for row in order]
        candidate_scores[order] = candidate
        self.comparison = fixed_k_comparison(y, {"base": baseline, "cand": candidate_scores}, np.arange(12),
                                             np.arange(12), [2, 4, 8], "base", "cand", 1)
        self.outcomes = scenario_outcomes(self.comparison, [5, 50], [5, 10, 15], 2.0)

    def test_fixed_k_comparison_is_model_generated(self):
        self.assertEqual(self.comparison[ev.EVIDENCE_KEY], ev.MODEL_GENERATED)
        rows = self.comparison["equal_alert_volume_comparison"]
        self.assertEqual([r["tp_baseline"] for r in rows], [2, 2, 3])
        self.assertEqual([r["tp_candidate"] for r in rows], [2, 3, 3])
        self.assertEqual([r["delta_tp_candidate_minus_baseline"] for r in rows], [0, 1, 0])
        self.assertTrue(all(r["delta_alerts"] == 0 for r in rows))

    def test_break_even_ratios(self):
        base = self.outcomes["per_model"]["base"]
        self.assertEqual([r["break_even_R"] for r in base], [1.0, 2.0, 8 / 3])  # K / TP = 1 / precision
        self.assertIsNone(base[0]["marginal_break_even_R"])
        self.assertFalse(base[1]["marginal_break_even_R"]["defined"])  # no additional fraud from K=2 to 4
        self.assertIsNone(base[1]["marginal_break_even_R"]["R_star"])
        self.assertEqual(base[2]["marginal_break_even_R"]["R_star"], 4.0)  # 4 more alerts for 1 more fraud
        between = self.outcomes["between_models_at_equal_k"]
        self.assertEqual([b["defined"] for b in between], [False, True, False])
        self.assertEqual(between[1]["R_star"], 0.0)
        self.assertEqual(between[1]["cheaper_for_every_positive_R"], "cand")
        self.assertIsNone(between[0]["R_star"])

    def test_cost_sensitivity_and_analyst_hours(self):
        row = self.outcomes["per_model"]["cand"][1]  # K = 4, TP = 3, FN = 1
        by_r = {r["R"]: r for r in row["by_R"]}
        self.assertEqual(by_r[5.0]["cost_in_review_units"], 4 + 5 * 1)
        self.assertEqual(by_r[5.0]["net_saving_vs_no_review_in_review_units"], 5 * 3 - 4)
        self.assertEqual(by_r[50.0]["no_review_cost_in_review_units"], 200)
        self.assertEqual(row["analyst_hours"], {"5_min": 4 * 5 / 60, "10_min": 4 * 10 / 60, "15_min": 1.0})
        self.assertEqual(row["analyst_hours_per_fraud_caught"]["15_min"], 1.0 / 3)
        self.assertEqual(row["alerts_per_hour_of_traffic"], 2.0)
        difference = self.outcomes["between_models_at_equal_k"][1]["net_saving_difference_by_R"]
        self.assertEqual([d["candidate_minus_baseline_in_review_units"] for d in difference], [5.0, 50.0])

    def test_scenario_blocks_are_hypothetical(self):
        inputs = scenario_inputs([5, 50], [5, 10, 15], "test")
        self.assertEqual(inputs[ev.EVIDENCE_KEY], ev.ASSUMED_SCENARIO_INPUT)
        self.assertEqual(self.outcomes[ev.EVIDENCE_KEY], ev.SIMULATED_OUTCOME)
        for block in (inputs, self.outcomes):
            self.assertTrue(block["hypothetical"])
            self.assertEqual(block["disclosure"], ev.HYPOTHETICAL_DISCLOSURE)
        self.assertEqual(ev.scenario_violations({"a": inputs, "b": self.outcomes}), [])


@unittest.skipUnless(RESULTS_FILE.is_file(), "recorded Phase 5B-3 results are absent")
class TestRecordedResults(unittest.TestCase):
    """Regression checks of the recorded real-data results (docs/reports/PHASE_5B_3_REPORT.md)."""

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RESULTS_FILE.read_text())

    def test_counts_and_test_isolation(self):
        data = self.r["data"]
        self.assertEqual((data["t_fit"]["n_rows"], data["t_fit"]["n_positive"]), (159_491, 356))
        self.assertEqual((data["t_cal"]["n_rows"], data["t_cal"]["n_positive"]), (39_873, 28))
        self.assertEqual((data["validation"]["n_rows"], data["validation"]["n_positive"]), (42_722, 56))
        self.assertEqual(self.r["test_isolation"]["test_rows_removed_before_label_access"], 42_721)
        self.assertEqual(self.r["test_isolation"]["max_time_in_processed_rows"], 151_328)

    def test_frozen_configuration_hash_is_unchanged(self):
        current = canonical_sha256(pre_registered_configuration())
        self.assertEqual(self.r["provenance"]["pre_registered_configuration_sha256"], current)
        if GRID_FILE.is_file():
            recorded = json.loads(GRID_FILE.read_text())["provenance"]["pre_registered_configuration_sha256"]
            self.assertEqual(current, recorded)

    def test_selected_pair_reproduces_phase5b2(self):
        continuity = self.r["selection_continuity"]
        self.assertEqual(continuity["selected"], {grid.LR: "LR-10", grid.HGB: "HGB-14"})
        self.assertTrue(continuity["recorded_phase5b2_grid"]["all_28_ap_identical"])
        self.assertTrue(continuity["recorded_phase5b2_grid"]["selection_identical"])

    def test_gates_and_decision_are_consistent(self):
        e1, e2 = self.r["e1_conditional_bootstrap"], self.r["e2_selection_aware"]
        self.assertEqual((e2["splits"]["n_attempted"], e2["splits"]["n_usable"]), (2_000, 1_984))
        cbb = e1["schemes"]["cbb"]
        self.assertEqual((cbb["n_replicates"], cbb["seed"]), (5_000, 0))
        for minutes in ("10", "15", "30"):
            lower = cbb["by_block_minutes"][minutes]["delta_ap_hgb_minus_lr"]["lower"]
            self.assertEqual(e1["gate"]["by_block_minutes"][minutes]["lower_bound"], lower)
            self.assertEqual(e1["gate"]["by_block_minutes"][minutes]["above_zero"], lower > 0)
        gate = e2["gate"]
        self.assertEqual(gate["frequency"], e2["heldout_delta_ap"]["share_positive"])
        self.assertEqual(gate["passed"], gate["frequency"] >= 0.90 and gate["median_heldout_delta_ap"] > 0)
        decision = self.r["champion_decision"]
        self.assertEqual(decision["champion"], champion_decision(e1["gate"], gate)["champion"])
        self.assertEqual(decision["champion_config_id"],
                         "HGB-14" if (e1["gate"]["passed"] and gate["passed"]) else "LR-10")

    def test_calibrators_and_disclosure(self):
        calibration = self.r["calibration"]
        for label in ("LR", "HGB"):
            calibrator = calibration["calibrators"][label][C_A]
            self.assertEqual((calibrator["n_rows"], calibrator["n_positive"]), (39_873, 28))
            if calibrator["calibrated"]:
                self.assertGreater(calibrator["A"], 0)
                equivalence = calibration["public_api_equivalence"][label]
                # a difference is documented, never silently accepted: it must be small and the own fit
                # must be at least as close to the optimum of the shared objective
                self.assertTrue(equivalence["max_abs_probability_difference_below_1e-6"])
                self.assertTrue(equivalence["own_objective_not_worse"])
                # gradient summed over the 39,873 T_cal rows
                self.assertLess(equivalence["platt_objective_on_t_cal"]["own_gradient_inf_norm"], 1e-6)
                if not equivalence["equivalent_within_1e-6_logit"]:
                    self.assertIn("NOT identical", equivalence["explanation"])
        ev.require_disclosed(self.r, EVIDENCE_SECTIONS)
        self.assertEqual(self.r["calibration"]["evaluation_bootstrap_B"]["n_replicates"], 5_000)


if __name__ == "__main__":
    unittest.main()
