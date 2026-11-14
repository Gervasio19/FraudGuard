"""Mandatory hypothetical-scenario disclosure (project-wide requirement, mentor decision 2026-11-14).

Any business or operational scenario with assumed parameters must be identified
as hypothetical in every artifact: result files, documentation, reports,
notebooks and figures. These tests enforce it mechanically:

* result blocks carry one of four evidence classes, and scenario content must
  sit inside a hypothetical block with the disclosure text;
* every documentation section that uses a scenario term must also say
  "hypothetical" (or carry the protocol's ``[H]`` mark);
* the scenario figure states the hypothetical label in its title.

Chronological logs and historical reports are not rewritten: the work log is
checked from the Phase 5B-3 entry onward, and reports written before this
requirement are listed in ``HISTORICAL`` and left as they are.
"""

from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.evaluation import evidence_labels as ev  # noqa: E402
from src.models.phase5_settings import pre_registered_configuration  # noqa: E402
from src.reproducibility.manifest import canonical_sha256  # noqa: E402
from src.visualization.phase5b3_figures import fixed_k_scenario_figure  # noqa: E402

DOCS = PROJECT_ROOT / "docs"
#: Living documents: every section is checked.
CHECKED_DOCUMENTS = (
    PROJECT_ROOT / "README.md",
    DOCS / "PROJECT_CONTEXT.md",
    DOCS / "TECHNICAL_DOCUMENTATION.md",
    DOCS / "business_understanding.md",
    DOCS / "FINAL_REPORT.md",
    DOCS / "reports" / "PHASE_5B_3_REPORT.md",
)
#: Written before the requirement and preserved unchanged (mentor: "preserve historical reports").
HISTORICAL = (
    DOCS / "phase_5a_experimental_protocol.md",  # uses its own [H] evidence label
    DOCS / "reports" / "PHASE_5B_0_COMPLETION_REPORT.md",
    DOCS / "reports" / "PHASE_5B_1_REPORT.md",
    DOCS / "reports" / "PHASE_5B_2_REPORT.md",
    DOCS / "reports" / "phase4_model_development.md",
)
RESULTS_FILE = PROJECT_ROOT / "reports" / "phase5b3_results.json"
FROZEN_CONFIGURATION_SHA256_PREFIX = "86ab5adb"


class TestEvidenceLabels(unittest.TestCase):
    def test_four_classes_and_two_are_hypothetical(self):
        self.assertEqual(ev.EVIDENCE_CLASSES, ("observed_dataset", "model_generated", "assumed_scenario_input",
                                               "simulated_outcome"))
        self.assertEqual(ev.HYPOTHETICAL_CLASSES, ("assumed_scenario_input", "simulated_outcome"))

    def test_labelled_blocks(self):
        observed = ev.labelled(ev.OBSERVED_DATASET, {"n_rows": 3})
        self.assertEqual(observed, {"evidence_class": "observed_dataset", "n_rows": 3})
        simulated = ev.labelled(ev.SIMULATED_OUTCOME, {"analyst_hours": 2.0})
        self.assertTrue(simulated["hypothetical"])
        self.assertEqual(simulated["disclosure"], ev.HYPOTHETICAL_DISCLOSURE)
        self.assertIn("not observed real-world facts", ev.HYPOTHETICAL_DISCLOSURE)
        self.assertIn("actual business outcome", ev.HYPOTHETICAL_DISCLOSURE)
        with self.assertRaises(ValueError):
            ev.labelled("business_fact", {})
        with self.assertRaises(ValueError):
            ev.labelled(ev.SIMULATED_OUTCOME, {"hypothetical": False})

    def test_scenario_content_outside_a_hypothetical_block_is_a_violation(self):
        good = {"a": ev.labelled(ev.MODEL_GENERATED, {"tp": 3}),
                "b": ev.labelled(ev.SIMULATED_OUTCOME, {"rows": [{"analyst_hours": 1.0, "break_even_R": 2.0}]})}
        self.assertEqual(ev.scenario_violations(good), [])
        for bad in (
            {"a": ev.labelled(ev.MODEL_GENERATED, {"analyst_hours": 1.0})},  # scenario under a model class
            {"a": {"review_cost_ratio_R": [5, 10]}},  # no class at all
            {"a": {"evidence_class": "simulated_outcome", "analyst_hours": 1.0}},  # disclosure missing
            {"a": {"evidence_class": "simulated_outcome", "hypothetical": True, "disclosure": "trust me"}},
            {"a": {"evidence_class": "real_saving"}},
        ):
            self.assertTrue(ev.scenario_violations(bad), bad)
        nested = {"a": ev.labelled(ev.SIMULATED_OUTCOME, {
            "inner": ev.labelled(ev.MODEL_GENERATED, {"net_saving_vs_no_review_in_review_units": 4})})}
        self.assertTrue(ev.scenario_violations(nested))  # the nearest class decides

    def test_require_disclosed_refuses_unlabelled_sections(self):
        result = {"x": ev.labelled(ev.MODEL_GENERATED, {"ap": 0.8}), "y": {"ap": 0.7}}
        ev.require_disclosed(result, ["x"])
        with self.assertRaises(ValueError):
            ev.require_disclosed(result, ["x", "y"])
        with self.assertRaises(ValueError):
            ev.require_disclosed({"x": ev.labelled(ev.MODEL_GENERATED, {"sla_hours": [1, 2]})}, ["x"])

    def test_markdown_section_check(self):
        text = ("intro about models\n"
                "# Costs\nAt R = 20 the break-even point is hypothetical and assumed.\n"
                "## Staffing\nTen analyst-hours would be saved.\n"
                "## Metrics\nAverage precision is 0.85.\n"
                "## Queue\nCapacity of 10 reviews per hour [H].\n")
        self.assertEqual(ev.undisclosed_sections(text), ["## Staffing"])
        self.assertEqual(ev.undisclosed_sections(text, lambda h: "Costs" in h), [])
        self.assertEqual(ev.undisclosed_sections("The review-cost ratio is 20.\n# Other\nnothing"),
                         ["(text before the first heading)"])


class TestDocumentation(unittest.TestCase):
    def test_living_documents_disclose_every_scenario_section(self):
        for path in CHECKED_DOCUMENTS:
            with self.subTest(document=str(path.relative_to(PROJECT_ROOT))):
                self.assertTrue(path.is_file(), f"{path} is missing")
                self.assertEqual(ev.undisclosed_sections(path.read_text()), [])

    def test_work_log_from_phase_5b3_onward(self):
        text = (DOCS / "WORK_LOG.md").read_text()
        self.assertIn("5B-3", text)
        self.assertEqual(ev.undisclosed_sections(text, lambda h: "5B-3" in h or h == "## Current status"), [])

    def test_documents_state_the_requirement(self):
        for path in (PROJECT_ROOT / "README.md", DOCS / "PROJECT_CONTEXT.md", DOCS / "TECHNICAL_DOCUMENTATION.md",
                     DOCS / "business_understanding.md", DOCS / "FINAL_REPORT.md",
                     DOCS / "reports" / "PHASE_5B_3_REPORT.md"):
            with self.subTest(document=path.name):
                text = path.read_text()
                self.assertIn("hypothetical", text.lower())
                for evidence_class in ev.EVIDENCE_CLASSES:
                    self.assertIn(evidence_class, text, f"{path.name} does not name {evidence_class}")

    def test_notebooks_disclose_every_scenario_cell(self):
        notebooks = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(notebooks)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                self.assertEqual(ev.undisclosed_sections(ev.notebook_markdown(path)), [])

    def test_historical_documents_are_listed_not_silently_skipped(self):
        for path in HISTORICAL:
            self.assertTrue(path.is_file(), f"{path} is listed as historical but missing")
        self.assertFalse(set(HISTORICAL) & set(CHECKED_DOCUMENTS))


class TestFiguresAndResults(unittest.TestCase):
    def test_scenario_figure_states_the_hypothetical_label(self):
        fig = fixed_k_scenario_figure({"LR": [1, 2], "HGB": [1, 3]}, [25, 50], 4, [5, 10, 15], "synthetic")
        try:
            self.assertIn(ev.HYPOTHETICAL_FIGURE_LABEL, fig._suptitle.get_text())
            titles = " ".join(ax.get_title() for ax in fig.axes)
            self.assertIn("MODEL-GENERATED", titles)
            self.assertIn("SIMULATED OUTCOME", titles)
            self.assertIn("ASSUMED", " ".join(t.get_text() for t in fig.axes[1].get_legend().get_texts()))
            self.assertIn("SIMULATED", fig.axes[1].get_ylabel())
        finally:
            plt.close(fig)

    @unittest.skipUnless(RESULTS_FILE.is_file(), "recorded Phase 5B-3 results are absent")
    def test_recorded_results_are_classified_and_disclosed(self):
        from src.models.run_phase5b3 import EVIDENCE_SECTIONS
        results = json.loads(RESULTS_FILE.read_text())
        ev.require_disclosed(results, EVIDENCE_SECTIONS)
        self.assertEqual(results["scenario_inputs"]["evidence_class"], ev.ASSUMED_SCENARIO_INPUT)
        self.assertEqual(results["scenario_outcomes"]["evidence_class"], ev.SIMULATED_OUTCOME)
        for figure in results["figures"]:
            self.assertIn(figure["evidence_class"], ev.EVIDENCE_CLASSES)
            self.assertTrue((PROJECT_ROOT / figure["file"]).is_file(), figure["file"])
            if figure["evidence_class"] in ev.HYPOTHETICAL_CLASSES:
                self.assertEqual(figure["disclosure"], ev.HYPOTHETICAL_DISCLOSURE)

    def test_labels_did_not_change_the_frozen_configuration(self):
        settings = pre_registered_configuration()
        self.assertTrue(canonical_sha256(settings).startswith(FROZEN_CONFIGURATION_SHA256_PREFIX))
        self.assertNotIn("evidence_class", json.dumps(settings))
        self.assertNotIn("hypothetical", json.dumps(settings).lower())


if __name__ == "__main__":
    unittest.main()
