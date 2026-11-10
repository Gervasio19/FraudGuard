"""Tests for dataset integrity, manifests, result-size policy and path portability (Phase 5B-0).

The dataset check is exercised on temporary files only. No test here reads
the real dataset, fits a model or scores anything.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import CREDITCARD, PROJECT_ROOT
from src.models.phase5_settings import pre_registered_configuration
from src.reproducibility import dataset, manifest

PERSONAL_PATH_MARKERS = ("/" + "Users/", "/" + "home/", "C:" + "\\Users")


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


class TestDatasetCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sha256_of_known_bytes(self):
        path = _write(self.dir / "x.csv", b"abc")
        self.assertEqual(dataset.sha256_file(path),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_missing_mismatch_and_ok(self):
        missing = dataset.verify_dataset(self.dir / "creditcard.csv")
        self.assertEqual(missing["status"], "missing")
        self.assertIsNone(missing["observed_sha256"])
        path = _write(self.dir / "creditcard.csv", b"Time,Class\n0,0\n")
        self.assertEqual(dataset.verify_dataset(path)["status"], "mismatch")
        with mock.patch.object(dataset, "EXPECTED_SHA256", dataset.sha256_file(path)):
            self.assertEqual(dataset.verify_dataset(path)["status"], "ok")

    def test_cli_exit_codes(self):
        path = self.dir / "creditcard.csv"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dataset.main(["--data", str(path)]), dataset.MISSING)
        self.assertIn("data/raw/creditcard.csv", out.getvalue())
        _write(path, b"changed")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dataset.main(["--data", str(path)]), dataset.MISMATCH)
        with mock.patch.object(dataset, "EXPECTED_SHA256", dataset.sha256_file(path)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dataset.main(["--data", str(path)]), dataset.OK)

    def test_record_never_contains_absolute_path(self):
        record = dataset.verify_dataset(self.dir / "creditcard.csv")
        self.assertEqual(record["path"], "creditcard.csv")
        self.assertEqual(manifest.find_absolute_paths(record), [])
        self.assertEqual(dataset.portable_path(CREDITCARD.path), "data/raw/creditcard.csv")

    def test_expected_fingerprint_matches_project_record(self):
        work_log = (PROJECT_ROOT / "docs" / "WORK_LOG.md").read_text()
        self.assertIn(dataset.EXPECTED_SHA256, work_log)


class TestJsonAndPaths(unittest.TestCase):
    def test_strict_json(self):
        text = manifest.dumps_strict({"a": math.nan, "b": [math.inf, 1.5], "c": "x"})
        self.assertEqual(json.loads(text), {"a": None, "b": [None, 1.5], "c": "x"})

    def test_absolute_path_detection(self):
        obj = {"ok": "data/raw/creditcard.csv", "posix": "/tmp/x", "win": "C:\\x\\y",
               "home": "~/x", "nested": [{"p": str(Path.home() / "f")}]}
        found = manifest.find_absolute_paths(obj)
        self.assertEqual(sorted(found), ["/home", "/nested/0/p", "/posix", "/win"])

    def test_project_paths_are_repository_relative(self):
        self.assertTrue(CREDITCARD.path.is_relative_to(PROJECT_ROOT))
        self.assertEqual(CREDITCARD.path.relative_to(PROJECT_ROOT).as_posix(), "data/raw/creditcard.csv")

    def test_no_personal_absolute_paths_in_code_or_onboarding_docs(self):
        files = [*PROJECT_ROOT.joinpath("src").rglob("*.py"), *PROJECT_ROOT.joinpath("tests").rglob("*.py"),
                 PROJECT_ROOT / "README.md", PROJECT_ROOT / "data" / "README.md",
                 PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "requirements-notebook.txt",
                 PROJECT_ROOT / "docs" / "result_storage_policy.md"]
        for path in files:
            text = path.read_text()
            for marker in PERSONAL_PATH_MARKERS:
                self.assertNotIn(marker, text, f"{path.relative_to(PROJECT_ROOT)} contains {marker!r}")


class TestResultSizePolicy(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(manifest.result_size_status(5_000_000), "ok")
        self.assertEqual(manifest.result_size_status(5_000_001), "over_target")
        self.assertEqual(manifest.result_size_status(10_000_000), "over_target")
        self.assertEqual(manifest.result_size_status(10_000_001), "over_hard_limit")

    def test_writer_enforces_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            self.assertEqual(manifest.write_result_json({"x": 1}, out)["status"], "ok")
            with self.assertRaises(ValueError):
                manifest.write_result_json({"path": "/abs/file"}, out)
            big = {"x": "a" * 10_000_100}
            with self.assertRaises(ValueError):
                manifest.write_result_json(big, out)
            written = manifest.write_result_json(big, out, size_exception_reason="synthetic test")
            self.assertEqual(written["status"], "over_hard_limit")
            mid = {"x": "a" * 6_000_000}
            self.assertEqual(manifest.write_result_json(mid, out)["status"], "over_target")

    def test_tracked_result_files_within_hard_limit(self):
        for path in sorted((PROJECT_ROOT / "reports").glob("*.json")):
            status = manifest.result_size_status(path.stat().st_size)
            self.assertNotEqual(status, "over_hard_limit", path.name)


class TestRequirements(unittest.TestCase):
    def test_core_requirements_exactly_pinned(self):
        requirements = manifest.parse_requirements(manifest.REQUIREMENTS_FILE)
        for name in ("numpy", "pandas", "scikit-learn", "scipy"):
            self.assertIn(name, requirements)
        for name, spec in requirements.items():
            self.assertTrue(spec.startswith("=="), f"{name} is not exactly pinned")

    def test_parser_skips_comments_and_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.txt"
            path.write_text("# c\n-r other.txt\nfoo==1.0  # note\n\nbar\n")
            self.assertEqual(manifest.parse_requirements(path), {"foo": "==1.0", "bar": ""})


class TestEnvironmentManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.built = manifest.environment_manifest(Path(self.tmp.name) / "creditcard.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_contents(self):
        m = self.built
        for key in ("git", "python", "packages", "dataset", "seeds", "configuration"):
            self.assertIn(key, m)
        self.assertEqual(m["seeds"], pre_registered_configuration()["seeds"])
        self.assertEqual(m["configuration"]["pre_registered_configuration_sha256"],
                         manifest.canonical_sha256(pre_registered_configuration()))
        self.assertIn("scikit-learn", m["packages"])
        self.assertEqual(m["dataset"]["status"], "missing")

    def test_portable_and_strict(self):
        self.assertEqual(manifest.find_absolute_paths(self.built), [])
        json.loads(manifest.dumps_strict(self.built))

    def test_environment_manifest_is_valid(self):
        """The manifest file in the working tree (reports/environment_manifest.json)."""
        path = manifest.ENVIRONMENT_MANIFEST_PATH
        self.assertTrue(path.is_file())
        on_disk = json.loads(path.read_text())
        self.assertEqual(on_disk["manifest_type"], "fraudguard_environment_manifest")
        self.assertEqual(manifest.find_absolute_paths(on_disk), [])
        self.assertEqual(on_disk["seeds"], pre_registered_configuration()["seeds"])
        self.assertEqual(manifest.result_size_status(path.stat().st_size), "ok")

    def test_environment_manifest_matches_current_configuration(self):
        """The working-tree manifest records the hash of the current pre-registered configuration."""
        on_disk = json.loads(manifest.ENVIRONMENT_MANIFEST_PATH.read_text())
        self.assertEqual(on_disk["configuration"]["pre_registered_configuration_sha256"],
                         manifest.canonical_sha256(pre_registered_configuration()))


class TestFreezeManifest(unittest.TestCase):
    def setUp(self):
        self.template = manifest.freeze_manifest_template()

    def _frozen(self):
        """A synthetic, fully populated manifest. Every value is a placeholder, not a result."""
        m = copy.deepcopy(self.template)
        m["status"] = "frozen"
        m["provenance"].update(git_commit="0" * 40, git_dirty=False, created_at_utc="2000-01-01T00:00:00Z",
                               frozen_at_utc="2000-01-01T00:00:00Z", approved_by="synthetic",
                               approval_date="2000-01-01")
        m["environment"] = {"python": "synthetic", "packages": {"numpy": "synthetic"}}
        m["dataset"]["observed_sha256"] = m["dataset"]["expected_sha256"]
        d = m["validation_stage_decisions"]
        d["selected_configurations"] = {"logistic_regression": {"C": "synthetic"},
                                        "hist_gradient_boosting": {"learning_rate": "synthetic"}}
        d["calibration"] = {"logistic_regression": {"status": "calibrated", "A": 1.0, "B": 0.0},
                            "hist_gradient_boosting": {"status": "uncalibrated", "A": None, "B": None}}
        d["reliability_k_derived_edges"] = {"synthetic": []}
        d["empirical_thresholds_per_cost_ratio"] = {"synthetic": []}
        d["champion"] = "logistic_regression"
        d["development_results_file"] = "reports/synthetic.json"
        return m

    def test_template_is_valid_template(self):
        self.assertEqual(self.template["status"], "template")
        self.assertEqual(manifest.validate_freeze_manifest(self.template), [])

    def test_template_file_matches_code(self):
        """The template file in the working tree equals the code-generated template."""
        on_disk = json.loads(manifest.FREEZE_TEMPLATE_PATH.read_text())
        self.assertEqual(on_disk, json.loads(manifest.dumps_strict(self.template)))

    def test_template_holds_no_results(self):
        self.assertNotIn("results", self.template)
        decisions = self.template["validation_stage_decisions"]
        self.assertIsNone(decisions["champion"])
        self.assertTrue(all(v is None for v in decisions["selected_configurations"].values()))

    def test_template_cannot_pass_as_frozen(self):
        m = copy.deepcopy(self.template)
        m["status"] = "frozen"
        self.assertNotEqual(manifest.validate_freeze_manifest(m), [])

    def test_synthetic_frozen_manifest_validates(self):
        self.assertEqual(manifest.validate_freeze_manifest(self._frozen()), [])

    def test_frozen_rules(self):
        cases = []
        m = self._frozen(); m["validation_stage_decisions"]["calibration"]["logistic_regression"]["A"] = -1.0
        cases.append(m)
        m = self._frozen(); m["validation_stage_decisions"]["champion"] = "random_forest"; cases.append(m)
        m = self._frozen(); m["validation_stage_decisions"]["calibration"]["hist_gradient_boosting"]["status"] = None
        cases.append(m)
        m = self._frozen(); m["dataset"]["observed_sha256"] = "f" * 64; cases.append(m)
        m = self._frozen(); m["validation_stage_decisions"]["development_results_file"] = "/abs/r.json"
        cases.append(m)
        m = self._frozen(); m["pre_registered_configuration"]["seeds"]["e2_split_seed"] = 7; cases.append(m)
        m = self._frozen(); m["results"] = {}; cases.append(m)
        for case in cases:
            self.assertNotEqual(manifest.validate_freeze_manifest(case), [])


if __name__ == "__main__":
    unittest.main()
