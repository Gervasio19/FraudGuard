"""Environment manifest, freeze-manifest template and result-file policy (Phase 5B-0).

* **Environment manifest** (``reports/environment_manifest.json``): Python and
  package versions, dataset fingerprint, Git commit, frozen seeds and a hash of
  the pre-registered configuration. It records the environment in which it was
  generated; it does not certify any other environment.
* **Freeze-manifest template** (``docs/templates/phase5_freeze_manifest.template.json``):
  the structure of the manifest that protocol Section 4.2 step 9 commits before
  any test scoring. It separates the *pre-registered configuration* (fixed
  now) from *validation-stage decisions* (filled at step 9). It never holds
  results; those live in the result files it points to.
* **Result-file policy** (``docs/result_storage_policy.md``): strict JSON, no
  absolute paths, target <= 5 MB, hard limit 10 MB (1 MB = 1,000,000 bytes).

Usage, from the repository root::

    python -m src.reproducibility.manifest environment   # write reports/environment_manifest.json
    python -m src.reproducibility.manifest template      # rewrite the freeze-manifest template
    python -m src.reproducibility.manifest check-size reports/*.json

Nothing here fits a model, reads a label or computes a metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.config import PROJECT_ROOT
from src.models.phase5_settings import RANKING_SCORE_METHOD, pre_registered_configuration
from src.reproducibility.dataset import EXPECTED_RELATIVE_PATH, EXPECTED_SHA256, verify_dataset

SCHEMA_VERSION = 1
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
NOTEBOOK_REQUIREMENTS_FILE = PROJECT_ROOT / "requirements-notebook.txt"
ENVIRONMENT_MANIFEST_PATH = PROJECT_ROOT / "reports" / "environment_manifest.json"
FREEZE_TEMPLATE_PATH = PROJECT_ROOT / "docs" / "templates" / "phase5_freeze_manifest.template.json"
#: Packages not listed in requirements but relevant to scikit-learn's numerics.
TRANSITIVE_PACKAGES = ("joblib", "threadpoolctl")

RESULT_TARGET_BYTES = 5_000_000
RESULT_HARD_LIMIT_BYTES = 10_000_000

FREEZE_MANIFEST_TYPE = "fraudguard_phase5_freeze_manifest"
FREEZE_STATUSES = ("template", "frozen")
FREEZE_TOP_LEVEL_KEYS = (
    "manifest_type", "schema_version", "status", "notes", "protocol", "provenance",
    "environment", "dataset", "pre_registered_configuration", "validation_stage_decisions",
    "test_scope_acknowledgement",
)
MODEL_FAMILIES = ("logistic_regression", "hist_gradient_boosting")

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def json_safe(value):
    """Replace NaN/inf floats with None, recursively (strict JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dumps_strict(obj) -> str:
    return json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n"


def find_absolute_paths(obj, location: str = "") -> list[str]:
    """Return the locations of strings that look like absolute or home-relative paths."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found += find_absolute_paths(value, f"{location}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found += find_absolute_paths(value, f"{location}/{i}")
    elif isinstance(obj, str):
        if obj.startswith(("/", "~", "\\\\")) or _WINDOWS_ABSOLUTE.match(obj) or str(Path.home()) in obj:
            found.append(location or "/")
    return found


def canonical_sha256(obj) -> str:
    text = json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_requirements(path: Path) -> dict[str, str]:
    """Map package name to its version specifier; skips blanks, comments and ``-r`` lines."""
    requirements: dict[str, str] = {}
    if not Path(path).is_file():
        return requirements
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT_NAME.match(line)
        if match:
            requirements[match.group(1)] = match.group(2)
    return requirements


def installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_state(root: Path = PROJECT_ROOT) -> dict:
    """``{"commit": <HEAD sha or None>, "dirty": <bool or None>}``; None when Git is unavailable."""
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                       text=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or None, "dirty": None if status is None else bool(status)}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Environment manifest
# ---------------------------------------------------------------------------

def environment_manifest(dataset_path: Path | None = None) -> dict:
    config = pre_registered_configuration()
    packages: dict[str, dict] = {}
    for file, role in ((REQUIREMENTS_FILE, "required"), (NOTEBOOK_REQUIREMENTS_FILE, "optional (notebook)")):
        for name, spec in parse_requirements(file).items():
            packages[name] = {"role": role, "specifier": spec or None, "installed": installed_version(name)}
    for name in TRANSITIVE_PACKAGES:
        packages.setdefault(name, {"role": "transitive (scikit-learn)", "specifier": None,
                                   "installed": installed_version(name)})
    dataset = verify_dataset(dataset_path) if dataset_path is not None else verify_dataset()
    return {
        "manifest_type": "fraudguard_environment_manifest",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "scope": ("Records the environment in which this file was generated. It does not certify "
                  "that any other environment reproduces the project's results."),
        "git": git_state(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": packages,
        "dataset": dataset,
        "seeds": config["seeds"],
        "configuration": {
            "ranking_score_method": RANKING_SCORE_METHOD,
            "pre_registered_configuration_sha256": canonical_sha256(config),
            "source": "src/models/phase5_settings.py",
        },
    }


# ---------------------------------------------------------------------------
# Freeze manifest
# ---------------------------------------------------------------------------

def freeze_manifest_template() -> dict:
    """The freeze-manifest structure with every step-9 field still unset (None)."""
    return {
        "manifest_type": FREEZE_MANIFEST_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "template",
        "notes": [
            "Template only: no decision has been frozen and no experiment has been run.",
            "pre_registered_configuration is fixed before any model is fitted (protocol Revision 4 "
            "and Addendum C). validation_stage_decisions are filled at protocol Section 4.2 step 9.",
            "This manifest never stores metric values or predictions; it points to result files.",
            "Paths are repository-relative. A frozen copy is committed before any test scoring.",
        ],
        "protocol": {
            "document": "docs/phase_5a_experimental_protocol.md",
            "revision": 4,
            "clarification_addendum": "Addendum C (Phase 5B-0)",
        },
        "provenance": {
            "git_commit": None, "git_dirty": None, "created_at_utc": None, "frozen_at_utc": None,
            "approved_by": None, "approval_date": None,
        },
        "environment": {"python": None, "packages": None},
        "dataset": {"path": EXPECTED_RELATIVE_PATH, "expected_sha256": EXPECTED_SHA256, "observed_sha256": None},
        "pre_registered_configuration": pre_registered_configuration(),
        "validation_stage_decisions": {
            "selected_configurations": {family: None for family in MODEL_FAMILIES},
            "calibration": {family: {"status": None, "A": None, "B": None} for family in MODEL_FAMILIES},
            "reliability_k_derived_edges": None,
            "empirical_thresholds_per_cost_ratio": None,
            "champion": None,
            "development_results_file": None,
        },
        "test_scope_acknowledgement": {"acknowledged": None, "acknowledged_by": None, "date": None},
    }


def _nulls(obj, location: str) -> list[str]:
    if obj is None:
        return [location]
    if isinstance(obj, dict):
        return [n for k, v in obj.items() for n in _nulls(v, f"{location}/{k}")]
    if isinstance(obj, list):
        return [n for i, v in enumerate(obj) for n in _nulls(v, f"{location}/{i}")]
    return []


def validate_freeze_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; an empty list means the manifest is valid for its status."""
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    missing = [k for k in FREEZE_TOP_LEVEL_KEYS if k not in manifest]
    extra = [k for k in manifest if k not in FREEZE_TOP_LEVEL_KEYS]
    problems += [f"missing key: {k}" for k in missing] + [f"unexpected key: {k}" for k in extra]
    if missing:
        return problems
    if manifest["manifest_type"] != FREEZE_MANIFEST_TYPE:
        problems.append("wrong manifest_type")
    if manifest["schema_version"] != SCHEMA_VERSION:
        problems.append("unsupported schema_version")
    if manifest["status"] not in FREEZE_STATUSES:
        problems.append(f"status must be one of {FREEZE_STATUSES}")
    if manifest["protocol"].get("revision") != 4:
        problems.append("protocol revision must be 4")
    if manifest["pre_registered_configuration"] != pre_registered_configuration():
        problems.append("pre_registered_configuration differs from src/models/phase5_settings.py")
    if manifest["dataset"].get("expected_sha256") != EXPECTED_SHA256:
        problems.append("dataset.expected_sha256 differs from the recorded fingerprint")
    problems += [f"absolute path at {loc}" for loc in find_absolute_paths(manifest)]

    if manifest["status"] == "frozen":
        for section in ("provenance", "environment", "dataset"):
            problems += [f"unset field {loc}" for loc in _nulls(manifest[section], section)]
        decisions = manifest["validation_stage_decisions"]
        for key in ("selected_configurations", "reliability_k_derived_edges",
                    "empirical_thresholds_per_cost_ratio", "champion", "development_results_file"):
            problems += [f"unset field {loc}" for loc in _nulls(decisions.get(key), f"decisions/{key}")]
        if manifest["dataset"].get("observed_sha256") not in (None, EXPECTED_SHA256):
            problems.append("dataset.observed_sha256 does not match the expected fingerprint")
        if decisions.get("champion") not in (None, *MODEL_FAMILIES):
            problems.append(f"champion must be one of {MODEL_FAMILIES}")
        for family in MODEL_FAMILIES:
            calibration = decisions.get("calibration", {}).get(family, {})
            status = calibration.get("status")
            if status == "calibrated":
                A, B = calibration.get("A"), calibration.get("B")
                if not (isinstance(A, (int, float)) and isinstance(B, (int, float))
                        and math.isfinite(A) and math.isfinite(B) and A > 0):
                    problems.append(f"{family}: a calibrated model needs finite A > 0 and finite B")
            elif status != "uncalibrated":
                problems.append(f"{family}: calibration status must be 'calibrated' or 'uncalibrated'")
    return problems


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------

def result_size_status(n_bytes: int) -> str:
    if n_bytes > RESULT_HARD_LIMIT_BYTES:
        return "over_hard_limit"
    if n_bytes > RESULT_TARGET_BYTES:
        return "over_target"
    return "ok"


def write_result_json(obj, path: Path, size_exception_reason: str | None = None) -> dict:
    """Write a result file under the storage policy.

    Strict JSON (NaN -> null), no absolute paths. Above the 5 MB target the
    write proceeds but returns ``over_target`` so the report can explain it.
    Above the 10 MB hard limit the write is refused unless an explicit
    ``size_exception_reason`` is given (it must then be stated in the report).
    """
    located = find_absolute_paths(obj)
    if located:
        raise ValueError(f"Result contains absolute paths at {located}.")
    text = dumps_strict(obj)
    n_bytes = len(text.encode("utf-8"))
    status = result_size_status(n_bytes)
    if status == "over_hard_limit" and not size_exception_reason:
        raise ValueError(f"Result is {n_bytes} bytes, above the {RESULT_HARD_LIMIT_BYTES}-byte hard limit.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": path.name, "bytes": n_bytes, "status": status, "size_exception_reason": size_exception_reason}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudGuard reproducibility manifests.")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("environment", help="Write the environment manifest.")
    env.add_argument("--output", type=Path, default=ENVIRONMENT_MANIFEST_PATH)
    tpl = sub.add_parser("template", help="Write the freeze-manifest template.")
    tpl.add_argument("--output", type=Path, default=FREEZE_TEMPLATE_PATH)
    size = sub.add_parser("check-size", help="Check result files against the size policy.")
    size.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.command == "environment":
        manifest = environment_manifest()
        print(json.dumps(write_result_json(manifest, args.output)))
        print(f"dataset: {manifest['dataset']['status']}; git commit: {manifest['git']['commit']} "
              f"(dirty={manifest['git']['dirty']})")
        return 0
    if args.command == "template":
        template = freeze_manifest_template()
        problems = validate_freeze_manifest(template)
        if problems:
            print("\n".join(problems))
            return 1
        print(json.dumps(write_result_json(template, args.output)))
        return 0
    worst = 0
    for path in args.paths:
        n_bytes = path.stat().st_size
        status = result_size_status(n_bytes)
        worst = max(worst, {"ok": 0, "over_target": 1, "over_hard_limit": 2}[status])
        print(f"{status:<16} {n_bytes:>12,} bytes  {path.as_posix()}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
