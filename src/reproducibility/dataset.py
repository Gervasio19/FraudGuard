"""Dataset integrity check (Phase 5B-0).

Verifies that ``data/raw/creditcard.csv`` is present and byte-identical to the
copy used throughout the project, by SHA-256 over the file bytes. The check
never parses rows, so it reads no label.

Usage, from the repository root::

    python -m src.reproducibility.dataset            # check the default location
    python -m src.reproducibility.dataset --data PATH

Exit status: 0 = matches, 1 = missing, 2 = present but different.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from src.config import CREDITCARD, PROJECT_ROOT

#: SHA-256 recorded since Phase 4a (docs/WORK_LOG.md, Phase 2-4 reports).
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_FILENAME = "creditcard.csv"
EXPECTED_RELATIVE_PATH = "data/raw/creditcard.csv"

OK, MISSING, MISMATCH = 0, 1, 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Repository-relative POSIX path when inside the repository, else the bare file name."""
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return path.name


def verify_dataset(path: Path = CREDITCARD.path) -> dict:
    """Return a JSON-ready integrity record; ``status`` is 'ok', 'missing' or 'mismatch'."""
    path = Path(path)
    record = {"path": portable_path(path), "expected_sha256": EXPECTED_SHA256, "observed_sha256": None}
    if not path.is_file():
        record["status"] = "missing"
        return record
    record["observed_sha256"] = sha256_file(path)
    record["status"] = "ok" if record["observed_sha256"] == EXPECTED_SHA256 else "mismatch"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the FraudGuard raw dataset fingerprint.")
    parser.add_argument("--data", type=Path, default=CREDITCARD.path, help="Path to creditcard.csv.")
    args = parser.parse_args(argv)
    record = verify_dataset(args.data)
    if record["status"] == "missing":
        print(f"MISSING: {record['path']} not found. Download {EXPECTED_FILENAME} from Kaggle and "
              f"place it at {EXPECTED_RELATIVE_PATH} (see data/README.md).")
        return MISSING
    if record["status"] == "mismatch":
        print(f"MISMATCH: {record['path']} has SHA-256 {record['observed_sha256']}, expected "
              f"{EXPECTED_SHA256}. The file differs from the project's dataset; results will not "
              "reproduce. Re-download the original file and do not edit it.")
        return MISMATCH
    print(f"OK: {record['path']} matches SHA-256 {EXPECTED_SHA256}.")
    return OK


if __name__ == "__main__":
    sys.exit(main())
