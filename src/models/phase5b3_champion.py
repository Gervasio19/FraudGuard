"""Champion decision under the frozen rule (protocol Section 4.3; decision #3), Phase 5B-3.

Logistic Regression is the default champion. HistGradientBoosting replaces it
**only if both** gates pass:

* **E1** (conditional, necessary): the 95% percentile interval of the paired
  circular-block bootstrap of validation ``delta AP = AP(HGB) - AP(LR)`` has a
  lower bound > 0 at **every** block length of the decision envelope
  {10, 15, 30} minutes.
* **E2** (selection-aware stability heuristic): at least 1,000 usable splits,
  held-out split-stability frequency >= 90% and median held-out ``delta AP`` > 0.

:func:`champion_decision` takes the two gate records and nothing else. By
construction, calibration results, business scenarios, Precision@K, the W1 -> W2
replay and test data cannot enter the decision (mentor decision D1). They are
reported separately as supporting evidence.
"""

from __future__ import annotations

import math

from src.models.phase5_grid import HGB, LR
from src.models.phase5_settings import pre_registered_configuration

RULE = ("LR is the default champion; HGB replaces it only if E1 and E2 both pass "
        "(protocol Section 4.3; Section 13.0, decision #3)")


def e1_gate(cbb_lower_bounds: dict, envelope_minutes=None) -> dict:
    """E1 passes iff the CBB lower bound is finite and > 0 at every envelope block length.

    ``cbb_lower_bounds`` maps block minutes to the lower bound of the 95%
    interval of ``delta AP`` (``None`` or NaN when the interval is not estimable).
    Block lengths outside the envelope (5 min) are ignored.
    """
    if envelope_minutes is None:
        envelope_minutes = pre_registered_configuration()["uncertainty"]["conditional_bootstrap"][
            "e1_envelope_block_minutes"]
    bounds = {int(k): v for k, v in cbb_lower_bounds.items()}
    missing = [m for m in envelope_minutes if m not in bounds]
    if missing:
        raise ValueError(f"E1 needs the CBB lower bound at every envelope block length; missing {missing}.")
    per_length = {}
    for m in envelope_minutes:
        value = bounds[m]
        ok = value is not None and math.isfinite(value) and value > 0
        per_length[str(m)] = {"lower_bound": value, "above_zero": bool(ok)}
    return {"envelope_block_minutes": list(envelope_minutes), "by_block_minutes": per_length,
            "passed": all(v["above_zero"] for v in per_length.values()),
            "rule": "CBB 95% percentile lower bound of delta AP > 0 at every envelope block length"}


def champion_decision(e1: dict, e2: dict) -> dict:
    """Apply the frozen rule to the E1 and E2 gate records. No other input exists."""
    e1_pass, e2_pass = bool(e1["passed"]), bool(e2["passed"])
    promoted = e1_pass and e2_pass
    if promoted:
        statement = ("HGB is adopted. The validation delta AP is an optimistic development estimate; "
                     "the test delta AP is the confirmatory figure.")
    elif not e2.get("estimable", True):
        statement = ("LR is retained because E2 was not estimable under the pre-registered rule, "
                     "not because of evidence against HGB.")
    else:
        statement = ("LR is retained: no sufficient evidence for replacement at this sample size. "
                     "This is not a finding that LR is better; with 56 validation frauds the rule has low power.")
    return {
        "rule": RULE,
        "champion": HGB if promoted else LR,
        "challenger": LR if promoted else HGB,
        "e1_passed": e1_pass, "e2_passed": e2_pass, "e2_estimable": bool(e2.get("estimable", True)),
        "statement": statement,
        "inputs_used": ["E1 gate record", "E2 gate record"],
        "inputs_not_used": ["calibration results", "Precision@K / Recall@K", "business scenarios",
                            "W1 -> W2 replay", "test partition"],
        "note": "both frozen models are scored on test at the end regardless of this decision",
    }
