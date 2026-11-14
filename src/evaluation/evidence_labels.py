"""Evidence classes and the mandatory hypothetical-scenario disclosure (Phase 5B-3).

Every result block is one of four evidence classes:

* ``observed_dataset``: counts and facts read from the dataset (rows, frauds,
  timestamps, amounts).
* ``model_generated``: quantities computed from fitted models and the data
  (scores, AP, calibration metrics, bootstrap intervals, the champion gates).
* ``assumed_scenario_input``: operational or business parameters that were
  **assumed**, not observed (review-cost ratios, review minutes, capacities).
* ``simulated_outcome``: quantities computed by combining model results with
  assumed scenario inputs (break-even ratios in a cost scenario, analyst-hours,
  net savings).

The last two are **hypothetical**. No real investigation cost, financial loss
or operational capacity has been provided to this project, so nothing in those
classes is an actual business outcome.

The labels live in result files, reports, figures and documentation. They are
deliberately *not* added to ``src/models/phase5_settings.py``: that would change
the frozen configuration hash for a documentation purpose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OBSERVED_DATASET = "observed_dataset"
MODEL_GENERATED = "model_generated"
ASSUMED_SCENARIO_INPUT = "assumed_scenario_input"
SIMULATED_OUTCOME = "simulated_outcome"

EVIDENCE_CLASSES = (OBSERVED_DATASET, MODEL_GENERATED, ASSUMED_SCENARIO_INPUT, SIMULATED_OUTCOME)
HYPOTHETICAL_CLASSES = (ASSUMED_SCENARIO_INPUT, SIMULATED_OUTCOME)
EVIDENCE_KEY = "evidence_class"

HYPOTHETICAL_DISCLOSURE = (
    "HYPOTHETICAL: assumed scenario inputs and simulated outcomes, not observed real-world facts. "
    "No real investigation cost, financial loss or operational capacity was provided; nothing here "
    "is an actual business outcome or saving."
)
#: Short label for figure titles and table captions.
HYPOTHETICAL_FIGURE_LABEL = "HYPOTHETICAL SCENARIO: assumed inputs, simulated outcomes (not observed business results)"

#: Result keys that hold assumed operational inputs or outcomes computed from them.
#: A block carrying one of them must be (or sit inside) a hypothetical block.
SCENARIO_KEYS = frozenset({
    "review_cost_ratio_R", "review_minutes", "analyst_hours", "analyst_hours_per_fraud_caught",
    "net_saving_vs_no_review_in_review_units", "cost_in_review_units", "break_even_R",
    "marginal_break_even_R", "bayes_threshold_alerts", "queue_capacity_per_hour", "sla_hours",
})

#: Documentation terms that signal a business or operational scenario.
SCENARIO_TERMS = re.compile(
    r"analyst[- ]hours?|review[- ]cost ratio|cost ratio|break-even|reviews? per hour|reviews/h(?:our)?\b"
    r"|net saving|simulated saving|review minutes|minutes per review|R\s*∈|SLA scenario",
    re.IGNORECASE,
)
DISCLOSURE_MARK = re.compile(r"hypothetical|\[H\]", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)


def labelled(evidence_class: str, payload: dict | None = None, **extra) -> dict:
    """Return ``payload`` as a result block with its evidence class first.

    Hypothetical classes also get ``hypothetical: True`` and the disclosure text.
    """
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"Unknown evidence class {evidence_class!r}; use one of {EVIDENCE_CLASSES}.")
    block: dict = {EVIDENCE_KEY: evidence_class}
    if evidence_class in HYPOTHETICAL_CLASSES:
        block["hypothetical"] = True
        block["disclosure"] = HYPOTHETICAL_DISCLOSURE
    body = {**(payload or {}), **extra}
    if EVIDENCE_KEY in body or (evidence_class in HYPOTHETICAL_CLASSES and
                                ({"hypothetical", "disclosure"} & set(body))):
        raise ValueError("The payload must not set the evidence-class fields itself.")
    block.update(body)
    return block


def missing_evidence_classes(result: dict, sections) -> list[str]:
    """Names of the required top-level sections that lack a valid evidence class."""
    missing = []
    for name in sections:
        block = result.get(name)
        if not isinstance(block, dict) or block.get(EVIDENCE_KEY) not in EVIDENCE_CLASSES:
            missing.append(name)
    return missing


def scenario_violations(obj, location: str = "", inherited: str | None = None) -> list[str]:
    """Locations of scenario content that is not inside a properly disclosed hypothetical block.

    A block inherits the nearest enclosing evidence class. Violations:

    * a :data:`SCENARIO_KEYS` key whose nearest class is not hypothetical;
    * a hypothetical block without ``hypothetical: True`` and the disclosure text;
    * an unknown evidence class.
    """
    found: list[str] = []
    if isinstance(obj, dict):
        current = inherited
        if EVIDENCE_KEY in obj:
            current = obj[EVIDENCE_KEY]
            if current not in EVIDENCE_CLASSES:
                found.append(f"{location or '/'}: unknown evidence class {current!r}")
            elif current in HYPOTHETICAL_CLASSES and not (
                    obj.get("hypothetical") is True and obj.get("disclosure") == HYPOTHETICAL_DISCLOSURE):
                found.append(f"{location or '/'}: hypothetical block without the disclosure")
        for key, value in obj.items():
            if key in SCENARIO_KEYS and current not in HYPOTHETICAL_CLASSES:
                found.append(f"{location}/{key}: scenario content outside a hypothetical block")
            found += scenario_violations(value, f"{location}/{key}", current)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found += scenario_violations(value, f"{location}/{i}", inherited)
    return found


def require_disclosed(result: dict, sections) -> None:
    """Refuse a result whose sections lack an evidence class or whose scenarios lack the disclosure."""
    problems = [f"/{name}: no evidence class" for name in missing_evidence_classes(result, sections)]
    problems += scenario_violations(result)
    if problems:
        raise ValueError("Evidence-class / hypothetical-disclosure check failed: " + "; ".join(problems))


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into ``(heading line, section text)`` pairs; text before the first heading is ''."""
    starts = [m.start() for m in _HEADING.finditer(text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    sections = []
    for begin, end in zip(starts, starts[1:] + [len(text)]):
        chunk = text[begin:end]
        heading = chunk.splitlines()[0] if chunk.startswith("#") else ""
        sections.append((heading, chunk))
    return sections


def undisclosed_sections(text: str, heading_filter=None) -> list[str]:
    """Headings of sections that use a scenario term without a hypothetical mark.

    A section is the text from one heading to the next heading of any level.
    ``heading_filter(heading) -> bool`` limits the check (for example to the
    sections of a chronological log that were written under this requirement).
    """
    bad = []
    for heading, chunk in markdown_sections(text):
        if heading_filter is not None and not heading_filter(heading):
            continue
        if SCENARIO_TERMS.search(chunk) and not DISCLOSURE_MARK.search(chunk):
            bad.append(heading or "(text before the first heading)")
    return bad


def notebook_markdown(path: Path) -> str:
    """Concatenated Markdown cells of a Jupyter notebook, one section per cell."""
    cells = json.loads(Path(path).read_text()).get("cells", [])
    parts = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            parts.append(f"# cell {i}\n" + ("".join(source) if isinstance(source, list) else source))
    return "\n".join(parts)
