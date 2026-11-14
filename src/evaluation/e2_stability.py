"""Criterion E2: selection-aware repeated held-out block splits (protocol Sections 4.3, 7.5), Phase 5B-3.

For each of the 2,000 frozen splits (21 validation blocks of 15 minutes, 10 in
the selection half, ``default_rng(42)``; Addendum C3, C4):

1. the held-out half is the other 11 blocks minus the 5-minute buffer zones;
2. a split is **usable** iff the buffered held-out half holds at least 10 frauds;
3. on the selection half, the best LR and the best HGB configuration are
   re-selected among all 28 cached score vectors with the Section 5.4 rule;
4. the re-selected pair is evaluated on the held-out half:
   ``delta AP = AP(HGB) - AP(LR)``.

E2 passes iff at least 1,000 splits are usable, the **held-out split-stability
frequency** (share of usable splits with ``delta AP > 0``; a difference of
exactly 0 is not positive) is at least 90%, and the median held-out ``delta AP``
is > 0. With fewer than 1,000 usable splits nothing is computed and E2 is "not
estimable under the pre-registered rule".

The frequency is a descriptive resampling stability statistic. It is **not** a
probability that HGB is better, a confidence level, a p-value or an independent
replication rate: every split reuses the same 21 blocks and 56 frauds.

Nothing is refitted and no frozen value is redefined here; block construction,
buffer, seed and fraud counting come from ``src/data/validation_blocks.py``,
``src/evaluation/e2_usability.py`` and ``src/models/phase5_settings.py``.
"""

from __future__ import annotations

import numpy as np

from src.data.split import VALIDATION_END_TIME
from src.data.validation_blocks import assign_blocks, block_edges, draw_selection_blocks, heldout_keep_mask
from src.evaluation.block_bootstrap import RankedScores
from src.evaluation.e2_usability import usability_decision
from src.models.phase5_grid import FAMILIES, HGB, LR, select_within_family
from src.models.phase5_settings import (
    E2_BLOCK_SECONDS, E2_BUFFER_SECONDS, E2_MIN_HELDOUT_FRAUDS, E2_MIN_USABLE_SPLITS, E2_N_BLOCKS,
    E2_N_SELECTION_BLOCKS, E2_N_SPLITS, E2_SPLIT_SEED, E2_STABILITY_THRESHOLD, VALIDATION_START_TIME,
)

NOT_ESTIMABLE = "not estimable under the pre-registered rule"
FREQUENCY_NAME = "held-out split-stability frequency"
FREQUENCY_MEANING = (
    "descriptive resampling stability statistic, conditional on the fixed validation period, the 28 "
    "fitted score vectors and the 21 x 15-min block partition with the 5-min buffer; NOT a probability "
    "that HGB is better, a confidence level, a p-value or an independent replication rate"
)


def _summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "min": float(v.min()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "max": float(v.max()),
            "mean": float(v.mean())}


def e2_gate(n_usable: int, frequency: float | None, median_delta: float | None,
            min_usable: int, threshold: float = E2_STABILITY_THRESHOLD) -> dict:
    """Apply the frozen E2 gate. ``frequency >= threshold`` and ``median > 0`` (strict)."""
    estimable = n_usable >= min_usable
    if not estimable:
        return {"estimable": False, "status": NOT_ESTIMABLE, "passed": False,
                "reason": f"{n_usable} usable splits < {min_usable}; no frequency or median is computed"}
    frequency_pass = bool(frequency >= threshold)
    median_pass = bool(median_delta > 0)
    return {"estimable": True, "status": "estimable",
            "frequency": frequency, "frequency_threshold": threshold, "frequency_pass": frequency_pass,
            "median_heldout_delta_ap": median_delta, "median_pass": median_pass,
            "passed": frequency_pass and median_pass,
            "rule": f"{FREQUENCY_NAME} >= {threshold:g} and median held-out delta AP > 0"}


def e2_selection_aware(times, y_true, scores_by_config: dict, configurations: list[dict],
                       reference_config_id: str, positive_label=1,
                       start_time: int = VALIDATION_START_TIME, end_time_inclusive: int = VALIDATION_END_TIME,
                       n_splits: int = E2_N_SPLITS, min_usable: int | None = None) -> dict:
    """Run criterion E2 on cached validation score vectors (no fit, no retry).

    ``configurations`` rows hold ``config_id``, ``family`` and ``params`` (the
    28 grid points); ``scores_by_config`` maps each ID to its validation
    ``decision_function`` scores. ``reference_config_id`` is the fixed,
    never-selected Phase 4 configuration of the optimism indicator.
    ``n_splits`` / ``min_usable`` exist for synthetic tests only; real runs use
    the frozen 2,000 / 1,000.
    """
    min_usable = E2_MIN_USABLE_SPLITS if min_usable is None else min_usable
    times = np.asarray(times)
    positive = np.asarray(y_true) == positive_label
    ids = [c["config_id"] for c in configurations]
    if set(ids) != set(scores_by_config) or reference_config_id not in scores_by_config:
        raise ValueError("Every configuration needs a score vector, and the reference must be one of them.")
    edges = block_edges(start_time, end_time_inclusive, E2_BLOCK_SECONDS)
    if len(edges) - 1 != E2_N_BLOCKS:
        raise ValueError(f"Expected {E2_N_BLOCKS} blocks, got {len(edges) - 1}.")
    blocks = assign_blocks(times, edges)
    draws = draw_selection_blocks(n_splits, E2_N_BLOCKS, E2_N_SELECTION_BLOCKS, E2_SPLIT_SEED)
    positions = np.arange(len(times))
    ranked = {cid: RankedScores(scores_by_config[cid], positive, times, positions, True) for cid in ids}

    heldout_frauds = np.empty(n_splits, dtype=np.int64)
    records = []
    for i, selection in enumerate(draws):
        keep = heldout_keep_mask(times, edges, selection, E2_BUFFER_SECONDS)
        heldout_frauds[i] = positive[keep].sum()
        if heldout_frauds[i] < E2_MIN_HELDOUT_FRAUDS:
            continue  # skipped: not usable (Section 7.5, step 7)
        in_selection = np.isin(blocks, selection)
        if not positive[in_selection].any():
            raise RuntimeError(
                f"Split {i}: the selection half holds no fraud, so no configuration can be selected. "
                "The protocol does not define this case; it needs an explicit decision, not a silent rule.")
        w_sel, w_held = in_selection.astype(np.float64), keep.astype(np.float64)
        sel_ap = {cid: ranked[cid].average_precision(w_sel) for cid in ids}
        rows = [{**c, "average_precision": sel_ap[c["config_id"]]} for c in configurations]
        chosen = {fam: select_within_family(rows, fam)["selected_config_id"] for fam in FAMILIES}
        held_ap = {fam: ranked[chosen[fam]].average_precision(w_held) for fam in FAMILIES}
        records.append({
            "chosen": chosen, "heldout_ap": held_ap,
            "selection_ap": {fam: sel_ap[chosen[fam]] for fam in FAMILIES},
            "reference": (sel_ap[reference_config_id], ranked[reference_config_id].average_precision(w_held)),
        })

    decision = usability_decision(heldout_frauds, min_usable_splits=min_usable)
    out = {
        "name": "criterion E2: selection-aware repeated held-out block splits (protocol Section 7.5)",
        "settings": {"seed": E2_SPLIT_SEED, "n_splits_attempted": int(n_splits), "n_blocks": E2_N_BLOCKS,
                     "n_selection_blocks": E2_N_SELECTION_BLOCKS, "block_seconds": E2_BLOCK_SECONDS,
                     "buffer_seconds": E2_BUFFER_SECONDS, "min_heldout_frauds": E2_MIN_HELDOUT_FRAUDS,
                     "min_usable_splits": int(min_usable), "stability_threshold": E2_STABILITY_THRESHOLD,
                     "n_configurations_reselected_from": len(ids), "reference_config_id": reference_config_id},
        "splits": {k: decision[k] for k in ("n_attempted", "n_usable", "n_skipped", "usable_share")},
        "frequency_name": FREQUENCY_NAME, "frequency_meaning": FREQUENCY_MEANING,
    }
    if not decision["e2_estimable"]:
        out["gate"] = e2_gate(decision["n_usable"], None, None, min_usable)
        return out

    delta = np.array([r["heldout_ap"][HGB] - r["heldout_ap"][LR] for r in records])
    frequency = float((delta > 0).mean())
    median = float(np.median(delta))
    out["gate"] = e2_gate(decision["n_usable"], frequency, median, min_usable)
    out["heldout_frauds_usable_splits"] = _summary(heldout_frauds[heldout_frauds >= E2_MIN_HELDOUT_FRAUDS])
    out["heldout_delta_ap"] = {**_summary(delta), "share_positive": frequency,
                               "share_exactly_zero": float((delta == 0).mean()),
                               "share_negative": float((delta < 0).mean())}
    out["heldout_ap"] = {fam: _summary([r["heldout_ap"][fam] for r in records]) for fam in FAMILIES}
    out["selection_frequency"] = {}
    for fam in FAMILIES:
        chosen, counts = np.unique([r["chosen"][fam] for r in records], return_counts=True)
        out["selection_frequency"][fam] = {c: float(n / len(records)) for c, n in
                                           sorted(zip(chosen.tolist(), counts.tolist()), key=lambda t: -t[1])}
    reference_gap = float(np.mean([r["reference"][0] - r["reference"][1] for r in records]))
    out["optimism_indicator"] = {
        "label": ("heuristic; NOT an unbiased estimate of selection optimism and no coverage claim; "
                  "the reference is an LR, so it matches the LR family better than HGB"),
        "reference_mean_selection_minus_heldout_ap": reference_gap,
    }
    for fam in FAMILIES:
        unadjusted = float(np.mean([r["selection_ap"][fam] - r["heldout_ap"][fam] for r in records]))
        out["optimism_indicator"][fam] = {"unadjusted_mean_selection_minus_heldout_ap": unadjusted,
                                          "reference_adjusted": unadjusted - reference_gap}
    out["limitations"] = [
        "held-out halves hold few frauds, so held-out AP is very noisy",
        "selection uses half of validation, so it is noisier than the real full-validation selection",
        "the halves are not independent; the buffer only limits short-range dependence",
        "E2 is a decision heuristic; it indicates selection optimism but neither removes nor estimates it without bias",
    ]
    return out
