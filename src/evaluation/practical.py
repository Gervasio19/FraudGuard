"""Practical relevance at equal alert volume (protocol Sections 8.2, 9.2; mentor decisions D2, D3), Phase 5B-3.

There is **no numeric AP threshold** for a "practically important" improvement:
none can be derived without real costs, and any value chosen now would be
informed by results already seen. The practical unit is instead **additional
frauds caught at equal alert volume (fixed K)**. At equal K the alert volumes
are identical, so the model with more true positives is cheaper for every
review-cost ratio R > 0; that comparison needs no cost assumption.

Two kinds of output, never mixed in one block:

* **model-generated**: TP / FP / Precision@K / Recall@K per model and the paired
  difference ``delta TP@K`` (with its paired block-bootstrap interval, added by
  the runner);
* **hypothetical** (``assumed_scenario_input`` -> ``simulated_outcome``):
  break-even ratios, costs in review units over the frozen R grid, and
  analyst-hours at the frozen 5 / 10 / 15 minutes per review. R, the review
  minutes and "one review = one cost unit" are assumptions; no real
  investigation cost, loss or capacity was provided.

Supporting evidence only: nothing here is a champion gate, and no threshold or
operating point is selected. The time-based queue and per-scenario thresholds
are deferred (D3).
"""

from __future__ import annotations

from src.evaluation.capacity import fixed_k_table
from src.evaluation.evidence_labels import ASSUMED_SCENARIO_INPUT, MODEL_GENERATED, SIMULATED_OUTCOME, labelled

COST_MODEL = ("Cost(policy) = c_r * (TP + FP) + L * FN, with c_r = 1 review unit and R = L / c_r "
              "(protocol Section 8.2); net saving vs no review = R * TP - K")


def fixed_k_comparison(y_true, scores: dict, times, positions, ks, baseline: str, candidate: str,
                       positive_label=1) -> dict:
    """Fixed-K rows per model and ``delta TP@K = TP(candidate) - TP(baseline)`` at equal K. Model-generated."""
    tables = {name: fixed_k_table(y_true, s, times, positions, ks, positive_label) for name, s in scores.items()}
    comparison = []
    for row_b, row_c in zip(tables[baseline], tables[candidate]):
        comparison.append({
            "k": row_b["k"], "tp_baseline": row_b["tp"], "tp_candidate": row_c["tp"],
            "delta_tp_candidate_minus_baseline": row_c["tp"] - row_b["tp"],
            "delta_alerts": 0,
            "tie_sensitive": bool(row_b["ties"]["tie_sensitive"] or row_c["ties"]["tie_sensitive"]),
            "reading": ("equal alert volume: more TP is better for every R > 0, with no cost assumption"
                        if row_c["tp"] != row_b["tp"] else "equal alert volume and equal TP: no difference"),
        })
    return labelled(MODEL_GENERATED, {
        "baseline": baseline, "candidate": candidate, "ks": [int(k) for k in ks],
        "by_model": tables, "equal_alert_volume_comparison": comparison,
    })


def scenario_inputs(r_grid, review_minutes, source: str) -> dict:
    """The assumed scenario inputs, labelled as hypothetical."""
    return labelled(ASSUMED_SCENARIO_INPUT, {
        "review_cost_ratio_R": [float(r) for r in r_grid],
        "review_minutes": [float(m) for m in review_minutes],
        "cost_unit": "one review = 1 unit for every alert, fraud or not; a missed fraud costs R units",
        "cost_model": COST_MODEL,
        "source": source,
        "assumptions": [
            "R (cost of a missed fraud in review units) is unknown; the grid spans two orders of magnitude",
            "minutes per review are unknown; 5 / 10 / 15 are illustrative",
            "a review, when performed, prevents the whole loss (100% efficacy) and costs the same for every alert",
            "every fraud costs the same L; transaction amounts and their currency are not used",
            "losses are additive and independent; follow-on fraud is not modelled",
        ],
    })


def _ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def scenario_outcomes(comparison: dict, r_grid, review_minutes, duration_hours: float) -> dict:
    """Break-even ratios, R-grid sensitivity and analyst-hours from the fixed-K rows. Hypothetical.

    * ``break_even_R`` (review vs no review at K): ``K / TP = 1 / Precision``.
    * ``marginal_break_even_R`` between consecutive K of one model:
      ``R* = delta alerts / delta TP``; not defined when no additional fraud is caught.
    * Between the two models at equal K: ``delta alerts = 0``, so ``R* = 0`` when
      ``delta TP != 0`` (the model with more TP wins for every R > 0) and is not
      defined when ``delta TP = 0``.
    """
    baseline, candidate = comparison["baseline"], comparison["candidate"]
    per_model = {}
    for name, rows in comparison["by_model"].items():
        out_rows, previous = [], None
        for row in rows:
            k, tp, fn = row["k"], row["tp"], row["n_positive"] - row["tp"]
            marginal = None
            if previous is not None:
                d_alerts, d_tp = k - previous["k"], tp - previous["tp"]
                marginal = {"from_k": previous["k"], "to_k": k, "delta_alerts": d_alerts, "delta_tp": d_tp,
                            "R_star": _ratio(d_alerts, d_tp),
                            "defined": d_tp > 0}
            out_rows.append({
                "k": k, "tp": tp, "fn": fn,
                "alerts_per_hour_of_traffic": k / duration_hours,
                "break_even_R": _ratio(k, tp),
                "marginal_break_even_R": marginal,
                "analyst_hours": {f"{m:g}_min": k * m / 60.0 for m in review_minutes},
                "analyst_hours_per_fraud_caught": {f"{m:g}_min": _ratio(k * m / 60.0, tp) for m in review_minutes},
                "by_R": [{"R": float(r), "cost_in_review_units": k + r * fn,
                          "no_review_cost_in_review_units": r * row["n_positive"],
                          "net_saving_vs_no_review_in_review_units": r * tp - k,
                          "review_beats_no_review": bool(r * tp > k)} for r in r_grid],
            })
            previous = row
        per_model[name] = out_rows
    between = []
    for row in comparison["equal_alert_volume_comparison"]:
        d_tp = row["delta_tp_candidate_minus_baseline"]
        between.append({
            "k": row["k"], "delta_tp_candidate_minus_baseline": d_tp, "delta_alerts": 0,
            "R_star": 0.0 if d_tp != 0 else None,
            "defined": d_tp != 0,
            "cheaper_for_every_positive_R": (candidate if d_tp > 0 else baseline if d_tp < 0 else None),
            "net_saving_difference_by_R": [
                {"R": float(r), "candidate_minus_baseline_in_review_units": r * d_tp} for r in r_grid],
        })
    return labelled(SIMULATED_OUTCOME, {
        "baseline": baseline, "candidate": candidate, "duration_hours": duration_hours,
        "per_model": per_model, "between_models_at_equal_k": between,
        "interpretation_limits": [
            "all values are in hypothetical review units, not money",
            "no operating point, K or threshold is selected from this table",
            "fixed-K is retrospective (it ranks the whole period at once); the time-based queue is deferred",
        ],
    })
