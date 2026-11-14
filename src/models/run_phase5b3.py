"""Runner for Phase 5B-3: calibration, criteria E1 and E2, and the champion decision.

Protocol Section 4.2, steps 4, 6 and 7, plus the fixed-K part of step 5 as
supporting evidence (mentor decisions D1-D8, 2026-11-14). In order:

1. Recompute the 28-configuration score cache (``compute_grid``; scores are
   never stored on disk) and check it against the recorded Phase 5B-2 grid.
2. Fit and check the sigmoid calibrator of each selected model on ``T_cal``
   only (own Platt fitter, compared with scikit-learn's public
   ``CalibratedClassifierCV(FrozenEstimator(...))``).
3. Freeze the K-derived reliability edges from validation scores **before** any
   validation label is used for calibration evaluation.
4. Evaluate calibration on validation (Section 6.5), with the ``T_cal``
   calibrator bootstrap (A), the calibration-evaluation bootstrap (B) and the
   isotonic and C-B sensitivities.
5. E1: conditional paired block bootstrap of the fixed selected pair.
6. E2: selection-aware repeated held-out block splits over all 28 score vectors.
7. Champion decision from the E1 and E2 gate records only.
8. Practical relevance at equal alert volume (supporting evidence; hypothetical
   scenario blocks are labelled as such).

Test rows are removed before any label access. The test partition is never
fitted, scored, calibrated on or used for any decision here. Calibration,
practical relevance and every other analysis outside E1/E2 cannot change the
champion: :func:`src.models.phase5b3_champion.champion_decision` takes the two
gate records only.

Usage, from the repository root::

    python -m src.models.run_phase5b3 [--output reports/phase5b3_results.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score

from src.config import CREDITCARD, PROJECT_ROOT, DatasetConfig
from src.data.split import VALIDATION_END_TIME
from src.data.temporal_windows import split_two_time_windows
from src.evaluation import block_bootstrap as bb
from src.evaluation import calibration as cal
from src.evaluation.e2_stability import e2_selection_aware
from src.evaluation.evidence_labels import (
    ASSUMED_SCENARIO_INPUT, HYPOTHETICAL_DISCLOSURE, MODEL_GENERATED, OBSERVED_DATASET, SIMULATED_OUTCOME,
    labelled, require_disclosed,
)
from src.evaluation.practical import fixed_k_comparison, scenario_inputs, scenario_outcomes
from src.models import phase5_grid as grid
from src.models.phase5_settings import (
    CONDITIONAL_BOOTSTRAP_SEED, OTHER_BOOTSTRAP_SEED, T_CAL_END_TIME, T_CAL_START_TIME, TCAL_BOOTSTRAP_SEED,
    VALIDATION_START_TIME, pre_registered_configuration,
)
from src.models.phase5b3_champion import champion_decision, e1_gate
from src.models.ranking_score import calibrated_logits
from src.models.run_phase5b2_grid import PHASE4_CONFIG, compute_grid
from src.reproducibility.dataset import verify_dataset
from src.reproducibility.manifest import canonical_sha256, git_state, installed_version, write_result_json
from src.visualization.phase5b3_figures import fixed_k_scenario_figure, reliability_figure

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase5b3_results.json"
GRID_RESULTS = PROJECT_ROOT / "reports" / "phase5b2_grid_results.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RUNTIME_KEY = "runtime_seconds"
CALIBRATION_EVALUATION_BLOCK_MINUTES = 15  # mentor decision D5
FAMILY_LABEL = {grid.LR: "LR", grid.HGB: "HGB"}
C_A, C_B_10, C_B_15, ISOTONIC = "sigmoid_c_a", "sigmoid_c_b_10pct", "sigmoid_c_b_15pct", "isotonic_c_a"
NOT_AVAILABLE = "not available (no valid calibrator)"
#: Top-level result sections that must carry an evidence class.
EVIDENCE_SECTIONS = (
    "data", "test_isolation", "selection_continuity", "calibration", "calibration_threshold_propagation",
    "e1_conditional_bootstrap", "e2_selection_aware", "champion_decision", "fixed_k_comparison",
    "scenario_inputs", "scenario_outcomes",
)


@dataclasses.dataclass(frozen=True)
class Replicates:
    """Replicate counts. Real runs use :func:`frozen_replicates`; smaller values are for synthetic tests only."""
    conditional_primary: int
    conditional_sensitivity: int
    t_cal: int
    calibration_evaluation: int
    iid_continuity: int = 2_000
    permutations: int = 2_000
    e2_splits: int | None = None
    e2_min_usable: int | None = None


def frozen_replicates() -> Replicates:
    s = pre_registered_configuration()
    cb = s["uncertainty"]["conditional_bootstrap"]
    return Replicates(conditional_primary=cb["n_replicates_primary"],
                      conditional_sensitivity=cb["n_replicates_sensitivity"],
                      t_cal=s["calibration"]["tcal_bootstrap"]["n_replicates"],
                      calibration_evaluation=cb["n_replicates_primary"])


# ---------------------------------------------------------------------------
# Step 1: selection continuity
# ---------------------------------------------------------------------------

def selection_from_cache(run: dict, y_val, recorded_grid: dict | None) -> tuple[dict, list[dict], dict]:
    """Re-apply the Section 5.4 rule to the recomputed cache and compare it with the recorded grid."""
    table = [{"config_id": cid, "family": f["family"], "params": f["params"],
              "average_precision": float(average_precision_score(y_val, f["scores"]["validation"]["decision_function"]))}
             for cid, f in run["fitted"].items()]
    selection = {fam: grid.select_within_family(table, fam) for fam in grid.FAMILIES}
    selected = {fam: selection[fam]["selected_config_id"] for fam in grid.FAMILIES}
    record = {"selected": selected,
              "selected_average_precision": {fam: selection[fam]["selected_average_precision"]
                                             for fam in grid.FAMILIES},
              "rule": "protocol Section 5.4, unchanged; recomputed from the in-memory score cache"}
    if recorded_grid is not None:
        recorded = {r["config_id"]: r["average_precision"] for r in recorded_grid["grid_table_track_b"]}
        differences = [abs(r["average_precision"] - recorded[r["config_id"]]) for r in table]
        recorded_selected = {fam: recorded_grid["step3_selection_record"][fam]["selected_config_id"]
                             for fam in grid.FAMILIES}
        record["recorded_phase5b2_grid"] = {
            "file": "reports/phase5b2_grid_results.json",
            "max_abs_ap_difference_over_28": max(differences),
            "all_28_ap_identical": all(d == 0.0 for d in differences),
            "recorded_selected": recorded_selected,
            "selection_identical": recorded_selected == selected,
        }
        if max(differences) > 1e-12 or recorded_selected != selected:
            raise AssertionError("The recomputed grid does not reproduce the recorded Phase 5B-2 grid; "
                                 "Phase 5B-3 must not continue on different scores.")
    return selected, table, record


# ---------------------------------------------------------------------------
# Step 2: calibrators on T_cal
# ---------------------------------------------------------------------------

def public_sigmoid_reference(configuration: dict, cached: dict, run: dict, config: DatasetConfig) -> dict:
    """scikit-learn's public sigmoid calibrator on the same fit, for the equivalence check (D6).

    The configuration is refitted on ``T_fit`` (deterministic); the scores must
    equal the cache. Only the public ``CalibratedClassifierCV`` /
    ``FrozenEstimator`` API is used; the frozen pipeline is never refitted by it.
    """
    f = grid.fit_configuration(configuration, run["fit_rows"], run["cal_rows"], run["validation"], config)
    for part in ("t_cal", "validation"):
        if not np.array_equal(f["scores"][part]["decision_function"], cached["scores"][part]["decision_function"]):
            raise AssertionError(f"{configuration['config_id']}: the refit does not reproduce the cached scores.")
    features = list(grid.model_features())
    X_cal, X_val = run["cal_rows"][features], run["validation"][features]
    public = CalibratedClassifierCV(FrozenEstimator(f["pipeline"]), method="sigmoid")
    public.fit(X_cal, run["cal_rows"][config.target_column])
    column = list(public.classes_).index(config.positive_label)
    return {"n_calibrators": len(public.calibrated_classifiers_),
            "t_cal": public.predict_proba(X_cal)[:, column].astype(np.float64),
            "validation": public.predict_proba(X_val)[:, column].astype(np.float64)}


def platt_objective_and_gradient(logits, positive) -> tuple[float, float]:
    """Platt's smoothed-target cross-entropy at ``logits`` and its gradient norm in ``(A, B)``-free form.

    The gradient is taken with respect to the logits' affine parameters through
    ``sum(p - t)`` and ``sum((p - t) * z)``; both vanish at the optimum.
    """
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = float(positive.sum()), float((~positive).sum())
    target = np.where(positive, (n1 + 1.0) / (n1 + 2.0), 1.0 / (n0 + 2.0))
    z = np.asarray(logits, dtype=np.float64)
    residual = expit(z) - target
    return (float(np.sum(np.logaddexp(0.0, z) - target * z)),
            float(max(abs(residual.sum()), abs(residual @ z))))


def equivalence_record(own: dict, public: dict, scores: dict, y_cal) -> dict:
    """Measured difference between the own Platt fit and the public scikit-learn calibrator (D6).

    Nothing is accepted silently: besides the probability and logit
    differences, the record compares both solutions on the objective they
    share (Platt's smoothed-target cross-entropy on ``T_cal``), so a difference
    can be attributed to the optimizer that stopped further from the optimum.
    """
    out = {"reference": "sklearn.calibration.CalibratedClassifierCV(FrozenEstimator(pipeline), method='sigmoid') "
                        "fitted on T_cal; public API only",
           "n_public_calibrators": public["n_calibrators"]}
    public_logits = {}
    for part in ("t_cal", "validation"):
        z = calibrated_logits(scores[part], own["A"], own["B"])
        p_own, p_pub = expit(z), public[part]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_pub = np.log(p_pub) - np.log1p(-p_pub)
        public_logits[part] = z_pub
        finite = np.isfinite(z_pub)
        out[part] = {"max_abs_probability_difference": float(np.abs(p_own - p_pub).max()),
                     "max_abs_logit_difference": float(np.abs(z[finite] - z_pub[finite]).max()),
                     "max_relative_probability_difference": float(np.abs(p_own[finite] / p_pub[finite] - 1).max())}
    own_objective, own_gradient = platt_objective_and_gradient(calibrated_logits(scores["t_cal"], own["A"], own["B"]), y_cal)
    pub_objective, pub_gradient = platt_objective_and_gradient(public_logits["t_cal"], y_cal)
    max_logit = max(out[p]["max_abs_logit_difference"] for p in ("t_cal", "validation"))
    max_probability = max(out[p]["max_abs_probability_difference"] for p in ("t_cal", "validation"))
    out["platt_objective_on_t_cal"] = {"own": own_objective, "public": pub_objective,
                                       "own_minus_public": own_objective - pub_objective,
                                       "own_gradient_inf_norm": own_gradient, "public_gradient_inf_norm": pub_gradient}
    out["equivalent_within_1e-6_logit"] = bool(max_logit < 1e-6)
    out["max_abs_probability_difference_below_1e-6"] = bool(max_probability < 1e-6)
    out["own_objective_not_worse"] = bool(own_objective <= pub_objective + 1e-9)
    if out["equivalent_within_1e-6_logit"]:
        out["explanation"] = ("identical within 1e-6 in logit: both minimize the same strictly convex Platt objective "
                              "(own: damped Newton, Lin, Lin and Weng 2007; scikit-learn: L-BFGS-B, gtol 1e-6)")
    else:
        out["explanation"] = (f"NOT identical within 1e-6 in logit (max {max_logit:.2e}; max probability difference "
                              f"{max_probability:.2e}). Both minimize the same strictly convex objective; the own fit "
                              f"has gradient {own_gradient:.1e} and the public fit {pub_gradient:.1e}, with objective "
                              f"difference {own_objective - pub_objective:+.1e}. The difference is the public optimizer "
                              "stopping earlier (L-BFGS-B, gtol 1e-6, after scikit-learn rescales scores with "
                              "max |s| >= 30), not a different calibrator. The own fit is used.")
    return out


# ---------------------------------------------------------------------------
# Step 4: calibration evaluation
# ---------------------------------------------------------------------------

def evaluate_vector(positive, probabilities, logits, band_index=None, band_labels=None) -> dict:
    out = cal.calibration_metrics(positive, probabilities, logits)
    out["reliability_decade_bins"] = cal.decade_reliability(positive, probabilities)
    if band_index is not None:
        out["reliability_k_derived_bins"] = cal.reliability_table(positive, probabilities, band_index, band_labels)
    return out


def bayes_alert_propagation(parameters: np.ndarray, point: tuple[float, float], val_scores, r_grid) -> list[dict]:
    """Alert counts at the Bayes threshold p* = 1/R on validation: point value and bootstrap range. Label-free."""
    rows = []
    for r in r_grid:
        threshold = math.log(1.0 / r) - math.log1p(-1.0 / r)
        point_alerts = int((calibrated_logits(val_scores, *point) >= threshold).sum())
        row = {"R": float(r), "bayes_probability_threshold": 1.0 / r, "bayes_threshold_alerts": point_alerts}
        if len(parameters):
            alerts = np.array([(calibrated_logits(val_scores, a, b) >= threshold).sum() for a, b in parameters])
            lower, upper = np.percentile(alerts, [2.5, 97.5])
            row["bayes_threshold_alerts_bootstrap_95"] = [float(lower), float(upper)]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 5: E1
# ---------------------------------------------------------------------------

def summarize_pair(rep: dict, observed: dict, ks, n_frauds_observed: int) -> dict:
    """Summaries of one paired bootstrap analysis; zero-fraud replicates are dropped and counted."""
    lr, hgb = grid.LR, grid.HGB
    valid = rep["n_frauds"] > 0
    nan = np.where(valid, 0.0, np.nan)
    out = {
        "n_replicates": int(valid.size), "n_zero_fraud_replicates_dropped": int((~valid).sum()),
        "replicate_rows": {"mean": float(rep["n_rows"].mean()), "min": float(rep["n_rows"].min()),
                           "max": float(rep["n_rows"].max())},
        "replicate_frauds": {"mean": float(rep["n_frauds"].mean()), "min": float(rep["n_frauds"].min()),
                             "max": float(rep["n_frauds"].max())},
        "delta_ap_hgb_minus_lr": bb.interval_summary(rep["ap"][hgb] - rep["ap"][lr], observed["delta_ap"]),
        "ap": {FAMILY_LABEL[f]: bb.interval_summary(rep["ap"][f], observed["ap"][f]) for f in (lr, hgb)},
        "delta_at_k_hgb_minus_lr": [],
    }
    for j, k in enumerate(ks):
        d_tp = rep["tp_at_k"][hgb][:, j] - rep["tp_at_k"][lr][:, j] + nan
        with np.errstate(divide="ignore", invalid="ignore"):
            d_recall = d_tp / rep["n_frauds"]
        obs_tp = observed["tp_at_k"][hgb][j] - observed["tp_at_k"][lr][j]
        out["delta_at_k_hgb_minus_lr"].append({
            "k": int(k),
            "delta_tp": bb.interval_summary(d_tp, obs_tp),
            "delta_recall": bb.interval_summary(d_recall, obs_tp / n_frauds_observed),
            "delta_precision": bb.interval_summary(d_tp / k, obs_tp / k),
        })
    return out


def dependence_diagnostics(ranked: dict, positive, times, ks, bin_minutes, n_permutations: int) -> dict:
    """Pre-registered Section 7.4 diagnostic: TP / FN / discordant frauds over time bins (descriptive)."""
    fraud_rows = np.flatnonzero(positive)
    fraud_times = np.asarray(times)[fraud_rows]
    out = {"note": "descriptive; selects nothing and does not re-tune the decision rule (protocol Section 7.4)",
           "permutation_seed": OTHER_BOOTSTRAP_SEED, "by_k": {}}
    for k in ks:
        in_top = {}
        for fam, model in ranked.items():
            mask = np.zeros(len(positive), dtype=bool)
            mask[model.order[:k]] = True
            in_top[fam] = mask[fraud_rows]
        block = {}
        for fam, hit in in_top.items():
            for name, status in (("tp", hit), ("fn", ~hit)):
                block[f"{FAMILY_LABEL[fam]}_{name}"] = bb.status_dependence(
                    fraud_times, status, VALIDATION_START_TIME, VALIDATION_END_TIME, bin_minutes,
                    n_permutations, OTHER_BOOTSTRAP_SEED)
        block["discordant"] = bb.status_dependence(
            fraud_times, in_top[grid.LR] != in_top[grid.HGB], VALIDATION_START_TIME, VALIDATION_END_TIME,
            bin_minutes, n_permutations, OTHER_BOOTSTRAP_SEED)
        out["by_k"][str(k)] = block
    return out


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyse(run: dict, config: DatasetConfig, replicates: Replicates | None = None,
            recorded_grid: dict | None = None, figures_dir: Path | None = None) -> dict:
    """All Phase 5B-3 analyses from the in-memory score cache of :func:`compute_grid`."""
    replicates = replicates or frozen_replicates()
    settings = pre_registered_configuration()
    operations, uncertainty = settings["operations"], settings["uncertainty"]["conditional_bootstrap"]
    ks = operations["fixed_k"]
    r_grid, review_minutes = operations["review_cost_ratio_R"], operations["review_minutes"]
    block_minutes = uncertainty["block_minutes_reported"]
    target, tcol, positive_label = config.target_column, config.time_column, config.positive_label
    validation, cal_rows, fit_rows = run["validation"], run["cal_rows"], run["fit_rows"]
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name], clock = now - clock, now

    t_val, pos_val = validation[tcol].to_numpy(), validation.index.to_numpy()
    t_cal = cal_rows[tcol].to_numpy()
    n_val, n_cal = len(validation), len(cal_rows)

    # --- step 1: selection continuity (validation labels: ranking only) -----------------------
    y_val = (validation[target] == positive_label).to_numpy()
    selected, table, selection_record = selection_from_cache(run, y_val, recorded_grid)
    fitted = {fam: run["fitted"][selected[fam]] for fam in grid.FAMILIES}
    val_scores = {fam: fitted[fam]["scores"]["validation"]["decision_function"] for fam in grid.FAMILIES}
    cal_scores = {fam: fitted[fam]["scores"]["t_cal"]["decision_function"] for fam in grid.FAMILIES}
    reference_id = next(c["config_id"] for c in table
                        if c["family"] == PHASE4_CONFIG["family"] and c["params"] == PHASE4_CONFIG["params"])
    lap("selection_continuity")

    # --- step 2: calibrators, on T_cal only ----------------------------------------------------
    y_cal = (cal_rows[target] == positive_label).to_numpy()
    holdouts = {C_A: np.ones(n_cal, dtype=bool)}
    for key, name in ((C_B_10, "C-B_10pct"), (C_B_15, "C-B_15pct")):
        holdouts[key] = t_cal >= settings["splits"]["train_subpartitions"]["sensitivity_holdouts_time_start"][name]
    calibrators = {fam: {key: cal.primary_calibrator(cal_scores[fam][mask], y_cal[mask], True)
                         for key, mask in holdouts.items()} for fam in grid.FAMILIES}
    equivalence = {}
    for fam in grid.FAMILIES:
        own = calibrators[fam][C_A]
        if own["calibrated"]:
            configuration = {k: fitted[fam][k] for k in ("config_id", "family", "params")}
            public = public_sigmoid_reference(configuration, fitted[fam], run, config)
            equivalence[FAMILY_LABEL[fam]] = equivalence_record(
                own, public, {"t_cal": cal_scores[fam], "validation": val_scores[fam]}, y_cal)
        else:
            equivalence[FAMILY_LABEL[fam]] = {"status": NOT_AVAILABLE}
    lap("calibrator_fit_and_public_equivalence")

    # --- step 3: K-derived reliability edges, from validation scores only (no labels) ----------
    k_edges = {fam: cal.k_derived_score_edges(val_scores[fam], t_val, pos_val, ks) for fam in grid.FAMILIES}
    bands = {fam: cal.k_band_index(val_scores[fam], k_edges[fam]) for fam in grid.FAMILIES}

    # --- step 4: calibration evaluation ---------------------------------------------------------
    windows = split_two_time_windows(validation[tcol]).labels.astype(str).to_numpy()
    window_names = sorted(set(windows))
    evaluation, vectors = {}, {}
    for fam in grid.FAMILIES:
        label = FAMILY_LABEL[fam]
        block: dict = {"config_id": selected[fam]}
        for key in (C_A, C_B_10, C_B_15):
            c = calibrators[fam][key]
            if not c["calibrated"]:
                block[key] = {"status": NOT_AVAILABLE}
                continue
            z = calibrated_logits(val_scores[fam], c["A"], c["B"])
            p = expit(z)
            block[key] = {"validation": evaluate_vector(y_val, p, z, *bands[fam])}
            vectors[f"{label}/{key}"] = {"probabilities": p, "logits": z, "fit_metrics": True}
            if key == C_A:
                z_cal = calibrated_logits(cal_scores[fam], c["A"], c["B"])
                block[key]["t_cal_in_sample_optimistic"] = evaluate_vector(y_cal, expit(z_cal), z_cal)
                block[key]["validation_windows_descriptive"] = {
                    w: cal.calibration_metrics(y_val[windows == w], p[windows == w], z[windows == w])
                    for w in window_names}
                block[key]["ranking_preserved"] = {
                    "average_precision_uncalibrated_score": float(average_precision_score(y_val, val_scores[fam])),
                    "average_precision_calibrated_logit": float(average_precision_score(y_val, z)),
                    "n_unique_scores": int(len(np.unique(val_scores[fam]))),
                    "n_unique_calibrated_logits": int(len(np.unique(z))),
                }
        p_iso = cal.isotonic_probabilities(cal_scores[fam], y_cal, val_scores[fam], True)
        block[ISOTONIC] = {"role": "sensitivity only; never replaces the sigmoid",
                           "n_unique_probabilities": int(len(np.unique(p_iso))),
                           "validation": evaluate_vector(y_val, p_iso, None, *bands[fam])}
        vectors[f"{label}/{ISOTONIC}"] = {"probabilities": p_iso, "logits": None, "fit_metrics": False}
        evaluation[label] = block
    lap("calibration_point_evaluation")

    primary = [f"{FAMILY_LABEL[f]}/{C_A}" for f in grid.FAMILIES]
    pairs = [(primary[0], primary[1])] if all(n in vectors for n in primary) else []
    evaluation_bootstrap = cal.calibration_evaluation_bootstrap(
        vectors, y_val, t_val, VALIDATION_START_TIME, VALIDATION_END_TIME,
        CALIBRATION_EVALUATION_BLOCK_MINUTES * 60, replicates.calibration_evaluation,
        CONDITIONAL_BOOTSTRAP_SEED, pairs=pairs, positive_label=True)
    lap("calibration_evaluation_bootstrap")

    tcal_block_seconds = settings["calibration"]["tcal_bootstrap"]["block_minutes"] * 60
    parameter_bootstrap, propagation = {}, {}
    for fam in grid.FAMILIES:
        label, c = FAMILY_LABEL[fam], calibrators[fam][C_A]
        if not c["calibrated"]:
            parameter_bootstrap[label] = {"status": NOT_AVAILABLE}
            propagation[label] = {"status": NOT_AVAILABLE}
            continue
        boot = cal.tcal_calibrator_bootstrap(cal_scores[fam], y_cal, t_cal, T_CAL_START_TIME, T_CAL_END_TIME,
                                             tcal_block_seconds, replicates.t_cal, TCAL_BOOTSTRAP_SEED, True)
        parameters = boot.pop("parameters")
        boot.pop("low_positive")
        parameter_bootstrap[label] = boot
        propagation[label] = (
            bayes_alert_propagation(parameters if boot["estimable"] else np.empty((0, 2)),
                                    (c["A"], c["B"]), val_scores[fam], r_grid)
            if boot["estimable"] else
            {"status": boot["status"], "point_only": bayes_alert_propagation(
                np.empty((0, 2)), (c["A"], c["B"]), val_scores[fam], r_grid)})
    lap("t_cal_calibrator_bootstrap")

    # --- step 5: E1, conditional paired block bootstrap of the fixed selected pair --------------
    ranked = {fam: bb.RankedScores(val_scores[fam], y_val, t_val, pos_val, True) for fam in grid.FAMILIES}
    observed = {"ap": {fam: ranked[fam].average_precision() for fam in grid.FAMILIES},
                "tp_at_k": {fam: ranked[fam].tp_at_k(ks) for fam in grid.FAMILIES}}
    observed["delta_ap"] = observed["ap"][grid.HGB] - observed["ap"][grid.LR]
    n_frauds = int(y_val.sum())
    schemes = ((bb.CBB, replicates.conditional_primary, "primary"),
               (bb.MBB, replicates.conditional_sensitivity, "sensitivity"),
               (bb.STATIONARY, replicates.conditional_sensitivity, "sensitivity"))
    bootstrap = {}
    for scheme, n_rep, role in schemes:
        bootstrap[scheme] = {"role": role, "n_replicates": n_rep, "seed": CONDITIONAL_BOOTSTRAP_SEED,
                             "by_block_minutes": {}}
        for minutes in block_minutes:
            rep = bb.paired_block_bootstrap(ranked, y_val, t_val, VALIDATION_START_TIME, VALIDATION_END_TIME,
                                            scheme, minutes * 60, n_rep, CONDITIONAL_BOOTSTRAP_SEED, ks, True)
            bootstrap[scheme]["by_block_minutes"][str(minutes)] = summarize_pair(rep, observed, ks, n_frauds)
    cbb = bootstrap[bb.CBB]["by_block_minutes"]
    gate_e1 = e1_gate({m: cbb[str(m)]["delta_ap_hgb_minus_lr"].get("lower") for m in block_minutes})
    lap("e1_block_bootstrap")

    central = CALIBRATION_EVALUATION_BLOCK_MINUTES
    jackknife = bb.block_jackknife(ranked, (grid.HGB, grid.LR), t_val, VALIDATION_START_TIME,
                                   VALIDATION_END_TIME, central * 60)
    iid = bb.stratified_iid_bootstrap(ranked, y_val, replicates.iid_continuity, OTHER_BOOTSTRAP_SEED, True)
    dependence = dependence_diagnostics(ranked, y_val, t_val, (50, 100), block_minutes, replicates.permutations)
    lap("e1_cross_checks")

    # --- step 6: E2 over all 28 cached validation score vectors ---------------------------------
    e2_kwargs = {}
    if replicates.e2_splits is not None:
        e2_kwargs = {"n_splits": replicates.e2_splits, "min_usable": replicates.e2_min_usable}
    e2 = e2_selection_aware(
        t_val, y_val, {cid: f["scores"]["validation"]["decision_function"] for cid, f in run["fitted"].items()},
        [{k: c[k] for k in ("config_id", "family", "params")} for c in table], reference_id, True, **e2_kwargs)
    e2["full_validation_selected_pair"] = {FAMILY_LABEL[f]: selected[f] for f in grid.FAMILIES}
    lap("e2_selection_aware")

    # --- step 7: champion decision (E1 and E2 gate records only) --------------------------------
    decision = champion_decision(gate_e1, e2["gate"])

    # --- step 8: practical relevance at equal alert volume (supporting evidence) ----------------
    names = {fam: f"{FAMILY_LABEL[fam]} ({selected[fam]})" for fam in grid.FAMILIES}
    comparison = fixed_k_comparison(validation[target].to_numpy(), {names[f]: val_scores[f] for f in grid.FAMILIES},
                                    t_val, pos_val, ks, names[grid.LR], names[grid.HGB], positive_label)
    for row, boot_row in zip(comparison["equal_alert_volume_comparison"],
                             cbb[str(central)]["delta_at_k_hgb_minus_lr"]):
        row["paired_block_bootstrap_delta_tp"] = {
            "scheme": "cbb", "block_minutes": central, "n_replicates": replicates.conditional_primary,
            **{k: boot_row["delta_tp"].get(k) for k in ("lower", "upper", "mean", "share_at_or_below_zero",
                                                        "n_valid")}}
    duration_hours = (VALIDATION_END_TIME - VALIDATION_START_TIME + 1) / 3600.0
    inputs = scenario_inputs(r_grid, review_minutes,
                             "frozen hypothetical grids of protocol Sections 8.2 and 9.2 [H] "
                             "(src/models/phase5_settings.py, operations); unchanged")
    outcomes = scenario_outcomes(comparison, r_grid, review_minutes, duration_hours)
    lap("practical_relevance")

    # --- figures ---------------------------------------------------------------------------------
    figures = []
    if figures_dir is not None:
        import matplotlib.pyplot as plt
        for fam in grid.FAMILIES:
            label = FAMILY_LABEL[fam]
            if C_A not in evaluation[label] or "validation" not in evaluation[label][C_A]:
                continue
            path = Path(figures_dir) / f"phase5b3_reliability_{label.lower()}.png"
            fig = reliability_figure(
                evaluation[label][C_A]["validation"]["reliability_decade_bins"],
                f"Reliability: {label} ({selected[fam]}), sigmoid calibrator fitted on T_cal",
                f"MODEL-GENERATED development result; validation, {n_val:,} rows, {n_frauds} frauds; "
                "frozen decade bins", path)
            plt.close(fig)
            figures.append({"file": f"reports/figures/{path.name}", "evidence_class": MODEL_GENERATED,
                            "content": f"reliability diagram, {label}, validation, frozen decade bins"})
        path = Path(figures_dir) / "phase5b3_fixed_k_scenarios.png"
        fig = fixed_k_scenario_figure(
            {name: [r["tp"] for r in rows] for name, rows in comparison["by_model"].items()}, ks, n_frauds,
            review_minutes, f"validation (development), {n_val:,} rows, {n_frauds} frauds; no operating point "
                            "is selected", path)
        plt.close(fig)
        figures.append({"file": f"reports/figures/{path.name}", "evidence_class": SIMULATED_OUTCOME,
                        "hypothetical": True, "disclosure": HYPOTHETICAL_DISCLOSURE,
                        "content": "frauds caught at fixed K (model-generated) and analyst-hours under "
                                   "assumed review minutes (simulated)"})
    lap("figures")

    results = {
        "data": labelled(OBSERVED_DATASET, {
            "t_fit": {"n_rows": int(len(fit_rows)), "n_positive": int((fit_rows[target] == positive_label).sum())},
            "t_cal": {"n_rows": n_cal, "n_positive": int(y_cal.sum()),
                      "time": [int(t_cal.min()), int(t_cal.max())],
                      "c_b_holdouts": {k: {"n_rows": int(m.sum()), "n_positive": int(y_cal[m].sum())}
                                       for k, m in holdouts.items() if k != C_A}},
            "validation": {"n_rows": n_val, "n_positive": n_frauds, "duration_hours": duration_hours,
                           "windows": {w: {"n_rows": int((windows == w).sum()),
                                           "n_positive": int(y_val[windows == w].sum())} for w in window_names}},
        }),
        "test_isolation": labelled(OBSERVED_DATASET, {
            "test_rows_removed_before_label_access": run["n_test_removed"],
            "max_time_in_processed_rows": run["max_time"],
            "statement": "No test row reached any fit, score, calibrator, bootstrap, gate or decision in this run.",
        }),
        "selection_continuity": labelled(MODEL_GENERATED, selection_record),
        "calibration": labelled(MODEL_GENERATED, {
            "role": ("supporting evidence, evaluated independently of discrimination; NOT a champion gate; "
                     "validation calibration results are reported, never acted upon (protocol Section 6.5)"),
            "method": "sigmoid (Platt) on decision_function scores; holdout C-A = T_cal; A = -a_, B = -b_",
            "calibrators": {FAMILY_LABEL[f]: calibrators[f] for f in grid.FAMILIES},
            "public_api_equivalence": equivalence,
            "k_derived_edges": {
                "timing": "computed from validation scores alone, before validation labels were used for "
                          "any calibration metric",
                "edges": {FAMILY_LABEL[f]: k_edges[f] for f in grid.FAMILIES}},
            "evaluation": evaluation,
            "evaluation_bootstrap_B": {
                "decision": "mentor decision D5: circular block bootstrap, 15-min blocks, B = 5,000; a "
                            "separate analysis from E1 (same seed rule: fresh default_rng(0))",
                **evaluation_bootstrap},
            "calibrator_parameter_bootstrap_A": parameter_bootstrap,
            "small_sample_limits": [
                f"{int(y_cal.sum())} positives fix each calibrator",
                f"with {n_frauds} validation positives, O/E cannot distinguish miscalibration below roughly "
                "25-30% from noise",
                "window-level figures are descriptive only",
                "ECE is not reported: with this few positives it is dominated by the binning choice",
            ],
        }),
        "calibration_threshold_propagation": labelled(SIMULATED_OUTCOME, {
            "what": ("number of validation alerts at the Bayes threshold p* = 1/R for each assumed R, with the "
                     "T_cal calibrator-parameter bootstrap range; label-free and descriptive; no threshold is "
                     "selected (per-scenario thresholds are deferred, decision D3)"),
            "assumed_inputs": labelled(ASSUMED_SCENARIO_INPUT, {"review_cost_ratio_R": [float(r) for r in r_grid]}),
            "by_model": propagation,
        }),
        "e1_conditional_bootstrap": labelled(MODEL_GENERATED, {
            "name": "criterion E1: conditional paired block bootstrap of the fixed selected pair (Section 7.2)",
            "pair": {FAMILY_LABEL[f]: selected[f] for f in grid.FAMILIES},
            "observed": {"average_precision": {FAMILY_LABEL[f]: observed["ap"][f] for f in grid.FAMILIES},
                         "delta_ap_hgb_minus_lr": observed["delta_ap"]},
            "conventions": {
                "zero_fraud_replicates": "dropped and counted (D7)", "interval": "percentile 95%",
                "pairing": "both models use identical replicate rows",
                "stationary": "geometric block lengths with mean L, truncated at T (D7)",
                "generator": "fresh numpy.random.default_rng(0) per (scheme, block length)"},
            "schemes": bootstrap,
            "gate": gate_e1,
            "block_jackknife_15_min": {**jackknife, "bootstrap_standard_error_cbb_15_min":
                                       cbb[str(central)]["delta_ap_hgb_minus_lr"].get("standard_error")},
            "iid_stratified_continuity": {
                "role": "continuity with the Phase 4 audit scheme only; ignores temporal dependence",
                "seed": OTHER_BOOTSTRAP_SEED,
                "delta_ap_hgb_minus_lr": bb.interval_summary(iid["ap"][grid.HGB] - iid["ap"][grid.LR],
                                                             observed["delta_ap"])},
            "dependence_diagnostics": dependence,
            "interpretation": [
                "conditional uncertainty of an already-selected pair; it does not include selection optimism",
                "block-bootstrap intervals with 11-31 effective blocks are approximate and possibly "
                "anti-conservative",
                "an interval containing 0 means no evidence of a difference at this sample size, not equivalence",
            ],
        }),
        "e2_selection_aware": labelled(MODEL_GENERATED, e2),
        "champion_decision": labelled(MODEL_GENERATED, {
            **decision,
            "champion_config_id": selected[decision["champion"]],
            "challenger_config_id": selected[decision["challenger"]],
            "status": "validation-stage decision, recorded before any replay or test evaluation",
        }),
        "fixed_k_comparison": comparison,
        "scenario_inputs": inputs,
        "scenario_outcomes": outcomes,
        "figures": figures,
        RUNTIME_KEY: timings,
    }
    require_disclosed(results, EVIDENCE_SECTIONS)
    return results


def run(config: DatasetConfig = CREDITCARD, figures_dir: Path | None = FIGURES_DIR) -> dict:
    dataset = verify_dataset(config.path)
    if dataset["status"] != "ok":
        raise SystemExit(f"Dataset check failed: {dataset['status']} ({dataset['path']}).")
    df = pd.read_csv(config.path)
    start = time.perf_counter()
    # HGB determinism was verified for all 16 configurations in Phase 5B-2; the recomputed APs are
    # compared with that recorded grid instead of fitting every HGB twice again.
    grid_run = compute_grid(df, config, check_hgb_determinism=False)
    grid_seconds = time.perf_counter() - start
    recorded = json.loads(GRID_RESULTS.read_text())
    results = analyse(grid_run, config, recorded_grid=recorded, figures_dir=figures_dir)
    results[RUNTIME_KEY] = {"grid_recompute": grid_seconds, **results[RUNTIME_KEY]}
    settings = pre_registered_configuration()
    return {
        "phase": "5B-3: calibration (step 4), E1 and E2 (step 6), champion decision (step 7); fixed-K part of step 5",
        "result_type": "development (validation); no test evaluation; no operational threshold selected",
        "scope": {
            "champion_gates": "E1 and E2 only (protocol Section 4.3); calibration and practical relevance are "
                              "supporting evidence and cannot change the decision (mentor decision D1)",
            "practical_relevance": "no numeric AP threshold; additional frauds caught at equal alert volume (D2)",
            "deferred": "time-based queue simulation and per-scenario thresholds of protocol step 5 (D3)",
            "ordering_deviation": "steps 6-7 were run before the deferred part of step 5; the champion rule "
                                  "uses step 6 only, so the decision is unaffected (logged, D3)",
            "mentor_decisions": "D1-D8, 2026-11-14",
        },
        "evidence_classes": {
            "observed_dataset": "read from the dataset", "model_generated": "computed from fitted models and data",
            "assumed_scenario_input": "assumed operational parameter (hypothetical)",
            "simulated_outcome": "computed from model results and assumed inputs (hypothetical)"},
        "provenance": {
            "dataset": dataset, "git": git_state(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__, "scipy": installed_version("scipy"),
                            "matplotlib": matplotlib.__version__},
            "seeds": settings["seeds"],
            "pre_registered_configuration_sha256": canonical_sha256(settings),
        },
        **results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5B-3: calibration, E1/E2 and the champion decision.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    output = run(figures_dir=args.figures_dir)
    record = write_result_json(output, args.output)
    print(f"wrote {record['path']} ({record['bytes']} bytes, {record['status']})")
    e1, e2, decision = output["e1_conditional_bootstrap"], output["e2_selection_aware"], output["champion_decision"]
    for minutes, row in e1["schemes"]["cbb"]["by_block_minutes"].items():
        d = row["delta_ap_hgb_minus_lr"]
        print(f"E1 CBB {minutes:>2} min: delta AP {d['observed']:+.4f}  95% [{d['lower']:+.4f}, {d['upper']:+.4f}]")
    print(f"E1 passed: {e1['gate']['passed']}; E2: {e2['gate']}")
    print(f"champion: {decision['champion_config_id']} ({decision['statement']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
