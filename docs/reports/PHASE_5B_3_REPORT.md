# Phase 5B-3 Report — Calibration, Criteria E1 and E2, Champion Decision

> **Status (2026-11-14):** implemented and verified; awaiting mentor review; not committed.
> **Result type:** development results on validation (42,722 rows, 56 frauds). No test row was fitted, scored or used. No operational threshold or operating point is selected.
> **Results file:** `reports/phase5b3_results.json`. **Figures:** `reports/figures/phase5b3_*.png`.
> **Evidence labels:** every number below is `observed_dataset`, `model_generated`, `assumed_scenario_input` or `simulated_outcome`. The last two are **hypothetical**. Section 6 is the only section that contains them, and it is marked as such.

## 1. Scope and governing decisions

Protocol Section 4.2, steps 4 (calibration), 6 (uncertainty: E1 and E2) and 7 (champion decision), plus the fixed-K part of step 5 as supporting evidence. The mentor decisions of 2026-11-14 govern the phase:

| # | Decision | How it is applied |
|---|---|---|
| D1 | E1 and E2 are the **only** champion gates; LR is the default | `champion_decision(e1, e2)` takes the two gate records and nothing else. Calibration and practical relevance are reported as supporting evidence |
| D2 | No numeric AP threshold for practical improvement | The practical unit is additional frauds caught at equal alert volume (ΔTP@K), with paired bootstrap intervals; break-even, R-grid sensitivity and analyst-hours are hypothetical |
| D3 | Fixed-K, break-even and sensitivity now; time-based queue and per-scenario thresholds deferred | **Logged ordering deviation:** steps 6–7 ran before the deferred part of step 5. The champion rule uses step 6 only, so the decision does not depend on the order |
| D4 | Full Section 6.5 | O/E, intercept, slope, Brier, log loss, reliability, both bootstraps, isotonic and C-B sensitivities |
| D5 | Calibration-evaluation bootstrap: CBB, 15 min, B = 5,000 | Separate analysis from E1; fresh `default_rng(0)` |
| D6 | Own Platt fitter with public APIs, checked against `CalibratedClassifierCV` + `FrozenEstimator` | Section 3.2; the measured difference is documented, not assumed away |
| D7 | Zero-fraud E1 replicates dropped and counted; ΔAP = 0 is not positive; stationary bootstrap with geometric lengths of mean L, truncated at T | Implemented and tested; no E1 replicate had zero frauds |
| D8 | Reliability PNGs in `reports/figures/`, frozen log-scaled decade bins | Three figures; the scenario figure carries the hypothetical label |

Nothing frozen changed: the configuration SHA-256 is still `86ab5adb…`. The evidence labels live in result files and documentation, not in `src/models/phase5_settings.py`.

## 2. Implementation [V]

| Module | Role |
|---|---|
| `src/evaluation/evidence_labels.py` | Four evidence classes, disclosure text, result and documentation checks |
| `src/evaluation/block_bootstrap.py` | CBB / MBB / stationary replicate weights; weighted AP and TP@K; paired bootstrap; jackknife; dependence diagnostic |
| `src/evaluation/calibration.py` | Platt fit (damped Newton, public NumPy/SciPy only); failure rule; Section 6.5 metrics; reliability tables; bootstraps (A) and (B); isotonic sensitivity |
| `src/evaluation/e2_stability.py` | Criterion E2 with re-selection among all 28 cached score vectors |
| `src/models/phase5b3_champion.py` | `e1_gate`, `champion_decision` |
| `src/evaluation/practical.py` | Fixed-K comparison (model-generated); scenario inputs and outcomes (hypothetical) |
| `src/visualization/phase5b3_figures.py` | Reliability diagrams; labelled scenario figure |
| `src/models/run_phase5b3.py` | Runner; CLI `python -m src.models.run_phase5b3` |

**Order of operations in the runner:**
1. The 28 score vectors are recomputed (scores are never stored) and must reproduce `reports/phase5b2_grid_results.json` exactly. They do: all 28 APs are identical and the selection is LR-10 / HGB-14.
2. Calibrators are fitted on `T_cal` only.
3. The K-derived reliability edges are computed from validation **scores** before any validation label enters a calibration metric.
4. Calibration evaluation, E1, E2, the decision, then the fixed-K comparison.

Total runtime is about 6.5 minutes. The largest parts are the calibration-evaluation bootstrap (about 150 s), the grid recompute (about 105 s) and E1 (about 57 s).

## 3. Calibration (supporting evidence; not a champion gate) [model_generated]

### 3.1 Calibrators on `T_cal` (39,873 rows, 28 frauds)

| Model | A | B | Logit SD on `T_cal` | Status |
|---|---:|---:|---:|---|
| LR-10 | 0.5132 | −7.386 | 1.196 | valid (A > 0, converged, not constant) |
| HGB-14 | 0.4615 | −3.165 | 0.916 | valid |

Both models are calibrated; neither triggers the technical-failure rule. Calibration preserves the ranking exactly: AP on the calibrated logit equals AP on `decision_function` (0.853574 and 0.868638), with the same number of unique values.

**Calibrator-parameter bootstrap (A)** (CBB of `T_cal`, 15 min, B = 2,000, seed 42, identical rows for both models): all 2,000 replicates are valid for each model, with no low-positive replicate. The 95% intervals are:
- LR: A [0.458, 0.605], B [−8.14, −6.93];
- HGB: A [0.412, 0.530], B [−4.02, −2.55].

### 3.2 Own fitter vs the public scikit-learn calibrator (D6)

| Model | Max \|Δ logit\| | Max \|Δ p\| | Gradient of own fit | Gradient of public fit | Objective, own − public |
|---|---:|---:|---:|---:|---:|
| LR-10 | 2.1 × 10⁻⁶ | 1.2 × 10⁻⁷ | 3.3 × 10⁻¹¹ | 1.2 × 10⁻⁵ | −5.8 × 10⁻¹³ (own not worse) |
| HGB-14 | 4.7 × 10⁻⁹ | 9.1 × 10⁻¹⁰ | 1.3 × 10⁻⁷ | 2.1 × 10⁻⁷ | 0.0 (equivalent within 10⁻⁶) |

Gradients are the larger of |Σ(p − t)| and |Σ(p − t)·z| over the 39,873 `T_cal` rows; both vanish at the optimum.

**The LR difference is above the 10⁻⁶-logit equivalence flag. It is documented, not accepted silently.** Both fitters minimize the same strictly convex Platt objective. The own fit sits at the optimum (gradient 3.3 × 10⁻¹¹). scikit-learn rescales scores with max |s| ≥ 30 (LR-10: 46.5) and stops L-BFGS-B at gtol 10⁻⁶, leaving a gradient of 1.2 × 10⁻⁵ in original units. The difference is therefore optimizer tolerance, not a different calibrator. The own fit is used. The exact figures are under `calibration.public_api_equivalence` in the results file.

### 3.3 Validation calibration (42,722 rows, 56 frauds; development figures)

| Metric | LR-10 | HGB-14 |
|---|---|---|
| Mean calibrated p (prevalence 0.00131) | 0.00107 | 0.00100 |
| O/E [Garwood 95%] | 1.22 [0.92, 1.59] | 1.31 [0.99, 1.70] |
| O/E, block bootstrap 95% | [1.05, 1.42] | [1.14, 1.50] |
| Calibration intercept [profile 95%] | +0.59 [0.14, 1.00] | +0.74 [0.31, 1.14] |
| Calibration slope [profile 95%] | 1.25 [1.09, 1.43] | 1.33 [1.16, 1.53] |
| Brier score [bootstrap 95%] | 0.000390 [0.000226, 0.000565] | 0.000395 [0.000239, 0.000560] |
| Brier skill vs constant prevalence | 0.70 | 0.70 |
| Log loss [bootstrap 95%] | 0.00227 [0.00134, 0.00324] | 0.00216 [0.00131, 0.00306] |

- **Paired LR − HGB (block bootstrap):** the Brier difference is −0.0000059 [−0.0000816, +0.0000738] and the log-loss difference is +0.000095 [−0.000178, +0.000419]. **No calibration difference between the models is detectable.**
- **Reading:** both models **under-predict on validation**, by about 20–30% in O/E. Their slope above 1 means the calibrated spread is too narrow. The direction is the one Section 6.4 anticipates: `T_cal` (hours 7–12, prevalence 0.070%) has a lower fraud rate than validation (0.131%). This is a dataset property (prevalence shift by hour), not a calibrator failure, and it is **not acted upon** (Section 6.5: no method switch, no re-sizing, no prevalence correction). The `T_cal` in-sample O/E is 1.00 for both models (optimistic by construction).
- **Windows (descriptive):** O/E is 1.22 / 1.22 for LR and 1.32 / 1.30 for HGB in W1 / W2. The shift is the same in both windows.

**Reliability (decade bins, frozen; figures `phase5b3_reliability_lr.png`, `phase5b3_reliability_hgb.png`):**
- Only the top bin [0.1, 1] is interpretable. It holds LR 53 rows, 44 frauds against 37.0 expected, and HGB 47 rows, 43 frauds against 34.2 expected.
- Every other bin has fewer than 5 observed or 5 expected frauds and is flagged "not interpretable".
- The K-derived bands show the same pattern: the top 25 rows hold 25 frauds against about 23–24 expected.

### 3.4 Sensitivities (reported only; they replace nothing)

| Variant | LR O/E | LR slope | HGB O/E | HGB slope |
|---|---:|---:|---:|---:|
| Sigmoid C-A (primary) | 1.22 | 1.25 | 1.31 | 1.33 |
| Sigmoid C-B 15% (24 frauds) | 1.06 | 1.16 | 1.17 | 1.29 |
| Sigmoid C-B 10% (20 frauds) | 0.97 | 1.15 | 1.08 | 1.30 |
| Isotonic C-A | 1.20 | n/a | 1.33 | n/a |

- The later, shorter C-B holdouts (closer to validation hours) under-predict less. This is consistent with the prevalence shift. As pre-registered, it is a sensitivity and does not change the holdout.
- Isotonic has the worst log loss (LR 0.00325, HGB 0.00388). It assigns probability 0 to some rows, so its log loss uses clipped probabilities, and its intercept and slope are not estimable.

## 4. Criterion E1 — conditional paired block bootstrap [model_generated]

Observed validation AP: LR-10 0.853574, HGB-14 0.868638. **ΔAP = +0.0151** (optimistic by selection).

| Block length | CBB (primary, B = 5,000) 95% | Share ΔAP ≤ 0 | MBB 95% | Stationary 95% |
|---|---|---:|---|---|
| 5 min (reported, not in envelope) | [−0.0107, +0.0467] | 0.147 | [−0.0112, +0.0472] | [−0.0103, +0.0465] |
| **10 min** | **[−0.0098, +0.0466]** | 0.138 | [−0.0132, +0.0440] | [−0.0108, +0.0434] |
| **15 min** | **[−0.0105, +0.0480]** | 0.155 | [−0.0127, +0.0459] | [−0.0108, +0.0421] |
| **30 min** | **[−0.0092, +0.0415]** | 0.150 | [−0.0114, +0.0410] | [−0.0102, +0.0380] |

- **E1 fails:** the CBB lower bound is below 0 at every envelope length (10, 15, 30 min).
- **Cross-checks:**
  - CBB centring bias is at most 0.001; MBB shows a slightly larger bias (−0.002), the expected endpoint effect.
  - The block-jackknife SE at 15 min is 0.0165 against a bootstrap SE of 0.0150. Every delete-one-block ΔAP is positive (0.003 to 0.020).
  - The i.i.d. stratified continuity interval is [−0.0082, +0.0410]; it ignores temporal dependence.
  - No replicate had zero frauds (mean 56 per replicate).
- **Dependence diagnostic (Section 7.4, descriptive):** at K = 50 and 100, TP, FN and discordant frauds show no dispersion beyond the permutation null at 15 or 30 min. The only p < 0.05 is HGB TP at K = 50 at 10 min (p = 0.038), out of 40 tests. Nothing indicates dependence beyond 30 min, so there is no reason to expect the block CIs to be systematically too narrow on that account. The decision rule is not re-tuned.
- **Reading:** the interval contains 0, so there is **no evidence of a difference at this sample size**. This is not evidence of equivalence. About 85% of replicates favour HGB, which is not a probability or a significance level.

## 5. Criterion E2 — selection-aware held-out block splits [model_generated]

Frozen design: 21 × 15-min blocks, 10 selection / 11 held-out, 300-s buffers, `default_rng(42)`, re-selection among all 28 configurations on each selection half.

| Quantity | Value | Gate |
|---|---|---|
| Splits attempted / usable / skipped | 2,000 / **1,984** / 16 | ≥ 1,000 usable: estimable |
| Held-out split-stability frequency | **77.2%** | ≥ 90%: **fails** |
| Median held-out ΔAP | **+0.0124** | > 0: passes |
| Held-out ΔAP, 5th–95th percentile | −0.013 to +0.050 | — |
| Held-out frauds per usable split | median 20 (10–34) | — |

**E2 fails**, because the frequency is below 90%. The frequency is a descriptive resampling stability statistic. It is not a probability that HGB is better and not a confidence level; every split reuses the same 21 blocks and 56 frauds.

- **Selection stability (k̂):**
  - LR picks LR-03 (25.8%), LR-10 (23.3%), LR-08 (18.1%), LR-01 (16.6%) and LR-12 (15.5%).
  - HGB picks HGB-06 (37.3%), HGB-16 (26.3%) and HGB-14 (22.8%).
  - The full-validation choices are therefore far from stable under half-sample re-selection.
- **Optimism indicator (heuristic; not an unbiased estimate):**
  - Unadjusted selection-minus-held-out AP is 0.042 for both families.
  - The reference LR (Phase 4 configuration) has 0.033.
  - Adjusted, that leaves 0.009 for both families. No family-specific optimism is visible.

## 6. Practical relevance at equal alert volume (supporting evidence)

### 6.1 Frauds caught at fixed K [model_generated]

| K (alerts) | TP LR-10 | TP HGB-14 | ΔTP (HGB − LR) | Paired CBB 15-min 95% for ΔTP | Tie-sensitive |
|---:|---:|---:|---:|---|---|
| 25 | 25 | 25 | 0 | [0, 0] | no |
| 50 | 44 | 44 | 0 | [−3, +4] | no |
| 100 | 49 | 48 | −1 | [−3, +3] | no |
| 250 | 51 | 51 | 0 | [−3, +3] | no |
| 500 | 52 | 52 | 0 | [0, 0] | no |

At equal alert volume, the two models catch the same frauds within one: HGB catches one fewer at K = 100. Every paired interval contains 0. **The +0.015 AP difference does not translate into additional frauds caught at any frozen capacity.** At equal K this statement needs no cost assumption: the model with more TP is cheaper for every R > 0, and here neither has more TP except LR by one at K = 100.

### 6.2 Hypothetical scenarios [assumed_scenario_input → simulated_outcome] — **HYPOTHETICAL**

> **HYPOTHETICAL.** The inputs below are **assumed**, not observed: the review-cost ratio R ∈ {5, 10, 20, 50, 100, 200, 500} (missed-fraud cost in review units) and 5 / 10 / 15 minutes per review, both frozen protocol grids `[H]`. No real investigation cost, financial loss or operational capacity was provided. Every figure in this subsection is a **simulated outcome** in hypothetical review units, **not** a business result or a saving. No operating point is selected.

| K | Break-even R (review vs none) = K/TP, both models | Marginal R\* = ΔAlerts/ΔTP from the previous K (LR / HGB) | Simulated analyst-hours at assumed 5 / 10 / 15 min |
|---:|---:|---|---|
| 25 | 1.00 | — | 2.1 / 4.2 / 6.3 |
| 50 | 1.14 | 1.3 / 1.3 | 4.2 / 8.3 / 12.5 |
| 100 | 2.04 (LR), 2.08 (HGB) | 10.0 / 12.5 | 8.3 / 16.7 / 25.0 |
| 250 | 4.90 | 75 / 50 | 20.8 / 41.7 / 62.5 |
| 500 | 9.62 | 250 / 250 | 41.7 / 83.3 / 125.0 |

- **Between models at equal K (simulated):** R\* is defined only at K = 100, where ΔTP = −1. There, LR is cheaper for every assumed R > 0, by R review units (5 to 500). At every other K, R\* is not defined (ΔTP = 0) and the simulated costs are identical for every R in the grid.
- **R-grid sensitivity (simulated):** reviewing the top K beats no review in every scenario except K = 500 at R = 5 (−240 review units). Moving from K = 250 to 500 pays only if a missed fraud is assumed to cost more than 250 reviews.
- **Bayes-threshold alert counts (simulated; label-free):** under p\* = 1/R, LR raises 46–247 alerts and HGB 44–317 alerts across the R grid. The calibrator bootstrap ranges are wide at large R (LR at R = 500: [113, 553]). No threshold is selected (per-scenario thresholds are deferred, D3).
- **Figure:** `reports/figures/phase5b3_fixed_k_scenarios.png`. Its title reads "HYPOTHETICAL SCENARIO: assumed inputs, simulated outcomes (not observed business results)", and its panels separate the model-generated counts from the simulated analyst-hours.

## 7. Champion decision [model_generated]

| Gate | Result |
|---|---|
| E1: CBB lower bound > 0 at 10, 15 and 30 min | **fail** (−0.0098, −0.0105, −0.0092) |
| E2: ≥ 1,000 usable splits, frequency ≥ 90%, median > 0 | **fail** (1,984 usable; 77.2%; +0.0124) |

**Champion: LR-10** (Logistic Regression, C = 1, `class_weight="balanced"`, the Phase 4 configuration). Challenger: HGB-14.

**LR is retained because there is no sufficient evidence for replacement at this sample size. This is not a finding that LR is better.** With 56 validation frauds the rule has low power, and HGB could genuinely be better while failing both gates. The decision used only the E1 and E2 gate records. Calibration, fixed-K results and scenarios were not inputs; they agree with the decision but could not have changed it. The decision is recorded before any W1 → W2 replay or test evaluation. Both frozen models will be scored on test at the end regardless.

## 8. Leakage and test-set isolation [V]

- Test rows (42,721) are removed before any label access; the maximum processed `Time` is 151,328.
- Calibrators are fitted on `T_cal` only. A test recomputes them from `T_cal` scores and labels alone and checks equality.
- The K-derived edges take scores only; their function signature has no label argument.
- E2 re-selection reads selection-half labels only. A test flips held-out labels and checks that the selection does not change.
- Synthetic runs give identical results with test labels flipped and test features set to 10⁶.
- The champion function has exactly two parameters; a test passes extra fields such as calibration or savings and checks that the decision ignores them.

## 9. Reproducibility [V]

- Seeds:
  - E1: `default_rng(0)` per (scheme, block length);
  - bootstrap (B): `default_rng(0)`;
  - `T_cal` bootstrap: `default_rng(42)` per model;
  - E2: `default_rng(42)`;
  - permutations, jackknife and i.i.d. continuity: 0.
- First replicates of each scheme are pinned in tests (NumPy 2.3.2).
- A second full run reproduced `reports/phase5b3_results.json` exactly except the wall-clock `runtime_seconds` field (Section 10).
- The result file is 0.32 MB, within the 5 MB target. It stores no per-row or per-replicate arrays.

## 10. Tests [V]

| File | Tests | Covers |
|---|---:|---|
| `tests/test_phase5b3_calibration.py` | 24 | Platt recovery, optimality, equivalence with the public calibrator (LR, HGB), weights = expanded sample, scale/shift invariance, separation, failure reporting and classification; Garwood, Clopper–Pearson, Brier and log loss vs scikit-learn, logit-based log loss, profile intervals, miscalibration detection, negative slope kept; reliability partition and interpretability flag; label-free K edges; bootstraps (A) and (B) incl. not-estimable and zero-fraud cases |
| `tests/test_phase5b3_bootstrap.py` | 20 | Block lengths, circular wrap, exact duration for all schemes, pinned seeded draws, equal CBB inclusion and MBB end down-weighting, shared timestamps; weighted AP and TP@K = expanded resample with ties; pairing, reproducibility, zero-fraud dropping; jackknife, i.i.d. continuity, dependence diagnostic; E1 gate boundaries; champion truth table and two-input signature |
| `tests/test_phase5b3_e2.py` | 11 | Frozen constants; gate boundaries (exactly 90%, median 0, 999/1,000 usable); clear pass, clear fail, ΔAP = 0 not positive; re-selection on the selection half only; held-out labels do not affect selection; not estimable computes nothing; undefined case refused |
| `tests/test_phase5b3_run.py` | 19 | Frozen replicate counts; synthetic end-to-end (evidence classes, champion from gates only, E1 structure, ranking preserved, calibrator and K edges from allowed inputs only, test isolation, rerun identity, figures, grid-mismatch stop); practical relevance by hand; recorded-results regression |
| `tests/test_hypothetical_disclosure.py` | 13 | Evidence-class helpers; scenario content outside hypothetical blocks rejected; every scenario section of the living documents, the 5B-3 work-log entries and notebooks disclosed; documents name all four classes; scenario figure title; recorded results; frozen hash unchanged |

Focused: 87/87. Full suite: **421/421** with `-W error::FutureWarning` (334 before Phase 5B-3). The figures are byte-identical across runs.

## 11. Limitations

- **Sample size:** 56 validation frauds, about 20 per held-out half. Both gates have low power; "retained" means "not enough evidence", not "LR is better".
- **Calibration:**
  - 28 calibration positives;
  - a prevalence shift between `T_cal` (morning hours) and validation that no train-internal holdout can remove;
  - O/E cannot resolve miscalibration below roughly ±25–30%.
- **E1:** block-bootstrap intervals with 11–31 effective blocks are approximate and possibly anti-conservative.
- **E2:** a heuristic. The splits share the same blocks, and the optimism indicator is not unbiased.
- **Scenarios:** every cost and capacity value is hypothetical. Fixed-K is retrospective; the time-based queue is deferred (D3).
- **Fitter equivalence:** the LR calibrator differs from scikit-learn's public calibrator by 2 × 10⁻⁶ in logit, because scikit-learn stops earlier (Section 3.2).
- **Not done:** the time-based queue, per-scenario thresholds, the W1 → W2 replay (step 8), the freeze manifest (step 9) and the final test evaluation (step 11).
