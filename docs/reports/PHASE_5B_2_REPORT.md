# Phase 5B-2 Report — Controlled Model Experiments E0–E3

> **Status (2026-11-11):** implemented and verified; **awaiting mentor review**; not committed. Phase 5B-2 was started on the project owner's explicit instruction, which set the scope: four experiments (E0–E3), three evaluation tracks, the metric set, the temporal-stability and capacity analyses, and the safeguards. Phase 5 is **not** complete.
>
> **Evidence labels:** **[V]** verified by execution in this phase; **[D]** quoted from a repository document; **[R]** recommendation or interpretation.
>
> **Naming:** the experiment IDs **E0–E3** are unrelated to the protocol's champion criteria **E1/E2** (protocol Section 4.3). This report writes "experiment E2" for the HGB experiment and "criterion E2" for the protocol gate.

## 1. Scope, and how it relates to the approved protocol

| Experiment | Model | Features | Role |
|---|---|---|---|
| **E0** | Logistic Regression, Phase 4 configuration | `V1`–`V28` + `Amount` (29) | Incumbent; identical to the Phase 5B-1 primary track |
| **E1** | Same as E0 | E0 + `log1p(Amount)` (30) | Effect of `log1p(Amount)` within LR |
| **E2** | HistGradientBoosting, one fixed configuration | `V1`–`V28` + `Amount` (29) | Model-family effect against E0 |
| **E3** | Same as E2 | E2 + `log1p(Amount)` (30) | Effect of `log1p(Amount)` within HGB |

**Relationship to the protocol [D][R]:**
- **Protocol deviation, logged:** protocol Section 5.5 explicitly excludes feature engineering, naming `log(Amount)`. E1 and E3 therefore run outside the pre-registered design, on the owner's instruction. They are **exploratory development results**: they do not enter the champion rule, and no feature set is adopted from them. Adopting `log1p(Amount)` later would be a validation-informed change and would need mentor approval.
- **No tuning.** The HGB configuration is **one point of the frozen 16-configuration grid** (Section 5.3), chosen a priori before any Phase 5B-2 score existed (Section 3). The LR is the Phase 4 configuration, which is one point of the frozen 12-configuration LR grid. Both will be refitted identically when the full grid runs (Section 4.2 step 2).
- **What this phase is not:** it is not the 28-configuration grid, not hyperparameter selection, not the champion decision (criteria E1 + E2), and not calibration, queue simulation, cost analysis or the W1 → W2 decision replay. Those remain later stages of the protocol.
- No value in `src/models/phase5_settings.py` changed. The configuration SHA-256 is still `86ab5adb…` **[V]**.

## 2. Implementation

New modules only. No Phase 4, Phase 5B-0 or Phase 5B-1 module was changed.

| File | Role |
|---|---|
| `src/features/amount_transform.py` | `log1p_amount`, `add_log1p_amount`: stateless, row-wise; appends `log1p_Amount` and keeps `Amount`; never modifies its input; refuses negative, missing or infinite amounts, a missing column or an existing output column |
| `src/evaluation/capacity.py` | `fixed_k_metrics`, `fixed_k_table`: Precision@K, Recall@K (fraud capture rate), TP/FP, alert share, oracle ceiling and the protocol Section 9.2 boundary-tie report (`[tp_min, tp_max]`, expected TP under a random tie order). The order uses the frozen key (score ↓, `Time` ↑, file position ↑), which never reads a label |
| `src/models/phase5b2_experiments.py` | Experiment and feature-set definitions, `HGB_PARAMS`, `build_hgb`, `prepare_frames` (refuses non-`T_fit` fit rows and non-validation scoring rows), `fit_experiment`, metric helpers, `validate_experiment_tracks`, `heldout_half_stability` |
| `src/models/run_phase5b2.py` | CLI. Verifies the dataset SHA-256; removes test rows straight after the label-free chronological assignment (reusing `run_phase5b1.drop_test_rows`); fits E0–E3 once each on `T_fit`; checks HGB determinism; evaluates; writes `reports/phase5b2_results.json` |
| `tests/test_phase5b2.py` | 41 tests (Section 10) |
| `reports/phase5b2_results.json` | 125,670 bytes, strict JSON, within the 5 MB target **[V]** |

Run: `python -m src.models.run_phase5b2` (about 65 s, including the two HGB determinism refits).

**Reuse.** Every fit goes through the Phase 4 runner `fit_and_score`, which refuses a test evaluation partition. Ranking scores come from `ranking_scores` (`decision_function` only, no fallback). The LR builder and refit rule are the Phase 5B-1 ones (`build_continuity_lr`; refit at `max_iter` 5,000 on a convergence warning). Windows come from `split_two_time_windows` (Phase 4c). The held-out halves use the frozen E2 block, draw and buffer functions. Track A is read with `quote_phase4_historical`.

## 3. Configurations (fixed before any Phase 5B-2 score)

| | Logistic Regression (E0, E1) | HistGradientBoosting (E2, E3) |
|---|---|---|
| Fixed settings | C = 1, `class_weight="balanced"`, lbfgs, `max_iter` 1,000 (5,000 on a convergence warning), L2 (scikit-learn default), `random_state` 42 | `learning_rate` 0.1, `max_leaf_nodes` 31, `min_samples_leaf` 20, `class_weight="balanced"`, `max_iter` 300, `early_stopping=False`, `l2_regularization` 1.0, `max_bins` 255, `random_state` 42 |
| Preprocessing | StandardScaler inside the pipeline, fitted on `T_fit` | None (a column selector only). Tree splits do not need scaling (protocol Section 5.3) |
| Why these values | The Phase 4 benchmark, unchanged | The three tuned parameters are scikit-learn 1.9.1's defaults. `balanced` follows the incumbent's imbalance convention. The fixed values are the protocol's. `early_stopping=False` prevents a hidden random (non-temporal) validation split inside training |
| Fit outcome **[V]** | E0 converged in 40 iterations, E1 in 45; no warning, no refit | 300 trees each; two fits gave bit-identical `decision_function` scores (E2 and E3) |

`class_weight` is supported by `HistGradientBoostingClassifier` in the installed scikit-learn 1.9.1 **[V]**. `T_fit` has 159,491 rows, below HGB's 200,000-row binning subsample, so binning is exact and the fit is deterministic **[V]**.

## 4. Evaluation methodology

**Data [V]:** fit on `T_fit` (`Time` 0–112,553; 159,491 rows; 356 frauds). Every experiment scores the **same 42,722 validation rows (56 frauds)**, checked row by row. `T_cal` is neither fitted nor scored; the test partition is never read.

**Three tracks, never mixed:**

| Track | Fit | Score | Role | Applies to |
|---|---|---|---|---|
| **A — historical** | full train (199,364; 384 frauds) | `predict_proba` | Phase 4 benchmark, **quoted** from `reports/phase4b_baseline_results.json`, never recomputed | E0 only. E1–E3 are marked `not_applicable`: no historical benchmark exists for them |
| **B — primary** | `T_fit` | `decision_function` | Phase 5 primary | E0–E3 |
| **C — diagnostic** | the same fit as B | `predict_proba` | Separate score representation; never used for selection or primary evaluation | E0–E3 |

Every metric block carries `score_representation`, and `validate_experiment_tracks` rejects a missing or mismatched one. Neither score is calibrated. LR and HGB raw scores are on different scales and are compared through rank-based metrics only.

**Metrics:** primary **AP**. Secondary: ROC-AUC, Precision@K, Recall@K (fraud capture rate), TP and FP (legitimate transactions in the queue), and the estimator's default `predict()` rule as a diagnostic.

**Capacities:** K ∈ {25, 50, 100, 250, 500} on full validation and {13, 25, 50, 125, 250} per window. These are the protocol's frozen Section 9.2 grids **[D]**. They are **illustrative alert-capacity scenarios, not business operating limits**; the project has no documented real capacity. For scale, validation lasts 5.11 h, so K = 50 is about 10 alerts/hour **[D]**.

**Temporal stability:**
- the existing Phase 4c windows W1 (`Time` 132,929–142,280; 21,362 rows; 23 frauds) and W2 (142,281–151,328; 21,360 rows; 33 frauds), boundary reproduced exactly **[V]**;
- the buffered held-out halves of the 2,000 frozen E2 splits (21 blocks of 15 min, `default_rng(42)`, 300-s buffers, usable iff ≥ 10 held-out frauds). The configurations are **fixed**, so this is a descriptive stability check. It is **not** criterion E2, which re-selects over the 28-configuration grid.

**Uncertainty:** the project has no bootstrap utility (the Phase 4 intervals were audit computations without a script [D]), and the protocol's block bootstrap belongs to a later stage. No new bootstrap was introduced here. Uncertainty is shown through the window split, the held-out-half distribution and the fraud counts.

## 5. Results [V]

### 5.1 Ranking metrics (validation; 42,722 rows; 56 frauds)

| Exp. | Track B AP | Track B ROC-AUC | Track C AP | Track C ROC-AUC | Track A AP / ROC-AUC |
|---|---:|---:|---:|---:|---|
| E0 | 0.853574 | 0.981031 | 0.853574 | 0.981031 | 0.839660 / 0.981517 (quoted; full train) |
| E1 | 0.853869 | 0.980860 | 0.853869 | 0.980860 | not applicable |
| E2 | **0.868638** | 0.979531 | 0.868638 | 0.979531 | not applicable |
| E3 | **0.868638** | 0.979531 | 0.868638 | 0.979531 | not applicable |

- **E0 reproduces the Phase 5B-1 primary track exactly** (AP difference 0).
- **Track B = Track C for all four experiments.** No probability reaches 1.0. The maximum logit is 31.55 for LR and 15.35 for HGB, below the 36.74 saturation point, so the two representations give identical rankings here **[V]**. They remain separate records: on other configurations or data they need not agree.
- E0 track B − track A = +0.0139 AP. As in Phase 5B-1, this mixes the `T_fit` change, the 5.66 h gap and the representation; it is not evidence that `T_fit` training is better **[D]**.

### 5.2 Contrasts (track B; same validation rows)

| Contrast | Isolates | ΔAP validation | ΔAP W1 | ΔAP W2 | ΔROC-AUC validation |
|---|---|---:|---:|---:|---:|
| E1 − E0 | `log1p(Amount)` in LR | +0.00029 | −0.00013 | −0.00014 | −0.00017 |
| E3 − E2 | `log1p(Amount)` in HGB | **0 (exactly)** | 0 | 0 | 0 |
| E2 − E0 | HGB vs LR, baseline features | **+0.01506** | +0.00292 | +0.02385 | −0.00150 |
| E3 − E1 | HGB vs LR, with `log1p(Amount)` | +0.01477 | +0.00304 | +0.02399 | −0.00133 |

**`log1p(Amount)` in LR:** negligible. The full-validation ΔAP is positive, yet it is negative in **both** windows. Pooled AP is not an average of window APs: it also depends on how scores from the two windows interleave. A sign that flips between pooled and per-window views is noise, not an effect **[R]**.

**`log1p(Amount)` in HGB: no effect, by construction [V].** E3 scores are bit-identical to E2's. Inspection of the fitted E3 model showed why: `log1p` is strictly increasing, so the quantile binning puts the training rows into the same 189 bins for `Amount` and `log1p_Amount`. Every candidate split on either column therefore has the same gain, and the tie goes to the earlier column (`Amount`). `log1p_Amount` was never used in any split. Every validation row also falls in the same bin under both columns. A synthetic regression test confirms the mechanism (`test_hgb_invariant_to_monotone_log1p_duplicate`). The lesson is that monotone transforms are useful for linear models and irrelevant to tree ensembles **[R]**.

### 5.3 Temporal stability

**Per window (track B):**

| Exp. | W1 AP (23 frauds) | W1 ROC-AUC | W2 AP (33 frauds) | W2 ROC-AUC |
|---|---:|---:|---:|---:|
| E0 | 0.8083 | 0.9575 | 0.8925 | 0.9968 |
| E1 | 0.8081 | 0.9574 | 0.8924 | 0.9966 |
| E2 / E3 | 0.8112 | 0.9535 | 0.9164 | 0.9974 |

- **AP ranking:** validation E2 = E3 > E1 > E0; W1 and W2 both E2 = E3 > E0 > E1. HGB ranks first in every view. The LR pair swaps between pooled and per-window views, consistent with the noise reading of Section 5.2.
- **HGB's advantage is concentrated in W2:** +0.024 AP in W2 and +0.003 in W1. In W1, HGB has the *lower* ROC-AUC (−0.004). All four models score much higher in W2 than in W1. W2's prevalence is higher (0.154% vs 0.108%), and raw AP rises with prevalence **[D]**, so the window difference is not drift.

**Held-out halves of the 1,984 usable frozen E2 splits** (median 20 held-out frauds, range 10–34; descriptive, not criterion E2):

| Contrast | Median ΔAP | 5th–95th percentile | Share of splits > 0 |
|---|---:|---:|---:|
| E2 − E0 (HGB vs LR) | +0.0162 | −0.0115 to +0.0573 | 80.3% |
| E3 − E1 | +0.0164 | −0.0117 to +0.0572 | 80.1% |
| E1 − E0 (`log1p` in LR) | −0.00004 | −0.0015 to +0.0018 | 44.5% |
| E3 − E2 (`log1p` in HGB) | 0 | 0 to 0 | 0% (identical) |

Held-out AP medians: E0 0.821, E1 0.820, E2/E3 0.839. HGB has the highest held-out AP in 79.6% of splits. E3 is credited 0% only because exact ties go to the earlier ID; E2 and E3 are identical. All splits reuse the same 21 blocks and 56 frauds, so these are not independent replications. The spread describes sensitivity to *which rows of this afternoon* are evaluated. It does not describe uncertainty about other periods, and the shares are not probabilities.

### 5.4 Capacity view: full validation, track B

| K (alert share) | E0 TP / FP | E1 TP / FP | E2, E3 TP / FP | Precision@K (E0 / E2) | Recall@K (E0 / E2) | Oracle recall ceiling |
|---:|---|---|---|---|---|---:|
| 25 (0.06%) | 25 / 0 | 25 / 0 | 25 / 0 | 1.00 / 1.00 | 0.446 / 0.446 | 0.446 |
| 50 (0.12%) | 44 / 6 | 44 / 6 | 44 / 6 | 0.88 / 0.88 | 0.786 / 0.786 | 0.893 |
| 100 (0.23%) | 49 / 51 | 49 / 51 | 48 / 52 | 0.49 / 0.48 | 0.875 / 0.857 | 1.00 |
| 250 (0.59%) | 51 / 199 | 51 / 199 | 51 / 199 | 0.204 / 0.204 | 0.911 / 0.911 | 1.00 |
| 500 (1.17%) | 52 / 448 | 52 / 448 | 52 / 448 | 0.104 / 0.104 | 0.929 / 0.929 | 1.00 |

No boundary is tie-sensitive at any K, for any model, in either track (`tp_min = tp_max`) **[V]**. Four frauds are missed by every model even at K = 500.

**Per window (track B):** in W1 at K = 25, HGB catches 19 frauds and LR 18; at K = 125, LR catches 20 and HGB 19. In W2 at K = 50, LR catches 30 and HGB 29; at K = 125, HGB catches 32 and LR 31. All other window K values are equal. Each difference is one fraud.

**Default `predict()` rule (diagnostic only; not an operating point):** balanced LR (E0, E1) flags 1,904 rows (4.5% of traffic; 52 TP, 1,852 FP). HGB (E2, E3) flags 61 (47 TP, 14 FP). This shows only that "logit > 0" has a different meaning on each model's uncalibrated scale. No threshold is selected from it.

### 5.5 Computational cost (wall clock; one 8-core laptop; not reproducible to the second)

| | LR (E0 / E1) | HGB (E2 / E3) |
|---|---:|---:|
| Fit + `predict_proba` on validation | 0.50 s / 0.55 s | 9.97 s / 10.18 s (second fits 11.1 s / 11.9 s) |
| `decision_function`, 42,722 rows | 0.005 s / 0.003 s | 0.154 s / 0.154 s |

HGB costs about 20× more to fit and about 30× more to score. Both are trivial at this data size. Runtime is the only non-reproducible field in the result file (key `runtime_seconds`).

## 6. Interpretation [R]

**Which model performed best?** Experiment E2 (= E3), HistGradientBoosting, has the highest AP in every view: full validation (+0.015 over E0), both windows, and 80% of the held-out halves. ROC-AUC tells a different story: HGB is slightly lower on full validation and in W1.

**Was the advantage stable?** Only partly:
- It is concentrated in W2 (+0.024) and nearly absent in W1 (+0.003).
- The held-out 5th percentile is negative (−0.012).
- It rests on 56 frauds. The Phase 4 bootstrap interval for one model's AP was about 0.18 wide ([0.746, 0.922]) **[D]**, an order of magnitude wider than this difference.
- It is a single a-priori configuration. It is not a selected model, and selection optimism is therefore not involved yet. It also means the other 15 HGB configurations could be better or worse.

**Does the AP gain matter operationally?** Not at the frozen illustrative capacities. At K = 25, 50, 250 and 500, all models catch the same frauds with the same number of legitimate alerts. At K = 100, HGB catches one fraud *fewer*. The AP gain comes from ordering *between* those K points, mostly in W2, which the capacity grid does not resolve. "Higher AP" does not automatically mean "more frauds caught at a given queue size".

**Does HGB justify its cost?** Its compute cost is negligible here. Its real costs are complexity, interpretability and a larger selection-optimism risk in the later grid stage (protocol Section 4.3). The protocol answers the question formally through criteria E1 + E2 after the grid, and this phase deliberately does not pre-empt that. On the current evidence, the favourable but small and window-dependent AP gain is consistent with the protocol's default of retaining LR unless both gates pass.

**`log1p(Amount)`:** no material effect on LR and exactly none on HGB. Nothing here supports adding it.

**No production readiness is claimed.** All figures are offline development results on one afternoon of one 2013 dataset.

## 7. Leakage and methodological safeguards [V]

| # | Safeguard | How it is enforced |
|---|---|---|
| 1 | No test evaluation | Test rows are removed before any label is read. `fit_and_score` refuses `evaluation_partition="test"`; `prepare_frames` refuses any scoring row outside the validation period. Results are invariant to flipping test labels and setting test features to 1e6, and a validation-label flip control changes them (synthetic tests) |
| 2 | Preprocessing fitted on training only | The scaler sits inside the pipeline. The test checks `n_samples_seen_ == len(T_fit)` and the fitted `Amount` mean. `log1p` has no parameters (test `test_row_wise_and_stateless`) |
| 3 | No validation-guided preprocessing | Feature sets and all parameters are module constants fixed before scoring. Nothing is chosen from the results |
| 4 | No `Time` feature | Asserted in `fit_experiment` and in tests for every experiment |
| 5 | Splits unchanged | Frozen `T_fit` boundary, Phase 4a partitions, Phase 4c window boundary (reproduced: 142,280) and the frozen E2 draws |
| 6 | Tracks not mixed | Separate blocks, each with a required `score_representation`, validated by `validate_experiment_tracks` |
| 7 | Phase 4 artifacts preserved | Track A is quoted read-only; a real-data test checks the Phase 4b file's SHA-256 before and after recomputation |
| 8 | No single-metric selection | No model is selected. AP, ROC-AUC, capacity, windows and held-out views are reported side by side |
| 9 | No undocumented tuning | One configuration per family, recorded in Section 3 and in the result file; no other configuration was fitted |
| 10 | No production claim | Section 6 |

Identical validation rows across experiments are asserted in `compute`. `T_cal` is neither fitted nor scored. **Disclosure:** as in Phase 5B-1, the consolidated suite includes pre-existing Phase 3/4 tests that read the full CSV and assert the already documented test-partition counts; no Phase 5B-2 code or test reads a test label.

## 8. Reproducibility [V]

- Rerunning the CLI produced a result file identical in every field except `runtime_seconds`.
- A real-data test recomputes everything and matches the recorded file to 1e-12.
- HGB determinism: two fits gave identical scores.
- Provenance (dataset SHA-256, Git state, Python 3.13.6, NumPy 2.3.2, pandas 2.3.1, scikit-learn 1.9.1, SciPy 1.17.1, seeds, configuration SHA-256) is recorded in the file.
- The file records `git.commit` = `aa4e63a`, `dirty: true`, because it was generated from the uncommitted working tree. It can be regenerated after a commit; only those two fields and `runtime_seconds` would change.

## 9. Limitations and uncertainty

- **56 validation frauds** (23 / 33 per window). One fraud is 1.8 percentage points of recall (4.3 in W1, 3.0 in W2). Every AP difference reported here is smaller than one model's bootstrap interval width.
- **No confidence intervals.** No project bootstrap utility exists, and the protocol's paired block bootstrap (criterion E1) is a later stage. The held-out halves are descriptive and reuse the same 56 frauds.
- **One HGB configuration.** It says nothing about the best HGB in the grid. The later grid stage will add selection optimism, which criterion E2 is designed to address.
- **Validation period:** one afternoon (hour values 12–18), about 5 hours, with no night hours **[D]**.
- **Uncalibrated scores.** Default-rule alert counts are not comparable across families. Nothing here is a probability statement.
- **`log1p` result for HGB** holds for this binning (exact, no subsampling). With binning subsampling (more than 200,000 rows), the bins could differ slightly.
- Inherited limitations: PCA look-ahead at the dataset level, and non-fraud model-input twins across partitions **[D]**.

## 10. Tests [V]

| Run | Result |
|---|---|
| Focused: `python -W error::FutureWarning -m unittest tests.test_phase5b2` | **41 passed** (about 75 s; includes one full real-data recomputation) |
| Consolidated: `python -m unittest discover -s tests` | **316 passed** (275 existing + 41 new), 0 failures, 0 skipped |

Coverage of `tests/test_phase5b2.py`:
- **Feature transform:** exact values, zero handling, appended column, input not modified, row-wise and stateless, refusals.
- **Definitions:** four experiments; pairs differ only by `log1p_Amount`; no `Time`; LR equals the Phase 4 parameters; HGB is a frozen grid point with `early_stopping=False`; each contrast isolates one factor.
- **Fitting (synthetic):** refusal of `T_cal`, test and train rows in the wrong role; scaler fitted on `T_fit` only; feature counts; identical validation rows; `decision_function` and `predict_proba` tracks equal the pipeline outputs, with `predict_proba = expit(decision_function)`; E0 equals the Phase 5B-1 primary track; HGB determinism; E3 ≡ E2 mechanism; LR scores change with `log1p`.
- **Fixed-K:** hand-computed P@K and R@K; the `Time`-then-position tie-break with the TP range and expected TP; label-independent order; K = 0, negative, fractional, boolean and K > n refused; no-fraud period gives NaN recall; length mismatch refused.
- **Tracks:** validator rejects wrong, missing or swapped representations and illegal `not_applicable` markings.
- **Metrics:** undefined AP and ROC-AUC; default-rule counts; windows partition the rows.
- **Held-out halves:** same usable count as `e2_usability`; identical scores give ΔAP 0; out-of-period refusal.
- **End-to-end (synthetic):** all tracks present; test isolation; test-label and test-feature invariance with a validation-label control.
- **Real data:** quoted track A equals the Phase 4b file; E0 equals the Phase 5B-1 primary; counts and settings; full recomputation matches the recorded file, and the Phase 4b file is unchanged.

## 11. Open items for mentor review [R]

1. **The `log1p(Amount)` deviation (E1, E3):** confirm that it is recorded as an owner-instructed exploratory deviation from Section 5.5 with no effect on the champion pathway. The results give no reason to adopt it.
2. **The next protocol stage is unchanged:** Section 4.2 step 2 (all 28 configurations on `T_fit`, score `T_cal` and validation once each), then selection, calibration, criteria E1 + E2, operational analysis and the replay. E0 and experiment E2 are grid points and will be refitted identically there.
3. **Bootstrap:** this phase reports no confidence intervals. The protocol's paired block bootstrap (criterion E1) is the planned source.
4. The test partition stays untouched until the freeze manifest and the test-scope acknowledgement. Do not write the final report until the project is complete.

---

# Part II — Protocol Section 4.2 step 2: the 28-configuration grid (2026-11-11)

> **Status:** implemented and verified; **awaiting mentor review**; not committed. Started on the owner's instruction after the mentor approved Part I with this scope clarification: E0 and experiment E2 are protocol-aligned; E1 and E3 (`log1p(Amount)`) stay documented exploratory deviations that must not influence champion selection or modify the protocol. **E1 and E3 are not part of the grid.** Phase 5 is not complete.

## II.1 What was run [D][V]

- **Grid:** exactly the frozen `pre_registered_configuration()["models"]` (protocol Sections 5.2, 5.3), enumerated as a Cartesian product in the frozen key order.
  - **12 LR:** `C` ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10} × `class_weight` ∈ {None, balanced}. StandardScaler fitted on `T_fit`; lbfgs; `max_iter` 1,000 with the pre-registered refit at 5,000 on a convergence warning; L2 (scikit-learn default); `random_state` 42.
  - **16 HGB:** `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} × `min_samples_leaf` ∈ {20, 200} × `class_weight` ∈ {None, balanced}. Fixed: `max_iter` 300, `early_stopping=False`, `l2_regularization` 1.0, `max_bins` 255, `random_state` 42. No scaler.
  - Features: the frozen 29 (`V1`–`V28` + `Amount`); never `Time`.
- **Execution (Section 4.2 step 2):** each configuration is fitted **once** on `T_fit` (159,491 rows; 356 frauds) and scores **`T_cal` (39,873 rows) and validation (42,722 rows; 56 frauds) once each**, with `decision_function` (track B) and `predict_proba` of the same fit (track C). Track A (the quoted Phase 4b value) is attached to LR-10 (C = 1, balanced) only.
- **Score storage:** the scores are returned as an in-memory cache by `compute_grid` for the later stages. Following Section 12.2 and the storage policy, per-row scores are not written to the result file; they are recomputed deterministically from code.
- **`T_cal`:** scored and cached for the calibration stage. Its labels are not read; only label-free score diagnostics are reported.
- **HGB determinism (Sections 5.3, 12.2):** every HGB configuration was fitted twice. **All 16 gave bit-identical `T_cal` and validation scores [V].**
- **Step 3 record:** the pre-registered Section 5.4 rule was applied to record one candidate per family for the later gates. That is the maximum validation AP on track B; ties within 1e-12 go to the simpler model. Nothing else was decided.
- New code: `src/models/phase5_grid.py`, `src/models/run_phase5b2_grid.py`, `tests/test_phase5b2_grid.py` (18 tests). Result file: `reports/phase5b2_grid_results.json` (810,757 bytes; within the 5 MB target). Runtime: 3 min 42 s on the verified laptop. Run: `python -m src.models.run_phase5b2_grid`.

## II.2 Full grid (validation, track B = `decision_function`; 42,722 rows, 56 frauds) [V]

Recall@100 is the share of the 56 frauds in the top 100. TP@K is for K = 25 / 50 / 100 / 250 / 500. Fit time is wall clock and not reproducible.

| ID | Parameters | AP | ROC-AUC | Recall@100 | TP@K | W1 AP | W2 AP | Fit (s) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| LR-01 | C 1e-4, None | 0.8475 | 0.9880 | 0.857 | 25/44/48/50/50 | 0.7954 | 0.8929 | 0.2 |
| LR-02 | C 1e-4, balanced | 0.7484 | 0.9865 | 0.839 | 24/33/47/50/52 | 0.6386 | 0.8283 | 0.2 |
| LR-03 | C 1e-3, None | 0.8490 | 0.9870 | 0.875 | 25/42/49/50/51 | 0.8045 | 0.8914 | 0.1 |
| LR-04 | C 1e-3, balanced | 0.7704 | 0.9828 | 0.857 | 25/35/48/51/52 | 0.6624 | 0.8464 | 0.2 |
| LR-05 | C 1e-2, None | 0.7667 | 0.9859 | 0.875 | 23/38/49/51/51 | 0.6310 | 0.8498 | 0.1 |
| LR-06 | C 1e-2, balanced | 0.8382 | 0.9811 | 0.875 | 25/42/49/51/52 | 0.7807 | 0.8822 | 0.3 |
| LR-07 | C 0.1, None | 0.7201 | 0.9832 | 0.875 | 19/36/49/51/51 | 0.5322 | 0.8286 | 0.2 |
| LR-08 | C 0.1, balanced | 0.8525 | 0.9809 | 0.875 | 25/44/49/51/52 | 0.8053 | 0.8907 | 0.3 |
| LR-09 | C 1, None | 0.7225 | 0.9805 | 0.875 | 19/37/49/51/51 | 0.5488 | 0.8272 | 0.2 |
| **LR-10** | **C 1, balanced (Phase 4 = E0)** | **0.8536** | 0.9810 | 0.875 | 25/44/49/51/52 | 0.8083 | 0.8925 | 0.4 |
| LR-11 | C 10, None | 0.7221 | 0.9803 | 0.875 | 19/37/49/51/51 | 0.5468 | 0.8275 | 0.2 |
| LR-12 | C 10, balanced | 0.8533 | 0.9810 | 0.875 | 25/44/49/51/52 | 0.8083 | 0.8918 | 0.4 |
| HGB-01 | 0.05, 15, 20, None | 0.8043 | 0.9837 | 0.839 | 25/40/47/48/50 | 0.7266 | 0.8645 | 2.1 |
| HGB-02 | 0.05, 15, 20, balanced | 0.8399 | 0.9886 | 0.875 | 24/42/49/51/52 | 0.7871 | 0.8836 | 8.7 |
| HGB-03 | 0.05, 15, 200, None | 0.7984 | 0.9823 | 0.821 | 24/42/46/48/52 | 0.7371 | 0.8538 | 2.4 |
| HGB-04 | 0.05, 15, 200, balanced | 0.8511 | 0.9899 | 0.875 | 24/43/49/50/52 | 0.7942 | 0.8971 | 9.3 |
| HGB-05 | 0.05, 31, 20, None | 0.8200 | 0.9862 | 0.857 | 25/41/48/48/50 | 0.7334 | 0.8874 | 4.1 |
| HGB-06 | 0.05, 31, 20, balanced | 0.8664 | 0.9790 | 0.875 | 25/46/49/51/52 | 0.8078 | 0.9136 | 10.6 |
| HGB-07 | 0.05, 31, 200, None | 0.8199 | 0.9853 | 0.839 | 25/43/47/49/50 | 0.7520 | 0.8728 | 2.9 |
| HGB-08 | 0.05, 31, 200, balanced | 0.8622 | 0.9805 | 0.857 | 25/44/48/51/51 | 0.8085 | 0.9063 | 10.8 |
| HGB-09 | 0.1, 15, 20, None | 0.7974 | 0.9855 | 0.839 | 24/41/47/50/51 | 0.7299 | 0.8524 | 2.3 |
| HGB-10 | 0.1, 15, 20, balanced | 0.8545 | 0.9870 | 0.875 | 25/44/49/51/52 | 0.8082 | 0.8938 | 9.9 |
| HGB-11 | 0.1, 15, 200, None | 0.7562 | 0.9780 | 0.821 | 23/40/46/46/49 | 0.7123 | 0.7932 | 2.7 |
| HGB-12 | 0.1, 15, 200, balanced | 0.8519 | 0.9907 | 0.875 | 25/44/49/50/51 | 0.8040 | 0.8924 | 9.2 |
| HGB-13 | 0.1, 31, 20, None | 0.7757 | 0.9817 | 0.821 | 23/40/46/49/50 | 0.6768 | 0.8585 | 2.9 |
| **HGB-14** | **0.1, 31, 20, balanced (= experiment E2)** | **0.8686** | 0.9795 | 0.857 | 25/44/48/51/52 | 0.8112 | 0.9164 | 9.5 |
| HGB-15 | 0.1, 31, 200, None | 0.7865 | 0.9800 | 0.821 | 24/40/46/49/49 | 0.7363 | 0.8209 | 2.9 |
| HGB-16 | 0.1, 31, 200, balanced | 0.8637 | 0.9778 | 0.875 | 25/44/49/51/51 | 0.7974 | 0.9151 | 9.3 |

HGB parameters are listed as `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `class_weight`.

**Fit diagnostics [V]:**
- Every LR converged within `max_iter` 1,000 (13–42 iterations). No convergence warning occurred, so the 5,000-iteration refit never triggered.
- Every HGB built 300 trees.
- Track C AP equals track B AP for all 28 configurations. No `predict_proba` equals 1.0; the largest logit is 31.6, below the 36.74 saturation point.
- No fixed-K boundary is tie-sensitive.
- **Continuity:** LR-10 reproduces E0 and HGB-14 reproduces experiment E2 exactly (AP difference 0).

## II.3 Step-3 candidates (pre-registered rule; development, optimistic by selection) [V]

| Family | Selected | Validation AP | Median AP of family | Spread (max − median) | Within 1e-12 |
|---|---|---:|---:|---:|---|
| LR | **LR-10** (C = 1, balanced) | 0.853574 | 0.804301 | 0.049273 | LR-10 only |
| HGB | **HGB-14** (0.1, 31, 20, balanced) | 0.868638 | 0.829923 | 0.038715 | HGB-14 only |

Validation ΔAP of the selected pair: HGB-14 − LR-10 = +0.015064. This is a selection-optimistic development figure. It is **not** a champion decision, which needs criteria E1 (conditional paired block bootstrap) and E2 (held-out split-stability over the 28 cached score vectors) (Section 4.3).

## II.4 Observations [R]

- **The selected pair is exactly the pair examined in Part I.** LR-10 is the Phase 4 configuration. HGB-14 is the HGB point chosen a priori for experiment E2, whose validation AP was therefore seen *before* the grid ran. The Section 5.4 rule is mechanical and fixed in advance, and it read only the grid's AP values, so no discretion entered. The order of events is nevertheless disclosed here: one of the 16 HGB AP values was known before the grid.
- **Class weighting dominates the LR grid.** For C ≥ 1e-2, `balanced` beats unweighted by 0.07–0.13 AP. At C ≤ 1e-3 the order reverses: LR-01 and LR-03 (unweighted) reach 0.847–0.849. Balanced LR is almost flat in C from 0.1 to 10 (0.8525–0.8536).
- **HGB:** `balanced` beats unweighted in all 8 pairs. Larger trees (31 leaves) score higher with `balanced`.
- **AP and ROC-AUC disagree.** The highest ROC-AUCs (HGB-12 0.9907, HGB-04 0.9899, LR-01 0.9880) belong to configurations with clearly lower AP, and the AP winners have some of the lowest ROC-AUCs. AP, the pre-registered primary metric, emphasises the top of the ranking.
- **Selection is near the noise level.** The best HGB beats the best LR by 0.015 AP, and HGB-06 is 0.002 below HGB-14. The family spreads (0.04–0.05) are of the same order. At the operational capacities, the selected pair differs by at most one fraud (K = 100: LR 49, HGB 48).
- **Window dependence:** for HGB-14 − LR-10, ΔAP is +0.003 in W1 and +0.024 in W2, as in Part I.
- **Cost:** balanced HGB takes 9–11 s per fit and unweighted HGB 2–4 s; LR takes at most 0.4 s.

## II.5 Safeguards and reproducibility [V]

- Test rows are removed before any label access; the maximum processed `Time` is 151,328. `check_row_roles` refuses any row outside its role (`T_fit`, `T_cal`, validation). `T_cal` rows are never passed to the fitting runner.
- The synthetic tests show that results are invariant to flipping test labels and setting test features to 1e6.
- No frozen value, feature, split or gate was changed (configuration SHA-256 `86ab5adb…`). No Part I or Phase 4 result file was modified.
- **Reproducibility:** a second full run produced a result file identical in every field except the wall-clock fields `runtime_seconds` and `hgb_determinism.*.second_fit_seconds`.
- **Tests:** `tests/test_phase5b2_grid.py` (18 tests) covers:
  - the grid equals the frozen product (12 + 16) and every fixed value is applied, with no deprecated `penalty=`;
  - the feature list;
  - row-role refusals; fit on `T_fit` only; `T_cal` and validation scores equal the pipeline outputs, with `predict_proba = expit(decision_function)`;
  - LR-10 and HGB-14 reproduce the Part I builders exactly;
  - the convergence-refit rule (forced with a patched `max_iter` of 1);
  - the Section 5.4 tie-break, hand-built for LR and HGB, plus refusals;
  - synthetic end-to-end: 28 configurations, track A only on LR-10, and test-isolation invariance;
  - real data: counts, continuity with Part I, 16/16 determinism, and that the selection can be recomputed from the recorded table.
  - Full suite: see WORK_LOG.

## II.6 Next stages (unchanged protocol; only on approval)

Step 4: sigmoid calibration of LR-10 and HGB-14 on their cached `T_cal` scores. Step 5: operational analysis. Step 6: the conditional paired block bootstrap (criterion E1) and the held-out split-stability over all 28 cached validation score vectors (criterion E2; estimable, with 1,984 usable splits). Step 7: the champion decision. The test partition stays untouched.
