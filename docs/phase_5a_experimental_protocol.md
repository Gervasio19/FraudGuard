# Phase 5A — Experimental Protocol (Preparation Only)

> **Status: APPROVED — Phase 5A officially closed (2026-11-09).** The project owner approved Revision 4 together with all remaining recommendations and decisions of the methodology review. The final decisions are recorded in Section 13.0 and Appendix B. This document is the frozen experimental protocol for Phase 5B. Phase 5B (implementation) has **not** started; it is a separate phase and requires its own explicit start. *History:* proposal for mentor review, prepared 2026-11-09. **Revision 2 (2026-11-09)** addresses methodology review findings M01–M08. **Revision 3 (2026-11-09)** applies targeted corrections R3-01–R3-05. **Revision 4 (2026-11-09)** applies targeted corrections R4-01–R4-06. See Appendix B for the revision log. No revision is approved; approval is recorded only by the human reviewer. **Nothing in this document is a result.** No model was trained, no experiment was run, no threshold was tuned, and no test-partition label was read while preparing it. All numeric values in this document are of three kinds: (a) repository facts, (b) descriptive counts computed on the train and validation partitions with pandas, and (c) explicitly hypothetical scenario parameters. Exact Poisson intervals for known fraud counts (Section 6.5) are arithmetic only. Phase 5B (implementation) starts only after explicit approval.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[V]** | Verified directly from code, data or artifacts. This includes descriptive statistics computed during this preparation (Appendix A): read-only, no model fitting, train and validation labels only |
| **[D]** | Documented in a committed repository document |
| **[R]** | Recommendation or interpretation by the protocol author; requires mentor approval |
| **[H]** | Hypothetical parameter (a scenario value, not a fact about any real operation) |

---

## 1. Executive summary

Phase 5 compares the incumbent Logistic Regression with exactly one HistGradientBoosting (HGB) challenger on the primary chronological split. It adds probability calibration, then evaluates both models for an Analyst Review Queue under fixed-K and time-based review capacity, transparent hypothetical cost scenarios, paired bootstrap comparison, and a Window 1 → Window 2 decision-transfer replay. The test partition is scored once, after every decision is frozen.

The six design recommendations, each justified in later sections:

| # | Question | Recommendation [R] | Main evidence |
|---|---|---|---|
| A | Review capacity | Fixed-K ∈ {25, 50, 100, 250, 500} per partition. Time-based: an online priority queue with capacity c ∈ {5, 10, 20, 50} reviews/hour **[H]** × SLA ∈ {1, 2, 4} h. The two are linked by K ≈ c × duration, so the gap between them measures the cost of the time constraint | Validation: 42,722 rows over 5.11 h, 56 frauds, 7–16 frauds per clock-aligned hour **[V]** |
| B | Cost model | Review-cost formulation: every alert costs one review unit, and a missed fraud costs R units, with R ∈ {5, 10, 20, 50, 100, 200, 500} **[H]**. Break-even R = 1 / Precision. An amount-aware variant is a sensitivity analysis. No currency is assumed | Currency of `Amount` is undocumented **[D]**. Validation fraud value is extremely concentrated: the top 5 frauds hold 56.1% of it **[V]** |
| C | Calibration | Temporal holdout = the last ~20% of train rows (`Time` 112,555–132,928; 39,873 rows; 28 frauds). Models are fitted on the earlier 80%. Sigmoid (Platt) calibration. If it fails technically, the safe alternative is **uncalibrated scores**. The cross-fitted temporal ensemble (C-F) is fully specified but is **not** an automatic fallback and never primary evidence. It is an optional retrospective sensitivity: two of its four calibration folds hold 4 frauds each, 71% of its out-of-fold positives come from night hours, and it is not fully prequential (its hyperparameters were selected with later validation labels) | Train-tail options hold only 20–28 frauds. The 25% option jumps to 96 frauds by entering night hours **[V]** |
| D | Hyperparameters | LR: C ∈ {10⁻⁴ … 10} × class_weight ∈ {None, balanced} = 12 configurations. HGB: learning_rate × max_leaf_nodes × min_samples_leaf × class_weight = 16 configurations, `early_stopping=False`, max_iter 300. Selection by validation AP | HGB's default `early_stopping='auto'` would draw a random 10% split inside training **[V]** |
| E | Paired bootstrap | Two distinct uncertainty layers. **(1) Conditional:** a paired circular moving-block bootstrap of the *fixed* selected pair on time-defined blocks. 15 min is a **provisional** central length; 5/10/30 min are always reported, and the decision uses the conservative envelope over 10/15/30 min. Non-circular moving-block and stationary bootstrap are sensitivities. **(2) Selection-aware:** repeated random half-splits of the validation blocks. Hyperparameters are re-selected on one half and evaluated on the buffered other half. Its "held-out split-stability frequency" is a descriptive stability statistic, not a probability or confidence level. Neither layer removes selection bias. The champion rule keeps the incumbent by default and promotes HGB only if E1 and E2 both pass | Fraud overdispersion at 5–10-min scales in validation (p ≈ 0.02–0.04) and at all scales in train. Train lag-1 autocorrelation falls from 0.17 (15-min bins) to 0.04 (30-min bins) **[V]**. Model hit/miss dependence is not yet observable (it requires scores; a 5B diagnostic is pre-registered). Only ~21 effective 15-min blocks in validation |
| F | Temporal transfer | A full "decision replay": every validation-stage decision is remade on Window 1 only, frozen, and applied to Window 2. Four quantities are kept apart: the W1-selected performance on W2 (the transferable estimate), its selection-aware uncertainty, the W2 retrospective oracle (a non-deployable reference), and the regret between them. The replay is a **diagnostic only**: its W2 outcome never decides the champion | Windows: 21,362 / 21,360 rows, 23 / 33 frauds, prevalence 0.108% / 0.154% **[D][V]** |

The main methodological risks are:
- the small fraud counts (56 validation, 52 test);
- the absence of night hours from validation and test;
- a prevalence that varies by hour of day, so a calibrator fitted on one period carries a systematic intercept error into another;
- selection optimism from tuning and choosing the champion on one 56-fraud validation set. Conditional bootstrap intervals do not capture this optimism (Sections 4.3 and 7.5);
- the few effective bootstrap blocks (Section 13).

---

## 2. Existing methodology and artifact inventory

### 2.1 Files inspected

- **Documentation:** `README.md`, `docs/PROJECT_CONTEXT.md`, `docs/validation_strategy.md`, `docs/business_understanding.md`, `docs/WORK_LOG.md` (Phase 5 and Phase 4 entries), `docs/reports/phase2_data_understanding.md` (Sections 3–4), `docs/reports/phase3_data_splitting.md` (Sections 4, 5, 7), `docs/reports/phase4_model_development.md` (all).
- **Code:** `src/config.py`, `src/data/split.py`, `src/data/temporal_windows.py`, `src/features/preprocessing.py`, `src/models/baselines.py`, `src/models/experiment.py`, `src/models/run_baselines.py`, `src/models/run_threshold_analysis.py`, `src/evaluation/metrics.py`, `src/evaluation/threshold.py`.
- **Artifacts:** `reports/phase4b_baseline_results.json` (read), `reports/phase4c_threshold_results.json` (size and structure via the code that writes it), and the folder listings of `artifacts/`, `data/processed/` and `reports/figures/`.
- **Environment:** scikit-learn 1.9.1 signatures of `HistGradientBoostingClassifier`, `CalibratedClassifierCV` and `sklearn.frozen.FrozenEstimator` **[V]**.

### 2.2 Inventory

| Item | Status | Label |
|---|---|---|
| Raw data `data/raw/creditcard.csv` | 284,807 rows, 492 frauds, SHA-256 `76274b69…51a89`. Git-ignored and must stay unchanged | [D][V] |
| Processed data `data/processed/` | Empty (`.gitkeep` only). No derived dataset exists | [V] |
| Model artifacts `artifacts/` | Empty (`.gitkeep` only). **No persisted model exists.** Every model is refitted deterministically from code | [V] |
| Results | `reports/phase4b_baseline_results.json` (5.5 KB) and `reports/phase4c_threshold_results.json` (9.5 MB, tracked by owner decision) | [V][D] |
| Figures `reports/figures/` | Empty | [V] |
| Tests | 147 `unittest` tests across 5 files (38 + 23 + 36 + 29 + 21) | [V] |
| Chronological split | `src/data/split.py`: train `Time ≤ 132,928`, validation `132,929–151,328`, test `≥ 151,329`. A pure function of `Time`; equal timestamps stay together | [V] |
| Phase 4c windows | `src/data/temporal_windows.py`: the boundary between unique timestamps that minimizes the row imbalance (earlier boundary on ties). On validation: W1 `Time` 132,929–142,280, W2 142,281–151,328 | [V][D] |
| Features | `V1`–`V28` + `Amount` (29 columns). `Time` is excluded from features but used for splitting (`src/config.py`) | [V] |
| Preprocessing | `StandardScaler` in a `ColumnTransformer` inside the `Pipeline`, fitted on training rows only. Other columns are dropped | [V] |
| Incumbent model | `LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)` | [V] |
| Test-access guard | `src/models/experiment.py::_prepare` raises if `evaluation_partition == "test"` | [V] |
| Metrics | `average_precision_score`, `roc_auc_score`, confusion matrix (`src/evaluation/metrics.py`). Threshold curve over unique scores with the inclusive `score >= t` rule; max-F1 with the lowest threshold on ties (`src/evaluation/threshold.py`) | [V] |
| Experiment configs | No config files. Settings are module constants (`LOGISTIC_REGRESSION_PARAMS`, `DEFAULT_RANDOM_STATE = 42`) and are echoed into the result JSON with environment versions and the data SHA-256 | [V] |
| Environment | `/Users/duc/Desktop/env/bin/python`: Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1 | [D][V] |

### 2.3 Prior results that constrain the design

| Fact | Source | Implication for Phase 5 [R] |
|---|---|---|
| Phase 4 LR validation AP 0.8397, 95% stratified bootstrap CI [0.746, 0.922] | [D] Phase 4 report §3.2 | Single-model AP uncertainty is about ±0.09. Small model differences cannot be resolved without a *paired* design |
| Balanced LR mean predicted probability 0.078 against a prevalence of 0.0013 | [D] Phase 4 report §3.4 | Probabilities are uncalibrated by about 60×. Cost thresholds need calibration |
| Max-F1 thresholds correspond to the top 17–47 alerts on validation | [D] WORK_LOG, Phase 4 report §4.3 | The interesting operating range is tens to low hundreds of alerts. This anchors the low end of the K grid |
| A transfer gap is ≤ 0 by construction; the observed gap is inside the selection-optimism band | [D] Phase 4 report §5.2 | Transfer analysis must judge magnitude against a reference band, not the sign |
| Validation and test contain no hour values 00–11; train night-band fraud rate is 0.474% | [D] Phase 3 report §4 | Prevalence depends on time of day, which constrains any calibration holdout |
| 1,335 model-input duplicate groups cross chronological partitions (all non-fraud); measured AP effect < 0.0002 | [D] Phase 3/4 reports | Recheck when the train partition is subdivided (Section 12) |
| Partition usage policy: train = fit; validation = selection, tuning, thresholds, calibration decisions; test = final evaluation only | [D] PROJECT_CONTEXT, validation_strategy | Followed throughout. The calibration holdout is placed *outside* test |
| Approved and awaiting specification: K values, time-based capacity definition, cost ratios, block length, search space | [D] PROJECT_CONTEXT | Exactly what Sections 5, 7, 8 and 9 propose |

---

## 3. Data partition and temporal validation diagram

### 3.1 Proposed partition (recommended option C-A; see Section 6)

```
Elapsed hours  0                                 31.27     36.92   39.52   42.04          48.0
               |---------------------------------|---------|-------|-------|--------------|
Time (s)       0                            112,553  112,555 132,928 142,280 151,328        172,792
               [========== T_fit ===============][ T_cal  ][  W1   ][  W2   ][    TEST      ]
                                                           [-- VALIDATION --]
Rows           159,491                            39,873    21,362  21,360   42,721
Frauds         356                                28        23      33       (not read in 5A)
Prevalence     0.223% (derived)                   0.070%    0.108%  0.154%   0.1217% [D]
Hour values    0–23 (day 1) + 0–7 (day 2)         7–12      12–15   15–18    18–23    [D][V]

T_fit : fit preprocessing and models (all hyperparameter configurations)
T_cal : fit the calibrator only (models frozen); never used for selection
VALID : hyperparameter selection, champion selection, threshold/cost/capacity decisions,
        paired bootstrap (development), W1 -> W2 decision replay
TEST  : one final scoring of frozen champion + challenger, after the freeze manifest is approved
```

T_fit counts and boundaries are **[V]** (Appendix A). The T_fit prevalence is derived as 356 / 159,491. Hour values are elapsed hour modulo 24, not clock hours **[D]**.

### 3.2 Why this layout [R]

- **Strict time order: fit → calibrate → select → test.** Each stage uses only earlier or same-stage information. The calibrator sees data later than the model's training data, the way a deployed calibrator would. It never sees validation or test.
- **Validation stays whole.** All 56 validation frauds remain available for selection and comparison. Taking the calibration holdout out of validation would cut selection data to 33 frauds or fewer and destroy the W1 → W2 replay.
- **Test is unchanged.** The 70/15/15 chronological partitions are not modified. T_cal is a sub-partition of train, consistent with `docs/validation_strategy.md` ("location and size … awaiting detailed specification").
- **A cost to accept:** models are fitted on 159,491 rows (356 frauds) instead of 199,364 (384). This removes 7.3% of the training frauds. Phase 5 results will therefore not reproduce Phase 4 numbers exactly. Section 4.4 quantifies the difference.

### 3.3 Gap effect

With T_cal between them, T_fit ends 5.66 hours before validation starts and 10.8 hours before test starts **[V]**. Phase 4 had no gap. A gap is realistic for a model retrained periodically, but it is a change in conditions and must be stated whenever Phase 4 and Phase 5 numbers appear together **[R]**.

---

## 4. Model comparison protocol

### 4.1 Candidates [R]

| Role | Family | Configurations | Notes |
|---|---|---|---|
| Reference (not tuned) | Majority class (`DummyClassifier(strategy="prior")`) | 1 | Floor for AP (= prevalence). Reported only |
| Incumbent | Logistic Regression + StandardScaler | 12 (Section 5) | The Phase 4 configuration (C=1, balanced) is one grid point, so the incumbent is nested |
| Challenger | HistGradientBoostingClassifier | 16 (Section 5) | **Exactly one challenger family.** One selected configuration enters the comparison |

Both families use the same 29 features, the same `T_fit` rows and the same fixed seed (42). No resampling, no feature engineering and no use of `Time` as a feature (Phase 4b decision) **[D]**.

### 4.2 Execution order (pre-registered) [R]

1. Build the sub-partitions `T_fit` / `T_cal` from the frozen boundary (Section 6). Check timestamp integrity and duplicate crossing.
2. Fit every configuration (28) on `T_fit` **once**. Score `T_cal` and validation **once** each. Store the scores. Every later analysis reuses them, as in Phase 4c; nothing is refitted per analysis.
3. **Hyperparameter selection** within each family: maximize validation AP on the *uncalibrated* scores, with a deterministic tie-break (Section 5.4).
4. **Calibration:** fit the sigmoid calibrator for each selected model on its `T_cal` scores (Section 6). Apply it to the validation scores. Calibration status is decided **separately for each model**: a model whose calibrator fails technically uses its own uncalibrated scores (Section 6.4).
5. **Operational analysis on validation:** fixed-K, time-based queue, cost scenarios, threshold selection per cost scenario.
6. **Uncertainty on validation:** the conditional paired block bootstrap of the fixed selected pair (Section 7.2), and the selection-aware repeated held-out block splits over all 28 cached score vectors (Section 7.5).
7. **Champion decision** under the pre-registered rule (Section 4.3). It uses only the full-validation evidence from step 6 (E1 and E2). It is recorded before step 8 runs.
8. **W1 → W2 decision replay** (Section 10): a separate temporal-transfer diagnostic. Its results are reported, but they never change the step 7 decision.
9. **Freeze manifest:** selected configurations, calibrator parameters (or the "uncalibrated" flag), thresholds per scenario, K grid and tie-break key, capacities, cost grid, block lengths, seeds, code commit and data SHA-256. Committed before step 10.
10. **Mentor approval plus acknowledgment of the test scope** (Phase 3 report §7).
11. **Final test evaluation:** score test once with the frozen pipelines (Section 11.3). Test results never feed back into steps 1–9.

### 4.3 Champion decision rule (to be fixed before step 2) [R]

The champion decision must be made **on validation, before test**. Test then reports both frozen models and their paired difference. The choice is *not revised* after seeing test results; otherwise test would become a selection set.

**Two kinds of uncertainty must not be confused (M01).**

1. **Conditional uncertainty.** This is the uncertainty of an *already-selected* pair (k̂_LR, k̂_HGB) when the validation sample is resampled. The paired block bootstrap of Section 7.2 measures it. It treats k̂ as fixed and answers: "if this exact pair were evaluated on other periods like this one, how much would ΔAP vary?"
2. **Selection uncertainty.** k̂ is itself a random outcome: it is the argmax of 12 (LR) or 16 (HGB) noisy validation AP values. This has two parts:
   - **(a) selection variability:** another validation sample could select another configuration;
   - **(b) selection optimism (winner's curse):** E[AP_validation(k̂)] > E[AP_new data(k̂)]. The maximum of noisy estimates is biased upward, and more so with more candidates. HGB's grid is larger and its fits more flexible, so the optimism is probably larger for HGB, and validation ΔAP is biased toward HGB.

**Why bootstrapping fixed selected predictions does not account for (2).** Every conditional replicate reuses k̂, which was chosen because it scored well on *these* rows. The replicate distribution is therefore centred near an optimistic point estimate, and its width ignores the fact that a different sample would have produced a different k̂. A conditional CI that excludes 0 can be produced by selection optimism alone. **No bootstrap on the same validation data removes selection bias.** The selection-aware procedure in Section 7.5 gives a heuristic, noisy *indication* of the bias. It neither estimates the bias without bias nor corrects it. Only the untouched test partition gives a selection-free estimate, and then only conditional on the frozen choice.

**Assessment of the three candidate rules**

| Rule | Assessment |
|---|---|
| (i) Conditional paired CI as the sole criterion | Not sufficient. Biased toward the larger grid. Kept as **descriptive development evidence** and as a *necessary* condition |
| (ii) Selection-aware resampling | Better aligned with the question "does the *procedure* that picks HGB beat the procedure that picks LR on unseen periods?". But the held-out halves hold only ~15–20 frauds, and they are not fully independent of the selection halves. The estimate is noisy and approximate |
| (iii) Incumbent by default | Matches the asymmetry: the LR is nested, simpler, interpretable and Phase-4-validated, and the risk of a spurious replacement is larger than the cost of a missed one in an educational, test-scarce setting |

**Recommended rule (combination of iii with i and ii as gates):** the LR is retained **unless both** of the following hold.

- **E1 (conditional, necessary):** the 95% conditional paired block-bootstrap CI for validation ΔAP = AP(HGB) − AP(LR) has a lower bound > 0 at **every** block length in the decision envelope {10, 15, 30} min (Section 7.4).
- **E2 (selection-aware stability heuristic):** in the repeated held-out block splits (Section 7.5), the **held-out split-stability frequency** is at least **90%**, and the median held-out ΔAP is > 0. The frequency is the share of usable splits in which the held-out ΔAP of the pair re-selected on the selection half is > 0.

**What the held-out split-stability frequency is, and is not (R3-01).**
- The 2,000 splits all reuse the same 21 validation blocks. They are **not** 2,000 independent validation samples.
- The frequency is a **descriptive resampling stability statistic**, conditional on three things:
  - the fixed validation period;
  - the 28 existing fitted configuration score vectors;
  - the chosen block partition and buffer rule.
- It is **not**:
  - a probability that HGB is better;
  - a confidence level;
  - a significance level or p-value;
  - an independent replication rate.
- The 90% value is a pre-registered **decision convention** (decision table #3 and #10b).
- E2 as a whole is a **decision heuristic, not a correction for selection bias**.
- **E2 requires at least 1,000 usable splits out of the 2,000 attempted** (Section 7.5, R4-01). If fewer are usable, E2 is reported as "**not estimable under the pre-registered rule**", the frequency is not calculated, and E2 cannot serve as a gate. Because promotion requires E1 **and** E2, an E2 that is not estimable means **HGB is not promoted and the LR is retained**. The report states that the incumbent was retained because E2 was not estimable, not because of evidence against HGB.

**Three distinct roles of the data (R3-02).**
1. **Full-validation champion selection: E1 + E2 only.** These use the whole validation partition, W1 and W2 rows alike, as the validation policy allows **[D]**. No other evidence enters the decision.
2. **W1 → W2 transfer diagnostic (Section 10):** a separate analysis. Decisions are made on W1 and frozen before W2 is evaluated, and W2 labels only *measure* them. Its outcome **never** promotes or blocks HGB in the primary rule.
   - *Scope note:* the replay is an independent check of the **W1-trained decision procedure** only. It is **not** an independent evaluation of the full-validation champion, because that champion's selection already used W2's rows as part of validation.
3. **Final test:** untouched until the freeze. It is the only evaluation of the frozen champion and challenger that is independent of every selection step.

*Revision 3 change:* revision 2 also had a temporal-replay gate (E3) that used W2 outcomes to decide promotion. That gate made W2 part of champion selection, and it is removed. The rule above has two gates, E1 and E2. Nothing else in the rule changes.

Interpretation duties:
- If the LR is retained, the report states "**no sufficient evidence for replacement at this sample size**", not "LR is better". With 56 frauds the rule has low power: HGB could be genuinely better and still fail E1 or E2. That is a limit of the data, not a finding.
- If HGB is adopted, the validation ΔAP is reported as an **optimistic development estimate**, together with the E2 split-stability frequency and the median held-out ΔAP. The test ΔAP is the confirmatory figure.
- Both frozen models are scored on test regardless of the decision, so the test report always contains the paired comparison.

Alternatives for the mentor (rejected as primary for the reasons above): **Option 2:** point estimate ΔAP > 0, plus Recall@K no worse at K = 50 and K = 100. **Option 3:** ΔAP ≥ 0.02 with the conditional lower bound > 0. Both rely on conditional evidence only.

### 4.4 Continuity with Phase 4 [R]

Report the Phase 4 configuration (C=1, balanced LR) fitted on `T_fit` beside the existing Phase 4b number (AP 0.8397, fitted on full train, `reports/phase4b_baseline_results.json` **[D]**). The difference shows the joint effect of removing `T_cal` and adding the gap. Phase 4 files are read, never rewritten.

### 4.5 Metrics roles [R]

- **Primary model-quality metric:** average precision on validation (uncalibrated scores). AP is the project's established ranking metric. It is threshold-free and focused on the rare class.
- **Secondary:** ROC-AUC (context only), Precision@K, Recall@K, calibration metrics (Section 6.5), and cost and queue metrics (Sections 8–9).
- **Calibration does not change AP** when the map is strictly increasing (sigmoid with positive slope). AP is therefore computed on uncalibrated scores to avoid ties. Isotonic regression creates ties and can change AP, which is another reason it is not the primary method.

---

## 5. Hyperparameter search space

### 5.1 Principles [R]

- **Small and pre-registered.** Validation has 56 frauds, and every extra configuration increases selection optimism. Twenty-eight configurations in total, chosen before any score is seen.
- **Full grid, no adaptive search.** Grid search is deterministic, reproducible and fully reportable: the whole grid table is published, not only the winner. Random or Bayesian search adds nothing at this size.
- **No inner cross-validation.** The policy assigns selection to validation **[D]**, and train-internal random CV would mix time order. Each configuration is fitted once on `T_fit`.

### 5.2 Logistic Regression (12 configurations)

| Parameter | Values | Rationale |
|---|---|---|
| `C` | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 | Log grid around the Phase 4 value (1). With ~160k rows and 29 standardized features, regularization matters only at small C, so the grid extends downward. Above C = 10 the fits are effectively unregularized |
| `class_weight` | None, "balanced" | Phase 4 used "balanced" without testing it **[D]**. Weighting changes the loss and can change the ranking, not only the calibration. Unweighted LR is the natural comparison |
| Fixed | penalty L2, `solver="lbfgs"`, `max_iter=1000` (raise to 5000 and record if a `ConvergenceWarning` appears), `random_state=42`, StandardScaler fitted on `T_fit` | Same pipeline as Phase 4. Convergence warnings are recorded per configuration, as the existing runner already does **[V]** |

### 5.3 HistGradientBoosting (16 configurations)

| Parameter | Values | Rationale |
|---|---|---|
| `learning_rate` | 0.05, 0.1 | The standard range. With fixed `max_iter` it sets the effective model capacity |
| `max_leaf_nodes` | 15, 31 | Tree complexity. 31 is the default; 15 is a lower-variance option for ~356 positive cases |
| `min_samples_leaf` | 20, 200 | The leaf-size floor. With few frauds, small leaves can isolate individual frauds (memorizing bursts). 200 forces smoother rules |
| `class_weight` | None, "balanced" | Same reason as LR. Supported in 1.9.1 **[V]** |
| Fixed: `max_iter` | 300 | Enough for convergence at learning_rate 0.05–0.1 without early stopping |
| Fixed: `early_stopping` | **False** | The 1.9.1 default is `'auto'`, which enables early stopping above 10,000 rows using a *random* `validation_fraction=0.1` split of the training data **[V]**. That would be a hidden, non-temporal selection step. Disabled explicitly |
| Fixed: `l2_regularization` | 1.0 | Mild shrinkage for stability. Not tuned, to keep the grid small |
| Fixed: `max_bins` | 255 (default) | — |
| Fixed: `random_state` | 42 | With early stopping off and `T_fit` (159,491 rows) below the 200,000-row binning subsample, the fit is expected to be deterministic. 5B verifies this by fitting twice and comparing scores |
| Preprocessing | No scaler | Tree splits are invariant to per-feature affine transforms. The scaler adds nothing and could only perturb bin edges by floating-point rounding. Same 29 columns; `Time` dropped |

Cost estimate **[R]**: 12 LR fits take seconds each, and 16 HGB fits take tens of seconds each on ~160k × 29. The total is minutes on a laptop; no parallel infrastructure is needed.

### 5.4 Selection rule [R]

- For each family, select the configuration with the highest validation AP. Ties within 1e-12 go to the simpler model: smaller `max_leaf_nodes`, then larger `min_samples_leaf`, then lower `learning_rate`; for LR, smaller `C`. Then `class_weight=None` before "balanced".
- Report the full grid with AP, ROC-AUC and Recall@100 per configuration, plus the spread (max − median AP) as a visible measure of how much the selection could exploit noise.
- *Alternative (mentor choice):* a "one-standard-error" rule that picks the simplest configuration within one bootstrap SE of the best. It reduces optimism at the cost of an extra bootstrap step.

### 5.5 Explicitly excluded (avoiding model proliferation) [R]

Other model families (random forest, XGBoost/LightGBM, neural networks), resampling (SMOTE, undersampling), feature engineering (time-of-day features, `log(Amount)`), stacking, and ensembles of LR and HGB. The only ensemble considered is the optional, non-automatic *calibration* investigation C-F (Section 6.7). It ensembles temporal fold models of the *same* selected configuration, not model families.

---

## 6. Calibration protocol

### 6.1 Why calibrate, and what calibration changes [R]

- The balanced LR's mean predicted probability is 0.078 against a prevalence of 0.0013 **[D]**. Scores are rank scores, not probabilities.
- Calibration matters for: (a) the **Bayes-rule cost threshold** (review if p ≥ 1/R, Section 8), which is only meaningful with calibrated p; (b) **expected alert and fraud counts** for planning; (c) the optional amount-aware decision policy (review iff p_i · L_i ≥ c_r, Section 8.3). Amount-aware *cost accounting* of rank-based policies does not need calibration.
- Calibration does **not** change: AP, ROC-AUC, Precision@K, Recall@K or the time-based queue. All are rank-based, and a strictly increasing sigmoid preserves ranks. The model comparison is therefore independent of the calibration decision, by design.

### 6.2 Holdout options (counts are [V], Appendix A)

| Option | Location | Rows | Frauds | Prevalence | Hour values | Assessment [R] |
|---|---|---:|---:|---:|---|---|
| **C-A (recommended)** | Last 20% of train, `Time` 112,555–132,928 | 39,873 | 28 | 0.070% | 7–12 | Temporally between fit and validation. Daytime, adjacent to validation's hours. Removes only 28 of 384 training frauds. Weakness: only 28 positives, and lower prevalence than validation (0.131%) |
| C-B | Last 10% / 15% of train | 19,936 / 29,904 | 20 / 24 | 0.100% / 0.080% | 10–12 / 9–12 | Fewer positives. Use as sensitivity checks on C-A |
| C-C | Last 25% of train, from `Time` 91,409 | 49,841 | 96 | 0.193% | 1–12 | More positives, but includes night hours (fraud rate 0.474% in train **[D]**), which validation and test lack. It removes 96 frauds (25%) from training. Composition mismatch: rejected as primary |
| C-D | Window 1 of validation | 21,362 | 23 | 0.108% | 12–15 | Closest to test in time. But it destroys the W1 → W2 replay, cuts selection data to 33 frauds, and holds even fewer positives. Rejected |
| C-E | All of validation (in-sample) | 42,722 | 56 | 0.131% | 12–18 | The calibrator and the thresholds would be fitted on the same data. Calibration quality could then only be checked on test. Rejected for the primary design |
| C-F (ensemble investigation; **not** an automatic fallback) | Cross-fitted expanding-window folds inside train, fully specified in Section 6.7 | 49,841 OOF rows | 96 OOF | mixed (0.04–0.68% by fold) | 1–12 | Out-of-fold with respect to model fitting, but **not fully prequential**: the fold models use hyperparameters selected with later validation labels. 2 of 4 calibration folds hold 4 frauds each, and 68 of its 96 out-of-fold positives (71%) come from night hours. Optional retrospective sensitivity only, on explicit mentor request |

Note on how the size was chosen: the row-fraction options were enumerated first, and their train-label counts were then inspected. Inspecting train labels is permitted (train is the fitting partition). No validation or test outcome informed the choice.

### 6.3 Method [R]

- **Sigmoid (Platt) calibration** on the selected model's `T_cal` scores. Two parameters suit 28 positives. Isotonic regression is non-parametric, overfits with this few positives and introduces score ties. Isotonic is reported as a *sensitivity only* (calibration metrics, not AP).
- Implementation: `CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method="sigmoid")` fitted on `T_cal`. `FrozenEstimator` is available in 1.9.1 **[V]**; it keeps the model from being refitted. An equivalent is a 1-D logistic regression on the logit (LR) or raw score (HGB).
- **Input scale:** for LR the decision-function logit is natural. HGB's `decision_function` is also a log-odds. The same procedure is used for both models.
- **Guard:** the fitted slope must be positive. Otherwise the calibrator would reverse the ranking. A non-positive slope is a technical failure (Section 6.4).

### 6.4 Technical-failure rule: uncalibrated scores as the safe alternative [R]

**Trigger:** a *technical* failure of a model's C-A calibrator, determined on `T_cal` alone, before any validation calibration metric is computed. Technical failure means any of:
- the fit raises an error, does not converge, or returns non-finite parameters;
- the fitted slope is ≤ 0;
- **numerically constant output** under the criterion of Section 6.5 (R4-02): the standard deviation of the calibrated logits on `T_cal` is below 1e-8.

The earlier "all calibrated p in a single reliability bin" criterion is withdrawn. A legitimate, varying but concentrated probability distribution can fall inside one decade bin.

**Per-model status (R4-05).** The LR and HGB calibrators are fitted and checked **independently**. Each model uses its calibrated probabilities if its own calibrator passes, and its own uncalibrated scores otherwise. One model's failure does not change the other model's status. Mixed outcomes (one calibrated, one not) are possible and are reported as such.

**What happens on failure (for the affected model only):** that model continues with **uncalibrated scores**. There is no automatic switch to another calibration method.
- **Unaffected, for both models:** every rank-based analysis (AP, ROC-AUC, fixed-K, time-based queue, the champion rule) and the empirical cost-minimizing thresholds selected on validation (these are rank cut-offs). These stay fully comparable between the models whatever their calibration status, because a strictly increasing calibrator does not change ranks.
- **Reported as "not available (no valid calibrator)" for the affected model only:** its Bayes thresholds (p\* = 1/R and the transaction-specific rule of Section 8.3) and its probability-based calibration metrics. Paired probability-based comparisons (for example the LR–HGB Brier difference) are then not available.
- **Why this is the safe alternative:** it changes nothing that the model comparison depends on, introduces no new fitted component, and cannot leak.

**Calibration-in-the-large is not a trigger.** The prevalence differs by hour of day (`T_cal` 0.070% vs validation 0.131% vs test 0.122%), and no train-internal holdout can remove that shift. It is a dataset property to report, not a defect to tune away on validation; doing so would turn validation into a calibration set.

**C-F is not an automatic fallback.** It is retained only as an optional, retrospective ensemble investigation, run on explicit mentor request and reported as a sensitivity. It is never primary evidence (Section 6.7).

*Note:* the approved Phase 5 design lists "ensemble investigation as the fallback, if needed" **[D]**. This revision keeps the investigation but removes its automatic role, because Section 6.7 shows it is not statistically defensible as a drop-in replacement. The change needs mentor confirmation (decision #1b).

### 6.5 Calibration evaluation and its uncertainty (validation; then test at the end) [R]

All calibration figures are reported **with uncertainty** and with the number of positives behind them. Positives available: `T_cal` 28 (in-sample for the calibrator, so its metrics are optimistic and labelled as such), validation 56, W1 23, W2 33.

| Quantity | Point estimate | Required uncertainty |
|---|---|---|
| **O/E** (observed frauds / Σ calibrated p) | O/E | (a) Exact Poisson (Garwood) 95% CI on O with E fixed. Worked widths for known counts **[V]** (arithmetic): O = 56 → [42.3, 72.7], i.e. ×0.76–1.30; O = 33 → ×0.69–1.40; O = 23 → ×0.63–1.50; O = 28 → ×0.66–1.45. (b) Conditional block-bootstrap CI (Section 7.2), which reflects burst clustering and is expected to be wider. Report both |
| **Calibration intercept** (logistic regression of y on offset logit(p̂), slope fixed at 1) | a | Profile-likelihood 95% CI, plus the block-bootstrap CI |
| **Calibration slope** (logistic regression of y on logit(p̂)) | b | Profile-likelihood 95% CI, plus the block-bootstrap CI. With 56 events the slope CI is expected to be wide. A slope CI containing 1 is "no evidence of miscalibrated spread", not proof of good calibration |
| **Brier score, Brier skill vs constant prevalence, log loss** | value | Block-bootstrap CI. A paired CI for the LR–HGB difference |
| **Reliability table** | per bin: rows, observed frauds, Σp, observed rate | Clopper–Pearson 95% CI per bin. Two kinds of edges, with different timing (R4-06): (i) **decade edges** of predicted probability (10⁻⁶, 10⁻⁵, …, 10⁻¹, 1) are data-independent and pre-registered in this protocol; (ii) **K-derived edges**, the score of the K-th ranked row for each K in the frozen grid under the Section 9.2 key, can only be computed *after* the model scores that partition. They are computed from scores alone, without labels, and frozen in the manifest **before** that partition's labels are joined to evaluate any calibration metric. Bins with fewer than 5 observed or 5 expected frauds are flagged "not interpretable" and never summarized into a single number |
| **Calibrator parameter uncertainty** | sigmoid (A, B) fitted on `T_cal` | Block bootstrap of `T_cal` (CBB, 15-min blocks, ≈ 23 blocks over 5.66 h, B = 2,000, seed recorded) refitting only the 2-parameter sigmoid. Failure handling is pre-registered below. Propagated to the rank position of each Bayes threshold p\* = 1/R on validation, to show how many alerts the threshold uncertainty moves. Descriptive only |
| **Isotonic sensitivity** | same metrics | Same uncertainty reporting. Pre-registered as a sensitivity only |

**Bootstrap validity rules (pre-registered implementation rules; R3-04, revised by R4-02).** Two different bootstraps are involved. They get **different** validity rules, because one *fits a calibrator* and the other *evaluates* calibration.

**(A) Calibrator-fit bootstrap: `T_cal` replicates that refit the Platt sigmoid** (the "calibrator parameter uncertainty" row above). With 28 positives over ≈ 23 blocks, some replicates will contain few or no frauds. Each replicate is classified *before* any validation quantity is computed from it.

| Replicate condition | Classification | Treatment |
|---|---|---|
| Zero frauds in the resampled `T_cal` | **Invalid: no positives** | Not fitted; excluded |
| Fit raises an error, reports non-convergence, hits its iteration limit, or returns non-finite parameters | **Invalid: fit failure** | Excluded |
| Fitted slope ≤ 0 (the calibrated probability does not increase with the score, which would reverse or flatten the ranking). This is a requirement *on a calibrator*, which must preserve the ranking | **Invalid: non-positive slope** | Excluded |
| **Numerically constant output:** the standard deviation of the replicate calibrator's **logits**, z_i = A·s_i + B computed directly from the scores, over the replicate's own `T_cal` rows, is < **1e-8**; or any z_i is non-finite | **Invalid: degenerate** | Excluded |
| 1–4 frauds, otherwise valid | **Valid, flagged "low-positive"** | Included. Their count is reported, plus a sensitivity summary with them removed |
| Otherwise | **Valid** (including outputs that are concentrated but not constant) | Included |

*Why this degeneracy criterion, and why it is pre-registered without inspecting results:*
- It is measured on the **logit** scale. Calibrated fraud probabilities are legitimately tiny and tightly concentrated, so on the probability scale (or in decade bins) a varying output can look constant; on the logit scale it does not.
- The cutoff 1e-8 is a **floating-point guard**, not a statistical judgment. Double precision resolves about 1e-16 relative to logits of order 1–10, so 1e-8 sits about eight orders above rounding noise. It is also about eight orders below any spread that could carry ranking or calibration information. It flags only outputs that are constant for all practical purposes.
- It deliberately does **not** flag "statistically flat but varying" calibrators (for example, a logit spread of 0.05). Any such near-flatness cutoff would be a substantive choice that cannot be justified without seeing results. It is left as **unresolved decision #1d**, and until the mentor sets one, no near-flatness exclusion is applied.
- The same criterion defines "numerically constant output" for the primary C-A calibrator (Section 6.4).

Rules for (A):
- **Exclusion and reporting:** invalid replicates are excluded from every interval. The counts of each invalid category are always reported. Exclusion conditions the distribution on fittable samples, which tends to understate uncertainty; this is stated with the result.
- **Minimum requirement:** at least **90% valid replicates** (≥ 1,800 of B = 2,000).
- **If the requirement is not met:** report "**calibrator parameter uncertainty not estimable under the pre-registered rule**", give the invalid counts by category, and **report no interval and no propagated threshold range**. There is no retry with a different B, seed, block length or validity rule, and no substitute method. The primary C-A calibrator and its own failure rule (Section 6.4) are unaffected: this rule concerns only the uncertainty bootstrap.

**(B) Calibration-evaluation bootstrap: conditional block-bootstrap replicates of validation (and, at the end, test)** (Section 7.2). These replicates evaluate a **frozen** calibrated model. No calibrator is refitted, and the calibrator-fit rules of (A) **do not apply**. Which metrics need a fit in each replicate:

| Metric | Needs a fit in each replicate? | Replicate excluded only when |
|---|---|---|
| O/E | No; direct ratio | Never on validity grounds. E = Σp > 0 always, and O = 0 gives O/E = 0, a valid value |
| Brier score, log loss | No; direct averages | Never on validity grounds (log loss is finite because calibrated p lies strictly in (0, 1); if any p rounds to exactly 0 or 1, log loss is computed from logits) |
| Brier skill vs constant prevalence | No | The replicate has zero frauds, so the reference Brier score is 0 and skill is undefined. Reported as "undefined" and counted |
| Reliability-bin observed rates and Clopper–Pearson CIs | No | Never; empty bins are reported as empty |
| **Calibration intercept** a (logistic regression of y on the offset logit(p̂), slope fixed at 1) | **Yes**: a one-parameter logistic fit | Zero frauds (the MLE does not exist), fit error, non-convergence, or non-finite estimate |
| **Calibration slope** b (logistic regression of y on logit(p̂)) | **Yes**: a two-parameter logistic fit | Zero frauds, fit error, non-convergence (including complete separation), or non-finite estimate |

- **A negative or zero estimated calibration slope b is *not* an invalid replicate.** It is an evaluation result: it may indicate a reversed score–outcome association in that resample. It is kept in the distribution, and the share of replicates with b ≤ 0 is reported. The positive-slope requirement belongs only to the *fitted Platt calibrator* in (A) and Section 6.4.
- **Per-metric minimum:** for each fitted metric (a and b) separately, at least 90% valid replicates. Otherwise that metric's bootstrap CI is reported as "not estimable under the pre-registered rule", with failure counts and no retry. The non-fitted metrics are unaffected.
- The profile-likelihood CIs for a and b on the full partition follow the same fit-failure rule. If the full-partition fit fails, the point estimate and CI are reported as "not estimable".
- (B) applies only to a model that is calibrated (Section 6.4). For an uncalibrated model the probability-based metrics are "not available", not "not estimable".

Nothing here presupposes that calibration will succeed. No calibration result exists.

**Small-positive-sample limitations (stated in every calibration table).**
- 28 positives fix the calibrator. Its intercept SE in log-odds is of the order of 1/√28 ≈ 0.19 even before the prevalence shift.
- With 56 validation positives, O/E cannot distinguish miscalibration smaller than about ±25–30% from noise.
- Window-level calibration (23/33) is weaker still. Any figure is descriptive.
- ECE is not used as a headline number. With 56 positives it is dominated by binning choices.

**No repeated modification on validation.** The method (sigmoid), the holdout (C-A), the input scale, the decade bin edges, the validity rules (A) and (B), and the metric list are frozen in the manifest *before* validation is scored. The K-derived bin edges are added to the manifest after scoring, but before any validation label is used for calibration evaluation (R4-06).
- Validation calibration results are **reported, never acted upon**. They are not used to switch method (e.g. to isotonic), resize the holdout (e.g. to C-B), add a prevalence correction, or trigger C-F.
- The pre-registered sensitivities (isotonic, C-B 10%/15%) stay sensitivities, even if they look better on validation.
- Any change after validation calibration metrics have been seen requires mentor approval. It is logged as a **validation-informed deviation**, and from then on only test calibration counts as an unbiased check.

### 6.6 How calibration enters threshold selection and final evaluation [R]

- **Rank-based policies** (fixed-K, time-based queue): unaffected.
- **Cost thresholds:** two thresholds per cost scenario R are reported: (i) the **Bayes threshold** p* = 1/R on calibrated probabilities, *derived, not tuned* (it uses no validation labels); (ii) the **empirical cost-minimizing threshold** chosen on validation. The gap between them on validation is itself a calibration diagnostic.
- Both thresholds are frozen in the manifest and applied unchanged to W2 (replay) and to test. The calibrator is fitted once on `T_cal` and never refitted on validation or test.

### 6.7 C-F specification: cross-fitted temporal calibration (optional investigation) [R]

C-F is specified fully so that the mentor can judge it. It is **not** part of the primary pipeline and **not** an automatic fallback (Section 6.4).

**Fold boundaries.** Expanding windows inside train, with boundaries between unique timestamps. The boundaries reuse the row-fraction cut points of Appendix A. Counts are **[V]** or derived from those cut points by subtraction.

| Fold | Model training period (`Time`, inclusive) | Training rows / frauds | Calibration period (`Time`) | Calibration rows / frauds | Cal. prevalence | Cal. hour values |
|---|---|---|---|---|---:|---|
| F1 | 0–91,407 | 149,523 / 288 | 91,409–112,553 | 9,968 / 68 | 0.682% | 1–7 (night) |
| F2 | 0–112,553 (= `T_fit`) | 159,491 / 356 | 112,555–119,714 | 9,969 / 4 | 0.040% | 7–9 |
| F3 | 0–119,714 | 169,460 / 360 | 119,715–124,074 | 9,968 / 4 | 0.040% | 9–10 |
| F4 | 0–124,074 | 179,428 / 364 | 124,075–132,928 | 19,936 / 20 | 0.100% | 10–12 |
| Total OOF | — | — | 91,409–132,928 | 49,841 / 96 | 0.193% | — |

**Procedure (out-of-fold with respect to model fitting; *not* fully prequential, see below).**
1. For each fold f, fit the **already-selected** configuration (hyperparameters from step 3, chosen on validation with `T_fit` models) on the training period of f. Preprocessing is fitted inside the pipeline on that period only.
2. Score the calibration period of f, which lies strictly after its training period. These are the **out-of-fold (OOF) predictions**. Each OOF row is scored by a model that never saw it or any later row.
3. Two variants are possible:
   - **(a) Pooled calibrator:** fit one sigmoid on all OOF scores (96 positives) and apply it to the scores of the primary `T_fit` model on validation and test.
   - **(b) Ensemble:** fit one sigmoid per fold on that fold's OOF scores. For validation and test, average the four calibrated fold-model probabilities.
4. Validation and test predictions come only from models and calibrators **fitted** on data with `Time ≤ 132,928`. No validation or test row enters any model or calibrator fit. However, the hyperparameters of every fold model were chosen with validation labels (step 1). Validation labels are otherwise used only to *evaluate* C-F. Test is scored only in the final evaluation, and only if the mentor has explicitly included C-F as a reported sensitivity in the frozen manifest.

**C-F is not fully prequential (R3-03).**
- The OOF rows are out-of-fold **with respect to direct model fitting**: each OOF row is scored by a fold model fitted only on earlier rows.
- The **complete** C-F procedure is not prequential. The configuration every fold model uses was selected with **validation labels** (`Time` 132,929–151,328). Those labels are in the future relative to every fold's training and calibration period (all `Time ≤ 132,928`). A forecaster standing at the end of fold F1 could not have known that configuration.
- The OOF scores therefore carry a dependence on future labels, through the hyperparameters. How large that effect is cannot be quantified within this protocol.
- **Nested (fold-internal) hyperparameter selection is deliberately not implemented.** It would add a further selection layer inside 4-fraud folds and is outside scope.
- **Consequence:** C-F is an **optional retrospective sensitivity only**. It is never an automatic fallback and never primary evidence for calibration quality, threshold choice or the champion decision.

**Why C-F is not statistically defensible as an automatic fallback (independent of the prequential issue).**
- **Too few positives where it matters:** F2 and F3 calibrate on 4 frauds each, so variant (b) has two nearly degenerate calibrators.
- **Wrong regime:** 68 of the 96 OOF positives (71%) come from F1, hour values 1–7. That is the night regime that validation and test do not contain. The pooled calibrator (a) would be dominated by a 0.68% prevalence period and would systematically over-predict on validation (0.131%) and test (0.122%).
- **Score-scale mismatch (a):** F1, F3 and F4 fold models are trained on different amounts of data than the `T_fit` model whose scores are calibrated. A pooled sigmoid learned on their score scales is not guaranteed to transfer to the `T_fit` model's scale. This is especially true for HGB and for class-weighted models.
- **Rank change (b):** averaging four calibrated fold models produces a *different scorer* from the selected model. AP, fixed-K and the champion comparison would then depend on the calibration choice, which breaks the separation of Section 6.1.
- **Daytime-only alternative:** restricting folds to daytime calibration periods leaves at most the C-A positives (28), so it gains nothing over C-A.

**Conclusion:** C-F keeps validation and test rows out of every model and calibrator fit, but it is **not fully prequential** (its hyperparameters come from validation labels) and it is not defensible as a drop-in replacement. On technical failure of C-A, the protocol uses uncalibrated scores (Section 6.4). C-F may be run as a reported **retrospective sensitivity** on explicit mentor request, with variant (a) recommended because it preserves the ranking. Its results are labelled "retrospective, not prequential" and are never used as primary evidence.

---

## 7. Bootstrap methodology

### 7.1 Temporal dependence: evidence [V] (Appendix A)

Fraud counts in consecutive time bins were compared with a permutation null that shuffles labels across rows and keeps the transaction-volume pattern (500 permutations; the smallest attainable p is 0.002).

| Partition | Bin | Dispersion index (var/mean) | Null 95th pct. | Permutation p | Lag-1 autocorrelation of bin counts |
|---|---|---:|---:|---:|---:|
| Validation | 5 min | 1.37 | 1.33 | 0.036 | −0.06 |
| Validation | 10 min | 1.60 | 1.45 | 0.024 | −0.21 |
| Validation | 15 min | 1.25 | 1.59 | 0.23 | −0.24 |
| Validation | 30 min | 1.55 | 2.18 | 0.32 | −0.35 |
| Train | 5 min | 3.63 | 1.43 | 0.002 | 0.24 |
| Train | 10 min | 4.56 | 1.77 | 0.002 | 0.22 |
| Train | 15 min | 5.28 | 2.13 | 0.002 | 0.17 |
| Train | 30 min | 6.19 | 3.15 | 0.002 | 0.04 |

Burst structure (frauds linked when consecutive gaps are ≤ 300 s):

- **Validation:** 23 clusters, 17 with more than one fraud, holding 50 of 56 frauds. Largest cluster 6 frauds, longest 9.1 min, 90th-percentile duration 4.6 min. Median fraud inter-arrival 165 s; 35% of gaps ≤ 60 s.
- **Train:** the largest cluster has 37 frauds, the longest lasts 30.2 min, and the 90th-percentile duration is 11.3 min.

**Interpretation [R]:** frauds arrive in short bursts. In validation the dependence is detectable at 5–10-minute scales and not beyond, though power is low with 56 frauds. In train it is strong, and the positive autocorrelation has decayed to near zero by 30-minute bins. An i.i.d. bootstrap treats the frauds of one burst as independent draws. If both models miss (or catch) a whole burst together, the i.i.d. scheme understates the variance of ΔAP.

*Caveat:* this is dependence in **labels**. What matters for a paired bootstrap of ΔAP is dependence in each row's contribution to the statistic: model hits and misses, and above all the **discordant** frauds that one model ranks highly and the other does not. That dependence cannot be observed in 5A without scoring models. It is a pre-registered 5B diagnostic (Section 7.4).

### 7.2 Conditional paired block bootstrap: scheme [R]

This layer measures **conditional uncertainty** of the *fixed* selected pair (Section 4.3). It does not measure selection uncertainty.

**Block construction with irregular timestamps.**
- Transactions are irregular in time; `Time` is in whole seconds, with many rows per second at peak.
- Blocks are therefore **time intervals**, not row counts: a block starting at s contains every row with `s ≤ Time < s + L`. Rows sharing a timestamp are always in the same block.
- Row counts per block vary with volume. Validation volume is fairly steady (7,780–8,667 rows per full elapsed hour **[V]**), but the first and last partial hours are thin (555 and 270 rows).
- Block starts are drawn from a 1-second grid over the period [t₀, t₁]. Each replicate concatenates m = ⌈T/L⌉ blocks and truncates the last one so the total resampled duration equals T. The resampled row count and fraud count therefore vary between replicates. That is intended: it reflects the variation of a real period of length T.

**Endpoint problem and choice of scheme.**

| Scheme | How it works | Endpoint behaviour | Other issues | Role |
|---|---|---|---|---|
| Non-circular moving-block (MBB) | Starts uniform on [t₀, t₁ − L] | **Under-weights the ends.** A row at time t is covered by starts in [t − L, t] ∩ [t₀, t₁ − L], so rows within L of either end are drawn less often. At L = 15 min that is ~10% of the 5.11-h validation period. The bootstrap distribution is then centred on an end-down-weighted statistic, not the observed one | No artificial adjacency | Sensitivity |
| **Circular block (CBB)** | Starts uniform on [t₀, t₁); a block crossing t₁ continues at t₀ | Equal inclusion weight for every row, so the replicates are centred on the observed statistic | **Artificial adjacency:** a wrapping block joins the end of the period (hour value ~18, W2 regime, prevalence 0.154%) to its start (hour value ~12.9, W1 regime, 0.108%). The share of wrapping starts is L/T: 4.9% at 15 min and 9.8% at 30 min over validation | **Primary (provisional)** |
| Stationary bootstrap (Politis–Romano) | Circular, with geometric block lengths of mean L | Equal weights, like CBB | Less sensitive to the exact L, but adds variance from random lengths. Also wraps | Sensitivity |

For every scheme the report shows the replicate mean minus the observed ΔAP (centring bias). A visible bias in MBB but not in CBB indicates an endpoint effect. A visible difference between CBB and the stationary bootstrap indicates sensitivity to block-length randomness.

**Replicates do not create information.** B = 5,000 (primary) or 2,000 (sensitivities) only reduces *Monte Carlo* error in the percentiles. The information in the interval is limited by the number of approximately independent blocks, about T/L:

| Period | 5 min | 10 min | 15 min | 30 min |
|---|---:|---:|---:|---:|
| Validation (5.11 h) | 62 | 31 | 21 | 11 |
| Test (5.96 h) | ~72 | ~36 | ~24 | ~12 |
| W1 (2.60 h) / W2 (2.51 h) | 32 / 31 | 16 / 16 | 11 / 11 | 6 / 6 |

Mean validation frauds per block: 0.90 / 1.81 / 2.67 / 5.09 at 5 / 10 / 15 / 30 min **[V]**.

With ~11–24 effective units, block-bootstrap variance estimates are known to be biased downward and percentile intervals to under-cover. The CIs are therefore reported as **approximate and possibly anti-conservative**. As a cross-check, the delete-one-block **block jackknife** standard error (21 blocks at 15 min) is reported next to the bootstrap standard error.

**Other elements (unchanged from revision 1).**
- **Pairing:** the same blocks for both models in each replicate.
- **Interval:** percentile 95% CI, plus the bootstrap proportion with ΔAP ≤ 0. No BCa: its acceleration assumes independent units.
- **Degenerate replicates:** replicates with zero frauds are counted, reported and dropped.
- **Continuity:** the class-stratified i.i.d. paired bootstrap of the Phase 4 audit (seed 0).
- **Seed:** `numpy.random.default_rng(0)`, recorded.
- **Statistics:** ΔAP (primary), each model's AP, and ΔRecall@K / ΔPrecision@K at K = 50 and 100.

### 7.3 Interpretation rules [R]

- The bootstrap holds the fitted models fixed. It measures **evaluation-sample uncertainty only**, not variation from refitting on different training periods (Phase 4 report §7 **[D]**).
- **On validation, the conditional CI is descriptive development evidence.** Both models were selected on the same data, so ΔAP carries selection optimism, probably larger for HGB. **The conditional CI does not include that optimism and does not correct it** (Section 4.3). In the champion rule it is a necessary condition only (E1).
- **On test, the paired CI is confirmatory** for the frozen pair, which was selected without test data. A CI containing 0 means "no evidence of a difference at this sample size", **not** equivalence.
- **Windows (≈ 2.5 h, 11 blocks of 15 min):** the block bootstrap is not considered feasible. Per the approved fallback, window-level results are point estimates with sensitivity analysis: leave-one-block-out (15 min) and leave-one-burst-out jackknife ranges, plus the selection-aware replay bootstrap (Q3, Section 10.3) with its stated independence assumption.

### 7.4 Block length: reassessment (M04) [R]

| Evidence | What it says | Strength |
|---|---|---|
| Label dependence in validation | Overdispersion detectable at 5–10 min (p = 0.036, 0.024), not at 15–30 min (p = 0.23, 0.32). Lag-1 autocorrelation negative at every width | Weak: 56 frauds give little power at longer scales |
| Label dependence in train | Overdispersion at every width (p = 0.002); lag-1 autocorrelation 0.24 / 0.22 / 0.17 / 0.04 at 5 / 10 / 15 / 30 min; P90 burst duration 11.3 min, longest 30.2 min | Strong, but train may differ from validation and test (night bursts) |
| Model TP/FN dependence | **Not observable in 5A.** It requires scores | None yet |
| Effective blocks | 21 at 15 min and 11 at 30 min in validation (table above) | Longer blocks sharply reduce the number of units |

**Conclusion:** the evidence supports blocks of *at least* ~10 min, the scale of the observed bursts. It does **not** establish 15 min as statistically optimal. **15 min is a provisional central choice for reporting**, not a validated primary setting. Data-driven automatic selection (e.g. the Politis–White estimator) is not used for the decision. With ~56 events it is unstable, and choosing L on the same validation data would add another selection step. It may be reported descriptively.

**Pre-registered 5B diagnostics (descriptive; they select nothing):** for each selected model at K = 50 and K = 100, bin the TP and FN frauds, and the **discordant** frauds (inside the top-K of exactly one model), by 5 / 10 / 15 / 30 min. Report the dispersion against a permutation null that shuffles hit/miss status among frauds while keeping fraud times, and the lag-1 autocorrelation.

**Decision use (removes the freedom to pick L after seeing results):**
- CIs at **5, 10, 15 and 30 min** are always reported, side by side, for every scheme (CBB, MBB, stationary).
- Criterion E1 (Section 4.3) requires the lower bound > 0 under CBB at **all of 10, 15 and 30 min** (the conservative envelope).
- 5 min is reported but excluded from the envelope, because it is shorter than train's P90 burst duration and would cut bursts.
- If the pre-registered TP/FN diagnostic shows dependence beyond 30 min, the report states that **every** block CI is likely too narrow. The decision rule is not re-tuned.

### 7.5 Selection-aware resampling: repeated held-out block splits (M01) [R]

**Purpose:** to indicate, heuristically, how stable the comparison is under re-selection, and how much of the validation result may be due to selecting k̂ on the same rows (Section 4.3, uncertainty type 2). It reuses the 28 cached validation score vectors, so nothing is refitted.

**Why not a classic out-of-bag bootstrap.** With 21 effective 15-min blocks, a bootstrap draw leaves about (1 − 1/20.4)²¹ ≈ 36% of the period uncovered. A dependence buffer of one block length on each side of every drawn block shrinks the out-of-bag share to about (1 − 3/20.4)²¹ ≈ 4%, i.e. ~2 frauds, which is unusable. A split-sample design over non-overlapping blocks keeps a usable held-out set.

**Procedure (B = 2,000 random splits, seed recorded):**
1. Partition validation into its 21 non-overlapping 15-min blocks, aligned to the validation start. The final block is partial.
2. In each split b, assign 10 blocks at random to a **selection half** and the other 11 to a **held-out half**. There are C(21, 10) = 352,716 possible splits, so 2,000 random splits are nearly all distinct.
3. **Buffer:** drop held-out rows within **5 min** of a selection-half block. 5 min exceeds validation's P90 burst duration (4.6 min **[V]**) and limits leakage of a burst across the cut. Expected held-out share after the buffer: about a third of validation, i.e. roughly 15–20 frauds.
4. Re-select the best LR and best HGB configuration by AP on the **selection half** (rule of Section 5.4).
5. Evaluate the re-selected pair on the **held-out half**: held-out AP of each model, and held-out ΔAP.
6. As a **fixed reference**, also evaluate the pre-specified, never-selected Phase 4 configuration (LR, C = 1, balanced) on both halves.
7. Skip splits whose held-out half holds fewer than 10 frauds, and report how many were skipped.
8. **Minimum usable splits (R4-01):** the Section 7.5 outputs are computed only if **at least 1,000 of the 2,000 attempted splits are usable**. The outputs covered are the held-out ΔAP distribution and median, the held-out split-stability frequency, the selection frequencies and the optimism indicator.
   - **If fewer than 1,000 are usable:** report E2 as "**not estimable under the pre-registered rule**", and give the attempted, skipped and usable split counts. Do **not** calculate the split-stability frequency or the median held-out ΔAP, and do not use either as a champion gate. Under Section 4.3, E2 then does not pass and the LR is retained.
   - **No retry** with different split sizes, buffers, seeds, B or fraud-count thresholds.
   - The 1,000 figure is a **Monte Carlo adequacy requirement only**. It makes the frequency and median stable as summaries of *this* resampling scheme. Usable splits are **not** independent validation samples: all of them reuse the same 21 blocks and 56 frauds. Requiring more of them does **not** increase the effective sample size or the evidential weight of E2.

**Reported quantities:**
- **Selection frequency** of each configuration across splits (the stability of k̂).
- **Distribution of the held-out ΔAP** across usable splits, with its median (input to criterion E2).
- **Held-out split-stability frequency** (input to criterion E2): the share of usable splits with held-out ΔAP > 0. It is computed only when the minimum of 1,000 usable splits (step 8) is met, and always reported together with the attempted / skipped / usable counts.
  - It is a **descriptive resampling stability statistic**, conditional on the fixed validation period, the 28 existing fitted configuration score vectors, and the chosen block partition (21 × 15 min) and 5-min buffer rule.
  - The splits reuse the same 21 blocks, so they are **not** independent validation samples.
  - The frequency is **not** a probability that HGB is better, **not** a confidence level, **not** a significance level, and **not** an independent replication rate. It is reported under this name only, never as a "probability" or "% confidence".
- **Reference-adjusted optimism indicator per family (heuristic, R3-05):**

  ```
  indicator = mean_b[ AP_selection(k*_b) − AP_held-out(k*_b) ] − mean_b[ AP_selection(ref) − AP_held-out(ref) ]
  ```

  where `ref` is the fixed, never-selected Phase 4 configuration. The first term mixes selection optimism with systematic differences between the two halves (composition, prevalence, burst placement). Subtracting the reference term is a **heuristic adjustment** for those systematic differences. It assumes they affect the selected configuration and the reference in the same additive way.

  **That assumption is not guaranteed.**
  - AP depends non-linearly on prevalence.
  - A flexible HGB and a linear reference can respond differently to the same burst or composition change.
  - The reference is an LR, so it is a closer match for the LR family than for HGB.

  The adjustment can therefore under- or over-correct. **It is not an unbiased estimate of selection optimism**, and it carries no coverage claim. It is reported with this label next to the unadjusted first term.

**Limitations (stated with the results):**
- The held-out halves are small (~15–20 frauds), so held-out AP is very noisy.
- Selection uses half of validation, so it is noisier than the real full-validation selection. The optimism indicator therefore tends to be *larger* than the optimism of the actual selection, but not provably so.
- Halves are not fully independent; the buffer only limits short-range dependence.
- The procedure re-selects within the frozen grid, but cannot represent choices made when the grid was designed.
- It **indicates** selection optimism heuristically; it does **not remove** it and does not estimate it without bias. The result is a development diagnostic. The selection-free estimate remains the single test evaluation.

---

## 8. Cost model and assumptions

### 8.1 What is known and unknown

- **Unknown [D]:** the currency of `Amount` (Phase 2 report §4), analyst cost per review, customer-friction cost, recovery rates, chargeback rules, follow-on fraud. `docs/business_understanding.md` states that a monetary cost function "cannot yet be specified credibly". **That statement still holds; nothing below claims monetary validity.**
- **Known [V]:** the `Amount` column. Validation frauds: 56, total 8,397.70 Amount units, median 6.25, mean 149.96, maximum 1,335.00. 7 are zero-amount and 20 are ≤ 1.00. The largest fraud holds 15.9% of the total value and the top five hold 56.1%. Train frauds: median 11.86, mean 118.65, top-five share 17.7%. Legitimate validation transactions have median 24.95.

### 8.2 Primary formulation: review-cost ratio [R][H]

For an Analyst Review Queue, every reviewed alert consumes analyst time, whether it is fraud or not:

```
Cost(policy) = c_r · (TP + FP)  +  L · FN          (units: c_r = 1 review unit)
R            = L / c_r                              (missed-fraud cost in review units)
Net saving vs no review = L·TP − c_r·(TP + FP)
```

- **Break-even ratio for an operating point:** reviewing beats not reviewing iff R > (TP + FP) / TP = **1 / Precision**.
- **Break-even ratio between two operating points A and B:** A is cheaper iff R · (TP_A − TP_B) > (Alerts_A − Alerts_B). The break-even is R* = ΔAlerts / ΔTP.
- **Consequence:** at **equal alert volume (fixed K)**, the model with more TP wins for *every* R > 0. Fixed-K comparisons are therefore cost-model-free. The cost model matters only when policies differ in volume, i.e. threshold policies. This is why capacity and cost analyses complement each other.
- **Bayes threshold** with calibrated p: review iff p · L ≥ c_r, i.e. **p\* = 1/R**.
- **Scenario grid [H]:** R ∈ {5, 10, 20, 50, 100, 200, 500}, log-spaced over two orders of magnitude, because nothing narrows the plausible range. Every reported cost carries the label "hypothetical".
- **Illustration from existing numbers [V] (arithmetic on Phase 4 documented values; not a Phase 5 result):**
  - The Phase 4 LR at the 0.5 rule had precision 0.0397, so break-even R = 25.2: reviewing those 1,309 alerts pays only if a missed fraud costs more than about 25 reviews.
  - At the retrospective full-validation max-F1 threshold (precision 0.9149), break-even R = 1.09.

*Alternative formulation (classical FP/FN costs, no cost for reviewing a true fraud):* break-even R = FP/TP = 1/Precision − 1, with Bayes threshold 1/(1+R). It is reported in an appendix table for readers used to that convention. The two differ materially only at high precision.

### 8.3 Sensitivity: amount-aware loss (corrected, M05) [R][H]

**Loss model.** For transaction i, the loss if it is fraudulent and *not* reviewed is

```
L_i = α + λ · Amount_i      α ≥ 0: fixed loss per missed fraud (handling, chargeback, follow-on), in Amount units
                            λ ∈ [0, 1]: share of the transaction amount that is lost if the fraud is missed
```

**Transaction-specific Bayes decision rule.** With review cost c_r per reviewed alert (fraud or not) and a calibrated fraud probability p_i, reviewing i has expected cost c_r and not reviewing has expected cost p_i · L_i. Review i **iff**

```
p_i · L_i ≥ c_r     ⇔     p_i ≥ t_i = c_r / (α + λ · Amount_i)      (when α + λ·Amount_i > 0; algebraic form only, see the tie policy below)
```

- The threshold **t_i differs per transaction**. Large amounts get low thresholds, small amounts high ones.
- **Three cases by L_i = α + λ·Amount_i (R4-03):**
  - **L_i < c_r:** t_i > 1. Since p_i ≤ 1, p_i·L_i < c_r always, so reviewing **cannot be beneficial**, even at p_i = 1, and the transaction is never reviewed. This includes L_i = 0, for example a zero-amount transaction when α = 0. With α = 0, every zero-amount transaction is never reviewed, including the 7 zero-amount validation frauds **[V]**.
  - **L_i = c_r:** t_i = 1. For p_i < 1 the transaction is not reviewed. At p_i = 1 the two actions have **equal expected cost**: an **indifference case**, not a gain.
  - **L_i > c_r:** t_i < 1, and the threshold is applied normally.
- **Pre-registered tie policy for exact equality p_i·L_i = c_r** (this covers the L_i = c_r, p_i = 1 case and any other exact equality): the transaction **is reviewed**.
  - This follows the rule's inclusive "≥" and the project's inclusive threshold convention (`score ≥ threshold`, Phase 4c **[D]**).
  - The decision is computed in one fixed form only: `p_i * L_i >= c_r` in float64, with p_i taken from the calibrated logit, exact comparison, and no tolerance. Algebraically equivalent forms such as `p_i >= c_r / L_i` are not used, because rounding can make them disagree at equality.
  - The number of exact-equality cases is reported. With continuous calibrated probabilities it is expected to be zero, so the policy exists for determinism, not for effect.
- Under a capacity limit, the same logic ranks by **expected avoidable loss p_i · L_i** and reviews the top ones while p_i · L_i ≥ c_r. Ties in p_i·L_i follow the Section 9.2 key.

**How this differs from the review-cost-ratio model (Section 8.2).**

| | Review-cost ratio (primary) | Amount-aware (sensitivity) |
|---|---|---|
| Loss of a missed fraud | Constant L for every fraud | L_i = α + λ·Amount_i, specific to each transaction |
| Bayes rule | One global threshold p\* = c_r / L = 1/R | A threshold t_i = c_r / L_i per transaction |
| Needs calibrated p? | Only for the Bayes threshold. Fixed-K and empirical thresholds are rank-based | **Yes, always.** p_i is multiplied by L_i, so its absolute level matters, not only its rank |
| Effect on ranking | None: it ranks by score | It **re-ranks** by p_i · L_i, so it is a different *policy* from score ranking |
| Break-even | R > 1/Precision | Σ_{TP} L_i ≥ c_r · (TP + FP) for the reviewed set |

**Assumptions (each stated with the results):**
1. p_i is calibrated **conditionally on Amount**, not only on average. Sigmoid calibration on `T_cal` guarantees neither; miscalibration that varies by amount band biases the rule. 5B reports calibration by amount tercile (descriptive, with the uncertainty rules of Section 6.5).
2. A review, when performed, prevents the whole loss L_i (100% efficacy), and reviews are instantaneous within the SLA.
3. c_r is the same for every review, whatever the amount.
4. Losses are additive and independent across transactions. Card-level follow-on fraud is not modelled; there is no card identifier **[D]**.
5. The calibrated p is used as if exact. Its uncertainty (Section 6.5, calibrator parameter bootstrap) is reported but not propagated into the decision.

**Two distinct uses (kept separate):**
- **(a) Amount-aware cost accounting** of the primary, score-ranked policies (fixed-K, queue, empirical thresholds): evaluate Σ L_i over missed frauds plus c_r × alerts. This needs no calibration and is always reported as a sensitivity.
- **(b) Amount-aware decision policy** (the transaction-specific Bayes rule, or ranking by p_i · L_i): this needs calibration and is a different policy. It is **optional** and outside the approved scope list; it is evaluated only if the mentor approves it (decision #9). If C-A fails technically (Section 6.4), (b) is "not available".

**Scenarios [H]:** c_r ∈ {1, 5, 10} Amount units, α ∈ {0, 20} Amount units, λ = 1. That gives 6 scenarios, all hypothetical, in "Amount units" (no currency).
- **Why α matters:** with α = 0, a zero-amount fraud costs nothing, yet 7 of 56 validation frauds have Amount 0 and 20 have Amount ≤ 1. Small frauds may be card testing that precedes larger losses. That is a *hypothesis*; no card identifier exists to test it **[D]**. α keeps those frauds from being valued at zero.
- **Fragility guard:** with the top five frauds holding 56% of validation fraud value, every amount-weighted figure (cost, fraud value captured@K) is also reported with the single largest fraud removed (leave-one-out).

### 8.4 Reporting [R]

Report cost per hour of traffic, because the windows, validation and test differ in duration (2.5 / 5.1 / 6.0 h), next to the TP/FP/FN counts behind it. For every scenario, show the no-review cost (= L × all frauds) and the oracle cost (perfect ranking at the same alert volume).

---

## 9. Analyst Review Queue capacity scenarios

### 9.1 Evidence [V]

- **Validation:** 5.11 h, 42,722 rows (8,359 rows/h), 56 frauds (10.96/h on average). W1: 8.85 frauds/h; W2: 13.13 frauds/h.
- **Clock-aligned 1-hour buckets from the validation start:** 9, 8, 14, 7, 16 frauds in the five full hours, plus 2 frauds in a final partial bucket of 0.11 h.
- **2-hour buckets:** 17, 21, then 18 in a final 1.11 h. **4-hour buckets:** 38, then 18 in a final 1.11 h.
- **Test row volume** (label-free counts only): 7,166 rows/h on average, falling to 4,856 in the last elapsed hour. A fixed hourly capacity covers a larger share of test traffic than of validation traffic. This is stated for interpretation only; no capacity value was chosen from it.

### 9.2 Fixed-K (batch ranking) [R]

- **Definition:** rank every transaction in the evaluation period by score and review **exactly K** transactions. The capacity is never exceeded.
- **Deterministic tie-break (M06):** the ranking key is (score descending, `Time` ascending, original file row position ascending).
  - The two secondary keys are label-free and fixed before scoring.
  - Earlier-first matches first-in-first-out handling of equal-priority alerts.
  - File position makes the order total, because rows can share a timestamp.
  - The same key is used in the time-based queue and the bucketed top-k.
- **Ties are reported separately** at every K, per model:
  - `n_tied_at_boundary`: the rows whose score equals the K-th score, split into how many were inside and outside the top K;
  - the frauds within that tied group;
  - the TP range over *all* orderings of the tied group, [TP_min, TP_max];
  - the expected TP under a uniformly random tie order: TP above the tied score + k_in × (frauds in tied group) / n_tied (hypergeometric mean).
  - If TP_min ≠ TP_max, the Precision@K and Recall@K row is flagged "tie-sensitive" and the range is shown beside the deterministic value.
- **Why ties are expected:**
  - Identical model inputs give identical scores. Validation has 824 rows with an exact model-input twin in train, all non-fraud **[D]**, and twins within validation tie with each other too.
  - HGB scores are sums of piecewise-constant trees, so they tie far more often than LR scores. In Phase 4, the LR had 41,807 unique scores for 42,722 validation rows **[D]**.
  - 5B reports the number of unique scores per model and partition.
- **AP and ties:** scikit-learn's `average_precision_score` treats tied scores as one threshold step and does not depend on any tie-break order. It is therefore unaffected by the key above. This is noted so that AP and P@K are not expected to agree on tied boundaries.
- **K grid:** {25, 50, 100, 250, 500} per partition. Validation and test have almost identical sizes (42,722 / 42,721), so the same K is comparable across them. In the windows, K is scaled by row share: {13, 25, 50, 125, 250}.
- **Rationale:** the low end (25–50) covers the Phase 4c max-F1 range (top 17–47 alerts **[D]**) and sits below and near the number of frauds. The middle (100–250) is 2–5× the fraud count, i.e. the regime where recall saturates. 500 (1.2% of traffic) approaches the high-volume regime but stays well below the 1,309 alerts of the 0.5 rule **[D]**.
- **Metrics at each K:** Precision@K, Recall@K, fraud value captured@K (Σ Amount of caught frauds / Σ Amount of all frauds, with the leave-largest-out version), workload (K, alerts/hour, FP per TP), and analyst-hours at hypothetical 5 / 10 / 15 min per review **[H]**. An **oracle ceiling** (max recall = min(K, frauds)/frauds) is shown next to each value.
- **Limitation:** fixed-K is retrospective. It ranks the whole period at once, which an online queue cannot do. It is the upper bound of what a time-constrained queue with the same total budget can achieve.

### 9.3 Time-based capacity (online priority queue) — primary time-based definition [R]

- **Definition:** transactions arrive in `Time` order. Analysts complete reviews at a constant rate of c per hour. At each review slot the analyst takes the **highest-scoring unreviewed transaction that arrived within the last s hours** (the SLA). Equal scores are ordered by the Section 9.2 tie-break key, and each slot reviews at most one transaction. A transaction not reviewed within s hours of arrival **expires** and counts as not caught.
- **Deterministic event ordering (R4-04).** All times are in `Time` seconds.
  - **Period:** t₀ is the earliest and t₁ the latest `Time` in the evaluated partition. These are label-free; for validation they are 132,929 and 151,328.
  - **Slot spacing and the first slot:** Δ = 3600 / c seconds (720, 360, 180 and 72 s for c = 5, 10, 20, 50, all whole seconds). Slots are τ_j = t₀ + j·Δ for j = 1, 2, …, while τ_j ≤ t₁. **The first slot is at t₀ + Δ, not at t₀**: an analyst needs one review interval before the first review is complete. The last slot is τ_last = max{τ_j ≤ t₁}.
  - **SLA window:** S = s·3600 seconds. A transaction with arrival time a is **eligible at slot τ** iff a ≤ τ **and** τ − a ≤ S. Both ends are inclusive.
    - Arrivals exactly at a slot's timestamp (a = τ) **are eligible** at that slot.
    - A transaction exactly at its SLA boundary (τ − a = S) **is still eligible**. It expires only when τ − a > S.
  - **Order of operations at each slot τ_j:**
    1. Admit all arrivals with a ≤ τ_j.
    2. Expire every unreviewed transaction with τ_j − a > S. Expiry is processed **before** selection, but the boundary case τ_j − a = S is not expired, by the inclusive rule above.
    3. Select the eligible unreviewed transaction ranked first by the Section 9.2 key (score descending, `Time` ascending, file position ascending), and mark it reviewed at time τ_j. Its detection delay is τ_j − a.
  - **Empty slot:** if no eligible unreviewed transaction exists at τ_j, the slot is **idle**. Idle capacity is **lost, not deferred**: it is not banked for later slots, so no later slot ever reviews more than one transaction. The reason is that capacity is a rate of analyst time; banking would let quiet periods pay for bursts beyond rate c. Idle slots are counted and reported. They are expected to be rare, because every transaction, not only flagged ones, is a candidate.
  - **End of period:** an unreviewed transaction is **expired** if it was not eligible at τ_last + Δ, i.e. if a + S < τ_last + Δ. It is **censored** if a + S ≥ τ_last + Δ, i.e. it would still have been eligible at the next slot, which falls outside the period. In the primary metric, censored transactions count as not caught (conservative). A "drain" sensitivity continues slots τ_last + Δ, τ_last + 2Δ, … with no new arrivals until no eligible transaction remains.
- **Capacities [H]:** c ∈ {5, 10, 20, 50} reviews/hour. The anchor is the average fraud arrival rate in validation of ~11/h:
  - c = 5 cannot review every fraud even with perfect ranking;
  - c = 10 roughly matches the average arrival rate but not the bursts (up to 16/h);
  - c = 20 and 50 give headroom.
  - For staffing intuition only: at a hypothetical 6 minutes per review, c = 10 is one analyst.
- **SLA:** s ∈ {1, 2, 4} h (approved **[D]**). Together with the capacities this gives 12 scenarios per model.
- **Metrics:** frauds reviewed within SLA (recall), precision of reviewed alerts, fraud value captured, mean and 90th-percentile detection delay (review time − arrival), expired count, censored count, idle-slot count, and peak backlog. An oracle ceiling is computed by running the same simulation with the labels as scores (a reference only, not a model).
- **Feasibility limits [V][R]:** validation lasts 5.11 h, so a 4-hour SLA is nearly degenerate there (one full 4-hour span plus 1.1 h). Inside a 2.5-hour window, a 4-hour SLA is never binding. SLA 4 h is therefore reported on validation and test only, and flagged as weakly identified.

### 9.4 Secondary cross-check: bucketed top-k [R]

Split the period into consecutive non-overlapping SLA-length buckets and review the top ⌊c·s⌋ transactions of each bucket, with capacity prorated for a partial last bucket. It is simple and deterministic. Because the bucket alignment is arbitrary, results are reported over start offsets {0, 15, 30, 45} min.

### 9.5 How the two capacity views complement each other [R]

- **Linked budgets:** K ≈ c × duration. On validation, c = 5, 10, 20, 50 corresponds to K ≈ 26, 51, 102, 256, close to the K grid.
- **Recall gap = cost of time:** Recall@K (fixed-K) minus recall of the time-based queue at the matching c is the **price of the time constraint**. It measures how much burstiness and the SLA deadline cost relative to retrospective ranking. A model whose advantage holds under fixed-K but not under the queue ranks well globally but not locally in time.
- **Calibration-free:** both views depend only on ranks. The model comparison under capacity therefore does not depend on the calibration choice (Section 6.1).

---

## 10. Temporal transfer protocol (Window 1 → Window 2)

### 10.1 Window facts [V][D]

| | Window 1 | Window 2 |
|---|---|---|
| `Time` | 132,929–142,280 | 142,281–151,328 |
| Elapsed hours | 36.92–39.52 (2.60 h) | 39.52–42.04 (2.51 h) |
| Rows | 21,362 | 21,360 |
| Frauds | 23 | 33 |
| Prevalence | 0.108% | 0.154% |
| Frauds/hour | 8.85 | 13.13 |

The boundary comes from the existing `split_two_time_windows` and is reused unchanged. No new window definition is introduced.

### 10.2 Design: full decision replay [R]

Phase 4c transferred one threshold. Phase 5 replays **every validation-stage decision** using W1 only, freezes it, and applies it unchanged to W2.

| Decision | Made on | Frozen before W2? |
|---|---|---|
| Features, preprocessing, `T_fit` model fits | `T_fit` | Yes (shared with the main protocol; nothing is refitted) |
| Calibrator | `T_cal` | Yes (shared) |
| Hyperparameters per family | **W1** AP only | Yes |
| Replay champion (W1-only, diagnostic; *not* the primary champion): a reduced version of the Section 4.3 rule. The LR is retained unless the W1 conditional ΔAP CI has a lower bound > 0, using the stratified i.i.d. bootstrap because a block bootstrap is infeasible in 2.6 h. E2 is not run (11 blocks are too few to split) | **W1** only | Yes |
| Empirical cost-minimizing threshold per R | **W1** only | Yes |
| K-equivalent threshold (the score of the K-th ranked row in W1 under the Section 9.2 key, scaled K) | **W1** only | Yes. In W2 it becomes a *threshold* policy (`score ≥ t`), which tests whether alert volume holds. Its realized alert count may differ from K by design and is always reported, including any rows tied at t in W1 |
| Bayes thresholds p\* = 1/R | none (derived) | Yes |
| Capacities c, SLA s, cost grid | policy | Yes |

The W2 labels are used **only** to measure. A test (as in Phase 4c **[D]**) confirms that flipping the W2 labels leaves every frozen W1 decision unchanged.

**Separation from champion selection (R3-02).** The replay is a temporal-transfer **diagnostic**. The primary champion is decided in step 7 (Section 4.2) by E1 + E2 on full validation, before the replay runs. Replay results, including the W1 replay champion and any W2 ΔAP, **never** promote, block or reverse it. The replay independently checks the *W1-trained decision procedure*. It is not an independent evaluation of the full-validation champion, whose selection used all validation rows. Only test provides that.

### 10.3 Four quantities that must not be confused (M07) [R]

| # | Quantity | Definition | Status | How it is reported |
|---|---|---|---|---|
| Q1 | **W1-selected performance on W2** | Every decision made on W1 (Section 10.2), frozen, and measured on W2. Example: AP_W2(k̂_W1) | **The transferable estimate.** W2 labels are used only to measure | Headline transfer figure |
| Q2 | **W2 retrospective oracle** | The best configuration or threshold chosen *with W2 labels*. Example: max_k AP_W2(k) | **Not deployable and not a model's performance.** Nobody could have chosen it before seeing W2 outcomes. A reference ceiling only | Never in a headline table. Always labelled "oracle (retrospective, not deployable)" |
| Q3 | **Selection-aware uncertainty of Q1** | The distribution of Q1 when the *whole procedure* is repeated: resample W1 and W2 independently, re-make every W1 decision on the W1 resample, and measure on the W2 resample | Uncertainty of the procedure, including selection variability | Percentile range next to Q1. Stratified i.i.d. (independence assumption stated; block resampling infeasible in 2.5 h). A 10-min block version with 16 units is a sensitivity |
| Q4 | **Actual transferable performance** | What deploying the W1-trained decision procedure delivered in W2: **Q1, with Q3 as its uncertainty** | The answer to "does the procedure transfer?" | Stated as Q1 [Q3 range]. The in-sample W1 value of the W1-selected configuration is *not* a transfer estimate: it is optimistic by selection |

**Regret** = Q2 − Q1 (≥ 0 by construction) is a diagnostic of how sensitive the selection is. It is **not** a degradation measure. A small regret means "the W1 choice was nearly as good on W2 as hindsight could have been", not "performance held up".

**What the replay cannot separate:** comparing the W1 in-sample value with Q1 mixes three effects: selection optimism in W1, the prevalence change (0.108% → 0.154%), and genuine drift. One transfer of ~2.5 h cannot identify them separately. The report says so instead of attributing the difference to drift.

### 10.4 What is compared [R]

| Question | Measure | Note |
|---|---|---|
| Does the W1-chosen configuration rank W2 well? | Q1 = AP_W2(W1-selected config) with its Q3 range. **Hyperparameter regret** = AP_W2(oracle config) − Q1 | Regret ≥ 0 by construction. Compared against the spread of W2 AP across the grid |
| Does the champion choice hold? | Agreement of the W1 replay champion and the full-validation champion; ΔAP on W2 of the W1-selected pair | Descriptive only. **Never feeds back** into the champion decision |
| Do thresholds keep their volume? | Realized alerts/hour in W2 vs W1 at the frozen K-threshold; precision and recall | The alert-volume drift is prevalence-independent at the row level and operationally the most important |
| Cost transfer | Q1: cost_W2(frozen t) per hour with its Q3 range. **Cost regret per hour** = cost_W2(frozen t) − cost_W2(W2-oracle t), for each R | Regret ≥ 0 by construction. The W2-oracle threshold is a reference only |
| Calibration transfer | O/E in W1 vs W2; Brier | O/E is expected to move with prevalence (0.108% → 0.154%) |
| Queue transfer | Time-based queue recall, delay and expiry in W2 at each c and s ≤ 2 h | 4 h excluded (never binding in a 2.5-hour window) |
| Ranking stability | ROC-AUC in W1 vs W2 (prevalence-insensitive); AP with prevalence shown alongside | Raw AP and F1 differences mix in the prevalence change (Phase 4 §5.3 **[D]**) |

### 10.5 How degradation is judged [R]

- **The sign carries no information.** All regrets are ≥ 0 by construction (the Phase 4 §5.2 lesson **[D]**).
- **Reference band:** the Q3 procedure bootstrap (re-select on resampled W1, apply to resampled W2), as in the Phase 4 audit §5.1. In each replicate the replicate's own regret is recorded as well. An observed regret is called **material** only if it lies outside the 95% band of those replicate regrets. Otherwise it is "not distinguishable from selection optimism and sampling noise".
- **Presentation rule:** every transfer table lists Q1 first, then the Q3 range, then Q2 and the regret in a column headed "retrospective oracle, not deployable". A W2 optimum is never presented as an achievable model, threshold or cost.
- **Granularity:** one fraud is 4.3 percentage points of recall in W1 and 3.0 in W2. Every rate is reported with its counts.
- **Scope:** both windows lie in one afternoon (hour values 12–18). A good transfer shows stability over ~2.5 hours, not long-term robustness (the existing wording in WORK_LOG **[D]**).
- **The main temporal test is still train → validation → test.** The replay is a rehearsal of the freeze-then-apply procedure. It is a diagnostic, not a selection step.

---

## 11. Metrics and reporting requirements

### 11.1 Metric catalogue [R]

| Group | Metrics | Partitions |
|---|---|---|
| Ranking (primary) | AP; ROC-AUC (secondary) | validation, W1, W2, test |
| Fixed-K | P@K, R@K, fraud value captured@K (+ leave-largest-out), alerts/hour, FP per TP, analyst-hours **[H]**; boundary-tie report (n tied, inside/outside, [TP_min, TP_max], expected TP) | validation, windows (scaled K), test |
| Time-based queue | recall within SLA, precision, value captured, mean/P90 delay, expired, censored, idle slots, peak backlog (event ordering of Section 9.3) | validation, W2, test |
| Cost | cost/hour per R, break-even R (1/Precision and ΔAlerts/ΔTP), no-review and oracle references | validation, W2, test |
| Calibration | O/E with exact Poisson and block-bootstrap CIs; intercept and slope with profile-likelihood and block-bootstrap CIs; Brier, Brier skill and log loss with CIs; reliability table with Clopper–Pearson CIs on pre-fixed bins; calibrator-parameter bootstrap; isotonic sensitivity; calibration by amount tercile (if the amount-aware policy is approved) | `T_cal` (in-sample, labelled as such), validation, W1, W2, test |
| Uncertainty (conditional) | paired block-bootstrap CI for CBB at 5/10/15/30 min, with MBB and stationary sensitivities and the centring bias; block-jackknife SE; i.i.d. continuity CI | validation, test |
| Uncertainty (selection-aware) | attempted / skipped / usable split counts (always); and, only if ≥ 1,000 splits are usable (otherwise "not estimable under the pre-registered rule"): selection frequencies, held-out ΔAP distribution and median, held-out split-stability frequency (a descriptive stability statistic, not a probability or confidence level); reference-adjusted optimism indicator (heuristic, shown with the unadjusted term) (Section 7.5); replay Q3 ranges (Section 10.3, diagnostic only) | validation; W1 → W2 |
| Bootstrap validity | (A) calibrator-fit bootstrap: invalid counts by category (no positives, fit failure, non-positive slope, numerically constant output) plus low-positive count and valid share; (B) calibration-evaluation bootstrap: per fitted metric (intercept, slope), failure counts and valid share, share of replicates with slope ≤ 0 (kept as results), and undefined Brier-skill count. A "not estimable" statement wherever a 90% requirement fails (Section 6.5) | `T_cal`, validation, test |
| Window uncertainty | point estimates with leave-one-block-out and leave-one-burst-out ranges | W1, W2 |

### 11.2 Reporting rules [R]

1. Every number carries its partition, the number of frauds behind it, and a tag: *selection* (chosen on this data), *development* (validation, after selection) or *final* (test).
2. Every hypothetical parameter is tagged **[H]**. No cost is printed with a currency symbol.
3. Uncalibrated vs calibrated scores are always named. AP is always computed on uncalibrated scores.
4. Full grids are published (all 28 configurations), not only the winners.
5. Random-split (secondary benchmark) results are not used in Phase 5 unless the mentor adds them (Section 13). They would never be pooled with chronological results **[D]**.
6. Outputs go to *new* files (for example `reports/phase5b_development_results.json` and `reports/phase5_final_test_results.json`) in the strict-JSON conventions of Phase 4 (NaN → null; environment, seeds and data SHA-256 recorded). Phase 4 files are never rewritten.
7. The champion decision is reported with E1 and E2 only. W1 → W2 replay results appear in a separate section headed "temporal-transfer diagnostic (does not affect the champion)".
8. Figures planned (in 5B, not now): PR curves; P@K / R@K vs K with the oracle ceiling; queue recall vs capacity per SLA; cost/hour vs R; reliability plots; the bootstrap ΔAP distribution.

### 11.3 Final test evaluation content [R]

The frozen champion and challenger, scored once.

**Calibration status is per model (R4-05):** each model is used with its calibrated probabilities if *its own* calibrator passed Section 6.4, and with its own uncalibrated scores otherwise. The status is fixed in the manifest before test and is not revisited.
- Probability-based metrics and thresholds (Bayes thresholds, the amount-aware policy if approved, O/E, intercept and slope, Brier, log loss, reliability) are "not available" **only for the affected model**, together with any paired probability-based comparison.
- Rank-based metrics (AP, the paired ΔAP, fixed-K, the time-based queue, and empirical rank cut-offs) are unaffected and remain fully comparable between the two models.

Reported: AP and the conditional paired block-bootstrap ΔAP (CBB at 10/15/30 min, ~36/24/12 blocks, with 5 min and the MBB/stationary sensitivities). On test this conditional CI is the confirmatory comparison, because no selection used test data, fixed-K at the frozen K grid, time-based queue at frozen c/s, cost at the frozen thresholds and the Bayes thresholds, and calibration. Opened with the mandatory test-scope statement (hour values 18–23 of day 2, 52 frauds, prevalence 0.1217% **[D]**). No retrospective test optimum is computed for selection. If one is shown at all, it is labelled "retrospective, not used".

---

## 12. Leakage and reproducibility safeguards

### 12.1 Leakage controls [R]

| Risk | Control |
|---|---|
| Test use before freeze | Keep the existing guard (`_prepare` refuses `test`). The final scoring uses a *separate* entry point that requires a committed freeze manifest and refuses to run without it. That code change needs approval (Section 13) |
| Calibrator sees validation or test | The calibrator is fitted on `T_cal` scores only. A unit test checks the calibrator's training indices ⊆ `T_cal` |
| Preprocessing leakage | The scaler stays inside the pipeline, fitted on `T_fit` (not all of train) |
| Hidden random split inside HGB | `early_stopping=False` set explicitly **[V]** |
| Timestamp splitting | The `T_fit`/`T_cal` boundary sits between unique timestamps (112,553 \| 112,555) **[V]**. A test asserts that no timestamp spans both |
| Model-input duplicates across `T_fit`/`T_cal`/validation | Report the cross-partition twin counts over `V1`–`V28` + `Amount` for the new sub-partitions (descriptive, as in Phase 3/4). Earlier evidence shows all twins are non-fraud with negligible AP effect **[D]** |
| Window labels leaking into replay decisions | Label-flip test on W2 (as in Phase 4c) |
| Selection optimism read as performance | Development vs final tags; full grids; the champion rule is fixed before any score exists; conditional and selection-aware uncertainty reported separately (Sections 4.3, 7.5); W2 oracle labelled non-deployable (Section 10.3) |
| Validation-driven calibration changes | Calibration method, holdout, decade bin edges, validity rules and metrics frozen before validation scoring. K-derived bin edges are computed from scores without labels and frozen before validation labels are evaluated. Any later change is logged as a validation-informed deviation (Section 6.5) |
| Choosing block length after seeing CIs | All block lengths are always reported; the decision envelope {10, 15, 30} min is fixed in advance (Section 7.4) |
| C-F fold leakage (if run) | Partial control only. Each fold model is *fitted* only on rows earlier than its calibration period, and every fold fit uses `Time ≤ 132,928` only. A unit test asserts max(train Time) < min(calibration Time) per fold. **Not controlled:** the fold models' hyperparameters were selected with validation labels, so C-F is not fully prequential. This is accepted only because C-F is a labelled retrospective sensitivity, never an automatic fallback or primary evidence (Section 6.7) |
| W2 labels deciding the champion | The champion is decided by E1 + E2 at step 7, before the replay. Replay outputs are written to a separate results key, and the champion decision code does not read them. A unit test asserts that changing replay outputs (or flipping W2 labels within the replay) leaves the recorded champion unchanged |
| Tie-break order depending on labels | The tie-break key uses score, `Time` and file position only. A unit test asserts that permuting labels leaves the reviewed set unchanged |
| Feedback from test | Test results never change a decision. If a defect is found after test, it is fixed and reported as a *post-test correction* with both results shown |
| Inherited PCA look-ahead | Unfixable at the dataset level **[D]**. Restated in the limitations |

### 12.2 Reproducibility [R]

- Seeds: model `random_state=42`; bootstrap `default_rng(0)`; both recorded.
- Pinned environment versions written into every JSON. Data SHA-256 checked at start, and the run aborts on a mismatch.
- HGB determinism verified (two fits, identical scores) before the grid runs.
- Scores computed once and cached in memory. Every analysis reads the same arrays, as in Phase 4c **[V]**.
- Unit tests for every new metric:
  - P@K exact-K tie handling and the [TP_min, TP_max] bounds;
  - the queue simulator on hand-computed toy cases, including these **boundary cases for the Section 9.3 event ordering (R4-04)**. Each uses t₀ = 0, Δ = 10 s and S = 20 s; arrivals are given as (id, arrival time, score), in file order:
    - **Q1, first slot and arrival at a slot time:** (x1, 0, 0.2), (x2, 10, 0.9); t₁ = 20. Expected: there is no slot at 0. At τ₁ = 10, x2 (arriving exactly at the slot) is eligible and is reviewed. At τ₂ = 20, x1 has age 20 = S, is still eligible, and is reviewed with delay 20. Nothing expires.
    - **Q2, expiry before selection, and an idle slot:** (x1, 0, 0.2), (x2, 5, 0.9), (x3, 6, 0.8); t₁ = 30. Expected: τ₁ = 10 reviews x2. τ₂ = 20 reviews x3 (x1, at age 20, is eligible but lower-scored). At τ₃ = 30, x1 (age 30 > S) expires first, the queue is empty, and the slot is idle. Result: 2 reviewed, 1 expired, 1 idle slot.
    - **Q3, idle capacity is lost, not deferred:** (x1, 0, 0.5), (x2, 21, 0.9), (x3, 22, 0.8); t₁ = 40. Expected: τ₁ = 10 reviews x1. τ₂ = 20 is idle. τ₃ = 30 reviews only x2. τ₄ = 40 reviews x3 (age 18). The test asserts that x3 is **not** reviewed at 30.
    - **Q4, tie-break inside the queue:** (x2, 1, 0.7), (x3, 1, 0.7), (x1, 3, 0.7), with x2 before x3 in file order; t₁ = 30. Expected: τ₁ = 10 reviews x2 (earliest `Time`, then file position). τ₂ = 20 reviews x3. At τ₃ = 30, x1 (age 27 > S) expires and the slot is idle. Result: 2 reviewed, 1 expired, 1 idle slot.
    - **Q5, expired vs censored at period end:** (x1, 0, 0.9), (x2, 0, 0.8), (x3, 0, 0.1), (x4, 15, 0.05); t₁ = 25. Expected: τ₁ = 10 reviews x1, τ₂ = 20 reviews x2, τ_last = 20, next slot 30. x3: a + S = 20 < 30, so **expired**. x4: a + S = 35 ≥ 30, so **censored**. In the drain sensitivity, τ = 30 reviews x4.
  - block construction (CBB wrap, MBB endpoint weights, stationary lengths, time-interval membership of equal timestamps);
  - held-out split buffers;
  - the E2 minimum-usable-splits rule: with 999 usable splits the outputs are "not estimable" and E2 does not pass; with 1,000 they are computed;
  - (A) calibrator-fit replicate validity: zero frauds, fit failure, slope ≤ 0, and numerically constant logits (SD < 1e-8) are invalid; a concentrated but varying output (for example logits with SD 0.01) is **valid**; low-positive replicates are flagged; plus the 90% "not estimable" rule;
  - (B) calibration-evaluation replicates: a negative recalibration slope is **kept** as a result; zero-fraud replicates make only the intercept/slope fits and Brier skill invalid or undefined, while O/E (= 0), Brier and log loss stay valid;
  - per-model calibration status: one failed calibrator leaves the other model calibrated, and rank-based outputs are identical either way;
  - cost formulas and break-even identities;
  - the transaction-specific rule in its three cases: L_i < c_r is never reviewed, even at p_i = 1; L_i = c_r with p_i = 1 is reviewed under the tie policy, and with p_i < 1 is not; L_i > c_r applies t_i. An exact-equality case p_i·L_i = c_r with p_i < 1 is also reviewed. The suite runs before and after implementation, and the existing 147 tests must still pass unchanged.
- The protocol (this document) is committed before 5B code. Deviations are logged in WORK_LOG with their reason.

---

## 13. Outstanding decisions requiring human approval

### 13.0 Final decisions (Phase 5A closure, approved by the project owner, 2026-11-09)

All decisions in the table below are **resolved**: every recommendation is approved as written, with the following explicit final decisions.

| Topic | Final decision | Table rows |
|---|---|---|
| C-F fallback | Retrospective sensitivity analysis only. Never an automatic fallback and never primary evidence | #1b |
| Near-flat calibrator | **No** additional statistical cutoff. The numerical guard SD(calibrated logits) < 1e-8 is retained. The calibrator slope and the logit dispersion (SD of calibrated logits) are always reported | #1c, #1d (resolved: none) |
| Calibration status | LR and HGB are evaluated independently. A calibration failure affects only that model's probability-based evaluation | #1e |
| Champion selection | LR is the default champion. HGB replaces LR only if both E1 and E2 pass | #3 |
| E2 minimum | At least 1,000 usable splits out of 2,000. Otherwise E2 is not estimable and LR remains champion | #3b |
| Amount-aware policy | Excluded from the primary decision-policy scope. Amount-aware cost accounting is retained as a sensitivity analysis | #9 |

All other Revision 4 recommendations remain unchanged and approved. The table below is kept as the record of what was decided.

| # | Decision | Recommendation | Alternatives |
|---|---|---|---|
| 1 | Calibration holdout | C-A: last 20% of train (`Time` ≥ 112,555; 28 frauds), sigmoid | C-B 10%/15%; C-C 25% (night hours) |
| 1b | Role of C-F and the safe alternative (*revised in rev. 2 and 3*) | Uncalibrated scores on a C-A technical failure. C-F is not automatic: an optional **retrospective, not fully prequential** sensitivity (variant (a)), only on request, never primary evidence. Confirms a change to the approved "ensemble fallback" wording | Keep C-F as an automatic fallback (not recommended, Section 6.7); drop C-F entirely. Nested hyperparameter selection inside folds is out of scope |
| 1c | Bootstrap validity rules (*new in rev. 3; split and revised in rev. 4*) | **(A) Calibrator-fit bootstrap:** invalid = zero frauds, fit failure or non-convergence or non-finite parameters, slope ≤ 0, numerically constant logits (SD < 1e-8, a floating-point guard); low-positive (1–4) kept but flagged; ≥ 90% valid (≥ 1,800 of 2,000), otherwise "not estimable" with no interval and no retry. **(B) Calibration-evaluation bootstrap:** only the intercept and slope refits can fail (zero frauds, fit failure, non-finite estimate); a negative slope is a kept result; O/E, Brier and log loss have no fit rule; Brier skill is undefined with zero frauds; 90% per fitted metric. The same numerical-constancy criterion applies to the primary calibrator (Section 6.4) | Another minimum (e.g. 80% or 95%); also exclude low-positive (1–4 frauds) replicates; a different floating-point tolerance (e.g. 1e-10) |
| 1d | Statistical near-flatness criterion for a fitted calibrator (*new in rev. 4; unresolved*) | **None applied** unless the mentor sets one before implementation. Only numerical constancy (#1c) is excluded | A logit-SD floor (e.g. 0.05 or 0.1) or a minimum calibrated-probability range. Any value must be fixed before scoring and justified without results |
| 1e | Calibration status per model (*new in rev. 4*) | Decided independently for LR and HGB; the affected model alone loses probability-based outputs; rank-based outputs unaffected | Force both models uncalibrated if either fails (not recommended: it discards a valid calibrator for no ranking benefit) |
| 2 | Accept the train reduction and the 5.66 h gap (Phase 4 numbers not directly reproduced) | Accept; report the C=1 balanced LR on `T_fit` vs Phase 4b | Refit on all train for test (rejected: no holdout would be left for the calibrator) |
| 3 | Champion rule (*revised in rev. 3: E3 removed*) | Incumbent by default; HGB only if **E1** (conditional CI lower bound > 0 at 10/15/30 min) **and E2** (held-out split-stability frequency ≥ 90% and median held-out ΔAP > 0). The W1 → W2 replay is a separate diagnostic and never a gate | Conditional-CI-only rule (biased toward the larger grid); E2 convention of 80% or 95%; Options 2 and 3 (Section 4.3). Reinstating a W2-based gate is **not recommended**: it would make W2 part of champion selection |
| 3b | Selection-aware procedure (*new in rev. 2; wording revised in rev. 3; minimum added in rev. 4*) | Repeated random half-splits of the 21 validation 15-min blocks, 5-min buffer, B = 2,000. **At least 1,000 usable splits** (held-out half with ≥ 10 frauds) are required, as a Monte Carlo adequacy rule; otherwise E2 is "not estimable", does not pass, and the LR is retained, with no retry. Outputs: held-out split-stability frequency (descriptive stability statistic) and a reference-adjusted optimism indicator (heuristic, not unbiased), with the Phase 4 configuration as the fixed reference | Classic out-of-bag bootstrap (held-out set too small with a buffer); no selection-aware step (not recommended); report the unadjusted optimism term only |
| 4 | Hyperparameter selection rule | Max validation AP with a simplicity tie-break | One-SE rule |
| 5 | Search grids | LR 12, HGB 16 as listed | Smaller or larger grids; tune `l2_regularization` |
| 6 | Fixed-K grid | {25, 50, 100, 250, 500}; windows scaled | Rate-based K (e.g. 0.05–1% of traffic) |
| 6b | Fixed-K tie-break key (*new*) | Exactly K reviewed; key = (score desc, `Time` asc, file position asc); ties reported with [TP_min, TP_max] and the expected TP | Random tie-break with a fixed seed; report the expected TP as primary |
| 7 | Time-based capacity definition and values (*event ordering added in rev. 4*) | Online priority queue with SLA expiry; c ∈ {5, 10, 20, 50}/h **[H]**. Event ordering: first slot at t₀ + Δ; arrivals at a slot time eligible; SLA boundary inclusive (expire only when age > S); expiry processed before selection; idle slots lost, not deferred; expired vs censored defined at τ_last + Δ | Bucketed top-k as primary; other c values; first slot at t₀; exclusive SLA boundary; deferred (banked) idle capacity |
| 8 | Cost formulation and grid | Review-cost ratio, R ∈ {5 … 500} **[H]**; amount-aware as sensitivity | Classical FP/FN as primary; different α/c_r grids |
| 9 | Amount-aware decision policy (review iff p_i·L_i ≥ c_r) (*revised wording; equality policy added in rev. 4*) | Not included as a policy (outside the approved scope list). Amount-aware *cost accounting* of rank-based policies is always reported. If the policy is included: L_i < c_r is never reviewed; exact equality p_i·L_i = c_r is reviewed (inclusive ≥), computed as `p_i * L_i >= c_r` in float64 | Include as an optional policy variant (requires calibration conditional on amount); resolve equality as "not reviewed" |
| 10 | Bootstrap scheme and block length (*revised*) | CBB primary (provisional); MBB and stationary as sensitivities; 15 min as the provisional central length; all of 5/10/15/30 min reported; decision envelope {10, 15, 30}; B = 5,000 | Non-circular MBB primary; stationary primary; a single fixed length; include 5 min in the envelope |
| 10b | E2 and envelope conventions (*new in rev. 2; wording revised in rev. 3*) | 90% held-out split-stability frequency (a decision convention, not a confidence level); 5-min buffer | Other conventions or buffer lengths |
| 10c | Calibration uncertainty reporting (*new in rev. 2; bin timing clarified in rev. 4*) | As in Section 6.5: Garwood and block-bootstrap O/E CIs, profile-likelihood and bootstrap slope/intercept CIs, Clopper–Pearson reliability bins (decade edges pre-registered; K-derived edges computed from scores without labels and frozen before labels are evaluated), calibrator-parameter bootstrap | Fewer items (e.g. drop the calibrator-parameter bootstrap); decade edges only |
| 11 | Random-split benchmark in Phase 5 | Excluded | Report HGB on the random split as a secondary benchmark only |
| 12 | Code structure for 5B | New modules and runners only (for example sub-partition helper, challenger builder, `ranking_at_k`, `cost`, `queue`, `bootstrap`, `calibration`, a development runner and a separate guarded final-test runner). No changes to Phase 4 modules | Extend existing runners (would touch Phase 4 code) |
| 13 | Isotonic calibration | Sensitivity only | Exclude entirely |
| 14 | Hypothetical review times for analyst-hours | 5 / 10 / 15 min **[H]** | Omit analyst-hours |

### Methodological risks (summary) [R]

1. **Statistical power:** 56 validation and 52 test frauds. Real but modest model differences may be undetectable, and the champion rule may favour the incumbent because of power, not quality.
2. **Selection optimism:** tuning and champion choice both use validation; the larger HGB grid benefits more. Mitigated, **not removed**, by the full-grid report, the pre-registered incumbent-default rule, the selection-aware estimate and the untouched test.
3. **Prevalence shift by hour of day:** `T_cal` 0.070%, validation 0.131%, test 0.122%, W1/W2 0.108/0.154%. Calibration-in-the-large will be off by design. Cost thresholds based on calibrated p inherit that error.
4. **Few calibration positives (28):** sigmoid parameters are uncertain. The isotonic and 10/15% sensitivities and the calibrator-parameter bootstrap show this; nothing corrects it. If C-A fails technically, probability-based analyses become unavailable (uncalibrated scores).
5. **Burstiness:** one burst can swing recall by several points. The block bootstrap addresses it on validation and test; the windows depend on point estimates.
6. **Coverage:** no night hours in validation or test. Everything describes hour values 12–23 of one day. Value metrics are dominated by a handful of large frauds.
7. **SLA 4 h is weakly identified** on 5–6-hour periods and not identifiable in the windows.
8. **Hypothetical costs:** results are conditional on R. Break-even values are the robust output; cost levels are not.

**New risks identified in revision 2 [R]**

9. **Few effective bootstrap units:** ~21 (validation) and ~24 (test) 15-min blocks. Block-bootstrap CIs are approximate and probably anti-conservative; more replicates do not fix this.
10. **Unverified hit/miss dependence:** block lengths rest on label dependence. If model hits and misses cluster over longer spans, every block CI is too narrow. Detectable only in 5B (pre-registered diagnostic), not correctable after the fact without a validation-informed deviation.
11. **Low power of the combined champion rule:** requiring both E1 and E2 makes replacement hard. The likely outcome "incumbent retained" must not be read as evidence that LR is better.
12. **Noisy selection-aware indicator:** the held-out halves hold ~15–20 frauds, and half-size selection tends to exaggerate optimism. The indicator is coarse, not a correction.
13. **CBB wrap artefact:** 4.9–9.8% of blocks join the end of validation to its start, across a prevalence change. Monitored through the MBB/stationary sensitivities and the centring bias.
14. **Tie sensitivity of HGB at fixed K:** piecewise-constant scores may produce boundary ties where TP_min ≠ TP_max, making P@K and R@K partly order-dependent. Reported as ranges.
15. **Conditional-calibration assumption in the amount-aware rule:** a sigmoid fitted on 28 positives cannot guarantee calibration by amount band. Mis-calibration there directly biases the transaction-specific thresholds.
16. **Deviation from approved wording:** demoting C-F from "fallback" to "optional investigation" changes an approved design item and needs explicit mentor confirmation (decision #1b).

**New or revised risks identified in revision 3 [R]**

17. **Misreading the E2 frequency:** a "90% of splits" figure invites being read as "90% probability" or "90% confidence". It is a stability statistic over resplits of one fixed 21-block period (Section 7.5). Mitigated by mandatory naming and wording rules; misreading remains a communication risk.
18. **E2's dependence on construction choices:** the frequency depends on the 15-min partition, the 5-min buffer and the 10/11 split sizes. Another reasonable construction could cross the 90% convention in either direction. These choices are frozen in advance, and they are not tuned.
19. **Heuristic optimism adjustment:** the reference-adjusted indicator can under- or over-correct, especially for HGB, because the reference is an LR. It must not be read as an unbiased optimism estimate.
20. **No independent check of the full-validation champion before test:** with E3 removed, and with the replay unable to evaluate the full-validation champion independently, test is the *only* independent evaluation of the chosen champion. A wrong promotion is detectable only at test, and test results cannot reverse the decision (Section 4.3).
21. **Non-prequential C-F:** if C-F is ever run, its results carry an unquantified dependence on future (validation) labels through the hyperparameters. It is mitigated only by restricting C-F to a labelled retrospective sensitivity.
22. **Calibrator-bootstrap invalidity:** with 28 positives over ≈ 23 blocks, zero-fraud or degenerate `T_cal` replicates may be frequent. If fewer than 90% are valid, calibrator parameter uncertainty is reported as not estimable. Excluding invalid replicates also tends to understate the uncertainty.

**New risks identified in revision 4 [R]**

23. **E2 not estimable:** bursts can leave many held-out halves with fewer than 10 frauds. If fewer than 1,000 of 2,000 splits are usable, E2 cannot pass and the LR is retained by rule, so the champion decision may turn on resampling feasibility rather than on model evidence. The report must say so explicitly.
24. **No statistical flatness guard:** only numerically constant calibrators are excluded. A calibrator that is nearly flat but varying (decision #1d, unresolved) passes and yields probabilities with little spread. It is visible in the reported calibrator slope and logit spread, but it is not excluded.
25. **Mixed calibration status:** if only one model's calibrator passes, the probability-based comparisons between the models are unavailable, and the cost-threshold analyses cover one model only. The champion decision is unaffected, because it is rank-based.
26. **Queue-convention sensitivity:** results depend on the pre-registered event-ordering conventions (first slot at t₀ + Δ, inclusive SLA boundary, lost idle capacity). Other defensible conventions would shift recall and delay slightly, especially at c = 5 and SLA 1 h. The conventions are fixed in advance and not tuned.

---

## Appendix A. Descriptive computations performed for this protocol [V]

- **What:** pandas/NumPy statistics on `data/raw/creditcard.csv` (read-only): partition durations and row volumes; per-hour and per-SLA-bucket row and fraud counts in validation; `T_fit`/`T_cal` option sizes at timestamp boundaries (10/15/20/25% of train rows); fraud-count dispersion with a label-permutation null (500 permutations, `default_rng(0)`); lag-1 autocorrelation of binned fraud counts; fraud inter-arrival and burst clustering; block counts per block length; and fraud `Amount` summaries.
- **Label scope:** train and validation labels only. Test contributed only label-free row counts per elapsed hour. No test label, test fraud count or test amount was read.
- **No model:** nothing was fitted, scored or tuned. Library signatures were inspected with `inspect.signature`.
- **Where:** the script lives in a session scratch directory outside the repository and is not committed (as in the Phase 2–4 audits). The values here are reproducible from the description above. 5B should reimplement any value it relies on inside tested project code.
- **Environment:** `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1), run with `-I -B`.
- **Revision 2 additions:** no new data access. The C-F fold counts (Section 6.7) are derived by subtraction from the revision 1 cut-point counts. The exact Poisson intervals (Section 6.5) were computed with `scipy.stats.chi2` from the fraud counts 23, 28, 33 and 56 alone. The out-of-bag and held-out coverage figures (Section 7.5) are closed-form arithmetic.

## Appendix B. Revision log

**Revision 2 — 2026-11-09 (methodology review M01–M08).** Only this document was changed. Not approved; approval is the human reviewer's.

| Finding | Severity | Sections changed | Change |
|---|---|---|---|
| M01 Bootstrap and selection uncertainty | Critical | 1, 4.2, 4.3, 7.3, 7.5 (new), 11.1, 12.1, 13 (#3, #3b), risks 11–12 | Separated conditional uncertainty from selection uncertainty (variability and optimism). Explained why bootstrapping fixed selected predictions cannot capture optimism, and stated that no same-data bootstrap removes it. Assessed three rule types. The champion rule is now incumbent-by-default with three gates (E1 conditional, E2 selection-aware, E3 temporal replay). Added a selection-aware procedure: repeated buffered half-splits of the validation blocks, with a fixed-reference optimism estimate. Execution order renumbered (the champion decision is now step 8) |
| M02 Temporal calibration fallback | Critical | 1, 4.2, 5.5, 6.2, 6.3, 6.4 (rewritten), 6.7 (new), 11.3, 12.1, 13 (#1, #1b), risk 16 | Specified C-F fully: four expanding-window folds with exact `Time` boundaries, training and calibration periods with counts, out-of-fold prediction, two variants, and leakage-free generation of validation and test predictions. Showed it is not defensible as an automatic fallback (4-fraud folds, 71% night positives, score-scale mismatch or rank change). Replaced the automatic fallback with **uncalibrated scores**, and listed which analyses become unavailable |
| M03 Circular bootstrap | Major | 7.2 (rewritten), 11.1, 12.1, 12.2, 13 (#10), risks 9 and 13 | Compared non-circular MBB, CBB and stationary bootstrap, including the endpoint under-weighting of MBB and the wrap artefact of CBB (4.9–9.8% of blocks). Described block construction on irregular timestamps (time intervals, 1-s start grid, truncation to duration T). Stated that B does not add information, that ~11–24 effective units make the CIs approximate and possibly anti-conservative, and added the block-jackknife SE and a centring-bias report |
| M04 Block length | Major | 1, 7.1 caveat, 7.4 (new), 12.1, 13 (#10, #10b), risk 10 | Reassessed using label dependence, effective block counts and frauds per block. Model TP/FN dependence is declared unobservable in 5A, with a pre-registered 5B diagnostic (including discordant frauds). **15 min is now provisional**; all of 5/10/15/30 min are reported, and the decision uses a fixed conservative envelope {10, 15, 30}. Automatic block-length selection is not used for decisions |
| M05 Amount-aware loss | Major | 6.1, 8.3 (rewritten), 13 (#9), risk 15 | Corrected the decision rule to the transaction-specific threshold: review iff p_i·L_i ≥ c_r, i.e. t_i = c_r/(α + λ·Amount_i), including the never-review case t_i ≥ 1. Stated assumptions (conditional calibration, full prevention, constant c_r, independence, p used as exact). Contrasted it with the global 1/R rule. Separated amount-aware cost *accounting* (always reported, calibration-free) from the amount-aware *policy* (optional) |
| M06 Fixed-K tie handling | Major | 9.2, 9.3, 10.2, 11.1, 12.1, 12.2, 13 (#6b), risk 14 | Exactly K reviewed; deterministic, label-free key (score desc, `Time` asc, file position asc), shared with the queue. Ties reported separately (n tied, inside/outside, [TP_min, TP_max], hypergeometric expected TP) with a "tie-sensitive" flag. Removed the earlier "include all tied rows" rule, which could exceed K |
| M07 Temporal transfer | Major | 1, 10.2, 10.3 (new), 10.4 (renumbered), 10.5 (renumbered) | Defined four distinct quantities: Q1 W1-selected on W2 (transferable), Q2 W2 oracle (never deployable), Q3 selection-aware uncertainty, Q4 actual transferable performance = Q1 with Q3. Regret is defined as a selection-sensitivity diagnostic, not degradation. Stated what one transfer cannot separate. Added a presentation rule and a reduced W1 champion rule |
| M08 Calibration uncertainty | Major | 6.5 (rewritten), 11.1, 12.1, 13 (#10c) | Required CIs for O/E (exact Poisson with worked widths, plus block bootstrap), intercept and slope (profile likelihood plus block bootstrap), Brier/log loss, and reliability bins (Clopper–Pearson on pre-fixed decade edges, minimum-event flag). Added calibrator-parameter bootstrap propagation, small-sample limitation statements, and a **no-repeated-modification** rule: frozen before validation scoring; changes count as validation-informed deviations |

Decisions preserved unchanged from revision 1: the partition layout and C-A holdout; the sigmoid method; the 28-configuration grid and selection rule; the K grid, capacities, SLAs and cost grid; the W1 → W2 replay structure; the reporting and leakage rules; the test policy.

**Revision 3 — 2026-11-09 (targeted corrections R3-01–R3-05).** Only this document was changed. Not approved; approval is the human reviewer's. No training, scoring or experiment, and no data access.

| Item | Sections changed | Change |
|---|---|---|
| R3-01 E2 interpretation | 1, 4.3, 7.5, 11.1, 13 (#3, #3b, #10b), risks 17–18 | The E2 quantity is named **held-out split-stability frequency**. It is defined as a descriptive resampling stability statistic, conditional on the fixed validation period, the 28 score vectors, and the block partition and buffer. It is explicitly not a probability, confidence level, significance level or independent replication rate. The 90% convention and the median held-out ΔAP requirement are kept. E2 is stated to be a decision heuristic, not a correction for selection bias |
| R3-02 W2 removed from champion promotion | 1, 4.2 (steps 7–8 swapped), 4.3, 10.2, 10.4, 10.5, 11.1, 11.2 (new rule 7), 12.1 (new row), 13 (#3), risks 11 and 20 | E3 removed. Primary rule: incumbent by default; HGB only if E1 **and** E2. The champion is decided at step 7, before the replay (now step 8). The replay is a diagnostic whose outcome never affects the champion. Three roles distinguished: full-validation selection (E1 + E2), the W1 → W2 diagnostic, and the final test. Added a scope note: the replay does not independently evaluate the full-validation champion, whose selection used W2 rows as part of validation. The W1 → W2 analysis is kept in full |
| R3-03 C-F not fully prequential | 1, 6.2, 6.4, 6.7, 12.1, 13 (#1b), risk 21 | Removed the "leakage-free" and "no future leakage" claims. OOF rows are out-of-fold only with respect to model fitting. The complete procedure is not prequential, because fold hyperparameters were selected with later validation labels. C-F is an optional retrospective sensitivity only, never an automatic fallback or primary evidence. Nested selection is explicitly not implemented |
| R3-04 Calibrator bootstrap failures | 6.5 (new rule table), 11.1, 12.2, 13 (#1c), risk 22 | Pre-registered invalid categories: zero frauds, fit failure or non-convergence, slope ≤ 0, degenerate output. Invalid replicates are excluded and counted; low-positive (1–4 frauds) replicates are kept but flagged. A minimum of 90% valid replicates (≥ 1,800 of 2,000), otherwise "not estimable" with no interval, no retry and no substitute. The same rule applies to the recalibration-fit bootstraps. Also corrected the `T_cal` duration from 5.65 h to 5.66 h and fixed the bootstrap as CBB with B = 2,000 |
| R3-05 Reference-adjusted optimism | 4.3, 7.5, 11.1, 13 (#3b), risk 19 | Renamed the "optimism estimate" to **reference-adjusted optimism indicator (heuristic)**. Stated its additive-equal-effect assumption and why that assumption can fail (non-linear prevalence dependence of AP; HGB vs LR reference). It is not unbiased, has no coverage claim, and is reported next to the unadjusted term. The calculation is unchanged |

*Correction to the revision 2 log:* the M02 row above says C-F has "leakage-free generation of validation and test predictions". Per R3-03, that holds only for model and calibrator *fitting*. The complete C-F procedure is not fully prequential. Likewise, the M01 row's three-gate rule (E1, E2, E3) and the M07 row's "reduced W1 champion rule" are superseded by R3-02: E3 is removed, and the W1 replay champion is a diagnostic only. The revision 2 rows are kept as a historical record.

Decisions preserved unchanged from revision 2: everything not listed above, including the E1 envelope, the 90% E2 convention, the median requirement, the half-split design, the C-A holdout, the grids, capacities, costs, the replay design and the test policy.

**Revision 4 — 2026-11-09 (targeted corrections R4-01–R4-06, following the mentor's revision 3 review).** Only this document was changed. Not approved; approval is the human reviewer's. No training, scoring or experiment, and no data or test-label access.

| Item | Sections changed | Change |
|---|---|---|
| R4-01 Minimum usable splits for E2 | 4.3, 7.5 (new step 8 and reported quantities), 11.1, 12.2, 13 (#3b), risk 23 | Requires ≥ 1,000 usable splits of the 2,000 attempted before any Section 7.5 output is computed. Otherwise: "not estimable under the pre-registered rule"; attempted, skipped and usable counts reported; frequency and median not calculated; no gate use; no retry. Stated as a Monte Carlo adequacy requirement only: not independent samples, and no increase in effective sample size. Made explicit that a not-estimable E2 does not pass, so the LR is retained. This follows from the existing E1-and-E2 rule and is not a change to it |
| R4-02 Separate fit vs evaluation bootstrap validity | 6.4, 6.5 (validity rules rewritten as (A) and (B)), 11.1, 12.2, 13 (#1c, new #1d), risk 24 | Withdrew the single-bin degeneracy rule, in the calibrator-fit bootstrap and in the primary failure trigger. It is replaced by a numerical-constancy criterion: SD of calibrated logits < 1e-8, a floating-point guard justified without results. A statistical near-flatness cutoff is left as unresolved decision #1d (none applied). Kept: zero-positive, fit-failure, non-finite and slope ≤ 0 invalidity for the *calibrator*; low-positive flagging; the 90% rule and the no-retry rule. For the evaluation bootstrap: only the intercept and slope refits can fail; a negative calibration slope is a kept result; O/E, Brier and log loss have no fit-validity rule; Brier skill is undefined with zero frauds; 90% per fitted metric |
| R4-03 Amount-aware equality case | 8.3, 12.2, 13 (#9) | Replaced "never reviewed if L_i ≤ c_r" with three cases: L_i < c_r is never beneficial; L_i = c_r with p_i = 1 is an indifference case; L_i > c_r applies the normal threshold. Pre-registered tie policy: exact equality p_i·L_i = c_r is reviewed (inclusive ≥), computed in one fixed float64 form with no tolerance, and equality cases are counted. The rest of the formulation is unchanged |
| R4-04 Queue event ordering | 9.3, 11.1, 12.2 (hand-computed cases Q1–Q5), 13 (#7), risk 26 | Defined t₀ and t₁; first slot at t₀ + Δ; arrivals at a slot time eligible; inclusive SLA boundary; expiry before selection; idle slots lost, not deferred; at most one review per slot; expired vs censored defined against τ_last + Δ; drain sensitivity. Capacity, SLA and tie-break design unchanged |
| R4-05 Per-model calibration failure | 4.2 (step 4), 6.4, 11.3, 12.2, 13 (new #1e), risk 25 | Calibration status is decided per model; only the affected model loses probability-based metrics and thresholds; rank-based metrics stay comparable. Removed "both calibrated or both uncalibrated" from 11.3 |
| R4-06 Reliability-bin timing | 6.5 (reliability row and the no-repeated-modification rule), 13 (#10c) | Decade edges are data-independent and pre-registered. K-derived edges are computed from model scores (so only after scoring), without labels, and frozen before that partition's labels are evaluated. Removed the claim that they are fixed "before scoring" |

*Correction to the revision 3 log:* the R3-04 row's "degenerate output" definition (all predictions in one reliability bin) and its statement that the same rules apply to the recalibration fits are superseded by R4-02. The revision 3 row is kept as a historical record.

Decisions preserved unchanged from revision 3: everything not listed above, including E1, the 90% E2 convention and the median requirement, the half-split design and the 10-fraud usability threshold, the C-A holdout and sigmoid method, the 90% valid-replicate rule, the K grid and tie-break key, capacities, SLAs, the cost grids, the W1 → W2 replay as a diagnostic, and the test policy.

**Phase 5A closure — 2026-11-09.** The project owner approved Revision 4 and all remaining review recommendations, and Phase 5A is closed. No methodology was changed at closure. Changes made: the approval status in the header; the final decisions recorded in Section 13.0; the editorial typo "Reported: Reported:" in Section 11.3 corrected. No model was trained or scored, and no test label was accessed. Phase 5B has not started.

**Revision 1 — 2026-11-09.** Initial protocol.
