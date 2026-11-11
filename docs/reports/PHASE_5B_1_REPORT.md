# Phase 5B-1 Report — E2 Usability, T_fit/T_cal Partition, Phase 4 Continuity

> **Status (2026-11-11):** implemented and verified; **awaiting mentor review**; not committed. Phase 5B-1 was started on the project owner's explicit instruction, which set the scope (5B-1A/B/C) and the training-partition and score-continuity decisions recorded in Section 2.
>
> **Evidence labels:** **[V]** verified by execution in this phase; **[D]** quoted from a repository document; **[R]** recommendation or interpretation.

## 1. Scope and what was not done

| Stage | Protocol source | Done |
|---|---|---|
| 5B-1A: E2 usable-split count | Section 7.5 steps 1–3, 7, 8; Appendix C3, C4 | Yes |
| 5B-1B: `T_fit`/`T_cal` partition helper | Sections 3.1, 4.2 step 1, 6.2 (C-A), 12.1 | Yes |
| 5B-1C: three-track Phase 4 continuity | Section 4.4; Appendix C5 | Yes |

**Not done [V]:** no hyperparameter grid fit; no HGB fit; no calibrator fit; no `T_cal` scoring; no selection, threshold, cost, queue or bootstrap analysis; no E1 or E2 statistic (re-selection, held-out ΔAP, split-stability frequency); no test-partition scoring or label access by Phase 5B-1 code. No Phase 4 module, Phase 4 result file or Phase 5B-0 module was changed. No locked value in `src/models/phase5_settings.py` was changed (configuration SHA-256 unchanged; see Section 7).

## 2. Owner decisions applied

- **Training partition:** chronological. `T_fit` holds the earlier observations and `T_cal` the later ones, with no shuffling. Fraud counts and estimability are checked, and the test set stays untouched. The boundaries are the frozen ones (`T_FIT_END_TIME` = 112,553, `T_CAL_START_TIME` = 112,555; protocol Section 3.1) **[D]**.
- **Score continuity:** three distinct tracks. (1) Historical Phase 4 LR: full train, `predict_proba`. (2) Phase 5 primary LR: `T_fit`, `decision_function`. (3) Phase 5 diagnostic LR: `T_fit`, `predict_proba`. Historical Phase 4 results are never overwritten. This matches the frozen `reporting.phase4_continuity_comparison` **[D]**.
- **Thresholds:** only the protocol's own (≥ 10 held-out frauds per split; ≥ 1,000 usable splits). No new threshold was introduced. The protocol sets **no** minimum fraud count for `T_fit` or `T_cal`, so estimability there means only that both classes are present (required for any fit).

## 3. Implementation

New modules only (decision #12: no changes to Phase 4 modules):

| File | Role |
|---|---|
| `src/data/train_subpartitions.py` | `assign_train_subpartitions` (pure function of `Time`; refuses validation/test rows and any timestamp strictly between 112,553 and 112,555), `check_temporal_integrity`, `summarize_subpartitions`, `require_estimable` |
| `src/evaluation/e2_usability.py` | `split_counts` (per-split selection, buffered held-out and buffer-dropped row/fraud counts), `usability_decision` (the two pre-registered rules), `e2_usability` (full run with frozen constants and diagnostics) |
| `src/models/phase4_continuity.py` | Builder for the C = 1 balanced LR; `quote_phase4_historical` (read-only); `fit_phase5_tracks` (one `T_fit` fit through the Phase 4 runner `fit_and_score`, scored with both representations); `validate_score_tracks` (every track must carry `score_representation`, matching the frozen design); `continuity_comparison` |
| `src/models/run_phase5b1.py` | CLI. Verifies the dataset SHA-256, removes test rows straight after the label-free chronological assignment, runs 5B-1B → 5B-1A → 5B-1C, and writes `reports/phase5b1_results.json` with `write_result_json` |
| `tests/test_phase5b1.py` | 34 tests (Section 6) |
| `reports/phase5b1_results.json` | 14,326 bytes, strict JSON, within the 5 MB target **[V]** |

Run: `python -m src.models.run_phase5b1` (about 7 s).

**Two implementation notes [V]:**
- `LogisticRegression(penalty="l2")` raises a FutureWarning in scikit-learn 1.9.1 (the parameter is deprecated since 1.8). The L2 penalty is the default, so it is not passed explicitly, exactly as in the Phase 4 builder; the result file records it as `"l2 (scikit-learn default)"`. The focused tests run with `-W error::FutureWarning`.
- The pre-registered rule "refit with `max_iter` 5,000 on a convergence warning" is implemented. It did not trigger: the fit converged in 40 iterations.

## 4. 5B-1B — T_fit / T_cal partition [V]

| Sub-partition | `Time` range | Rows | Frauds | Non-frauds | Prevalence | Estimable |
|---|---|---:|---:|---:|---:|---|
| `T_fit` | 0 – 112,553 | 159,491 | 356 | 159,135 | 0.2232% | yes |
| `T_cal` | 112,555 – 132,928 | 39,873 | 28 | 39,845 | 0.0702% | yes |

- Counts equal the protocol's Section 3.1 / Appendix A values exactly (`matches_protocol_section_3_1: true`).
- `max(T_fit Time)` = 112,553 < `min(T_cal Time)` = 112,555 (2-second gap; no row at 112,554); no timestamp spans both.
- The two sub-partitions cover the 199,364 train rows exactly once, and file order is preserved within each.
- **Model-input twins (`V1`–`V28` + `Amount`, descriptive, Section 12.1):** 566 groups cross `T_fit`/`T_cal`/validation, involving 140 `T_fit`, 800 `T_cal` and 824 validation rows; **0 fraud rows**. Under the Phase 3 key (all columns except `Class`) 0 groups cross, which holds by construction because the key includes `Time`. Consistent with the earlier finding that cross-partition twins are all non-fraud **[D]**; they remain potential leakage of unmeasured effect, not proven leakage. Test rows are excluded from this count.
- `T_cal` holds only 28 frauds and has a lower prevalence than validation (0.131%). This is the known C-A weakness (protocol Section 6.2) **[D]**. It does not affect 5B-1, which fits no calibrator.

## 5. 5B-1A — E2 usability [V]

Settings (all frozen): 21 half-open 15-min blocks from 132,929 (final block [150,929, 151,329), 400 s); 2,000 sequential draws of 10 selection blocks from `default_rng(42)`, repeats kept; buffer zones `[a−300, a) ∪ [b, b+300)`; usability counted after the buffer. Inputs: 42,722 validation rows and 56 validation frauds. Nothing else.

| Quantity | Value |
|---|---:|
| Splits attempted | 2,000 |
| **Usable** (buffered held-out frauds ≥ 10) | **1,984** (99.2%) |
| Skipped | 16 (held-out frauds 7: 1 split, 8: 5, 9: 10) |
| Minimum required | 1,000 |
| **E2 estimable under the pre-registered rule** | **Yes** |
| Distinct splits among the 2,000 | 1,990 (10 repeats retained, Appendix C3) |
| First three splits | identical to the Phase 5B-0 snapshot (NumPy 2.3.2) |

| Per split (all 2,000) | min | p5 | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| Buffered held-out frauds | 7 | 12 | 20 | 27 | 34 |
| Buffered held-out rows | 9,407 | 12,308 | 15,174 | 17,921 | 20,580 |
| Held-out share of validation rows | 22.0% | 28.8% | 35.5% | 41.9% | 48.2% |
| Selection-half frauds | 15 | 20 | 27 | 34 | 40 |
| Frauds dropped by the buffer | 0 | 3 | 9 | 17 | 23 |

- For every split, selection + held-out + dropped frauds = 56 (conservation check).
- Per-block fraud counts range from 0 to 6 (3 of the 21 blocks hold none); full table in the result file.
- The median of 20 held-out frauds is at the top of the protocol's expected "roughly 15–20" **[D]**.

**[R] Interpretation limits:**
- "Estimable" means only that the E2 statistics *may be computed* in a later stage. It says nothing about whether E2 passes, and no E2 statistic was computed.
- The 1,000-split minimum is a Monte Carlo adequacy rule. The 1,984 usable splits all reuse the same 21 blocks and 56 frauds; they are not independent samples (Section 7.5).
- Held-out halves of about 20 frauds keep held-out AP noisy, as the protocol anticipates.

## 6. 5B-1C — three-track Phase 4 continuity [V]

All tracks are evaluated on the same 42,722 validation rows (56 frauds). Configuration: LR C = 1, `class_weight="balanced"`, lbfgs, `max_iter` 1,000, `random_state` 42, StandardScaler inside the pipeline.

| Track | Fit rows | Score | Role | AP | ROC-AUC |
|---|---|---|---|---:|---:|
| 1. Phase 4 historical | full train (199,364; 384 frauds) | `predict_proba` | historical reference, **quoted** from `reports/phase4b_baseline_results.json` | 0.839660 | 0.981517 |
| 2. Phase 5 primary | `T_fit` (159,491; 356 frauds) | `decision_function` | primary | **0.853574** | 0.981031 |
| 3. Phase 5 diagnostic | same fit as 2 | `predict_proba` | diagnostic only | 0.853574 | 0.981031 |

| Difference | ΔAP | ΔROC-AUC | What it reflects |
|---|---:|---:|---|
| 2 − 1 | +0.013913 | −0.000486 | `T_fit` (dropping `T_cal`), the 5.66 h gap and the representation, jointly |
| 3 − 1 | +0.013913 | −0.000486 | the data change only (same representation) |
| 2 − 3 | 0.000000 | 0.000000 | the representation only (same fit) |

**Score diagnostics (validation):** both representations have 41,807 unique values; 1,378 rows fall in tied groups (identical model inputs). There are no probabilities equal to 1.0. The largest logit is 31.55, below the 36.74 saturation point, and the resolution loss above about 30 created no additional ties here. `predict_proba` equals `expit(decision_function)` exactly (maximum absolute difference 0).

**[R] Interpretation:**
- For this one configuration, `predict_proba` and `decision_function` give identical rankings, so the Phase 4 and Phase 5 representations do not diverge for the Phase 4 model. The `decision_function` rule still matters for grid points that reach the saturation regime (Appendix C5).
- The +0.014 AP is a single point estimate from 56 frauds without an interval. It is **not** evidence that training on `T_fit` is better. It combines fewer training rows, the removal of the most recent 28 training frauds, and the gap. The independent Phase 4 bootstrap interval for the historical AP was [0.746, 0.922] **[D]** (Phase 4 report, Section 3.2), much wider than this difference.
- These are development figures. Nothing was selected from them; the Phase 4 configuration is one of the 12 LR grid points and will be refitted identically in the grid stage.
- `reports/phase4b_baseline_results.json` was read, not written; its SHA-256 is unchanged (test `TestHistoricalTrackIsQuotedReadOnly`).

## 7. Leakage and test-set isolation [V]

| Check | Result |
|---|---|
| Test rows removed before any label access | 42,721 test rows removed right after the label-free `Time` assignment; the maximum processed `Time` is 151,328 |
| Test-label invariance | Flipping every test label and setting every test feature to 1e6 leaves partition, E2 and continuity results unchanged. Counts and E2 are exact; fitted floats agree within 1e-12 (`test_results_invariant_to_test_labels_and_features`) |
| Sensitivity control | Flipping *validation* labels changes the E2 result, so the invariance test is not vacuous |
| Refusals | `assign_train_subpartitions` refuses any `Time` > 132,928. `e2_usability` refuses train- or test-period timestamps. The Phase 4 runner refuses a test evaluation partition |
| `T_fit` fit only | The pipeline (scaler and LR) is fitted on the `T_fit` rows only; `T_cal` is neither fitted nor scored |
| E2 inputs | Validation timestamps and validation labels only; no scores |
| No new thresholds | All constants are imported from `src/models/phase5_settings.py`. The configuration SHA-256 recorded in the result file equals the Phase 5B-0 environment manifest's value |
| Reproducibility | Rerunning the CLI produced a **byte-identical** result file. Seeds and environment (Python 3.13.6, NumPy 2.3.2, pandas 2.3.1, scikit-learn 1.9.1, SciPy 1.17.1) are recorded in the file |

**Numerical note [V]:** bit-level identity of fitted floats needs identical input memory layout as well as identical values. A `DataFrame.copy()` of the same data changed LR coefficients by about 2e-15 and scores by about 4e-15 (BLAS summation order). AP and every count were unaffected. This refines the Phase 5B-0 statement that bit-identical scores are expected only in the verified environment.

**Disclosure:** the consolidated regression suite (Section 8) includes pre-existing Phase 3/4 real-data tests that read the whole CSV and assert already-documented counts, including the test partition's 42,721 rows and 52 frauds (`test_split.TestDocumentedSplitOnRealData`). They score nothing on test and feed nothing into Phase 5B-1. No Phase 5B-1 module or test reads a test label.

## 8. Tests [V]

| Run | Result |
|---|---|
| Focused: `python -W error::FutureWarning -m unittest tests.test_phase5b1` | **34 passed** |
| Consolidated: `python -m unittest discover -s tests` | **275 passed** (241 existing + 34 new), 0 skipped, 0 failures |

Coverage of `tests/test_phase5b1.py`: inclusive frozen boundaries; refusal of the gap timestamp, validation/test rows and invalid timestamps; order, index and equal-timestamp preservation; label-free determinism; integrity violations (order, shared timestamp, empty side); counts and estimability (missing positive or negative). It also covers: the 10-fraud and 1,000-split boundaries (9 vs 10, 999 vs 1,000); a hand-computed buffer case; fraud and row conservation; refusal of out-of-period timestamps; frozen draws and determinism; the score-track validator (missing field, wrong or swapped representation, missing track); the read-only historical quote; T_fit-only fitting; recomputation of the primary AP; the test-isolation invariance and its control; and real-data checks (`T_fit`/`T_cal` counts; E2 result equal to the recorded file).

## 9. Remaining items and blockers

- **No blocker** for the next stage.
- The result file's provenance records `git.commit` = `726a354` with `dirty: true`, because it was generated from the uncommitted working tree. **[R]** As with the Phase 5B-0 manifest, it can be regenerated after the commit to record a clean commit; the results are byte-identical apart from those two fields.
- Phase 5B-0 recommendation (d) is met only for the continuity result: `score_representation` is enforced on its metric blocks. **[R]** Later 5B result writers should reuse `validate_score_tracks` or apply the same rule.
- `T_cal` holds 28 frauds; its adequacy for calibration is decided by the pre-registered technical-failure rule (Section 6.4) at the calibration stage, not here.
- **[R] Next stage (on approval):** protocol Section 4.2 step 2: fit all 28 configurations on `T_fit` once, check HGB determinism (two fits), score `T_cal` and validation once each, and cache the scores. Then step 3, hyperparameter selection by validation AP on `decision_function` scores.
