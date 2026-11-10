# Phase 5B-0 Completion Report — Protocol and Documentation Remediation

| Item | Value |
|---|---|
| Phase | 5B-0 (preparation and remediation; no experimentation) |
| Date | 2026-11-11 |
| Starting commit | `760ccc3` (`docs: close Phase 5A with approved experimental protocol`); clean working tree at start. After the 2026-11-11 local attribution rewrite this commit is `9b6d8e8`, with an identical tree (Section 16) |
| Approved methodology | `docs/phase_5a_experimental_protocol.md`, Revision 4 (unchanged); clarifications in its Appendix C |
| Mentor review | Two rounds on 2026-11-11. First: listed conventions approved, five corrections applied. Final review: remaining implementation conventions approved, subject to final corrections (applied, Section 6.4). Final approval (sign-off) of Phase 5B-0 as a whole was pending at that point; **final mentor approval was subsequently reported by the owner (2026-11-11)** |
| Status | **Complete and mentor-approved; committed** |
| Implementation commit | `7c4edcdc400b1f4332c7d961f845662f65685ea9` (`Complete Phase 5B-0 methodology and reproducibility foundation`) |
| Environment manifest commit | `aa67802353f3646e7b5eb53237f3da8e8d263626` (`Record clean Phase 5B-0 environment manifest`); a snapshot generated from the clean tree at `7c4edcd`, `dirty: false` |
| Push state | Not recorded here, because it changes without a commit; see `git log` / `git status` |
| Verdict | **COMPLETE WITH CONDITIONS** (Section 15) |

**Evidence labels:** **[V]** verified in this phase (command output, source inspection or test) · **[D]** stated in a committed project document · **[R]** recommendation or interpretation by the implementer.

---

## 1. Executive summary

Phase 5B-0 removes every documentation contradiction found by the Phase 5B readiness audit, freezes the implementation details that Revision 4 left open, and makes the repository reproducible for a new user. **No model was fitted on, and no prediction was made for, the FraudGuard dataset.** No experiment was run, and no test label was read **[V]** (Sections 10 and 14).

The only models fitted in this phase are small estimators and a sigmoid calibrator on *generated* arrays. They exist in one mentor-requested integration test, which runs with dataset reads blocked **[V]**.

Main outcomes:
- **Protocol Appendix C**, an editorial and implementation addendum, not a Revision 5 **[V]**, with items C1–C8:
  - the approval record (owner approvals attributed precisely);
  - the amount-tercile scope;
  - frozen seeds and generator handling;
  - a deterministic E2 block and buffer convention;
  - the score representation and the Phase 4 continuity comparison;
  - storage and manifests;
  - the mentor-review record;
  - the reliability bins and final corrections (C8).
- **Score representation:** `decision_function` for both model families and as the calibrator input. It is verified from the scikit-learn 1.9.1 source, by float64 saturation evidence, and by a synthetic fitted integration test **[V]**. The mentor approved it.
- **Reproducibility:**
  - exact pins verified by a fresh install;
  - a dataset fingerprint check that reads no rows;
  - an environment manifest, a freeze-manifest template and validator, and a result-storage policy;
  - a beginner-level README.
- **Tests:** 94 new tests. 180 safe tests passed under a guard that makes any `fit` call or real-dataset read fail. The 5 integration tests passed with dataset reads blocked **[V]**.

## 2. Scope completed

| Workstream | Status |
|---|---|
| A. Documentation synchronization | Done (Section 3) |
| B. Protocol clarification addendum (B1–B5) | Done; protocol Appendix C, items C1–C5; mentor record in C7 |
| C. Reproducibility and onboarding (C1–C5) | Done; the fresh-clone check is limited to the same machine (Section 13) |
| D. Experiment-tracking infrastructure | Done: freeze-manifest template, validator, environment manifest, storage policy |
| E. Tests and validation | Done (Sections 10–11) |
| Mentor corrections 1–5 | Done (Section 6) |
| Mentor final corrections (wording fixes, reliability lower-tail bin) | Done (Section 6.4; Appendix C8) |

## 3. Files changed

**Modified (10) [V]:**

| File | Change |
|---|---|
| `README.md` | Rewritten as an onboarding guide (15 steps). Status, roadmap and documentation index updated; the personal absolute path removed |
| `data/README.md` | Exact file name, location, size, SHA-256, integrity check, outcomes and dataset policy |
| `requirements.txt` | Exact pins of the verified environment; SciPy added; `jupyter` and `ipykernel` moved to the optional file |
| `.gitignore` | Added `venv/`, `env/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `*.npy`, `*.npz`, `*.parquet`, `*.feather` |
| `docs/phase_5a_experimental_protocol.md` | Header pointer; pre-closure sentences annotated in place (not deleted); Appendix C (C1–C8) appended |
| `docs/PROJECT_CONTEXT.md` | Current phase, documentation list, environment wording, architecture rows. "Awaiting specification" and "ensemble fallback" marked superseded; champion rule; handoff state |
| `docs/WORK_LOG.md` | Current status; Phase 5B-0 session, including the mentor-review entry; next-task chain; superseded note on the briefing's fallback line |
| `docs/validation_strategy.md` | Calibration-holdout, fallback and specification statements marked superseded, with protocol pointers |
| `docs/business_understanding.md` | Specification statement superseded (hypothetical grids, protocol sections); status wording |
| `docs/TECHNICAL_DOCUMENTATION.md` | Phase 5B-0 module reference; new test rows; portable run instructions |

**Created (15) [V]:**
- Requirements and policy: `requirements-notebook.txt`, `docs/result_storage_policy.md`.
- Manifests: `docs/templates/phase5_freeze_manifest.template.json`, `reports/environment_manifest.json`.
- This report.
- Source: `src/models/phase5_settings.py`, `src/models/ranking_score.py`, `src/data/validation_blocks.py`, `src/reproducibility/__init__.py`, `src/reproducibility/dataset.py`, `src/reproducibility/manifest.py`.
- Tests: `tests/test_phase5_settings.py`, `tests/test_validation_blocks.py`, `tests/test_ranking_score.py`, `tests/test_reproducibility.py`.

**Not changed [V]:**
- all Phase 4 source modules, Phase 4 tests and `reports/phase4*.json`;
- the Phase 1–4 reports and the notebook;
- `data/raw/creditcard.csv` (SHA-256 unchanged);
- the split boundaries.

## 4. Decisions implemented

| Decision | Source | Implementation [V] |
|---|---|---|
| E2 split seed 42; `T_cal` bootstrap seed 42; conditional bootstrap `default_rng(0)` | Owner, 5B-0 briefing; mentor-approved | `src/models/phase5_settings.py`; Appendix C3; `test_phase5_settings.py` |
| Amount-tercile calibration not required | Owner, 5B-0 briefing | Appendix C2 and in-place notes (Sections 8.3, 11.1); settings key `amount_tercile_calibration` |
| Git tracking and result-storage policy (≤ 5 MB target, 10 MB hard limit) | Owner, 5B-0 briefing | `docs/result_storage_policy.md`; `.gitignore`; `write_result_json`, `check-size`; tests |
| Addendum, not Revision 5 | Owner, 5B-0 briefing | Appendix C; "Revision 4" kept everywhere |
| Conventions listed in Section 6.1 | Mentor review, 2026-11-11 | Appendix C3–C5, C7 |
| Three-column Phase 4 continuity comparison | Mentor review, 2026-11-11 | Appendix C5; settings key `reporting.phase4_continuity_comparison` |
| Reliability bins [0, 1e-6), [1e-6, 1e-5), …, [0.1, 1.0] (last bin closed) | Mentor final review, 2026-11-11 | Protocol Section 6.5 note; Appendix C8; `RELIABILITY_BIN_EDGES` and `calibration.reliability_bins`; `reliability_bin_index` |
| Section 6.3 conventions | Mentor final review: implementation conventions approved, subject to the final corrections; phase sign-off pending at the time, final approval later reported by the owner | Appendix C7 note |

## 5. Protocol clarification details (Appendix C)

- **C1 Approval wording:**
  - Each resolution cites its source. The six final decisions are the **project owner's**, listed in Section 13.0. Decision #12 (new modules, guarded final-test runner) is covered only by Section 13.0's general owner approval of "all other Revision 4 recommendations", not by an individual listing.
  - **The repository records no mentor sign-off of Revision 4 or of the Phase 5A closure [V].** The mentor's review of Phase 5B-0 is recorded separately in C7.
- **C2 Amount tercile:** not computed in Phase 5B. No amount-aware calibration and no amount-based selection criterion. Amount-aware cost accounting is unchanged.
- **C3 Seeds and generators:**
  - E2: one sequential `default_rng(42)`, split *i* = `sorted(rng.choice(21, 10, replace=False))`, 2,000 draws, repeats kept.
  - `T_cal`: `default_rng(42)`, with identical resampled rows for the LR and HGB calibrators.
  - Conditional bootstrap: `default_rng(0)`.
  - Other bootstraps: `default_rng(0)` under Section 12.2.
  - A fresh generator per analysis makes results **reproducible independently of execution order**. It does **not** make different analyses' streams statistically independent; streams from the same seed are identical. No cross-analysis independence is claimed. Replicates are paired between models within an analysis.
- **C4 Buffer convention:**
  - Blocks are half-open [e_j, e_{j+1}) in whole seconds from 132,929; the final block [150,929, 151,329) holds 400 s and is an ordinary block.
  - Each selection block's buffer zone is [a − 300, a) ∪ [b, b + 300).
  - Boundary seconds: a − 301 is kept, a − 300 is dropped, b is dropped, b + 300 is kept. Kept held-out rows are ≥ 301 s from every selection second.
  - No buffer applies beyond the period, and selection rows are never dropped.
  - Usability (≥ 10 frauds) is counted on the buffered held-out half.
- **C5 Score representation:**
  - `decision_function` (Section 7 below).
  - The constancy guard is SD(z) < 1e-8 with population SD (ddof = 0) and a strict comparison; an SD of exactly 1e-8 is not constant. ddof can change the outcome near the cutoff, which is why it is fixed and tested.
  - The three-column Phase 4 continuity comparison (Section 8).
- **C6 Storage and manifests:** policy file, freeze-manifest template, environment manifest.
- **C7 Mentor review:** the approved decisions, the corrections applied, and the remaining implementation conventions, which the final review approved subject to the final corrections; final approval of Phase 5B-0 as a whole is pending. The former open item is resolved by C8.
- **C8 Reliability bins and final corrections:**
  - Edges 0, 1e-6, …, 1e-1, 1 give seven bins that partition [0, 1]: [0, 1e-6), [1e-6, 1e-5), …, [1e-2, 1e-1), [0.1, 1.0]. Each is half-open except the last, which is closed so that p = 1.0 is included.
  - Comparisons are exact in float64; values outside [0, 1] are rejected.
  - K-derived edges and the "not interpretable" rule are unchanged.
  - C8 also lists the four final wording corrections.

## 6. Mentor review (2026-11-11)

### 6.1 Approved by the mentor (not to be changed)

- E2 sampling: sequential `default_rng(42)`, 2,000 draws, 10 of 21 blocks without replacement per draw, repeated draws retained.
- Half-open block intervals.
- The final 400-second partial block treated as an ordinary block.
- The documented 300-second half-open buffer zones.
- E2 usability counted after buffering.
- The `T_cal` bootstrap seed 42, with identical resampled rows across model families.
- The conditional bootstrap seed 0.
- Other bootstraps seed 0 under protocol Section 12.2.
- `decision_function` for both model families and as the calibration input.
- The Platt mapping A = −a_, B = −b_.

### 6.2 Corrections required and applied [V]

| # | Correction | Where |
|---|---|---|
| 1 | RNG wording: per-analysis generators give order-independent reproducibility only; no cross-analysis independence claimed; pairing preserved | Appendix C3; `phase5_settings.py` comment and the `seeds.generator_handling` key |
| 2 | Constancy guard: ddof = 0 and the strict `SD < 1e-8` kept; tests below, exactly at (SD of `[-1e-8, 1e-8]` is exactly 1e-8 in float64, giving *not* constant) and above the threshold, plus a ddof-sensitive case. ddof is not described as immaterial | `tests/test_ranking_score.py`: `test_constancy_guard_below_threshold`, `test_constancy_guard_exactly_at_threshold_is_not_constant`, `test_constancy_guard_above_threshold`, `test_constancy_guard_uses_population_sd`; Appendix C5 |
| 3 | Synthetic-only integration test with fitted estimators and calibration | `TestSyntheticFittedIntegration` (5 tests), Section 7 |
| 4 | Three-column Phase 4 continuity comparison documented; third column diagnostic only | Appendix C5; settings `reporting.phase4_continuity_comparison`; Section 8 |
| 5 | Governance: C1 attributions tightened; mentor decisions recorded | Appendix C1, C7; this report; WORK_LOG |

### 6.3 Remaining conventions: approved in the final review

These were not individually approved in the first round. In the final review (2026-11-11), the mentor approved them as implementation conventions, **subject to the final corrections in Section 6.4** (Appendix C7). This approval covers the conventions only; **final approval (sign-off) of Phase 5B-0 as a whole was still pending at that point** *(it was later given; see the header)*:
- score negation for a positive label at `classes_[0]`;
- no `predict_proba` fallback, and the exemption of the majority-class reference;
- exact float comparison in ranking, with −0.0 equal to 0.0;
- thresholds on the score scale;
- fresh generators per analysis;
- rejection of fractional timestamps;
- the C-B start times;
- the NumPy-version scope of seeded draws;
- the freeze-manifest structure;
- the storage byte definition and refusal behaviour;
- the requirement pins;
- the dataset-check exit codes.

### 6.4 Final corrections required and applied (mentor final review) [V]

| # | Correction | Where |
|---|---|---|
| F1 | Test-module docstring acknowledges the synthetic fitted integration tests | `tests/test_ranking_score.py`, module docstring |
| F2 | Appendix C opening: no model fitted, scored or calibrated on FraudGuard data; synthetic fitting did occur | Appendix C, opening paragraph |
| F3 | C3 "Other bootstraps": stale request for mentor confirmation removed (seed 0 approved) | Appendix C3 |
| F4 | Section 12.1: decision #12 attributed to the owner's general approval in Section 13.0, not to the six listed final decisions | Section 12.1 note |
| F5 | Reliability lower-tail bin [0, 1e-6); last bin [0.1, 1.0] closed; full coverage of [0, 1] | Section 6.5 note; Appendix C8; `phase5_settings.py`; `ranking_score.reliability_bin_index`; tests `TestReliabilityBinSettings` (2), `TestReliabilityBins` (6); template and manifest regenerated; new consistency test `test_environment_manifest_matches_current_configuration` for the working-tree manifest's configuration hash |
| F6 | Appendix C7, WORK_LOG and this report no longer imply that the overall Phase 5B-0 sign-off has happened. Approval of the implementation conventions is kept separate from final approval of the phase | Appendix C7 note and "Who decided what"; WORK_LOG; this report (header, Sections 4, 5, 6.3, 13, 15) |
| F7 | "Committed" terminology removed from the manifest and template tests; the files were in the working tree and not yet committed at the time (both have since been committed): `test_environment_manifest_is_valid`, `test_environment_manifest_matches_current_configuration`, `test_template_file_matches_code` (formerly `test_committed_…`) | `tests/test_reproducibility.py`; this report (Sections 10, 12) |

## 7. Score representation investigation and evidence

**Installed versions [V]:** scikit-learn 1.9.1, NumPy 2.3.2, SciPy 1.17.1, Python 3.13.6.

**API evidence (installed source, read with `inspect.getsource`) [V]:**

| Component | Finding |
|---|---|
| `LinearClassifierMixin.decision_function` (used by `LogisticRegression`) | `safe_sparse_dot(X, coef_.T) + intercept_`, reshaped to 1-D; documented as the score of `classes_[1]` |
| `LogisticRegression.predict_proba` (binary) | `_predict_proba_lr`: `expit(decision_function)`, stacked as `[1 − p, p]` |
| `HistGradientBoostingClassifier.decision_function` | `_raw_predict`: `_baseline_prediction` plus the leaf values of every iteration, float64, raveled to 1-D |
| `HistGradientBoostingClassifier.predict_proba` | `HalfBinomialLoss.predict_proba`: `proba[:, 1] = LogitLink.inverse(raw) = expit(raw)` |
| `CalibratedClassifierCV` | Calibrator input from `response_method=["decision_function", "predict_proba"]`. With a `FrozenEstimator`, `ensemble="auto"` resolves to `False`, giving a single calibrator |
| `_SigmoidCalibration.predict` | `expit(-(a_ * T + b_))`, so protocol A = −a_, B = −b_. The fit rescales inputs when max \|F\| ≥ 30 and converts the slope back; it uses Platt's smoothed targets |
| `DummyClassifier` | No `decision_function`. The reference is reported only; a constant score has AP = prevalence |

**Numerical evidence (arithmetic) [V]:**
- `expit(z) == 1.0` for z ≥ 36.74.
- 1,001 distinct logits in [30, 31] give 267 distinct probabilities.
- The stored Phase 4 validation probabilities reach 1 − 8.2×10⁻¹⁵ (logit ≈ 32).
- Synthetic AP: 1.0 on logits vs 0.5 on `expit` when saturation ties a positive with a negative.

**Fitted synthetic confirmation (`TestSyntheticFittedIntegration`; generated data only, dataset reads blocked) [V]:**
- Data: 2-feature Gaussian inputs with a logistic label model, `default_rng(0)`, 3,000 rows per sample.
- Models: an LR (`max_iter=1000`) and an HGB (`max_iter=30`, `early_stopping=False`).

| Test | What it confirms |
|---|---|
| `test_positive_class_is_classes_1_for_both_families` | `classes_ == [0, 1]`; `ranking_scores == decision_function`; mean score of positives > negatives, for LR and HGB |
| `test_predict_proba_is_expit_of_decision_function` | `predict_proba[:, 1] == expit(decision_function)` (rtol 1e-15), for LR and HGB |
| `test_positive_label_sorting_first_is_negated_correctly` | With labels `"a_fraud"`/`"b_legit"`, the positive is `classes_[0]`; `ranking_scores == -decision_function`, and positives still score higher |
| `test_platt_sign_convention_on_fitted_calibrator` | `CalibratedClassifierCV(FrozenEstimator(lr), method="sigmoid")` gives exactly 1 calibrator. Fitted `a_ < 0` and protocol `A > 0`. Calibrated p equals `expit(A·s + B)` with s = `decision_function` (rtol 1e-12), so the calibrator consumed `decision_function`. Calibrated p is non-decreasing in s, and the calibrated logits are not constant |
| `test_reversed_association_gives_non_positive_protocol_slope` | With the label association reversed, `a_ > 0` and `A ≤ 0`, which is the protocol's technical-failure condition |

## 8. Phase 4 continuity comparison (documented for Phase 5B-1; nothing fitted or scored now)

| Column | Model | Score | Role |
|---|---|---|---|
| 1. Phase 4 historical | LR C = 1, balanced, full train | `predict_proba` | Historical reference, quoted from `reports/phase4b_baseline_results.json` (AP 0.8397 **[D]**), never recomputed |
| 2. Phase 5 primary | LR C = 1, balanced, `T_fit` | `decision_function` | Primary |
| 3. Phase 5 diagnostic | The same fit as column 2 | `predict_proba` | **Diagnostic only**: never used for hyperparameter selection, the champion decision, thresholds, calibration or any primary evaluation |

- Columns 1 vs 2 mix the data change (`T_fit` and the gap) with the representation change. Columns 2 vs 3 isolate the representation.
- Frozen in `pre_registered_configuration()["reporting"]["phase4_continuity_comparison"]`.

**How the representations are told apart [V]:**
- Phase 4 files already state probabilities: the `decision_rule_note` in `phase4b` and `evaluation_protocol.scores` in `phase4c`.
- Phase 5 records `ranking_score.method = "decision_function"` in the settings, the environment manifest and the freeze manifest.
- **[R]** The 5B-1 result writer should require a `score_representation` field on every metric block. It is not implemented yet.

## 9. Reproducibility, dependencies and environment

- **README onboarding** covers 15 steps: prerequisites; clone; venv; activation on macOS/Linux, PowerShell and cmd; installation; Kaggle download; exact file name and location; directory tree; fingerprint check with outcomes; tests; pipeline; reproduction expectations; reports; troubleshooting. No personal absolute paths; a test enforces this for code and onboarding files [V].
- **Dataset check:** `python -m src.reproducibility.dataset`, with exit codes 0 / 1 / 2. It hashes bytes and parses no rows [V].
- **Git hygiene:** `git check-ignore` confirms that the dataset, `*.joblib`, `*.npy`, `.venv/` and `env/` are ignored. The 7 `.gitkeep` placeholders stay tracked. A dataset copied into a fresh clone is not listed by `git status` [V].
- **Verified environment** (`reports/environment_manifest.json`) [V]:
  - CPython 3.13.6 on Darwin arm64;
  - numpy 2.3.2, pandas 2.3.1, scikit-learn 1.9.1, scipy 1.17.1, matplotlib 3.10.5, seaborn 0.13.2;
  - ipykernel 7.1.0; joblib 1.6.0; threadpoolctl 3.7.0.
  - The manifest was regenerated after the mentor corrections. Pre-registered configuration SHA-256 `86ab5adb…` (after the final corrections); after the Phase 5B-0 implementation commit it was regenerated from the clean tree. The committed manifest (`aa67802`) records Git commit `7c4edcd` with `dirty: false`. An earlier working-tree version, which recorded `9b6d8e8` with `dirty: true`, is kept in the rewrite backup directory, not in Git.

| Decision | Reason |
|---|---|
| Exact `==` pins | Results depend on library numerics; pins equal the verified versions [V] |
| SciPy added | Required by the protocol's statistics and a hard scikit-learn dependency [D][V] |
| statsmodels not added | Installed (0.14.6) but imported by no code; add and pin it if a 5B module uses it [R] |
| `jupyter` moved to the optional file, unpinned | Not installed in the verified environment [V] |

## 10. Tests performed and results (final run, after the mentor corrections)

| Check | Result |
|---|---|
| New tests: `test_phase5_settings` 16, `test_validation_blocks` 18, `test_ranking_score` 36 (31 non-fitting + 5 integration), `test_reproducibility` 24 | 94 tests |
| **Part 1:** 91 safe existing + 89 new non-fitting tests, under a guard that replaces `fit`/`fit_transform` of LR, HGB, Dummy, Pipeline, CalibratedClassifierCV, `_SigmoidCalibration`, IsotonicRegression, StandardScaler and ColumnTransformer with a raising stub, and blocks `pandas.read_csv` of `creditcard.csv` | **180 passed, 0 violations** [V] (final run, after the final corrections) |
| **Part 2:** `TestSyntheticFittedIntegration`, with fitting allowed and the dataset-read guard still active | **5 passed, 0 dataset reads** [V] (final run) |
| Fresh clone (local `git clone` of `760ccc3`, now `9b6d8e8` with identical files, plus the working-tree changes of the first correction round) with the fresh venv installed from PyPI earlier in this phase | Dataset check MISSING (exit 1); Parts 1 and 2 passed (171 + 5); the 21 real-data tests skipped [V]. **Not repeated after the final corrections.** Earlier in this phase the same venv was created and every pin installed from PyPI, and the dataset check returned OK (exit 0) on a temporary copy, since deleted [V] |
| Freeze-manifest template (10,126 B, working tree) | Regenerated after the final corrections; the working-tree file equals the code-generated template (`test_template_file_matches_code`); a synthetic frozen copy validates; 7 corrupted variants are rejected [V] |
| Size policy | `phase4b` 5,554 B ok; `phase4c` 9,495,967 B over the target but under the hard limit (documented pre-policy exception); environment manifest 2,589 B ok [V] |
| JSON validity, `git diff --check` | All strict JSON; diff check clean [V] |

## 11. Tests intentionally not run, and why

56 existing tests were not run in Phase 5B-0 **[V]**:

| Test class | Count | Reason |
|---|---:|---|
| `test_split.TestDocumentedSplitOnRealData` | 8 | Reads the real dataset, including test-partition labels |
| `test_random_split.TestRandomSplitOnRealData` | 4 | Reads the real dataset, including labels |
| `test_baselines.TestPreprocessing` | 4 | Fits a `StandardScaler` |
| `test_baselines.TestBaselineModels`, `TestExperimentRunner` | 5 + 12 | Fit LR and Dummy models (synthetic data, not requested by the mentor) |
| `test_baselines.TestBaselinesOnRealData` | 4 | Fits the Phase 4 LR on the real dataset |
| `test_threshold_analysis.TestProtocol`, `TestIsolation` | 9 + 4 | Fit LR on synthetic data |
| `test_threshold_analysis.TestSerialization.test_cli_end_to_end` | 1 | Runs the CLI, which fits |
| `test_threshold_analysis.TestThresholdAnalysisOnRealData` | 5 | Fits on the real dataset |

No Phase 4 module changed, so their last documented passing run (147 OK, Phase 4 closure) still applies **[D]**. The full suite now has 241 tests.

## 12. Git tracking verification

- **[V]** Nothing is staged.
- **[V]** Before committing, `git status` listed only the 10 modified and 15 new files of Section 3. They were committed as 24 files in `7c4edcd`, plus the manifest in `aa67802`; the working tree was then clean.
- **[V]** No dataset, model, array or cache file is untracked-but-unignored.
- **[V]** The new JSON files (`reports/environment_manifest.json` and the freeze-manifest template) are committed: the template in `7c4edcd`, the manifest in `aa67802`. They are tracked under the storage policy, and are small, strict and free of absolute paths.

## 13. Remaining limitations

1. **Phase sign-off:** given. The mentor approved the Section 6.3 implementation conventions subject to the final corrections (F1–F7, applied), and final approval of Phase 5B-0 was then reported by the owner (2026-11-11).
2. **No mentor sign-off of Revision 4 or of the Phase 5A closure** is recorded; only the owner's approval is. The mentor's Phase 5B-0 review is recorded in Appendix C7.
3. **Fresh-clone scope:** same macOS arm64 machine and Python build. Linux and Windows instructions and the Kaggle download were not executed.
4. **The environment manifest is a snapshot of `7c4edcd`** (committed in `aa67802`, `dirty: false`). By design it records the commit it was generated from, not later commits. It is regenerated only when the environment, the dataset or the pre-registered configuration changes.
5. **The E2 usable-split count is unknown.** It needs validation labels; this is a 5B-1 step.
6. **Cross-analysis RNG streams are not independent** (by design). Monte Carlo errors at the 10/15/30-min block lengths are correlated, which is reported, not corrected.
7. **The `score_representation` field is not yet enforced** in Phase 5 result files. It is a 5B-1 writer requirement.
8. **Bit-level reproducibility** of scores and ties is expected only in the verified environment (BLAS and NumPy versions).
9. **Private API use:** the sign-convention tests use scikit-learn's private `_SigmoidCalibration` and `calibrated_classifiers_[0].calibrators[0]`. They are stable under the pin but may break on an upgrade.
10. **Synthetic only:** the integration test checks conventions on synthetic data. It says nothing about the FraudGuard models' calibration quality or slopes.
11. **Historical records** still name the original developer's interpreter path (WORK_LOG history, protocol Sections 2.2 and Appendix A, Phase 1–4 reports). They are kept as history; every current-state and onboarding document is free of it.
12. **Resolved:** reliability bins below 10⁻⁶ ([0, 1e-6)), Appendix C8. No open methodological item remains in Appendix C.
13. The freeze-manifest validator does not check that the selected configurations belong to the grid. That is a 5B-1 runner responsibility.

## 14. Confirmation: what was and was not done

**[V]** During Phase 5B-0:
- No model, preprocessor or calibrator was fitted on FraudGuard data.
- No prediction was generated for, or scored on, FraudGuard data, and no AP or ROC-AUC of any project model was computed.
- E1 and E2 were not executed, and no test-partition label was read.

**Fitting that did occur:** only inside `TestSyntheticFittedIntegration`, as the mentor requested. That means one LR, one HGB and an LR with string labels fitted on generated arrays, plus two sigmoid calibrators fitted on generated calibration samples. It is a convention check, not an experiment, selection step or evaluation.

**Data access in this phase:**
- (a) SHA-256 over the raw file bytes;
- (b) `Time`-column-only row counts, to confirm the C-B start times.

The preceding read-only readiness audit also counted train and validation labels. The only random draws over project structures were E2 block-index draws with seed 42 for the regression snapshot, which use no labels or scores.

The split, the raw data, the Phase 4 artifacts and the Revision 4 decisions are unchanged.

## 15. Phase 5B-1 readiness recommendation

**Verdict: COMPLETE WITH CONDITIONS.** Every acceptance criterion and every mentor correction is met, with the limitations of Section 13. **[R]** Before Phase 5B-1:

1. ~~The mentor reviews the final corrections and gives the overall Phase 5B-0 sign-off.~~ Done (2026-11-11).
2. ~~The owner commits Phase 5B-0, then regenerates and commits `reports/environment_manifest.json` (`dirty: false`).~~ Done: `7c4edcd` and `aa67802`.
3. The first Phase 5B-1 actions, each subject to approval:
   - (a) the full 241-test suite (it refits the Phase 4 LR, a reproduction);
   - (b) the label-only E2 usable-split count with seed 42;
   - (c) the `T_fit`/`T_cal` sub-partition helper with a timestamp-integrity test;
   - (d) a result writer that requires `score_representation`.

## 16. Git attribution rewrite (owner-approved, 2026-11-11; pushed 2026-11-11)

Two pushed commits carried `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. With owner approval, the three affected commits were rewritten **locally**. Each file tree is identical **[V]**. Author, committer and timestamps are unchanged **[V]**. Only the trailer was removed **[V]**.

| Old | New |
|---|---|
| `4c463f4` | `a84f94a` |
| `7f3387a` | `c349f04` |
| `760ccc3` | `9b6d8e8` (re-parented only) |

- References in this report and in the other Phase 5B-0 documents now use the new IDs. Where an action used the old object, the old ID is kept next to the new one.
- `reports/environment_manifest.json` was later regenerated from the clean tree after the implementation commit. It records `7c4edcd` (`dirty: false`) and is committed in `aa67802`.
- Backups and the full mapping are in `docs/WORK_LOG.md` ("Local Git attribution rewrite").
- **Pushed (2026-11-11):** after mentor review, `main` was pushed with `--force-with-lease` from `760ccc3` to `9b6d8e8`; the remote history has no Claude trailer.
