# FraudGuard Work Log

## Current status

- **Current phase:** Phase 5 — **Phase 5B-3 (calibration, criteria E1 and E2, champion decision; protocol Section 4.2 steps 4, 6, 7 and the fixed-K part of step 5) implemented and verified, awaiting mentor review (2026-11-14); not committed.** Champion: **LR-10** (E1 and E2 both fail; no sufficient evidence for replacement). See the Phase 5B-3 session below and `docs/reports/PHASE_5B_3_REPORT.md`.
- *Previous status line (end of Phase 5B-2):* Phase 5B-2 parts I and II are **committed in `e4f60ee`** ("Complete Phase 5B-2"; verified as HEAD on 2026-11-14). *(Corrected 2026-11-14: the lines below that say Phase 5B-2 is "awaiting mentor review; not committed" describe the state before that commit.)*
- *Earlier status line (end of Phase 5B-2 part II):* Phase 5 — **Phase 5B-2 part II (protocol Section 4.2 step 2: the 28-configuration grid; step-3 candidates LR-10 and HGB-14 recorded) implemented and verified, awaiting mentor review (2026-11-11); not committed.** Part I (experiments E0–E3) was mentor-approved with a scope clarification (E1/E3 exploratory only).
- *Previous status line (end of part I):* Phase 5 — **Phase 5B-2 (controlled experiments E0–E3: LR / HGB × baseline / + `log1p(Amount)`) implemented and verified, awaiting mentor review (2026-11-11); not committed.** See the Phase 5B-2 session below and `docs/reports/PHASE_5B_2_REPORT.md`. Phase 5 is not complete.
- *Previous status line (end of Phase 5B-1):* Phase 5 — **Phase 5B-1 (E2 usability, `T_fit`/`T_cal` partition helper, three-track Phase 4 continuity) implemented and verified, awaiting mentor review (2026-11-11); not committed.** *(Resolved: Phase 5B-1 is committed in `aa4e63a`; see `git log`.)* See the Phase 5B-1 session below and `docs/reports/PHASE_5B_1_REPORT.md`.
- *Previous status line (end of Phase 5B-0):* Phase 5 — **Phase 5B-0 (protocol and documentation remediation, reproducibility infrastructure) complete and mentor-approved (final approval reported by the owner, 2026-11-11).** Phase 5B-0 is committed: implementation in `7c4edcd` (`7c4edcdc400b1f4332c7d961f845662f65685ea9`), environment manifest in `aa67802` (`aa67802353f3646e7b5eb53237f3da8e8d263626`); see `git log` for the current push state. The manifest is a snapshot of the clean tree at `7c4edcd` (`dirty: false`). Phase 5B-1 has not started (session at the top).
  - Phase 5A (experimental protocol) was approved by the project owner and closed on 2026-11-09, committed in `9b6d8e8` (originally `760ccc3`; see the commit-ID mapping below).
  - The frozen protocol is `docs/phase_5a_experimental_protocol.md` (Revision 4 plus closure), with implementation clarifications in its Appendix C (Phase 5B-0).
  - **No Phase 5 model has been fitted or scored; no Phase 5 experiment or result exists.**
  - Next task: Phase 5B-1, only on explicit approval. *(Done 2026-11-11, awaiting review; next is protocol Section 4.2 step 2, only on approval.)*
  - *Superseded wording (2026-11-09):* "Phase 5B (implementation) has not started; no Phase 5 code … exist". Phase 5B-0 added infrastructure code, but no experiment code.
- **Phase 4:** Model Development & Evaluation, **complete and mentor-approved (2026-11-08)**. Phase 4a (chronological partitioning) is committed in `37d4521`. Phase 4b (baseline modeling) is committed in `4b5b94b`. **Phase 4c (threshold analysis and temporal stability) is approved and complete**, committed in `cffc32c`. The retrospective documentation remediation of Phases 1–4 is complete (see the sessions below). The Phase 1, Phase 2 and Phase 3 reports are committed. The Phase 4c completion work (common-reference comparison) and the Phase 4 retrospective report are committed in `1f510f7`.
- **Key commits on branch `main`** (verified 2026-11-08): the Phase 1 remediation (`0459921`), the Phase 2 report (`9b52a8a`), the Phase 3 report (`ac178c3`), the Phase 4 completion (`1f510f7`, `Complete Phase 4 model development and evaluation`) and the Phase 5 design documentation (`a84f94a`, originally `4c463f4`, `docs: finalize approved Phase 5 design documentation`). Use `git log` for the current `HEAD`.
- **Documentation set:** `README.md`, `docs/PROJECT_CONTEXT.md`, `docs/WORK_LOG.md`, `docs/TECHNICAL_DOCUMENTATION.md` (created in Phase 4b), and `docs/FINAL_REPORT.md` (a pending placeholder, created in Phase 4b). Topic documents: `docs/business_understanding.md`, `docs/validation_strategy.md`. Retrospective phase reports: `docs/reports/` (Phase 1: `phase1_foundation.md`; Phase 2: `phase2_data_understanding.md`; Phase 3: `phase3_data_splitting.md`; Phase 4: `phase4_model_development.md`, committed in `1f510f7`; Phase 5B-0: `PHASE_5B_0_COMPLETION_REPORT.md`; created in the sessions below). Phase 5 protocol: `docs/phase_5a_experimental_protocol.md`. Policy: `docs/result_storage_policy.md`. Dataset: `data/README.md`.

## Phase 5B-3 session — calibration, criteria E1/E2, champion decision (2026-11-14)

Started on the mentor's approval of the readiness assessment with decisions D1–D8 (2026-11-14). Nothing frozen was changed (configuration SHA-256 `86ab5adb…`). Full detail: `docs/reports/PHASE_5B_3_REPORT.md`.

- **Mentor decisions applied:**
  - D1: E1 and E2 are the only champion gates.
  - D2: no numeric AP threshold; practical relevance = additional frauds at equal alert volume.
  - D3: fixed-K, break-even and sensitivity now; time-based queue and per-scenario thresholds deferred. **Logged ordering deviation:** steps 6–7 ran before the deferred part of step 5; the decision uses step 6 only.
  - D4: full Section 6.5.
  - D5: calibration-evaluation bootstrap CBB, 15 min, B = 5,000.
  - D6: own Platt fitter with public APIs only, compared with `CalibratedClassifierCV` + `FrozenEstimator`.
  - D7: zero-fraud E1 replicates dropped and counted; ΔAP = 0 not positive; stationary lengths geometric with mean L, truncated at T.
  - D8: reliability PNGs in `reports/figures/`.
- **New files:** `src/evaluation/{evidence_labels,block_bootstrap,calibration,e2_stability,practical}.py`, `src/models/{phase5b3_champion,run_phase5b3}.py`, `src/visualization/phase5b3_figures.py`, five test files, `reports/phase5b3_results.json` (about 0.32 MB), three figures, `docs/reports/PHASE_5B_3_REPORT.md`. No earlier module or result file changed.
- **Continuity [V]:** the recomputed 28 score vectors reproduce `reports/phase5b2_grid_results.json` exactly (all 28 APs identical; LR-10 / HGB-14 re-selected).
- **Calibration [V] (model-generated; supporting evidence, not a gate):**
  - Both calibrators are valid. LR A = 0.513, B = −7.39; HGB A = 0.461, B = −3.17.
  - All 2,000 `T_cal` bootstrap replicates are valid for each model.
  - Validation O/E: LR 1.22 [Garwood 0.92, 1.59], HGB 1.31 [0.99, 1.70]. Slopes 1.25 and 1.33. Brier 0.000390 vs 0.000395; log loss 0.00227 vs 0.00216.
  - The paired Brier and log-loss differences both contain 0.
  - Both models under-predict on validation, consistent with the prevalence shift between `T_cal` and validation. Reported, not acted on.
  - **Fitter check:** LR differs from the public calibrator by 2.1e-6 in logit, above the 1e-6 flag. The own fit is at the optimum (gradient 3e-11); scikit-learn stops earlier (1e-5). This is documented in the results file; HGB agrees to 5e-9.
- **E1 [V]:** observed ΔAP = +0.0151. CBB 95% intervals: 10 min [−0.0098, +0.0466], 15 min [−0.0105, +0.0480], 30 min [−0.0092, +0.0415]. **Fails.** MBB and stationary agree.
- **E2 [V]:** 1,984 of 2,000 splits usable. Held-out split-stability frequency 77.2% (< 90%); median held-out ΔAP +0.0124. **Fails.**
- **Champion:** **LR-10 retained**: no sufficient evidence for replacement at this sample size; not a finding that LR is better.
- **Fixed-K [V] (model-generated):** TP LR / HGB = 25/25, 44/44, 49/48, 51/51, 52/52 at K = 25/50/100/250/500. Every paired ΔTP interval contains 0.
- **Hypothetical scenarios (assumed_scenario_input → simulated_outcome; HYPOTHETICAL, not business results):** over the assumed R grid {5 … 500} and assumed 5/10/15 min per review, the two models have identical simulated costs at every K except K = 100, where LR is cheaper by R review units. No operating point is selected.
- **Hypothetical disclosure:** four evidence classes in every result block; `require_disclosed` refuses unlabelled scenario content. README, PROJECT_CONTEXT, TECHNICAL_DOCUMENTATION, business_understanding and FINAL_REPORT now state the requirement; `tests/test_hypothetical_disclosure.py` enforces it.
- **Status correction:** PROJECT_CONTEXT and this log now record Phase 5B-2 as committed in `e4f60ee`. Historical lines are kept and marked.
- **Reproducibility [V]:** a second full run is identical except `runtime_seconds`.
- **Tests [V]:** focused 87/87 (calibration 24, bootstrap 20, E2 11, runner 19, hypothetical disclosure 13); full suite 421/421 with `-W error::FutureWarning` (334 before Phase 5B-3). Result file 325,156 B (within the 5 MB target); the three figures are byte-identical across runs.
- **Not done:** the time-based queue, per-scenario thresholds, the W1 → W2 replay, the freeze manifest and any test evaluation. No commit.

## Phase 5B-2 session, part II — protocol Section 4.2 step 2: 28-configuration grid (2026-11-11)

The mentor approved part I (below) with a scope clarification: E0 and experiment E2 are protocol-aligned; E1 and E3 (`log1p(Amount)`) remain documented exploratory deviations that must not influence champion selection or modify the protocol. The owner then instructed step 2. Nothing frozen was changed (configuration SHA-256 `86ab5adb…`).

- **New files:** `src/models/phase5_grid.py`, `src/models/run_phase5b2_grid.py`, `tests/test_phase5b2_grid.py` (18 tests), `reports/phase5b2_grid_results.json` (810,757 B). Report: `docs/reports/PHASE_5B_2_REPORT.md`, Part II.
- **Run [V]:** all 28 frozen configurations (12 LR, 16 HGB; 29 features, no `Time`). Each is fitted once on `T_fit` and scores `T_cal` and validation once each (track B `decision_function`, track C `predict_proba`; track A on LR-10 only). Scores are cached in memory (Section 12.2); none is written to disk. `T_cal` labels are not read. Total runtime 3 min 42 s.
- **Fits [V]:** every LR converged within 1,000 iterations (no refit was needed). All 16 HGB configurations are deterministic: two fits gave bit-identical scores. Track C AP equals track B AP for all 28. No saturation; no tie-sensitive K boundary.
- **Results [V]** (validation AP, track B):
  - LR: 0.7201–0.8536 (median 0.8043).
  - HGB: 0.7562–0.8686 (median 0.8299).
  - LR-10 = E0 and HGB-14 = experiment E2, exactly.
- **Step-3 record (Section 5.4 rule; development, optimistic by selection):** LR → **LR-10** (C = 1, balanced; AP 0.853574; spread 0.0493); HGB → **HGB-14** (0.1 / 31 / 20 / balanced; AP 0.868638; spread 0.0387). No tie within 1e-12. Selected-pair ΔAP is +0.0151 (W1 +0.003, W2 +0.024). Not a champion decision.
- **Disclosure:** HGB-14 is the point whose validation AP was seen in part I before the grid ran. The selection rule is mechanical and pre-registered, so no discretion entered.
- **Reproducibility [V]:** a second full run is identical except the wall-clock fields (`runtime_seconds`, `hgb_determinism.*.second_fit_seconds`).
- **Tests [V]:** focused 18/18 (with `-W error::FutureWarning`); full suite 334/334.
- **Not done:** calibration (step 4), operational analysis, criteria E1/E2, the champion decision, the replay and any test evaluation. No commit.

## Phase 5B-2 session — controlled experiments E0–E3 (2026-11-11)

Started on the project owner's explicit instruction, which set the scope: E0 = LR baseline features; E1 = LR + `log1p(Amount)`; E2 = HGB baseline features; E3 = HGB + `log1p(Amount)`. It also set three evaluation tracks (A historical / B primary / C diagnostic), AP as the primary metric, P@K/R@K, temporal stability, capacity interpretation and the safeguards. The experiment IDs E0–E3 are unrelated to the protocol's champion criteria E1/E2. No locked Phase 5B-0 value was changed; the configuration SHA-256 is still `86ab5adb…`.

- **Protocol deviation (logged per protocol Section 12.2):** protocol Section 5.5 excludes feature engineering, naming `log(Amount)`. E1 and E3 run on the owner's instruction as **exploratory development results outside the champion pathway**. No feature set is adopted from them; any later adoption would be a validation-informed change needing mentor approval. *Not a deviation:* the HGB configuration is one point of the frozen grid (Section 5.3), chosen a priori. The three tuned parameters take scikit-learn's defaults (0.1 / 31 / 20), `class_weight="balanced"`, and the protocol's fixed values. No other configuration was fitted.
- **New files:** `src/features/amount_transform.py`, `src/evaluation/capacity.py`, `src/models/phase5b2_experiments.py`, `src/models/run_phase5b2.py`, `tests/test_phase5b2.py` (41 tests), `reports/phase5b2_results.json` (125,670 B), `docs/reports/PHASE_5B_2_REPORT.md`. No earlier module changed.
- **Design [V]:** each experiment is fitted once on `T_fit` (159,491 / 356) through the Phase 4 runner and scores the same 42,722 validation rows (56 frauds) once. `T_cal` is unused; test rows are removed before any label access. Track A (E0 only) is quoted from Phase 4b; tracks B and C are the same fit scored with `decision_function` and `predict_proba`.
- **Results [V]** (validation; track B AP / ROC-AUC):
  - E0 0.853574 / 0.981031 (equal to the Phase 5B-1 primary);
  - E1 0.853869 / 0.980860;
  - E2 0.868638 / 0.979531;
  - E3 identical to E2, bit for bit.
  - Track C equals track B for all four (no saturation; max logit 31.55 LR, 15.35 HGB). Track A for E0 is 0.839660 / 0.981517.
- **Temporal stability [V]:**
  - HGB − LR ΔAP is +0.003 in W1 (23 frauds) and +0.024 in W2 (33 frauds); HGB's W1 ROC-AUC is lower (−0.004).
  - On the 1,984 usable held-out halves of the frozen E2 splits (fixed configurations; descriptive, not criterion E2), HGB − LR has median ΔAP +0.016, 5th–95th percentile −0.011 to +0.057, and is positive in 80% of splits.
  - `log1p` in LR: median −0.00004, and negative in both windows although +0.0003 on pooled validation (noise).
- **Capacity [V]:** at the frozen illustrative K grid {25, 50, 100, 250, 500}, all models catch 25 / 44 / 48–49 / 51 / 52 of 56 frauds with 0 / 6 / 51–52 / 199 / 448 legitimate alerts. HGB catches one fraud fewer at K = 100. No boundary is tie-sensitive. The AP gain does not appear at these capacities.
- **Finding:** `log1p(Amount)` cannot change an HGB. It is monotone, so it gets the same bins (189) as `Amount`; every split ties in gain and goes to the earlier column. It was never used in a tree, and E3 = E2 exactly. For LR its effect is negligible.
- **Cost [V] (wall clock):** LR fit about 0.5 s, HGB about 10 s (300 trees); scoring 0.005 s vs 0.15 s. HGB determinism holds: two fits give bit-identical scores.
- **Reproducibility [V]:** a CLI rerun is identical except `runtime_seconds`.
- **Tests [V]:** focused 41/41 (with `-W error::FutureWarning`); full suite 316/316.
- **Not done:** the 28-configuration grid, selection, calibration, criteria E1/E2, bootstrap intervals, queue/cost analysis, the W1 → W2 replay and any test evaluation. No commit.

## Phase 5B-1 session — E2 usability, T_fit/T_cal partition, Phase 4 continuity (2026-11-11)

Started on the project owner's explicit instruction. Owner decisions applied: a chronological `T_fit`/`T_cal` split (no shuffling; fraud counts and estimability checked; test untouched), and three separate score tracks (historical Phase 4 `predict_proba` on full train; Phase 5 primary `decision_function` on `T_fit`; Phase 5 diagnostic `predict_proba` on the same fit). No locked Phase 5B-0 value was changed; the configuration SHA-256 is still `86ab5adb…`.

- **New files:** `src/data/train_subpartitions.py`, `src/evaluation/e2_usability.py`, `src/models/phase4_continuity.py`, `src/models/run_phase5b1.py`, `tests/test_phase5b1.py` (34 tests), `reports/phase5b1_results.json` (14,326 B), `docs/reports/PHASE_5B_1_REPORT.md`. No Phase 4 or Phase 5B-0 module changed.
- **T_fit/T_cal [V]:** `T_fit` = `Time` 0–112,553, 159,491 rows, 356 frauds; `T_cal` = 112,555–132,928, 39,873 rows, 28 frauds. Equal to protocol Section 3.1. No timestamp spans both; both sides are estimable. Model-input twins crossing `T_fit`/`T_cal`/validation: 566 groups, 0 fraud rows.
- **E2 usability [V]:** 2,000 attempted (1,990 distinct), **1,984 usable**, 16 skipped (held-out frauds 7–9). **E2 estimable** (≥ 1,000). Buffered held-out frauds: median 20, range 7–34. No E2 statistic was computed.
- **Continuity [V]** (validation, 56 frauds), as AP / ROC-AUC:
  - historical (quoted): 0.839660 / 0.981517;
  - Phase 5 primary (`decision_function`, `T_fit`): 0.853574 / 0.981031;
  - diagnostic (`predict_proba`, same fit): identical to the primary. There are no saturation ties (max logit 31.55; 41,807 unique scores in both representations).
  - The fit converged in 40 iterations with `max_iter` 1,000.
- **Isolation [V]:** test rows are removed before any label access. Results are invariant to flipping test labels and corrupting test features. Rerunning the CLI gives a byte-identical file.
- **Tests [V]:** focused 34/34 (with `-W error::FutureWarning`); full suite 275/275, 0 skipped.
- **Findings:**
  - `penalty="l2"` is deprecated in scikit-learn 1.9.1, so it is not passed; L2 is the default.
  - Fitted floats depend on input memory layout at about 1e-15 (`DataFrame.copy()`); counts and AP are unaffected.
- **Not done:** the grid, HGB, calibration, E1/E2 statistics, operational analyses and any test evaluation. No commit.

## Local Git attribution rewrite (2026-11-11; owner-approved; pushed 2026-11-11)

Scope: removal of two `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailers from commit messages, so Claude is not listed as a contributor. Local only: **nothing pushed**; `origin/main` on GitHub still points to the old `760ccc3`. *(Resolved, 2026-11-11: after mentor review, `main` was pushed with `--force-with-lease` from `760ccc3` to `9b6d8e8`; the remote history has no Claude trailer.)* Not done: changes to any file content in history, to methodology or code; no commit of Phase 5B-0.

- **Remote check before rewriting:** `git ls-remote origin` showed one branch, `refs/heads/main` (and `HEAD`) at `760ccc3f229d9af6f79203f1d4ece44d1a9da517`, unchanged.
- **Method:**
  - The raw commit objects were rewritten with `git hash-object`.
  - The exact trailer and its preceding blank line were removed from two messages, and the parent lines were updated.
  - Tree, author, committer, timestamps and all other message bytes are unchanged. No commit was signed.
  - `main` was then moved with `git update-ref`, which leaves the index and working tree untouched.
- **Commit-ID mapping** (the files of each commit are identical; only the IDs changed):

  | Old | New | Subject | Change |
  |---|---|---|---|
  | `4c463f474711dc78a1ee4fe047072ecd49c712d4` | `a84f94a503a4f70cbaba2186f96cae5202e554a0` | docs: finalize approved Phase 5 design documentation | Trailer removed |
  | `7f3387a52765751c7355aed390322d5969c79d72` | `c349f04ffab5597bc5f8b2a454092fd4b112849f` | docs: synchronize Phase 5 project status | Trailer removed |
  | `760ccc3f229d9af6f79203f1d4ece44d1a9da517` | `9b6d8e8855fe7ad27995df7727fd17215551d27e` | docs: close Phase 5A with approved experimental protocol | Re-parented only |

  The 11 earlier commits (`e4e866d` … `1f510f7`) are unchanged.
- **Historical files:** files committed inside the rewritten commits still mention `4c463f4` as they always did. They were deliberately not altered; use this table to resolve old IDs. Current documents now cite the new IDs, and the old ID where an action used the old object.
- **Backups:**
  - branch `backup/pre-attribution-rewrite` (at `760ccc3`);
  - `.git/backups/phase5b0-attribution-rewrite-2026-11-11/`: a Git bundle of the old history, a patch of the tracked working-tree changes, a tar of the untracked files, the pre-rewrite `git status`, and `SHA256SUMS`;
  - a copy of the same files in the session scratchpad.
  - Restore was verified: the bundle clone plus patch plus tar matches all 62 working-tree files byte-for-byte.
- **Status:** awaiting owner review. A push would require `git push --force-with-lease=main:760ccc3f229d9af6f79203f1d4ece44d1a9da517 origin main`, and is not done. *(Resolved, 2026-11-11: pushed with exactly this command after mentor review; GitHub `main` = `9b6d8e8`.)*

## Phase 5B-0 protocol and documentation remediation (2026-11-11)

Scope: implementation of the owner-approved Phase 5B-0 briefing, following the read-only Phase 5B readiness audit of `760ccc3` (now `9b6d8e8`, identical files; verdict READY WITH CONDITIONS). **Not done:**
- model fitting of any kind (LR, HGB, calibrators; also no synthetic fits);
- prediction, scoring or metric evaluation; E1/E2 execution;
- test-label access; changes to the split, the raw data or the Phase 4 code and results; changes to the Revision 4 scientific decisions;
- commits or pushes.

Full report: `docs/reports/PHASE_5B_0_COMPLETION_REPORT.md`.

- **Owner decisions implemented:**
  - seeds: E2 splits 42, `T_cal` calibrator bootstrap 42, conditional bootstrap `default_rng(0)` (unchanged);
  - amount-tercile calibration is not required in Phase 5B;
  - Git tracking and result-storage policy: compact JSON tracked, target ≤ 5 MB, hard limit 10 MB;
  - an editorial and implementation addendum instead of a Revision 5.
- **Protocol:**
  - Appendix C added (C1 approval record and pre-closure wording; C2 amount tercile; C3 seeds; C4 E2 block and buffer convention; C5 score representation; C6 storage and manifests).
  - Pre-closure sentences annotated in place, not deleted.
  - No mentor sign-off of Revision 4 is recorded in the repository, and none is claimed.
- **Score representation (C5):** `decision_function` for both LR and HGB, for ranking, AP, Precision@K, the queue and the calibrator input. It is verified from the scikit-learn 1.9.1 source, and float64 `expit` saturation (exactly 1.0 for logits ≥ 36.74) rules out `predict_proba`. The Platt sign mapping is A = −a_, B = −b_.
- **Code added (no fitting, no data access except the byte-level fingerprint):** `src/models/phase5_settings.py`, `src/models/ranking_score.py`, `src/data/validation_blocks.py`, `src/reproducibility/{__init__,dataset,manifest}.py`.
- **Tests added:** 76 synthetic tests in `tests/test_phase5_settings.py`, `test_validation_blocks.py`, `test_ranking_score.py` and `test_reproducibility.py`.
- **Reproducibility:**
  - `requirements.txt` exactly pinned to the verified environment, adding SciPy 1.17.1. statsmodels is not added, because no code needs it yet.
  - `requirements-notebook.txt` added. *Superseded:* the old unpinned list included `jupyter`, which the verified environment does not have.
  - `reports/environment_manifest.json` added; freeze-manifest template at `docs/templates/phase5_freeze_manifest.template.json`; `docs/result_storage_policy.md`; `.gitignore` extended (environments, caches, large arrays).
  - README rewritten as a new-user onboarding guide; `data/README.md` extended.
- **Documentation synchronized:**
  - README, PROJECT_CONTEXT, validation_strategy, business_understanding, TECHNICAL_DOCUMENTATION and this file.
  - Outdated "awaiting specification" lists and the "ensemble fallback" wording are marked superseded, with pointers to the protocol.
  - The roadmap is aligned across files.
- **Verification:**
  - 167 safe tests (91 existing + 76 new) passed under a guard that disables every `fit` method and blocks reading the real CSV; no guard violation.
  - In a fresh clone with a fresh Python 3.13.6 virtual environment installed from `requirements.txt`, on the same machine: the same 167 tests passed, the 21 real-data tests were skipped, and the dataset check returned MISSING and then OK.
  - Not run: 56 existing tests that fit models (synthetic or real) or read the real dataset including test labels.
- **Status:** awaiting mentor review; the owner commits. *(Resolved, 2026-11-11: committed in `7c4edcd`; environment manifest in `aa67802`.)*
- **Mentor review and final corrections (2026-11-11):**
  - The mentor approved the E2 sampling, half-open blocks, the ordinary partial block, the 300-s buffer zones, post-buffer usability, the seeds (`T_cal` 42 with identical rows across families; conditional 0; others 0 under §12.2), `decision_function`, and the Platt mapping A = −a_, B = −b_ (protocol Appendix C7).
  - Corrections applied:
    - RNG wording: order-independent reproducibility, no cross-analysis independence claimed.
    - Constancy-guard tests below, at and above 1e-8, plus a ddof-sensitive case.
    - A synthetic-only fitted integration test (`TestSyntheticFittedIntegration`): the only fitting in Phase 5B-0, on generated data.
    - The three-column Phase 4 continuity comparison, documented and frozen in settings; its `predict_proba` diagnostic column is never used for selection or primary evaluation.
    - C1 approval attributions tightened.
  - The template and environment manifest were regenerated.
  - Tests: 171 safe tests passed under the no-fit, no-dataset guard; the 5 integration tests passed with dataset reads blocked. No real-data test was run.
  - Still uncommitted; awaiting final approval. *(Resolved: see the current status at the top.)*
- **Mentor final review and corrections (2026-11-11):**
  - The mentor approved the remaining implementation conventions (completion report Section 6.3), subject to final corrections. That approval covers the conventions only; final approval (sign-off) of Phase 5B-0 as a whole is still pending.
  - Applied:
    - the four wording fixes: test-module docstring; Appendix C opening (no fitting on FraudGuard data, synthetic fitting acknowledged); C3 stale confirmation request removed; Section 12.1 attribution of decision #12 to the owner's general approval;
    - the reliability lower-tail bin [0, 1e-6). Bins are [0, 1e-6), [1e-6, 1e-5), …, [0.1, 1.0], the last one closed so that p = 1.0 is included (protocol Section 6.5 note, Appendix C8, `RELIABILITY_BIN_EDGES`, `reliability_bin_index`).
  - The template and environment manifest were regenerated in the working tree (configuration SHA-256 `86ab5adb…`; not yet committed at the time, later committed: template in `7c4edcd`, manifest regenerated from the clean tree and committed in `aa67802`), and a consistency test was added that checks the working-tree manifest against the current configuration.
  - Tests: 180 safe tests passed under the no-fit, no-dataset guard; the 5 synthetic integration tests passed with dataset reads blocked.
  - Still uncommitted; awaiting the mentor's review of these corrections before sign-off. *(Resolved: see the current status at the top.)*
- **Mentor documentation corrections (2026-11-11):**
  - Appendix C7, the completion report and this log now keep approval of the implementation conventions separate from final approval (sign-off) of Phase 5B-0 as a whole. **The sign-off has not been given.** *(Resolved, 2026-11-11: final mentor approval of Phase 5B-0 was later reported by the owner.)*
  - Manifest and template tests renamed to drop "committed", since the files exist only in the working tree: `test_environment_manifest_is_valid`, `test_environment_manifest_matches_current_configuration`, `test_template_file_matches_code`.
  - No methodology, model code, Phase 4 artifact or historical result changed. Nothing staged, committed or pushed; awaiting mentor approval. *(Resolved: committed in `7c4edcd` and `aa67802`.)*

## Phase 5A experimental protocol — approved and closed (2026-11-09)

Scope: documentation only. Protocol preparation in four review revisions, then closure. Not done: training, scoring, experiments, threshold tuning, test-label access, or changes to code, data, notebooks, artifacts, requirements or Phase 1–4 documents and results. Phase 5B not started.

- **Deliverable:** `docs/phase_5a_experimental_protocol.md`. Revision 1 is the initial protocol; Revisions 2–4 address the methodology review findings M01–M08, R3-01–R3-05 and R4-01–R4-06. Revision history is in its Appendix B.
- **Approval:** the project owner approved Revision 4 and all remaining review recommendations on 2026-11-09.
- **Final decisions** (protocol Section 13.0):
  - **C-F:** retrospective sensitivity only, never an automatic fallback or primary evidence.
  - **Near-flat calibrator:** no statistical cutoff. The numerical guard SD(calibrated logits) < 1e-8 is kept, and the calibrator slope and logit dispersion are reported.
  - **Calibration status:** decided independently for LR and HGB.
  - **Champion:** LR by default; HGB only if both E1 and E2 pass.
  - **E2:** needs ≥ 1,000 usable splits of 2,000; otherwise E2 is not estimable and LR remains champion.
  - **Amount-aware policy:** excluded from the primary decision-policy scope; amount-aware cost accounting is kept as a sensitivity.
- **Descriptive evidence used:** read-only train/validation statistics only (protocol Appendix A). No test labels were read.
- **Status:** closed. The protocol is the frozen specification for Phase 5B.

## Phase 5 methodology and design briefing (2026-11-08)

Scope: documentation housekeeping only. The approved Phase 5 design decisions are recorded in the documentation. Not done: Phase 5 code, experiments, training, tuning, calibration, test-set evaluation, or changes to notebooks, data, models, source, tests, configuration or Phase 1–4 results. No commits or pushes. *Resolved:* the documentation review is complete, and the changes are committed in `a84f94a` (originally `4c463f4`).

- **Status:** the briefing is complete and its decisions are approved as **design**. They are not results, and nothing has been implemented.
- **Approved decisions** (full list in `docs/PROJECT_CONTEXT.md`, "Approved Phase 5 design decisions"):
  - Validation: a temporal validation and tuning protocol; a validation and test-set usage policy; Window 1 → Window 2 transfer evaluation.
  - Models: HistGradientBoosting as the challenger model.
  - Calibration: a model calibration strategy that prioritizes a temporal calibration holdout, with ensemble investigation as a fallback if needed. *(Superseded by the approved protocol, decision #1b: uncalibrated scores are the safe alternative; C-F is retrospective sensitivity only, never a fallback.)*
  - Statistical comparison: paired bootstrap evaluation; temporal-block bootstrap for the online queue evaluation when feasible, otherwise point estimates with sensitivity analysis.
  - Operational evaluation: the Analyst Review Queue use case; cost-sensitive evaluation with break-even analysis; review capacity as both fixed-K and time-based capacity; Precision@K and Recall@K; SLA scenarios of 1 h, 2 h and 4 h.
- **Not in the repository:** the briefing's full text and detailed parameters. They will be documented when each item is implemented and reviewed.
- **Documentation updated:** `README.md`, `docs/PROJECT_CONTEXT.md`, this file, `docs/business_understanding.md`, `docs/validation_strategy.md`. The stale "Phase 4 not yet committed" status was also corrected, because Phase 4 is committed in `1f510f7`.

## Phase 4b/4c audits and Phase 4c completion (2026-11-08)

Scope: independent read-only audits of Phase 4b (`4b5b94b`) and Phase 4c (`cffc32c`), then the mentor-approved completion items. Not done: model, split, window or transfer changes; retraining; changes to Phase 4b code or artifacts; changes to Phase 3 documentation; refactoring of optional low-priority findings; commits or pushes. **Status: implemented and mentor-approved (2026-11-08) after a final integrated Phase 4 closure audit (PASS WITH MINOR ISSUES; the minor documentation items were fixed in this session). Ready for commit; not yet committed.** *Resolved:* committed in `1f510f7`.

- **Audit verdicts:** Phase 4b PASS WITH MINOR ISSUES; Phase 4c PASS WITH MINOR ISSUES. All reported metrics were reproduced independently, and both CLIs regenerated byte-identical results files before any change.
- **Mentor decisions recorded:**
  - Window sizes 21,362 / 21,360 are correct. Timestamp groups stay intact (a 21,361 / 21,361 split would split `Time` 142,280, which has 5 rows). The split is unchanged.
  - A descriptive common-reference threshold comparison was approved during the Phase 4c design discussion but had not been implemented. It is added now as a separate experiment; the frozen Window 1 → Window 2 transfer is unchanged.
  - The comparison uses the Phase 4c inclusive rule (`>=`). No validation score equals exactly 0.5.
  - Phase 4b code and `reports/phase4b_baseline_results.json` stay unchanged, including the `decision_rule_note` text ">= 0.5". The correction is recorded in the documentation and the Phase 4 report.
  - Bootstrap results are reported as independent audit computations (seed 0, 2,000 stratified resamples). No script is added to the repository.
- **Files modified:**
  - `src/models/run_threshold_analysis.py`: adds the `common_reference_thresholds` output key and four console lines. It reuses the in-memory validation scores, window labels and `evaluate_threshold`. No extra `fit` or `predict_proba` call.
  - `tests/test_threshold_analysis.py`: 4 new tests (reference thresholds; independent per-window recomputation; window counts summing to full-validation counts; real-data regression). The CLI key-set test and the corrupted-test-rows test now include the new key.
  - `reports/phase4c_threshold_results.json`: regenerated. Only `common_reference_thresholds` was added; every existing key is identical to `HEAD` (text diff: 67 lines added, 0 removed).
  - Documentation: this file, `docs/PROJECT_CONTEXT.md`, `docs/TECHNICAL_DOCUMENTATION.md`.
- **File created:** `docs/reports/phase4_model_development.md`.

### Common-reference comparison (descriptive; verified by running `python -m src.models.run_threshold_analysis`)

Window 1 prevalence 0.108% (23 / 21,362); Window 2 prevalence 0.154% (33 / 21,360). Rule: `y_score >= threshold`.

| Reference threshold | Window | TP | FP | FN | TN | Precision | Recall | F1 | Alerts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 (fixed, diagnostic) | 1 | 20 | 646 | 3 | 20,693 | 0.0300 | 0.8696 | 0.0581 | 666 |
| 0.5 (fixed, diagnostic) | 2 | 32 | 611 | 1 | 20,716 | 0.0498 | 0.9697 | 0.0947 | 643 |
| 0.999995978456591 (full-validation max-F1, retrospective) | 1 | 17 | 0 | 6 | 21,339 | 1.0000 | 0.7391 | 0.8500 | 17 |
| 0.999995978456591 (full-validation max-F1, retrospective) | 2 | 26 | 4 | 7 | 21,323 | 0.8667 | 0.7879 | 0.8254 | 30 |

- This is a descriptive comparison, not a transfer test. The full-validation threshold was selected using the labels of both windows.
- For each threshold, the window counts sum to the full-validation counts. At 0.5 this gives 52 TP / 1,257 FP, the Phase 4b Experiment A confusion matrix. At the max-F1 threshold it gives 43 / 4 / 13 / 42,662, the full-validation optimum.
- The full-validation max-F1 threshold and the Window 1 optimum produce identical Window 1 counts (17 / 0 / 6 / 21,339).

### Verification

- `/Users/duc/Desktop/env/bin/python -B -W always -m unittest discover -s tests -v` → **147 tests, OK** (143 existing + 4 new). None were skipped and no warnings appeared.
- `reports/phase4b_baseline_results.json` and `src/models/run_baselines.py` are unchanged against `HEAD`. The raw-data SHA-256 is unchanged.
- **Raw data:** `data/raw/creditcard.csv` is local and ignored by Git. It has not been modified (SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, checked before and after Phases 4a, 4b and 4c, and again during the 2026-11-08 retrospective audit).

## Phase 3 retrospective audit and documentation (2026-11-08)

Scope: a read-only audit of Phase 3, covering the design commit `fadebd7` and the Phase 4a implementation `37d4521` that realized it, then documentation-only changes approved by the mentor. No source code, tests, notebooks, datasets, `requirements.txt`, Phase 1/2 reports, Phase 4b/4c results or other phase artifacts were changed. No experiment was rerun and the split was not changed. Nothing was committed or pushed. **Status: report drafted, awaiting mentor review.** The Phase 3 audit is recorded as complete only after that approval. *Resolved:* the Phase 3 report and these documentation changes were committed in `ac178c3`.

- **Mentor decisions recorded:**
  - The report covers both `fadebd7` (design and documentation) and `37d4521` (implementation). It states that Phase 3 was design-only in Git.
  - Model-based diagnostics (distribution-shift classifiers, time-of-day model experiments, bootstrap intervals) are not reported in Phase 3. They belong to the Phase 4 retrospective; Phase 3 only cross-references them.
  - The tautological duplicate-isolation check and the model-input duplicate findings are documented. Cross-partition duplicates are not called leakage as such. No split change or rerun.
  - No source code changes. A model-input duplicate checker is recorded as a future improvement needing separate approval.
  - Test fraud counts known during Phase 3 design are disclosed. The boundary rule was label-independent, and no model or threshold was selected on test performance.
  - The test partition's scope (hour values 18–23 of day 2 of one approximately 48-hour dataset, prevalence 0.1217%) is documented and must be acknowledged before final test evaluation.
- **Audit verification (newly performed, not historical):**
  - Raw-data SHA-256 matched the recorded value. Test suite: 143 tests, OK (`-B -W always`).
  - `src/data/split.py` and `tests/test_split.py` are unchanged since `37d4521`; `docs/validation_strategy.md` changed after `fadebd7` only by the Phase 4b implementation note.
  - Independent recomputation reproduced every documented partition count. The documented cumulative-count rule reproduces both boundaries exactly (closest counts to 70% and 85% of 284,807 rows: 199,364 at `Time` 132,928 and 242,086 at `Time` 151,328) without using labels. The assignment is deterministic and independent of row order.
  - The pipeline's scaler is fitted on the 199,364 training rows only; `Time` is not a model input.
- **Findings recorded in the report** (retrospective observations, not original Phase 3 results):
  - Phase 3 (`fadebd7`) contains no code; the boundary derivation and diagnostics were not committed. The work log's Phase 3 verification claims were first written in `c0ecd3c`.
  - The zero-crossing duplicate check is tautological for the chronological split, because the Phase 3 key includes `Time`. At the model-input level (`V1`–`V28` + `Amount`), 1,335 groups cross chronological partitions and 2,391 cross the random split. All are non-fraud with consistent labels. 824 validation and 896 test rows (chronological) have exact model-input twins in train. These are documented as potential leakage of unmeasured effect, not as leakage or future-data contamination.
  - Validation and test cover only hour values 12–23 of day 2. Train's higher fraud rate is mainly a composition effect (train hour values 12–23: 0.136%; validation 0.131%; test 0.122%).
  - Undocumented assumptions: no label delay (the one-second adjacency between partitions is not an embargo), no entity grouping, a single cutoff, and an inherited PCA look-ahead of unknown magnitude.
- **Correction flagged:** the audit findings gave the "same elapsed span one day earlier" comparisons as 39,898 and 50,265 transactions, from rounded hour boundaries. Exact second boundaries give 39,902 and 50,264. Fraud counts (62, 64) and rates (0.155%, 0.127%) are unchanged. The report uses the exact figures.
- **Stale status corrected:** this file gave `HEAD` as `0459921` and the Phase 2 changes as uncommitted, and `docs/PROJECT_CONTEXT.md` described the Phase 2 report as awaiting review. Git history shows the Phase 2 report committed in `9b52a8a` (= `origin/main`). Both files now say so. Historical session text is unchanged; resolved items carry *Resolved* notes.
- **Files created:** `docs/reports/phase3_data_splitting.md`.
- **Files modified:** `docs/WORK_LOG.md`, `docs/PROJECT_CONTEXT.md`.

## Phase 2 retrospective audit and documentation (2026-11-08)

Scope: a read-only audit of Phase 2 (`812c356`), then documentation-only changes approved by the project owner. No source code, tests, notebooks, datasets, `requirements.txt`, `docs/business_understanding.md` or other phase artifacts were changed. Nothing was committed or pushed. *Resolved:* the Phase 2 report and these documentation changes were committed in `9b52a8a`, which is `origin/main`.

- **Owner decisions recorded:**
  - `notebooks/01_data_understanding.ipynb` stays exactly as committed in `812c356`, without outputs. The Phase 2 report records its re-executed outputs instead.
  - `docs/business_understanding.md` is not edited. Its Phase 2 wording, partly rewritten in `fadebd7`, is quoted from `812c356` in the report.
  - The complete time-of-day table is included in the Phase 2 report.
- **Audit verification (newly performed, not historical):**
  - An unchanged copy of the notebook, executed outside the repository with the designated environment, completed without error. It reproduced every documented Phase 2 figure: 284,807 × 31, 492 frauds (0.1727%), majority-class accuracy 99.8273%, 0 missing values, 1,081 full-row duplicates, and no label values outside {0, 1}.
  - The raw-data SHA-256 matched the recorded value.
- **Findings recorded in the report** (retrospective observations, not original Phase 2 results):
  - The notebook was committed without outputs. The only historical execution evidence is the later entry in this log (`c0ecd3c`).
  - Duplicates were checked over full rows only: 1,081 excess rows in 773 groups. Over the model features (`V1`–`V28` + `Amount`) there are 9,144 excess rows in 5,149 groups, all with consistent labels. Duplicates are not treated as leakage in themselves.
  - Fraud rate by elapsed hour modulo 24 (full table in the report): 0.474% in hours 00–06 (28,035 transactions, 133 frauds) against 0.140% in hours 07–23. Validation and test cover only hour values 12–18 and 18–23.
  - `V1`–`V28` are centered and uncorrelated over the full file but not over the training rows. This is consistent with, but does not prove, a PCA fitted on the full 48 hours. `V3` correlates with `Time` (r = −0.42).
  - `Amount` has 1,825 zero values (27 fraud) and skewness 16.98.
  - Amount-weighted recall is proposed only as an optional supplementary metric. It is not implemented.
- **Deferred:** partition-level consequences of duplicates, time-of-day coverage and PCA provenance are left to the Phase 3 and Phase 4 retrospective reports.
- **File created:** `docs/reports/phase2_data_understanding.md`.

## Retrospective audit and Phase 1 documentation remediation (2026-11-08)

Scope: a read-only audit of Phases 1–4c, then documentation-only remediation approved by the project owner. No source code, tests, datasets, model configuration, experiment results or `requirements.txt` were changed. Nothing was committed or pushed.

- **Owner decisions recorded:**
  - The canonical validation/test boundary is `Time = 151328`, as in `src/data/split.py`. A value of `165950` from an earlier project briefing was a recollection error; it appears nowhere in the repository.
  - Phase 4c is formally approved and complete (`cffc32c`).
  - `reports/phase4c_threshold_results.json` (about 9.5 MB) stays in Git. Git history is not rewritten.
  - The remediation plan is executed incrementally, with review between stages.
- **Audit verification (newly performed, not historical):**
  - The full test suite passed: 143 tests, OK.
  - The raw-data SHA-256 matched the value above.
  - An independent scratch script, kept outside the repository, reproduced the Phase 4b Experiment A Logistic Regression values (average precision 0.8397, ROC-AUC 0.9815, 1,309 alerts at the 0.5 rule, 32 iterations).
  - Further audit observations are queued for the Phase 2–4 retrospective reports and are not yet project documentation.
- **Stale status corrected:** `README.md`, `docs/PROJECT_CONTEXT.md`, `docs/TECHNICAL_DOCUMENTATION.md`, `docs/FINAL_REPORT.md` and this file described Phase 4c as uncommitted or awaiting review. This file also gave `HEAD` as `4b5b94b`, and `docs/PROJECT_CONTEXT.md` described Phase 4c as built on top of `4b5b94b`. All five now describe Phase 4c as approved. All except `docs/FINAL_REPORT.md` cite `cffc32c`, and `docs/PROJECT_CONTEXT.md` and this file record the owner's decision on the results file. Historical session text below is unchanged; resolved items carry *Resolved* notes.
- **File created:** `docs/reports/phase1_foundation.md`, a retrospective Phase 1 report built from commit `e4e866d` and later Git history.

### Phase 1 record (reconstructed retrospectively from Git)

Phase 1 had no work-log entry at the time; this log was created later, in `c0ecd3c`. The following is reconstructed only from Git evidence:

- **Commit:** `e4e866d` `Initialize FraudGuard project structure`, 2026-11-06, 17 files and 100 lines added.
- **Files:** `README.md`, `requirements.txt` (six unpinned packages: pandas, numpy, matplotlib, seaborn, jupyter, ipykernel), `.gitignore`, `data/README.md`, empty `src/` subpackages (`data`, `evaluation`, `features`, `models`, `visualization`), and `.gitkeep` placeholders for `app/components`, `artifacts`, `data/raw`, `data/processed`, `notebooks`, `reports/figures` and `tests`.
- **Replaced later:** the Phase 1 README gave the environment as "Python 3.14.6". Commit `812c356` (Phase 2) replaced this with "Development environment verified for this project: Python 3.13.6 (`/Users/duc/Desktop/env/bin/python`)".
- **Not done in Phase 1:** no dataset hash, no pinned dependencies, no environment specification file, no tests, and no phase report. Details and rationale: `docs/reports/phase1_foundation.md`.

## Phase 4c session — threshold analysis and temporal stability (2026-11-08)

Scope: implemented the mentor's Phase 4c Technical Specification. Out of scope and not done: new models, retraining per threshold or window, hyperparameter tuning, feature engineering, resampling, calibration, SHAP, test-set evaluation, threshold optimization on the random split, raw-data edits, commits or pushes.

- **Files created:**
  - `src/evaluation/threshold.py`: `threshold_candidates`, `threshold_curve`, `evaluate_threshold` and `select_max_f1_threshold`.
  - `src/data/temporal_windows.py`: `split_two_time_windows`.
  - `src/models/run_threshold_analysis.py`: the CLI `python -m src.models.run_threshold_analysis`.
  - `reports/phase4c_threshold_results.json`: about 9.5 MB, holding the three full threshold curves.
  - `tests/test_threshold.py` (29 tests) and `tests/test_threshold_analysis.py` (17 tests).
- **Files modified:**
  - `src/models/experiment.py`: extracted `fit_and_score` and `ScoredModel` (fit once on train, score once on validation) from `fit_and_evaluate` so that Phase 4c reuses the Phase 4b fitting code. This is behavior-preserving: the regenerated `reports/phase4b_baseline_results.json` was byte-identical to the committed file, and all Phase 4b tests pass.
  - Documentation: README, PROJECT_CONTEXT, WORK_LOG, TECHNICAL_DOCUMENTATION and FINAL_REPORT (still pending).
- **Not modified:** the Phase 3 boundaries, `src/data/split.py`, `src/data/random_split.py`, the model configuration, the Phase 4b runner's behavior, and the raw data.

### Protocol (as specified)

- **Model:** `logistic_regression_balanced` only (StandardScaler fitted on train; `class_weight="balanced"`, `C=1.0`, `lbfgs`, `max_iter=1000`, `random_state=42`). It is fitted once, on the chronological train partition (199,364 rows / 384 frauds), and converged in 32 iterations. The majority baseline is excluded because its scores are constant.
- **Scores:** `predict_proba` is called once on chronological validation (42,722 rows / 56 frauds). The scores keep the original index and row order and are reused for every analysis below. The test partition is never scored.
- **Threshold candidates:** every unique validation score (41,807 candidates for full validation; 21,050 for Window 1; 20,929 for Window 2). The rule `y_score >= threshold` is inclusive.
- **Selection:** maximize F1, breaking ties by the lowest threshold. F1 is computed as `2TP/(2TP+FP+FN)`, which equals `2PR/(P+R)` when both are defined. Undefined values are `null`.
- **Windows:** the boundary minimizes `|n_W1 - n_W2|` over boundaries between unique validation timestamps, choosing the earlier boundary on ties. Rows sharing a timestamp stay together.

### Results (verified by running `python -m src.models.run_threshold_analysis`)

| Analysis | Rows / frauds | Threshold | Precision | Recall | F1 | TP | FP | FN | TN | Alerts (rate) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full validation, max-F1 (retrospective) | 42,722 / 56 | 0.999995978456591 | 0.9149 | 0.7679 | 0.8350 | 43 | 4 | 13 | 42,662 | 47 (0.110%) |
| Window 1, max-F1 (selection) | 21,362 / 23 | 0.9999978229787393 | 1.0000 | 0.7391 | 0.8500 | 17 | 0 | 6 | 21,339 | 17 (0.080%) |
| **Window 2, frozen Window 1 threshold (transfer)** | 21,360 / 33 | 0.9999978229787393 | 0.8889 | 0.7273 | **0.8000** | 24 | 3 | 9 | 21,324 | 27 (0.126%) |
| Window 2, max-F1 (retrospective diagnostic) | 21,360 / 33 | 0.9999996024880934 | 0.9600 | 0.7273 | 0.8276 | 24 | 1 | 9 | 21,326 | 25 (0.117%) |

- **Windows:** Window 1 is `Time` 132,929–142,280 (prevalence 0.108%); Window 2 is `Time` 142,281–151,328 (prevalence 0.154%). The split is 21,362 / 21,360 rows.
- **Transfer gap:** transferred F1 − retrospective Window 2 F1 = **−0.0276**. Both thresholds catch the same 24 of 33 Window 2 frauds; the frozen threshold adds 2 extra false positives (3 vs 1).
- **Comparison with the Phase 4b 0.5 rule:** at 0.5, chronological validation produced 1,309 alerts. The max-F1 threshold produces 47.

### Interpretation (required by the specification)

1. Full validation contains only 56 frauds, and the windows contain only 23 and 33. A change of one or two cases moves precision, recall and F1 materially. For example, the whole transfer gap above is 2 false positives.
2. The full-validation and Window 2 optima are **retrospective**: they are chosen and measured on the same rows. They are not independently validated or production thresholds.
3. The Window 1 → Window 2 transfer is the more realistic temporal check. Its result (F1 0.80 against a retrospective 0.83) suggests the threshold carried over reasonably *within this ~48-hour dataset*. It does not establish long-term temporal robustness.
4. Breaking ties by the lowest threshold is a methodological convention. On the real data, the selected maxima were unique (no tie was triggered at the selected points).
5. Maximizing F1 is not the same as minimizing financial loss or review cost. No operational threshold has been chosen.
6. **Score saturation (observed):** the selected thresholds lie within about 5×10⁻⁶ of 1.0. The highest validation score is 0.9999999999999918, and no score equals exactly 1.0, so the candidates are still distinct. Class weighting makes the probabilities uncalibrated, so the thresholds are better read as rank cut-offs (about the top 17–47 alerts) than as fraud probabilities. Because the values sit near floating-point resolution, a different numerical environment could shift them slightly. The counts reported above were reproduced exactly in this environment. *Corrected (2026-11-08 audit):* the selected thresholds are 4.0×10⁻⁶, 2.2×10⁻⁶ and 4.0×10⁻⁷ below 1.0. That is close to 1.0 but about ten orders of magnitude above floating-point resolution near 1 (1.1×10⁻¹⁶). The 41,807 unique probabilities match 41,807 unique logits, so saturation merged no scores.

### Verification

- `/Users/duc/Desktop/env/bin/python -W always -m unittest discover -s tests -v` → **143 tests, OK**: the 97 existing tests (Phases 4a and 4b, no regression) plus 29 threshold and window unit tests and 17 integration tests. None were skipped and no warnings appeared.
- **Mutation checks**, run in a scratch copy outside the repository. Each of the following made tests fail:
  - transferring the full-validation threshold or the Window 2 optimum instead of the Window 1 threshold;
  - an exclusive `>` rule, in either the curve or the single-threshold evaluator;
  - breaking ties by the highest threshold;
  - choosing the later window boundary on ties;
  - fitting on train plus validation.
- **Integration checks:** `fit` sees only train rows, once. `predict_proba` sees only validation rows, once, and `predict` is never called. Corrupting the test rows (NaN features, flipped labels) leaves every result unchanged. Flipping Window 2 labels leaves the transferred threshold and all Window 1 results unchanged.
- **Reproducibility:** three CLI runs produced byte-identical JSON. The file parses as strict JSON, rejecting `NaN` and `Infinity`.
- **Phase 4b regression:** `reports/phase4b_baseline_results.json` is unchanged against `HEAD`. The raw-data SHA-256 is unchanged.

### Outstanding issues (Phase 4c)

- **Results file size:** `reports/phase4c_threshold_results.json` is about 9.5 MB because the specification requires the complete curves at full precision. Each curve column is written on one line, which leaves the content identical and cuts the file from 13.4 MB. Whether to track it in Git or keep only a summary in `reports/` is a mentor decision. *Resolved (2026-11-08):* the file is tracked in `cffc32c` and stays in Git, by project-owner decision.
- No confidence intervals or bootstrap estimates are reported. *Resolved (2026-11-08):* independent bootstrap intervals for the Window 1 selection, the transferred Window 2 F1 and the transfer gap are documented in `docs/reports/phase4_model_development.md`, Section 5.1.
- The acceptance criterion "changes are committed and pushed" is not yet met: committing needs the user's go-ahead. *Resolved:* Phase 4c was committed as `cffc32c`, which is the current `origin/main`, and was formally approved on 2026-11-08.

## Phase 4b decision finalization session (2026-11-07)

The mentor approved the three decisions raised in the Phase 4b report. No models, tuning, threshold optimization, Phase 4c work, raw-data edits, commits or pushes were done in this session.

### Approved decisions

1. **Raw `Time` is excluded from the baseline model features.** It is still used for the chronological split and duplicate grouping. Reason: it is elapsed seconds within this 48-hour dataset, and the chronological validation period lies entirely outside the training time range. This does not claim that temporal information has no predictive value; temporal feature engineering remains a possible future direction.
2. **Stratified random split:** 70/15/15, seed 42, stratified by `Class`. Duplicate groups are defined by all columns except `Class` and never cross partitions. All rows are preserved. This split remains a secondary benchmark; chronological evaluation stays primary.
3. **The default 0.5 rule is used for diagnostic reporting only.** Average precision and ROC-AUC are ranking metrics; precision, recall and the confusion matrix at 0.5 are diagnostic metrics. 0.5 is not an optimal or production threshold.

The code already implemented all three decisions, so no model, split or metric code changed. The only code change is the `decision_rule_note` text in `src/models/run_baselines.py`, which now uses the "ranking" and "diagnostic" wording. `reports/phase4b_baseline_results.json` was regenerated: a diff against the previous file showed that only that note changed, and every metric is identical.

### Final verification

I wrote an independent script in the session scratchpad, outside the repository. It recomputes values directly from the raw CSV where possible.

1. **Data hash:** SHA-256 of `data/raw/creditcard.csv` (the path in `CREDITCARD.path` and in the results JSON) is `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. It matches the JSON and the value recorded since Phase 4a. The file has 284,807 rows, 492 frauds and 31 columns.
2. **Class proportions in the random split** (overall fraud rate 0.1727%):

   | Partition | Rows (share) | Frauds (share of all frauds) | Fraud rate |
   |---|---:|---:|---:|
   | Train | 199,364 (70.000%) | 344 (69.92%) | 0.1725% |
   | Validation | 42,722 (15.000%) | 74 (15.04%) | 0.1732% |
   | Test | 42,721 (15.000%) | 74 (15.04%) | 0.1732% |

3. **Duplicate isolation:** I computed it independently by hashing all non-`Class` columns. Both splits contain 773 repeated groups and 1,081 excess rows, all 284,807 rows are retained, and 0 groups cross partitions.
4. **Average precision uses probabilities:** I recomputed it from the captured `predict_proba` outputs, and it matches the reported values exactly (A: 0.839660, B: 0.752375). Logistic Regression produced 41,807 and 42,104 distinct scores. Average precision computed from the 0.5 labels instead would be 0.0370 and 0.0572, so the reported values are not label-based. The majority baseline has one constant score, so its average precision equals the prevalence.
5. **Test isolation:** I wrapped every pipeline's `fit`, `predict_proba` and `predict`. In both experiments and for both models, `fit` received only the 199,364 training rows and prediction received only the 42,722 validation rows; 0 test rows were touched. No model selection takes place. The two baselines are fixed and are reported side by side.
6. **Reproducibility:** two fresh CLI runs were byte-identical to each other and to the recorded results file, before the note text changed.

Tests: `/Users/duc/Desktop/env/bin/python -W always -m unittest discover -s tests -v` → 97 tests, OK, with no warnings and none skipped.

## Phase 4b session — baseline modeling (2026-11-07)

Scope: two fixed baselines, evaluated on the validation partition of two separate experiments. No threshold selection, hyperparameter tuning, test-set scoring, feature engineering, Phase 4c work, raw-data edits, commits or pushes.

- **Files created:**
  - `src/config.py`: the `DatasetConfig` dataclass (target, positive label, feature selection or exclusion, optional time column, optional duplicate-group columns) and the `CREDITCARD` instance.
  - `src/data/random_split.py`: `assign_stratified_group_partitions`, the secondary benchmark split (stratified by label and keeping duplicate groups together, seeded). This split was specified in Phase 3 but had not been implemented; it returns the same partition labels and dtype as `src/data/split.py`.
  - `src/features/preprocessing.py`: `build_preprocessor` (a `StandardScaler` on the configured features inside a `ColumnTransformer` that drops all other columns) and `validate_feature_frame`.
  - `src/models/baselines.py`: `build_majority_baseline` (`DummyClassifier(strategy="prior")`) and `build_logistic_regression` (`LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000)`). Each returns an unfitted preprocess→model `Pipeline`.
  - `src/evaluation/metrics.py`: `compute_binary_metrics` (average precision, ROC-AUC, precision, recall, confusion matrix; undefined values are returned as NaN).
  - `src/models/experiment.py`: `fit_and_evaluate`, which fits on `train` rows only and evaluates on `validation` (passing `evaluation_partition="test"` raises an error); `summarize_partitions`.
  - `src/models/run_baselines.py`: the CLI `python -m src.models.run_baselines`, which runs Experiments A and B and writes `reports/phase4b_baseline_results.json`.
  - `reports/phase4b_baseline_results.json`: the recorded results, including configuration, data SHA-256 and library versions.
  - `tests/test_random_split.py` (23 tests) and `tests/test_baselines.py` (38 tests).
  - `docs/TECHNICAL_DOCUMENTATION.md` and `docs/FINAL_REPORT.md` (pending placeholder).
- **Files modified:** `docs/WORK_LOG.md`, `docs/PROJECT_CONTEXT.md`, `README.md`, and `docs/validation_strategy.md` (an appended "Implementation note" for the secondary split; the Phase 3 text is unchanged). `src/data/split.py` and `tests/test_split.py` were **not** modified.

### Experiment configuration

- **Model features:** `V1`–`V28` and `Amount` (29 columns). `Time` is **excluded from the model features** but still drives the chronological split and duplicate grouping. Reason: `Time` counts seconds since the first transaction in this 48-hour extract, so in Experiment A every validation value lies outside the training range. This was proposed as a default needing approval. *Resolved:* approved in the decision finalization session above.
- **Preprocessing:** a `StandardScaler` on all 29 features, fitted inside the pipeline on the training partition only. There is no imputation, resampling or feature engineering.
- **Experiment A (primary):** the chronological split from `src/data/split.py`, unchanged (Phase 3 boundaries).
- **Experiment B (secondary):** the stratified random split with duplicate groups kept together. Fractions are 70/15/15 within each class, `random_state=42`, and groups are defined by all columns except `Class`.
- **Evaluation:** validation partition only. The test partition was never passed to any model. Partition label counts for test are reported for descriptive purposes only.
- **Decision rule for precision, recall and the confusion matrix:** each estimator's default `predict()`. For Logistic Regression this is probability ≥ 0.5; for the majority baseline it means always predicting non-fraud. **This is not a selected or optimized operating threshold.** *Corrected (2026-11-08 audit):* scikit-learn's binary `predict` flags a row only when the probability is strictly greater than 0.5. No validation score equals exactly 0.5, so the reported counts are unaffected. *Approved wording, from the finalization session:* these are diagnostic metrics, and 0.5 is not an optimal or production threshold.
- **Environment:** Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1 (`/Users/duc/Desktop/env/bin/python`).

### Split verification

| Experiment | Train rows / frauds | Validation rows / frauds | Test rows / frauds | Duplicate groups crossing partitions |
|---|---:|---:|---:|---:|
| A: chronological | 199,364 / 384 | 42,722 / 56 | 42,721 / 52 | 0 |
| B: stratified random (seed 42) | 199,364 / 344 | 42,722 / 74 | 42,721 / 74 | 0 |

The B partition sizes coincide with A's because both target 70/15/15 of 284,807 rows. A's sizes come from the Phase 3 timestamp boundaries; B's are computed per class (199,020 + 344 train rows).

### Validation results (verified by running `python -m src.models.run_baselines`)

The two experiments are reported separately and must not be pooled.

**Experiment A: chronological (primary)**, validation: 42,722 rows, 56 frauds, prevalence 0.131%.

| Model | Average precision | ROC-AUC | Precision* | Recall* | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.0013 | 0.5000 | undefined | 0.0000 | 42,666 | 0 | 56 | 0 |
| Logistic Regression (balanced) | 0.8397 | 0.9815 | 0.0397 | 0.9286 | 41,409 | 1,257 | 4 | 52 |

**Experiment B: stratified random (secondary benchmark)**, validation: 42,722 rows, 74 frauds, prevalence 0.173%.

| Model | Average precision | ROC-AUC | Precision* | Recall* | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.0017 | 0.5000 | undefined | 0.0000 | 42,648 | 0 | 74 | 0 |
| Logistic Regression (balanced) | 0.7524 | 0.9725 | 0.0649 | 0.8784 | 41,711 | 937 | 9 | 65 |

Average precision and ROC-AUC are ranking metrics. \* These columns are diagnostic metrics at the default 0.5 rule (Logistic Regression probability ≥ 0.5; *corrected 2026-11-08:* strictly > 0.5, with identical counts because no score equals 0.5); 0.5 is not an optimal or production threshold. Precision for the majority baseline is undefined because it makes no positive predictions.

- Logistic Regression converged in both experiments (32 and 22 lbfgs iterations; no `ConvergenceWarning`).
- As expected, the majority baseline's average precision equals the validation prevalence and its ROC-AUC is 0.5.
- With class weighting, the 0.5 rule flags 1,309 (A) and 1,002 (B) validation transactions, about 2–3% of rows. That gives high recall and low precision. Choosing an operating point is deferred to later work with agreed operational criteria.
- Experiment A's average precision is higher than B's. The validation sets have only 56 and 74 frauds and cover different transactions, so this gap is within the range that a small number of cases can move. It should not be read as evidence that chronological generalization is easier.

### Tests executed

- `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v` → **97 tests, OK** (36 existing from Phase 4a, 23 new random-split tests, 38 new baseline tests). No tests were skipped because the raw CSV is present, and `-W always` showed no warnings.
- Coverage: majority-baseline behavior; Logistic Regression configuration, training, probability outputs and reproducibility; the scaler fitted on training rows only; models unchanged when validation/test rows are altered; NaN-corrupted test rows never touched; refusal to evaluate on test; `Time` excluded from model inputs; hand-computed metric values and NaN for undefined metrics; compatibility with both split outputs; deterministic results; a CLI run end to end on synthetic data; and regression checks on the real data against the tables above.
- Mutation check (in a scratch copy, outside the repository): fitting on train+validation made 5 tests fail, including both leakage tests.
- Reproducibility: two CLI runs produced byte-identical JSON output. `git diff --check` passed.

### Outstanding issues (Phase 4b)

- Mentor approval was needed for: excluding `Time` from the model features; the seed and procedure of the random split; and reporting precision, recall and the confusion matrix at the default 0.5 rule. *Resolved:* all three were approved (see the decision finalization session).
- No confidence intervals are reported. With 56 and 74 validation frauds, the metrics have substantial sampling uncertainty. *Resolved (2026-11-08):* independent bootstrap intervals for the Logistic Regression average precision (A: [0.746, 0.922]; B: [0.644, 0.857]) are documented in `docs/reports/phase4_model_development.md`, Section 3.2.
- The chronological boundaries in `src/data/split.py` remain specific to this dataset. A new dataset would need its own documented boundaries, although the runner accepts any partition Series.

## Phase 4a session — chronological partitioning module (2026-11-07)

Scope was authorized for Phase 4a only. No model training, feature engineering, threshold selection, test-set evaluation, random benchmark split, raw-data edits, commits, or pushes.

- **Files created:**
  - `src/data/split.py` — `assign_partitions(df)` returns an ordered categorical `train`/`validation`/`test` Series aligned to `df.index`, using only `Time` and the frozen boundaries (`TRAIN_END_TIME = 132928`, `VALIDATION_END_TIME = 151328`, both inclusive). `split_partitions(df)` returns the three frames via boolean `.loc` selection, keeping original index labels, values and row order. `count_cross_partition_duplicate_groups(df, partitions)` counts identical-feature groups (all columns except `Class`) that span partitions, as a leakage check.
  - `tests/test_split.py` — 36 `unittest` tests: boundary inclusivity, documented constants, equal-timestamp and duplicate-record grouping, the leakage detector, preservation of records/index/order, no input mutation, determinism, row-order and label independence, invalid inputs, and real-data checks of the documented row, fraud, time-range and duplicate-group figures. Real-data tests are skipped if the raw CSV is absent.
- **Files modified:** `requirements.txt` (added `scikit-learn`), `docs/PROJECT_CONTEXT.md` and `docs/WORK_LOG.md` (status and handoff references only; earlier decisions and history unchanged).
- **Input validation:** rejects non-DataFrame input, a missing or repeated `Time` column, non-numeric or boolean `Time`, missing/infinite values, negative values, and non-integer values. Non-integer times are rejected because the documented inclusive integer ranges leave values such as `132928.5` undefined; the raw data contains only whole seconds. Values above `172792` are assigned to test.
- **Verification:** `/Users/duc/Desktop/env/bin/python -m unittest discover -s tests -v` → 36 tests, OK. Real-data results matched the documentation exactly: train 199,364 rows / 384 frauds, validation 42,722 / 56, test 42,721 / 52; 284,807 rows retained; 773 duplicate groups and 1,081 excess rows retained; zero duplicate groups or timestamps crossing partitions. As a sanity check, moving `TRAIN_END_TIME` by one second in memory made 4 real-data tests fail.
- **Methodology:** unchanged. Boundaries, duplicate retention, and the undecided threshold are as approved in Phase 3.

## Tasks completed in the Phase 3 finalization session

The latest Phase 3 commit finalized the validation design and aligned the project status text.

- **Committed files changed:**
  - `README.md` — updated roadmap and current status through Phase 3.
  - `docs/business_understanding.md` — aligned the evaluation plan with the finalized validation strategy and undecided threshold.
  - `docs/validation_strategy.md` — created the validation strategy, split diagnostics, leakage mitigation, limitations, and Phase 3 completion summary.
- **Files deleted:** none.
- **Commit:** `fadebd7 Complete Phase 3: Data Preparation and Validation`.
- **Decisions:** chronological 70/15/15 is primary; boundaries are train `Time 0–132928`, validation `132929–151328`, test `151329–172792`; equal timestamps stay together; all duplicate rows are retained; identical feature groups must not cross partitions; a duplicate-group-aware stratified random split is secondary; no classification threshold is selected.
- **Implementation scope:** documentation and validation diagnostics only. No data preparation, feature engineering, split implementation, model training, raw-data edits, commits, or pushes were performed in that finalization session.

## Verified diagnostics and execution

- **Environment used for the split checks:** `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1 as reported in the prior inspection).
- **Dataset counts:** 284,807 rows, 492 fraud rows. `Time` is nondecreasing and spans 0–172,792 seconds, approximately 48 hours.
- **Frozen chronological partitions:**

  | Partition | Time range | Rows | Non-fraud | Fraud | Fraud rate |
  |---|---:|---:|---:|---:|---:|
  | Train | 0–132,928 | 199,364 | 198,980 | 384 | 0.1926% |
  | Validation | 132,929–151,328 | 42,722 | 42,666 | 56 | 0.1311% |
  | Test | 151,329–172,792 | 42,721 | 42,669 | 52 | 0.1217% |

- Equal timestamps are kept in one partition. Grouping the 30 feature columns excluding `Class` identified 773 repeated feature groups and 1,081 excess rows; labels were consistent and no repeated feature group crossed the finalized partitions.
- The specified Python environment reran these partition/count and duplicate-group checks during finalization; assertions passed.
- **Notebook execution evidence from Phase 2:** all code cells in `notebooks/01_data_understanding.ipynb` were executed in order under Python 3.13.6 with pandas 2.3.1 and NumPy 2.3.2; they completed successfully. No models were trained.
- **Other checks:** `git diff --check` passed for the Phase 3 finalization changes. No conventional automated test suite was run or requested.

## Known issues, limitations, and unresolved questions

### Verified limitations

- Validation and test contain only 56 and 52 fraud cases, so metrics can be sensitive to a small number of observations.
- The dataset covers only about 48 hours in 2013; chronological evaluation cannot establish robustness across longer time periods or current production traffic.
- `V1`–`V28` are anonymized PCA components; individual original business meanings are unavailable from this dataset documentation.
- The dataset documentation does not establish label availability timing or the production prediction point.
- The chronological split is implemented as reusable code in `src/data/split.py` (Phase 4a). The secondary random benchmark split was not implemented in Phase 4a; *superseded:* it was implemented in Phase 4b (`src/data/random_split.py`).
- No feature engineering has been performed. *Superseded in Phase 4b:* two baseline models were trained (Phase 4b session above). The classification threshold remains intentionally undecided.

### Unresolved for later project work

- Confirm the intended prediction point and which fields would be available then.
- Decide operational costs, review capacity, and evaluation criteria before any threshold selection. *Phase 5 design (2026-11-08):* the evaluation approach is approved (cost-sensitive evaluation with break-even analysis, fixed-K and time-based capacity, 1/2/4-hour SLA scenarios). No cost values, capacity values or operational threshold have been set.
- Choose models and preprocessing only after the split is implemented with leakage controls and the Phase 3 boundaries remain fixed.

No blocking implementation errors are known from the completed Phase 3 validation checks. An earlier notebook cell-execution attempt lacked Jupyter's `display` binding; the notebook was corrected, and the full code-cell execution subsequently passed. The default Python environment also lacked pandas; use the designated environment listed under "Verified diagnostics and execution" above.

## Exact files created for this handoff

- `docs/WORK_LOG.md` — this chronological project handoff log.
- `docs/PROJECT_CONTEXT.md` — concise verified project context and continuation notes.

No files were deleted. No raw data or notebook was changed during handoff preparation. These two files were later committed in `c0ecd3c`; they are the complete handoff documentation, and no separate protocol file exists.

## Next recommended task

*Historical (end of Phase 4a):* begin Phase 4b after approval. That work has now been done (see the Phase 4b session).

*Historical (end of Phase 4b):* mentor approval of Phase 4b, then Phase 4c. Phase 4b is committed in `4b5b94b`, and Phase 4c has been implemented.

*Historical (end of Phase 4c):* mentor review of Phase 4c, including the results-file size question. Both were resolved on 2026-11-08: Phase 4c is approved (`cffc32c`) and the results file stays in Git.

*Historical (after the Phase 1 remediation):* review of the Phase 1 documentation remediation, then the Phase 2 lesson and remediation. The Phase 1 remediation was committed in `0459921`, and the Phase 2 audit and report were completed (see the 2026-11-08 Phase 2 session at the top).

*Historical (after the Phase 2 report):* mentor review of the Phase 2 retrospective report, then the Phase 3 audit. The Phase 2 report was committed in `9b52a8a`, and the Phase 3 audit and report were completed as a draft (see the Phase 3 session at the top).

*Historical (after the Phase 3 report):* mentor review of the Phase 3 retrospective report, then the Phase 4 retrospective audit. The Phase 3 report was committed in `ac178c3`; the Phase 4b and 4c audits were completed (see the session at the top).

*Historical (after the Phase 4 completion work):* mentor review of the Phase 4c completion work and the Phase 4 retrospective report. Both were approved on 2026-11-08 after the final Phase 4 closure audit.

*Historical (after the Phase 4 closure audit):* commit the approved Phase 4 changes. Done in `1f510f7`. The Phase 5 methodology and design briefing has since been completed (session at the top).

*Historical (after the Phase 5 briefing):* review of the Phase 5 design documentation. The review is complete, and the documentation is committed in `a84f94a` (originally `4c463f4`).

*Historical (after the Phase 5 status sync):* Phase 5 experimental protocol preparation. Done: Phase 5A approved and closed on 2026-11-09 (session at the top).

*Historical (after the Phase 5A closure):* Phase 5B, implementation of `docs/phase_5a_experimental_protocol.md`. Its preparation stage, Phase 5B-0, was implemented on 2026-11-11 (session at the top).

*Historical (after Phase 5B-0):* mentor review of Phase 5B-0, then Phase 5B-1. Done: Phase 5B-0 committed (`7c4edcd`, `aa67802`, `726a354`); Phase 5B-1 implemented 2026-11-11 (session at the top).

*Historical (after Phase 5B-1):* mentor review of Phase 5B-1, then protocol Section 4.2 step 2. Phase 5B-1 is committed (`aa4e63a`). On the owner's instruction, Phase 5B-2 then ran the controlled experiments E0–E3 (session at the top) before the grid stage.

**Current:** mentor review of Phase 5B-2 part II (the grid, step 2, and the recorded step-3 candidates). Then, only on approval, step 4: sigmoid calibration of LR-10 and HGB-14 on their `T_cal` scores. *Superseded:* "mentor review of Phase 5B-2, including the logged `log1p(Amount)` deviation; then step 2". Done: part I approved with a scope clarification, and step 2 implemented (session at the top). *Earlier text of this entry:* mentor review of Phase 5B-0, then Phase 5B-1 (the first implementation and experiment stage). Phase 5B-1 starts only on explicit approval and proceeds with review between stages. Before any final test evaluation, acknowledge the test-partition scope recorded in the Phase 3 report (Section 7). Do not write the final report until the project is complete. The test partition stays untouched until final evaluation. The Phase 4c thresholds are analytical; an operational threshold still requires agreed costs and review capacity.
