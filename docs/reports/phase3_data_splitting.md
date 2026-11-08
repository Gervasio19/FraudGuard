# Phase 3 Report — Data Splitting and Validation Design (Retrospective)

> **Report type:** retrospective. This report was written on 2026-11-08, after Phases 1–4c were complete. No Phase 3 report was written at the time. It covers two commits: the Phase 3 **design** (`fadebd7`) and the Phase 4a **implementation** that realized it (`37d4521`). Historical claims are drawn from those commits and later Git history. Every number marked **[V]** was computed during the 2026-11-08 retrospective audit and was **not** known or recorded during Phase 3 unless the text says so. Explanations of *why* something matters are retrospective interpretation, not the original author's recorded reasoning.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[I]** | Implemented in the repository (files or code exist; commit cited) |
| **[D]** | Documented in the repository (stated in a committed document) |
| **[V]** | Newly verified or computed during the 2026-11-08 retrospective audit |
| **[R]** | Retrospective explanation, interpretation or recommendation |

**Phase 3 at a glance**

| Item | Value |
|---|---|
| Design commit | `fadebd780320b9c58263f10c33ba9ffaa2a06315`, "Complete Phase 3: Data Preparation and Validation", 2026-11-07 17:21:37 +0700 |
| Design files | `docs/validation_strategy.md` (new, 42 lines); `README.md` and `docs/business_understanding.md` (edited) |
| Code, scripts, tests in the design commit | None **[I]** |
| Implementation commit | `37d4521`, "Complete Phase 4a chronological data splitting", 2026-11-07 17:38:28 +0700: `src/data/split.py` (136 lines), `tests/test_split.py` (36 tests) |
| Later changes to `src/data/split.py` and `tests/test_split.py` | None **[V]** |
| Later changes to `docs/validation_strategy.md` | One appended "Implementation note" in `4b5b94b` (Phase 4b); the Phase 3 text is unchanged **[V]** |
| Dataset fingerprint (SHA-256) at audit | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, matching the value recorded since Phase 4a **[V]** |
| Test suite at audit | 143 tests, OK (`tests.test_split`: 36 tests, OK) **[V]** |

## 1. Phase objectives and historical scope

The Phase 3 commit renamed roadmap step 3 in `README.md` from "Data preparation and feature engineering" to "Data preparation and validation design (complete)" **[D]**. Its status line read: "Business framing, basic dataset inspection, and validation design are documented. No data preparation, feature engineering, or model training has been performed." **[D]**

The decisions recorded in `docs/validation_strategy.md` (`fadebd7`) are **[D]**:

1. **Primary design:** a chronological 70/15/15 train/validation/test holdout on `Time`, with all records sharing a timestamp kept in one partition.
2. **Boundaries:** train `Time <= 132928`; validation `132929 <= Time <= 151328`; test `Time >= 151329`.
3. **Duplicates:** retain all records. "Exact feature groups (the 30 columns excluding `Class`)" must not cross partitions.
4. **Secondary benchmark:** a stratified random split that keeps duplicate groups together, reported separately.
5. **Leakage controls for later phases:** fit preprocessing on training data only; do not tune on validation or test outcomes; confirm the prediction point and label availability before modeling.
6. **Threshold:** none selected.

**What Phase 3 was, in Git terms [I][R]:** Phase 3 was **design-only**. The commit contains prose and a table of counts, but no code that assigns partitions, derives the boundaries or reproduces the counts. The executable split arrived 17 minutes later in Phase 4a (`37d4521`), which hard-codes the Phase 3 boundaries. This report judges the design against its stated goals and audits the Phase 4a code as its realization. Phase 4b and 4c work that consumed the split is cross-referenced (Section 11), not analysed.

**Note on boundary values [D]:** an earlier project briefing quoted a validation/test boundary of `165950`. The owner confirmed on 2026-11-08 that this was a recollection error. The canonical value is `151328`, as in `src/data/split.py`; `165950` appears nowhere in the repository (`docs/WORK_LOG.md`, Phase 1 remediation session).

## 2. Timeline

| When (commit time, +0700) | Commit | What happened | Label |
|---|---|---|---|
| 2026-11-07 17:17:42 | `812c356` | Phase 2: notebook and business understanding. No split. | [I] |
| 2026-11-07 17:21:37 | `fadebd7` | **Phase 3 design:** `docs/validation_strategy.md` created with boundaries, counts and duplicate statement; README and `business_understanding.md` aligned. No code. | [I][D] |
| 2026-11-07 17:28:37 | `c0ecd3c` | `docs/WORK_LOG.md` and `docs/PROJECT_CONTEXT.md` created. The work log's Phase 3 entries ("Verified diagnostics and execution") were first written here, **after** the Phase 3 commit. They state that the partition and duplicate checks were rerun and "assertions passed". | [D] |
| 2026-11-07 17:38:28 | `37d4521` | **Phase 4a implementation:** `src/data/split.py` (`assign_partitions`, `split_partitions`, `count_cross_partition_duplicate_groups`) and `tests/test_split.py`. `scikit-learn` added to `requirements.txt`. | [I] |
| 2026-11-08 00:00:02 | `4b5b94b` | Phase 4b: secondary random split implemented (`src/data/random_split.py`); `Time` excluded from model features; implementation note appended to `docs/validation_strategy.md`. | [I][D] |
| 2026-11-08 (audit) | — | Retrospective audit; this report. | [V] |

Commit times record when work was committed, not when it was done. They do not show how the Phase 3 counts were produced **[R]**.

## 3. Split methodology, boundary derivation and reproducibility

### 3.1 How the split works

`assign_partitions` (`src/data/split.py`) reads only `Time` and assigns code 0 (train) when `Time <= 132928`, 1 (validation) when `Time <= 151328`, and 2 (test) otherwise. The result is an ordered categorical Series aligned to the input index **[I]**. It never reads `Class`. It rejects non-integer, negative, missing and non-numeric times **[I]**. Because assignment is a pure function of `Time`, rows with equal timestamps always share a partition **[I]**.

### 3.2 Boundary derivation

Phase 3 described the rule only in prose: boundaries "were selected by cumulative row count, choosing complete timestamps closest to each target" **[D]**. No script was committed. The audit reconstructed the rule from the raw file **[V]**:

- Sort unique `Time` values and accumulate their row counts. The dataset has 284,807 rows.
- **Train end, 70% target = 199,364.90 rows.** The closest cumulative count is **199,364**, reached at `Time` **132,928** (difference −0.90). Neighbouring timestamps give 199,362 (`Time` 132,927) and 199,368 (`Time` 132,929).
- **Validation end, 85% target = 242,085.95 rows.** The closest cumulative count is **242,086**, reached at `Time` **151,328** (difference +0.05). Neighbours give 242,083 (`Time` 151,327) and 242,087 (`Time` 151,329).
- The prose could also be read as "validation *size* closest to 15% of rows" (42,721.05). That reading selects the same boundary, `Time` 151,328, so the ambiguity has no effect here.

The documented rule therefore **reproduces both boundaries exactly** **[V]**. It uses only row counts per timestamp, never labels.

### 3.3 Verified partitions

Computed independently of `src/` and cross-checked against `assign_partitions` **[V]**:

| Partition | `Time` range (s, inclusive) | Rows (share) | Non-fraud | Fraud | Fraud rate | Unique timestamps | File rows (0-based) |
|---|---|---:|---:|---:|---:|---:|---|
| Train | 0–132,928 | 199,364 (70.000%) | 198,980 | 384 | 0.1926% | 89,844 | 0–199,363 |
| Validation | 132,929–151,328 | 42,722 (15.000%) | 42,666 | 56 | 0.1311% | 16,495 | 199,364–242,085 |
| Test | 151,329–172,792 | 42,721 (15.000%) | 42,669 | 52 | 0.1217% | 18,253 | 242,086–284,806 |

These match the counts documented in `fadebd7` **[D]**.

**Disjointness and ordering [V]:**
- `Time` takes whole-second values only and is non-decreasing in file order. Each partition is therefore one contiguous block of rows.
- 0 timestamps are shared between partitions; every row is in exactly one partition.
- Rows at the boundary seconds: `Time` 132,928 → 2 rows (train); 132,929 → 4 rows (validation); 151,328 → 3 rows (validation); 151,329 → 1 row (test).
- Consecutive partitions are separated by one second of `Time` (132,928 → 132,929 and 151,328 → 151,329). This is just the next timestamp. It is **not** a gap or embargo between training and evaluation (Section 8).

**Reproducibility [V]:**
- The independent assignment equals the repository assignment row for row.
- The SHA-256 of the assignment codes is identical across repeated calls and after shuffling the input rows (`0a48ae06…`).
- `tests/test_split.py` (36 tests, OK) checks boundary inclusivity, timestamp grouping, label independence, order independence, and the documented real-data counts **[I][V]**. The Phase 4a work log records that shifting `TRAIN_END_TIME` by one second made 4 real-data tests fail **[D]**.

### 3.4 Disclosure: test labels were known during design

The Phase 3 table reports fraud counts and rates for every partition, including **test (52 frauds, 0.1217%)** **[D]**. The test labels were therefore inspected while the design was being finalized. This report records that transparently.

What limits the consequence:

- The boundary rule uses row counts per timestamp only, and reproduces both boundaries without reading `Class` **[V]** (Section 3.2).
- No model, feature set, hyperparameter or threshold was selected using test performance. No model has been scored on test; the Phase 4 runner refuses to evaluate on the test partition **[I][D]**.

**[R]:** seeing descriptive test prevalence is a mild exposure, not a selection step. Best practice is to report test-label statistics only at final evaluation, or to state explicitly that they were seen. This report does the latter.

## 4. Partition time coverage and fraud prevalence

**What "hour value" means here [R]:** `Time` counts seconds from the first transaction in the file. The clock time of that first transaction is not documented. This report therefore uses the **elapsed hour**, `floor(Time / 3600)`, and the **hour value**, elapsed hour modulo 24. Hour values are **not** clock hours. The same convention is used in the Phase 2 report (Section 6.2).

| Partition | Elapsed hours (exact) | Elapsed day(s) | Hour values present |
|---|---|---|---|
| Train | 0.000–36.924 | Day 1 (all) and day 2 (part) | 00–23 |
| Validation | 36.925–42.036 | Day 2 | 12–18 |
| Test | 42.036–47.998 | Day 2 | 18–23 |

**[V]**. Two hour values are split between neighbouring partitions (Phase 2 report, Section 6.2):

- **Hour value 12:** 7,133 rows (7 frauds) in train, 555 (1) in validation.
- **Hour value 18:** 270 rows (0 frauds) in validation, 8,162 (18) in test.

**Train composition [V]:** day 1 contributes 144,786 rows (281 frauds) and day 2 contributes 54,578 rows (103 frauds).

**Fraud rate by hour band [V]:**

| Rows | Transactions | Frauds | Fraud rate |
|---|---:|---:|---:|
| Train, hour values 00–06 | 28,035 | 133 | 0.474% |
| Train, hour values 07–11 | 66,811 | 109 | 0.163% |
| Train, hour values 12–23 | 104,518 | 142 | 0.136% |
| Validation (hour values 12–18) | 42,722 | 56 | 0.131% |
| Test (hour values 18–23) | 42,721 | 52 | 0.122% |
| Same elapsed span as validation, one day earlier (`Time` 46,529–64,928) | 39,902 | 62 | 0.155% |
| Same elapsed span as test, one day earlier (`Time` 64,929–86,392) | 50,264 | 64 | 0.127% |

*Correction note:* the audit findings presented on 2026-11-08 gave the last two rows as 39,898 and 50,265 transactions, from hour boundaries rounded to three decimals. The figures above use exact second boundaries. Fraud counts and rates are unchanged.

**Fraud is concentrated in time [V]:** 17 of the 56 validation frauds fall in elapsed hour 41 (hour value 17). 18 of the 52 test frauds fall in elapsed hour 42 (hour value 18), all after the validation/test boundary.

**How to read this [R]:**

- **Train's higher overall fraud rate (0.193%) is mainly a composition effect.** Train contains the hour values 00–11, which hold 242 of its 384 frauds (63.0%) **[V]**. Restricted to the same hour values, train's rate (0.136%) is close to validation's (0.131%) and test's (0.122%). The same elapsed spans one day earlier show similar rates (0.155% and 0.127%). The lower validation and test prevalence is therefore **not evidence that fraud declined over time**.
- **Prevalence matters for metrics.** Average precision, precision and alert volume all depend on prevalence. A model evaluated at 0.13% prevalence will show lower precision than the same model at 0.47%, even with the same ranking ability.
- **Small, bursty counts.** With about a third of each evaluation set's frauds in a single hour, one burst of related fraud, possibly one card or campaign, can dominate the metrics. Without card identifiers this cannot be checked.

## 5. Duplicate definitions and findings

### 5.1 Three duplicate keys

| Key | Columns | Excess rows | Repeated groups | Rows in groups | Fraud rows in groups | Groups with conflicting labels | Groups crossing chronological partitions | Groups crossing the random split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full row (Phase 2) | 31 | 1,081 | 773 | 1,854 | 32 | n/a (key includes `Class`) | 0 | 0 |
| **Phase 3 key:** all columns except `Class` | 30 (includes `Time`) | 1,081 | 773 | 1,854 | 32 | 0 | **0** | 0 |
| **Model input (Phase 4b):** `V1`–`V28` + `Amount` | 29 | **9,144** | **5,149** | 14,293 | 32 | **0** | **1,335** | **2,391** |

**[V]**. Grouping on `V1`–`V28` alone gives exactly the same groups as `V1`–`V28` + `Amount`: whenever the components match, the amount matches too **[V]**. The Phase 3 counts (773 groups, 1,081 excess rows, consistent labels, zero crossing) match the design document **[D][V]**.

### 5.2 Why the Phase 3 zero-crossing result is tautological

The Phase 3 key includes `Time`. The chronological split is a function of `Time` alone. Every member of a Phase 3 duplicate group therefore has the same `Time` and, necessarily, the same partition. **Zero crossing groups is guaranteed by construction. The check could not have returned any other value.** **[R]**

The design text half-recognized this: "assigning by `Time` already keeps the observed duplicate groups together because each repeated feature group has the same `Time`" **[D]**. The implementation's docstring says so directly: "duplicate feature vectors (which include `Time`) always share a partition too" **[I]**. Yet the zero result is still presented as a verified safeguard in `docs/PROJECT_CONTEXT.md`, `docs/WORK_LOG.md` and `docs/TECHNICAL_DOCUMENTATION.md` ("It should be 0") **[D]**.

**Why it matters [R]:** a diagnostic that cannot fail gives false assurance. The question that matters for evaluation is whether the **model** sees the same input in training and evaluation. Since Phase 4b, the model's inputs exclude `Time`, so the relevant key is the 29-column model-input key. That key was never checked at the partition level during Phases 3–4c **[V]**.

### 5.3 Model-input duplicates across partitions

| Measure | Value |
|---|---|
| Model-input groups crossing chronological partitions | 1,335 |
| Rows in those groups | 4,171 (train 1,334; validation 1,406; test 1,431) |
| Fraud rows in those groups | **0** |
| Partition combinations | validation + test 425; train + test 405; train + validation 318; all three 187 |
| Time between the first and last member of a crossing group | 664 s minimum; 28,282 s (about 7.9 h) median; 85,833 s (about 23.8 h) maximum |
| Validation rows with an exact model-input twin in train | 824 (1.93%), 0 frauds |
| Test rows with an exact model-input twin in train | 896 (2.10%), 0 frauds |
| Median `Amount`, rows in model-input duplicate groups vs other rows | 1.98 vs 24.85 |

All values **[V]**.

**Fraud duplicates [V]:** all 32 fraud rows in duplicate groups belong to 13 groups whose members share the **same second**. No fraud group crosses a partition: 12 groups (30 rows) are in train and 1 group (2 rows, `Time` 148,053) is in validation. Validation's 56 frauds therefore contain 55 distinct fraud input vectors.

**Within-partition duplicates [V]:**

| Key | Train excess rows (fraud) | Validation excess rows (fraud) | Test excess rows (fraud) |
|---|---:|---:|---:|
| Phase 3 key | 716 (18) | 182 (1) | 183 (0) |
| Model input | 5,782 (18) | 915 (1) | 925 (0) |

Repeated rows are counted once each in any metric, so they give some input vectors extra weight **[R]**.

### 5.4 Duplicates, leakage and future-data contamination are different things

**[R]** These three terms are often merged. They should not be:

- **Duplicate observation:** two rows with identical values. This is a fact about the data and may be legitimate. The observed pattern (non-fraud, small amounts, members hours apart) is **consistent with** recurring legitimate purchases, but that is **only a hypothesis**. The dataset has no card, merchant or customer identifier to test it, and anonymized, rounded components could also coincide by chance or through a technical artifact.
- **Potential leakage:** the model is trained on an input that also appears in evaluation, so part of the evaluation measures recall of seen inputs rather than generalization. Cross-partition twins create this *potential*. Its size depends on how many there are, which class they belong to, and how flexible the model is. Here the twins are about 2% of evaluation rows, all non-fraud, with consistent labels. This report **does not claim** that they changed any reported metric. That effect was not measured, and measuring it belongs to the Phase 4 retrospective.
- **Future-data contamination:** information that would not exist at prediction time reaches training or preprocessing. A twin in train that **precedes** its validation or test copy is not contamination. Seeing the same legitimate pattern again later is what production traffic looks like. Twins in validation and test do not reach training at all.

The cross-partition model-input duplicates are therefore documented as an **evaluation-design observation**, not as leakage.

### 5.5 The secondary random split

Phase 3 specified that the random benchmark should "split by duplicate-group identity rather than individual rows; otherwise copies could enter training and evaluation, inflating performance" **[D]**. Phase 4b implemented it with the 30-column Phase 3 key, which includes `Time` **[I]**. At the model-input level, 2,391 groups cross that split, and 1,510 validation rows (3.53%) and 1,622 test rows (3.80%) have exact twins in train, all non-fraud **[V]**. The protection the design intended therefore does not hold at the level the model sees. Per the mentor's decision (2026-11-08), this is documented only. The split is not changed, and no experiment is rerun. Consequences for Phase 4b are left to the Phase 4 report (Section 11).

## 6. Feature handling and preprocessing leakage assessment

**What Phase 3 specified [D]:** no feature list. Its duplicate statement called all 30 non-`Class` columns "feature groups", which implicitly included `Time`. It required that imputers, scalers, feature selection, resampling and learned transformations be fitted on training data only.

**What was implemented [I][D]:**
- Phase 4b set the model features to `V1`–`V28` + `Amount`. `Time` is excluded from model inputs but still drives the split and duplicate grouping (`src/config.py`, mentor-approved 2026-11-07).
- Preprocessing is a `StandardScaler` inside a `ColumnTransformer` with `remainder="drop"`, placed inside the model `Pipeline` (`src/features/preprocessing.py`). Fitting the pipeline on train is the only place statistics are learned.

**Verified at audit [V]:** fitting the repository pipeline on the chronological split gives:
- a scaler that was fitted on the configured 29 columns, in order;
- `n_samples_seen_` = 199,364, the training row count;
- scaler means equal to the training-row means and different from the full-data means;
- no `Time` column among the transformed outputs.

The Phase 4b and 4c work logs record further checks: `fit` received only training rows, and corrupting test rows left results unchanged **[D]**.

**Leakage pathways [R]:**

| Pathway | Assessment |
|---|---|
| Preprocessing fitted on validation or test | Not found. Fit scope verified **[V]** |
| Labels used to choose boundaries | Not found. The rule is label-free **[V]** |
| `Time` used as a model input | Not since Phase 4b **[I]** |
| Test performance used for selection | Not found. Test never scored **[I][D]** |
| Test labels seen during design | Descriptive counts only (Section 3.4) |
| Inherited PCA look-ahead | Dataset-level, unquantifiable (Section 8) |
| Time information inside `V1`–`V28` | Present. `V3`, `V11` and `V25` correlate with `Time` (Phase 2 report, Section 6.3) **[V]**. Excluding raw `Time` does not remove all temporal signal |
| Cross-partition model-input twins | Potential leakage, all non-fraud. Effect not measured (Section 5.4) |

No **implementation-level** leakage was found. The remaining pathways are properties of the supplied dataset or of the evaluation design.

## 7. Temporal representativeness and limitations

**What a chronological split simulates [R]:** a model trained on everything before a cutoff, frozen, and then applied to the transactions that follow. It tests whether patterns learned from the past still hold a few hours later, under whatever drift has occurred. A random split cannot test this. Phase 3's choice of a chronological primary design was sound **[R]**.

**What it does not simulate here [R]:**

- **Full daily coverage.** Validation and test contain no hour values 00–11 (Section 4). Hour values 00–06 have the highest fraud rate in the file (0.474% in train). Every validation and test metric describes hour values 12–23 of day 2 only.
- **Longer horizons.** The whole file covers about 48 hours. Evaluation spans about 5.1 hours (validation) and 6.0 hours (test) immediately after training. Nothing here establishes stability across days, weeks, seasons or fraud campaigns.
- **Representative prevalence.** Validation and test prevalence (0.131%, 0.122%) is below the dataset average (0.173%) because of the hours they cover (Section 4).

**Evidence versus hypothesis:**
- *Evidence* **[V]:** the coverage and prevalence figures in Section 4.
- *Hypothesis* **[R]:** model performance at hour values 00–11 could differ from performance at hour values 12–23, in prevalence-dependent metrics at least. This cannot be tested on validation or test, because those hours are absent from them. Later model-based investigations of time-of-day sensitivity and distribution shift belong to the Phase 4 retrospective report and are not reproduced here.

**Final test scope (to be acknowledged before final test evaluation) [V][R]:**

> The existing test partition represents **hour values 18–23 (elapsed hour modulo 24, not clock time) of the second day of one approximately 48-hour dataset**: `Time` 151,329–172,792, 42,721 transactions, 52 frauds, measured prevalence **0.1217%**. Of those frauds, 18 fall in a single elapsed hour (hour 42). Any final test result describes model behaviour in that window at that prevalence. It does not describe night-time traffic, other days, or production performance.

## 8. Assumptions not documented at the time

Phase 3 noted that "the dataset does not document when labels become known" and asked that the prediction point be confirmed before modeling **[D]**. It did not state the following assumptions, which the chosen design makes implicitly **[R]**:

1. **Label delay.**
   - The design assumes that every training label is known at the moment the training period ends.
   - In practice, fraud confirmations (chargebacks, customer reports) arrive days or weeks later, so a model retrained at a cutoff would not have labels for its most recent transactions.
   - The usual remedy is an **embargo**: a gap of unlabeled time between training and evaluation.
   - Here, consecutive partitions are separated by one second because the next transaction simply follows. That is adjacency, **not an embargo**, and it models no label delay.
2. **Entity grouping.**
   - The dataset has no card, account or merchant identifier.
   - A fraud campaign on one card, or one customer's habitual purchases, can straddle a boundary.
   - A chronological split prevents training on the future, but it does not prevent training on the **same entity** that is later evaluated.
   - This cannot be measured or controlled with the supplied data.
3. **Single cutoff.**
   - Each partition is one contiguous window, and each metric comes from one evaluation period.
   - Phase 3 mentioned that a larger holdout or rolling-origin evaluation could give a more stable temporal assessment **[D]**, but it kept the single 70/15/15 split without recording why.
   - A single cutoff gives one sample of "the future", so its result cannot be separated from the particulars of that afternoon and evening.
4. **Inherited PCA look-ahead.**
   - `V1`–`V28` are centered and uncorrelated over the full file but not over the training rows (Phase 2 report, Section 6.3) **[V]**.
   - This is **consistent with**, but does not prove, a PCA fitted on the full 48 hours.
   - If so, the transformation applied to training rows was partly shaped by the validation and test periods.
   - The **magnitude of any such effect is unknown** and cannot be estimated without the original attributes. It is a dataset-level limitation of every chronological result in this project.

## 9. Reproducibility and documentation gaps

| Gap | Evidence | Why it matters [R] |
|---|---|---|
| No code in the Phase 3 commit; boundaries derived by an unrecorded procedure | `fadebd7` contains prose only **[I]** | The design could not be re-run from the repository until Phase 4a. The audit had to reconstruct the derivation |
| Boundaries are hard-coded constants with no derivation function | `TRAIN_END_TIME`, `VALIDATION_END_TIME` in `src/data/split.py` **[I]** | Correct for this file **[V]**, but a changed or new dataset has no executable rule. Phase 4b notes this limitation **[D]** |
| Phase 3 verification claims recorded after the fact | "Assertions passed" first written in `c0ecd3c`; the assertions are not in the repository at `fadebd7` **[D]** | Documentary evidence only, until Phase 4a tests made the checks reproducible |
| Duplicate check described as a safeguard although it is tautological | Sections 5.1–5.2 | False assurance; the model-input level went unchecked |
| "Feature groups" used for a key that includes `Time`; never reconciled with the Phase 4b feature set | `validation_strategy.md` **[D]**; `src/config.py` **[I]** | The same word names two different column sets |
| No rationale recorded for 70/15/15, for a three-way split, or for choosing a single holdout over rolling-origin evaluation | `validation_strategy.md` **[D]** | Reviewers cannot see what alternatives were weighed |
| Test-label exposure not disclosed | Section 3.4 | Transparency about what was seen before final evaluation |
| Time-of-day coverage of the partitions not examined | Phase 2 deferred it; Phase 3 did not profile it **[D]** | Evaluation scope (Section 7) was implicit |
| Phase name says "Data Preparation" though no data preparation occurred | README `fadebd7` **[D]** | Minor naming mismatch; the README itself states that no preparation was performed |

## 10. What went well, lessons learned and recommendations

**What went well** (judgments **[R]**; facts labelled):

- A chronological primary design was chosen **before** any model existed, with the reasoning stated (future-to-past leakage) **[D]**.
- Boundaries fall between whole timestamps, with inclusive integer ranges documented **[D]**. Equal timestamps never cross partitions **[V]**.
- The boundary rule is label-free and reproduces exactly **[V]**.
- All records were retained instead of deduplicating by reflex **[D]**.
- The random split was framed as secondary, with its weaknesses named **[D]**.
- The threshold decision was deferred to agreed costs and capacity **[D]**.
- The small number of evaluation frauds and the 48-hour coverage were stated as limitations **[D]**.
- The Phase 4a implementation is careful **[I]**:
  - input validation and a pure function of `Time`;
  - 36 tests, including label- and order-independence and real-data regression checks;
  - a mutation-style sensitivity check on the boundary.

**Lessons [R]:**

1. **Check duplicates at the level the model sees.** A duplicate key that includes the split variable cannot detect anything under that split. When the feature set changes, re-run the duplicate analysis on the new feature set.
2. **A safeguard that cannot fail is not a safeguard.** Before relying on a check, ask what input would make it fail.
3. **Commit the procedure, not only its result.** A short script that derives boundaries and counts makes a design self-evidencing.
4. **Profile the evaluation window, not only the dataset.** Hour coverage and like-for-like prevalence explain differences that would otherwise look like drift.
5. **Say what was seen.** Recording that test-label statistics were inspected costs nothing and strengthens trust in the final result.
6. **Name the assumptions a split makes:** label delay, entity independence and a single cutoff, even when the data cannot address them.

**Recommendations** (proposals only; **none implemented**, and each needs separate approval):

- **Model-input duplicate checker (future improvement):** a diagnostic that counts cross-partition groups over the configured model features (`V1`–`V28` + `Amount`), reported alongside the existing check. Per the mentor's decision (2026-11-08), no source code is added or modified during documentation recovery.
- **Boundary derivation as code:** an executable, tested function or script that derives the boundaries from cumulative counts, kept separate from the frozen constants.
- **Pre-registered final test scope:** carry the statement in Section 7 into the final evaluation and the final report.
- **Documentation wording:** where the zero-crossing result is cited, note that it holds by construction for the chronological split. Distinguish the Phase 3 key from the model-input key.
- **Future temporal designs (if more data were available):** an embargo for label delay, entity-grouped splitting, and rolling-origin evaluation.

## 11. Implications for Phase 4

Cross-referenced only. Model-based analyses (including distribution-shift diagnostics, time-of-day sensitivity and uncertainty estimates) belong to the Phase 4 retrospective report, which is pending, and are not reproduced here.

| Phase 3 observation (this report) | Where it matters later | To be analysed in |
|---|---|---|
| Validation covers hour values 12–18 of day 2 at 0.131% prevalence (Sections 4, 7) | Phase 4b Experiment A metrics describe that window only. Average precision is prevalence-dependent, so Experiment A (0.131%) and Experiment B (0.173%) values are not directly comparable | Phase 4 report |
| Random split's duplicate key includes `Time` (Section 5.5) | Experiment B: 1,510 validation rows have model-input twins in train, all non-fraud **[V]**. The effect on reported metrics is not measured | Phase 4 report |
| Chronological model-input twins, about 2% of evaluation rows, all non-fraud (Section 5.3) | Experiment A and Phase 4c, same caveat | Phase 4 report |
| Phase 4c windows lie inside validation (Section 4) | Window 1 is elapsed hours 36.92–39.52 and Window 2 is 39.52–42.04 **[V]**, both within hour values 12–18. The threshold-transfer check covers about 5 hours of one afternoon | Phase 4 report |
| Prevalence and composition differ between train and evaluation (Section 4) | Relevant to the uncalibrated probabilities and the meaning of any threshold | Phase 4 report |
| Final test scope (Section 7) | Must be acknowledged before final test evaluation | Final evaluation and final report |
| Assumptions in Section 8 | Limits every chronological result | Final report limitations |

## Appendix: audit method

- **Environment:** `/Users/duc/Desktop/env/bin/python` (Python 3.13.6, pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.9.1), with bytecode writing disabled.
- **Scripts:** kept in a session scratch directory outside the repository, not committed. They read `data/raw/creditcard.csv` read-only.
- **Independence:**
  - Partition counts, boundary derivation, coverage and duplicate statistics were computed with independent pandas/NumPy code, then compared with `src/data/split.py` and `src/data/random_split.py`.
  - The preprocessing check fitted the repository pipeline on chronological train and inspected the fitted scaler. No test-partition rows were passed to any model.
- **Duplicate keys:** groups formed with `DataFrame.groupby(columns).ngroup()` over the exact column sets in Section 5.1. "Crossing" means a group whose rows carry more than one partition label.
- **Integrity:** the raw-data SHA-256 matched the recorded value. The test suite (`python -B -m unittest discover -s tests`) passed: 143 tests, OK.
