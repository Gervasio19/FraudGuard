# Phase 2 Report — Business & Data Understanding (Retrospective)

> **Report type:** retrospective. This report was written on 2026-11-08, after Phases 1–4c were complete. No Phase 2 report was written at the time. Historical claims are drawn from commit `812c356` and later Git history. Every number marked **[V]** was computed during the 2026-11-08 audit. These numbers are **not** original Phase 2 outputs, because the Phase 2 notebook was committed without saved outputs (Section 7). Explanations of *why* something matters are retrospective interpretation, not the original author's recorded reasoning.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[I]** | Implemented in the repository (files or code exist; commit cited) |
| **[D]** | Documented in the repository (stated in a committed document) |
| **[V]** | Newly verified or computed during the 2026-11-08 retrospective audit |
| **[R]** | Retrospective explanation or interpretation |

**Phase 2 at a glance**

| Item | Value |
|---|---|
| Commit | `812c356f370d26b7f98a60d03ba39fffe1e85699`, "Complete Phase 2" |
| Date | 2026-11-07 17:17:42 +0700 |
| Files | `docs/business_understanding.md` (new, 59 lines), `notebooks/01_data_understanding.ipynb` (new, 14 cells), `README.md` (7 lines changed) |
| Code modules, tests, processed data, figures | None |
| Later changes to Phase 2 files | `docs/business_understanding.md` was edited in `fadebd7` (Phase 3); see Section 8. The notebook has not changed since `812c356` **[V]** |
| Dataset fingerprint (SHA-256) at audit | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, matching the value recorded since Phase 4a **[V]** |

## 1. Phase objectives

The Phase 2 README (`812c356`) gave the status as "Under development — Phase 2: Business Understanding & Data Understanding. Business framing and basic dataset inspection are documented; no machine learning models have been trained." **[D]** The roadmap entry became "Business understanding and dataset exploration" **[D]**.

The original scope paragraph in `docs/business_understanding.md` (`812c356`) read **[D]**:

> Phase 2 documents the business framing and performs basic data understanding only. The current scope does not include model training, advanced exploratory analysis, production deployment, live payment decisions, or claims of operational effectiveness.

The notebook's opening cell says the same: it "checks the local Credit Card Fraud Detection dataset's structure, documented feature meanings, target balance, and basic data quality. It does not train models or perform advanced exploratory analysis." **[D]**

Phase 2 therefore set itself two goals: **frame the business problem** and **inspect the data's basic structure and quality**. This report judges Phase 2 against those goals. It does not hold Phase 2 responsible for analyses it explicitly placed out of scope. Where an unexamined property later mattered, the report says so as a retrospective observation **[R]**.

## 2. Business framing

**What Phase 2 documented [D] (`docs/business_understanding.md`, `812c356`):**

- **Problem:** payment card fraud causes direct losses and operational work. Detection must be timely enough to act on, while letting legitimate purchases proceed.
- **Dataset context:** European cardholder transactions from two days in September 2013. The data is a historical, anonymized research set, not a production feed.
- **Four example uses:** real-time authorization support, an analyst review queue, customer step-up verification, and portfolio monitoring. The document states that these are framing examples and that the dataset does not describe any bank's workflow.
- **Costs of errors:**
  - A *false negative* (missed fraud) leads to unrecovered loss, follow-on fraud, disputes and customer harm.
  - A *false positive* (false alarm) leads to declined purchases, unnecessary verification, customer frustration, lost sales and analyst workload.
- **No cost function:** "The costs depend on transaction amount, recovery rates, downstream fraud, review capacity, and customer impact. Those quantities are not supplied in this dataset, so a monetary cost function cannot yet be specified credibly."

**Why this matters [R]:** a classifier is only good or bad relative to the decision it supports. The same model can be useful for a review queue that takes the top 50 alerts a day and useless for automatic declines. By naming the decisions first and refusing to invent costs, Phase 2 kept later phases from optimizing a number that had no business meaning. Phase 4c kept to this: its F1-maximizing thresholds are labelled analytical, not operational **[D]**.

### Why accuracy is misleading

Phase 2 states that predicting "legitimate" for every transaction gives about 99.83% accuracy while detecting none of the 492 frauds **[D]**. Recomputed: majority-class accuracy is **99.8273%** **[V]**.

**Concept [R]:** accuracy weights every row equally. When 99.83% of rows belong to one class, accuracy mostly measures how well that class is handled. It can sit near its maximum while the rare class, the one the business cares about, is ignored entirely. A useful metric for rare-event detection must look at the positive class directly.

## 3. Evaluation metrics proposed, and why

Phase 2 recommended reporting, at minimum **[D]**:

| Metric | What it answers | Why it suits this problem [R] |
|---|---|---|
| Recall | What share of fraud do we catch? | Measures missed fraud (false negatives) directly |
| Precision | What share of alerts are real fraud? | Measures alert quality and analyst burden |
| Precision–recall curve and average precision | How does the precision/recall trade-off look across all thresholds? | Both axes concern the positive class, so the large number of legitimate rows cannot inflate it |
| Confusion matrix at a stated threshold | How many of each error at one operating point? | Makes both error types countable |
| Alert volume and threshold behaviour | How many cases need action? | Translates a score into workload |
| ROC-AUC, "with context", secondary | How well are frauds ranked above non-frauds overall? | See below |

**Why ROC-AUC is secondary [R]:** the ROC false-positive rate divides false positives by *all* legitimate transactions, about 284,000 here. A model can raise thousands of false alarms and still have a tiny false-positive rate, so ROC-AUC can look excellent while precision is poor. Phase 4b later showed this on validation **[D]**: Logistic Regression had ROC-AUC 0.9815 but precision 0.0397 at the 0.5 rule.

Phase 2 also said that threshold selection should reflect agreed capacity and error costs, that no production threshold should be chosen in Phase 2, and that later evaluations should compare against a simple baseline **[D]**. Phase 4b implemented the baseline comparison, using a majority-class model **[I]**.

### Supplementary metric: amount-weighted recall (retrospective suggestion)

Phase 2 concluded that no monetary cost function could be specified **[D]**. That conclusion stands. A narrower, *optional* supplementary metric is still available, because the dataset contains `Amount` **[R]**:

```
amount-weighted recall = (sum of Amount over detected frauds) / (sum of Amount over all frauds)
```

- **What it adds [R]:** standard recall counts every fraud equally. Amount-weighted recall asks what share of the fraudulent *transaction value* the model catches. Two models with the same recall can differ here, if one catches mainly small frauds and the other mainly large ones.
- **What it is not [R]:**
  - **It is not financial loss.** Actual loss depends on chargebacks, recovery, liability rules, follow-on fraud and operating costs, none of which are in the data. A €500 fraud may be fully recovered and a €5 card-testing transaction may precede large losses.
  - **It does not replace standard recall.** Fraud recall stays the primary detection measure. Amount-weighted recall, if reported, is an addition.
  - **It is fragile with few cases.** With 52–74 frauds per evaluation partition **[D]**, a single large transaction can dominate the ratio.
- **Status:** a suggestion only. It is not implemented, and adopting it would be a mentor decision.

## 4. Dataset provenance and feature meaning

**Publisher documentation, as quoted by the project [D]** (`docs/business_understanding.md`, notebook cell 3): the Kaggle dataset covers two days and contains 284,807 transactions with 492 frauds.
- `Time` is the number of seconds elapsed since the first transaction.
- `Amount` is the transaction amount.
- `V1`–`V28` are principal components obtained with PCA, and the original attributes are not disclosed.
- `Class` is 1 for fraud and 0 otherwise.

**Verified [V]:** 284,807 rows, 31 columns in the order `Time, V1 … V28, Amount, Class`, and 492 frauds.

**What PCA anonymization means [R]:** principal component analysis rotates the original variables into new uncorrelated axes, ordered by how much variance each explains. The publisher used it to hide the original attributes. Two consequences follow:

1. **No business meaning for individual features.** Phase 2 recorded this correctly **[D]**: "this notebook does not assign them individual business meanings."
2. **The transformation was fitted by someone else, on data we cannot see.** We do not know which rows the PCA was fitted on, whether variables were scaled first, or how. That has a methodological consequence, discussed in Section 6.3, which Phase 2 did not raise.

The currency of `Amount` is not stated in the project documentation **[V]**. Calling it euros would be an assumption **[R]**.

## 5. Data quality checks performed

Phase 2's notebook performs these checks **[I]**. Because the committed notebook contains no outputs (Section 7), the **Result** column comes from re-executing an unchanged copy of the notebook during the audit and from an independent script **[V]**.

| Check (notebook cell) | Result at audit [V] | Why the check matters [R] |
|---|---|---|
| Shape (2) | 284,807 rows × 31 columns | The loaded dataset has the expected dimensions, consistent with the publisher's description. Matching dimensions alone do not prove that the file is complete or uncorrupted; the SHA-256 fingerprint identifies the exact file |
| Column names and `df.info()` (4) | Expected 31 names, all non-null | Catches a wrong or truncated file and renamed columns |
| Data types (6) | 30 `float64`, 1 `int64` (`Class`) | Columns read as text would silently break statistics and models |
| Missing values (6, 12) | 0 in every column; 0 rows with any missing value | Decides whether imputation is needed. If it were, imputation would have to be fitted on training data only |
| Class balance (8) | 284,315 non-fraud and 492 fraud; prevalence 0.1727% | Sets the scale of the imbalance problem and the baseline |
| Majority-class accuracy (8) | 99.8273% | Quantifies why accuracy is misleading (Section 2) |
| `describe()` (10) | Printed for all 31 columns | Gives scale and range. `Time` and `Amount` are on very different scales from `V1`–`V28` |
| Full-row duplicates (12) | 1,081 | See Section 6.1 |
| Target values outside {0, 1} (12) | None | An invalid label would corrupt every metric |

Additional properties checked independently during the audit **[V]**, none of which the notebook checked:

- **Infinite values:** 0.
- **`Time`:**
  - Stored as `float64`, but every value is a whole number.
  - Non-decreasing in file order, ranging from 0 to 172,792 seconds (about 48 hours).
  - 124,592 distinct values.
- **`Amount`:**
  - Minimum 0.00, maximum 25,691.16, median 22.00. No negative values, and no value has more than two decimal places.
  - Strongly right-skewed (skewness 16.98).
  - **1,825 transactions have `Amount = 0`, 27 of them fraud.**
  - The fraud median is 9.25, against 22.00 for non-fraud. The fraud maximum is 2,125.87.

**Why these matter [R]:**
- A heavily skewed feature can dominate a scaled linear model through a few extreme values. Phase 4b uses `StandardScaler` on `Amount` **[I]**. Standardization subtracts the training mean and divides by the training standard deviation. That changes the feature's location and scale only. The shape of the distribution is unchanged, so the skewness stays the same, and extreme values remain extreme relative to the rest. Standardization does not remove skewness or eliminate the influence of extreme values.
- Zero-amount transactions may be card verifications rather than purchases. They are unexplained in the documentation.
- The whole-number `Time` values were first enforced in code in Phase 4a, where `src/data/split.py` rejects non-integer times **[I]**.

## 6. What Phase 2 did not examine

These are retrospective observations. Phase 2 called duplicates, invalid values and other issues something that "can be considered in later preparation work" (notebook cell 7) **[D]**. Each item below records **what was found**, **why it matters**, and **where its consequences are analysed**. The consequences belong to the Phase 3 and Phase 4 reports, which are pending.

### 6.1 Duplicate definitions: full-row vs model-feature duplicates

**What Phase 2 computed [I]:** `df.duplicated()`, over all 31 columns, including `Time` and `Class`.

**Three possible definitions, computed during the audit [V]:**

| Definition | Columns compared | Excess rows* | Groups of identical rows | Groups with conflicting labels |
|---|---|---:|---:|---:|
| **Exact full-row duplicate** (Phase 2) | All 31 | 1,081 | 773 | – |
| Identical apart from the label (Phase 3 rule) | All except `Class` (30) | 1,081 | 773 | 0 |
| **Model-feature duplicate** | `V1`–`V28` + `Amount` (29), the Phase 4b model inputs | **9,144** | **5,149** | 0 |

\* Excess rows means rows beyond the first in each group.

In every definition, 32 fraud rows belong to duplicate groups, and no group mixes fraud and non-fraud labels **[V]**.

**Why excluding `Time` changes the definition [R]:** `Time` is a timestamp. Two transactions with identical `V1`–`V28` and `Amount` but recorded a few seconds or hours apart are *not* full-row duplicates, because their `Time` differs. Phase 4b removed `Time` from the model inputs **[D]**. To the model, those two rows are the same input vector. Dropping one distinguishing column makes rows collapse together, which is why the count rises from 1,081 to 9,144. The right duplicate definition depends on the question:
- **Data-quality question** ("was a record written twice?"): full rows are the natural unit.
- **Model-evaluation question** ("has the model already seen this exact input?"): the model's own feature set is the relevant unit.

**Duplicates are not automatically leakage [R].** Identical feature vectors can be legitimate. One customer may make the same purchase from the same merchant repeatedly, and anonymized, rounded features can coincide by chance. They can also be technical artifacts. The dataset documentation does not say which. A duplicate is therefore an *observation about the data*, not a defect to delete. The project's decision to keep all rows (Phase 3) is consistent with this **[D]**.

**When duplicates can bias evaluation [R]:** if the same feature vector appears in both training and evaluation data, the model is partly scored on inputs it was fitted on. A flexible model can memorize them, so the evaluation can be **optimistic**: it reflects recall of seen inputs more than generalization to new ones. Whether this matters depends on how many such vectors cross partitions, which classes they belong to, and the model type. All 9,144 model-feature excess rows have consistent labels, and the fraud rows among them are few (32 in duplicate groups overall). Phase 2 evidence alone does not show the size of any effect.

**Where the consequences are analysed:** the partition-level counts and their effect on the chronological and random splits belong in the Phase 3 and Phase 4 retrospective reports (pending). This report does not analyse them.

### 6.2 Time-of-day structure

Phase 2 recommended "a validation design that respects the data's time structure" (`812c356`) **[D]**, but the notebook did not profile how fraud varies over time **[V]**.

**Important caveat on clock time [R]:** `Time` counts seconds from the first transaction in the file. The project documentation does not state the clock time of that first transaction. The table below therefore uses **elapsed hour modulo 24**, `floor(Time / 3600) mod 24`, and **not** the true local hour. The low transaction volume in hours 01–06 is consistent with night-time if `Time = 0` falls near midnight. This is an inference, not a documented fact.

**Table: transactions and frauds by elapsed hour modulo 24 (both days combined).** Computed during the 2026-11-08 audit from `data/raw/creditcard.csv` **[V]**. Not an original Phase 2 output.

| Elapsed hour mod 24 | Transactions | Frauds | Fraud rate |
|---:|---:|---:|---:|
| 00 | 7,695 | 6 | 0.078% |
| 01 | 4,220 | 10 | 0.237% |
| 02 | 3,328 | 57 | 1.713% |
| 03 | 3,492 | 17 | 0.487% |
| 04 | 2,209 | 23 | 1.041% |
| 05 | 2,990 | 11 | 0.368% |
| 06 | 4,101 | 9 | 0.219% |
| 07 | 7,243 | 23 | 0.318% |
| 08 | 10,276 | 9 | 0.088% |
| 09 | 15,838 | 16 | 0.101% |
| 10 | 16,598 | 8 | 0.048% |
| 11 | 16,856 | 53 | 0.314% |
| 12 | 15,420 | 17 | 0.110% |
| 13 | 15,365 | 17 | 0.111% |
| 14 | 16,570 | 23 | 0.139% |
| 15 | 16,461 | 26 | 0.158% |
| 16 | 16,453 | 22 | 0.134% |
| 17 | 16,166 | 29 | 0.179% |
| 18 | 17,039 | 33 | 0.194% |
| 19 | 15,649 | 19 | 0.121% |
| 20 | 16,756 | 18 | 0.107% |
| 21 | 17,703 | 16 | 0.090% |
| 22 | 15,441 | 9 | 0.058% |
| 23 | 10,938 | 21 | 0.192% |
| **Total** | **284,807** | **492** | **0.173%** |

**Summary bands [V]:**

| Band (elapsed hour mod 24) | Transactions | Frauds | Fraud rate |
|---|---:|---:|---:|
| 00–06 (low-volume hours) | 28,035 | 133 | 0.474% |
| 07–23 | 256,772 | 359 | 0.140% |

By day: elapsed day 1 (`Time` < 86,400) has 144,786 transactions and 281 frauds; day 2 has 140,021 transactions and 211 frauds **[V]**.

**How to read this [R]:**

- **Volume and rate move in opposite directions.** The low-volume hours 00–06 hold 9.8% of transactions but 27.0% of frauds. Their fraud rate is about 3.4 times that of the remaining hours.
- **Single-hour rates rest on small counts and need care.** Hour 02 (57 frauds, 1.713%) and hour 11 (53 frauds, 0.314%) stand out. Spikes like these can come from a single burst of related fraud, for example one compromised card tested repeatedly. Each hour is observed on at most two days, so neither spike can be called a stable pattern.
- **Why this matters for evaluation:** if fraud prevalence and possibly fraud *behaviour* differ by time of day, then a model's performance measured over one part of the day need not carry over to another.

**Consequence for later partitions (stated only) [V]:** in this subsection, "hour value" means elapsed hour modulo 24, `floor(Time / 3600) mod 24`. It is not a clock hour. The Phase 3 boundaries are defined in seconds as closed intervals, and every row falls in exactly one partition:

| Partition | `Time` interval (seconds, inclusive) | Elapsed hours (exact) | Hour values (elapsed hour mod 24) present |
|---|---|---|---|
| Train | 0–132,928 | 0.000–36.924 | All, 00–23 |
| Validation | 132,929–151,328 | 36.925–42.036 | 12–18 |
| Test | 151,329–172,792 | 42.036–47.998 | 18–23 |

The two partition boundaries fall *inside* an elapsed hour, so two hour values are split between neighbouring partitions **[V]**:

- **Hour value 12** (elapsed hour 36, `Time` 129,600–133,199): 7,133 rows (7 frauds) are in train (`Time` ≤ 132,928), and 555 rows (1 fraud) are in validation.
- **Hour value 18** (elapsed hour 42, `Time` 151,200–154,799): 270 rows (0 frauds) are in validation (`Time` 151,200–151,328), and 8,162 rows (18 frauds) are in test (`Time` ≥ 151,329).

Hour value 18 therefore appears in both validation and test, but no row belongs to both; the partitions are disjoint in `Time`. Validation and test contain **no observations with hour values 00–06** (the low-volume hours), nor 07–11. Whether this biases the Phase 4 metrics, and in which direction, needs a **separate evaluation analysis**. It cannot be concluded from this table. That analysis belongs in the Phase 3 and Phase 4 retrospective reports (pending).

### 6.3 PCA provenance and possible full-dataset preprocessing

**What the publisher documents [D]:** `V1`–`V28` are "principal components obtained with PCA", and `Time` and `Amount` were not transformed. The project documentation does not record which rows the PCA was fitted on, or whether the inputs were scaled first.

**What the data shows [V]:**

| Property, over all 284,807 rows | Value |
|---|---|
| Largest absolute mean of `V1`–`V28` | 4.87 × 10⁻¹⁵ |
| Largest absolute pairwise correlation between `V1`–`V28` | 2.32 × 10⁻¹⁴ |
| Variances of `V1` → `V28` | Non-increasing, from 3.836 to 0.109 |
| Same, restricted to the chronological training rows (`Time` ≤ 132,928) | Largest absolute mean 0.3208; largest absolute pairwise correlation 0.1049 |

**Interpretation, qualified [R]:** PCA output is centered (mean zero), uncorrelated and ordered by variance, *on the data it was fitted on*. These properties hold almost exactly over the full 284,807 rows and visibly fail on the training rows alone. This pattern is **consistent with** a PCA fitted on the full 48-hour dataset, or on a set very close to it. It does not prove it. Other procedures could produce the same pattern, for example a PCA fitted on a larger source followed by re-centering of this extract. The publisher's fitting procedure cannot be verified from the repository or the file.

**Why it would matter [R]:** in a chronological evaluation, the training data should be processed using only information available up to the end of training. If the PCA was fitted on all 48 hours, the mean and rotation applied to the training rows were partly shaped by the validation and test periods. This would be an **inherited** form of look-ahead. Its magnitude and impact on downstream model performance are unknown. The effect cannot be quantified from the supplied transformed dataset. It also cannot be removed, because the original attributes are not available. The honest response is to **state it as a dataset-level limitation** of every chronological result in this project, not to claim that chronological evaluation is entirely free of future information.

**Time information inside the features [V]:**
- Several components are correlated with `Time` across the full dataset: `V3` (r = −0.420), `V11` (−0.248) and `V25` (−0.233).
- `V2` (−0.531) and `V7` (0.397) are correlated with `Amount`.

**[R]:** removing the raw `Time` column (Phase 4b) does not remove all temporal signal from the model inputs. This does not contradict the Phase 4b decision, which excluded *raw* `Time` because validation values lay outside the training range **[D]**. It does mean that "the model uses no time information" would be an overstatement.

## 7. Reproducibility of the notebook

**Historical evidence [D]:**
- `docs/WORK_LOG.md`, in a section first written in `c0ecd3c` (after Phase 3), states that all code cells were executed in order under Python 3.13.6, pandas 2.3.1 and NumPy 2.3.2 and completed successfully.
- It also records an earlier failed attempt that lacked Jupyter's `display` binding, corrected in the notebook before it was committed.

**What the committed notebook contains [I][V]:**
- Every code cell has `execution_count: null` and no outputs.
- The duplicate count, 1,081, is not written in any Phase 2 file. Cell 13 says only that the "duplicate and target checks above provide additional basic checks when run".
- The figure first appears in Phase 3 documentation (`fadebd7`, `docs/validation_strategy.md`).

**Re-execution at audit [V]:**
- **Method:** an unchanged copy of the notebook was executed with `jupyter nbconvert --execute`, using the `python3` kernel of the designated environment `/Users/duc/Desktop/env`. The copy sat in a scratch directory outside the repository, with a symbolic link to `data/raw/creditcard.csv`.
- **Result:** every cell completed without error, and the printed versions were Python 3.13.6, pandas 2.3.1 and NumPy 2.3.2. Every printed figure matched Section 5. The repository copy was not modified.

**Observations [V]:**

- The notebook metadata names the generic `python3` kernel ("Python 3"), not the registered `fraudguard` kernel. On this machine `python3` resolves to the designated environment. On another machine it would resolve to whatever interpreter is installed there.
- `np.random.seed(42)` is set, but no cell uses randomness, so the seed has no effect. Phase 4 code passes `random_state` explicitly instead **[I]**.
- `find_project_root()` searches upward from the working directory. The notebook therefore works from the repository root or from `notebooks/`, but not when executed from elsewhere. An execution attempt from outside the repository failed with `FileNotFoundError`, as designed.
- No dataset hash was recorded in Phase 2. It was first recorded in Phase 4a; see `docs/reports/phase1_foundation.md`, Section 4.
- No plots were produced, although Matplotlib and Seaborn were in `requirements.txt` **[I]**. This is consistent with the "basic" scope.

**Why saved outputs matter [R]:** a notebook committed with its outputs is its own evidence. A reviewer can see what the code printed on the day it ran. A notebook without outputs is a *recipe*: it shows what would be computed, not what was. Here the recipe re-runs and reproduces exactly, so no result is in doubt. The historical evidence is still weaker than it could have been, and the Phase 2 figures had to be reconstructed from later documents and re-execution. Per the owner's decision (2026-11-08), the notebook stays unchanged; this report serves as the record of its outputs.

## 8. Document history and provenance

| File | Created | Later changes | Label |
|---|---|---|---|
| `docs/business_understanding.md` | `812c356` (Phase 2) | `fadebd7` (Phase 3) rewrote three paragraphs (below). Unchanged since | [I][V] |
| `notebooks/01_data_understanding.ipynb` | `812c356` | None | [V] |
| `README.md` (Phase 2 lines) | `812c356` | Rewritten in every later phase | [V] |
| Phase 2 execution evidence in `docs/WORK_LOG.md` | `c0ecd3c`, written after Phase 3 | Carried forward unchanged | [D] |

**Paragraphs of `docs/business_understanding.md` changed in `fadebd7` (Phase 3).** The original Phase 2 wording, quoted from `812c356` **[D]**:

1. *Objectives and scope:* "Phase 2 documents the business framing and performs basic data understanding only. …" became "Phases 2 and 3 document business understanding, basic data understanding, and the validation design. …"
2. *Proposed evaluation considerations, first paragraph:* "Before modeling, agree on a validation design that respects the data's time structure and avoids information leakage. The dataset spans only two days, so a time-based holdout may be especially sensitive to the small number of fraud cases and should be reviewed carefully." This was replaced by a description of the finalized chronological 70/15/15 design.
3. *Threshold paragraph:* "… this phase should not choose a production threshold …" became "… no classification threshold has been selected …"

All other sections are the original Phase 2 text **[V]**. Anyone citing what Phase 2 decided should cite the `812c356` version. Per the owner's decision, the current file is not edited.

**`README.md` changes in `812c356` [D]:** the status, objectives, dataset sentence ("A local copy is present at `data/raw/creditcard.csv`"), and roadmap were updated. The environment line changed from "Python 3.14.6" to "Development environment verified for this project: Python 3.13.6 (`/Users/duc/Desktop/env/bin/python`)".

## 9. Consequences for later phases

Each item is cross-referenced, not analysed, here. The Phase 3 and Phase 4 retrospective reports are pending, and their content is not pre-empted.

| Phase 2 observation (this report) | Where it matters later | To be analysed in |
|---|---|---|
| Duplicates defined over full rows only (6.1) | Phase 3 adopted "all columns except `Class`" for duplicate grouping; Phase 4b removed `Time` from the model features **[D]** | Phase 3 and Phase 4 reports |
| Time-of-day structure not profiled (6.2) | Phase 3 chose chronological boundaries; validation and test contain only hour values (elapsed hour mod 24) 12–18 and 18–23; hour value 18 is split between them at the boundary (6.2) **[V]** | Phase 3 and Phase 4 reports |
| PCA properties consistent with full-dataset fitting; `V` components correlated with `Time` (6.3) | Limits the claim that chronological evaluation uses no future information; relevant to the Phase 4b `Time` exclusion | Phase 3 and Phase 4 reports |
| `Amount` skew and zero amounts not profiled (5) | Phase 4b scales `Amount` with `StandardScaler` **[I]** | Phase 4 report |
| Amount-weighted recall available as an optional supplementary metric (3) | Any later evaluation | Mentor decision |
| Business framing: no threshold without costs (2) | Followed in Phase 4b (0.5 used only for diagnostics) and Phase 4c (analytical thresholds) **[D]** | — |

## 10. Lessons learned and future improvements

**What Phase 2 did well** (the judgments are **[R]**, the underlying facts are labelled):

- It framed the problem around decisions and error costs before touching models, and refused to invent a cost function the data cannot support **[D]**.
- It identified why accuracy fails under imbalance and recommended precision- and recall-based metrics, with ROC-AUC only as a secondary measure **[D]**. Later phases followed this **[I]**.
- It recommended a time-aware, leakage-aware validation design and a simple baseline, both later implemented **[D][I]**.
- Every claim it made reproduces exactly from the raw file **[V]**.
- It stated its limitations honestly: 48 hours of 2013 data, anonymized features, no production claims **[D]**.

**Lessons [R]:**

1. **Save the outputs of evidence notebooks.** A notebook committed without outputs records a method, not a result. Saving outputs, or writing the key figures into a document when they are computed, makes the phase self-evidencing.
2. **Define duplicates relative to the question.** "Duplicate" is not one fixed property. Data quality and model evaluation need different definitions (Section 6.1).
3. **Profile the axis you will split on.** If validation is to be chronological, the time structure of the target (prevalence by hour and by day) is part of data understanding, not an advanced extra.
4. **Ask how inherited transformations were fitted.** Preprocessing done by the data publisher can carry information across any split the project later makes. When it cannot be verified, state it as a limitation in qualified terms.
5. **Profile skewed and boundary values.** Zero amounts and extreme skew are cheap to find and inform later preprocessing choices.
6. **Keep the original phase record distinct from later edits.** When a later phase rewrites a document, its earlier wording survives only in Git. Phase reports should cite the commit that holds the original.

**Future improvements** (proposals only; none implemented, and each needs approval):

- Partition-level analysis of model-feature duplicates, and of time-of-day coverage and its effect on metrics (Phase 3 and Phase 4 reports).
- State the PCA provenance limitation in the project's limitations section once the Phase 3 and Phase 4 reports are reviewed.
- Optionally report amount-weighted recall as a supplementary metric, alongside standard recall.
- In future notebooks, name the project kernel in the metadata and commit outputs for evidence notebooks.
