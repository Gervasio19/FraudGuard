# Phase 1 Report — Project Foundation (Retrospective)

> **Report type:** retrospective. This report was written on 2026-11-08, after Phases 1–4c were complete. No Phase 1 report or work-log entry was written at the time. Every historical claim is drawn from commit `e4e866d` and later Git history. Explanations of *why* a choice is sound are labelled as retrospective interpretation, not as the original author's recorded reasoning.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[I]** | Implemented in the repository (files or code exist; commit cited) |
| **[D]** | Documented in the repository (stated in a committed document) |
| **[V]** | Newly verified during the 2026-11-08 retrospective audit |
| **[R]** | Retrospective explanation or interpretation |

**Phase 1 at a glance**

| Item | Value |
|---|---|
| Commit | `e4e866d15087b0f939634f726b713b986e7e8772`, "Initialize FraudGuard project structure" |
| Date | 2026-11-06 00:23:20 +0700 |
| Size | 17 files, 100 lines added |
| Code, data or tests added | None |

## 1. Phase objectives

The Phase 1 README (`e4e866d`) states four project objectives **[D]**:

1. Establish a reproducible project structure and development environment.
2. Explore the selected public credit card fraud dataset in a later phase.
3. Develop and evaluate fraud detection approaches in subsequent phases.
4. Present findings and project outputs clearly.

Objectives 2 and 3 are placed in later phases by their own wording **[D]**. The README's status line read: "Under development — Phase 1: Project Initialization. No data analysis or machine learning functionality has been implemented." **[D]** Objective 4's wording states no timing **[D]**. Reading objective 1 as the only Phase 1 objective is an interpretation **[R]**.

The Phase 1 roadmap had five phases **[D]**:

1. Project initialization.
2. Dataset exploration and quality assessment.
3. Data preparation and feature engineering.
4. Model development and evaluation.
5. Results communication and project presentation.

The current README renames two of these phases **[D]**: Phase 2 became "Business understanding and dataset exploration", and Phase 3 became "Data preparation and validation design", replacing "feature engineering" (see also commit `fadebd7`, "Complete Phase 3: Data Preparation and Validation"). The Phase 1 roadmap mentioned no explainable AI, Streamlit application or deployment stage, and neither does the current README **[V]**.

## 2. Project structure and separation of concerns

Commit `e4e866d` created this layout **[I]**:

```
FraudGuard/
├── README.md, requirements.txt, .gitignore
├── data/
│   ├── README.md              # how to obtain the dataset
│   ├── raw/.gitkeep           # untouched source data (ignored by Git)
│   └── processed/.gitkeep     # derived data (ignored by Git)
├── notebooks/.gitkeep         # exploratory analysis
├── src/__init__.py            # importable Python package
│   ├── data/__init__.py       # loading and splitting
│   ├── features/__init__.py   # preprocessing / feature engineering
│   ├── models/__init__.py     # model definitions and training
│   ├── evaluation/__init__.py # metrics and analysis
│   └── visualization/__init__.py
├── tests/.gitkeep
├── reports/figures/.gitkeep   # generated results and figures
├── artifacts/.gitkeep         # trained model files (ignored by Git)
└── app/components/.gitkeep    # application layer
```

The comments describe each folder's apparent purpose **[R]**. Inside the new folders, Phase 1 added only zero-byte `__init__.py` and `.gitkeep` files, apart from `data/README.md` **[I]**. No Phase 1 document described folder responsibilities in general **[V]**. The only folder-level statements are in `data/README.md`, which names `data/raw/` as the location for the downloaded file and says raw and processed data are excluded from Git **[D]**.

**Why this separation matters [R]:**

- **Raw vs processed data:** keeping the source data apart from anything derived means a mistake in a later step can always be traced back to, and rebuilt from, an unchanged original.
- **`notebooks/` vs `src/`:** notebooks are for exploration and explanation. Logic that is reused or must be correct belongs in importable modules that can be tested. In later phases the project followed this rule: all splitting, modelling and evaluation logic lives in `src/` and is covered by `tests/` (Phases 4a–4c) **[I]**.
- **`src/` subpackages:** the `data → features → models → evaluation` structure mirrors the stages of an ML workflow. Phases 4a–4c placed their modules accordingly, for example `src/data/split.py` and `src/evaluation/threshold.py` **[I]**.
- **`artifacts/` vs `reports/`:** trained models (large, binary, regenerable) are kept apart from results (small, human-readable, worth versioning).

**How the placeholders were used later [V]:**

| Placeholder | Later use |
|---|---|
| `src/data`, `src/features`, `src/models`, `src/evaluation` | Filled in Phases 4a–4c |
| `notebooks/` | One notebook, added in Phase 2 |
| `tests/` | Five test files, from Phase 4a onward |
| `reports/` | Two JSON result files, from Phase 4b onward |
| `src/visualization/`, `app/components/`, `artifacts/`, `reports/figures/` | Still empty: the project has no figures or saved model files |
| `tests/.gitkeep` | Remains, although the folder now has real files; harmless |

The `app/components/` placeholder suggests an application layer was expected **[R]**. No Phase 1 document describes its purpose **[V]**.

## 3. Why raw data is kept outside Git

**What was implemented [I] (`.gitignore`, `e4e866d`):**

- `/data/raw/*` and `/data/processed/*` are ignored. The `.gitkeep` markers are re-included (`!/data/raw/.gitkeep`, `!/data/processed/.gitkeep`), and both are tracked, so the folders exist in a fresh clone **[I][V]**. The reason for this choice is not recorded; keeping the folder layout is the likely purpose **[R]**.
- `/artifacts/*` and common model-file extensions (`*.pkl`, `*.joblib`, `*.h5`, `*.keras`, `*.pt`, `*.pth`, `*.onnx`) are ignored.
- Python caches, `.venv/`, `.env*` secrets (except `.env.example`), IDE files and OS files are ignored.
- The line `!/data/README.md` is redundant, because `data/README.md` is not matched by any ignore rule. It is harmless **[V]**.

**What was documented [D] (`data/README.md`):** raw and processed data "are excluded from Git to avoid committing large data files and to respect dataset distribution terms." The file names the source (Kaggle, `mlg-ulb/creditcardfraud`), the expected path `data/raw/creditcard.csv`, and the download steps.

**Verified [V]:**

- `git check-ignore` confirms `data/raw/creditcard.csv` is ignored by `.gitignore` line 20 (`/data/raw/*`).
- No version of the file was found in the reachable history of the local Git refs examined during this audit (`main`, `origin/main` and two tool-created checkpoint refs under `refs/codex/`): `git log --all -- data/raw/creditcard.csv` returns no commits, and `git rev-list --all --objects` lists no path containing `creditcard.csv`. At `HEAD`, the only tracked file under `data/raw/` is `.gitkeep`.
- The local file is 150,828,752 bytes.

**Why this is the right decision [R]:**

1. **Size.** At about 144 MiB, the file exceeds GitHub's 100 MB per-file limit. Large binary data also makes every clone slower, permanently, because Git keeps every version.
2. **Licensing.** The dataset is distributed by its publisher under Kaggle's terms. Re-hosting it in a public repository could breach those terms. The repository records *where to get* the data, not the data itself. The terms themselves are not recorded in the repository.
3. **Immutability.** Raw data should never be edited. Keeping it outside version control, with a documented source, encourages treating it as read-only input.

**Trade-off [R]:** the repository can no longer reproduce results on its own. Anyone who clones it must download the data separately, and nothing in Phase 1 let them confirm they had the *same* file. Section 4 covers how that gap was closed later.

## 4. The role of data integrity and SHA-256

**Concept [R]:** SHA-256 is a cryptographic hash function. It maps a file of any size to a 256-bit fingerprint, written as 64 hexadecimal characters. Changing even one byte produces a completely different fingerprint, and finding two different files with the same fingerprint is computationally infeasible. If the data file is not stored in Git, its hash is the practical way to prove that a result was computed from exactly that file.

**Phase 1 recorded no dataset hash [V].** `e4e866d` contains no SHA-256 value **[V]**. The Phase 1 README states that the dataset "has not been downloaded or analyzed" **[D]**.

When the hash entered the project, and how:

| Step | Evidence | Label |
|---|---|---|
| Hash first **recorded** in documentation | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` first appears in commit `37d4521` (Phase 4a, 2026-11-07), in `docs/WORK_LOG.md` | [D] |
| Hash first **computed by project code** | `_sha256()` in `src/models/run_baselines.py` (commit `4b5b94b`, Phase 4b) writes it into `reports/phase4b_baseline_results.json`. The Phase 4c runner reuses it in `reports/phase4c_threshold_results.json` | [I] |
| Hash **reported as checked** before and after Phases 4a, 4b and 4c | `docs/WORK_LOG.md` | [D] |
| Hash **independently re-verified** | `shasum -a 256 data/raw/creditcard.csv` returned the recorded value during the retrospective audit | [V] |

The distinction matters:

- A hash written in a document is a **claim**.
- A hash computed by code and stored next to the results **ties those results to one exact file**.
- Re-running the hash against the local file is a **verification**.

The project has all three, but **none of them comes from Phase 1**.

**Remaining gaps [V]:**

- `data/README.md` does not record the expected hash.
- There is no documented command or automated check that someone could run right after downloading the data.

## 5. Initial dependency and environment setup

**Implemented in Phase 1 [I]:** `requirements.txt` listed six packages with **no version pins**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter`, `ipykernel`.

**Documented in Phase 1 [D]:** the README said "Development environment: Python 3.14.6."

**What later evidence shows [V]:**

- The designated project environment `/Users/duc/Desktop/env/bin/python` runs **Python 3.13.6**.
- Commit `812c356` (Phase 2) replaced the Phase 1 statement with "Development environment verified for this project: Python 3.13.6 (`/Users/duc/Desktop/env/bin/python`)".
- The Phase 1 value therefore does not match the project environment. Which interpreter, if any, it described is not recorded.
- The Phase 1 environment statement should **not** be treated as a verified environment specification.

**Changes after Phase 1:**

| Change | When | Label |
|---|---|---|
| `scikit-learn` added to `requirements.txt`, also unpinned | `37d4521`, Phase 4a | [I] |
| Python 3.13.6 recorded in the README | `812c356`, Phase 2 | [D] |
| Exact library versions recorded in prose | pandas 2.3.1 (`docs/PROJECT_CONTEXT.md` and `docs/WORK_LOG.md`) and NumPy 2.3.2 (`docs/WORK_LOG.md` only) first appear in `c0ecd3c`, written after Phase 3, which reports them as the versions used in earlier Phase 2–3 runs. scikit-learn 1.9.1 first appears in `37d4521` (Phase 4a) in `docs/PROJECT_CONTEXT.md`, and in `docs/WORK_LOG.md` from `4b5b94b` (Phase 4b) | [D] |
| Exact versions captured at run time by code (`platform.python_version()` and each library's `__version__`) | Phase 4b results JSON (`4b5b94b`): Python, pandas and scikit-learn, no NumPy. Phase 4c results JSON (`cffc32c`): Python, NumPy, pandas and scikit-learn | [I] |

**Not established at any point [V]:**

- No pinned `requirements.txt` and no lock file.
- No `pyproject.toml`, `environment.yml` or `.python-version`.
- The designated environment, `/Users/duc/Desktop/env`, is outside the repository, so the repository's `.venv/` ignore rule does not apply to it.
- A Jupyter kernel named "Python (FraudGuard)" exists in the user's Jupyter configuration and points to that environment. It is not part of the repository, and no repository evidence shows when it was created.

**Why this matters [R]:**

- An unpinned `requirements.txt` lets `pip install -r requirements.txt` install whatever versions are newest *today*. Six months from now that may give a different scikit-learn, with different solver defaults or numerical behaviour.
- This is a concrete risk for FraudGuard. The Phase 4c thresholds lie within about 5×10⁻⁶ of 1.0 **[D]**, and the real-data regression tests assert exact counts **[I]**.
- Recording versions in prose and JSON tells a reader *what was used*. It does not *recreate* that environment.

## 6. What was actually implemented

Everything below is from commit `e4e866d` **[I]**:

- Directory skeleton, with 7 `.gitkeep` placeholders and 6 empty `__init__.py` files (Section 2).
- `.gitignore`, 42 lines (Section 3).
- `requirements.txt`, 6 unpinned packages (Section 5).
- `README.md`, 34 lines: overview, business problem, objectives, dataset link, technology stack, roadmap, status.
- `data/README.md`, 18 lines: dataset source, expected path, download steps, exclusion rationale.

Phase 1 implemented no code, tests, notebooks, data processing or analysis. The README stated this itself **[D]**.

## 7. What was documented

At the time **[D]**:

- `README.md`: project purpose, a one-sentence business problem, objectives, the planned dataset ("has not been downloaded or analyzed"), the Python version and dependencies, the five-phase roadmap, and status.
- `data/README.md`: dataset provenance, placement and the exclusion rationale.

Not documented at the time **[V]**:

- A Phase 1 report.
- A work-log entry. `docs/WORK_LOG.md` was first created in `c0ecd3c`, after Phase 3.
- Folder responsibilities in general. `data/README.md` covers only the dataset location in `data/raw/` and the exclusion of raw and processed data (Section 2).
- Decision rationale beyond the one-line data-exclusion note.
- Setup instructions: environment creation, install command, kernel registration.

## 8. What remains incomplete

As of 2026-11-08. The **Gap** column was checked against the repository during the audit **[V]**, except the part of the last row that comes from the mentor's project brief, a document outside the repository. The **Impact** and **Suggested remediation** columns are interpretation **[R]**.

| Gap | Impact | Suggested remediation (needs approval) |
|---|---|---|
| Dependencies are unpinned | Results may not reproduce in a fresh environment | Pin the verified versions, for example `scikit-learn==1.9.1`, or add a constraints or lock file |
| No Python version specification | Interpreter drift | Record `3.13.6` in a machine-readable form, such as `.python-version` or `pyproject.toml` |
| No setup instructions | A new user cannot rebuild the environment or kernel | Add a "Setup" section to the README |
| Dataset hash not in `data/README.md`, no post-download check | Users cannot confirm they have the same file | Add the expected SHA-256 and a verification command |
| No LICENSE file; dataset terms not recorded | Reuse terms are unclear for a public portfolio repository | Owner decision |
| Placeholder folders still empty (`src/visualization`, `app/components`, `artifacts`, `reports/figures`) | Folder structure suggests more than exists | Leave until used, or document them as reserved |
| No continuous integration | Tests run only by hand | Optional, later |
| The README roadmap lists no explainable AI, application or deployment stage (repository-verified). The mentor's project brief, which is not in the repository, includes explainable AI and application stages; that part cannot be verified from repository evidence | The README and the brief describe different scopes | Owner and mentor decision before the README rewrite |

## 9. Technical decisions and their rationale

| Decision | Evidence | Rationale | Alternatives |
|---|---|---|---|
| Keep data out of Git and document how to obtain it | `.gitignore`, `data/README.md` [I][D] | Stated reason: file size and distribution terms [D]. Also keeps raw data read-only [R] | Git LFS or DVC would version the data too, at the cost of more tooling and the same licensing question [R] |
| Use a `src/` package with workflow-stage subpackages | `e4e866d` tree [I] | Makes logic importable and testable, separate from notebooks [R]. Later phases followed this [I] | Notebook-only projects are quicker to start but hard to test and reuse [R] |
| Ignore model files by extension and in `artifacts/` | `.gitignore` [I] | Model files are large, binary and regenerable from code and data [R] | Track small models, or use a model registry [R] |
| Plain `requirements.txt` without versions | `e4e866d` [I] | No rationale recorded. Common in early scaffolding [R] | Pinned requirements, `pip-tools`, `uv`/`poetry` lock files, or conda `environment.yml` [R] |
| Python version stated in prose | README [D] | No rationale recorded [V]. Replaced in Phase 2 (`812c356`) with "verified … Python 3.13.6" [D] | A machine-readable version file [R] |

## 10. Lessons learned and future improvements

**Lessons** (interpretation **[R]**; the facts each lesson rests on are labelled inline):

1. **A claim in a document is not a verified environment.** The Phase 1 Python version was later replaced **[D]**. Versions should be captured by a command, such as `python --version` or `pip freeze`, not typed by hand **[R]**. Phases 4b and 4c capture versions at run time with `platform.python_version()` and each library's `__version__`, and write them into the results JSON **[I]**.
2. **Excluding data from Git creates a duty to fingerprint it.** The SHA-256 was first recorded in documentation in Phase 4a (`37d4521`, `docs/WORK_LOG.md`) **[D]**. Recording it in Phase 1, or as soon as the data was downloaded, would have covered Phases 2–3 as well **[R]**.
3. **Scaffolding creates expectations.** Empty folders such as `app/components/` and `src/visualization/` suggest planned capabilities. They should be described, or not created until needed.
4. **Write a short phase record at the time.** Reconstructing Phase 1 required reading Git diffs. A few lines in a work log on 2026-11-06 would have recorded the reasoning as well as the result.
5. **What went well:** `git log --diff-filter=RD` on `main` detects no renamed or deleted tracked files in any commit **[V]**, and no version of the dataset was found in the reachable history of the local refs examined (Section 3) **[V]**. Reading this as evidence that the Phase 1 structure was sound is a judgment **[R]**.

**Future improvements** (proposals only; none implemented, and each needs approval):

- Pin dependency versions to the verified environment and record the Python version in machine-readable form.
- Add setup instructions: environment creation, `pip install -r requirements.txt`, Jupyter kernel registration, and dataset download with hash verification.
- Record the expected dataset SHA-256 in `data/README.md`.
- Decide on a repository license and record the dataset's terms.
- Align the README roadmap with the agreed project scope.
