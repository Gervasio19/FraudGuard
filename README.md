# FraudGuard

FraudGuard is an educational portfolio project on machine learning for credit-card fraud detection. It uses a public, anonymized dataset. It is not a production system and makes no claim of operational effectiveness.

**Status (2026-11-11):**
- Phases 1–4 are complete; Phase 4 (model development and evaluation) is mentor-approved.
- Phase 5A, the experimental protocol, is approved and closed. The approved protocol is [docs/phase_5a_experimental_protocol.md](docs/phase_5a_experimental_protocol.md) (Revision 4), with implementation clarifications in its Appendix C.
- **Phase 5B-0** (protocol and documentation remediation, reproducibility infrastructure) is **complete and mentor-approved** (2026-11-11).
- No Phase 5 model has been fitted or scored, and no Phase 5 result exists.

Details: [Roadmap](#roadmap) and [docs/WORK_LOG.md](docs/WORK_LOG.md).

## Business problem

Payment fraud creates financial losses and operational costs. The project studies how transaction data can support fraud detection when fraud is very rare (about 0.17% of transactions). The Phase 5 use case is an **Analyst Review Queue**: ranking transactions so that analysts with limited capacity review the most suspicious first ([docs/business_understanding.md](docs/business_understanding.md)).

## Objectives

- A reproducible project structure and environment that a new user can set up from this README.
- Documented business understanding, data understanding and validation design.
- Fraud-detection models evaluated with imbalance-aware, time-respecting methods.
- Clear communication of results and limitations.

---

## Getting started (new users)

Run every command in this guide **from the repository root** (the folder that contains this `README.md`), unless a step says otherwise. Commands are given for macOS/Linux (bash or zsh) and Windows (PowerShell).

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.13** | The verified version is **3.13.6**. Other versions are not verified; the pinned NumPy and pandas releases need at least Python 3.11 |
| **Git** | To clone the repository |
| **A Kaggle account** | To download the dataset (free) |
| Disk space | About 150 MB for the dataset, plus about 1 GB for the virtual environment |

Check your Python version:

```bash
python3 --version        # macOS/Linux
```
```powershell
py -3.13 --version       # Windows (Python launcher)
```

### 2. Clone the repository

```bash
git clone https://github.com/Gervasio19/FraudGuard.git
cd FraudGuard
```

All later commands run from this `FraudGuard` folder.

### 3. Create a virtual environment

A virtual environment keeps the project's packages separate from the rest of your system. Create it in a folder named `.venv` inside the repository; Git ignores this folder.

```bash
python3.13 -m venv .venv          # macOS/Linux (or: python3 -m venv .venv if python3 is 3.13)
```
```powershell
py -3.13 -m venv .venv            # Windows
```

### 4. Activate the environment

You must activate it in every new terminal before running project commands.

| System | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

After activation, `python --version` should print `Python 3.13.x`. To leave the environment later, run `deactivate`.

*Windows note:* if PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

### 5. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the exact versions verified for this project: NumPy 2.3.2, pandas 2.3.1, scikit-learn 1.9.1, SciPy 1.17.1, Matplotlib 3.10.5 and seaborn 0.13.2.

*Optional:* to open the exploratory notebook in `notebooks/`, also run `python -m pip install -r requirements-notebook.txt`. The tests and the pipeline do not need this.

### 6. Download the dataset

The dataset is **not** in this repository: it is large, and its distribution is governed by Kaggle's terms. Download it yourself:

1. Sign in to Kaggle and open [Credit Card Fraud Detection (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Accept the dataset terms and click **Download**. You get a ZIP archive.
3. Extract `creditcard.csv` from the archive (150,828,752 bytes, about 144 MiB).

### 7–9. Place the file exactly here

- **File name:** `creditcard.csv`. Keep the exact name, in lowercase.
- **Location:** `data/raw/creditcard.csv`, relative to the repository root.

Expected structure after placement (abridged):

```
FraudGuard/
├── README.md
├── requirements.txt
├── requirements-notebook.txt
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── creditcard.csv      <- you add this file (never committed)
│   └── processed/              (empty; reserved)
├── docs/                       methodology, protocol, reports, policies
├── notebooks/                  01_data_understanding.ipynb
├── reports/                    result JSON files and the environment manifest
├── artifacts/                  (empty; fitted models would go here and are never committed)
├── src/                        project code (run as python -m src....)
└── tests/                      unittest suite
```

Do not open and re-save the CSV in a spreadsheet program. That changes the file and breaks the fingerprint check.

### 10. Verify the dataset fingerprint

```bash
python -m src.reproducibility.dataset
```

The command prints `OK: data/raw/creditcard.csv matches SHA-256 76274b69…`. It reads the file's bytes only, never its rows. Possible outcomes:

| Output | Exit code | Meaning and fix |
|---|---|---|
| `OK` | 0 | The file is identical to the project's copy |
| `MISSING` | 1 | No file at `data/raw/creditcard.csv`. Check the name and location (step 7–9) |
| `MISMATCH` | 2 | The file differs (re-saved, truncated, another version). Results will not reproduce. Download it again and do not edit it |

The expected SHA-256 is `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. You can also check it by hand with `shasum -a 256 data/raw/creditcard.csv` (macOS/Linux) or `Get-FileHash data\raw\creditcard.csv -Algorithm SHA256` (PowerShell).

### 11. Run the tests

```bash
python -m unittest discover -s tests -v
```

- The suite uses Python's built-in `unittest`; pytest is not needed.
- **Without the dataset**, the real-data test classes are skipped (reported as `skipped`), and everything else runs on small synthetic data.
- **With the dataset**, the real-data classes also run. They read the whole file, check the documented partition counts, and refit the Phase 4 Logistic Regression to compare against the committed results. This takes longer.
- To run one file, use for example `python -m unittest tests.test_validation_blocks -v`.

### 12–13. Run the pipeline and reproduce the experiments

The implemented pipeline is Phase 4. Each runner reads `data/raw/creditcard.csv`, fits on the training partition only, evaluates on the validation partition only (never on test), and writes a JSON summary:

```bash
python -m src.models.run_baselines             # Phase 4b: writes reports/phase4b_baseline_results.json
python -m src.models.run_threshold_analysis    # Phase 4c: writes reports/phase4c_threshold_results.json
```

- Both accept `--data PATH`, `--output PATH` and `--random-state N`. The project uses seed 42.
- **Reproducing:** in the verified environment, rerunning the runners regenerated byte-identical result files (Phase 4 audit, recorded in [docs/WORK_LOG.md](docs/WORK_LOG.md)). Compare your output with `git diff reports/`.
- Fields such as the recorded Python or package versions differ if your environment differs. Identical metric values are expected only with the pinned versions.
- Phase 5 experiments are **not implemented yet**. They will follow the approved protocol, and their commands will be added here when they exist.

To record your own environment, run `python -m src.reproducibility.manifest environment`. It rewrites `reports/environment_manifest.json` with your Python and package versions, the dataset fingerprint status, the Git commit and the frozen seeds. It contains no absolute paths.

### 14. Generated reports and artifacts

| Path | What it is | In Git? |
|---|---|---|
| `reports/phase4b_baseline_results.json` | Phase 4b validation metrics (majority baseline, Logistic Regression) | yes |
| `reports/phase4c_threshold_results.json` | Phase 4c threshold curves and window transfer (about 9.5 MB) | yes |
| `reports/environment_manifest.json` | Environment, dataset fingerprint, commit, seeds | yes |
| `docs/templates/phase5_freeze_manifest.template.json` | Empty template of the Phase 5 freeze manifest (no results) | yes |
| `reports/figures/` | Figures (none yet) | yes, when added |
| `artifacts/` | Fitted models, if ever saved | **no** |
| `data/raw/`, `data/processed/` | Datasets | **no** |

The rules are in [docs/result_storage_policy.md](docs/result_storage_policy.md): compact JSON only, target ≤ 5 MB per file, hard limit 10 MB.

### 15. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not in the repository root, or you ran a file directly. `cd` to the root and use `python -m src....` |
| `ModuleNotFoundError: No module named 'pandas'` (or numpy, sklearn) | The virtual environment is not active, or the install failed. Activate it (step 4) and rerun step 5 |
| `pip` cannot find a pinned version, or tries to compile from source | Your Python version is probably not 3.13. Create the environment with Python 3.13 (steps 1 and 3) |
| `python3.13: command not found` | Install Python 3.13 from python.org, or use the interpreter that reports 3.13 (`python3 --version`) |
| `MISSING: data/raw/creditcard.csv not found` | The file is absent, misnamed (for example `creditcard (1).csv`) or in the wrong folder. Fix the name and location |
| `MISMATCH …` | The file was modified or is a different version. Download it again; do not open and save it in Excel |
| Many tests reported as `skipped` | The dataset is missing. Expected without the data; add it to run the real-data tests |
| PowerShell will not activate `.venv` | See the Windows note in step 4 |
| Results differ slightly from the committed JSON | Check that the dataset fingerprint is `OK` and that package versions match `requirements.txt`. Other library versions can change numerical results |

---

## Methodology in brief

- **Primary validation:** a chronological 70/15/15 split by `Time` (train ≤ 132,928 s; validation 132,929–151,328 s; test ≥ 151,329 s), keeping equal timestamps together. The **test partition is untouched until a single final evaluation**, after every decision is frozen.
- **Features:** `V1`–`V28` and `Amount`. `Time` is used for splitting only.
- **Phase 4:** a majority-class baseline and a class-weighted Logistic Regression, followed by threshold analysis with a Window 1 → Window 2 transfer check.
- **Phase 5 (approved protocol, not yet run):**
  - Logistic Regression is the default model. One HistGradientBoosting challenger replaces it only if both pre-registered criteria, E1 and E2, pass.
  - Sigmoid calibration on a temporal holdout taken from the end of training. If calibration fails technically, the model keeps its uncalibrated scores; the cross-fitted ensemble (C-F) is a retrospective sensitivity only, never a fallback.
  - Paired block bootstrap; fixed-K and time-based review capacity with 1/2/4-hour SLAs; hypothetical review-cost scenarios; a Window 1 → Window 2 decision replay.

## Documentation index

| Document | Content |
|---|---|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Architecture, confirmed methodology, approved decisions, current state |
| [docs/WORK_LOG.md](docs/WORK_LOG.md) | Chronological history of every phase and recorded results |
| [docs/phase_5a_experimental_protocol.md](docs/phase_5a_experimental_protocol.md) | Approved Phase 5 experimental protocol (Revision 4) with Appendix C clarifications |
| [docs/validation_strategy.md](docs/validation_strategy.md) | Split design, leakage controls, partition usage policy |
| [docs/business_understanding.md](docs/business_understanding.md) | Business framing and evaluation considerations |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Reference for every implemented module |
| [docs/result_storage_policy.md](docs/result_storage_policy.md) | What is tracked in Git; result-file size limits |
| [docs/reports/](docs/reports/) | Phase reports (Phases 1–4 retrospective audits; Phase 5B-0 completion report) |
| [data/README.md](data/README.md) | Dataset source, placement, integrity and policy |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Placeholder, written only when the project is complete |

## Roadmap

1. Project initialization: complete.
2. Business understanding and dataset exploration: complete.
3. Data preparation and validation design: complete.
4. Model development and evaluation (4a chronological split, 4b baselines, 4c threshold analysis): complete, mentor-approved.
5. Challenger model, calibration, and cost- and capacity-aware evaluation for an Analyst Review Queue:
   - **5A** experimental protocol: approved and closed (2026-11-09).
   - **5B-0** protocol and documentation remediation, reproducibility infrastructure: complete, mentor-approved (2026-11-11).
   - **5B-1 onward** implementation and experiments: not started; each stage starts only on explicit approval.
6. Results communication and project presentation: not started.

## Dataset and license note

The data is the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset: 284,807 transactions from two days in September 2013, 492 of them fraud. `V1`–`V28` are PCA components. The dataset is used under Kaggle's terms and is never committed or redistributed by this project.
