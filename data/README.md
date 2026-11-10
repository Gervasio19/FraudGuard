# Dataset

## Source

The project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`mlg-ulb/creditcardfraud`). It contains 284,807 card transactions from two days in September 2013, 492 of them fraudulent. Columns: `Time` (seconds since the first transaction), `V1`–`V28` (anonymized PCA components), `Amount` and `Class` (1 = fraud).

## Expected file

| Item | Value |
|---|---|
| File name | `creditcard.csv` (exact, lowercase) |
| Location | `data/raw/creditcard.csv`, relative to the repository root |
| Size | 150,828,752 bytes (about 144 MiB) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |

The code finds the file through `src/config.py` (`CREDITCARD.path`), which is built relative to the repository, so no machine-specific path is needed.

## Obtain the dataset

1. Sign in to Kaggle and open the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Accept the dataset terms and download it with Kaggle's download option. You get a ZIP archive.
3. Extract `creditcard.csv` into `data/raw/`.

## Integrity check (required)

From the repository root:

```bash
python -m src.reproducibility.dataset
```

The check computes SHA-256 over the file bytes and compares it with the value above. It never parses rows, so it reads no labels.

| Result | What happens |
|---|---|
| `OK` (exit 0) | The file is the project's dataset. All recorded counts and results apply |
| `MISSING` (exit 1) | No file at `data/raw/creditcard.csv`. The real-data tests are skipped, and the runners fail when they try to read the file |
| `MISMATCH` (exit 2) | The file is present but differs. The documented partition counts, test expectations and results are **not** valid for it. Download the original again and do not edit it |

Phase 5 runs will check the fingerprint at start and abort on a mismatch (protocol Section 12.2).

## Policy

- The dataset is **never committed** to Git. `.gitignore` excludes everything in `data/raw/` and `data/processed/` except the `.gitkeep` placeholders.
- It is never uploaded, redistributed or published as a downloadable artifact by this project. Each user downloads it from Kaggle under Kaggle's terms.
- `data/raw/creditcard.csv` must remain unchanged. No code writes to it.
- `data/processed/` is reserved; no derived dataset exists.
