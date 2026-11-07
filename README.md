# FraudGuard

FraudGuard is an educational portfolio project exploring machine learning for financial fraud detection. The project is under development. Phases 1–3 are complete. Phase 4 (model development) is in progress: chronological partitioning (4a) is implemented, and two baseline models (4b) have been implemented and evaluated on validation data. Their methodology decisions are approved, and Phase 4b completion awaits mentor approval.

## Business problem

Payment fraud creates financial losses and operational costs. This project will explore how transaction data can support fraud detection while accounting for the challenges of highly imbalanced data.

## Objectives

- Establish a reproducible project structure and development environment.
- Document the business problem, inspect the selected public credit card fraud dataset, and finalize validation design.
- Develop and evaluate fraud detection approaches in subsequent phases.
- Present findings and project outputs clearly.

## Dataset

The project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). A local copy is present at `data/raw/creditcard.csv`; it is excluded from Git. See [data/README.md](data/README.md) for source and placement details.

## Technology stack

Development environment verified for this project: Python 3.13.6 (`/Users/duc/Desktop/env/bin/python`). Dependencies (`requirements.txt`): pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter, and ipykernel.

## Usage

Run these from the repository root, with `data/raw/creditcard.csv` in place:

```bash
python -m unittest discover -s tests -v   # run the test suite
python -m src.models.run_baselines        # run the baseline experiments (writes reports/phase4b_baseline_results.json)
```

See [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) for the components, and [docs/WORK_LOG.md](docs/WORK_LOG.md) for recorded results.

## Roadmap

1. Project initialization (complete).
2. Business understanding and dataset exploration (complete).
3. Data preparation and validation design (complete).
4. Model development and evaluation (in progress: 4a chronological split done; 4b baselines awaiting completion approval).
5. Results communication and project presentation.

## Status

**Under development — Phase 4b: Baseline Modeling, awaiting completion approval.** A majority-class baseline and a class-weighted Logistic Regression have been evaluated on validation data only, under a primary chronological split and a secondary duplicate-aware stratified random split (70/15/15, seed 42). They are reported with ranking metrics (average precision, ROC-AUC) and diagnostic metrics at the default 0.5 rule; 0.5 is not an operating threshold. No feature engineering, hyperparameter tuning, threshold selection, or test-set evaluation has been performed.
