# FraudGuard

FraudGuard is an educational portfolio project exploring machine learning for financial fraud detection. The project is under development. Phases 1–3 are complete. Phase 4 (model development) is complete and mentor-approved: chronological partitioning (4a) is implemented, two baseline models (4b) have been implemented and evaluated on validation data, and threshold analysis with a temporal stability check and a descriptive common-reference threshold comparison (4c) is implemented. Retrospective reports for Phases 1–4 are in `docs/reports/`. The Phase 5 methodology and design briefing is complete and its decisions are approved; Phase 5 has not been implemented.

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
python -m src.models.run_threshold_analysis  # run the threshold analysis (writes reports/phase4c_threshold_results.json)
```

See [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) for the components, and [docs/WORK_LOG.md](docs/WORK_LOG.md) for recorded results.

## Roadmap

1. Project initialization (complete).
2. Business understanding and dataset exploration (complete).
3. Data preparation and validation design (complete).
4. Model development and evaluation (complete, mentor-approved: 4a chronological split, 4b baselines, 4c threshold analysis).
5. Challenger model, calibration and cost- and capacity-aware evaluation for an analyst review queue (design approved; not implemented).
6. Results communication and project presentation.

## Status

**Current status — Phase 4 (Model Development & Evaluation) is complete, mentor-approved (2026-11-08) and committed (`1f510f7`). The Phase 5 design decisions are approved; no Phase 5 code, experiments or results exist yet.**

- **Phase 4b:** a majority-class baseline and a class-weighted Logistic Regression were evaluated on validation data only, under a primary chronological split and a secondary duplicate-aware stratified random split (70/15/15, seed 42).
- **Phase 4c:** the Logistic Regression's validation scores were analysed across every unique threshold. An F1-maximizing threshold chosen on the first half of validation was then tested, unchanged, on the second half.

- **Phase 5 (planned, approved design):** a HistGradientBoosting challenger, model calibration, paired bootstrap comparison, and cost-sensitive (break-even) and review-capacity evaluation (fixed-K and time-based, Precision@K and Recall@K, 1/2/4-hour SLA scenarios) for an Analyst Review Queue. The full list is in [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

These thresholds are analytical; no operational threshold has been chosen. No feature engineering, hyperparameter tuning, or test-set evaluation has been performed. Results and their limitations are in [docs/WORK_LOG.md](docs/WORK_LOG.md).
