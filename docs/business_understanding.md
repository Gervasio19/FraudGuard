# Business Understanding

## Business problem and motivation

Payment card fraud creates direct financial losses and adds work for issuers, payment processors, merchants, and customers. A fraud monitoring process must identify suspicious transactions soon enough to support a useful response while allowing legitimate purchases to proceed. FraudGuard is an educational project that studies how transaction data can support that detection task.

The selected public dataset contains European cardholder transactions from two days in September 2013. It is a historical, anonymized research dataset, not a live production feed. Its narrow time window, age, and transformed feature set limit how directly results can be applied to today's payment operations.

## Fraud detection use cases

A transaction scoring system could support several operational workflows:

- **Real-time authorization support:** flag a transaction for approval, decline, or additional verification before completion.
- **Analyst review queue:** rank transactions for investigation when immediate automated action is not appropriate.
- **Customer verification:** trigger a step-up check, such as contacting the cardholder, when a transaction appears suspicious.
- **Portfolio monitoring:** summarize alerts and detected cases to help teams track fraud activity and review operational rules.

These are example use cases for framing the problem. The dataset itself does not describe an issuing bank's workflow or establish which action should be taken for any transaction.

## Costs of errors

- **False negative:** a fraudulent transaction is treated as legitimate. This can lead to unrecovered transaction loss, follow-on fraud, disputes, and customer harm.
- **False positive:** a legitimate transaction is flagged as fraud. This can cause a declined purchase, unnecessary verification, customer frustration, lost merchant sales, and investigation workload.

The costs depend on transaction amount, recovery rates, downstream fraud, review capacity, and customer impact. Those quantities are not supplied in this dataset, so a monetary cost function cannot yet be specified credibly.

## Why accuracy is misleading

The target is highly imbalanced: fraud is a very small fraction of all transactions. A classifier that labels every transaction as legitimate would achieve about 99.83% accuracy on the inspected file while detecting none of the 492 fraud cases. Accuracy therefore obscures whether the rare class is being found and how many legitimate transactions are incorrectly flagged. Evaluation should make class-specific outcomes and operating thresholds visible.

## Objectives and scope

The project objectives are to:

1. Understand the dataset's provenance, structure, target, and basic quality before modeling.
2. Build a reproducible educational workflow for fraud classification.
3. In later phases, compare suitable approaches using evaluation that reflects imbalance and the tradeoff between missed fraud and false alerts.
4. Communicate findings and limitations clearly.

Phases 2 and 3 document business understanding, basic data understanding, and the validation design. At that stage, the scope did not include model training or advanced exploratory analysis. The project scope has since expanded to model development and evaluation in later phases: Phase 4 trained and evaluated baseline models on validation data, and Phase 5 (approved protocol, not yet run) extends this work. The project does not include production deployment, live payment decisions, or claims of operational effectiveness. The dataset contains anonymized PCA components rather than the original transaction attributes, which also limits interpretation.

## Proposed evaluation considerations

The finalized primary design is a chronological 70/15/15 train/validation/test split, with equal `Time` values kept together and duplicate feature groups kept within a single partition. A duplicate-group-aware stratified random split is a secondary benchmark only. The dataset spans only two days, so validation and test contain few fraud cases and resulting metrics may be sensitive to individual examples; see [validation strategy](validation_strategy.md) for boundaries and counts.

For model comparisons, report at least:

- **Recall (fraud detection rate):** the share of actual fraud cases detected; useful for understanding missed fraud.
- **Precision:** the share of flagged transactions that are fraud; useful for understanding alert quality and review burden.
- **Precision-recall curve and average precision:** summarize the precision/recall tradeoff under rare positive cases.
- **Confusion matrix at a stated threshold:** makes false positives and false negatives explicit.
- **Alert volume and threshold behavior:** translate a threshold into the number of cases requiring action.
- **ROC-AUC, with context:** may be reported as a secondary ranking measure, alongside precision-recall results and class-specific counts.

Threshold selection should reflect an agreed operational capacity and relative error costs. Because the dataset does not provide those costs or a deployment workflow, no classification threshold has been selected; one metric alone should not be treated as business value. Any later evaluation should state the split strategy, threshold, and limitations, and compare against a simple baseline.

## Phase 5 design note (approved design recorded 2026-11-08; protocol approved 2026-11-09; not yet run)

The Phase 5 briefing chose the **Analyst Review Queue** as the use case for operational evaluation. The approved evaluation design:

- cost-sensitive evaluation with break-even analysis;
- review capacity modeled both as fixed-K capacity and as time-based capacity;
- Precision@K and Recall@K;
- SLA scenarios of 1, 2 and 4 hours.

These are planned methods, not results. *Superseded (2026-11-09):* K values, the definition of time-based review capacity and the cost ratios were "awaiting detailed specification". The approved protocol (`docs/phase_5a_experimental_protocol.md`) now specifies them as **hypothetical scenario grids**, not facts about any real operation:
- fixed-K: Section 9.2;
- the online priority queue with SLA expiry: Section 9.3;
- review-cost ratios R ∈ {5 … 500}: Section 8.2.

The amount-aware decision policy is excluded; amount-aware cost accounting is kept as a sensitivity. No operational threshold has been set, and the statement above that a monetary cost function cannot yet be specified credibly still holds. The full list of Phase 5 decisions is in `docs/PROJECT_CONTEXT.md`.

## Hypothetical-scenario disclosure (project-wide requirement, 2026-11-14)

Every business or operational scenario in this project uses **assumed** parameters. No real investigation cost, financial loss or operational capacity has been provided. Every artifact therefore separates four evidence classes:

| Evidence class | Meaning | Hypothetical? |
|---|---|---|
| `observed_dataset` | Read from the dataset: rows, frauds, timestamps, amounts | no |
| `model_generated` | Computed from fitted models and the data: scores, AP, calibration metrics, bootstrap intervals, the champion gates | no |
| `assumed_scenario_input` | An assumed operational parameter: review-cost ratio R, minutes per review, review capacity, SLA | **yes** |
| `simulated_outcome` | Computed from model results and assumed inputs: break-even ratios, costs in review units, analyst-hours | **yes** |

Simulated savings, hypothetical costs and assumed capacities are never actual business outcomes and must not be presented as such.

- The requirement applies to result files, reports, documentation, notebooks, figures and any presentation.
- It is enforced by code and tests: `src/evaluation/evidence_labels.py` and `tests/test_hypothetical_disclosure.py` (see `docs/TECHNICAL_DOCUMENTATION.md`).
- The protocol's `[H]` label marks the same hypothetical inputs in `docs/phase_5a_experimental_protocol.md`.

## Dataset reference

The local dataset documentation points to the Kaggle **Credit Card Fraud Detection** dataset. The publisher's description says the data covers two days and contains 284,807 transactions with 492 frauds; `V1`–`V28` are principal components obtained with PCA, while `Time` and `Amount` are not PCA-transformed. `Class` is the fraud target (1 for fraud, 0 otherwise). See the [dataset source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and the repository's [dataset instructions](../data/README.md).
