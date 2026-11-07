# Validation Strategy

## Primary design: chronological holdout

Use the sorted `Time` values to assign earlier transactions to training, then validation, then testing. The finalized 70/15/15 boundaries were selected by cumulative row count, choosing complete timestamps closest to each target. All records with the same `Time` remain in one partition. Training includes `Time <= 132928`; validation includes `132929 <= Time <= 151328`; testing includes `Time >= 151329`.

| Partition | Time range (seconds) | Records | Non-fraud (`Class=0`) | Fraud (`Class=1`) | Fraud rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0–132,928 | 199,364 | 198,980 | 384 | 0.1926% |
| Validation | 132,929–151,328 | 42,722 | 42,666 | 56 | 0.1311% |
| Test | 151,329–172,792 | 42,721 | 42,669 | 52 | 0.1217% |

The file has 284,807 records and 492 fraud cases overall. The count-based boundaries are close to 70/15/15, while equal timestamps remain intact. These boundaries are the finalized project design. Validation and test have only 56 and 52 fraud cases, so estimates—especially recall, precision, and threshold-specific rates—will be sensitive to a small number of cases. This limitation should be stated when results are reported; the project has chosen to retain the three-part 70/15/15 design.

## Duplicate-group leakage prevention

Duplicates are retained in the dataset. Exact feature groups (the 30 columns excluding `Class`) contain 773 repeated groups and 1,081 excess rows. Repeated feature groups have consistent labels and none crosses the finalized partitions. This was checked by grouping identical feature vectors and counting distinct partition assignments.

For any alternate split, keep each identical feature group together. For the chronological design, assigning by `Time` already keeps the observed duplicate groups together because each repeated feature group has the same `Time`. For the random benchmark, split by duplicate-group identity rather than individual rows; otherwise copies could enter training and evaluation, inflating performance. Do not drop records as part of this validation design.

## Secondary benchmark: stratified random split

A stratified random benchmark can preserve the overall class proportions and distribute more of the scarce positive cases across evaluation partitions, making it useful for comparison with prior work and for a less time-sensitive estimate. It does not simulate scoring future transactions, can mix temporal patterns across partitions, and can overstate generalization when neighboring or repeated observations are shared. Keep duplicate groups together, report the random seed and exact split procedure, and interpret results as secondary to chronological evaluation.

## Leakage risks and mitigation

- **Future-to-past leakage:** random splits can train on later transactions and evaluate on earlier ones. Use chronological order for the primary estimate.
- **Timestamp ties:** splitting rows at an arbitrary index can divide transactions with the same timestamp. Choose boundaries on unique `Time` values and document inclusive ranges.
- **Duplicate leakage:** identical feature vectors in different partitions can expose evaluation examples during training. Group duplicate feature vectors for any non-chronological split.
- **Preprocessing leakage:** in later phases, fit imputers, scalers, feature selection, resampling, and any learned transformations using training data only; apply the fitted transformations to validation/test. Do not use validation or test outcomes to tune model choices or thresholds.
- **Label availability and prediction timing:** the dataset does not document when labels become known relative to a transaction decision. Confirm the intended prediction point before modeling and exclude information unavailable at that point if identified.

## Dataset limitations and decision points

`Time` runs from 0 to 172,792 seconds (about 48 hours) and is nondecreasing in the file. The dataset has only two days of historical activity, so a chronological holdout tests a narrow later period, not robust performance across seasons, policy changes, or current fraud behavior. There are 492 fraud cases total; the validation and test sets are small in positive cases. Consider a larger time holdout or rolling-origin evaluation if the project needs a more stable temporal assessment, while recognizing that multiple folds from this short window are not independent datasets.

The 70/15/15 chronological partition is confirmed. Keep the classification threshold undecided until evaluation criteria and operational costs or review capacity have been agreed.


## Phase 3 completion summary

The Phase 3 validation design is finalized: use the chronological 70/15/15 partitions in the table above, keep equal timestamps together, and retain all records including duplicate rows. Identical feature groups must remain within one partition; the confirmed chronological boundaries satisfy this constraint for the inspected dataset. Use a duplicate-group-aware stratified random split only as a secondary benchmark. No classification threshold has been selected. No records were removed, no feature engineering was performed, and no models were trained. The principal limitation is the small number of fraud cases in validation and test and the dataset's approximately 48-hour coverage.

## Implementation note (Phase 4b, configuration approved 2026-11-07)

The secondary benchmark is implemented in `src/data/random_split.py`. Rows are grouped by identical values in every column except `Class`. Groups must have a single label. Within each class, the groups are shuffled with `numpy.random.default_rng(42)` and allocated by cumulative row count to 70/15/15 train/validation/test. On the inspected data, this gives train 199,364 rows (344 frauds), validation 42,722 (74) and test 42,721 (74), with no duplicate group crossing partitions. All original rows are preserved. The design decisions above are unchanged. Results from this split are reported separately from the chronological results (see `docs/WORK_LOG.md`).
