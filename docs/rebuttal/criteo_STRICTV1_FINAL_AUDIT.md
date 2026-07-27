# Criteo Strict-v1 Final Audit

Status: **completed**

This experiment is an **LLP-Bench-style Criteo feature-bag experiment** using
public RecZoo `Criteo_x1`. It is not an exact full LLP-Bench reproduction
because the public RecZoo preprocessing and train/validation/test partitions
differ from LLP-Bench's original five-fold preprocessing.

## Completion Matrix

All requested method/seed jobs completed.

| Method | Seeds |
|---|---|
| DLLP-BCE | 0, 1, 2 |
| DLLP-MSE | 0, 1, 2 |
| EasyLLP | 0, 1, 2 |
| FS-Conv | 0, 1, 2 |

No failed or partial run entered the summary tables.

## Data And Bags

- Dataset ZIP SHA256 matched the expected value.
- `train.csv`, `valid.csv`, and `test.csv` MD5 hashes matched the expected
  values.
- Row counts matched the expected RecZoo Criteo_x1 counts.
- Fixed bags used grouping key `C4,C11`, bag size 128, and deterministic
  label-blind MD5 ordering/selection.
- All requested bags were produced for every seed:
  - train: 12000 bags;
  - validation: 3000 bags;
  - test: 5000 bags.
- Train, validation, and test use the public dataset's disjoint CSV partitions.
- Within each split and seed, selected row IDs are unique.

## Leakage Checks

- Training loaders expose numerical features, categorical features, row IDs,
  and aggregate counts only.
- Validation loaders expose numerical features, categorical features, row IDs,
  and aggregate counts only.
- Train/validation shards do not contain the `hidden_labels` key.
- Validation metrics are aggregate-only:
  - validation expected-count MAE;
  - method-native aggregate objective.
- No validation ROC-AUC, PR-AUC, instance accuracy, log loss, Brier score, ECE,
  or other per-instance click metric was computed.
- Final test hidden labels are loaded only after checkpoint selection.

## Summary Construction

- Tables are reconstructed from the 12 top-level per-run JSON summaries under
  `results/rebuttal/4090_criteo_strictv1/`.
- Mean and uncertainty use seeds 0,1,2 only.
- `+/-` denotes sample standard deviation with `ddof=1`.
- Aggregate metrics are common post-hoc metrics computed from each method's
  instance probabilities using the same exact Poisson-binomial code.
- Checkpoints and raw probability tensors are excluded from Git and recorded in
  `artifacts/rebuttal/criteo_STRICTV1_EXTERNAL_ARTIFACTS.json`.

## Result Files

- `results/rebuttal/4090_criteo_strictv1/run_inventory.csv`
- `results/rebuttal/4090_criteo_strictv1/instance_table.md`
- `results/rebuttal/4090_criteo_strictv1/instance_table.csv`
- `results/rebuttal/4090_criteo_strictv1/aggregate_table.md`
- `results/rebuttal/4090_criteo_strictv1/aggregate_table.csv`
- `results/rebuttal/4090_criteo_strictv1/failures.md`
- `docs/rebuttal/criteo_STRICTV1_FINAL_REPORT.md`
