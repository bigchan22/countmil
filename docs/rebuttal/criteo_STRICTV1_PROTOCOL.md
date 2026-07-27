# Criteo Strict-v1 Protocol

Experiment description: **LLP-Bench-style Criteo feature-bag experiment**.

This is not an exact reproduction of the complete LLP-Bench benchmark. It uses
the public RecZoo `Criteo_x1` preprocessing and its train/validation/test CSV
partitions, which differ from LLP-Bench's original five-fold preprocessing.

## Fixed Definition

- Protocol tag: `criteo_x1_c4_c11_k128_strictv1`
- Dataset: Hugging Face `reczoo/Criteo_x1`
- Grouping key: `C4`, `C11`
- Bag size: `128`
- Train bags: `12000`
- Validation bags: `3000`
- Test bags: `5000`
- Seeds: `0, 1, 2`
- Methods: `DLLP-BCE`, `DLLP-MSE`, `EasyLLP`, `FS-Conv`

The grouping key and bag size were selected before inspecting FS-Conv test
performance because C4+C11 and bag size 128 appear in the published LLP-Bench
fixed-size feature-bag evaluation.

## Splits

- `train.csv`: training bags only.
- `valid.csv`: aggregate-only validation bags only.
- `test.csv`: final evaluation only, after checkpoint selection.

No per-instance click labels are exposed by training or validation loaders.
Final test labels are stored separately and loaded only for final evaluation.

## Bag Construction

For each split and seed:

1. Rows are grouped by `(C4, C11)`.
2. Rows inside each group are ordered by deterministic MD5 hash of protocol,
   split, seed, grouping key, and original row ID.
3. Ordered groups are chunked into non-overlapping bags of 128 rows.
4. Residual chunks smaller than 128 are discarded.
5. Candidate bags are selected by deterministic MD5 hash of protocol, split,
   seed, grouping key, and chunk index.
6. Selection is label-blind: click labels, click counts, and model predictions
   are not used.

Click labels are used offline only to compute one aggregate count per bag.

## Model And Training

Every method uses the same instance model:

- categorical embeddings, embedding dimension 8;
- 13 numerical features concatenated with flattened categorical embeddings;
- MLP hidden layers 256 and 128 with ReLU;
- one scalar click logit, sigmoid probability.

Training settings:

- optimizer: Adam;
- learning rate: `1e-3`;
- weight decay: `1e-6`;
- bag batch size: `32`;
- maximum epochs: `12`;
- global gradient clipping: `5`;
- early-stopping patience: `3` validation evaluations.

Checkpoint selection is common across methods:

- primary: validation expected-count MAE;
- tie-breaker: method-native aggregate validation objective.

Validation does not compute ROC-AUC, PR-AUC, instance accuracy, instance log
loss, Brier score, ECE, or any other statistic requiring individual validation
click labels.

## Common Final Metrics

After the selected checkpoint is loaded, the final test split is evaluated once.

Instance metrics:

- ROC-AUC;
- PR-AUC;
- binary log loss;
- Brier score;
- ECE with 15 equal-width bins;
- accuracy at threshold 0.5 as secondary.

Aggregate metrics:

- expected count MAE;
- rounded expected-count accuracy;
- exact Poisson-binomial PMF-mode accuracy;
- exact Poisson-binomial aggregate NLL.

The post-hoc Poisson-binomial metrics are computed identically for every method
from its instance probabilities.
