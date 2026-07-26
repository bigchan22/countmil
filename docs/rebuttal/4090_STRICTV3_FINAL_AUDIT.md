# 4090 Strict-v3 Final Audit

This file is populated by strict-v3 completion checks. Current source state
contains the strict protocol, masked-backbone utility, stable Gaussian
integer-bin evaluation, strict MNIST/CIFAR trainers, and launch/summarization
scripts.

Strict-v3 summaries must include only runs whose config records:

- protocol tag `strictv3_train_holdout_val_official_test_no_hidden_val`;
- full Git SHA;
- clean-run config hash;
- train/validation/test split hashes;
- train/validation/test manifest hashes;
- aggregate-only validation selection metric.

Rows invalidated by test-set checkpoint selection or validation/test image-pool
overlap must not enter the strict-v3 tables.
