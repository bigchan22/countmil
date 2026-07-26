# 4090 Strict-v3 Protocol

Protocol tag: `strictv3_train_holdout_val_official_test_no_hidden_val`.

The official training split is deterministically divided into train and
validation image pools. The official test split is reserved exclusively for
final evaluation after aggregate-only checkpoint selection. Validation does not
compute hidden instance accuracy, hidden AUC, hidden digit accuracy, or
pseudo-label quality.

All strict-v3 bag manifests are explicit saved artifacts with canonical content
hashes over tensor keys, dtype, shape, bytes, and scalar metadata. Methods
within a seed share the same train, validation, and final test bag manifests.

Historical rows using official test data for validation selection or validation
image pools are not eligible for strict-v3 summaries.
