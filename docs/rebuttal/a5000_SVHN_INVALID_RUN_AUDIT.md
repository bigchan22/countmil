# A5000 SVHN Invalid Run Audit

The SVHN jobs launched at `20260726T064205Z` are not used for final rebuttal
tables.

Reasons:

- Validation bags were sampled from the official SVHN test archive with a
  different deterministic bag seed from final test bags. This avoided identical
  bag reuse but did not keep validation and test image pools disjoint.
- Validation evaluation logged hidden instance digit accuracy at every epoch.
  The checkpoint-selection code used aggregate validation loss only, but the
  strict rebuttal protocol should not expose hidden validation instance metrics
  during training.

The completed artifacts are left on disk for provenance under
`results/rebuttal/a5000_svhn_dependence/`, but strict reruns use protocol tag
`strictv2_train_holdout_val_no_hidden_val_metrics` and summary filenames
containing `strictv2`.

The strict rerun changes:

- train: deterministic 85% split of the official SVHN train archive;
- validation: deterministic 15% holdout from the official SVHN train archive;
- test: official SVHN test archive;
- validation metrics: aggregate-only, with no hidden instance metrics computed
  or logged;
- final test metrics: aggregate metrics plus hidden instance digit accuracy.
