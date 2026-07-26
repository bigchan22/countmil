# Matched Strict-v3 LLP-PVC Diagnostics

Source rows: strict-v3 corrected fixed-bag CIFAR-10 summaries. Hidden labels were used only in final test evaluation in those runs.

| Seed | LR | Selected epoch | Best val NLL | Best val count MAE | Test acc. | Macro-F1 | Test count MAE | Test composite NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0005 | 5 | 6.069 | 6.094 | 0.509 | 0.449 | 6.141 | 6.091 |
| 1 | 0.0005 | 2 | 6.714 | 6.886 | 0.135 | 0.101 | 7.090 | 6.873 |
| 2 | 0.0005 | 3 | 6.774 | 6.883 | 0.283 | 0.217 | 6.824 | 6.773 |

## Prediction Diagnostics

The strict-v3 frozen Git evidence includes top-level per-seed JSON summaries and
bag manifests. It does not include the larger run directories, per-epoch curve
files, unique-image prediction dumps, checkpoints, or feature tensors; those
were intentionally excluded from Git/artifact-freeze packaging. Therefore the
following diagnostics cannot be reconstructed from this branch alone and should
be treated as unavailable, not as measured zero-valued quantities:

- mean and standard deviation of sigmoid probabilities;
- softmax entropy;
- predicted class frequency;
- saturated-probability fraction;
- count-loss floor-hit rate;
- training and validation curves.

Available evidence from the frozen summaries is limited to selected epoch,
aggregate validation metrics, final aggregate test metrics, final unique-image
instance metrics, split hashes, feature hashes, and bag-manifest hashes.

### Seed 0
- Additional probability/curve diagnostics: unavailable from committed
  strict-v3 evidence.

### Seed 1
- Additional probability/curve diagnostics: unavailable from committed
  strict-v3 evidence.

### Seed 2
- Additional probability/curve diagnostics: unavailable from committed
  strict-v3 evidence.
