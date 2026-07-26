# Criteo Failures

## NOT_LAUNCHED: `criteo_llpbench_protocol_probe`

The experiment was stopped before data download or training.

Concrete blockers:

- Missing local files: `train_x.txt`, `train_y.txt`, `train_i.txt`,
  `preprocessed_criteo.csv`, and LLP-Bench fixed-size Criteo bag files.
- The official LLP-Bench Criteo training code monitors instance-labeled
  `val_auc` for early stopping, which violates the rebuttal protocol.
- The official fixed-size feature-bag training path expects a test CSV from the
  random-bag path, while the fixed-size feature-bag creator only constructs the
  training bag `.ftr` file.

No failed model run exists because no model process was launched.
