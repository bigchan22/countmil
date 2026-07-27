# 4090 SVHN Masked Strict-v3 Final Audit

- Protocol version: `strictv3_train_holdout_val_official_test_no_hidden_val_masked_resnet_bag_forward`.
- Defect repaired: padded zero images in 5-D variable-length bags are not forwarded through ResNet-18/BatchNorm.
- Training and bag-level validation/test evaluation use `forward_valid_instances(model, instances, mask)` followed by softmax.
- Flat unpadded unique-instance final-test evaluation remains unchanged.
- Validation metrics are aggregate-only; hidden instance labels are evaluated only after checkpoint selection on the final test split.
- No old checkpoints or old result summaries are reused in this table.
- All rows are reconstructed from raw final-test prediction files.

## Final Table

| Method | Seeds | Expected-sum MAE | Rounded sum acc. | PMF mode acc. | FS-Conv NLL | Gaussian-bin NLL | Instance acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE | 3 | 2.165 +/- 0.149 | 0.219 +/- 0.054 | 0.231 +/- 0.074 | 2.350 +/- 0.239 | 2.455 +/- 0.134 | 0.781 +/- 0.079 |
| Gaussian-AMLE | 3 | 2.342 +/- 0.248 | 0.228 +/- 0.094 | 0.341 +/- 0.133 | 2.204 +/- 0.340 | 2.576 +/- 0.105 | 0.875 +/- 0.054 |
| FS-Conv | 3 | 1.983 +/- 0.359 | 0.328 +/- 0.116 | 0.479 +/- 0.181 | 1.777 +/- 0.465 | 2.466 +/- 0.038 | 0.916 +/- 0.049 |
