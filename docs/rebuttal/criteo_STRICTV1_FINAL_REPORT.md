# Criteo Strict-v1 Final Report

This is an LLP-Bench-style Criteo feature-bag experiment, not an exact full LLP-Bench reproduction.
Checkpoint selection used aggregate validation expected-count MAE only; hidden validation metrics were not computed.

## Instance Metrics

| Method | Seeds | ROC-AUC | PR-AUC | Log loss | Brier | ECE-15 | Acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DLLP-BCE | 3 | 0.7522 +/- 0.0011 | 0.5507 +/- 0.0034 | 0.5171 +/- 0.0015 | 0.1700 +/- 0.0003 | 0.0301 +/- 0.0031 | 0.7512 +/- 0.0007 |
| DLLP-MSE | 3 | 0.7472 +/- 0.0012 | 0.5429 +/- 0.0036 | 0.5263 +/- 0.0031 | 0.1724 +/- 0.0009 | 0.0395 +/- 0.0029 | 0.7486 +/- 0.0009 |
| EasyLLP | 3 | 0.6904 +/- 0.0058 | 0.4211 +/- 0.0064 | 4.9454 +/- 0.1715 | 0.3277 +/- 0.0106 | 0.3271 +/- 0.0106 | 0.6698 +/- 0.0106 |
| FS-Conv | 3 | 0.7491 +/- 0.0004 | 0.5456 +/- 0.0030 | 0.5175 +/- 0.0011 | 0.1705 +/- 0.0002 | 0.0254 +/- 0.0038 | 0.7501 +/- 0.0004 |

## Aggregate Metrics

| Method | Seeds | Count MAE | Rounded count acc. | PMF-mode acc. | Exact count NLL |
|---|---:|---:|---:|---:|---:|
| DLLP-BCE | 3 | 4.6380 +/- 0.0477 | 0.0735 +/- 0.0028 | 0.0728 +/- 0.0019 | 3.2845 +/- 0.0121 |
| DLLP-MSE | 3 | 4.5948 +/- 0.0363 | 0.0765 +/- 0.0055 | 0.0763 +/- 0.0057 | 3.2975 +/- 0.0288 |
| EasyLLP | 3 | 36.4932 +/- 1.5613 | 0.0043 +/- 0.0010 | 0.0043 +/- 0.0008 | 26.8902 +/- 0.1442 |
| FS-Conv | 3 | 4.6381 +/- 0.0793 | 0.0770 +/- 0.0037 | 0.0789 +/- 0.0018 | 3.2838 +/- 0.0356 |
