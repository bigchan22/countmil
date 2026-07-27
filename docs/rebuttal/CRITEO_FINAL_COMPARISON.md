# Criteo Final Comparison

This comparison uses the same strict C4+C11 bag-size-128 manifests, feature shards, preprocessing, architecture, batch size, training budget, aggregate-only validation protocol, and final-test evaluation code.

## Instance Metrics

| Method | Seeds | ROC-AUC | PR-AUC | Log loss | Brier | ECE-15 | Acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Group prior | 3 | 0.6969 +/- 0.0026 | 0.4867 +/- 0.0047 | 0.5408 +/- 0.0005 | 0.1807 +/- 0.0002 | 0.0043 +/- 0.0005 | 0.7391 +/- 0.0003 |
| DLLP-BCE | 3 | 0.7522 +/- 0.0011 | 0.5507 +/- 0.0034 | 0.5171 +/- 0.0015 | 0.1700 +/- 0.0003 | 0.0301 +/- 0.0031 | 0.7512 +/- 0.0007 |
| DLLP-MSE | 3 | 0.7472 +/- 0.0012 | 0.5429 +/- 0.0036 | 0.5263 +/- 0.0031 | 0.1724 +/- 0.0009 | 0.0395 +/- 0.0029 | 0.7486 +/- 0.0009 |
| EasyLLP | 3 | 0.6904 +/- 0.0058 | 0.4211 +/- 0.0064 | 4.9454 +/- 0.1715 | 0.3277 +/- 0.0106 | 0.3271 +/- 0.0106 | 0.6698 +/- 0.0106 |
| GenBags | 3 | 0.7463 +/- 0.0012 | 0.5404 +/- 0.0045 | 0.5304 +/- 0.0018 | 0.1727 +/- 0.0002 | 0.0422 +/- 0.0010 | 0.7482 +/- 0.0006 |
| OT-LLP | 3 | 0.6524 +/- 0.0027 | 0.4260 +/- 0.0009 | 2.7261 +/- 0.1167 | 0.2982 +/- 0.0009 | 0.2910 +/- 0.0011 | 0.6792 +/- 0.0020 |
| FS-Conv | 3 | 0.7491 +/- 0.0004 | 0.5456 +/- 0.0030 | 0.5175 +/- 0.0011 | 0.1705 +/- 0.0002 | 0.0254 +/- 0.0038 | 0.7501 +/- 0.0004 |
| Supervised oracle | 3 | 0.7882 +/- 0.0011 | 0.6042 +/- 0.0022 | 0.4827 +/- 0.0002 | 0.1583 +/- 0.0001 | 0.0142 +/- 0.0017 | 0.7689 +/- 0.0007 |

## Aggregate Metrics

| Method | Seeds | Count MAE | Rounded count acc. | PMF-mode acc. | Exact count NLL |
|---|---:|---:|---:|---:|---:|
| Group prior | 3 | 5.7981 +/- 0.0210 | 0.0709 +/- 0.0036 | 0.0702 +/- 0.0026 | 3.9762 +/- 0.0201 |
| DLLP-BCE | 3 | 4.6380 +/- 0.0477 | 0.0735 +/- 0.0028 | 0.0728 +/- 0.0019 | 3.2845 +/- 0.0121 |
| DLLP-MSE | 3 | 4.5948 +/- 0.0363 | 0.0765 +/- 0.0055 | 0.0763 +/- 0.0057 | 3.2975 +/- 0.0288 |
| EasyLLP | 3 | 36.4932 +/- 1.5612 | 0.0043 +/- 0.0010 | 0.0043 +/- 0.0008 | 26.8902 +/- 0.1442 |
| GenBags | 3 | 4.9334 +/- 0.3351 | 0.0663 +/- 0.0069 | 0.0690 +/- 0.0051 | 3.4804 +/- 0.1521 |
| OT-LLP | 3 | 5.8189 +/- 0.0708 | 0.0570 +/- 0.0044 | 0.0561 +/- 0.0043 | 8.8693 +/- 0.4781 |
| FS-Conv | 3 | 4.6381 +/- 0.0793 | 0.0770 +/- 0.0037 | 0.0789 +/- 0.0018 | 3.2838 +/- 0.0356 |
| Supervised oracle | 3 | 4.7992 +/- 0.0916 | 0.0686 +/- 0.0037 | 0.0669 +/- 0.0031 | 3.3383 +/- 0.0246 |

## Paired Cluster Bootstrap

| Method | Metric | Mean diff. | 95% CI |
|---|---|---:|---:|
| Group prior | expected_count_mae | 1.1599 | [1.0157, 1.2905] |
| Group prior | poisson_binomial_nll | 0.6924 | [0.6263, 0.7599] |
| DLLP-BCE | expected_count_mae | -0.0001 | [-0.1204, 0.0991] |
| DLLP-BCE | poisson_binomial_nll | 0.0007 | [-0.0499, 0.0401] |
| DLLP-MSE | expected_count_mae | -0.0433 | [-0.1204, 0.0282] |
| DLLP-MSE | poisson_binomial_nll | 0.0137 | [-0.0158, 0.0445] |
| EasyLLP | expected_count_mae | 31.8551 | [30.1435, 33.1888] |
| EasyLLP | poisson_binomial_nll | 23.6064 | [23.4287, 23.7742] |
| GenBags | expected_count_mae | 0.2953 | [0.0570, 0.6784] |
| GenBags | poisson_binomial_nll | 0.1966 | [0.0823, 0.3755] |
| OT-LLP | expected_count_mae | 1.1808 | [1.0486, 1.3047] |
| OT-LLP | poisson_binomial_nll | 5.5855 | [5.0243, 5.9737] |
| Supervised oracle | expected_count_mae | 0.1611 | [-0.0094, 0.3247] |
| Supervised oracle | poisson_binomial_nll | 0.0545 | [-0.0098, 0.1083] |
