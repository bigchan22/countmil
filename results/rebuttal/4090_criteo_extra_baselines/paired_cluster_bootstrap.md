# Criteo Paired Cluster Bootstrap

Differences are method minus FS-Conv. Negative values favor the method for error/NLL metrics. Whole test bags are resampled within seed; 10,000 bootstrap replicates.

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
