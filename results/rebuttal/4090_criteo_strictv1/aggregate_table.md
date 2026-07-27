# Criteo Strict-v1 Aggregate Metrics

All aggregate metrics are common post-hoc evaluations using each method's instance probabilities.

| Method | Seeds | Count MAE | Rounded count acc. | PMF-mode acc. | Exact count NLL |
|---|---:|---:|---:|---:|---:|
| DLLP-BCE | 3 | 4.6380 +/- 0.0477 | 0.0735 +/- 0.0028 | 0.0728 +/- 0.0019 | 3.2845 +/- 0.0121 |
| DLLP-MSE | 3 | 4.5948 +/- 0.0363 | 0.0765 +/- 0.0055 | 0.0763 +/- 0.0057 | 3.2975 +/- 0.0288 |
| EasyLLP | 3 | 36.4932 +/- 1.5613 | 0.0043 +/- 0.0010 | 0.0043 +/- 0.0008 | 26.8902 +/- 0.1442 |
| FS-Conv | 3 | 4.6381 +/- 0.0793 | 0.0770 +/- 0.0037 | 0.0789 +/- 0.0018 | 3.2838 +/- 0.0356 |
