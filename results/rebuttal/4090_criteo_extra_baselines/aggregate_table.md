# Criteo Final Aggregate Metrics

Aggregate metrics are common post-hoc evaluations using exact Poisson-binomial PMFs from each method's instance probabilities.

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
