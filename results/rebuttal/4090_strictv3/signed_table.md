# Strict-v3 Signed MNIST Table

Signed target-digit MNIST, mean bag size 10, 1000 fixed training bags. Mean +/- sample standard deviation over seeds 0,1,2.

| Protocol | Method | Seeds | Expected signed MAE | Rounded MAE | Rounded acc. | PMF-mode acc. | Discrete NLL | Instance acc. | Instance AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signed_random | mse | 3 | 0.354 +/- 0.292 | 0.307 +/- 0.332 | 0.739 +/- 0.254 | 0.743 +/- 0.257 | 3.093 +/- 4.736 | 0.957 +/- 0.054 | 0.814 +/- 0.309 |
| signed_random | gaussian_amle | 3 | 0.198 +/- 0.046 | 0.147 +/- 0.024 | 0.855 +/- 0.022 | 0.859 +/- 0.026 | 0.398 +/- 0.091 | 0.984 +/- 0.004 | 0.991 +/- 0.004 |
| signed_random | fsconv | 3 | 0.110 +/- 0.005 | 0.077 +/- 0.004 | 0.923 +/- 0.004 | 0.922 +/- 0.005 | 0.237 +/- 0.015 | 0.992 +/- 0.001 | 0.997 +/- 0.001 |
| signed_cancellation | mse | 3 | 0.119 +/- 0.007 | 0.077 +/- 0.008 | 0.923 +/- 0.008 | 0.924 +/- 0.007 | 0.234 +/- 0.023 | 0.992 +/- 0.001 | 0.998 +/- 0.000 |
| signed_cancellation | gaussian_amle | 3 | 0.223 +/- 0.038 | 0.160 +/- 0.025 | 0.843 +/- 0.025 | 0.846 +/- 0.025 | 0.431 +/- 0.058 | 0.982 +/- 0.003 | 0.993 +/- 0.002 |
| signed_cancellation | fsconv | 3 | 0.094 +/- 0.004 | 0.067 +/- 0.004 | 0.934 +/- 0.005 | 0.935 +/- 0.004 | 0.215 +/- 0.008 | 0.993 +/- 0.001 | 0.998 +/- 0.000 |
