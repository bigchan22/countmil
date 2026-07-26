# Strict-v3 Digit-Sum Table

MNIST digit sum, mean bag size 10, 1000 fixed training bags. Mean +/- sample standard deviation over seeds 0,1,2.
Validation used aggregate metrics only; hidden digit accuracy is final-test only.

| Method | Seeds | Expected-sum MAE | Rounded expected acc. | PMF-mode acc. | Discrete aggregate NLL | Instance digit acc. |
|---|---:|---:|---:|---:|---:|---:|
| mse | 3 | 2.615 +/- 0.081 | 0.121 +/- 0.010 | 0.100 +/- 0.008 | 3.251 +/- 0.024 | 0.199 +/- 0.008 |
| gaussian_amle | 3 | 2.147 +/- 0.169 | 0.172 +/- 0.013 | 0.171 +/- 0.011 | 3.019 +/- 0.416 | 0.616 +/- 0.074 |
| fsconv | 3 | 1.498 +/- 0.366 | 0.370 +/- 0.172 | 0.436 +/- 0.272 | 1.658 +/- 0.727 | 0.878 +/- 0.083 |
