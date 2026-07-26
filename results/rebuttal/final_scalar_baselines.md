# Final Scalar Baselines

All entries are mean +/- sample standard deviation over seeds 0,1,2.
FS-Conv reports both PMF-mode aggregate accuracy and rounded expected-aggregate accuracy.
Gaussian-AMLE and MSE have no discrete PMF mode, so mode accuracy is the rounded mean/mode surrogate.
Continuous Gaussian NLL is not compared directly with exact discrete FS-Conv NLL.

## Digit Sum, MNIST, n=10, train bags=1000

| Method | Expected aggregate MAE | PMF/aggregate-mode acc. | Rounded expected acc. | Instance acc. | Exact discrete NLL | Gaussian NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FS-Conv | 0.704 +/- 0.029 | 0.846 +/- 0.002 | 0.756 +/- 0.028 | 0.982 +/- 0.000 | 0.670 +/- 0.039 | - |
| Gaussian-AMLE | 1.668 +/- 0.221 | 0.268 +/- 0.061 | 0.268 +/- 0.061 | 0.791 +/- 0.062 | - | 1.385 +/- 0.042 |
| MSE | 2.861 +/- 0.152 | - | 0.107 +/- 0.005 | 0.214 +/- 0.029 | - | - |

## Signed MNIST, n=10, train bags=1000

| Sign protocol | Method | Expected aggregate MAE | Aggregate-mode acc. | Rounded expected acc. | Instance acc. | Instance AUC | Exact discrete NLL | Gaussian NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | FS-Conv | 0.109 +/- 0.011 | 0.907 +/- 0.011 | 0.908 +/- 0.009 | 0.990 +/- 0.001 | 0.995 +/- 0.001 | 0.338 +/- 0.026 | - |
| cancellation | FS-Conv | 0.079 +/- 0.007 | 0.930 +/- 0.006 | 0.931 +/- 0.008 | 0.992 +/- 0.000 | 0.998 +/- 0.000 | 0.266 +/- 0.030 | - |
| random | Gaussian-AMLE | 0.234 +/- 0.023 | 0.851 +/- 0.012 | 0.851 +/- 0.012 | 0.983 +/- 0.002 | 0.992 +/- 0.001 | - | -0.695 +/- 0.071 |
| cancellation | Gaussian-AMLE | 0.212 +/- 0.024 | 0.848 +/- 0.019 | 0.848 +/- 0.019 | 0.983 +/- 0.002 | 0.991 +/- 0.002 | - | -0.713 +/- 0.140 |
| random | MSE | 0.330 +/- 0.258 | 0.753 +/- 0.238 | 0.753 +/- 0.238 | 0.959 +/- 0.052 | 0.973 +/- 0.033 | - | - |
| cancellation | MSE | 0.131 +/- 0.010 | 0.918 +/- 0.010 | 0.918 +/- 0.010 | 0.991 +/- 0.001 | 0.996 +/- 0.001 | - | - |

## Raw Sources

- digit_sum FS-Conv seed 0: `runs/neurips_pilot_preleave_20260504_1619/digit_sum_n10_train1000_s0/mnist_digit_sum/conv/MNIST/20260504T163350Z_gpusystem_s0`
- digit_sum FS-Conv seed 1: `runs/neurips_pilot_preleave_20260504_1619/digit_sum_n10_train1000_s1/mnist_digit_sum/conv/MNIST/20260504T172126Z_gpusystem_s1`
- digit_sum FS-Conv seed 2: `runs/neurips_pilot_preleave_20260504_1619/digit_sum_n10_train1000_s2/mnist_digit_sum/conv/MNIST/20260504T174455Z_gpusystem_s2`
- signed FS-Conv seed 0: `runs/neurips_pilot_preleave_20260504_1619/signed_cancel_n10_train1000_s0/signed_mnist/conv/MNIST/20260504T165706Z_gpusystem_s0`
- signed FS-Conv seed 1: `runs/neurips_pilot_preleave_20260504_1619/signed_cancel_n10_train1000_s1/signed_mnist/conv/MNIST/20260504T172126Z_gpusystem_s1`
- signed FS-Conv seed 2: `runs/neurips_pilot_preleave_20260504_1619/signed_cancel_n10_train1000_s2/signed_mnist/conv/MNIST/20260504T180841Z_gpusystem_s2`
- signed FS-Conv seed 0: `runs/neurips_pilot_preleave_20260504_1619/signed_random_n10_train1000_s0/signed_mnist/conv/MNIST/20260504T165706Z_gpusystem_s0`
- signed FS-Conv seed 1: `runs/neurips_pilot_preleave_20260504_1619/signed_random_n10_train1000_s1/signed_mnist/conv/MNIST/20260504T172126Z_gpusystem_s1`
- signed FS-Conv seed 2: `runs/neurips_pilot_preleave_20260504_1619/signed_random_n10_train1000_s2/signed_mnist/conv/MNIST/20260504T180841Z_gpusystem_s2`
- digit_sum Gaussian-AMLE seed 0: `results/rebuttal/gaussian_selected/runs/digit_sum_gaussian_amle_n10_train1000_sum_s0_lr1e-03_eps1e-04_cfdf9520_20260725T234858Z`
- digit_sum Gaussian-AMLE seed 1: `results/rebuttal/gaussian_selected/runs/digit_sum_gaussian_amle_n10_train1000_sum_s1_lr1e-03_eps1e-04_cfdf9520_20260725T234911Z`
- digit_sum Gaussian-AMLE seed 2: `results/rebuttal/gaussian_selected/runs/digit_sum_gaussian_amle_n10_train1000_sum_s2_lr1e-03_eps1e-04_cfdf9520_20260725T234924Z`
- signed Gaussian-AMLE seed 0: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_cancel_s0_lr1e-03_eps1e-04_cfdf9520_20260725T234858Z`
- signed Gaussian-AMLE seed 1: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_cancel_s1_lr1e-03_eps1e-04_cfdf9520_20260725T234917Z`
- signed Gaussian-AMLE seed 2: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_cancel_s2_lr1e-03_eps1e-04_cfdf9520_20260725T234937Z`
- signed Gaussian-AMLE seed 0: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_random_s0_lr1e-03_eps1e-04_cfdf9520_20260725T234858Z`
- signed Gaussian-AMLE seed 1: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_random_s1_lr1e-03_eps1e-04_cfdf9520_20260725T234917Z`
- signed Gaussian-AMLE seed 2: `results/rebuttal/gaussian_selected/runs/signed_gaussian_amle_n10_train1000_random_s2_lr1e-03_eps1e-04_cfdf9520_20260725T234936Z`
- digit_sum MSE seed 0: `runs/rebuttal_mse_digit_sum_n10_train1000/mnist_digit_sum_baseline/mse/MNIST/20260725T233330Z_gpusystem_s0`
- digit_sum MSE seed 1: `runs/rebuttal_mse_digit_sum_n10_train1000/mnist_digit_sum_baseline/mse/MNIST/20260725T233434Z_gpusystem_s1`
- digit_sum MSE seed 2: `runs/rebuttal_mse_digit_sum_n10_train1000/mnist_digit_sum_baseline/mse/MNIST/20260725T233535Z_gpusystem_s2`
- signed MSE seed 0: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_cancel_s0/signed_mnist_baseline/mse/MNIST/20260507T021232Z_gpusystem_s0`
- signed MSE seed 1: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_cancel_s1/signed_mnist_baseline/mse/MNIST/20260507T021232Z_gpusystem_s1`
- signed MSE seed 2: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_cancel_s2/signed_mnist_baseline/mse/MNIST/20260507T021232Z_gpusystem_s2`
- signed MSE seed 0: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_random_s0/signed_mnist_baseline/mse/MNIST/20260507T020935Z_gpusystem_s0`
- signed MSE seed 1: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_random_s1/signed_mnist_baseline/mse/MNIST/20260507T020935Z_gpusystem_s1`
- signed MSE seed 2: `runs/server4090_signed_mse_baseline/signed_mse_n10_train1000_random_s2/signed_mnist_baseline/mse/MNIST/20260507T020935Z_gpusystem_s2`
