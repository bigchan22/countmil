# Existing Result Audit

Date: 2026-07-25 UTC.

This audit covers which existing results can be reused for rebuttal tables and which claims require new runs. Hidden instance labels were used for evaluation in these scripts, but the key risk is whether they were also used for checkpoint selection.

## Locally Available Seed-Level Results

These have local JSON/CSV artifacts and can be aggregated with sample standard deviation.

### Binary MNIST CountMIL

- Path: `results/neurips_pilot_preleave_20260504_1619/aggregate.csv`
- Seeds: `0,1,2`
- Settings: bag means `10`, `50`; train bags `1000`, `5000`
- Objective: exact Bernoulli count NLL
- Reuse status: usable as validation / Shukla-style count-loss connection.
- Caveat: old posterior/EM variants include some objectives that are not central to rebuttal.

### MNIST Digit-Sum FS-Conv

- Path: `results/neurips_pilot_preleave_20260504_1619/aggregate.csv`
- Seeds: `0,1,2`
- Settings: bag means `10`, `50`; train bags `1000`, `5000`
- Objective: exact finite-support digit-sum NLL
- Reuse status: usable.
- Selection: existing script selects best by expected aggregate MAE, not hidden digit accuracy.

### Signed MNIST FS-Conv

- Path: `results/neurips_pilot_preleave_20260504_1619/aggregate.csv`
- Seeds: `0,1,2`
- Settings: bag means `10`, `50`; train bags `1000`, `5000`; random and cancellation-heavy signs
- Objective: exact signed count NLL
- Reuse status: usable with caution.
- Important caveat: `scripts/train_signed_mnist.py` selects checkpoints by hidden `instance_auc`. For new rebuttal runs, checkpoint selection must use aggregate validation loss/error instead. Existing signed FS-Conv rows should be reported transparently or recomputed if strict aggregate-only model selection is required.

### Signed MNIST MSE Baseline

- Path: `results/server4090_signed_mse_baseline/aggregate.csv`
- Seeds: `0,1,2`
- Settings: bag means `10`, `50`; train bags `1000`, `5000`; random and cancellation-heavy signs
- Objective: expected signed-count MSE
- Reuse status: usable as a point aggregate-regression baseline.
- Caveat: this is not a distributional baseline.

### MNIST Histogram LLP and PVC-Style Count Likelihood

- Paths:
  - `results/server4090_histogram_llp/aggregate.csv`
  - `results/server4090_histogram_pvc/aggregate.csv`
- Seeds: `0,1,2`
- LLP objectives: CE, KL, MSE proportion matching
- PVC-style objective: local classwise OVR count likelihood
- Reuse status: usable if local PVC rows are labeled as the count-likelihood component, not full official LLP-PVC.

### UltraMNIST-Style Sum FS-Conv

- Path: `results/atomic_sum_server4090_ultramnist_20260506_1758/aggregate.csv`
- Seeds: `0,1,2,3,4`
- Objective: exact finite-support sum NLL
- Reuse status: usable.
- Instruction status: do not rerun existing FS-Conv UltraMNIST runs.

### FashionMNIST MSE Add-On

- Path: `results/server4090_fashion_mse2/`
- Seeds: `0,1`
- Reuse status: usable as an auxiliary baseline check.
- Caveat: not requested for mandatory rebuttal runs.

## Results Recorded from Other Servers

These were recorded in `paper/experiment_report_2026-05-06.md`, but seed-level raw JSONs are not all present locally.

### FashionMNIST Robustness

- Server: A5000
- Seeds: `0,1`
- Grid: 28 jobs
- Reuse status: usable as recorded means, but sample std cannot be recomputed locally without A5000 raw JSONs.

### CIFAR-100 Coarse Pilot

- Server: A5000
- Seeds: `0,1`
- Reuse status: exploratory only. Not a headline result.

### CIFAR-10 Scratch ResNet-18

- Server: A5000
- Seeds: `0,1`
- Reuse status: useful to show representation dependence. Not sufficient as a strong rebuttal baseline.

### CIFAR-10 Pretrained ResNet-18 PVC-Style Runs

- Server: A5000
- Seeds: `0,1`
- Reuse status: usable only as `PVC-style / classwise count likelihood` under the old resampled-bag protocol.
- Caveat: the existing `n64/train250` result is fixed-instance-budget, not a true fixed finite aggregate-label dataset because bags are resampled each epoch.

## Checkpoints

No `.pt` or `.pth` checkpoints were found under local `runs/` or `results/` during this audit. Existing metrics therefore cannot be recomputed from checkpoints unless those checkpoints are copied back from remote servers. New rebuttal scripts must save best and final checkpoints.

## Mandatory New Evidence

The rebuttal should add:

1. Gaussian-AMLE for MNIST digit-sum `n=10/train1000`, seeds `0,1,2`.
2. Gaussian-AMLE for signed MNIST `n=10/train1000`, seeds `0,1,2`.
3. After smoke tests, Gaussian-AMLE for `n=50/train1000` digit-sum and signed MNIST, seeds `0,1,2`.
4. True fixed-bag CIFAR-10 protocol with saved bag manifests, fixed validation/test bags, and identical manifests across aggregate methods.

## Metric Definitions

- `Instance acc.`: hidden instance top-1 accuracy for multiclass tasks, or thresholded binary accuracy for signed/binary target detection.
- `Instance AUC`: hidden binary target-detection ROC AUC.
- `Expected aggregate MAE`: absolute error of posterior/point expected aggregate.
- `Mode/rounded aggregate acc.`: exact aggregate match after PMF mode or rounded expectation.
- `Composite count NLL`: mean classwise OVR count negative log likelihood. This is not a normalized joint histogram NLL.
