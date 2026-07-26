# 4090 Official LLP-PVC Final Report

The full official LLP-PVC row completed under the strict-v3 corrected CIFAR-10
protocol on seeds 0,1,2. The local PVC count-component row remains a
component-equivalence result and must not be renamed as full official LLP-PVC.

Protocol:

- frozen ImageNet ResNet-18 CIFAR-10 features;
- 45,000/5,000 stratified split of the official CIFAR-10 training archive for
  train/validation;
- official CIFAR-10 test archive reserved for final evaluation only;
- bag size 64, 250 fixed train bags, 250 fixed validation bags, 1000 fixed test
  bags;
- identical manifests across CE/KL, FS-Conv, and full official LLP-PVC within
  each seed;
- aggregate validation composite count NLL primary checkpoint metric, count MAE
  tie-break;
- hidden instance labels used only in final test evaluation.

Learning-rate lock:

- seed-999 grid: `5e-4`, `1e-3`, `2.5e-3`;
- selected by aggregate validation composite count NLL;
- selected LR: `5e-4`.

Strict-v3 CIFAR result, mean +/- sample standard deviation over seeds 0,1,2:

| Method | Unique-image acc. | Macro-F1 | Count MAE | Classwise composite NLL |
|---|---:|---:|---:|---:|
| CE/KL proportion matching | 0.848 +/- 0.001 | 0.847 +/- 0.002 | 1.647 +/- 0.005 | 2.782 +/- 0.065 |
| Full official LLP-PVC | 0.309 +/- 0.188 | 0.256 +/- 0.178 | 6.685 +/- 0.489 | 6.579 +/- 0.425 |
| FS-Conv classwise count likelihood | 0.851 +/- 0.005 | 0.851 +/- 0.005 | 1.775 +/- 0.106 | 2.688 +/- 0.059 |

Paired CIFAR bootstrap:

- FS-Conv minus CE/KL count MAE: 0.1286, 95% CI [0.0610, 0.2432].
- FS-Conv minus CE/KL composite NLL: -0.0939, 95% CI [-0.1652, -0.0243].
- Full official LLP-PVC minus CE/KL count MAE: 5.0427, 95% CI [4.5093, 5.4341].
- Full official LLP-PVC minus CE/KL composite NLL: 3.7944, 95% CI [3.2505, 4.1414].

Interpretation:

Under the corrected finite-bag protocol, CE/KL and FS-Conv are comparable in
instance accuracy. FS-Conv is better on the classwise composite NLL used by its
count-likelihood objective, while CE/KL has lower count MAE. The faithful full
official LLP-PVC learner is weak in this frozen-feature, strict fixed-bag setup;
do not present the old local count-component result as full LLP-PVC.
