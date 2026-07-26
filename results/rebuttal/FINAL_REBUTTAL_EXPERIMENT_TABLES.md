# Final Rebuttal Experiment Tables

This file integrates the locally available 4090 and merged A5000 artifacts.
All mean +/- values use sample standard deviation over seeds unless noted.
Missing rows are left explicit; no values are fabricated or interpolated.

## 1. Stronger Scalar Baselines

MNIST digit sum, n=10, train bags=1000, seeds 0,1,2.

| Method | Expected aggregate MAE | Aggregate-mode acc. | Rounded expected acc. | Instance acc. | NLL field | Raw summary |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MSE | 2.861 +/- 0.152 | - | 0.107 +/- 0.005 | 0.214 +/- 0.029 | - | `results/rebuttal/scalar_audit_summary.csv` |
| Gaussian-AMLE | 1.668 +/- 0.221 | 0.268 +/- 0.061 | 0.268 +/- 0.061 | 0.791 +/- 0.062 | 1.385 +/- 0.042 | `results/rebuttal/scalar_audit_summary.csv` |
| FS-Conv | 0.704 +/- 0.029 | 0.846 +/- 0.002 | 0.756 +/- 0.028 | 0.982 +/- 0.000 | 0.670 +/- 0.039 | `results/rebuttal/scalar_audit_summary.csv` |

Signed MNIST, n=10, train bags=1000, random/cancellation protocols, seeds 0,1,2.

| Sign protocol | Method | Expected aggregate MAE | Aggregate-mode acc. | Instance acc. | Instance AUC | Raw summary |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| random | MSE | 0.330 +/- 0.258 | 0.753 +/- 0.238 | 0.959 +/- 0.052 | 0.973 +/- 0.033 | `results/rebuttal/scalar_audit_summary.csv` |
| random | Gaussian-AMLE | 0.234 +/- 0.023 | 0.851 +/- 0.012 | 0.983 +/- 0.002 | 0.992 +/- 0.001 | `results/rebuttal/scalar_audit_summary.csv` |
| random | FS-Conv | 0.109 +/- 0.011 | 0.907 +/- 0.011 | 0.990 +/- 0.001 | 0.995 +/- 0.001 | `results/rebuttal/scalar_audit_summary.csv` |
| cancellation | MSE | 0.131 +/- 0.010 | 0.918 +/- 0.010 | 0.991 +/- 0.001 | 0.996 +/- 0.001 | `results/rebuttal/scalar_audit_summary.csv` |
| cancellation | Gaussian-AMLE | 0.212 +/- 0.024 | 0.848 +/- 0.019 | 0.983 +/- 0.002 | 0.991 +/- 0.002 | `results/rebuttal/scalar_audit_summary.csv` |
| cancellation | FS-Conv | 0.079 +/- 0.007 | 0.930 +/- 0.006 | 0.992 +/- 0.000 | 0.998 +/- 0.000 | `results/rebuttal/scalar_audit_summary.csv` |

## 2. LLP-PVC Relationship

- Existing local `PVC` classification: `locally modified approximation of the classwise count-likelihood component`.
- Existing local rows may be called full LLP-PVC: `False`.
- Forward loss difference: `4.441e-16`.
- Per-class PMF difference: `2.220e-16`.
- Logit-gradient difference: `2.116e-16`.
- This is a component-equivalence check, not a competitive full-method baseline.

Official LLP-PVC full-pipeline status:

- Development-only seed 999 LR check completed; selected LR by aggregate validation NLL would be `0.001`.
- Best dev row: val NLL `6.522`, val count MAE `5.542`, test NLL `6.515`, test count MAE `5.552`, unique-image acc. `0.523`.
- Required official LLP-PVC seeds 0,1,2 are not present in committed result summaries.

## 3. Fixed Finite-Bag CIFAR-10

Protocol: frozen ImageNet-pretrained ResNet-18 features, n=64, 250 fixed train bags, 250 validation bags, 1000 test bags, Dirichlet alpha=0.3.

| Method | Seeds | Instance acc. | Macro-F1 | Count MAE | Composite NLL | Raw summaries |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CE/KL proportion matching | 6 | 0.852 +/- 0.002 | 0.852 +/- 0.002 | 1.552 +/- 0.011 | 2.812 +/- 0.049 | `results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_ce_n64_train250_s0.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_ce_n64_train250_s1.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_ce_n64_train250_s2.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_kl_n64_train250_s0.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_kl_n64_train250_s1.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_kl_n64_train250_s2.json` |
| Full official LLP-PVC | 0 | - | - | - | - | not available |
| FS-Conv classwise count likelihood | 3 | 0.853 +/- 0.001 | 0.853 +/- 0.002 | 1.744 +/- 0.058 | 2.676 +/- 0.032 | `results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_fsconv_count_n64_train250_s0.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_fsconv_count_n64_train250_s1.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_fsconv_count_n64_train250_s2.json` |
| PVC count-component only | 3 | 0.838 +/- 0.006 | 0.837 +/- 0.007 | 1.867 +/- 0.169 | 2.861 +/- 0.097 | `results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_official_pvc_count_component_n64_train250_s0.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_official_pvc_count_component_n64_train250_s1.json; results/rebuttal/fixed_bag_cifar10/summaries/fixed_cifar10_official_pvc_count_component_n64_train250_s2.json` |

## 4. Natural-Image Replication

No strict committed three-seed pretrained SVHN result summaries are present after merging `origin/exp/a5000-svhn-dependence`.
The A5000 branch includes protocol and invalid-run audits plus rerun scripts, but not reportable strict result JSONs.
See `docs/rebuttal/a5000_SVHN_PROTOCOL_AUDIT.md` and `docs/rebuttal/a5000_SVHN_INVALID_RUN_AUDIT.md`.

## 5. Conditional-Independence Sensitivity

No committed dependence-stress result summaries are present after merging `origin/exp/a5000-svhn-dependence`.
The merged branch includes the generator and training scripts under `src/experiments/dependence_stress/` and `scripts/rebuttal/a5000_train_dependence.py`.
