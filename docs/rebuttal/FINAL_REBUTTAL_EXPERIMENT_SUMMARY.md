# Final Rebuttal Experiment Summary

- Integration branch: `neurips26-rebuttal`.
- Integration HEAD before final summary commit: `f34a6737ddc7f41bb20d4e0131065dd674c2b78e`.
- 4090 feature tip: `c9928f33bb42b168800c2e7488b4cfdd51ca516b`; recorded final marker: `MISSING`.
- A5000 feature tip: `796d0b468b88a0a92deb912abe35bb63074080c3`; recorded final marker: `MISSING`.
- Scalar MNIST rebuttal baselines are complete for three seeds and distinguish PMF-mode from rounded expectation metrics.
- Fixed finite-bag CIFAR-10 CE/KL and FS-Conv rows are complete for three seeds with fixed manifests.
- The old `official_pvc_count_component` row is explicitly not labeled full LLP-PVC.
- Full official LLP-PVC has upstream pin/audit and adapter validation, but final seeds 0,1,2 are not available in the merged artifacts.
- Natural-image SVHN and conditional-dependence final rows are unavailable in committed artifacts.

Primary result files:

- `results/rebuttal/FINAL_REBUTTAL_EXPERIMENT_TABLES.md`
- `results/rebuttal/FINAL_REBUTTAL_EXPERIMENT_TABLES.csv`
- `results/rebuttal/FINAL_RUN_INVENTORY.csv`
- `results/rebuttal/FINAL_FAILURE_INVENTORY.md`
