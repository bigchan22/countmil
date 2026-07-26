# Final Experiment Audit

## PVC Verdict

- Classification: C
- Recommended label: FS-Conv classwise count likelihood / LLP-PVC count-likelihood component
- Existing local `PVC` rows must not be called full official LLP-PVC unless a separate faithful official run is completed.
- Equivalence result is numerical verification of the shared count-likelihood component, not a competitive baseline.

## Scalar Metrics

- Scalar summary rows available: 9
- FS-Conv digit-sum aggregate accuracy is split into PMF-mode accuracy and rounded expected-sum accuracy.
- Gaussian-AMLE NLL is continuous-density NLL and is not directly compared with exact discrete FS-Conv NLL.

## Gaussian-AMLE Tuning

- Selected by dev seed aggregate validation loss: lr=0.001, eps=0.0001, validation loss=0.913620.

## Fixed-Bag CIFAR-10

- Completed fixed-bag CIFAR-10 runs: 12

## Artifact Checks

- Manuscript files were intentionally not edited by these scripts.
- Missing values are left absent/pending rather than fabricated.
