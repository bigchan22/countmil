# PVC Audit Verdict

**Verdict: C.** The existing method named `PVC` is a locally implemented approximation of the shared classwise Poisson-binomial count-likelihood component. It is not the complete official LLP-PVC method.

Existing rows should be labeled as `FS-Conv classwise count likelihood` or `LLP-PVC count-likelihood component`, not as full official LLP-PVC.

## Numerical Verification

This is a special-case relationship check, not a competitive baseline result.

| Quantity | Max absolute difference |
| --- | ---: |
| Forward loss | 4.44e-16 |
| Per-class PMF | 2.22e-16 |
| Logit gradient | 2.12e-16 |

## Key Differences

- Output parameterization: local code uses `softmax`; official code contains a `sigmoid` path for the count loss and `softmax` for evaluation/pseudo-label logic.
- Complete loss/procedure: local code implements the classwise count likelihood only; official code includes broader training procedure details.
- Prediction/checkpointing: local code predicts by `argmax softmax` and checkpoints by observed histogram metric; official code has its own evaluation/EMA bookkeeping.
- Bag construction: local old runs used on-the-fly random bags; official code supports multiple bag construction strategies.
