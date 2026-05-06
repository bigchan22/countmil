# A1 colleague result summary for remote reproduction

Date: 2026-05-06
Source directory: `refs/Colleague_work/`
Purpose: preserve the colleague A1 verification code, result summaries, and verdict notes so a remote server clone can inspect or reproduce the experiments.

## Verdict

Final A1 verdict: `A1_CONFIRMED`.

A1 tests atomic PMF generality: the same one-dimensional convolutional aggregate likelihood is reused while changing only the atomic support.

| Atom support | Task family | Representative result |
|---|---|---|
| `S=2`, `{0,1}` | binary count | B1 binary reuse passes, instance AUC about `0.9985` at `N=50` |
| `S=10`, `{0,...,9}` | digit-sum / ordinal sum | MNIST-sum, SVHN-sum, and UltraMNIST all pass relative top-1 criteria |
| `S=3`, `{-1,0,+1}` | signed count | MNIST-signed passes, PCA acc `0.692` vs mean-pool `0.302` |

## Key aggregate results

All values below are from the colleague `tail5` selector summaries.

| Dataset | PCA acc. | Best baseline acc. | PCA MAE | PCA NLL | Main interpretation |
|---|---:|---:|---:|---:|---|
| MNIST-sum | `0.635` | `0.541` | `1.360` | `3.290` | best point prediction, NLL not significant under 5 seeds |
| SVHN-sum | `0.245` | `0.116` | `3.081` | `3.653` | strongest harder natural-image evidence, H1 NLL pass `p=0.001` |
| UltraMNIST | `0.899` | `0.572` | `0.401` | `0.606` | strongest clean synthetic evidence, H1 NLL pass `p=0.005` |
| MNIST-signed | `0.692` | `0.302` | `0.564` | `3.293` | signed atomic support works |

Cross-dataset hypothesis summary:

- H1 NLL primary: pass on 2 of 3 multiclass datasets (`SVHN-sum`, `UltraMNIST`).
- H2 atomic-shape generality: pass on 5 of 5 checks.
- H3 calibration: pass on 1 of 3 multiclass datasets; calibration remains a limitation.
- H4 point prediction: pass on 3 of 3 multiclass datasets.

## Files to inspect

- `docs/notes/2026-05-06-a1-final-verdict.md`: final narrative and hypothesis verdict.
- `results/summary_overall.md`: aggregate result summary.
- `results/*/summary.md` and `results/*/summary.json`: per-dataset result summaries.
- `pca/losses.py`: `atomic_conv` and aggregate likelihood code.
- `pca/data.py`: MNIST-sum, MNIST-signed, SVHN-sum, and UltraMNIST bag datasets.
- `pca/models.py`: SmallCNN, ResNet18-from-scratch, and PatchEncoder backbones.
- `pca/train.py`: mask-aware variable-size bag training loop.
- `scripts/analyze_a1.py`: H1/H2/H3/H4 verdict logic.
- `scripts/run_svhn_sum.py` and `scripts/run_ultramnist.py`: remote-relevant reproduction entry points.
- `tests/test_atomic_conv.py` and `tests/test_analyze_a1.py`: algebra and verdict tests.

## Paper integration note

The NeurIPS draft now uses this result as supporting evidence for the problem formulation:

> The atomic support size `S` defines the local contribution PMF; changing `S` and the contribution values changes the weak-label problem while the convolutional bag likelihood remains unchanged.

This should be cited in the paper as internal experimental evidence rather than as a separate prior-work claim.
