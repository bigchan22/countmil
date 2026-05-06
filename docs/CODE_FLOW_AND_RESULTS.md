# CountMIL Code Flow and Result Interpretation

This note summarizes how the code supports the paper and what the current experiment results justify. Use `paper/experiment_report_2026-05-06.md` as the detailed source of truth for numbers.

## Code Flow

The repository is organized around a simple pipeline:

1. Bag datasets sample instances on the fly and expose only aggregate labels for training.
2. Instance networks predict latent instance PMFs.
3. Aggregate layers convolve those PMFs into an exact bag-level PMF.
4. Training minimizes aggregate negative log likelihood or a baseline aggregate loss.
5. Hidden instance labels are used only for evaluation.
6. Scripts write run metadata, per-epoch JSONL metrics, compact JSON summaries, and aggregate CSVs.

## Core Method Code

- `src/countmil/aggregators.py`
  - `finite_support_convolution`: sequential exact finite-support convolution.
  - `finite_support_convolution_fft_tree`: balanced FFT-tree convolution for larger batched PMFs.
  - `binary_count_dp`: Shukla-style Bernoulli count DP baseline.
  - `grouped_signed_binary_convolution`: grouped GPU convolution for signed Bernoulli aggregates.
  - `aggregate_nll`: reads the observed aggregate probability and returns NLL.
- `src/countmil/posteriors.py`
  - `binary_count_posterior`: exact count-conditioned posterior marginals for binary Count-MIL.
- `src/countmil/baselines/proportion_matching.py`
  - CE/KL/MSE proportion matching for LLP-style baselines.

## Dataset and Model Code

- `src/countmil/datasets/mnist.py`
  - `MNISTBags`: binary count and contains-target labels.
  - `MNISTDigitSumBags`: scalar digit-sum labels.
  - `MNISTDigitHistogramBags`: digit histogram labels.
  - `SignedMNISTBags`: signed target-count labels.
- `src/countmil/datasets/cifar.py`
  - `CIFARHistogramBags`: CIFAR-10/100 histogram labels.
- `src/countmil/models/mnist_cnn.py`
  - `ShuklaMNISTSelector`: binary LeNet-style selector.
  - `MNISTDigitClassifier`: 10-class LeNet-style classifier.
  - `AttentionMILMNIST`: attention/gated-attention MIL baselines.
- `src/countmil/models/cifar_cnn.py`
  - small CNN and ResNet-18 wrappers, including optional ImageNet-pretrained ResNet-18.

## Experiment Entry Points

- `scripts/train_mnist_bags.py`: binary Count-MIL, DP vs convolution, plus exploratory posterior objectives.
- `scripts/train_mnist_bags_attention.py`: attention and gated-attention MIL baselines.
- `scripts/train_mnist_digit_sum.py`: exact finite-support digit-sum likelihood.
- `scripts/train_mnist_digit_sum_baseline.py`: expected-sum MSE/MAE/Huber baselines.
- `scripts/train_signed_mnist.py`: signed Count-MIL with random or cancellation-heavy signs.
- `scripts/train_mnist_histogram_llp.py`: MNIST histogram CE/KL/MSE proportion matching.
- `scripts/train_mnist_histogram_pvc.py`: MNIST one-vs-rest count likelihood/PVC.
- `scripts/train_cifar_histogram.py`: CIFAR histogram PVC or proportion matching.
- `scripts/bench_exactness_runtime.py`: broad exactness/runtime benchmark.
- `scripts/bench_multiclass_ovr_fft.py`: focused CPU DP vs GPU FFT-tree benchmark for paper runtime.

## Current Result Interpretation

### Strongest Supported Claims

- Binary Bernoulli CountMIL recovers the Shukla-style count probability. It is a sanity check and baseline connection, not the novelty.
- Finite-support convolution cleanly extends exact aggregate likelihoods to signed counts and ordinal digit sums.
- MNIST digit-sum results show exact likelihood is much more data-efficient than expected-value matching in the low-data small-bag regime.
- Signed Count-MIL is a real novelty case: random signs work well, while cancellation-heavy large bags expose genuine ambiguity.
- MNIST histogram PVC improves over CE/KL proportion matching in completed grids, especially on count MAE and hidden digit accuracy.
- GPU FFT-tree convolution is numerically close to CPU DP and much faster in the focused batched one-vs-rest benchmark.

### CIFAR-10 Interpretation After Fixed-Budget Diagnostic

The pretrained ResNet-18 PVC results are strong for `n=16/train1000` and `n=64/train1000`, but the completed `n=64/train250` diagnostic changes the claim.

| Setting | Total train instances/epoch | Mean final instance accuracy |
|---|---:|---:|
| n16/train1000 | 16,000 | 90.24% |
| n64/train250 | 16,000 | 80.34% |
| n64/train1000 | 64,000 | 91.45% |

Conclusion: the earlier `n64/train1000` result should not be described as evidence that larger bags are intrinsically better. Under fixed sampled image exposure, `n64` is much worse than `n16`. Larger bags appear to need more total image exposure or more aggregate observations to work well.

### Weak or Risky Claims

- Do not claim binary count loss is new.
- Do not claim histogram PVC is independent of LLP-PVC.
- Do not claim superiority to fully supervised learning.
- Do not claim large bags are intrinsically easier or more scalable.
- Do not use CIFAR-100 coarse pilots as headline evidence; they are above chance but not strong.
- Treat expected-sum baseline rows with incomplete seeds as provisional.

## Overall Paper Evaluation

The codebase supports a coherent paper, but the paper should be positioned as a unifying exact-likelihood framework rather than as a broad empirical dominance paper. The most defensible center is:

> Finite-support convolution gives exact aggregate likelihoods for weak labels beyond ordinary binary counts, including signed cancellation and ordinal sums, with posterior marginals and practical GPU/FFT backends.

The current evidence is strongest on MNIST-family aggregate supervision and runtime exactness. CIFAR supports feasibility under a strong pretrained representation, but the fixed-budget result argues for caution.
