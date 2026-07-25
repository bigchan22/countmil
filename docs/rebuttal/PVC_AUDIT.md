# PVC Implementation Audit

Date: 2026-07-25 UTC.

## Bottom Line

The current local `pvc` implementation should **not** be described as the complete official LLP-PVC method from Ma et al. It is best described as:

> the one-vs-rest Poisson-binomial / count-likelihood component used by LLP-PVC, implemented locally through FS-Conv's classwise finite-support convolution backend.

Existing MNIST and CIFAR rows named `PVC` are therefore honest only if labeled as `PVC-style classwise count likelihood`, `OVR count likelihood`, or `LLP-PVC count-likelihood component`. They should not be called full official LLP-PVC results unless we run the official implementation or a verified faithful reproduction of the full official training recipe.

## Local Implementations Inspected

- `scripts/train_mnist_histogram_pvc.py`
- `scripts/train_cifar_histogram.py`
- `src/countmil/baselines/proportion_matching.py`
- `tests/test_histogram_pvc_training.py`

## What Local `PVC` Does

For class probabilities `p_{ic}` and observed class counts `h_c`, the local implementation computes one Bernoulli count likelihood per class:

```text
K_c = sum_i 1[y_i = c]
loss = - mean_c log P(K_c = h_c)
```

Implementation details:

- output parameterization: model logits are converted by `softmax(logits, dim=-1)`;
- classwise probabilities: each class is treated one-vs-rest as Bernoulli probabilities `p_{ic}`;
- loss: mean of per-class negative log Poisson-binomial count probabilities;
- backend: FFT-tree finite-support convolution for classwise count PMFs;
- evaluation: predicted instance label is `argmax softmax(logits)`;
- checkpoint selection: current CIFAR code selects by observed `hist_count_mae`, not hidden instance accuracy.

## Official LLP-PVC Code Inspected

OpenReview lists the official code at:

- `https://github.com/TianhaoMa5/ICLR2026_LLP-PVC`

The inspected official script contains functions named:

- `compute_CC_loss_dp_precise`
- `compute_CC_loss_fft_precise`
- `compute_CC_loss_fft_precise_batched`

These functions compute the same conceptual object: a per-class Poisson-binomial count likelihood from class probabilities and integerized bag proportions. The official code also includes broader training-protocol details, including dataset loaders, bag construction modes, ResNet setup, warmup/cosine scheduling, and evaluation code.

One important implementation distinction is that the official training script contains a path where `torch.sigmoid(logits_u_w)` is passed into the count-likelihood component, while `torch.softmax(logits_u_w, dim=1)` is used later for pseudo-label/evaluation logic. The local implementation uses `softmax` probabilities directly for the count likelihood.

## Classification

The current local implementation is:

**C. a local approximation / implementation of the shared classwise count-likelihood component.**

It is also close to:

**B. only the one-vs-rest Poisson-binomial/count-likelihood component**

when the parameterization is matched.

It is **not**:

**A. the complete official LLP-PVC method from Ma et al.**

## Consequence for Existing Results

Existing local result labels should be revised in rebuttal language:

- acceptable: `FS-Conv classwise count likelihood`
- acceptable: `OVR Poisson-binomial count likelihood`
- acceptable: `PVC-style count-likelihood component`
- not acceptable without further official-code run: `full LLP-PVC`

If the rebuttal includes a direct comparison against full LLP-PVC, it should either:

1. run the official implementation under a matched protocol; or
2. explicitly report numerical equivalence only for the count-likelihood component and avoid duplicate method rows.

## Numerical Equivalence Test

The script `scripts/rebuttal/check_pvc_equivalence.py` compares:

- FS-Conv classwise FFT-tree count likelihood;
- a reference one-vs-rest Poisson-binomial dynamic-programming count component.

It reports maximum absolute differences for:

- forward loss;
- per-class count probabilities;
- gradients with respect to logits.

This test validates equivalence of the count-likelihood component under a matched softmax parameterization. It does not validate equivalence to the complete official training recipe.
