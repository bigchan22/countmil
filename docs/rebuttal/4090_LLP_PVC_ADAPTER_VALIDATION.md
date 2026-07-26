# 4090 LLP-PVC Adapter Validation

Date: 2026-07-26 UTC.

## Adapter Scope

The adapter is implemented in:

- `src/baselines/llp_pvc/official.py`
- `scripts/rebuttal/4090_train_official_llppvc_fixed_cifar.py`

It preserves the upstream method-defining training components while replacing only:

- the upstream image dataset loader;
- the upstream bag loader;
- the upstream image backbone input interface.

The experiment input is the already cached frozen ImageNet-pretrained ResNet-18 feature representation used by the fixed-bag CE/KL and FS-Conv experiments.

## Preserved Official Components

- Count-loss probabilities: `sigmoid(logits)`.
- Evaluation predictions: `argmax softmax(logits)`.
- Count loss: classwise Poisson-binomial count likelihood, summed over classes and averaged over bags.
- Count target conversion: rounded proportions with largest-proportion correction for sum mismatch.
- Optimizer: SGD with Nesterov momentum.
- Scheduler: official warmup-cosine rule with linear warmup.
- Default EMA: off, matching upstream default.

## Fixed-Protocol Safeguards

- Feature train/test hashes are checked and must differ.
- Train, validation, and test manifest hashes are loaded from existing fixed-bag manifests.
- The training manifest hash is rechecked every epoch.
- Training batches contain features, counts, proportions, and masks only; hidden labels are not yielded to the optimizer.
- Checkpoint selection uses aggregate validation composite count NLL, tie-breaking by aggregate validation count MAE.

## Tests

Implemented in `tests/test_4090_llppvc_adapter.py`.

Covered checks:

- Synthetic official-style loss agrees with an independent reference implementation.
- Gradients with respect to logits agree to machine precision.
- Softmax prediction rule is used at evaluation.
- Probability normalization and output shape are valid.
- Official warmup-cosine scheduler produces finite positive learning rates.
- Fixed manifests remain stable across shuffled training epochs.
- Training batches do not expose hidden instance labels.

Latest targeted validation:

```text
6 passed
```
