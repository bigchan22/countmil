# 4090 LLP-PVC Upstream Audit

Date: 2026-07-26 UTC.

## Upstream Pin

- Repository: `https://github.com/TianhaoMa5/ICLR2026_LLP-PVC`
- Local path: `third_party/ICLR2026_LLP-PVC/`
- Pinned commit: `fe11a007f5f7953666370129105aa2a0c3a6d6a2`
- Integration method: read-only Git submodule.

## Files Inspected / Used

- `LLP-PVC.py`: primary training loop, count likelihood, optimizer setup, evaluation rule.
- `utils.py`: `WarmupCosineLrScheduler`, accuracy helpers, logging setup.
- `README.md`: defaults and command-line argument descriptions.

No `LICENSE`, `COPYING`, or `NOTICE` file was present in the upstream repository at the pinned commit. Treat reuse as source-code audit/adaptation for internal rebuttal experiments unless the authors clarify licensing.

## Method-Defining Components

- Output parameterization for count loss: `torch.sigmoid(logits_u_w)`.
- Prediction/evaluation rule: `torch.softmax(logits, dim=1)` followed by argmax.
- Count likelihood: one classwise Poisson-binomial count likelihood per class; integer targets are obtained by rounding bag proportions times bag size, with any count-sum mismatch assigned to the largest-proportion class.
- Loss scaling: per-class negative log count probabilities are summed for each bag; batched training uses `reduce="mean"` over bags.
- Optimizer: SGD with momentum `0.9`, Nesterov enabled, weight decay on parameters other than names containing `bn`.
- Default learning rate: `2.5e-3`.
- Numerical epsilon: README reports `1e-30`; upstream FFT implementation effectively clamps with a tiny probability floor.
- Warmup: linear warmup by default, `warmup_frac=0.08`, `warmup_lr=5e-5`, then cosine rule from `WarmupCosineLrScheduler`.
- EMA: supported by flags, but default `eval_ema=False`; therefore not used in our default matched adapter.
- Bag construction: official loaders support `random`, `cluster`, and `alphafirst`. Our matched protocol replaces only the loader with the already fixed CIFAR-10 bag manifests.
- Checkpointing in upstream script: logs test accuracy and periodically saves model state; for the rebuttal adapter, hidden labels are not used for model selection. We select by aggregate validation composite count NLL, tie-breaking by aggregate validation count MAE.

## Difference From Local `PVC` Count Component

The existing local row previously called `PVC` is not full official LLP-PVC. It is a local FS-Conv implementation of the classwise count-likelihood component.

Important differences:

- Local old count component uses `softmax(logits)` probabilities for the classwise count likelihood.
- Official LLP-PVC training path uses `sigmoid(logits)` probabilities for the count likelihood, then `softmax(logits)` for prediction/evaluation.
- Official learner includes SGD+Nesterov and warmup-cosine scheduling.
- Official code includes method-specific training bookkeeping and optional EMA.
- Official code has its own bag loader strategies; the matched rebuttal adapter replaces only the loader/backbone interface to force identical fixed bags and frozen features.

## Preserved Equivalence Result

The previously completed equivalence check validates only the shared count-likelihood component under matched parameterization. It is not a competitive baseline result.

| Quantity | Max absolute difference |
| --- | ---: |
| Forward loss | `4.44e-16` |
| Per-class PMF | `2.22e-16` |
| Logit gradient | `2.12e-16` |

The official full learner row must be produced by the matched adapter on seeds 0, 1, and 2; the old local count-component row must not be presented as full LLP-PVC.
