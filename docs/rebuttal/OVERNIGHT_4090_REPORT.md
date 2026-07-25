# Overnight 4090 Rebuttal Setup Report

Date: 2026-07-25 UTC.

## Branch

- Branch: `neurips26-rebuttal-overnight-4090`
- Initial pre-work checkpoint pushed: `cb9a49e`

## What Was Added

- `src/countmil/baselines/gaussian_amle.py`
  - Generalized Gaussian-AMLE moment and loss functions.
  - Supports categorical finite-support sums and signed Bernoulli sums.
  - Uses documented default epsilon `1e-4`.
  - Floors negative variance caused by floating-point roundoff.

- `tests/test_gaussian_amle.py`
  - Moment equivalence with exact FS-Conv PMFs.
  - Signed Bernoulli moment checks.
  - Finite-gradient checks.
  - Deterministic-atom stability.
  - Mask/padding behavior.
  - Batch/per-bag agreement.

- `scripts/rebuttal/train_gaussian_amle_scalar.py`
  - MNIST digit-sum and signed MNIST Gaussian-AMLE training.
  - Aggregate-only validation selection using Gaussian-AMLE loss.
  - Saves config, command, metadata, best/final checkpoints, metrics, and raw test predictions.

- `scripts/rebuttal/train_fixed_cifar10_features.py`
  - True fixed-bag CIFAR-10 feature-level protocol.
  - Caches frozen ImageNet ResNet-18 penultimate features.
  - Saves fixed bag manifests and manifest statistics.
  - Supports CE/KL proportion matching, FS-Conv count likelihood, and an official-code-inspired count component using sigmoid probabilities.

- `scripts/rebuttal/check_pvc_equivalence.py`
  - Numerical equivalence test between FS-Conv classwise count likelihood and a reference Poisson-binomial count component under matched softmax parameterization.

- `scripts/rebuttal/run_overnight_4090.py`
  - Sequential resumable runner.
  - Uses `CUDA_VISIBLE_DEVICES=0`.
  - Writes per-job stdout/stderr and `RUNNING` / `COMPLETED` / `FAILED` status files.
  - Continues to independent jobs after failures.

- `scripts/rebuttal/summarize_overnight_4090.py`
  - Reads completed runs and writes summary CSV/Markdown result tables.
  - Uses sample standard deviation when multiple seeds are available.
  - Distinguishes composite count NLL from joint histogram NLL.

- Audit documents:
  - `docs/rebuttal/PVC_AUDIT.md`
  - `docs/rebuttal/EXISTING_RESULT_AUDIT.md`

## Validation Completed

- Full unit test suite:
  - `PYTHONPATH=src .venv/bin/python -m pytest`
  - Result: `53 passed`

- Gaussian-AMLE smoke tests:
  - digit-sum, one epoch, train bags 16, seed 99
  - signed MNIST, one epoch, train bags 16, seed 99
  - Result: both completed and saved smoke outputs under `results/rebuttal/overnight_4090/`.

- PVC count-component equivalence:
  - Result file: `results/rebuttal/overnight_4090/pvc_equivalence.json`
  - Max forward loss diff: `4.44e-16`
  - Max per-class probability diff: `2.22e-16`
  - Max logit-gradient diff: `2.12e-16`

## CIFAR Fixed-Bag Status

The true fixed-bag CIFAR-10 code is implemented but was not smoke-tested because this server does not currently have a complete CIFAR-10 archive:

- present: `data/cifar-10-python.tar.gz.part`
- missing: `data/cifar-10-python.tar.gz`

No cached ResNet-18 ImageNet weights were found locally. Because project instructions say not to download datasets unless explicitly asked, the active overnight manifest does not include CIFAR jobs.

## Active Overnight Manifest

- File: `configs/rebuttal/overnight_4090_manifest.yaml`
- Jobs included:
  - Phase 0 equivalence check.
  - P1 Gaussian-AMLE MNIST digit-sum, n=10/train1000, seeds 0,1,2.
  - P2 Gaussian-AMLE signed MNIST, n=10/train1000, random and cancellation-heavy signs, seeds 0,1,2.

P3/P4 are deferred because the user instruction says not to start them before the fixed-bag experiment is prepared and smoke-tested.

Deferred jobs are recorded in:

- `configs/rebuttal/deferred_after_fixed_cifar_smoke.yaml`

## Manuscript Safety

No manuscript files under `paper/` were edited.
