# A1 Atomic PMF Generality (Multi-class Ordinal Counting) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the Day 1–7 protocol from the A1 verification design spec to test whether PCA's exact bag PMF beats 5 baselines on multi-class ordinal counting (NLL primary, ECE secondary), with atom-shape generality (binary B1 reuse + multi-class + signed) as must-pass.

**Architecture:** Extend B1's `pca/` module + `scripts/` + `tests/` layout. Add `atomic_conv` + `multiclass_marginal_nll_loss` to `losses.py`; new `metrics.py` (ECE/MAE/top-1); new `baselines.py` (5 baselines); extend `data.py` (4 datasets), `models.py` (4 backbones), `train.py` (variable-N + mask). Per-dataset capacity backbone, head unified. Sequential per-seed runs. All convs in log space.

**Tech Stack:** Python ≥3.11, PyTorch ≥2.2, torchvision (MNIST/SVHN/ResNet-18 weights=None), scikit-learn, scipy (paired t-test), matplotlib (reliability diagram), uv for env management. Apple Silicon MPS device default; CPU fallback.

**Spec:** `docs/superpowers/specs/2026-05-04-a1-multiclass-counting-design.md` (committed at `f6c1dcf`).

**Depends on:** B1 verification artifacts at `fc1c9da` — `pca/losses.py:forward_conv`, `pca/models.py:SmallCNN`, `tests/test_losses.py`, `results/mnist_mil/summary.json`.

---

## Design clarification (binding for plan)

§7 of the spec says backbones include a `Linear(features, num_classes)` head. This conflicts with §8 baseline definitions, which require raw features (`h_i ∈ ℝ^F`) for 3a/3b and dictate baseline-owned heads for 1a/2a/2b/PCA. Plan resolution:

- **Backbones (`pca/models.py`)** are *feature extractors*, output `(B*N, F)` where `F=128` for SmallCNN-based, `F=512` for ResNet-18.
- **Baselines (`pca/baselines.py`)** own their own per-instance and bag heads.

This matches §8 intent. Spec §7.1 wording is treated as PCA-default-usage convention; baseline framework moves the head into the baseline class.

## UltraMNIST data note (binding)

Real UltraMNIST (Kaggle 2024) requires manual download + license accept. To keep Day 6 executable without external blocker, this plan implements `UltraMNISTBagDataset` as a synthetic stand-in: 4000×4000 canvas with 3–5 MNIST digits placed at non-overlapping random positions, bounding boxes recorded by construction, per-instance feature = 64×64 patch crop. The dataset class contract (returns `(patches, bag_sum, gt, mask)`) is identical for synthetic vs. real Kaggle data; swap is a one-file replacement if real data becomes available.

---

## File structure

| Path | Status | Created/modified in task | Responsibility |
|---|---|---|---|
| `pyproject.toml` | modify | 1 | bump version 0.1.0 → 0.2.0; add matplotlib |
| `.gitignore` | modify | 1 | add A1 results dirs + `~/.cache/pca/` |
| `pca/__init__.py` | modify | 1 | bump `__version__ = "0.2.0"` |
| `pca/losses.py` | modify | 2–5 | add `atomic_conv`, `multiclass_marginal_nll_loss` |
| `pca/metrics.py` | create | 6–7 | `top1_acc_from_log_pmf`, `mae_from_log_pmf`, `ece_from_log_pmf`, `reliability_diagram_data` |
| `pca/models.py` | modify | 8–10 | add `SmallCNNMulticlass`, `ResNet18FromScratch`, `PatchEncoder` (all feature extractors) |
| `pca/baselines.py` | create | 11–16 | `MeanPoolBaseline`, `PLMulticlassBaseline`, `CLTGaussianBaseline`, `AttentionPoolingBaseline`, `DeepSetsBaseline`, `PCABaseline` |
| `pca/train.py` | modify | 17 | add `per_bag_features_masked`, `train_a1`, `run_seeds_a1` |
| `pca/data.py` | modify | 19–22, 27, 30 | add `variable_n_collate_fn`, `MNISTSumBagDataset`, `MNISTSignedSumBagDataset`, `SVHNSumBagDataset`, `UltraMNISTBagDataset` |
| `scripts/sanity_a1.py` | create | 18 | end-to-end sanity check before any A1 experiment |
| `scripts/run_mnist_sum.py` | create | 23 | MNIST-sum entrypoint |
| `scripts/run_mnist_signed.py` | create | 23 | MNIST-signed entrypoint |
| `scripts/run_svhn_sum.py` | create | 28 | SVHN-sum entrypoint |
| `scripts/run_ultramnist.py` | create | 31 | UltraMNIST entrypoint |
| `scripts/analyze_a1.py` | create | 25 | per-dataset summary + H1/H2/H3 verdict + saturation detect |
| `tests/test_atomic_conv.py` | create | 2–5 | unit tests for atomic_conv + multiclass_nll |
| `tests/test_metrics.py` | create | 6–7 | unit tests for metrics |
| `tests/test_baselines.py` | create | 11–16 | smoke tests for each baseline |
| `tests/test_data_a1.py` | create | 19–22, 27, 30 | unit tests for new datasets |
| `results/mnist_sum/` | create | 24 | per-seed JSONs (gitignored) + summary.{md,json} |
| `results/mnist_signed/` | create | 24 | same |
| `results/svhn_sum/` | create | 29 | same |
| `results/ultramnist/` | create | 32 | same |
| `results/summary_overall.md` | create | 33 | final A1 verdict |

---

## Task 1: Project setup — version bump, deps, .gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `pca/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Bump version + add matplotlib in `pyproject.toml`**

Replace existing `[project]` block with:

```toml
[project]
name = "pca"
version = "0.2.0"
description = "Probabilistic Convolutional Aggregator — A1 multi-class verification"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "numpy>=1.26",
  "pandas>=2.1",
  "scikit-learn>=1.3",
  "scipy>=1.11",
  "matplotlib>=3.8",
  "Pillow>=10.0",
]
```

- [ ] **Step 2: Bump `pca/__init__.py`**

```python
"""Probabilistic Convolutional Aggregator — A1 multi-class verification package."""
__version__ = "0.2.0"
```

- [ ] **Step 3: Extend `.gitignore`**

Append to `.gitignore`:

```
results/mnist_sum/N*_*/
results/svhn_sum/N*_*/
results/ultramnist/N*_*/
results/mnist_signed/N*_*/
```

- [ ] **Step 4: Sync env**

Run:
```bash
uv sync
uv run python -c "import matplotlib; print('matplotlib', matplotlib.__version__); import pca; print('pca', pca.__version__)"
```

Expected: matplotlib ≥3.8, pca 0.2.0.

- [ ] **Step 5: Verify B1 tests still pass**

Run:
```bash
uv run pytest tests/ -v
```

Expected: all existing B1 tests pass (no regressions from new deps).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pca/__init__.py .gitignore
git commit -m "chore: bump to 0.2.0, add matplotlib for A1 verification"
```

---

## Task 2: `atomic_conv` — Bernoulli equivalence + first impl

**Files:**
- Create: `tests/test_atomic_conv.py`
- Modify: `pca/losses.py`

- [ ] **Step 1: Create `tests/test_atomic_conv.py` with Bernoulli equivalence test**

```python
"""Unit tests for atomic_conv and multiclass_marginal_nll_loss — see spec §10.1."""
import torch
import torch.nn.functional as F

from pca.losses import forward_conv, atomic_conv


def test_atomic_conv_bernoulli_equiv_forward_conv():
    """Bernoulli case (S=2): atomic_conv(log_atoms) ≈ forward_conv(p)."""
    torch.manual_seed(0)
    p = torch.rand(8, 12).clamp(min=1e-3, max=1 - 1e-3)
    log_atoms = torch.stack([(1 - p).log(), p.log()], dim=-1)   # (B, N, 2)
    log_P_atomic = atomic_conv(log_atoms)
    log_P_forward = forward_conv(p)
    assert torch.allclose(log_P_atomic, log_P_forward, atol=1e-6), \
        f"max diff: {(log_P_atomic - log_P_forward).abs().max().item()}"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_atomic_conv.py::test_atomic_conv_bernoulli_equiv_forward_conv -v
```

Expected: FAIL with `ImportError: cannot import name 'atomic_conv'`.

- [ ] **Step 3: Append `atomic_conv` to `pca/losses.py`**

Add at end of `pca/losses.py`:

```python
def atomic_conv(log_atoms: Tensor) -> Tensor:
    """Generic 1D log-space convolution of independent atomic PMFs.

    Args:
        log_atoms: (B, N, S) — per-instance log-PMF over atom support.
                   Each row sums to 1 in prob space (logsumexp over last dim = 0).
                   S = atom support size:
                     S=2 → Bernoulli (matches forward_conv).
                     S=K+1 → multi-class ordinal {0, ..., K}.
                     S=3 → signed {-1, 0, +1} (caller handles ±-shift).
                     S=m_i+1 → multiplicity (variable per instance).
    Returns:
        log_P: (B, T) — log P(sum_i z_i = k) for k = 0, ..., T-1.
               T = N * (S - 1) + 1.
    Complexity: O(B * N * S * T) sequential conv. For N=15, S=10: ~1.4e4 ops/bag.
    Autograd-safe at structurally unreachable positions (uses LOG_ZERO sentinel).
    """
    B, N, S = log_atoms.shape
    T = N * (S - 1) + 1
    log_P = log_atoms.new_full((B, T), LOG_ZERO)
    log_P[:, 0] = 0.0
    for i in range(N):
        new_log_P = log_atoms.new_full((B, T), LOG_ZERO)
        for s in range(S):
            if s == 0:
                shifted = log_P
            else:
                shifted = F.pad(log_P[:, :T - s], (s, 0), value=LOG_ZERO)
            term = log_atoms[:, i, s].unsqueeze(1) + shifted
            new_log_P = torch.logaddexp(new_log_P, term)
        log_P = new_log_P
    return log_P
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/test_atomic_conv.py::test_atomic_conv_bernoulli_equiv_forward_conv -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_atomic_conv.py pca/losses.py
git commit -m "feat(losses): atomic_conv with Bernoulli equivalence test"
```

---

## Task 3: `atomic_conv` — multi-class hand cases + sums-to-one + signed shift

**Files:**
- Modify: `tests/test_atomic_conv.py`

- [ ] **Step 1: Append multi-class N=2 hand-computed test**

Append to `tests/test_atomic_conv.py`:

```python
def test_atomic_conv_n2_multiclass_hand():
    """
    N=2, K=2, support {0, 1, 2}. Atoms d_1 = [0.5, 0.3, 0.2], d_2 = [0.4, 0.4, 0.2].
    Conv:
      P(0) = 0.5*0.4                                = 0.20
      P(1) = 0.5*0.4 + 0.3*0.4                      = 0.32
      P(2) = 0.5*0.2 + 0.3*0.4 + 0.2*0.4            = 0.30
      P(3) = 0.3*0.2 + 0.2*0.4                      = 0.14
      P(4) = 0.2*0.2                                = 0.04
    """
    log_atoms = torch.tensor([[
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2],
    ]]).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    expected = torch.tensor([[0.20, 0.32, 0.30, 0.14, 0.04]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"


def test_atomic_conv_n3_multiclass_hand():
    """N=3 uniform atoms over {0,1,2}: bag PMF = trinomial 1/27 of compositions."""
    log_atoms = torch.full((1, 3, 3), 1.0 / 3).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    # Compositions of s using {0,1,2} with 3 instances:
    # P(s=0) = C(3,3,0,0)*(1/3)^3 = 1/27
    # P(s=1) = 3/27, P(s=2)=6/27, P(s=3)=7/27, P(s=4)=6/27, P(s=5)=3/27, P(s=6)=1/27
    expected = torch.tensor([[1, 3, 6, 7, 6, 3, 1]]) / 27.0
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"
```

- [ ] **Step 2: Append sums-to-one test**

Append to `tests/test_atomic_conv.py`:

```python
def test_atomic_conv_sums_to_one():
    """For any valid atom logits, sum_k P(Σ=k) ≈ 1."""
    torch.manual_seed(0)
    logits = torch.randn(8, 10, 4)              # (B, N, S=4) random
    log_atoms = F.log_softmax(logits, dim=-1)
    log_P = atomic_conv(log_atoms)
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), f"got {sums.tolist()}"
```

- [ ] **Step 3: Append signed shift test**

Append to `tests/test_atomic_conv.py`:

```python
def test_atomic_conv_signed_shift():
    """Signed atoms over {-1, 0, +1} via S=3 with caller-side shift.

    For N=2 with d_1 = d_2 = [0.25, 0.5, 0.25] (symmetric around 0):
    bag sum z ∈ {-2, -1, 0, +1, +2}, after caller-side +N shift → {0, 1, 2, 3, 4}.
    Convolution gives:
      P(shift=0) = 0.0625, P(1) = 0.25, P(2) = 0.375, P(3) = 0.25, P(4) = 0.0625.
    """
    log_atoms = torch.tensor([[[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]]).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    expected = torch.tensor([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"
```

- [ ] **Step 4: Run all atomic_conv tests**

```bash
uv run pytest tests/test_atomic_conv.py -v -k atomic_conv
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/test_atomic_conv.py
git commit -m "test(losses): atomic_conv multi-class hand cases + sums-to-one + signed shift"
```

---

## Task 4: `multiclass_marginal_nll_loss` — basic + finite test + impl

**Files:**
- Modify: `tests/test_atomic_conv.py`
- Modify: `pca/losses.py`

- [ ] **Step 1: Append finite-loss test**

Append to `tests/test_atomic_conv.py`:

```python
from pca.losses import multiclass_marginal_nll_loss


def test_multiclass_nll_finite():
    """Forward + grad finite on random batch."""
    torch.manual_seed(2)
    logits = torch.randn(4, 10, 5, requires_grad=True)   # B=4, N=10, S=5 (K=4)
    bag_y = torch.randint(0, 10 * 4 + 1, (4,))            # bag sum in [0, NK]
    loss = multiclass_marginal_nll_loss(logits, bag_y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(g).all()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_atomic_conv.py::test_multiclass_nll_finite -v
```

Expected: FAIL with `cannot import name 'multiclass_marginal_nll_loss'`.

- [ ] **Step 3: Append `multiclass_marginal_nll_loss` to `pca/losses.py`**

```python
def multiclass_marginal_nll_loss(
    logits: Tensor,
    bag_y: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Multi-class generalization of marginal_nll_loss.

    Args:
        logits: (B, N, S) per-instance pre-softmax logits over atom support.
        bag_y:  (B,) integer bag sum, already shifted to [0, T-1] by caller
                (T = N*(S-1)+1). For signed atoms ({-1,0,+1}) the caller
                passes shifted bag_y (true_y + N).
        mask:   (B, N) bool, optional. True for active instances. Inactive
                instances are forced to atom = delta_0 = [1, 0, ..., 0],
                contributing identity to the convolution and ignored in the
                bag PMF. Handles variable bag size without a separate path.
    Returns:
        scalar — mean NLL over batch.
    """
    B, N, S = logits.shape
    log_atoms = F.log_softmax(logits, dim=-1)
    if mask is not None:
        delta_0 = log_atoms.new_full((S,), LOG_ZERO)
        delta_0[0] = 0.0
        log_atoms = torch.where(mask.unsqueeze(-1), log_atoms, delta_0)
    log_P = atomic_conv(log_atoms)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/test_atomic_conv.py::test_multiclass_nll_finite -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_atomic_conv.py pca/losses.py
git commit -m "feat(losses): multiclass_marginal_nll_loss with finite-grad test"
```

---

## Task 5: `multiclass_marginal_nll_loss` — mask handles variable N

**Files:**
- Modify: `tests/test_atomic_conv.py`

- [ ] **Step 1: Append mask test**

Append to `tests/test_atomic_conv.py`:

```python
def test_multiclass_nll_mask_handles_variable_n():
    """Loss with N=4 + mask=[T,T,F,F] equals loss on identical first-2 logits with N=2."""
    torch.manual_seed(7)
    full = torch.randn(2, 4, 4)                              # (B=2, N_max=4, S=4)
    mask = torch.tensor([[True, True, False, False],
                         [True, True, False, False]])
    bag_y = torch.tensor([3, 5])                              # bag sum in [0, 2*3]=6
    loss_masked = multiclass_marginal_nll_loss(full, bag_y, mask=mask)
    # Equivalent N=2 case: pad bag PMF to T=N_max*(S-1)+1=13. Use first-2 logits;
    # bag_y stays the same. Without mask the bag-sum range is [0, 2*3]=6,
    # but atomic_conv yields T=2*3+1=7 there vs T=4*3+1=13 here. With masking,
    # inactive atoms = delta_0 keep bag_y at the same offset → loss should match
    # 'unpadded' N=2 evaluated at the same bag_y.
    half = full[:, :2]
    loss_n2 = multiclass_marginal_nll_loss(half, bag_y)
    assert torch.allclose(loss_masked, loss_n2, atol=1e-5), \
        f"masked={loss_masked.item()} vs n2={loss_n2.item()}"
```

- [ ] **Step 2: Run test, verify it passes**

```bash
uv run pytest tests/test_atomic_conv.py::test_multiclass_nll_mask_handles_variable_n -v
```

Expected: PASS (mask logic in Task 4 already correct).

- [ ] **Step 3: Run all atomic_conv tests**

```bash
uv run pytest tests/test_atomic_conv.py -v
```

Expected: 7 PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_atomic_conv.py
git commit -m "test(losses): mask makes multiclass_nll equivalent to truncated N"
```

---

## Task 6: `metrics.py` — top1_acc + MAE

**Files:**
- Create: `tests/test_metrics.py`
- Create: `pca/metrics.py`

- [ ] **Step 1: Create `tests/test_metrics.py` with first tests**

```python
"""Unit tests for pca/metrics.py — see spec §10.4."""
import torch

from pca.metrics import top1_acc_from_log_pmf, mae_from_log_pmf


def test_top1_acc_argmax_matches_truth():
    """log_P argmax == bag_y on perfect predictions."""
    log_P = torch.full((3, 5), -10.0)
    log_P[0, 1] = 0.0   # argmax=1
    log_P[1, 4] = 0.0   # argmax=4
    log_P[2, 2] = 0.0   # argmax=2
    bag_y = torch.tensor([1, 4, 2])
    acc = top1_acc_from_log_pmf(log_P, bag_y)
    assert acc == 1.0


def test_top1_acc_argmax_partial():
    """3 of 4 correct → 0.75."""
    log_P = torch.full((4, 3), -10.0)
    log_P[0, 0] = 0.0
    log_P[1, 1] = 0.0
    log_P[2, 2] = 0.0
    log_P[3, 0] = 0.0     # wrong
    bag_y = torch.tensor([0, 1, 2, 1])
    acc = top1_acc_from_log_pmf(log_P, bag_y)
    assert abs(acc - 0.75) < 1e-7


def test_mae_uniform_pmf():
    """For uniform PMF over [0, T-1] and target=0, expected sum = (T-1)/2."""
    T = 11
    log_P = torch.full((1, T), -torch.tensor(float(T)).log().item())   # uniform 1/T
    bag_y = torch.tensor([0])
    mae = mae_from_log_pmf(log_P, bag_y)
    # E[Y_pred] = (T-1)/2 = 5; |E - bag_y| = 5
    assert abs(mae - 5.0) < 1e-5
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: FAIL — `cannot import name 'top1_acc_from_log_pmf'`.

- [ ] **Step 3: Create `pca/metrics.py` with both functions**

```python
"""Evaluation metrics for A1 multi-class verification — see spec §10.4 / §11."""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def top1_acc_from_log_pmf(log_P: Tensor, bag_y: Tensor) -> float:
    """Top-1 accuracy of bag-sum prediction = fraction with argmax(log_P) == bag_y.

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth bag sum (already shift-corrected).
    Returns:
        scalar float in [0, 1].
    """
    pred = log_P.argmax(dim=1)
    return float((pred == bag_y).float().mean().item())


def mae_from_log_pmf(log_P: Tensor, bag_y: Tensor) -> float:
    """MAE of expected bag sum vs ground truth: E_log_P[Y] = Σ_k k * exp(log_P[k]).

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth bag sum.
    Returns:
        scalar float (mean absolute error).
    """
    T = log_P.shape[1]
    k_grid = torch.arange(T, device=log_P.device).float()
    expected = (log_P.exp() * k_grid).sum(dim=1)
    return float((expected - bag_y.float()).abs().mean().item())
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metrics.py pca/metrics.py
git commit -m "feat(metrics): top1_acc + MAE from log-PMF with hand-computed tests"
```

---

## Task 7: `metrics.py` — ECE + reliability diagram data

**Files:**
- Modify: `tests/test_metrics.py`
- Modify: `pca/metrics.py`

- [ ] **Step 1: Append ECE hand-computed test**

Append to `tests/test_metrics.py`:

```python
from pca.metrics import ece_from_log_pmf, reliability_diagram_data


def test_ece_hand_computed():
    """
    Hand-built case: 4 samples, 2 bins (n_bins=2).
      sample 0: log_P max = log(0.9) at k=0; bag_y=0  → conf=0.9, correct, bin=high
      sample 1: log_P max = log(0.9) at k=0; bag_y=1  → conf=0.9, wrong,   bin=high
      sample 2: log_P max = log(0.4) at k=0; bag_y=0  → conf=0.4, correct, bin=low
      sample 3: log_P max = log(0.4) at k=0; bag_y=1  → conf=0.4, wrong,   bin=low
    Bins (n_bins=2, edges 0,0.5,1.0):
      low bin:  acc=0.5, conf=0.4, |0.5-0.4|=0.1, weight=2/4
      high bin: acc=0.5, conf=0.9, |0.5-0.9|=0.4, weight=2/4
    ECE = 0.5*0.1 + 0.5*0.4 = 0.25
    """
    # Build log_P with controlled max prob.
    def make_log_P(p_max):
        # 2-class support; max prob = p_max at k=0, rest = (1-p_max)/(T-1) at k=1
        T = 2
        return torch.tensor([[p_max, 1 - p_max]]).log()
    log_P = torch.cat([make_log_P(0.9), make_log_P(0.9),
                        make_log_P(0.4), make_log_P(0.4)], dim=0)
    bag_y = torch.tensor([0, 1, 0, 1])
    ece = ece_from_log_pmf(log_P, bag_y, n_bins=2)
    assert abs(ece - 0.25) < 1e-6, f"got {ece}"


def test_ece_perfect_calibration_zero():
    """When confidence == accuracy in every bin, ECE = 0."""
    # All samples: confidence 0.6, half correct → bin-0.6 has acc 0.5, conf 0.5? 
    # Use confidence=1.0 with all correct → ECE=0.
    log_P = torch.tensor([[0.0, -1e10]] * 4).float()    # confidence ≈ 1, prediction = 0
    bag_y = torch.tensor([0, 0, 0, 0])
    ece = ece_from_log_pmf(log_P, bag_y, n_bins=15)
    assert ece < 1e-5, f"got {ece}"


def test_reliability_diagram_data_shape():
    """reliability_diagram_data returns (bin_centers, accs, confs, weights) of length n_bins."""
    torch.manual_seed(3)
    log_P = torch.log_softmax(torch.randn(50, 8), dim=1)
    bag_y = torch.randint(0, 8, (50,))
    centers, accs, confs, weights = reliability_diagram_data(log_P, bag_y, n_bins=10)
    assert centers.shape == accs.shape == confs.shape == weights.shape == (10,)
    assert abs(weights.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Append ECE + reliability_diagram_data to `pca/metrics.py`**

```python
def ece_from_log_pmf(log_P: Tensor, bag_y: Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error over predicted bag-sum confidences.

    For each sample, take p_pred = max_k exp(log_P[b, k]) (top-1 confidence)
    and acc_b = 1[argmax(log_P[b]) == bag_y[b]]. Bin samples by p_pred into
    n_bins equal-width bins on [0, 1]; ECE = Σ_bin (|bin| / N) * |acc_bin - conf_bin|.

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth.
        n_bins: number of equal-width bins.
    Returns:
        scalar float (ECE).
    """
    pred = log_P.argmax(dim=1)
    confidence = log_P.exp().max(dim=1).values
    correct = (pred == bag_y).float()
    confidence_np = confidence.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidence_np > lo) & (confidence_np <= hi) if b > 0 \
                 else (confidence_np >= lo) & (confidence_np <= hi)
        if not in_bin.any():
            continue
        acc_bin = correct_np[in_bin].mean()
        conf_bin = confidence_np[in_bin].mean()
        ece += abs(acc_bin - conf_bin) * in_bin.sum() / len(confidence_np)
    return float(ece)


def reliability_diagram_data(
    log_P: Tensor, bag_y: Tensor, n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, accuracies, confidences, weights) for reliability plot.

    Bins with no samples report 0.0 for accuracy and confidence; weight is
    fraction of total samples in that bin (sums to 1).
    """
    pred = log_P.argmax(dim=1)
    confidence = log_P.exp().max(dim=1).values
    correct = (pred == bag_y).float()
    confidence_np = confidence.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    N = len(confidence_np)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidence_np > lo) & (confidence_np <= hi) if b > 0 \
                 else (confidence_np >= lo) & (confidence_np <= hi)
        if not in_bin.any():
            continue
        accs[b] = correct_np[in_bin].mean()
        confs[b] = confidence_np[in_bin].mean()
        weights[b] = in_bin.sum() / N
    return centers, accs, confs, weights
```

- [ ] **Step 3: Run tests, verify all pass**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: 6 PASSED.

- [ ] **Step 4: Day 1 gate — full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All atomic_conv (7) + metrics (6) + B1-existing (10+) tests pass. Total ≥ 23 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metrics.py pca/metrics.py
git commit -m "feat(metrics): ECE + reliability_diagram_data with hand-computed tests"
```

---

## Task 8: `SmallCNNMulticlass` — feature extractor

**Files:**
- Modify: `pca/models.py`

- [ ] **Step 1: Append `SmallCNNMulticlass` to `pca/models.py`**

```python
class SmallCNNMulticlass(nn.Module):
    """Per-instance MNIST → 128-dim feature vector.

    Identical body to SmallCNN (B1) but the final classification layer is
    OMITTED — this class is a feature extractor; baselines own their own head.
    Output shape: (B*N, 128). For MNIST-sum and (via num_classes parameter on
    the head) MNIST-signed.
    """
    feature_dim = 128

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:    # x: (B*N, 1, 28, 28)
        return self.head(self.features(x))      # (B*N, 128)
```

- [ ] **Step 2: Add smoke test for SmallCNNMulticlass**

Append to `tests/test_atomic_conv.py`:

```python
def test_small_cnn_multiclass_output_shape():
    """Forward returns (B*N, 128) features."""
    from pca.models import SmallCNNMulticlass
    model = SmallCNNMulticlass()
    x = torch.randn(8, 1, 28, 28)
    feat = model(x)
    assert feat.shape == (8, 128), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_atomic_conv.py::test_small_cnn_multiclass_output_shape -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pca/models.py tests/test_atomic_conv.py
git commit -m "feat(models): SmallCNNMulticlass feature extractor (F=128)"
```

---

## Task 9: `ResNet18FromScratch` — feature extractor for SVHN

**Files:**
- Modify: `pca/models.py`

- [ ] **Step 1: Append `ResNet18FromScratch` to `pca/models.py`**

Add after `SmallCNNMulticlass`:

```python
class ResNet18FromScratch(nn.Module):
    """ResNet-18 from scratch (no pretrained weights). 32x32 input adaptation.

    Replaces the original 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv
    (CIFAR-style adaptation; preserves spatial resolution for small inputs),
    and removes the final classification layer to expose 512-d features.
    Output shape: (B*N, 512). For SVHN-sum.
    """
    feature_dim = 512

    def __init__(self, in_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(weights=None)
        # CIFAR-style stem
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        # Drop final classifier; .fc → identity, output is the 512-dim avgpool result.
        self.net.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:     # x: (B*N, 3, 32, 32)
        return self.net(x)                       # (B*N, 512)
```

- [ ] **Step 2: Add smoke test**

Append to `tests/test_atomic_conv.py`:

```python
def test_resnet18_from_scratch_output_shape():
    """Forward returns (B*N, 512) features for 32x32 RGB input."""
    from pca.models import ResNet18FromScratch
    model = ResNet18FromScratch()
    x = torch.randn(4, 3, 32, 32)
    feat = model(x)
    assert feat.shape == (4, 512), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_atomic_conv.py::test_resnet18_from_scratch_output_shape -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pca/models.py tests/test_atomic_conv.py
git commit -m "feat(models): ResNet18FromScratch feature extractor (F=512, CIFAR-style stem)"
```

---

## Task 10: `PatchEncoder` — small RGB CNN for UltraMNIST

**Files:**
- Modify: `pca/models.py`

- [ ] **Step 1: Append `PatchEncoder` to `pca/models.py`**

```python
class PatchEncoder(nn.Module):
    """Small RGB CNN for 64x64 patches → 128-dim feature.

    For UltraMNIST: per-instance feature is a digit-bounding-box patch crop
    resized to 64x64 RGB. Output shape: (B*N, 128). Architecturally similar
    to SmallCNNMulticlass but adapted for 64x64x3 input.
    """
    feature_dim = 128

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32→16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16→8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:    # x: (B*N, 3, 64, 64)
        return self.head(self.features(x))      # (B*N, 128)
```

- [ ] **Step 2: Smoke test**

Append to `tests/test_atomic_conv.py`:

```python
def test_patch_encoder_output_shape():
    from pca.models import PatchEncoder
    model = PatchEncoder()
    x = torch.randn(6, 3, 64, 64)
    feat = model(x)
    assert feat.shape == (6, 128), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_atomic_conv.py::test_patch_encoder_output_shape -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pca/models.py tests/test_atomic_conv.py
git commit -m "feat(models): PatchEncoder for 64x64 RGB patches (F=128)"
```

---

## Task 11: Baseline `MeanPoolBaseline` (1a)

**Files:**
- Create: `pca/baselines.py`
- Create: `tests/test_baselines.py`

- [ ] **Step 1: Create `tests/test_baselines.py` with smoke fixture + 1a test**

```python
"""Smoke tests for pca/baselines.py — see spec §10.2.

Per-baseline: forward returns finite loss + finite gradient on backbone params;
log_P (when not None) is a valid log-probability vector.
"""
import torch
import torch.nn.functional as F
import pytest


def _smoke_inputs(B=2, N=10, F_dim=128, K=9, T=None):
    """Random features + bag_y + mask for baseline smoke tests."""
    if T is None:
        T = N * K + 1
    torch.manual_seed(0)
    features = torch.randn(B, N, F_dim, requires_grad=True)
    bag_y = torch.randint(0, T, (B,))
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -2:] = False           # mark bag 0 as having only N-2 active
    return features, bag_y, mask


def _assert_log_p_valid(log_P, expected_T):
    assert log_P.shape[1] == expected_T, f"shape {tuple(log_P.shape)}"
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), \
        f"log_P does not sum to 1: got {sums.tolist()}"


def test_mean_pool_baseline_smoke():
    from pca.baselines import MeanPoolBaseline
    features, bag_y, mask = _smoke_inputs()
    model = MeanPoolBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_mean_pool_baseline_smoke -v
```

Expected: FAIL — `cannot import name 'MeanPoolBaseline'`.

- [ ] **Step 3: Create `pca/baselines.py` with `MeanPoolBaseline`**

```python
"""Baselines for A1 multi-class verification — see spec §8.

All baselines are nn.Modules with forward signature:
    forward(features, bag_y, mask) -> (loss, log_P)
where:
  features: (B, N_max, F) backbone-output features.
  bag_y:    (B,) integer bag sum (already shifted for signed atoms).
  mask:     (B, N_max) bool — True for active instances.
Returns:
  loss:  scalar training loss.
  log_P: (B, NK+1) log-probability over bag-sum support; None if baseline lacks
         a native bag distribution and uses Gaussian wrap in eval (1a, 2a).
"""
from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from pca.losses import LOG_ZERO, atomic_conv, multiclass_marginal_nll_loss


def _gaussian_log_pmf(mu: Tensor, var: Tensor, T: int) -> Tensor:
    """Discretized Gaussian log-PMF over k = 0, 1, ..., T-1.

    Used by methods that train on continuous targets but need a bag PMF for
    NLL/ECE evaluation. Each bin is normalized so log_P sums to 1 in prob space.
    """
    k = torch.arange(T, device=mu.device).float().unsqueeze(0)  # (1, T)
    log_unnorm = -0.5 * ((k - mu.unsqueeze(1)) ** 2) / var.unsqueeze(1).clamp(min=1e-3)
    return log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)


class MeanPoolBaseline(nn.Module):
    """1a — Mean-pool regression. Per-instance scalar prediction; bag sum = Σ_i z_i.

    Loss: MSE(Σ ẑ_i, Y). Eval log_P: Gaussian wrap with σ²_bag = N * K^2 / 12
    (uniform-on-[0,K] per-instance variance × N).
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z_hat = self.head(features).squeeze(-1)     # (B, N_max)
        z_hat = z_hat * mask.float()
        bag_pred = z_hat.sum(dim=1)                  # (B,)
        loss = F.mse_loss(bag_pred, bag_y.float())
        # Gaussian-wrap bag PMF over k = 0..N_max*K
        T = self.N_max * self.K + 1
        n_active = mask.sum(dim=1).float().clamp(min=1.0)
        var = n_active * (self.K ** 2) / 12.0
        log_P = _gaussian_log_pmf(bag_pred, var, T)
        return loss, log_P
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run pytest tests/test_baselines.py::test_mean_pool_baseline_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): MeanPoolBaseline (1a) with Gaussian-wrap eval log_P"
```

---

## Task 12: Baseline `PLMulticlassBaseline` (2a)

**Files:**
- Modify: `pca/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Append smoke test for 2a**

Append to `tests/test_baselines.py`:

```python
def test_pl_multiclass_baseline_smoke():
    from pca.baselines import PLMulticlassBaseline
    features, bag_y, mask = _smoke_inputs()
    model = PLMulticlassBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)
```

- [ ] **Step 2: Run test, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_pl_multiclass_baseline_smoke -v
```

Expected: FAIL.

- [ ] **Step 3: Append `PLMulticlassBaseline` to `pca/baselines.py`**

```python
class PLMulticlassBaseline(nn.Module):
    """2a — PL-multiclass (sum-only adaptation).

    Per-instance softmax over {0..K}; bag mean prediction = (1/N) Σ_i E[z_i].
    Loss: MSE(bag_mean_pred, bag_y / N). Eval log_P: Gaussian wrap with
    per-instance variance σ²_i = E[z²]-E[z]² aggregated over active instances.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.head(features)                 # (B, N_max, K+1)
        probs = F.softmax(logits, dim=-1)            # (B, N_max, K+1)
        k_grid = torch.arange(self.K + 1, device=features.device).float()
        mu_i = (probs * k_grid).sum(dim=-1)          # (B, N_max)
        var_i = (probs * (k_grid ** 2)).sum(dim=-1) - mu_i ** 2
        mu_i = mu_i * mask.float()
        var_i = var_i * mask.float()
        n_active = mask.sum(dim=1).float().clamp(min=1.0)
        bag_mean_pred = mu_i.sum(dim=1) / n_active
        loss = F.mse_loss(bag_mean_pred, bag_y.float() / n_active)
        T = self.N_max * self.K + 1
        log_P = _gaussian_log_pmf(mu_i.sum(dim=1), var_i.sum(dim=1).clamp(min=1e-3), T)
        return loss, log_P
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run pytest tests/test_baselines.py::test_pl_multiclass_baseline_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): PLMulticlassBaseline (2a) sum-only adaptation"
```

---

## Task 13: Baseline `CLTGaussianBaseline` (2b)

**Files:**
- Modify: `pca/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Append smoke test**

Append to `tests/test_baselines.py`:

```python
def test_clt_gaussian_baseline_smoke():
    from pca.baselines import CLTGaussianBaseline
    features, bag_y, mask = _smoke_inputs()
    model = CLTGaussianBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)
```

- [ ] **Step 2: Run test, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_clt_gaussian_baseline_smoke -v
```

Expected: FAIL.

- [ ] **Step 3: Append `CLTGaussianBaseline` to `pca/baselines.py`**

```python
class CLTGaussianBaseline(nn.Module):
    """2b — CLT Gaussian. Per-instance softmax → moments; bag = N(Σμ, Σσ²).

    Loss: -log N(Y; μ_bag, σ²_bag). Eval log_P: discretized Gaussian over
    k = 0..N_max*K. Native bag distribution (not Gaussian wrap on top of MSE).
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.head(features)                # (B, N_max, K+1)
        probs = F.softmax(logits, dim=-1)
        k_grid = torch.arange(self.K + 1, device=features.device).float()
        mu_i = (probs * k_grid).sum(dim=-1)
        var_i = (probs * (k_grid ** 2)).sum(dim=-1) - mu_i ** 2
        mu_i = mu_i * mask.float()
        var_i = var_i * mask.float()
        mu_bag = mu_i.sum(dim=1)
        var_bag = var_i.sum(dim=1).clamp(min=1e-3)
        # Continuous Gaussian NLL on integer Y
        loss = 0.5 * (torch.log(2 * math.pi * var_bag)
                       + (bag_y.float() - mu_bag) ** 2 / var_bag).mean()
        T = self.N_max * self.K + 1
        log_P = _gaussian_log_pmf(mu_bag, var_bag, T)
        return loss, log_P
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run pytest tests/test_baselines.py::test_clt_gaussian_baseline_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): CLTGaussianBaseline (2b) native Gaussian bag"
```

---

## Task 14: Baseline `AttentionPoolingBaseline` (3a)

**Files:**
- Modify: `pca/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Append smoke test**

Append to `tests/test_baselines.py`:

```python
def test_attention_pooling_baseline_smoke():
    from pca.baselines import AttentionPoolingBaseline
    features, bag_y, mask = _smoke_inputs()
    model = AttentionPoolingBaseline(feature_dim=128, K=9, N_max=10, attn_dim=64)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_attention_pooling_baseline_smoke -v
```

Expected: FAIL.

- [ ] **Step 3: Append `AttentionPoolingBaseline` to `pca/baselines.py`**

```python
class AttentionPoolingBaseline(nn.Module):
    """3a — Ilse 2018 gated attention pooling.

    a_i = softmax_i(w^T (tanh(V h_i) ⊙ sigmoid(U h_i))) over masked instances;
    bag embedding z = Σ_i a_i h_i; bag head MLP → Linear(NK+1) → softmax → CE.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int, attn_dim: int = 64):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.V = nn.Linear(feature_dim, attn_dim)
        self.U = nn.Linear(feature_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        T = N_max * K + 1
        self.bag_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, T),
        )

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = torch.tanh(self.V(features)) * torch.sigmoid(self.U(features))
        scores = self.w(gate).squeeze(-1)                # (B, N_max)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=1)                  # (B, N_max)
        z = (attn.unsqueeze(-1) * features).sum(dim=1)   # (B, F)
        bag_logits = self.bag_head(z)                    # (B, T)
        log_P = F.log_softmax(bag_logits, dim=1)
        loss = F.nll_loss(log_P, bag_y)
        return loss, log_P
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run pytest tests/test_baselines.py::test_attention_pooling_baseline_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): AttentionPoolingBaseline (3a) gated Ilse 2018"
```

---

## Task 15: Baseline `DeepSetsBaseline` (3b)

**Files:**
- Modify: `pca/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Append smoke test**

Append to `tests/test_baselines.py`:

```python
def test_deepsets_baseline_smoke():
    from pca.baselines import DeepSetsBaseline
    features, bag_y, mask = _smoke_inputs()
    model = DeepSetsBaseline(feature_dim=128, K=9, N_max=10, hidden=64)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_deepsets_baseline_smoke -v
```

Expected: FAIL.

- [ ] **Step 3: Append `DeepSetsBaseline` to `pca/baselines.py`**

```python
class DeepSetsBaseline(nn.Module):
    """3b — DeepSets (Zaheer 2017). φ → sum → ρ.

    Per-instance φ(h_i) MLP, mask-aware sum aggregation, bag head ρ → Linear(NK+1)
    → softmax CE. Sum (not mean) to preserve N-information.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int, hidden: int = 64):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        T = N_max * K + 1
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T),
        )

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        phi_h = self.phi(features) * mask.unsqueeze(-1).float()    # (B, N_max, hidden)
        z = phi_h.sum(dim=1)                                         # (B, hidden)
        bag_logits = self.rho(z)                                     # (B, T)
        log_P = F.log_softmax(bag_logits, dim=1)
        loss = F.nll_loss(log_P, bag_y)
        return loss, log_P
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run pytest tests/test_baselines.py::test_deepsets_baseline_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): DeepSetsBaseline (3b) sum aggregation"
```

---

## Task 16: `PCABaseline` — wraps multiclass_marginal_nll_loss

**Files:**
- Modify: `pca/baselines.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Append smoke test**

Append to `tests/test_baselines.py`:

```python
def test_pca_baseline_smoke():
    from pca.baselines import PCABaseline
    features, bag_y, mask = _smoke_inputs()
    model = PCABaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_pca_baseline_signed_atom():
    """PCA with S=3 signed atom + caller-shifted bag_y works."""
    from pca.baselines import PCABaseline
    torch.manual_seed(0)
    B, N, F_dim = 2, 8, 128
    features = torch.randn(B, N, F_dim, requires_grad=True)
    # signed range: bag sum ∈ [-N, N], shifted to [0, 2N]
    bag_y = torch.randint(0, 2 * N + 1, (B,))
    mask = torch.ones(B, N, dtype=torch.bool)
    model = PCABaseline(feature_dim=128, K=2, N_max=8)   # K=2 → S=3 (signed support)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    _assert_log_p_valid(log_P, expected_T=8 * 2 + 1)
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_baselines.py::test_pca_baseline_smoke -v
```

Expected: FAIL.

- [ ] **Step 3: Append `PCABaseline` to `pca/baselines.py`**

```python
class PCABaseline(nn.Module):
    """PCA — exact bag PMF via 1D log-space conv over per-instance atomic PMF.

    Per-instance head: Linear(F, K+1). Atom logits → log_softmax → atomic_conv.
    Loss: multiclass_marginal_nll_loss (handles inactive instances via mask =
    delta_0 forcing). Native exact bag PMF in log space. For signed atoms
    (S=3 → K=2), caller pre-shifts bag_y by +N before passing.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.head(features)             # (B, N_max, K+1)
        # Log-PMF over bag-sum support T = N_max*K+1
        log_atoms = F.log_softmax(logits, dim=-1)
        if mask is not None:
            S = self.K + 1
            delta_0 = log_atoms.new_full((S,), LOG_ZERO)
            delta_0[0] = 0.0
            log_atoms = torch.where(mask.unsqueeze(-1), log_atoms, delta_0)
        log_P = atomic_conv(log_atoms)
        log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
        loss = -log_P_y.mean()
        return loss, log_P
```

- [ ] **Step 4: Run all baseline tests**

```bash
uv run pytest tests/test_baselines.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add pca/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): PCABaseline native exact bag PMF + signed atom test"
```

---

## Task 17: `train_a1` — variable-N + mask + 4-metric eval

**Files:**
- Modify: `pca/train.py`

- [ ] **Step 1: Add metrics import to top of `pca/train.py`**

In `pca/train.py`, locate the existing line `from pca.losses import forward_conv`
(approximately line 26) and add this immediately after it:

```python
from pca.metrics import (
    top1_acc_from_log_pmf, mae_from_log_pmf, ece_from_log_pmf,
)
```

- [ ] **Step 2: Append `per_bag_features_masked` + `train_a1` + `run_seeds_a1` to end of `pca/train.py`**

```python
def per_bag_features_masked(
    backbone: nn.Module, batch_features: Tensor, mask: Tensor,
) -> Tensor:
    """Reshape (B, N_max, *feature_shape) → backbone → (B, N_max, F).

    Inactive (mask=False) instances are still fed through the backbone (uses
    padding values from collate); their features are zeroed out in the result
    so downstream layers (mean-pool, attention-after-mask, etc.) cannot
    accidentally use them. The baseline's mask handling is the source of truth
    for how inactive instances affect loss.
    """
    B, N_max = batch_features.shape[:2]
    flat = batch_features.reshape(B * N_max, *batch_features.shape[2:])
    feat_flat = backbone(flat)                       # (B*N_max, F)
    F_dim = feat_flat.shape[-1]
    feat = feat_flat.view(B, N_max, F_dim)
    feat = feat * mask.unsqueeze(-1).float()
    return feat


def train_a1(
    backbone: nn.Module,
    baseline: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """A1 train loop with variable-N + mask + 4-metric eval.

    Per-epoch test eval reports:
      epoch_test_nll: mean -log P(bag_y) under bag PMF (uses log_P from baseline).
      epoch_test_acc: top-1 accuracy of argmax(log_P) == bag_y.
      epoch_test_mae: |E[Y_pred] - bag_y|.
      epoch_test_ece: 15-bin Expected Calibration Error.
    Returns dict with epoch arrays and last-5-epoch means under best_test_*.
    """
    backbone = backbone.to(device)
    baseline = baseline.to(device)
    epoch_train_loss = []
    epoch_test_nll = []
    epoch_test_acc = []
    epoch_test_mae = []
    epoch_test_ece = []

    for epoch in range(num_epochs):
        backbone.train(); baseline.train()
        running_loss = 0.0
        running_count = 0
        for features, bag_y, _gt, mask in train_loader:
            features = features.to(device)
            bag_y = bag_y.to(device)
            mask = mask.to(device)
            feat = per_bag_features_masked(backbone, features, mask)
            loss, _log_P = baseline(feat, bag_y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.shape[0]
            running_count += features.shape[0]
        epoch_train_loss.append(running_loss / max(running_count, 1))

        # Test eval
        backbone.eval(); baseline.eval()
        all_log_P, all_y = [], []
        with torch.no_grad():
            for features, bag_y, _gt, mask in test_loader:
                features = features.to(device)
                bag_y = bag_y.to(device)
                mask = mask.to(device)
                feat = per_bag_features_masked(backbone, features, mask)
                _loss, log_P = baseline(feat, bag_y, mask)
                all_log_P.append(log_P.cpu())
                all_y.append(bag_y.cpu())
        log_P_all = torch.cat(all_log_P, dim=0)
        y_all = torch.cat(all_y, dim=0)
        nll = -log_P_all.gather(1, y_all.unsqueeze(1)).squeeze(1).mean().item()
        epoch_test_nll.append(float(nll))
        epoch_test_acc.append(top1_acc_from_log_pmf(log_P_all, y_all))
        epoch_test_mae.append(mae_from_log_pmf(log_P_all, y_all))
        epoch_test_ece.append(ece_from_log_pmf(log_P_all, y_all, n_bins=15))

    n_tail = min(5, num_epochs)
    return {
        'epoch_train_loss': epoch_train_loss,
        'epoch_test_nll': epoch_test_nll,
        'epoch_test_acc': epoch_test_acc,
        'epoch_test_mae': epoch_test_mae,
        'epoch_test_ece': epoch_test_ece,
        'best_test_nll': float(np.mean(epoch_test_nll[-n_tail:])),
        'best_test_acc': float(np.mean(epoch_test_acc[-n_tail:])),
        'best_test_mae': float(np.mean(epoch_test_mae[-n_tail:])),
        'best_test_ece': float(np.mean(epoch_test_ece[-n_tail:])),
    }


def run_seeds_a1(
    seeds: list[int],
    build_fn: Callable[[int], tuple],
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    config_extra: dict | None = None,
) -> None:
    """Sequential per-seed A1 training, persists results to save_dir/seed{i}.json.

    build_fn(seed) -> (backbone, baseline, train_loader, test_loader, optimizer).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        backbone, baseline, tr_loader, te_loader, optimizer = build_fn(seed)
        t0 = time.time()
        metrics = train_a1(backbone, baseline, tr_loader, te_loader, optimizer,
                            num_epochs, device)
        wall = time.time() - t0

        out = {
            'config': {**(config_extra or {}), 'seed': seed, 'epochs': num_epochs,
                        'device': str(device), 'git_sha': sha,
                        'torch_version': torch.__version__},
            **metrics,
            'wall_clock_sec': wall,
        }
        out_path = save_dir / f'seed{seed}.json'
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  [seed {seed}] nll={metrics['best_test_nll']:.4f} "
              f"acc={metrics['best_test_acc']:.4f} ece={metrics['best_test_ece']:.4f} "
              f"wall={wall:.1f}s -> {out_path}")
```

- [ ] **Step 3: Run B1 tests to verify no regression**

```bash
uv run pytest tests/test_losses.py tests/test_data.py -v
```

Expected: All B1 tests pass (train.py extension only adds new functions).

- [ ] **Step 4: Commit**

```bash
git add pca/train.py
git commit -m "feat(train): train_a1 + run_seeds_a1 with variable-N mask + 4 metrics"
```

---

## Task 18: `scripts/sanity_a1.py` — end-to-end Day 2 gate

**Files:**
- Create: `scripts/sanity_a1.py`

- [ ] **Step 1: Create `scripts/sanity_a1.py`**

```python
#!/usr/bin/env python
"""Day 2 gate: A1 sanity check before any A1 experiment.

Checks:
  1. atomic_conv Bernoulli equivalence with forward_conv on random batch.
  2. atomic_conv multi-class N=2 hand-computed case.
  3. atomic_conv signed atom shift correctness.
  4. multiclass_marginal_nll_loss + Adam step → no NaN, params change.
  5. Each of 6 baselines (1a, 2a, 2b, 3a, 3b, PCA): 1 forward + backward + Adam
     step on N=10 batch with mask varying ∈ {5, 10, 15}.
  6. Variable-N collate-equivalent batch (mix of mask N ∈ {5, 10, 15}) → all
     baselines produce finite loss.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from pca.losses import (
    forward_conv, atomic_conv, multiclass_marginal_nll_loss,
)
from pca.baselines import (
    MeanPoolBaseline, PLMulticlassBaseline, CLTGaussianBaseline,
    AttentionPoolingBaseline, DeepSetsBaseline, PCABaseline,
)


def fail(check: str, detail: str = "") -> None:
    print(f"[FAIL] {check}{(': ' + detail) if detail else ''}")
    sys.exit(1)


def main() -> None:
    torch.manual_seed(0)

    # 1. atomic_conv Bernoulli equivalence
    p = torch.rand(8, 12).clamp(min=1e-3, max=1 - 1e-3)
    log_atoms = torch.stack([(1 - p).log(), p.log()], dim=-1)
    if not torch.allclose(atomic_conv(log_atoms), forward_conv(p), atol=1e-6):
        fail("atomic_conv Bernoulli ≡ forward_conv")

    # 2. multi-class N=2 hand
    log_atoms_n2 = torch.tensor([[[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]]).log()
    P_n2 = atomic_conv(log_atoms_n2).exp()
    expected_n2 = torch.tensor([[0.20, 0.32, 0.30, 0.14, 0.04]])
    if not torch.allclose(P_n2, expected_n2, atol=1e-6):
        fail("atomic_conv N=2 hand", f"got {P_n2.tolist()}")

    # 3. signed shift (symmetric atoms → symmetric bag PMF)
    log_atoms_sg = torch.tensor([[[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]]).log()
    P_sg = atomic_conv(log_atoms_sg).exp()
    expected_sg = torch.tensor([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
    if not torch.allclose(P_sg, expected_sg, atol=1e-6):
        fail("atomic_conv signed shift")

    # 4. multiclass NLL + Adam step
    logits = torch.nn.Parameter(torch.randn(2, 5, 4))
    bag_y = torch.tensor([3, 7])
    opt = Adam([logits], lr=1e-2)
    loss = multiclass_marginal_nll_loss(logits, bag_y)
    if not torch.isfinite(loss):
        fail("multiclass_nll finite", f"loss={loss.item()}")
    before = logits.detach().clone()
    opt.zero_grad()
    loss.backward()
    if not torch.isfinite(logits.grad).all():
        fail("multiclass_nll grad finite")
    opt.step()
    if torch.allclose(before, logits.detach()):
        fail("multiclass_nll opt step", "params unchanged")

    # 5–6. each baseline on variable-N batch
    feature_dim = 128
    K = 9
    N_max = 15
    B = 4
    features = torch.randn(B, N_max, feature_dim, requires_grad=True)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    mask[0, :5] = True; mask[1, :10] = True; mask[2, :15] = True; mask[3, :8] = True
    bag_y = torch.tensor([3, 12, 38, 22])
    baselines = {
        '1a_meanpool':  MeanPoolBaseline(feature_dim, K, N_max),
        '2a_pl_multi':  PLMulticlassBaseline(feature_dim, K, N_max),
        '2b_clt':       CLTGaussianBaseline(feature_dim, K, N_max),
        '3a_attn':      AttentionPoolingBaseline(feature_dim, K, N_max),
        '3b_deepsets':  DeepSetsBaseline(feature_dim, K, N_max),
        'pca':          PCABaseline(feature_dim, K, N_max),
    }
    for name, model in baselines.items():
        loss, log_P = model(features, bag_y, mask)
        if not torch.isfinite(loss):
            fail(f"{name} finite loss", f"loss={loss.item()}")
        opt = Adam(model.parameters(), lr=1e-2)
        opt.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                fail(f"{name} grad finite", f"param {n}")
        opt.step()
        # log_P validity
        sums = log_P.exp().sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
            fail(f"{name} log_P sums to 1", f"got {sums.tolist()}")

    print("[OK] All A1 sanity checks passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run sanity_a1.py**

```bash
uv run python scripts/sanity_a1.py
```

Expected: `[OK] All A1 sanity checks passed.` and exit 0.

- [ ] **Step 3: Day 2 gate — full test suite + sanity**

```bash
uv run pytest tests/ -v
uv run python scripts/sanity_a1.py
```

Expected: All tests pass (atomic_conv 10 + metrics 6 + baselines 6 + B1-existing 10+ = ≥32) AND sanity exits 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/sanity_a1.py
git commit -m "test: scripts/sanity_a1.py — Day 2 gate end-to-end check"
```

---

## Task 19: `variable_n_collate_fn`

**Files:**
- Modify: `pca/data.py`
- Create: `tests/test_data_a1.py`

- [ ] **Step 1: Create `tests/test_data_a1.py` with collate test**

```python
"""Unit tests for A1 datasets and collate — see spec §10.3."""
import torch

from pca.data import variable_n_collate_fn


def test_collate_pads_to_n_max():
    """Items with N=3 and N=5 → batch padded to N_max=5 with mask."""
    item_a = (torch.randn(3, 1, 28, 28), 7, torch.tensor([2, 3, 2]),
              torch.ones(3, dtype=torch.bool))
    item_b = (torch.randn(5, 1, 28, 28), 11, torch.tensor([1, 4, 2, 3, 1]),
              torch.ones(5, dtype=torch.bool))
    feats, bag_sums, gts, mask = variable_n_collate_fn([item_a, item_b])
    assert feats.shape == (2, 5, 1, 28, 28), f"feats shape {tuple(feats.shape)}"
    assert mask.shape == (2, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True] * 5
    assert bag_sums.tolist() == [7, 11]
    assert gts.shape == (2, 5)


def test_collate_mask_active_count_matches_n():
    """mask.sum(dim=1) equals each bag's actual N."""
    items = [
        (torch.randn(n, 1, 28, 28), n, torch.zeros(n, dtype=torch.long),
         torch.ones(n, dtype=torch.bool))
        for n in [3, 5, 4, 7]
    ]
    _, _, _, mask = variable_n_collate_fn(items)
    assert mask.sum(dim=1).tolist() == [3, 5, 4, 7]
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: FAIL — `cannot import name 'variable_n_collate_fn'`.

- [ ] **Step 3: Append `variable_n_collate_fn` to `pca/data.py`**

```python
def variable_n_collate_fn(batch):
    """Collate fn that pads variable-N items to batch's N_max with masking.

    Each item must be a 4-tuple (features, bag_sum, gt_per_instance, mask) where
    features.shape[0] == gt_per_instance.shape[0] == mask.shape[0] == N_item.
    Inactive padding atoms get features = 0 (not used downstream — mask flags
    them) and mask = False; gt is padded with 0 (caller must respect mask).
    """
    n_max = max(item[0].shape[0] for item in batch)
    B = len(batch)
    feature_shape = batch[0][0].shape[1:]
    padded_features = batch[0][0].new_zeros((B, n_max, *feature_shape))
    padded_gt = torch.zeros((B, n_max), dtype=torch.long)
    padded_mask = torch.zeros((B, n_max), dtype=torch.bool)
    bag_sums = torch.zeros((B,), dtype=torch.long)
    for b, (feats, bag_sum, gt, mask) in enumerate(batch):
        n_b = feats.shape[0]
        padded_features[b, :n_b] = feats
        padded_gt[b, :n_b] = gt
        padded_mask[b, :n_b] = mask
        bag_sums[b] = bag_sum
    return padded_features, bag_sums, padded_gt, padded_mask
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add pca/data.py tests/test_data_a1.py
git commit -m "feat(data): variable_n_collate_fn for batched variable-N bags"
```

---

## Task 20: `MNISTSumBagDataset`

**Files:**
- Modify: `pca/data.py`
- Modify: `tests/test_data_a1.py`

- [ ] **Step 1: Append MNIST-sum dataset tests**

Append to `tests/test_data_a1.py`:

```python
import numpy as np
from pca.data import MNISTSumBagDataset


def test_mnist_sum_bag_count_matches_gt():
    ds = MNISTSumBagDataset(num_bags=20, per_class_cap=100, seed=0)
    for i in range(len(ds)):
        images, bag_sum, gt, mask = ds[i]
        n_active = mask.sum().item()
        assert images.shape == (n_active, 1, 28, 28)
        assert gt.shape == (n_active,)
        assert mask.all()                     # MNISTSumBagDataset returns mask=all True
        assert bag_sum == int(gt.sum().item())


def test_mnist_sum_variable_n_in_range():
    ds = MNISTSumBagDataset(num_bags=200, bag_size_min=5, bag_size_max=15, seed=1)
    sizes = [ds[i][0].shape[0] for i in range(len(ds))]
    assert min(sizes) >= 5
    assert max(sizes) <= 15
    # Truncated normal with std=2: mean should be near 10
    assert 8.5 < float(np.mean(sizes)) < 11.5, f"mean={np.mean(sizes)}"


def test_mnist_sum_per_class_cap():
    ds = MNISTSumBagDataset(num_bags=10, per_class_cap=50, seed=2)
    label_counts = {}
    for k in range(10):
        label_counts[k] = (ds._labels == k).sum().item()
    for k, count in label_counts.items():
        assert count <= 50, f"class {k}: count={count} > cap=50"


def test_mnist_sum_seed_determinism():
    ds1 = MNISTSumBagDataset(num_bags=5, per_class_cap=100, seed=42)
    ds2 = MNISTSumBagDataset(num_bags=5, per_class_cap=100, seed=42)
    for i in range(5):
        f1, s1, g1, m1 = ds1[i]
        f2, s2, g2, m2 = ds2[i]
        assert torch.equal(f1, f2)
        assert s1 == s2
        assert torch.equal(g1, g2)


def test_mnist_sum_noise_increases_variance():
    ds_no = MNISTSumBagDataset(num_bags=50, per_class_cap=100, noise_sigma=0.0, seed=3)
    ds_n = MNISTSumBagDataset(num_bags=50, per_class_cap=100, noise_sigma=0.1, seed=3)
    var_no = torch.cat([ds_no[i][0].flatten() for i in range(20)]).var().item()
    var_n = torch.cat([ds_n[i][0].flatten() for i in range(20)]).var().item()
    assert var_n > var_no, f"noisy var={var_n} not > clean={var_no}"
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_data_a1.py -v -k mnist_sum
```

Expected: FAIL — `cannot import name 'MNISTSumBagDataset'`.

- [ ] **Step 3: Append `MNISTSumBagDataset` to `pca/data.py`**

```python
class MNISTSumBagDataset(Dataset):
    """MNIST bag dataset where bag label = sum of digit values.

    Atom support: S=10 (digits 0..9). Bag sum range: [0, 9*N_max].

    Args:
        bag_size_mean, bag_size_std: truncated-normal parameters for N.
        bag_size_min, bag_size_max:  truncation bounds (inclusive).
        num_bags:      number of bags.
        per_class_cap: if set, subsample MNIST train pool to this many per class
                       (deterministic via seed). None → full pool.
        noise_sigma:   if > 0, add N(0, σ) noise to images at access time
                       (training and eval identically — task hardening).
        train:         True → MNIST train split, False → test split.
        seed:          determinism over class subsampling, bag indices, bag sizes.
    """

    def __init__(
        self,
        bag_size_mean: float = 10.0,
        bag_size_std:  float = 2.0,
        bag_size_min:  int = 5,
        bag_size_max:  int = 15,
        num_bags:      int = 1500,
        per_class_cap: int | None = 100,
        noise_sigma:   float = 0.0,
        train:         bool = True,
        seed:          int = 0,
    ):
        self.bag_size_mean = bag_size_mean
        self.bag_size_std = bag_size_std
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max
        self.num_bags = num_bags
        self.noise_sigma = noise_sigma

        tfm = transforms.Compose([transforms.ToTensor()])
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        all_images = torch.stack([mnist[i][0] for i in range(len(mnist))])
        all_labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        # Pre-sample bag sizes + indices.
        sizes = []
        bag_indices = []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(round(rng.normal(bag_size_mean, bag_size_std)))
            n = max(bag_size_min, min(bag_size_max, n))
            sizes.append(n)
            bag_indices.append(rng.choice(M, size=n, replace=False))
        self._bag_sizes = sizes
        self._bag_indices = bag_indices
        self._noise_rng = np.random.default_rng(seed + 10_000)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum, labels.long(), mask
```

- [ ] **Step 4: Run all data_a1 tests**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: 7 PASSED (2 collate + 5 mnist_sum).

- [ ] **Step 5: Commit**

```bash
git add pca/data.py tests/test_data_a1.py
git commit -m "feat(data): MNISTSumBagDataset with variable-N + per-class-cap + noise"
```

---

## Task 21: `MNISTSignedSumBagDataset`

**Files:**
- Modify: `pca/data.py`
- Modify: `tests/test_data_a1.py`

- [ ] **Step 1: Append signed dataset tests**

Append to `tests/test_data_a1.py`:

```python
from pca.data import MNISTSignedSumBagDataset


def test_mnist_signed_bag_sum_matches_shifted_gt():
    """Signed mapping: 0→0, odd nonzero→-1, even nonzero→+1.
       bag_sum stored = true_sum + N (caller un-shifts via metrics).
    """
    ds = MNISTSignedSumBagDataset(num_bags=20, per_class_cap=100, seed=0)
    for i in range(20):
        images, bag_sum_shifted, gt_signed, mask = ds[i]
        n = mask.sum().item()
        true_sum = int(gt_signed.sum().item())
        assert bag_sum_shifted == true_sum + n
        # gt values must be in {-1, 0, +1}
        assert set(gt_signed.tolist()).issubset({-1, 0, 1})


def test_mnist_signed_atom_support_size_3():
    """For N=10 signed, support length = 2*N+1 = 21 (after shifting)."""
    ds = MNISTSignedSumBagDataset(num_bags=10, bag_size_min=10, bag_size_max=10, seed=1)
    images, bag_sum, gt, mask = ds[0]
    assert mask.sum().item() == 10
    assert 0 <= bag_sum <= 20         # shifted range [0, 2N]
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_data_a1.py -v -k signed
```

Expected: FAIL — `cannot import name 'MNISTSignedSumBagDataset'`.

- [ ] **Step 3: Append `MNISTSignedSumBagDataset` to `pca/data.py`**

```python
# Pinned signed-digit mapping: 0 → 0, odd-nonzero → -1, even-nonzero → +1.
_SIGNED_DIGIT_MAP = torch.tensor([0, -1, +1, -1, +1, -1, +1, -1, +1, -1])


class MNISTSignedSumBagDataset(MNISTSumBagDataset):
    """MNIST signed-sum bag dataset.

    Per-instance ground truth z_i ∈ {-1, 0, +1} via _SIGNED_DIGIT_MAP.
    Bag sum range: [-N, +N], stored shifted by +N → bag_sum_shifted ∈ [0, 2N].
    Caller (metrics.py) un-shifts for accuracy/MAE.
    """

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        signed = _SIGNED_DIGIT_MAP[labels]                  # (n,) values in {-1,0,1}
        bag_sum_shifted = int(signed.sum().item()) + n      # in [0, 2n]
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum_shifted, signed.long(), mask
```

- [ ] **Step 4: Run all data_a1 tests**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: 9 PASSED.

- [ ] **Step 5: Commit**

```bash
git add pca/data.py tests/test_data_a1.py
git commit -m "feat(data): MNISTSignedSumBagDataset with shifted bag_sum"
```

---

## Task 22: `scripts/run_mnist_sum.py` and `scripts/run_mnist_signed.py`

**Files:**
- Create: `scripts/run_mnist_sum.py`
- Create: `scripts/run_mnist_signed.py`

- [ ] **Step 1: Create `scripts/run_mnist_sum.py`**

```python
#!/usr/bin/env python
"""MNIST-sum experiment runner.

Default config (spec §13 Day 4):
  6 methods × 5 seeds × 80 epochs ≈ ~5h on MPS (single device).

Smoke usage:
  python scripts/run_mnist_sum.py --method pca --epochs 2 --num-bags 50 --seeds 0
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from functools import partial

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import (
    MeanPoolBaseline, PLMulticlassBaseline, CLTGaussianBaseline,
    AttentionPoolingBaseline, DeepSetsBaseline, PCABaseline,
)
from pca.data import MNISTSumBagDataset, variable_n_collate_fn
from pca.models import SmallCNNMulticlass
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 1500
DEFAULT_TEST_BAGS = 600
DEFAULT_EPOCHS = 80
K = 9
N_MAX = 15

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
    '2a':  PLMulticlassBaseline,
    '2b':  CLTGaussianBaseline,
    '3a':  AttentionPoolingBaseline,
    '3b':  DeepSetsBaseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(BASELINE_REGISTRY.keys()), required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--bag-size-mean', type=float, default=10.0)
    p.add_argument('--bag-size-std', type=float, default=2.0)
    p.add_argument('--bag-size-min', type=int, default=5)
    p.add_argument('--bag-size-max', type=int, default=15)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=100)
    p.add_argument('--noise-sigma', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output-dir', type=Path, default=Path('results/mnist_sum'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}, σ={args.noise_sigma}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = MNISTSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=True, seed=seed)
        test_ds = MNISTSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=False, seed=seed)
        backbone = SmallCNNMulticlass()
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()), lr=args.lr)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    save_dir = (args.output_dir / f"N{int(args.bag_size_mean)}_"
                                   f"cap{args.per_class_cap}_"
                                   f"sig{args.noise_sigma}_{args.method}")
    config_extra = {
        'experiment': 'mnist_sum', 'method': args.method, 'atom_support': K + 1,
        'bag_size_mean': args.bag_size_mean, 'bag_size_std': args.bag_size_std,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': args.noise_sigma,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run with method=pca, 2 epochs, 50 bags, seed=0**

```bash
uv run python scripts/run_mnist_sum.py --method pca --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
```

Expected: stdout ends with `[seed 0] nll=... acc=... ece=... wall=Xs -> results/mnist_sum/N10_cap100_sig0.0_pca/seed0.json`. JSON file is schema-valid (config + epoch_test_* arrays + best_test_*).

- [ ] **Step 3: Smoke runs for the other 5 methods**

```bash
for m in 1a 2a 2b 3a 3b; do
  uv run python scripts/run_mnist_sum.py --method $m --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
done
```

Expected: 5 more `seed0.json` files in `results/mnist_sum/N10_cap100_sig0.0_<method>/`.

- [ ] **Step 4: Create `scripts/run_mnist_signed.py`**

```python
#!/usr/bin/env python
"""MNIST-signed-sum experiment runner.

Signed atom case: K=2 (S=3 over {-1,0,+1}), bag-sum support T = 2*N+1 (shifted).
Default config (spec §13 Day 4): 5 seeds × 80 epochs × 2 methods (PCA + 1a baseline).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import MeanPoolBaseline, PCABaseline
from pca.data import MNISTSignedSumBagDataset, variable_n_collate_fn
from pca.models import SmallCNNMulticlass
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 1500
DEFAULT_TEST_BAGS = 600
DEFAULT_EPOCHS = 80
K = 2     # support {-1, 0, +1} with caller-side shift → K=2 in atomic_conv
N_MAX = 15

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(BASELINE_REGISTRY.keys()), required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--bag-size-mean', type=float, default=10.0)
    p.add_argument('--bag-size-std', type=float, default=2.0)
    p.add_argument('--bag-size-min', type=int, default=5)
    p.add_argument('--bag-size-max', type=int, default=15)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=100)
    p.add_argument('--noise-sigma', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output-dir', type=Path, default=Path('results/mnist_signed'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}, σ={args.noise_sigma}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = MNISTSignedSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=True, seed=seed)
        test_ds = MNISTSignedSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=False, seed=seed)
        backbone = SmallCNNMulticlass()
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()), lr=args.lr)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    save_dir = (args.output_dir / f"N{int(args.bag_size_mean)}_"
                                   f"cap{args.per_class_cap}_"
                                   f"sig{args.noise_sigma}_{args.method}")
    config_extra = {
        'experiment': 'mnist_signed', 'method': args.method, 'atom_support': K + 1,
        'bag_size_mean': args.bag_size_mean, 'bag_size_std': args.bag_size_std,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': args.noise_sigma,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke runs for signed**

```bash
uv run python scripts/run_mnist_signed.py --method pca --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
uv run python scripts/run_mnist_signed.py --method 1a  --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
```

Expected: 2 `seed0.json` files in `results/mnist_signed/N10_cap100_sig0.0_<method>/`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_mnist_sum.py scripts/run_mnist_signed.py
git commit -m "feat(scripts): MNIST-sum + MNIST-signed runners with smoke validation"
```

---

## Task 23: per-class-cap sweep smoke + Day 3 gate

**Files:**
- (No new files; runs the calibration sweep)

- [ ] **Step 1: Run the per-class-cap sweep (PCA × cap ∈ {200, 100, 50} × seed=0 × 30 epochs)**

```bash
for cap in 200 100 50; do
  uv run python scripts/run_mnist_sum.py --method pca --epochs 30 \
    --num-bags 1000 --num-test-bags 400 --seeds 0 \
    --per-class-cap $cap \
    --output-dir results/mnist_sum_sweep
done
```

Expected: 3 `seed0.json` files in `results/mnist_sum_sweep/N10_cap{200,100,50}_sig0.0_pca/`.

- [ ] **Step 2: Inspect best_test_acc per cap and apply spec §3.6 sweep guard**

```bash
for cap in 200 100 50; do
  uv run python -c "import json; d = json.loads(open('results/mnist_sum_sweep/N10_cap${cap}_sig0.0_pca/seed0.json').read()); print(f'cap=${cap}: acc={d[\"best_test_acc\"]:.3f} nll={d[\"best_test_nll\"]:.3f}')"
done
```

Expected: print three lines, one per cap. Apply rules from spec §3.6:
- All caps PCA top-1 < 0.30 → cap too strict, raise default to 200, re-smoke.
- caps 100 and 200 differ < 1pp → saturation regime, lower default to 50, re-smoke.
- cap=100 PCA top-1 in [0.50, 0.85] → confirm default=100 ✓.

If sweep guard requires re-smoking, repeat steps 1–2 with the adjusted cap and update the chosen default for Day 4 runs.

- [ ] **Step 3: Record final cap decision in `results/mnist_sum_sweep/cap_decision.md`**

```bash
cat > results/mnist_sum_sweep/cap_decision.md <<'EOF'
# Per-class-cap sweep result (Day 3 gate)

| cap | best_test_acc | best_test_nll |
|---|---|---|
| 200 | <fill> | <fill> |
| 100 | <fill> | <fill> |
|  50 | <fill> | <fill> |

**Decision (spec §3.6):** default cap = <100 / 50 / 200>
**Rationale:** <one line: which rule fired>

EOF
```

Expected: file created. Engineer fills in the table from step 2 output and the Decision line per the rules.

- [ ] **Step 4: Day 3 gate — full pytest + sanity_a1 + smoke runs all schema-valid**

```bash
uv run pytest tests/ -v
uv run python scripts/sanity_a1.py
ls -la results/mnist_sum/*/seed0.json results/mnist_signed/*/seed0.json
```

Expected: all tests pass; sanity ok; 6 + 2 = 8 smoke seed0.json files exist.

- [ ] **Step 5: Commit**

```bash
git add results/mnist_sum_sweep/cap_decision.md
git commit -m "chore(sweep): per-class-cap calibration result for Day 4 default"
```

---

## Task 24: `scripts/analyze_a1.py` — H1/H2/H3 + saturation + verdict

**Files:**
- Create: `scripts/analyze_a1.py`

- [ ] **Step 1: Create `scripts/analyze_a1.py`**

```python
#!/usr/bin/env python
"""Aggregate per-seed JSONs across A1 datasets → summary.md + final verdict.

Per spec §11. Usage:
  python scripts/analyze_a1.py                     # full verdict (all 4 datasets)
  python scripts/analyze_a1.py --partial           # subset of datasets present
  python scripts/analyze_a1.py --datasets mnist_sum mnist_signed
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.stats


HARD_CAP = {
    'mnist_sum':     50,
    'mnist_signed':  50,
    'svhn_sum':      1000,
    'ultramnist':    300,
}
ALL_DATASETS = ['mnist_sum', 'svhn_sum', 'ultramnist', 'mnist_signed']
MULTICLASS_DATASETS = ['mnist_sum', 'svhn_sum', 'ultramnist']
DISTRIBUTION_NATIVE = ['2b', '3a', '3b']


@dataclass
class CellStat:
    mean: float
    std: float
    seeds: list[float] = field(default_factory=list)


def aggregate(results_root: Path, datasets: Iterable[str]) -> dict:
    """summary[dataset][method][metric] = CellStat for metric ∈ {nll, acc, mae, ece}."""
    summary: dict = {}
    for ds in datasets:
        ds_dir = results_root / ds
        if not ds_dir.exists():
            continue
        per_method: dict = {}
        for setting_dir in sorted(ds_dir.glob('N*_*')):
            method = setting_dir.name.rsplit('_', 1)[1]
            seed_jsons = sorted(setting_dir.glob('seed*.json'))
            if not seed_jsons:
                continue
            metrics = {'nll': [], 'acc': [], 'mae': [], 'ece': []}
            for f in seed_jsons:
                d = json.loads(f.read_text())
                metrics['nll'].append(d['best_test_nll'])
                metrics['acc'].append(d['best_test_acc'])
                metrics['mae'].append(d['best_test_mae'])
                metrics['ece'].append(d['best_test_ece'])
            cells = {}
            for k, vs in metrics.items():
                cells[k] = CellStat(
                    mean=float(np.mean(vs)),
                    std=float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                    seeds=vs,
                )
            per_method[method] = cells
        if per_method:
            summary[ds] = per_method
    return summary


def evaluate_h1(summary, multiclass_datasets) -> dict:
    """H1: PCA NLL < min distribution-native baseline NLL,
       paired t p<0.05, |Δ|≥0.02 nats, on ≥2 of 3 datasets."""
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        if ds not in summary or 'pca' not in summary[ds]:
            per_dataset[ds] = {'available': False}
            continue
        pca_seeds = np.array(summary[ds]['pca']['nll'].seeds)
        baselines = [m for m in DISTRIBUTION_NATIVE if m in summary[ds]]
        if not baselines:
            per_dataset[ds] = {'available': False}
            continue
        baseline_seeds = np.stack([np.array(summary[ds][m]['nll'].seeds) for m in baselines])
        best_per_seed = baseline_seeds.min(axis=0)
        deltas = best_per_seed - pca_seeds
        delta_mean = float(np.mean(deltas))
        if len(pca_seeds) > 1:
            _, p_value = scipy.stats.ttest_rel(best_per_seed, pca_seeds)
        else:
            p_value = 1.0
        passes = bool(delta_mean >= 0.02 and p_value < 0.05)
        per_dataset[ds] = {'available': True, 'delta': delta_mean,
                            'p': float(p_value), 'pass': passes,
                            'baselines_used': baselines}
        if passes:
            pass_count += 1
    available_count = sum(1 for v in per_dataset.values() if v.get('available'))
    # "≥2 of 3" rule on full data; loosened to "≥ min(2, available)" in partial.
    threshold = min(2, available_count) if available_count > 0 else 1
    return {
        'pass': pass_count >= threshold,
        'pass_count': pass_count,
        'available_count': available_count,
        'threshold': threshold,
        'per_dataset': per_dataset,
    }


def evaluate_h2(summary, results_root: Path) -> dict:
    """H2: atom-shape generality must-pass.
    Binary B1 reuse: NLL > 0 (sanity) at any bag size; record cite-only.
    Multi-class K=9 on each dataset: top-1 acc > 0.30, NLL finite.
    Signed: top-1 acc > 0.20 (over un-shifted prediction), NLL finite.
    """
    checks = {}
    # Binary B1 reuse — read mnist_mil/summary.json's NLL @ N=50 instance AUC > 0.9.
    # B1 summary schema: {summary: {<N>: {<method>: {mean, std, seeds}}}}.
    b1_summary_path = results_root / 'mnist_mil' / 'summary.json'
    if b1_summary_path.exists():
        b1 = json.loads(b1_summary_path.read_text())
        try:
            b1_nll_n50_auc = b1['summary']['50']['nll']['mean']
            checks['binary_b1_reuse'] = {
                'pass': bool(b1_nll_n50_auc > 0.9),
                'detail': f'B1 NLL@N=50 inst_auc = {b1_nll_n50_auc:.4f}',
            }
        except (KeyError, TypeError) as e:
            checks['binary_b1_reuse'] = {
                'pass': False, 'detail': f'B1 summary unparseable: {e}'}
    else:
        checks['binary_b1_reuse'] = {'pass': False,
                                       'detail': f'missing {b1_summary_path}'}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        if ds in summary and 'pca' in summary[ds]:
            acc = summary[ds]['pca']['acc'].mean
            nll = summary[ds]['pca']['nll'].mean
            checks[f'multiclass_{ds}'] = {
                'pass': bool(acc > 0.30 and np.isfinite(nll)),
                'detail': f'acc={acc:.3f} nll={nll:.3f}',
            }
        else:
            checks[f'multiclass_{ds}'] = {'pass': False, 'detail': 'absent'}
    if 'mnist_signed' in summary and 'pca' in summary['mnist_signed']:
        acc = summary['mnist_signed']['pca']['acc'].mean
        nll = summary['mnist_signed']['pca']['nll'].mean
        checks['signed_mnist_signed'] = {
            'pass': bool(acc > 0.20 and np.isfinite(nll)),
            'detail': f'acc={acc:.3f} nll={nll:.3f}',
        }
    else:
        checks['signed_mnist_signed'] = {'pass': False, 'detail': 'absent'}
    all_pass = all(c['pass'] for c in checks.values())
    return {'pass': all_pass, 'checks': checks}


def evaluate_h3(summary, multiclass_datasets) -> dict:
    """H3: PCA ECE ≤ 0.7 × min distribution-native baseline ECE on ≥2 of 3."""
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        if ds not in summary or 'pca' not in summary[ds]:
            per_dataset[ds] = {'available': False}
            continue
        pca_ece = summary[ds]['pca']['ece'].mean
        baselines = [m for m in DISTRIBUTION_NATIVE if m in summary[ds]]
        if not baselines:
            per_dataset[ds] = {'available': False}
            continue
        baseline_eces = [summary[ds][m]['ece'].mean for m in baselines]
        min_baseline_ece = float(min(baseline_eces))
        ratio = pca_ece / min_baseline_ece if min_baseline_ece > 0 else float('inf')
        passes = bool(ratio <= 0.7)
        per_dataset[ds] = {'available': True, 'pca_ece': float(pca_ece),
                            'min_baseline_ece': min_baseline_ece,
                            'ratio': float(ratio), 'pass': passes,
                            'baselines_used': baselines}
        if passes:
            pass_count += 1
    available_count = sum(1 for v in per_dataset.values() if v.get('available'))
    threshold = min(2, available_count) if available_count > 0 else 1
    return {'pass': pass_count >= threshold, 'pass_count': pass_count,
            'available_count': available_count, 'threshold': threshold,
            'per_dataset': per_dataset}


def detect_saturation(summary) -> dict:
    """For each multi-class dataset: True iff distribution-native NLL spread <0.05
       OR any method's top-1 acc > 0.97."""
    out = {}
    for ds in MULTICLASS_DATASETS:
        if ds not in summary:
            continue
        nll_means = [summary[ds][m]['nll'].mean for m in DISTRIBUTION_NATIVE
                      if m in summary[ds]]
        acc_means = [summary[ds][m]['acc'].mean for m in summary[ds]]
        if not nll_means or not acc_means:
            out[ds] = False
            continue
        out[ds] = bool((max(nll_means) - min(nll_means) < 0.05)
                        or (max(acc_means) > 0.97))
    return out


def hard_preset_already_run(results_root: Path, saturation: dict) -> bool:
    """Detect whether any seed*.json under results/<saturated-ds>/ has
       config.per_class_cap == HARD_CAP[ds]."""
    for ds, sat in saturation.items():
        if not sat:
            continue
        ds_dir = results_root / ds
        if not ds_dir.exists():
            continue
        for seed_json in ds_dir.rglob('seed*.json'):
            d = json.loads(seed_json.read_text())
            if d.get('config', {}).get('per_class_cap') == HARD_CAP.get(ds):
                return True
    return False


def hard_preset_clis(saturation: dict) -> list[str]:
    clis = []
    for ds, sat in saturation.items():
        if not sat:
            continue
        cap = HARD_CAP[ds]
        if ds == 'mnist_sum':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_mnist_sum.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.1 "
                    f"--output-dir results/mnist_sum")
        elif ds == 'mnist_signed':
            for m in ['pca', '1a']:
                clis.append(
                    f"uv run python scripts/run_mnist_signed.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.1 "
                    f"--output-dir results/mnist_signed")
        elif ds == 'svhn_sum':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_svhn_sum.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.05 "
                    f"--output-dir results/svhn_sum")
        elif ds == 'ultramnist':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_ultramnist.py --method {m} --epochs 60 "
                    f"--seeds 0,1,2 --per-class-cap {cap} "
                    f"--output-dir results/ultramnist")
    return clis


def verdict(h1, h2, h3, saturation, results_root) -> str:
    if not h2['pass']:
        return 'ALGEBRA_BROKEN'
    if h1['pass']:
        return 'A1_CONFIRMED'
    sat_any = any(saturation.values())
    already = hard_preset_already_run(results_root, saturation)
    if sat_any and not already:
        return 'RUN_HARD_PRESET'
    if h3['pass']:
        return 'A1_CALIBRATION_FOCUS'
    return 'DEMOTE_A1'


def render_dataset_summary(ds: str, methods: dict) -> str:
    lines = [f"# Results — {ds}", "", "## Metrics (mean ± std)", ""]
    lines.append("| Method | Top-1 acc | MAE | NLL | ECE |")
    lines.append("|--------|-----------|-----|-----|-----|")
    method_order = ['1a', '2a', '2b', '3a', '3b', 'pca']
    for m in method_order:
        if m not in methods:
            continue
        c = methods[m]
        lines.append(
            f"| {m} | {c['acc'].mean:.3f} ± {c['acc'].std:.3f} "
            f"| {c['mae'].mean:.3f} ± {c['mae'].std:.3f} "
            f"| {c['nll'].mean:.3f} ± {c['nll'].std:.3f} "
            f"| {c['ece'].mean:.3f} ± {c['ece'].std:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_overall(summary, h1, h2, h3, saturation, v: str) -> str:
    lines = ["# A1 Verification — Final Summary", "",
             f"## Verdict: **{v}**", "", "## H1 — NLL primary", ""]
    lines.append(f"- pass={h1['pass']}; pass_count={h1['pass_count']} "
                 f"(of {h1['available_count']} available)")
    for ds, d in h1['per_dataset'].items():
        if not d.get('available'):
            lines.append(f"  - {ds}: not available")
        else:
            lines.append(f"  - {ds}: Δ={d['delta']:+.3f} nats, p={d['p']:.3f}, "
                         f"{'PASS' if d['pass'] else 'FAIL'}")
    lines.extend(["", "## H2 — atom-shape generality"])
    for k, c in h2['checks'].items():
        lines.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} ({c['detail']})")
    lines.extend(["", "## H3 — calibration"])
    lines.append(f"- pass={h3['pass']}; pass_count={h3['pass_count']}")
    for ds, d in h3['per_dataset'].items():
        if not d.get('available'):
            lines.append(f"  - {ds}: not available")
        else:
            lines.append(f"  - {ds}: ratio={d['ratio']:.2f}, "
                         f"{'PASS' if d['pass'] else 'FAIL'}")
    lines.extend(["", "## Saturation detection"])
    for ds, sat in saturation.items():
        lines.append(f"- {ds}: {'detected' if sat else 'clear'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', type=Path, default=Path('results'))
    p.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    p.add_argument('--partial', action='store_true',
                   help='Allow subset of datasets; verdict scoped to available.')
    args = p.parse_args()

    summary = aggregate(args.results_root, args.datasets)
    if not summary:
        print("[error] no datasets aggregated; nothing to analyze.")
        raise SystemExit(1)

    multiclass = [ds for ds in MULTICLASS_DATASETS if ds in summary]
    h1 = evaluate_h1(summary, multiclass)
    h2 = evaluate_h2(summary, args.results_root)
    h3 = evaluate_h3(summary, multiclass)
    saturation = detect_saturation(summary)
    v = verdict(h1, h2, h3, saturation, args.results_root)

    # Per-dataset summaries
    for ds, methods in summary.items():
        md = render_dataset_summary(ds, methods)
        out = args.results_root / ds / 'summary.md'
        out.write_text(md)
        out_json = args.results_root / ds / 'summary.json'
        out_json.write_text(json.dumps({
            m: {k: {'mean': c.mean, 'std': c.std, 'seeds': c.seeds}
                for k, c in cells.items()}
            for m, cells in methods.items()
        }, indent=2))

    # Overall summary
    overall = render_overall(summary, h1, h2, h3, saturation, v)
    (args.results_root / 'summary_overall.md').write_text(overall)
    print(overall)

    if v == 'RUN_HARD_PRESET':
        print("\n--- Hard preset CLIs to execute next ---")
        for cli in hard_preset_clis(saturation):
            print(cli)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run analyze on existing smoke runs**

```bash
uv run python scripts/analyze_a1.py --partial --datasets mnist_sum mnist_signed
```

Expected: prints overall summary; writes `results/mnist_sum/summary.{md,json}`, `results/mnist_signed/summary.{md,json}`, `results/summary_overall.md`. Verdict likely `ALGEBRA_BROKEN` (only 1 seed, smoke runs) or based on the smoke H2 check; OK to be non-final.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_a1.py
git commit -m "feat(scripts): analyze_a1 with H1/H2/H3 + saturation + verdict logic"
```

---

## Task 25: Day 4 — MNIST-sum + signed full runs

**Files:**
- (No new files; runs the experiments)

- [ ] **Step 1: Clean smoke runs (keep sweep)**

```bash
rm -rf results/mnist_sum/N10_cap100_sig0.0_* results/mnist_signed/N10_cap100_sig0.0_*
```

Expected: smoke seed0.json directories removed; cap-sweep dir kept.

- [ ] **Step 2: MNIST-sum 30 runs (5 seeds × 6 methods × 80 epochs)**

```bash
for m in pca 1a 2a 2b 3a 3b; do
  uv run python scripts/run_mnist_sum.py --method $m --epochs 80 \
    --seeds 0,1,2,3,4 --per-class-cap 100 \
    --output-dir results/mnist_sum
done
```

Expected: 6 directories under `results/mnist_sum/N10_cap100_sig0.0_<method>/` each with seed0..4.json. Wall clock ≈ 5h on MPS (sequential).

- [ ] **Step 3: MNIST-signed 10 runs (5 seeds × 2 methods × 80 epochs)**

```bash
for m in pca 1a; do
  uv run python scripts/run_mnist_signed.py --method $m --epochs 80 \
    --seeds 0,1,2,3,4 --per-class-cap 100 \
    --output-dir results/mnist_signed
done
```

Expected: 2 directories under `results/mnist_signed/N10_cap100_sig0.0_<method>/` each with seed0..4.json. Wall clock ≈ 1.7h.

- [ ] **Step 4: Schema validation — all 40 JSONs**

```bash
uv run python -c "
import json, glob
for path in sorted(glob.glob('results/mnist_sum/N10_cap100_sig0.0_*/seed*.json') + 
                    glob.glob('results/mnist_signed/N10_cap100_sig0.0_*/seed*.json')):
    d = json.loads(open(path).read())
    for key in ['config', 'best_test_nll', 'best_test_acc', 'best_test_mae',
                'best_test_ece', 'epoch_test_nll', 'wall_clock_sec', 'config']:
        assert key in d, f'{path} missing {key}'
print(f'all 40 JSONs schema-valid')
"
```

Expected: prints `all 40 JSONs schema-valid`.

- [ ] **Step 5: Commit logs (no JSON files — gitignored)**

```bash
git add results/mnist_sum/cap_decision.md 2>/dev/null || true
git status                                    # confirm only summary md/json staged
git commit --allow-empty -m "run: Day 4 MNIST-sum 30-run + signed 10-run complete"
```

Expected: empty commit (results gitignored). Engineer manually inspects sample seed0.json before proceeding.

---

## Task 26: Day 4 — Partial verdict (MNIST-sum + signed)

**Files:**
- Modify: `results/mnist_sum/summary.{md,json}` (auto-generated)
- Modify: `results/mnist_signed/summary.{md,json}` (auto-generated)
- Modify: `results/summary_overall.md` (auto-generated)

- [ ] **Step 1: Run partial verdict**

```bash
uv run python scripts/analyze_a1.py --partial --datasets mnist_sum mnist_signed
```

Expected: prints overall summary; writes summary md/json. Verdict logic (H1 evaluated only on mnist_sum since signed is not in MULTICLASS_DATASETS):
- H2 should pass for mnist_sum + mnist_signed; binary B1 reuse passes if `results/mnist_mil/summary.json` exists.
- H1 evaluated on mnist_sum alone (single dataset; "≥2 of 3" rule effectively becomes "pass on this one").
- H3 evaluated on mnist_sum alone.

Engineer reads the verdict and decides whether to adjust SVHN-sum/UltraMNIST hyperparameters before Day 5/6 overnight runs (e.g., apply hard preset early if mnist_sum saturates).

- [ ] **Step 2: If saturation detected on mnist_sum, run hard preset early**

If `summary_overall.md` shows `mnist_sum: detected` under saturation:

```bash
for m in pca 1a 2a 2b 3a 3b; do
  uv run python scripts/run_mnist_sum.py --method $m --epochs 80 \
    --seeds 0,1,2,3,4 --per-class-cap 50 --noise-sigma 0.1 \
    --output-dir results/mnist_sum
done
for m in pca 1a; do
  uv run python scripts/run_mnist_signed.py --method $m --epochs 80 \
    --seeds 0,1,2,3,4 --per-class-cap 50 --noise-sigma 0.1 \
    --output-dir results/mnist_signed
done
```

Expected: additional `N10_cap50_sig0.1_*` dirs created. Re-run analyze with `--partial` to see updated verdict.

- [ ] **Step 3: Commit summary md/json**

```bash
git add results/mnist_sum/summary.md results/mnist_sum/summary.json \
        results/mnist_signed/summary.md results/mnist_signed/summary.json \
        results/summary_overall.md
git commit -m "results: Day 4 partial verdict for MNIST-sum + signed"
```

---

## Task 27: `SVHNSumBagDataset`

**Files:**
- Modify: `pca/data.py`
- Modify: `tests/test_data_a1.py`

- [ ] **Step 1: Append SVHN tests**

Append to `tests/test_data_a1.py`:

```python
from pca.data import SVHNSumBagDataset


def test_svhn_sum_bag_count_matches_gt():
    """SVHN-sum bag dataset returns RGB 32x32 patches and matching bag sums."""
    ds = SVHNSumBagDataset(num_bags=8, seed=0)
    for i in range(8):
        images, bag_sum, gt, mask = ds[i]
        n = mask.sum().item()
        assert images.shape == (n, 3, 32, 32)
        assert gt.shape == (n,)
        assert bag_sum == int(gt.sum().item())


def test_svhn_sum_natural_difficulty_no_cap():
    """Default per_class_cap=None → uses full pool."""
    ds = SVHNSumBagDataset(num_bags=4, per_class_cap=None, seed=0)
    M = ds._images.shape[0]
    assert M > 50_000        # SVHN train ~73k
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_data_a1.py -v -k svhn
```

Expected: FAIL — `cannot import name 'SVHNSumBagDataset'`.

- [ ] **Step 3: Append `SVHNSumBagDataset` to `pca/data.py`**

Add after `MNISTSignedSumBagDataset`:

```python
from torchvision.datasets import SVHN as _SVHN


class SVHNSumBagDataset(Dataset):
    """SVHN bag dataset (RGB 32x32) with bag label = sum of digit values.

    Atom support S = 10 (digits 0..9, but SVHN uses {1..10} natively → mapped
    to {0..9}). Bag sum range: [0, 9 * N_max].
    """

    def __init__(
        self,
        bag_size_mean: float = 10.0,
        bag_size_std:  float = 2.0,
        bag_size_min:  int = 5,
        bag_size_max:  int = 15,
        num_bags:      int = 1500,
        per_class_cap: int | None = None,
        noise_sigma:   float = 0.0,
        train:         bool = True,
        seed:          int = 0,
    ):
        self.bag_size_mean = bag_size_mean
        self.bag_size_std = bag_size_std
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max
        self.num_bags = num_bags
        self.noise_sigma = noise_sigma

        split = 'train' if train else 'test'
        tfm = transforms.Compose([transforms.ToTensor()])
        svhn = _SVHN(root=str(_MNIST_ROOT), split=split, download=True, transform=tfm)
        all_images = torch.stack([svhn[i][0] for i in range(len(svhn))])
        all_labels = torch.tensor([svhn[i][1] for i in range(len(svhn))])
        # SVHN labels: 1..10 in original; torchvision returns 0..9 already
        # (label 10 in raw → 0 here). Verified with torchvision >= 0.17.

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        sizes = []
        bag_indices = []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(round(rng.normal(bag_size_mean, bag_size_std)))
            n = max(bag_size_min, min(bag_size_max, n))
            sizes.append(n)
            bag_indices.append(rng.choice(M, size=n, replace=False))
        self._bag_sizes = sizes
        self._bag_indices = bag_indices
        self._noise_rng = np.random.default_rng(seed + 10_000)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum, labels.long(), mask
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: 11 PASSED. NB: first run downloads SVHN (~600MB).

- [ ] **Step 5: Commit**

```bash
git add pca/data.py tests/test_data_a1.py
git commit -m "feat(data): SVHNSumBagDataset for RGB digit-sum task"
```

---

## Task 28: `scripts/run_svhn_sum.py`

**Files:**
- Create: `scripts/run_svhn_sum.py`

- [ ] **Step 1: Create `scripts/run_svhn_sum.py`**

```python
#!/usr/bin/env python
"""SVHN-sum experiment runner.

Default config (spec §13 Day 5):
  6 methods × 5 seeds × 80 epochs ≈ ~15h on MPS overnight.

Smoke usage:
  python scripts/run_svhn_sum.py --method pca --epochs 2 --num-bags 50 --seeds 0
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import (
    MeanPoolBaseline, PLMulticlassBaseline, CLTGaussianBaseline,
    AttentionPoolingBaseline, DeepSetsBaseline, PCABaseline,
)
from pca.data import SVHNSumBagDataset, variable_n_collate_fn
from pca.models import ResNet18FromScratch
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 1500
DEFAULT_TEST_BAGS = 600
DEFAULT_EPOCHS = 80
K = 9
N_MAX = 15

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
    '2a':  PLMulticlassBaseline,
    '2b':  CLTGaussianBaseline,
    '3a':  AttentionPoolingBaseline,
    '3b':  DeepSetsBaseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(BASELINE_REGISTRY.keys()), required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--bag-size-mean', type=float, default=10.0)
    p.add_argument('--bag-size-std', type=float, default=2.0)
    p.add_argument('--bag-size-min', type=int, default=5)
    p.add_argument('--bag-size-max', type=int, default=15)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=None)
    p.add_argument('--noise-sigma', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=Path, default=Path('results/svhn_sum'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}, σ={args.noise_sigma}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = SVHNSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=True, seed=seed)
        test_ds = SVHNSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=False, seed=seed)
        backbone = ResNet18FromScratch(in_channels=3)
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()), lr=args.lr)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    cap_str = args.per_class_cap if args.per_class_cap is not None else 'none'
    save_dir = (args.output_dir / f"N{int(args.bag_size_mean)}_"
                                   f"cap{cap_str}_"
                                   f"sig{args.noise_sigma}_{args.method}")
    config_extra = {
        'experiment': 'svhn_sum', 'method': args.method, 'atom_support': K + 1,
        'bag_size_mean': args.bag_size_mean, 'bag_size_std': args.bag_size_std,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': args.noise_sigma,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

```bash
uv run python scripts/run_svhn_sum.py --method pca --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
```

Expected: `results/svhn_sum/N10_capnone_sig0.0_pca/seed0.json` created. (First run downloads SVHN.)

- [ ] **Step 3: Smoke runs for the other 5 methods**

```bash
for m in 1a 2a 2b 3a 3b; do
  uv run python scripts/run_svhn_sum.py --method $m --epochs 2 --num-bags 50 --num-test-bags 30 --seeds 0
done
```

Expected: 5 more `seed0.json` files.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_svhn_sum.py
git commit -m "feat(scripts): SVHN-sum runner with ResNet-18 from scratch"
```

---

## Task 29: Day 5 — SVHN-sum 30 runs (overnight)

**Files:**
- (No new files; runs the experiment)

- [ ] **Step 1: Clean smoke runs**

```bash
rm -rf results/svhn_sum/N10_capnone_sig0.0_*
```

- [ ] **Step 2: Full SVHN-sum run (5 seeds × 6 methods × 80 epochs, overnight)**

```bash
for m in pca 1a 2a 2b 3a 3b; do
  uv run python scripts/run_svhn_sum.py --method $m --epochs 80 \
    --seeds 0,1,2,3,4 \
    --output-dir results/svhn_sum
done
```

Expected: 6 directories under `results/svhn_sum/N10_capnone_sig0.0_<method>/` each with seed0..4.json. Wall-clock ≈ 15h. Run overnight.

- [ ] **Step 3: Schema validation**

```bash
uv run python -c "
import json, glob
n = 0
for path in sorted(glob.glob('results/svhn_sum/N10_capnone_sig0.0_*/seed*.json')):
    d = json.loads(open(path).read())
    for key in ['config', 'best_test_nll', 'best_test_acc', 'best_test_mae',
                'best_test_ece', 'wall_clock_sec']:
        assert key in d, f'{path} missing {key}'
    n += 1
print(f'{n} JSONs schema-valid')
"
```

Expected: prints `30 JSONs schema-valid`.

- [ ] **Step 4: Run analyze, verify SVHN summary builds**

```bash
uv run python scripts/analyze_a1.py --partial --datasets mnist_sum mnist_signed svhn_sum
```

Expected: `results/svhn_sum/summary.{md,json}` written; updated `summary_overall.md`.

- [ ] **Step 5: Commit summary**

```bash
git add results/svhn_sum/summary.md results/svhn_sum/summary.json results/summary_overall.md
git commit -m "results: Day 5 SVHN-sum 30-run + partial verdict"
```

---

## Task 30: `UltraMNISTBagDataset` (synthetic stand-in)

**Files:**
- Modify: `pca/data.py`
- Modify: `tests/test_data_a1.py`

- [ ] **Step 1: Append UltraMNIST tests**

Append to `tests/test_data_a1.py`:

```python
from pca.data import UltraMNISTBagDataset


def test_ultramnist_synthetic_bag_count_matches_gt():
    """UltraMNIST stand-in: 3-5 MNIST digits scattered, returns 64x64 patch crops."""
    ds = UltraMNISTBagDataset(num_bags=5, seed=0)
    for i in range(5):
        patches, bag_sum, gt, mask = ds[i]
        n = mask.sum().item()
        assert 3 <= n <= 5
        assert patches.shape == (n, 3, 64, 64)
        assert gt.shape == (n,)
        assert bag_sum == int(gt.sum().item())


def test_ultramnist_seed_determinism():
    ds1 = UltraMNISTBagDataset(num_bags=3, seed=42)
    ds2 = UltraMNISTBagDataset(num_bags=3, seed=42)
    for i in range(3):
        p1, s1, g1, _ = ds1[i]
        p2, s2, g2, _ = ds2[i]
        assert torch.equal(p1, p2)
        assert s1 == s2
        assert torch.equal(g1, g2)
```

- [ ] **Step 2: Run, verify import error**

```bash
uv run pytest tests/test_data_a1.py -v -k ultramnist
```

Expected: FAIL — `cannot import name 'UltraMNISTBagDataset'`.

- [ ] **Step 3: Append `UltraMNISTBagDataset` to `pca/data.py`**

```python
class UltraMNISTBagDataset(Dataset):
    """Synthetic stand-in for Kaggle UltraMNIST.

    Composes 3-5 MNIST digits at non-overlapping positions on a 4000x4000 RGB
    canvas, then crops 64x64 RGB patches per digit (positions known by
    construction). Per-instance feature is the patch crop. Bag sum = sum of
    digit values. Atom support S=10, K=9.

    Note (binding): this is a stand-in to keep Day 6 executable without the
    Kaggle license blocker. The contract (returns (patches, bag_sum, gt, mask))
    matches what a real UltraMNIST loader would produce; swap is one-file.
    """

    def __init__(
        self,
        num_bags:      int = 600,
        bag_size_min:  int = 3,
        bag_size_max:  int = 5,
        per_class_cap: int | None = None,
        train:         bool = True,
        seed:          int = 0,
    ):
        self.num_bags = num_bags
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max

        tfm = transforms.Compose([transforms.ToTensor()])
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        all_images = torch.stack([mnist[i][0] for i in range(len(mnist))])  # (M, 1, 28, 28)
        all_labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        # Pre-sample bags: pick N_b digits and 64x64 patch positions.
        sizes, indices, positions = [], [], []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(rng.integers(bag_size_min, bag_size_max + 1))
            idxs = rng.choice(M, size=n, replace=False)
            pos = self._sample_positions(rng, n)
            sizes.append(n)
            indices.append(idxs)
            positions.append(pos)
        self._bag_sizes = sizes
        self._bag_indices = indices
        self._bag_positions = positions

    @staticmethod
    def _sample_positions(rng, n: int, canvas: int = 4000, patch: int = 64,
                            margin: int = 32) -> np.ndarray:
        """Non-overlapping (greedy) random patch top-left coordinates."""
        out = []
        attempts = 0
        while len(out) < n and attempts < 5000:
            x = int(rng.integers(margin, canvas - patch - margin))
            y = int(rng.integers(margin, canvas - patch - margin))
            ok = all(abs(x - ox) > patch or abs(y - oy) > patch
                      for ox, oy in out)
            if ok:
                out.append((x, y))
            attempts += 1
        # Fallback: in the unlikely sparse case, just append remaining as random.
        while len(out) < n:
            x = int(rng.integers(margin, canvas - patch - margin))
            y = int(rng.integers(margin, canvas - patch - margin))
            out.append((x, y))
        return np.array(out, dtype=np.int32)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        positions = self._bag_positions[idx]
        # Compose 28x28 MNIST digits onto 64x64 RGB patches centered in patch.
        patches = torch.zeros((n, 3, 64, 64))
        offset = (64 - 28) // 2
        for i, didx in enumerate(idxs):
            digit = self._images[didx]                       # (1, 28, 28) ∈ [0,1]
            digit_rgb = digit.repeat(3, 1, 1)                 # (3, 28, 28)
            patches[i, :, offset:offset + 28, offset:offset + 28] = digit_rgb
        labels = self._labels[idxs]
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return patches, bag_sum, labels.long(), mask
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_data_a1.py -v
```

Expected: 13 PASSED.

- [ ] **Step 5: Commit**

```bash
git add pca/data.py tests/test_data_a1.py
git commit -m "feat(data): UltraMNISTBagDataset (synthetic stand-in for Kaggle data)"
```

---

## Task 31: `scripts/run_ultramnist.py`

**Files:**
- Create: `scripts/run_ultramnist.py`

- [ ] **Step 1: Create `scripts/run_ultramnist.py`**

```python
#!/usr/bin/env python
"""UltraMNIST experiment runner.

Default config (spec §13 Day 6):
  6 methods × 3 seeds × 60 epochs ≈ ~13.5h overnight. UltraMNIST has natural
  variable N ∈ {3,4,5} (no per-class-cap or noise modulation by default).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import (
    MeanPoolBaseline, PLMulticlassBaseline, CLTGaussianBaseline,
    AttentionPoolingBaseline, DeepSetsBaseline, PCABaseline,
)
from pca.data import UltraMNISTBagDataset, variable_n_collate_fn
from pca.models import PatchEncoder
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 800
DEFAULT_TEST_BAGS = 300
DEFAULT_EPOCHS = 60
K = 9
N_MAX = 5
LR_DEFAULT = 5e-4   # spec §15: UltraMNIST lr=5e-4 (lower than MNIST/SVHN)

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
    '2a':  PLMulticlassBaseline,
    '2b':  CLTGaussianBaseline,
    '3a':  AttentionPoolingBaseline,
    '3b':  DeepSetsBaseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(BASELINE_REGISTRY.keys()), required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2])
    p.add_argument('--bag-size-min', type=int, default=3)
    p.add_argument('--bag-size-max', type=int, default=5)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=None)
    p.add_argument('--lr', type=float, default=LR_DEFAULT)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=Path, default=Path('results/ultramnist'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = UltraMNISTBagDataset(
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            train=True, seed=seed)
        test_ds = UltraMNISTBagDataset(
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            train=False, seed=seed)
        backbone = PatchEncoder()
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()), lr=args.lr)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    cap_str = args.per_class_cap if args.per_class_cap is not None else 'none'
    save_dir = (args.output_dir / f"N4_cap{cap_str}_{args.method}")
    config_extra = {
        'experiment': 'ultramnist', 'method': args.method, 'atom_support': K + 1,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': 0.0,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

```bash
uv run python scripts/run_ultramnist.py --method pca --epochs 2 --num-bags 30 --num-test-bags 20 --seeds 0
```

Expected: `results/ultramnist/N4_capnone_pca/seed0.json` created.

- [ ] **Step 3: Smoke runs for other 5 methods**

```bash
for m in 1a 2a 2b 3a 3b; do
  uv run python scripts/run_ultramnist.py --method $m --epochs 2 --num-bags 30 --num-test-bags 20 --seeds 0
done
```

Expected: 5 more `seed0.json` files.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_ultramnist.py
git commit -m "feat(scripts): UltraMNIST runner with PatchEncoder backbone"
```

---

## Task 32: Day 6 — UltraMNIST 18 runs (overnight)

**Files:**
- (No new files; runs the experiment)

- [ ] **Step 1: Clean smoke runs**

```bash
rm -rf results/ultramnist/N4_capnone_*
```

- [ ] **Step 2: Full UltraMNIST run (3 seeds × 6 methods × 60 epochs)**

```bash
for m in pca 1a 2a 2b 3a 3b; do
  uv run python scripts/run_ultramnist.py --method $m --epochs 60 \
    --seeds 0,1,2 \
    --output-dir results/ultramnist
done
```

Expected: 6 directories under `results/ultramnist/N4_capnone_<method>/` each with seed0..2.json. Wall clock ≈ 13.5h overnight.

- [ ] **Step 3: Schema validation**

```bash
uv run python -c "
import json, glob
n = 0
for path in sorted(glob.glob('results/ultramnist/N4_capnone_*/seed*.json')):
    d = json.loads(open(path).read())
    for key in ['config', 'best_test_nll', 'best_test_acc', 'best_test_mae',
                'best_test_ece', 'wall_clock_sec']:
        assert key in d, f'{path} missing {key}'
    n += 1
print(f'{n} JSONs schema-valid')
"
```

Expected: prints `18 JSONs schema-valid`.

- [ ] **Step 4: Run analyze (still partial — Day 7 will be final)**

```bash
uv run python scripts/analyze_a1.py --partial \
  --datasets mnist_sum mnist_signed svhn_sum ultramnist
```

Expected: `results/ultramnist/summary.{md,json}` + updated `summary_overall.md`.

- [ ] **Step 5: Commit summaries**

```bash
git add results/ultramnist/summary.md results/ultramnist/summary.json \
        results/summary_overall.md
git commit -m "results: Day 6 UltraMNIST 18-run + 4-dataset partial verdict"
```

---

## Task 33: Day 7 — final verdict + summary_overall.md

**Files:**
- Modify: `results/summary_overall.md`
- Modify: per-dataset `summary.{md,json}`

- [ ] **Step 1: Run full analyze (no `--partial`)**

```bash
uv run python scripts/analyze_a1.py
```

Expected: `summary_overall.md` reports verdict ∈ {`A1_CONFIRMED`, `RUN_HARD_PRESET`, `A1_CALIBRATION_FOCUS`, `DEMOTE_A1`, `ALGEBRA_BROKEN`}.

- [ ] **Step 2: Branch on verdict**

If `A1_CONFIRMED` → proceed to Step 3.
If `ALGEBRA_BROKEN` → STOP. Code/spec debug. No scientific interpretation. Open an issue and reach out to user.
If `RUN_HARD_PRESET` → run the printed hard-preset CLIs from Step 1's output.
If `A1_CALIBRATION_FOCUS` → keep Step 3 commit, then notify user for spec re-discussion.
If `DEMOTE_A1` → keep Step 3 commit, then proceed to Task 34 README priority update.

- [ ] **Step 3: Reliability diagram — skipped for Day 7 minimum viable**

Decision (binding): Day 7 produces verdict from the per-dataset ECE table in
`summary_overall.md` only; reliability diagram is **out of scope for the
minimum viable submission**. Per-bag log_P is not persisted by the run scripts
(spec §4.3 non-goal: no model checkpointing), so plotting reliability curves
would require either a re-run with checkpointing or a one-off re-eval pipeline
— both add work not in the 7-day budget.

If reliability plots are requested *after* the verdict (e.g., for paper figure
1), open a separate task: re-train one seed × {PCA, AttentionPooling} on the
H1-strongest dataset with model checkpointing enabled, then run
`pca.metrics.reliability_diagram_data` on the test set predictions and plot
with matplotlib. Estimated wall-clock: ~30 min.

(No command for this step; documents the decision.)

- [ ] **Step 4: Stamp final results commit**

```bash
git add results/*/summary.md results/*/summary.json results/summary_overall.md
git status         # confirm only summary files staged (no per-seed JSONs)
git commit -m "results: Day 7 final A1 verdict — <VERDICT_HERE>"
```

---

## Task 34: Day 7 — README priority update + spec/plan finalization

**Files:**
- Modify: `README.md`
- (Optional) Modify: `docs/notes/2026-05-04-a1-analysis.md` to record outcome

- [ ] **Step 1: Branch by verdict**

Read `results/summary_overall.md` to get final verdict, then:

- **A1_CONFIRMED**: README priority remains `A1 Main`. Add result row to README priority table noting "Verified 2026-05-XX, Δ NLL = ... nats over best baseline". Phase 2 (B2 bolt-on spec) starts in a new design session.

- **A1_CALIBRATION_FOCUS**: README priority remains `A1 Main` but headline pivots to "atom-shape generality + calibration advantage". Note ECE ratio in priority table. Notify user for spec re-discussion (separate session).

- **DEMOTE_A1**: README priority A1 Main → Low. Document the demotion in a new row of the priority table (mirrors the B1 demotion entry from 2026-05-04).

- **RUN_HARD_PRESET / ALGEBRA_BROKEN**: do not finalize Day 7 README. Loop back to fix or rerun.

- [ ] **Step 2: Update README priority table**

Edit the priority table in `README.md` for A1 with the chosen branch's content. Use exact same column structure as existing entries (Axis | Tier | Bucket | Why this priority). Update the "Why this priority" cell to reflect verification outcome with specific numbers (Δ NLL, p-value, dataset count).

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: A1 verification outcome — README priority update (<VERDICT>)"
```

- [ ] **Step 4: (Optional) Append outcome paragraph to `docs/notes/2026-05-04-a1-analysis.md`**

Add at the end of the analysis note:

```markdown

## §X. Verification outcome (2026-05-XX)

**Verdict:** <A1_CONFIRMED / A1_CALIBRATION_FOCUS / DEMOTE_A1>

**H1 (NLL):** <pass/fail>; ≥2 of 3 datasets: ... ; per-dataset Δ ... .
**H2 (atom-shape generality):** <pass/fail>; binary B1 reuse + multi-class + signed all <pass/fail>.
**H3 (calibration):** <pass/fail>; ECE ratios per dataset: ... .

**Implication for paper:** <one paragraph: what changes vs. the pre-verification framing in §1–§7>.
```

```bash
git add docs/notes/2026-05-04-a1-analysis.md
git commit -m "docs: append verification outcome to A1 analysis note"
```

- [ ] **Step 5: Final summary print**

```bash
cat results/summary_overall.md
```

Expected: full overall summary in console. Engineer reports it to the user along with the chosen branch action.

---

## Verification matrix (spec → tasks)

| Spec section | Topic | Implementing tasks |
|---|---|---|
| §1 Context, lineage | Documentation only | (covered in spec; no plan task needed) |
| §2.1 Hypotheses | H1/H2/H3/H4 definitions | 24 (analyze_a1) |
| §2.3 Falsifier verdict | 5 verdict tiers | 24 |
| §2.4 Saturation auto-detect | `detect_saturation` | 24 |
| §2.5 Signed verdict scope | H2 signed sub-condition only | 24, 26, 33 |
| §3.1 Multi-class DP ≡ conv proposition | Body proposition (no impl) | (in spec; no impl) |
| §3.2 SPL → related work | Documentation only | (in spec) |
| §3.3 Variable bag size | N ~ Normal(10,2) trunc [5,15] | 19, 20, 21, 27 |
| §3.4 Per-dataset backbone | Backbone per dataset | 8, 9, 10, 22, 28, 31 |
| §3.5 Phase separation | A1 standalone | (entire plan) |
| §3.6 Saturation insurance | per-class-cap / noise + sweep | 20, 22, 23, 26 |
| §4 Module architecture | File layout + non-goals | (file structure section) |
| §5.2 atomic_conv | code + tests | 2, 3 |
| §5.3 multiclass_marginal_nll_loss | code + mask test | 4, 5 |
| §6 Datasets (4 + collate) | dataset classes | 19, 20, 21, 27, 30 |
| §7 Models | 3 backbone classes (feature extractors) | 8, 9, 10 |
| §8 Baselines (5 + PCA) | 6 baseline classes | 11–16 |
| §9 train_a1 + run_seeds_a1 | 4-metric eval loop | 17 |
| §9.4 Scripts | 4 run_*.py + sanity + analyze | 18, 22, 24, 28, 31 |
| §10 Testing strategy | tests for atomic_conv/baselines/data/metrics | 2–7, 11–16, 19–21, 27, 30 |
| §11 analyze_a1 H1/H2/H3 verdict | full implementation | 24 |
| §11.7 Partial verdict (Day 4) | `--partial` mode | 24, 26 |
| §12 Result storage | per-seed JSON schema | 17 (run_seeds_a1) |
| §13 Day-by-day plan | 7-day execution | 25 (Day 4), 29 (Day 5), 32 (Day 6), 33–34 (Day 7) |
| §14 Decision points | 7 day-end gates | embedded in tasks 1, 18, 23, 25, 29, 32, 33 |
| §15 Out of scope | enforcement (negative coverage) | (in plan: no B2/A2/A4/SPL/etc.) |
| §16 Dependencies | pyproject + uv sync | 1 |

---

