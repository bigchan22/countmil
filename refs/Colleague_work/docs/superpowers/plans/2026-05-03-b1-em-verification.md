# B1 Count-conditioned EM Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the Day 1–7 protocol from the B1 verification design spec to test whether explicit count-conditioned EM beats marginal NLL on instance-level AUC.

**Architecture:** Module + scripts layout (`pca/` reusable code, `scripts/` entry points, `tests/` unit tests). Approach A (joint EM loss with `lam=0` ≡ NLL for free sanity). All convs in log space. Sequential per-seed runs (no parallelization).

**Tech Stack:** Python ≥3.11, PyTorch ≥2.2, torchvision, scikit-learn (AUC + Adult fetch), scipy (paired t-test), pandas (analyze), uv for env management.

**Spec:** `docs/superpowers/specs/2026-05-03-b1-em-verification-design.md` (committed at `46897e1`).

---

## File structure

| Path | Created in task | Responsibility |
|---|---|---|
| `pyproject.toml` | 1 | uv-managed project config |
| `pca/__init__.py` | 1 | package marker |
| `pca/losses.py` | 2–8 | forward_conv, leave_one_out_posterior, marginal_nll_loss, em_joint_loss |
| `pca/data.py` | 10–11 | MNISTBagDataset, LLPBagDataset |
| `pca/models.py` | 12 | SmallCNN, MLP |
| `pca/train.py` | 13–15 | per_bag_forward, select_device, train(), run_seeds() |
| `scripts/sanity_n4.py` | 16 | end-to-end sanity check before any experiment |
| `scripts/run_mnist_mil.py` | 17 | MNIST-MIL experiment entrypoint |
| `scripts/run_llp_adult.py` | 21 | LLP Adult experiment entrypoint |
| `scripts/analyze.py` | 19 | results → markdown summary + verdict |
| `tests/test_losses.py` | 2–9 | unit tests for losses (10 tests) |
| `tests/test_data.py` | 10–11 | unit tests for data (5 tests) |
| `.gitignore` | 1 | exclude results/, .venv/, caches |

---

## Task 1: Project setup (uv + scaffold)

**Files:**
- Create: `pyproject.toml`
- Create: `pca/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "pca"
version = "0.1.0"
description = "Probabilistic Convolutional Aggregator — B1 verification"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "numpy>=1.26",
  "pandas>=2.1",
  "scikit-learn>=1.3",
  "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pca"]
```

- [ ] **Step 2: Create `pca/__init__.py`**

```python
"""Probabilistic Convolutional Aggregator — B1 verification package."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
results/*/N*_*/
*.DS_Store
.DS_Store
~/.cache/pca/
```

- [ ] **Step 5: Set up environment**

Run:
```bash
cd /Users/bhwang/Documents/Projects/probabilistic-convolutional-aggregator
uv venv
uv pip install -e ".[dev]"
```

Expected: `.venv/` created, package installed in editable mode.

- [ ] **Step 6: Verify environment**

Run:
```bash
uv run python -c "import torch; print('torch', torch.__version__); import pca; print('pca', pca.__version__)"
uv run pytest --version
```

Expected: torch ≥2.2, pca 0.1.0, pytest ≥7.4.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml pca/__init__.py tests/__init__.py .gitignore
git commit -m "chore: initial uv project scaffold for B1 verification"
```

---

## Task 2: `forward_conv` — first test + implementation

**Files:**
- Create: `tests/test_losses.py`
- Create: `pca/losses.py`

- [ ] **Step 1: Create `tests/test_losses.py` with the first test**

```python
"""Unit tests for pca/losses.py — see spec §10.1."""
import torch
import torch.nn.functional as F

from pca.losses import forward_conv


def test_forward_conv_n2_hand():
    """
    p = [0.3, 0.4]
    P(Σ=0) = 0.7 * 0.6           = 0.42
    P(Σ=1) = 0.3*0.6 + 0.7*0.4   = 0.46
    P(Σ=2) = 0.3 * 0.4           = 0.12
    """
    p = torch.tensor([[0.3, 0.4]])
    log_P = forward_conv(p)
    P = log_P.exp()
    expected = torch.tensor([[0.42, 0.46, 0.12]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_losses.py::test_forward_conv_n2_hand -v
```

Expected: FAIL with `ImportError: cannot import name 'forward_conv'`.

- [ ] **Step 3: Create `pca/losses.py` with `forward_conv`**

```python
"""Loss functions for the Probabilistic Convolutional Aggregator (PCA).

All routines work in log space for numerical stability. See
docs/superpowers/specs/2026-05-03-b1-em-verification-design.md §5 for math.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-7

# Log-zero sentinel: avoids logaddexp(-inf, -inf) → NaN gradient
# (exp(-inf - (-inf)) = exp(NaN) = NaN). -1e30 gives finite (0.5, 0.5)
# backward, with exp(-1e30) ≈ 0 to far below float32 precision so forward
# values at structurally unreachable positions are unchanged. Used in
# forward_conv (which is differentiable). leave_one_out_posterior keeps
# float('-inf') because it runs under torch.no_grad().
LOG_ZERO = -1e30


def forward_conv(p: Tensor) -> Tensor:
    """Compute log P(sum z_i = k), k = 0..N, via sequential conv in log space.

    Args:
        p: (B, N) per-instance Bernoulli probability ∈ (0, 1).
    Returns:
        log_P: (B, N+1) — log_P[b, k] = log P(sum z_{b,i} = k).
    Complexity: O(B * N^2).
    Autograd-safe at structurally unreachable positions (uses LOG_ZERO sentinel).
    """
    B, N = p.shape
    log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
    log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()
    log_P = p.new_full((B, N + 1), LOG_ZERO)
    log_P[:, 0] = 0.0
    for i in range(N):
        stay = log_p0[:, i:i + 1] + log_P
        shifted = F.pad(log_P[:, :-1], (1, 0), value=LOG_ZERO)
        shift = log_p1[:, i:i + 1] + shifted
        log_P = torch.logaddexp(stay, shift)
    return log_P
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/test_losses.py::test_forward_conv_n2_hand -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_losses.py pca/losses.py
git commit -m "feat(losses): forward_conv with N=2 hand-computed test"
```

---

## Task 3: `forward_conv` — additional tests

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Add N=3 hand-computed test**

Append to `tests/test_losses.py`:

```python
def test_forward_conv_n3_hand():
    """
    p = [0.5, 0.5, 0.5]
    Independent fair coins → Binomial(3, 0.5):
      P(0) = 1/8, P(1) = 3/8, P(2) = 3/8, P(3) = 1/8.
    """
    p = torch.tensor([[0.5, 0.5, 0.5]])
    log_P = forward_conv(p)
    P = log_P.exp()
    expected = torch.tensor([[0.125, 0.375, 0.375, 0.125]])
    assert torch.allclose(P, expected, atol=1e-6)


def test_forward_conv_sums_to_one():
    """For any valid p, sum_k P(Σ=k) ≈ 1."""
    torch.manual_seed(0)
    p = torch.rand(8, 50)
    log_P = forward_conv(p)
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), f"got {sums.tolist()}"
```

- [ ] **Step 2: Run all forward_conv tests**

```bash
uv run pytest tests/test_losses.py -v -k forward_conv
```

Expected: 3 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): N=3 binomial + sum-to-one for forward_conv"
```

---

## Task 4: `leave_one_out_posterior` — N=2 hand-computed test + implementation

**Files:**
- Modify: `tests/test_losses.py`
- Modify: `pca/losses.py`

- [ ] **Step 1: Add N=2 posterior test**

Append to `tests/test_losses.py`:

```python
from pca.losses import leave_one_out_posterior


def test_posterior_n2_hand():
    """
    p = [0.3, 0.4], y = 1.
    P_{j≠0}(Σ=0) = 1-p_1 = 0.6,  P_{j≠0}(Σ=1) = p_1 = 0.4
    P_{j≠1}(Σ=0) = 1-p_0 = 0.7,  P_{j≠1}(Σ=1) = p_0 = 0.3
    P(Σ=1) = 0.46
    q_0 = p_0 * P_{j≠0}(Σ=0) / P(Σ=1) = 0.3 * 0.6 / 0.46 ≈ 0.39130
    q_1 = p_1 * P_{j≠1}(Σ=0) / P(Σ=1) = 0.4 * 0.7 / 0.46 ≈ 0.60870
    """
    p = torch.tensor([[0.3, 0.4]])
    bag_y = torch.tensor([1])
    q = leave_one_out_posterior(p, bag_y)
    expected = torch.tensor([[0.391304, 0.608696]])
    assert torch.allclose(q, expected, atol=1e-4), f"got {q.tolist()}"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_losses.py::test_posterior_n2_hand -v
```

Expected: FAIL — `cannot import name 'leave_one_out_posterior'`.

- [ ] **Step 3: Add `leave_one_out_posterior` to `pca/losses.py`**

Append to `pca/losses.py`:

```python
def leave_one_out_posterior(p: Tensor, bag_y: Tensor) -> Tensor:
    """Posterior q_u = P(z_u=1 | sum=bag_y) via prefix·suffix conv in log space.

    Args:
        p:     (B, N) per-instance Bernoulli probability ∈ (0, 1).
        bag_y: (B,)   integer count target ∈ [0, N].
    Returns:
        q: (B, N) detached posterior.
    Complexity: O(B * N^2).
    """
    with torch.no_grad():
        # Note: uses float('-inf') (not LOG_ZERO) because the no_grad
        # context disables autograd, so the logaddexp(-inf,-inf) NaN-gradient
        # bug that motivated LOG_ZERO in forward_conv does not apply here.
        B, N = p.shape
        log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
        log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()

        # prefix[:, i, :] = log PMF of d_0..d_{i-1}
        prefix = p.new_full((B, N + 1, N + 1), float('-inf'))
        prefix[:, 0, 0] = 0.0
        for i in range(N):
            stay = log_p0[:, i:i + 1] + prefix[:, i, :]
            shift = log_p1[:, i:i + 1] + F.pad(prefix[:, i, :-1], (1, 0), value=float('-inf'))
            prefix[:, i + 1, :] = torch.logaddexp(stay, shift)

        # suffix[:, i, :] = log PMF of d_i..d_{N-1}
        suffix = p.new_full((B, N + 1, N + 1), float('-inf'))
        suffix[:, N, 0] = 0.0
        for i in reversed(range(N)):
            stay = log_p0[:, i:i + 1] + suffix[:, i + 1, :]
            shift = log_p1[:, i:i + 1] + F.pad(suffix[:, i + 1, :-1], (1, 0), value=float('-inf'))
            suffix[:, i, :] = torch.logaddexp(stay, shift)

        # log P(Σ=bag_y) — recovered from prefix[N], same as forward_conv(p)
        log_P_full = prefix[:, N, :]                                       # (B, N+1)
        log_P_y = log_P_full.gather(1, bag_y.unsqueeze(1)).squeeze(1)      # (B,)

        # log P_{j≠u}(Σ=bag_y - 1) via 1-D conv in log space
        log_P_excl_at = p.new_full((B, N), float('-inf'))
        k_grid = torch.arange(N + 1, device=p.device).unsqueeze(0)         # (1, N+1)
        target = (bag_y - 1).unsqueeze(1)                                   # (B, 1)
        for u in range(N):
            sk_idx = target - k_grid                                       # (B, N+1)
            valid = (sk_idx >= 0) & (sk_idx <= N)
            sk_clamped = sk_idx.clamp(min=0, max=N)
            suffix_term = suffix[:, u + 1, :].gather(1, sk_clamped)        # (B, N+1)
            terms = torch.where(valid, prefix[:, u, :] + suffix_term,
                                torch.full_like(prefix[:, u, :], float('-inf')))
            log_P_excl_at[:, u] = torch.logsumexp(terms, dim=1)

        log_q = log_p1 + log_P_excl_at - log_P_y.unsqueeze(1)
        q = log_q.exp().clamp(min=0.0, max=1.0)

        # Edge cases
        q = torch.where(bag_y.unsqueeze(1) == 0, torch.zeros_like(q), q)
        q = torch.where(bag_y.unsqueeze(1) == N, torch.ones_like(q), q)

    return q
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/test_losses.py::test_posterior_n2_hand -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_losses.py pca/losses.py
git commit -m "feat(losses): leave_one_out_posterior with N=2 hand-computed test"
```

---

## Task 5: `leave_one_out_posterior` — invariants + edge cases

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Add invariant tests**

Append to `tests/test_losses.py`:

```python
def test_posterior_sum_equals_count():
    """Σ_u q_u should equal bag_y (posterior expected count = observed count)."""
    torch.manual_seed(1)
    p = torch.rand(8, 20)
    bag_y = torch.randint(0, 21, (8,))
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q.sum(dim=1), bag_y.float(), atol=1e-3), f"got {q.sum(dim=1).tolist()} vs {bag_y.tolist()}"


def test_posterior_y_zero():
    """If bag_y == 0, no positives → q ≡ 0."""
    p = torch.rand(4, 10)
    bag_y = torch.zeros(4, dtype=torch.long)
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q, torch.zeros_like(q))


def test_posterior_y_n():
    """If bag_y == N, all positive → q ≡ 1."""
    p = torch.rand(4, 10)
    bag_y = torch.full((4,), 10, dtype=torch.long)
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q, torch.ones_like(q))
```

- [ ] **Step 2: Run all posterior tests**

```bash
uv run pytest tests/test_losses.py -v -k posterior
```

Expected: 4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): posterior invariants (sum=count, y=0, y=N)"
```

---

## Task 6: `marginal_nll_loss` — implementation + smoke test

**Files:**
- Modify: `pca/losses.py`
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Add `marginal_nll_loss` to `pca/losses.py`**

Append to `pca/losses.py`:

```python
def marginal_nll_loss(logits: Tensor, bag_y: Tensor) -> Tensor:
    """Shukla-style aggregate NLL: -log P(Σ z = bag_y).

    Args:
        logits: (B, N) per-instance pre-sigmoid scores.
        bag_y:  (B,)  long integer in [0, N].
    Returns:
        scalar mean over batch.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()
```

- [ ] **Step 2: Add a smoke test for `marginal_nll_loss`**

Append to `tests/test_losses.py`:

```python
from pca.losses import marginal_nll_loss


def test_marginal_nll_returns_scalar_finite():
    torch.manual_seed(2)
    logits = torch.randn(4, 10)
    bag_y = torch.randint(0, 11, (4,))
    loss = marginal_nll_loss(logits, bag_y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_losses.py::test_marginal_nll_returns_scalar_finite -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pca/losses.py tests/test_losses.py
git commit -m "feat(losses): marginal_nll_loss + smoke test"
```

---

## Task 7: `em_joint_loss` + `lam=0` equivalence

**Files:**
- Modify: `pca/losses.py`
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Add `em_joint_loss` to `pca/losses.py`**

Append to `pca/losses.py`:

```python
def em_joint_loss(logits: Tensor, bag_y: Tensor, lam: float = 1.0) -> Tensor:
    """Approach A joint loss: marginal NLL + λ × CE(p, q.detach()).

    `lam=0.0` returns marginal_nll_loss exactly (numerical equivalence,
    no posterior computation). See spec §3.3 and §5.5.

    Args:
        logits: (B, N) per-instance pre-sigmoid scores.
        bag_y:  (B,)  long integer in [0, N].
        lam:    weight on the soft cross-entropy.
    Returns:
        scalar mean over batch.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    nll = -log_P_y.mean()

    if lam == 0.0:
        return nll

    q = leave_one_out_posterior(p, bag_y)
    log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
    log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()
    ce = -(q * log_p1 + (1 - q) * log_p0).mean()
    return nll + lam * ce
```

- [ ] **Step 2: Add `lam=0` equivalence test**

Append to `tests/test_losses.py`:

```python
from pca.losses import em_joint_loss


def test_em_loss_lam_zero_equals_nll():
    """lam=0 must be numerically equivalent to marginal_nll_loss within 1e-6."""
    torch.manual_seed(3)
    logits = torch.randn(4, 10)
    bag_y = torch.randint(0, 11, (4,))
    nll = marginal_nll_loss(logits, bag_y)
    em0 = em_joint_loss(logits, bag_y, lam=0.0)
    assert torch.allclose(nll, em0, atol=1e-6), f"nll={nll.item()} em0={em0.item()}"
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_losses.py::test_em_loss_lam_zero_equals_nll -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pca/losses.py tests/test_losses.py
git commit -m "feat(losses): em_joint_loss with lam=0 ≡ NLL equivalence test"
```

---

## Task 8: Gradient cosine similarity + extreme stability tests

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Add gradient cosine + extremes tests**

Append to `tests/test_losses.py`:

```python
def test_em_loss_lam_zero_grad_cosine():
    """Gradient direction equivalence: cos(grad NLL, grad EM(lam=0)) ≥ 0.9999."""
    torch.manual_seed(4)
    logits = torch.randn(4, 10, requires_grad=True)
    bag_y = torch.randint(0, 11, (4,))

    g_nll, = torch.autograd.grad(marginal_nll_loss(logits, bag_y), logits, retain_graph=False)
    g_em0, = torch.autograd.grad(em_joint_loss(logits, bag_y, lam=0.0), logits, retain_graph=False)
    cos = (g_nll * g_em0).sum() / (g_nll.norm() * g_em0.norm() + 1e-12)
    assert cos.item() > 0.9999, f"cos={cos.item()}"


def test_em_loss_no_nan_at_extremes():
    """Saturated probabilities + boundary bag_y must remain finite."""
    torch.manual_seed(5)
    # near-saturated logits (large positive → p≈1)
    logits = torch.full((4, 10), 8.0, requires_grad=True)
    for y in [0, 5, 10]:
        bag_y = torch.full((4,), y, dtype=torch.long)
        loss = em_joint_loss(logits, bag_y, lam=1.0)
        assert torch.isfinite(loss), f"loss not finite at y={y}: {loss.item()}"
        g, = torch.autograd.grad(loss, logits, retain_graph=False)
        assert torch.isfinite(g).all(), f"grad not finite at y={y}"
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/test_losses.py -v
```

Expected: all 10 tests PASS, < 5s total.

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): gradient cosine + extreme-value stability"
```

---

## Task 9: Day 1 gate — full unit test sweep

- [ ] **Step 1: Run full pytest sweep + count tests**

```bash
uv run pytest tests/test_losses.py -v --tb=short
```

Expected: 10 passed in < 5s.

- [ ] **Step 2: Tag Day 1 completion in git log**

```bash
git tag -a day1-losses-complete -m "Day 1 gate: pca/losses.py + 10 unit tests passing"
git log --oneline -10
```

(No commit needed; just a tag for reference.)

**End-of-Day-1 deliverable**: `pca/losses.py` complete, all 10 tests pass.

---

## Task 10: Data — `MNISTBagDataset` + tests

**Files:**
- Create: `pca/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create `tests/test_data.py` with MNIST tests (will fail)**

```python
"""Unit tests for pca/data.py — see spec §10.2."""
import torch

from pca.data import MNISTBagDataset


def test_mnist_bag_count_matches_gt():
    ds = MNISTBagDataset(bag_size=10, num_bags=20, positive_digit=9, train=True, seed=0)
    for i in range(len(ds)):
        images, bag_count, gt_labels = ds[i]
        assert bag_count == int(gt_labels.sum().item())
        assert images.shape == (10, 1, 28, 28)
        assert gt_labels.shape == (10,)
        assert gt_labels.dtype == torch.long


def test_mnist_seed_determinism():
    ds1 = MNISTBagDataset(bag_size=10, num_bags=5, positive_digit=9, train=True, seed=42)
    ds2 = MNISTBagDataset(bag_size=10, num_bags=5, positive_digit=9, train=True, seed=42)
    for i in range(5):
        i1, c1, g1 = ds1[i]
        i2, c2, g2 = ds2[i]
        assert torch.equal(i1, i2)
        assert c1 == c2
        assert torch.equal(g1, g2)
```

- [ ] **Step 2: Create `pca/data.py` with `MNISTBagDataset`**

```python
"""Bag datasets for B1 verification.

MNISTBagDataset: bags of MNIST digits with count-of-positives label.
LLPBagDataset:   bags of UCI tabular instances (Adult, Magic) with count label.

See spec §6 for math/conventions.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

_MNIST_ROOT = Path.home() / ".cache" / "torch" / "datasets"


class MNISTBagDataset(Dataset):
    """MNIST-MIL bag dataset with count supervision.

    Returns per item: (images, bag_count, gt_labels)
        images:    (N, 1, 28, 28) float in [0, 1]
        bag_count: int — number of positive_digit instances in bag
        gt_labels: (N,) long — 1 iff label == positive_digit (eval only)
    """

    def __init__(
        self,
        bag_size: int,
        num_bags: int,
        positive_digit: int = 9,
        train: bool = True,
        seed: int = 0,
    ):
        self.bag_size = bag_size
        self.num_bags = num_bags
        self.positive_digit = positive_digit

        # Load + normalize underlying MNIST
        tfm = transforms.Compose([transforms.ToTensor()])  # → float [0, 1], shape (1, 28, 28)
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        # Materialize into tensors for fast indexing.
        self._images = torch.stack([mnist[i][0] for i in range(len(mnist))])     # (M, 1, 28, 28)
        self._labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])    # (M,)

        # Pre-sample bag indices deterministically from a seeded RNG.
        rng = np.random.default_rng(seed)
        M = self._images.shape[0]
        self._bag_indices = np.stack([
            rng.choice(M, size=bag_size, replace=False) for _ in range(num_bags)
        ])  # (num_bags, bag_size)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        idxs = self._bag_indices[idx]
        images = self._images[idxs]                                    # (N, 1, 28, 28)
        labels = self._labels[idxs]                                    # (N,)
        gt = (labels == self.positive_digit).long()                    # (N,)
        bag_count = int(gt.sum().item())
        return images, bag_count, gt
```

- [ ] **Step 3: Run MNIST tests**

```bash
uv run pytest tests/test_data.py -v -k mnist
```

Expected: First run downloads MNIST (~10MB, ~30s), then 2 PASSED.

- [ ] **Step 4: Commit**

```bash
git add pca/data.py tests/test_data.py
git commit -m "feat(data): MNISTBagDataset with bag/gt invariants + determinism tests"
```

---

## Task 11: Data — `LLPBagDataset` for Adult

**Files:**
- Modify: `pca/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Add LLP test stubs**

Append to `tests/test_data.py`:

```python
from pca.data import LLPBagDataset


def test_llp_adult_loads_and_caches():
    """First load fetches+caches; second load is fast (uses cache)."""
    import time
    t0 = time.time()
    ds = LLPBagDataset('adult', bag_size=8, num_bags=10, train=True, seed=0)
    t1 = time.time()
    # Second instantiation should hit cache (under 5s)
    ds2 = LLPBagDataset('adult', bag_size=8, num_bags=10, train=True, seed=0)
    t2 = time.time()
    assert t2 - t1 < 5.0, f"second load took {t2 - t1:.2f}s — cache not used?"


def test_llp_adult_feature_dim():
    ds = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=0)
    feats, bag_count, gt = ds[0]
    assert feats.shape[0] == 8
    assert feats.shape[1] >= 100  # Adult D ≈ 108 after one-hot
    assert gt.shape == (8,)
    assert bag_count == int(gt.sum().item())


def test_llp_seed_determinism():
    ds1 = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=7)
    ds2 = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=7)
    for i in range(5):
        f1, c1, g1 = ds1[i]
        f2, c2, g2 = ds2[i]
        assert torch.equal(f1, f2)
        assert c1 == c2
```

- [ ] **Step 2: Add `LLPBagDataset` to `pca/data.py`**

Append to `pca/data.py`:

```python
import pandas as pd
from sklearn.datasets import fetch_openml

_LLP_CACHE_DIR = Path.home() / ".cache" / "pca" / "llp"


def _build_adult_cache(cache_path: Path) -> dict:
    """Fetch Adult, one-hot encode, standardize, split. Cache as torch dict."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_openml('adult', version=2, as_frame=True)
    X_df = raw.data.copy()
    y_raw = raw.target

    # Drop rows with '?' markers in any column
    mask = ~(X_df.astype(str) == '?').any(axis=1)
    X_df = X_df[mask]
    y_raw = y_raw[mask]

    y = (y_raw == '>50K').astype(int).to_numpy()
    cat_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    X_cat = pd.get_dummies(X_df[cat_cols], dummy_na=False).astype(float)
    X_num = X_df[num_cols].astype(float)
    X_full = pd.concat([X_num, X_cat], axis=1).to_numpy().astype('float32')

    # Train/test split: stratified 80/20, fixed seed
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.2, random_state=0, stratify=y,
    )

    # Standardize numerical columns based on train statistics
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    blob = {
        'X_train': torch.from_numpy(X_tr).float(),
        'y_train': torch.from_numpy(y_tr).long(),
        'X_test': torch.from_numpy(X_te).float(),
        'y_test': torch.from_numpy(y_te).long(),
    }
    torch.save(blob, cache_path)
    return blob


class LLPBagDataset(Dataset):
    """LLP bag dataset for tabular UCI datasets (Adult).

    Returns per item: (features, bag_count, gt_labels)
        features:  (N, D) float
        bag_count: int
        gt_labels: (N,) long
    """

    def __init__(
        self,
        dataset_name: str,
        bag_size: int,
        num_bags: int,
        train: bool = True,
        seed: int = 0,
    ):
        assert dataset_name == 'adult', f"only 'adult' supported (got {dataset_name})"
        self.bag_size = bag_size
        self.num_bags = num_bags

        cache_path = _LLP_CACHE_DIR / f"{dataset_name}.pt"
        if cache_path.exists():
            blob = torch.load(cache_path, weights_only=True)
        else:
            blob = _build_adult_cache(cache_path)

        if train:
            self._features = blob['X_train']
            self._labels = blob['y_train']
        else:
            self._features = blob['X_test']
            self._labels = blob['y_test']

        rng = np.random.default_rng(seed)
        M = self._features.shape[0]
        self._bag_indices = np.stack([
            rng.choice(M, size=bag_size, replace=False) for _ in range(num_bags)
        ])

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        idxs = self._bag_indices[idx]
        feats = self._features[idxs]
        gt = self._labels[idxs]
        bag_count = int(gt.sum().item())
        return feats, bag_count, gt
```

- [ ] **Step 3: Run LLP tests**

```bash
uv run pytest tests/test_data.py -v -k llp
```

Expected: First run fetches Adult (~30s), then 3 PASSED. Second run < 10s total.

- [ ] **Step 4: Commit**

```bash
git add pca/data.py tests/test_data.py
git commit -m "feat(data): LLPBagDataset for Adult with cached preprocessing"
```

---

## Task 12: Models — `SmallCNN` + `MLP`

**Files:**
- Create: `pca/models.py`

- [ ] **Step 1: Create `pca/models.py`**

```python
"""Per-instance backbones for B1 verification.

SmallCNN: MNIST → single logit.
MLP:      tabular features → single logit.

See spec §7.
"""
from torch import nn, Tensor


class SmallCNN(nn.Module):
    """Per-instance MNIST → single logit. ~206k params, Ilse 2018 / Shukla scale."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28→14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B*N, 1, 28, 28)
        return self.classifier(self.features(x))   # (B*N, 1)


class MLP(nn.Module):
    """Per-instance tabular feature vector → single logit."""

    def __init__(self, in_dim: int, hidden=(64, 32), dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Dropout(dropout), nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # x: (B*N, D)
        return self.net(x)
```

- [ ] **Step 2: Quick smoke check (no dedicated test file — just inline)**

Run:
```bash
uv run python -c "
import torch
from pca.models import SmallCNN, MLP
m1 = SmallCNN()
m2 = MLP(in_dim=108)
print('SmallCNN out:', m1(torch.randn(8, 1, 28, 28)).shape)
print('MLP out:    ', m2(torch.randn(8, 108)).shape)
print('SmallCNN params:', sum(p.numel() for p in m1.parameters()))
print('MLP params:    ', sum(p.numel() for p in m2.parameters()))
"
```

Expected: both output shapes `torch.Size([8, 1])`. SmallCNN ~205k params (close to spec target).

- [ ] **Step 3: Commit**

```bash
git add pca/models.py
git commit -m "feat(models): SmallCNN (MNIST) + MLP (LLP) backbones"
```

---

## Task 13: `pca/train.py` — helpers (`per_bag_forward`, `select_device`)

**Files:**
- Create: `pca/train.py`

- [ ] **Step 1: Create `pca/train.py` with helpers**

```python
"""Training utilities for B1 verification.

Functions:
  select_device()       — single-source device selection (MPS/CUDA/CPU).
  per_bag_forward()     — bag-level reshape wrapper around any backbone.
  train()               — train loop returning epoch-level metrics dict.
  run_seeds()           — multi-seed sweep, persists per-seed JSON.

See spec §8.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from pca.losses import forward_conv


def select_device() -> torch.device:
    """Mac MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def per_bag_forward(model: nn.Module, batch: Tensor) -> Tensor:
    """Reshape (B, N, *features) → backbone → (B, N) logits.

    Args:
        model: backbone returning (B*N, 1) logits.
        batch: (B, N, *features) — (1, 28, 28) for MNIST or (D,) for LLP.
    """
    B, N = batch.shape[:2]
    flat = batch.reshape(B * N, *batch.shape[2:])
    logits_flat = model(flat).squeeze(-1)
    return logits_flat.view(B, N)
```

- [ ] **Step 2: Smoke check**

```bash
uv run python -c "
import torch
from pca.train import select_device, per_bag_forward
from pca.models import SmallCNN
print('device:', select_device())
m = SmallCNN()
x = torch.randn(4, 10, 1, 28, 28)
print('per_bag out:', per_bag_forward(m, x).shape)
"
```

Expected: device printed (mps/cuda/cpu), `per_bag out: torch.Size([4, 10])`.

- [ ] **Step 3: Commit**

```bash
git add pca/train.py
git commit -m "feat(train): select_device + per_bag_forward helpers"
```

---

## Task 14: `pca/train.py` — `train()` loop

**Files:**
- Modify: `pca/train.py`

- [ ] **Step 1: Add `train()` to `pca/train.py`**

Append to `pca/train.py`:

```python
import sklearn.metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """Train + per-epoch eval. Returns metrics dict per spec §12.1.

    Metric definitions:
      - epoch_train_loss: mean per-sample loss over the epoch.
      - epoch_inst_auc:   ROC-AUC of test per-instance probabilities vs ground-truth.
      - epoch_bag_acc:    test bag-level top-1 count accuracy via PMF argmax.
      - best_inst_auc:    mean of last 5 epoch_inst_auc values.
      - best_bag_acc:     mean of last 5 epoch_bag_acc values.
    """
    model = model.to(device)
    epoch_train_loss = []
    epoch_inst_auc = []
    epoch_bag_acc = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for images, bag_count, _gt in train_loader:
            images = images.to(device)
            bag_count = bag_count.to(device)
            logits = per_bag_forward(model, images)
            loss = loss_fn(logits, bag_count)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
            running_count += images.shape[0]
        epoch_train_loss.append(running_loss / max(running_count, 1))

        # Test eval
        model.eval()
        all_p, all_z = [], []
        bag_correct = 0
        bag_total = 0
        with torch.no_grad():
            for images, bag_count, gt in test_loader:
                images = images.to(device)
                logits = per_bag_forward(model, images)
                p = torch.sigmoid(logits)
                all_p.append(p.cpu().flatten())
                all_z.append(gt.flatten())
                log_P = forward_conv(p)
                pred = log_P.argmax(dim=1).cpu()
                bag_correct += (pred == bag_count).sum().item()
                bag_total += bag_count.shape[0]
        inst_auc = float(sklearn.metrics.roc_auc_score(
            torch.cat(all_z).numpy(), torch.cat(all_p).numpy()
        ))
        epoch_inst_auc.append(inst_auc)
        epoch_bag_acc.append(bag_correct / max(bag_total, 1))

    n_tail = min(5, num_epochs)
    return {
        'epoch_train_loss': epoch_train_loss,
        'epoch_inst_auc': epoch_inst_auc,
        'epoch_bag_acc': epoch_bag_acc,
        'best_inst_auc': float(np.mean(epoch_inst_auc[-n_tail:])),
        'best_bag_acc': float(np.mean(epoch_bag_acc[-n_tail:])),
    }
```

- [ ] **Step 2: Smoke check (1-epoch tiny run)**

```bash
uv run python -c "
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pca.data import MNISTBagDataset
from pca.models import SmallCNN
from pca.losses import marginal_nll_loss
from pca.train import train, select_device

device = select_device()
ds_tr = MNISTBagDataset(bag_size=10, num_bags=20, train=True, seed=0)
ds_te = MNISTBagDataset(bag_size=10, num_bags=10, train=False, seed=0)
m = SmallCNN()
opt = Adam(m.parameters(), lr=1e-3)
metrics = train(m,
    DataLoader(ds_tr, batch_size=8),
    DataLoader(ds_te, batch_size=8),
    marginal_nll_loss, opt, num_epochs=1, device=device)
print('inst_auc:', metrics['best_inst_auc'])
print('bag_acc :', metrics['best_bag_acc'])
"
```

Expected: completes in < 30s on MPS, prints two floats in (0, 1).

- [ ] **Step 3: Commit**

```bash
git add pca/train.py
git commit -m "feat(train): train() loop with per-epoch instance AUC + bag accuracy"
```

---

## Task 15: `pca/train.py` — `run_seeds()`

**Files:**
- Modify: `pca/train.py`

- [ ] **Step 1: Add `run_seeds()` to `pca/train.py`**

Append to `pca/train.py`:

```python
import random
import subprocess


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_seeds(
    seeds: list[int],
    build_fn: Callable[[int], tuple],
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    config_extra: dict | None = None,
) -> None:
    """Sequential per-seed training, persists results to save_dir/seed{i}.json.

    Args:
        build_fn(seed) -> (model, train_loader, test_loader, loss_fn, optimizer).
            Must produce identical data and model init for the same seed; method
            differs only via loss_fn for paired comparison validity (spec §8.3).
        config_extra: extra fields to merge into the persisted config dict.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, tr_loader, te_loader, loss_fn, optimizer = build_fn(seed)
        t0 = time.time()
        metrics = train(model, tr_loader, te_loader, loss_fn, optimizer, num_epochs, device)
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
        print(f"  [seed {seed}] inst_auc={metrics['best_inst_auc']:.4f} "
              f"bag_acc={metrics['best_bag_acc']:.4f} wall={wall:.1f}s -> {out_path}")
```

- [ ] **Step 2: Commit (smoke-tested by Task 16 sanity_n4)**

```bash
git add pca/train.py
git commit -m "feat(train): run_seeds with deterministic seeding + JSON persistence"
```

---

## Task 16: `scripts/sanity_n4.py` — Day 2 gate

**Files:**
- Create: `scripts/sanity_n4.py`

> **Editable-install shim required for `scripts/*.py`:** Python 3.13's `site.py` silently skips `.pth` files whose names start with `_`, and uv/hatchling's editable install creates `_editable_impl_pca.pth`. Without a workaround, `python scripts/foo.py` fails with `ModuleNotFoundError: No module named 'pca'`. Add this 3-line shim at the top of every `scripts/*.py` (before `from pca... import ...`):
> ```python
> import sys
> from pathlib import Path
> sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
> ```
> The same shim is needed for `scripts/run_mnist_mil.py` (Task 17), `scripts/run_llp_adult.py` (Task 21), and `scripts/analyze.py` (Task 19).

- [ ] **Step 1: Create `scripts/sanity_n4.py`**

```python
#!/usr/bin/env python
"""Day 2 gate: end-to-end sanity check before any experiment.

Checks (spec §9.1):
  1. forward_conv on hand-computed N=2 case
  2. leave_one_out_posterior on hand-computed N=2 case
  3. marginal_nll_loss == em_joint_loss(lam=0)         (1e-6)
  4. gradient cosine similarity ≥ 0.9999                (NLL vs EM lam=0)
  5. Σ_u q_u ≈ bag_y                                    (1e-3)
  6. lam=1 + one Adam step → no NaN/Inf, params changed
"""
from __future__ import annotations
import sys

import torch
from torch.optim import Adam

from pca.losses import (
    forward_conv, leave_one_out_posterior,
    marginal_nll_loss, em_joint_loss,
)


def fail(check: str, detail: str = "") -> None:
    print(f"[FAIL] {check}{(': ' + detail) if detail else ''}")
    sys.exit(1)


def main() -> None:
    torch.manual_seed(0)

    # 1. forward_conv N=2
    p = torch.tensor([[0.3, 0.4]])
    P = forward_conv(p).exp()
    if not torch.allclose(P, torch.tensor([[0.42, 0.46, 0.12]]), atol=1e-6):
        fail("forward_conv N=2", f"got {P.tolist()}")

    # 2. leave_one_out_posterior N=2
    q = leave_one_out_posterior(p, torch.tensor([1]))
    expected_q = torch.tensor([[0.391304, 0.608696]])
    if not torch.allclose(q, expected_q, atol=1e-4):
        fail("posterior N=2", f"got {q.tolist()}")

    # 3. lam=0 numerical equivalence
    logits = torch.randn(4, 4, requires_grad=True)
    bag_y = torch.tensor([0, 1, 2, 3])
    nll = marginal_nll_loss(logits, bag_y)
    em0 = em_joint_loss(logits, bag_y, lam=0.0)
    if not torch.allclose(nll, em0, atol=1e-6):
        fail("lam=0 == NLL", f"nll={nll.item():.7f} em0={em0.item():.7f}")

    # 4. gradient cosine
    g_nll, = torch.autograd.grad(marginal_nll_loss(logits, bag_y), logits, retain_graph=False)
    g_em0, = torch.autograd.grad(em_joint_loss(logits, bag_y, lam=0.0), logits, retain_graph=False)
    cos = (g_nll * g_em0).sum() / (g_nll.norm() * g_em0.norm() + 1e-12)
    if cos.item() < 0.9999:
        fail("grad cosine", f"cos={cos.item():.6f}")

    # 5. posterior sum invariant
    p2 = torch.rand(8, 16)
    y2 = torch.randint(0, 17, (8,))
    q2 = leave_one_out_posterior(p2, y2)
    if not torch.allclose(q2.sum(dim=1), y2.float(), atol=1e-3):
        fail("posterior sum", f"got {q2.sum(dim=1).tolist()} vs {y2.tolist()}")

    # 6. one optimizer step
    p_init = torch.nn.Parameter(torch.randn(2, 4))
    opt = Adam([p_init], lr=1e-2)
    loss = em_joint_loss(p_init, torch.tensor([2, 1]), lam=1.0)
    if not torch.isfinite(loss):
        fail("lam=1 finite", f"loss={loss.item()}")
    p_before = p_init.detach().clone()
    opt.zero_grad()
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in [p_init]):
        fail("lam=1 grad finite")
    opt.step()
    if torch.allclose(p_before, p_init.detach()):
        fail("optimizer step", "params unchanged")

    print("[OK] All sanity checks passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run sanity check**

```bash
uv run python scripts/sanity_n4.py
```

Expected: `[OK] All sanity checks passed.` (under 5 seconds).

- [ ] **Step 3: Commit + tag Day 2**

```bash
git add scripts/sanity_n4.py
git commit -m "feat(scripts): sanity_n4 — Day 2 gate"
git tag -a day2-sanity-passed -m "Day 2 gate: end-to-end sanity passing"
```

**End-of-Day-2 deliverable**: sanity script output `[OK] All sanity checks passed.`

---

## Task 17: `scripts/run_mnist_mil.py` + smoke test

**Files:**
- Create: `scripts/run_mnist_mil.py`

- [ ] **Step 1: Create `scripts/run_mnist_mil.py`**

```python
#!/usr/bin/env python
"""MNIST-MIL experiment runner.

Per spec §9.2: --bag-size {10,50,100} --method {nll,em} --epochs 100 \\
               --seeds 0,1,2,3,4 --output-dir results/mnist_mil/

Smoke usage:
  python scripts/run_mnist_mil.py --bag-size 10 --method nll \\
    --epochs 2 --num-bags 50 --seeds 0
"""
from __future__ import annotations
import argparse
from functools import partial
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.data import MNISTBagDataset
from pca.losses import em_joint_loss, marginal_nll_loss
from pca.models import SmallCNN
from pca.train import run_seeds, select_device

# Default training pool sizes per spec §8.4.
DEFAULT_TRAIN_BAGS = 1000
DEFAULT_TEST_BAGS = 500
DEFAULT_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bag-size', type=int, required=True, choices=[10, 50, 100])
    p.add_argument('--method', choices=['nll', 'em'], required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS,
                   help='Training-pool bag count (test pool fixed at half).')
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', type=Path, default=Path('results/mnist_mil'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, bag_size={args.bag_size}, method={args.method}, "
          f"seeds={args.seeds}, epochs={args.epochs}")

    batch_size = 32 if args.bag_size <= 50 else 16

    def build_fn(seed: int):
        train_ds = MNISTBagDataset(bag_size=args.bag_size, num_bags=args.num_bags,
                                   positive_digit=9, train=True, seed=seed)
        # Test pool: half training count, fixed across methods (paired comparison).
        test_ds = MNISTBagDataset(bag_size=args.bag_size, num_bags=DEFAULT_TEST_BAGS,
                                  positive_digit=9, train=False, seed=seed)
        model = SmallCNN()
        opt = Adam(model.parameters(), lr=args.lr)
        loss_fn = (marginal_nll_loss if args.method == 'nll'
                   else partial(em_joint_loss, lam=args.lam))
        return (
            model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            loss_fn, opt,
        )

    save_dir = args.output_dir / f"N{args.bag_size}_{args.method}"
    config_extra = {
        'experiment': 'mnist_mil', 'bag_size': args.bag_size, 'method': args.method,
        'lam': args.lam if args.method == 'em' else None,
        'lr': args.lr, 'batch_size': batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': DEFAULT_TEST_BAGS,
    }
    run_seeds(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (Day 3 gate)**

```bash
uv run python scripts/run_mnist_mil.py \
    --bag-size 10 --method nll --epochs 2 --num-bags 50 --seeds 0
```

Expected: completes in under 5 minutes on MPS, creates `results/mnist_mil/N10_nll/seed0.json`.

- [ ] **Step 3: Verify smoke JSON schema**

```bash
uv run python -c "
import json
d = json.loads(open('results/mnist_mil/N10_nll/seed0.json').read())
assert 'config' in d and 'best_inst_auc' in d
assert 0.0 < d['best_inst_auc'] < 1.0
print('schema ok, inst_auc =', d['best_inst_auc'])
"
```

Expected: prints schema ok with inst_auc in (0, 1).

- [ ] **Step 4: Commit + tag Day 3**

```bash
git add scripts/run_mnist_mil.py
git commit -m "feat(scripts): run_mnist_mil with smoke-tested pipeline"
git tag -a day3-smoke-passed -m "Day 3 gate: MNIST-MIL smoke run produces valid JSON"
```

---

## Task 18: Day 3–4 — full MNIST-MIL sweep

**Files:** none modified (execution only).

- [ ] **Step 1: Run all 30 MNIST-MIL experiments sequentially**

Run each combination (5 seeds × 3 sizes × 2 methods = 30 runs). Single shell loop:

```bash
for N in 10 50 100; do
  for M in nll em; do
    uv run python scripts/run_mnist_mil.py --bag-size $N --method $M --epochs 100
  done
done
```

Expected total wall-clock: ~4 hours on MPS (per spec §16.3 estimate).

- [ ] **Step 2: Verify all 30 result JSONs present**

```bash
ls -1 results/mnist_mil/*/seed*.json | wc -l
```

Expected: 30.

- [ ] **Step 3: Spot-check a couple of JSONs**

```bash
for d in results/mnist_mil/N50_nll results/mnist_mil/N50_em; do
  echo "== $d =="
  uv run python -c "
import json, glob
for f in sorted(glob.glob('$d/seed*.json')):
    d = json.loads(open(f).read())
    print(f, 'inst_auc=', round(d['best_inst_auc'], 4))
"
done
```

Expected: 5 lines per directory, AUC values look reasonable (>0.7).

- [ ] **Step 4: Commit any state shifts (if results/summary.md was touched)**

(No commit needed in this task — raw seed JSONs are gitignored.)

**End-of-Day-4 deliverable**: 30 valid seed JSONs in `results/mnist_mil/`.

---

## Task 19: `scripts/analyze.py` — aggregation + verdict

**Files:**
- Create: `scripts/analyze.py`

- [ ] **Step 1: Create `scripts/analyze.py`**

```python
#!/usr/bin/env python
"""Aggregate per-seed JSONs → markdown summary + H1/H2/H3 verdict.

Per spec §11. Usage:
  python scripts/analyze.py --results-dir results/mnist_mil/
  python scripts/analyze.py --results-dir results/llp_adult/
"""
from __future__ import annotations
import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats


@dataclass
class CellStat:
    mean: float
    std: float
    seeds: list[float] = field(default_factory=list)


def aggregate(results_dir: Path) -> dict[int, dict[str, CellStat]]:
    """Returns summary[bag_size][method] = CellStat over best_inst_auc."""
    summary: dict[int, dict[str, CellStat]] = {}
    for setting_dir in sorted(results_dir.glob('N*_*')):
        name = setting_dir.name           # e.g., N50_em
        N_str, method = name.split('_', 1)
        N = int(N_str[1:])
        seed_jsons = sorted(setting_dir.glob('seed*.json'))
        aucs = [json.loads(f.read_text())['best_inst_auc'] for f in seed_jsons]
        cell = CellStat(mean=float(np.mean(aucs)), std=float(np.std(aucs, ddof=1)),
                        seeds=aucs)
        summary.setdefault(N, {})[method] = cell
    return summary


def evaluate_h1(summary, target_N: int) -> dict:
    """H1: Δ at target_N ≥ 1.0 pp, paired t-test p < 0.05."""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    delta_pp = (em.mean - nl.mean) * 100
    diffs = np.array(em.seeds) - np.array(nl.seeds)
    _, p_value = scipy.stats.ttest_rel(em.seeds, nl.seeds)
    return {'pass': delta_pp >= 1.0 and p_value < 0.05,
            'delta_pp': delta_pp, 'p_value': float(p_value),
            'paired_diffs_pp': (diffs * 100).tolist()}


def evaluate_h2(summary, sizes: list[int]) -> dict:
    """H2: Δ monotone non-decreasing across bag sizes (0.2 pp slack)."""
    deltas_pp = [(summary[N]['em'].mean - summary[N]['nll'].mean) * 100 for N in sizes]
    monotone = all(deltas_pp[i] <= deltas_pp[i + 1] + 0.2 for i in range(len(deltas_pp) - 1))
    return {'pass': monotone, 'deltas_pp': deltas_pp}


def evaluate_h3(summary, target_N: int) -> dict:
    """H3: std(EM) ≤ 0.7 × std(NLL) at target_N."""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    ratio = em.std / nl.std if nl.std > 0 else float('inf')
    return {'pass': ratio <= 0.7, 'std_ratio': ratio,
            'em_std': em.std, 'nll_std': nl.std}


def verdict(h1, h2, h3) -> str:
    delta = h1['delta_pp']
    if delta < 0.5:
        return 'DEMOTE_B1'
    if delta < 1.0:
        return 'F2_ABLATION_NEEDED'
    if not h1['pass']:
        return 'F2_ABLATION_NEEDED'
    if h2['pass'] or h3['pass']:
        return 'PROCEED_TO_LLP'
    return 'PROCEED_BUT_WEAK'


def render_markdown(summary, h1, h2, h3, target_N, sizes, exp_name) -> str:
    lines = [f"# Results — {exp_name}", "", "## Instance-level AUC (mean ± std over 5 seeds)", "",
             "| N   | NLL              | EM               | Δ (pp)        |",
             "|-----|------------------|------------------|---------------|"]
    for N in sizes:
        nl = summary[N]['nll']; em = summary[N]['em']
        delta = (em.mean - nl.mean) * 100
        mark = '✓' if delta >= 1.0 else ('~' if delta >= 0.5 else '✗')
        lines.append(f"| {N:<3} | {nl.mean:.4f} ± {nl.std:.4f}  "
                     f"| {em.mean:.4f} ± {em.std:.4f}  | {delta:+.2f} {mark}      |")
    v = verdict(h1, h2, h3)
    lines += ["", "## Hypothesis verdicts", "",
              f"- H1 (Δ @ N={target_N} ≥ 1pp, p<0.05): "
              f"**{'PASS' if h1['pass'] else 'FAIL'}**  Δ = {h1['delta_pp']:+.2f} pp, "
              f"p = {h1['p_value']:.3f}",
              f"- H2 (monotone in N): "
              f"**{'PASS' if h2['pass'] else 'FAIL'}**  deltas_pp = "
              f"[{', '.join(f'{d:+.2f}' for d in h2['deltas_pp'])}]",
              f"- H3 (std reduction ≥ 30%): "
              f"**{'PASS' if h3['pass'] else 'FAIL'}**  ratio = {h3['std_ratio']:.2f}",
              "", f"## Verdict: {v}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, required=True)
    args = p.parse_args()

    summary = aggregate(args.results_dir)
    sizes = sorted(summary.keys())
    target_N = 50 if 50 in sizes else (128 if 128 in sizes else sizes[len(sizes) // 2])
    h1 = evaluate_h1(summary, target_N)
    h2 = evaluate_h2(summary, sizes)
    h3 = evaluate_h3(summary, target_N)
    md = render_markdown(summary, h1, h2, h3, target_N, sizes, args.results_dir.name)
    print(md)
    (args.results_dir / 'summary.md').write_text(md)
    (args.results_dir / 'summary.json').write_text(json.dumps({
        'h1': h1, 'h2': h2, 'h3': h3, 'verdict': verdict(h1, h2, h3),
        'summary': {N: {m: {'mean': c.mean, 'std': c.std, 'seeds': c.seeds}
                        for m, c in d.items()} for N, d in summary.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on synthetic stub (verify the script runs end-to-end)**

```bash
mkdir -p /tmp/fake_results/{N10_nll,N10_em,N50_nll,N50_em,N100_nll,N100_em}
for d in /tmp/fake_results/*/; do
  for s in 0 1 2 3 4; do
    uv run python -c "
import json
d = '$d'
import random
random.seed(int('$s') + (10 if 'em' in d else 0))
import json
val = 0.85 + random.random() * 0.05
json.dump({'best_inst_auc': val, 'config': {}, 'epoch_train_loss': []}, open('${d}seed${s}.json', 'w'))
"
  done
done
uv run python scripts/analyze.py --results-dir /tmp/fake_results/
rm -rf /tmp/fake_results/
```

Expected: prints markdown table + verdict; both `summary.md` and `summary.json` written before cleanup.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze.py
git commit -m "feat(scripts): analyze.py with H1/H2/H3 paired-t-test verdict"
```

---

## Task 20: Day 5 gate — run analyze on real MNIST-MIL results

- [ ] **Step 1: Run analyze on actual results**

```bash
uv run python scripts/analyze.py --results-dir results/mnist_mil/
```

Expected: prints markdown table with NLL vs EM at N ∈ {10, 50, 100}, ends with one of `PROCEED_TO_LLP / PROCEED_BUT_WEAK / F2_ABLATION_NEEDED / DEMOTE_B1`.

- [ ] **Step 2: Inspect summary.json**

```bash
uv run python -c "
import json
s = json.loads(open('results/mnist_mil/summary.json').read())
print('verdict:', s['verdict'])
print('H1 delta:', round(s['h1']['delta_pp'], 2), 'p:', round(s['h1']['p_value'], 3))
print('H2 deltas:', [round(x, 2) for x in s['h2']['deltas_pp']])
print('H3 ratio:', round(s['h3']['std_ratio'], 2))
"
```

Expected: structured verdict info matches stdout markdown.

- [ ] **Step 3: Commit summary artifacts**

```bash
git add results/mnist_mil/summary.md results/mnist_mil/summary.json
git commit -m "docs(results): MNIST-MIL summary + Day 5 verdict"
git tag -a day5-mnist-verdict -m "Day 5 gate: MNIST-MIL verdict captured"
```

- [ ] **Step 4: Branch on verdict (manual decision)**

| Verdict | Next step |
|---|---|
| `PROCEED_TO_LLP` | Continue to Task 21 (LLP runs). |
| `PROCEED_BUT_WEAK` | Continue to Task 21 with note in `02_training_strengthening.md` reframing B1 as a boost-axis. |
| `F2_ABLATION_NEEDED` | Add entropy reg in M-step (modify `em_joint_loss` per spec §11.4 with `lam_ent=0.01`); rerun MNIST-MIL N=50 only; re-evaluate H1. If still <1pp → DEMOTE. |
| `DEMOTE_B1` | Update `README.md` priority table (B1 → Low). Stop here; Track 1 step 3 (B2) starts in a separate plan. |

Pick the path that matches the verdict before proceeding to Task 21.

---

## Task 21: `scripts/run_llp_adult.py` + smoke test

**Files:**
- Create: `scripts/run_llp_adult.py`

- [ ] **Step 1: Create `scripts/run_llp_adult.py`**

```python
#!/usr/bin/env python
"""LLP Adult experiment runner.

Per spec §9.3: --bag-size {32,128,512} --method {nll,em} --epochs 200 \\
               --seeds 0,1,2,3,4 --output-dir results/llp_adult/
"""
from __future__ import annotations
import argparse
from functools import partial
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.data import LLPBagDataset
from pca.losses import em_joint_loss, marginal_nll_loss
from pca.models import MLP
from pca.train import run_seeds, select_device


DEFAULT_TRAIN_BAGS = 2000
DEFAULT_TEST_BAGS = 1000
DEFAULT_EPOCHS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bag-size', type=int, required=True, choices=[32, 128, 512])
    p.add_argument('--method', choices=['nll', 'em'], required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', type=Path, default=Path('results/llp_adult'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, bag_size={args.bag_size}, method={args.method}, "
          f"seeds={args.seeds}, epochs={args.epochs}")

    if args.bag_size <= 128:
        batch_size = 32
    else:
        batch_size = 8

    # Inspect feature dim once to pin MLP in_dim.
    probe = LLPBagDataset('adult', bag_size=args.bag_size, num_bags=1, train=True, seed=0)
    feats, _, _ = probe[0]
    in_dim = feats.shape[1]

    def build_fn(seed: int):
        train_ds = LLPBagDataset('adult', bag_size=args.bag_size,
                                 num_bags=args.num_bags, train=True, seed=seed)
        test_ds = LLPBagDataset('adult', bag_size=args.bag_size,
                                num_bags=DEFAULT_TEST_BAGS, train=False, seed=seed)
        model = MLP(in_dim=in_dim)
        opt = Adam(model.parameters(), lr=args.lr)
        loss_fn = (marginal_nll_loss if args.method == 'nll'
                   else partial(em_joint_loss, lam=args.lam))
        return (
            model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            loss_fn, opt,
        )

    save_dir = args.output_dir / f"N{args.bag_size}_{args.method}"
    config_extra = {
        'experiment': 'llp_adult', 'bag_size': args.bag_size, 'method': args.method,
        'lam': args.lam if args.method == 'em' else None,
        'lr': args.lr, 'batch_size': batch_size, 'in_dim': in_dim,
        'num_train_bags': args.num_bags, 'num_test_bags': DEFAULT_TEST_BAGS,
    }
    run_seeds(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

```bash
uv run python scripts/run_llp_adult.py \
    --bag-size 32 --method nll --epochs 5 --num-bags 100 --seeds 0
```

Expected: completes in under 5 minutes, creates `results/llp_adult/N32_nll/seed0.json`.

- [ ] **Step 3: Commit + tag Day 6 entry**

```bash
git add scripts/run_llp_adult.py
git commit -m "feat(scripts): run_llp_adult"
git tag -a day6-llp-smoke -m "Day 6 gate: LLP Adult smoke run produces valid JSON"
```

---

## Task 22: Day 6–7 — full LLP Adult sweep

**Files:** none modified.

- [ ] **Step 1: Run all 30 LLP Adult experiments**

```bash
for N in 32 128 512; do
  for M in nll em; do
    uv run python scripts/run_llp_adult.py --bag-size $N --method $M --epochs 200
  done
done
```

Expected wall-clock: ~2–3 hours (per spec §16.3).

- [ ] **Step 2: Verify all 30 JSONs present**

```bash
ls -1 results/llp_adult/*/seed*.json | wc -l
```

Expected: 30.

---

## Task 23: Day 7 final analyze + week-1 wrap-up

**Files:**
- Modify: `README.md` (priority table or status note depending on verdict)

- [ ] **Step 1: Run analyze on LLP results**

```bash
uv run python scripts/analyze.py --results-dir results/llp_adult/
```

Expected: markdown table at N ∈ {32, 128, 512}, target_N = 128 picked automatically; verdict printed.

- [ ] **Step 2: Commit LLP summary**

```bash
git add results/llp_adult/summary.md results/llp_adult/summary.json
git commit -m "docs(results): LLP Adult summary + Day 7 verdict"
```

- [ ] **Step 3: Update `README.md` with the week-1 outcome**

Edit the §3 "결정 사항" section to add a new bullet capturing the B1 outcome:

```markdown
- **B1 week-1 verdict (2026-05-XX)**: <one of>
  - "B1 verified — H1/H2/H3 results inserted from `results/mnist_mil/summary.md` and `results/llp_adult/summary.md`. Track 1 step 3 (B2) starts next."
  - "B1 demoted to Low — H1 < 0.5 pp on MNIST-MIL. Priority table updated; Track 1 step 3 (B2) starts immediately."
  - "B1 partial — passes MNIST-MIL but fails LLP (or vice versa). Reframed as bag-size-conditional contribution; B2 verification proceeds."
```

- [ ] **Step 4: Commit README update + tag week-1 close**

```bash
git add README.md
git commit -m "docs(readme): record B1 week-1 verdict"
git tag -a week1-b1-complete -m "Week 1: B1 verification complete"
```

**End-of-week-1 deliverable**:
- `pca/` package with 4 modules.
- 4 scripts run end-to-end.
- 60 seed JSONs (30 MNIST-MIL + 30 LLP Adult) plus summaries.
- `README.md` reflects B1 outcome.
- Plan + spec committed to repo for the next track to reference.

---

## Self-review checklist

- [ ] **Spec coverage:** every spec section maps to a task.
  - §1 motivation → Task 0 (context only, no code).
  - §2 hypotheses → Task 19 (analyze.py implements H1/H2/H3).
  - §3 approach choice → Task 7 (em_joint_loss with lam=0 equivalence).
  - §4 architecture → Task 1 + module layout pinned in plan header.
  - §5 losses → Tasks 2–8.
  - §6 data → Tasks 10–11.
  - §7 models → Task 12.
  - §8 train → Tasks 13–15.
  - §9 scripts → Tasks 16, 17, 19, 21.
  - §10 testing → Tests in Tasks 2–11; gate runs in Tasks 9, 16, 17.
  - §11 hypothesis evaluation → Task 19.
  - §12 result storage → Task 15 (run_seeds JSON) + .gitignore in Task 1.
  - §13 day-by-day → mapping below.
  - §14 decision points → Tasks 9 (Day 1), 16 (Day 2), 17 (Day 3), 18 (Day 4), 20 (Day 5), 21 (Day 6), 23 (Day 7).
  - §15 non-goals → not implemented (correct — they're out of scope).
  - §16 dependencies → Task 1.

- [ ] **Day-to-task mapping**:
  - Day 1: Tasks 1–9. Gate: 10 unit tests pass.
  - Day 2: Tasks 10–16. Gate: sanity_n4 passes.
  - Day 3: Task 17. Gate: smoke MNIST-MIL JSON valid.
  - Day 4: Task 18. Gate: 30 JSONs present.
  - Day 5: Tasks 19–20. Gate: verdict emitted.
  - Day 6: Task 21. Gate: smoke LLP JSON valid.
  - Day 7: Tasks 22–23. Gate: final verdict + README update.

- [ ] **Placeholder scan**: no TBD/TODO inside steps; all code blocks are complete.

- [ ] **Type consistency**:
  - `forward_conv(p) → log_P` everywhere.
  - `leave_one_out_posterior(p, bag_y) → q` (detached) everywhere.
  - `marginal_nll_loss(logits, bag_y)` and `em_joint_loss(logits, bag_y, lam=...)` with consistent signatures.
  - `train(model, train_loader, test_loader, loss_fn, optimizer, num_epochs, device) → dict` consistent in train.py and scripts.
  - `run_seeds(seeds, build_fn, num_epochs, device, save_dir, config_extra)` arg order consistent.
  - JSON schema (per spec §12.1) is what `run_seeds` writes and what `analyze.aggregate` reads.

- [ ] **Test gates have explicit pass criteria** (return code, output text, file presence) — not "looks reasonable".

---

## References

- **Spec**: `docs/superpowers/specs/2026-05-03-b1-em-verification-design.md` (commit `46897e1`).
- **Project notes**: `README.md`, `02_training_strengthening.md` §B1, `03_evaluation_strategy.md` §4 Track 1.
- **Baseline paper**: `sources/Shukla2023/camera_ready.tex`.
