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
    # Note: features is recreated per baseline to avoid stale graph from
    # previous backward() calls accumulating on a shared leaf tensor.
    feature_dim = 128
    K = 9
    N_max = 15
    B = 4
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
        features = torch.randn(B, N_max, feature_dim, requires_grad=True)
        mask = torch.zeros(B, N_max, dtype=torch.bool)
        mask[0, :5] = True; mask[1, :10] = True; mask[2, :15] = True; mask[3, :8] = True
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
