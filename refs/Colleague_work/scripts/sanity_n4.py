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
from pathlib import Path

# Make `pca` importable when this file is run directly via `python scripts/sanity_n4.py`.
# (Python 3.13's site.py skips `_`-prefixed editable .pth files, so the editable install
# created by uv/hatchling doesn't register the project root on sys.path.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
