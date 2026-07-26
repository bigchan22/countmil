"""Synthetic binary-count data with hidden bag-level dependence."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DependenceData:
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    intercept: float
    tau: float
    seed: int

    @property
    def prevalence(self) -> float:
        return float(self.z.float().mean().item())

    @property
    def within_bag_correlation(self) -> float:
        return within_bag_label_correlation(self.z)


def within_bag_label_correlation(z: torch.Tensor) -> float:
    """Mean off-diagonal correlation between fixed within-bag positions."""

    if z.ndim != 2:
        raise ValueError("z must have shape (bags, bag_size)")
    work = z.float()
    if work.shape[1] < 2:
        return 0.0
    centered = work - work.mean(dim=0, keepdim=True)
    std = centered.square().mean(dim=0).sqrt()
    valid = std > 1e-8
    if int(valid.sum().item()) < 2:
        return 0.0
    c = centered[:, valid] / std[valid].clamp_min(1e-8)
    corr = (c.T @ c) / c.shape[0]
    n = corr.shape[0]
    offdiag = corr[~torch.eye(n, dtype=torch.bool, device=corr.device)]
    return float(offdiag.mean().item())


def generate_dependence_data(
    *,
    bags: int,
    bag_size: int,
    dim: int,
    tau: float,
    seed: int,
    w: torch.Tensor | None = None,
    intercept: float = -0.5,
) -> DependenceData:
    """Generate fixed bags from the hidden-random-effect DGP."""

    if bags < 1 or bag_size < 1 or dim < 1:
        raise ValueError("bags, bag_size, and dim must be positive")
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if w is None:
        w_raw = torch.randn(dim, generator=gen)
        w = w_raw / w_raw.norm().clamp_min(1e-12)
    else:
        w = w.detach().float()
        if w.numel() != dim:
            raise ValueError("w dimension mismatch")
        w = w / w.norm().clamp_min(1e-12)
    x = torch.randn(bags, bag_size, dim, generator=gen)
    u = torch.randn(bags, generator=gen) * float(tau)
    logits = torch.einsum("bnd,d->bn", x, w) + float(intercept) + u[:, None]
    z = torch.bernoulli(torch.sigmoid(logits), generator=gen).long()
    y = z.sum(dim=1).long()
    data = DependenceData(x=x, z=z, y=y, u=u, w=w, intercept=float(intercept), tau=float(tau), seed=int(seed))
    if data.prevalence <= 0.05 or data.prevalence >= 0.95:
        raise ValueError(f"degenerate generated prevalence {data.prevalence:.4f}")
    return data
