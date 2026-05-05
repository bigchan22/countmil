"""Posterior marginals under aggregate constraints."""

from __future__ import annotations

import torch


def binary_count_posterior(probs: torch.Tensor, counts: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute q_i = P(z_i=1 | sum_j z_j = Y) for Bernoulli atoms.

    Args:
        probs: `(B, N)` Bernoulli probabilities.
        counts: `(B,)` observed integer counts.

    Returns:
        Tensor `(B, N)` of posterior marginals.
    """

    if probs.ndim != 2:
        raise ValueError("probs must have shape (B, N)")
    batch, n_atoms = probs.shape
    device = probs.device
    counts = counts.to(device=device, dtype=torch.long)

    prefix = probs.new_zeros(batch, n_atoms + 1, n_atoms + 1)
    suffix = probs.new_zeros(batch, n_atoms + 1, n_atoms + 1)
    prefix[:, 0, 0] = 1.0
    suffix[:, n_atoms, 0] = 1.0

    for i in range(n_atoms):
        p = probs[:, i : i + 1]
        prefix[:, i + 1, : i + 1] += prefix[:, i, : i + 1] * (1.0 - p)
        prefix[:, i + 1, 1 : i + 2] += prefix[:, i, : i + 1] * p

    for i in range(n_atoms - 1, -1, -1):
        p = probs[:, i : i + 1]
        suffix[:, i, : n_atoms - i] += suffix[:, i + 1, : n_atoms - i] * (1.0 - p)
        suffix[:, i, 1 : n_atoms - i + 1] += suffix[:, i + 1, : n_atoms - i] * p

    y_idx = counts.clamp(0, n_atoms)
    denom = prefix[torch.arange(batch, device=device), n_atoms, y_idx].clamp_min(eps)
    out = probs.new_zeros(batch, n_atoms)

    k_grid = torch.arange(n_atoms + 1, device=device)
    for i in range(n_atoms):
        needed = counts.view(batch, 1) - 1 - k_grid.view(1, -1)
        valid = (needed >= 0) & (needed <= (n_atoms - i - 1))
        needed_safe = needed.clamp(0, n_atoms).long()
        suffix_vals = torch.gather(suffix[:, i + 1], 1, needed_safe)
        terms = prefix[:, i] * suffix_vals * valid.float()
        out[:, i] = probs[:, i] * terms.sum(dim=1) / denom

    impossible = (counts < 0) | (counts > n_atoms)
    out[impossible] = 0.0
    return out

