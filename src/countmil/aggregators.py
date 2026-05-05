"""Finite-support aggregate likelihoods.

The functions here are intentionally independent of datasets and selectors. A
model should produce instance-level atomic PMFs; this module turns those atoms
into an exact PMF over the integer aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AggregatePMF:
    """PMF over a contiguous integer support.

    `probs[..., j]` corresponds to aggregate value `support_min + j`.
    """

    probs: torch.Tensor
    support_min: int

    @property
    def support_max(self) -> int:
        return self.support_min + self.probs.shape[-1] - 1

    def value_to_index(self, values: torch.Tensor) -> torch.Tensor:
        return values.to(device=self.probs.device, dtype=torch.long) - self.support_min


def _flatten_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("expected at least one batch/instance dimension")
    batch_shape = x.shape[:-2]
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    return flat, batch_shape


def finite_support_convolution(atom_pmfs: torch.Tensor, support_min: int = 0) -> AggregatePMF:
    """Convolve independent finite-support atomic PMFs.

    Args:
        atom_pmfs: Tensor with shape `(..., N, K)`. Each atomic PMF has common
            contiguous support `[support_min, support_min + K - 1]`.
        support_min: Minimum integer value for every atom.

    Returns:
        AggregatePMF with batch shape `...` and aggregate support
        `[N * support_min, N * (support_min + K - 1)]`.
    """

    if atom_pmfs.ndim < 2:
        raise ValueError("atom_pmfs must have shape (..., N, K)")
    if atom_pmfs.shape[-1] < 1:
        raise ValueError("atomic support must be non-empty")

    atoms, batch_shape = _flatten_batch(atom_pmfs)
    batch, n_atoms, width = atoms.shape
    probs = atoms.new_ones(batch, 1)

    for i in range(n_atoms):
        atom = atoms[:, i]
        next_width = probs.shape[-1] + width - 1
        out = atoms.new_zeros(batch, next_width)
        for offset in range(width):
            out[:, offset : offset + probs.shape[-1]] += probs * atom[:, offset : offset + 1]
        probs = out

    support = n_atoms * int(support_min)
    return AggregatePMF(probs.reshape(*batch_shape, probs.shape[-1]), support)


def binary_count_dp(probs: torch.Tensor) -> AggregatePMF:
    """Exact Bernoulli count PMF using the Shukla-style DP recurrence."""

    if probs.ndim < 1:
        raise ValueError("probs must have shape (..., N)")
    flat = probs.reshape(-1, probs.shape[-1])
    batch, n_atoms = flat.shape
    pmf = flat.new_zeros(batch, n_atoms + 1)
    pmf[:, 0] = 1.0

    for i in range(n_atoms):
        p = flat[:, i : i + 1]
        prev = pmf.clone()
        pmf[:, 0 : i + 2] = 0.0
        pmf[:, 0 : i + 1] += prev[:, 0 : i + 1] * (1.0 - p)
        pmf[:, 1 : i + 2] += prev[:, 0 : i + 1] * p

    return AggregatePMF(pmf.reshape(*probs.shape[:-1], n_atoms + 1), 0)


def brute_force_binary_count(probs: torch.Tensor) -> AggregatePMF:
    """Brute-force Bernoulli count PMF for small N, useful for tests."""

    if probs.ndim != 1:
        raise ValueError("brute_force_binary_count expects a 1D probability vector")
    n_atoms = probs.numel()
    if n_atoms > 20:
        raise ValueError("brute force is intended only for small N")

    out = probs.new_zeros(n_atoms + 1)
    for assignment in product((0, 1), repeat=n_atoms):
        k = sum(assignment)
        mass = probs.new_tensor(1.0)
        for i, z_i in enumerate(assignment):
            mass = mass * (probs[i] if z_i else (1.0 - probs[i]))
        out[k] += mass
    return AggregatePMF(out, 0)


def brute_force_finite_support(atom_pmfs: torch.Tensor, support_min: int = 0) -> AggregatePMF:
    """Brute-force finite-support aggregate PMF for small N.

    This is exponential in the number of atoms and should only be used in tests.
    """

    if atom_pmfs.ndim != 2:
        raise ValueError("brute_force_finite_support expects shape (N, K)")
    n_atoms, width = atom_pmfs.shape
    if n_atoms > 10:
        raise ValueError("brute force is intended only for small N")
    support_max = support_min + width - 1
    out_min = n_atoms * support_min
    out_max = n_atoms * support_max
    out = atom_pmfs.new_zeros(out_max - out_min + 1)

    values = list(range(support_min, support_max + 1))
    for assignment in product(range(width), repeat=n_atoms):
        total = sum(values[j] for j in assignment)
        mass = atom_pmfs.new_tensor(1.0)
        for i, j in enumerate(assignment):
            mass = mass * atom_pmfs[i, j]
        out[total - out_min] += mass
    return AggregatePMF(out, out_min)


def grouped_signed_binary_convolution(
    probs: torch.Tensor,
    signs: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: Optional[int] = None,
) -> AggregatePMF:
    """Batched grouped-conv PMFs for signed Bernoulli sums.

    Args:
        probs: `(B, L)` selection probabilities.
        signs: `(B, L)` values in `{-1, +1}`.
        group_ids: `(B, L)` integer group ids, or `-1` for ignored tokens.
        num_groups: Number of groups per batch item. If omitted, inferred from
            nonnegative `group_ids`.

    Returns:
        AggregatePMF with shape `(B, G, 2*K_max+1)` and support `[-K_max,K_max]`,
        where `K_max` is the maximum token count in any `(B,G)` group.
    """

    if probs.shape != signs.shape or probs.shape != group_ids.shape:
        raise ValueError("probs, signs, and group_ids must have the same shape")
    if probs.ndim != 2:
        raise ValueError("grouped_signed_binary_convolution expects (B, L) tensors")

    device = probs.device
    batch, length = probs.shape
    if num_groups is None:
        valid_group_ids = group_ids[group_ids >= 0]
        num_groups = int(valid_group_ids.max().item()) + 1 if valid_group_ids.numel() else 0
    if num_groups < 1:
        return AggregatePMF(probs.new_ones(batch, 0, 1), 0)

    valid = (group_ids >= 0) & (group_ids < num_groups)
    if not valid.any():
        return AggregatePMF(probs.new_ones(batch, num_groups, 1), 0)

    b_full = torch.arange(batch, device=device).unsqueeze(1).expand(batch, length)
    keys = b_full[valid] * num_groups + group_ids[valid].long()
    p = probs[valid]
    s = signs[valid].long()

    order = torch.argsort(keys)
    keys = keys[order]
    p = p[order]
    s = s[order]

    total_groups = batch * num_groups
    counts = torch.bincount(keys, minlength=total_groups)
    k_max = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts
    pos = torch.arange(keys.numel(), device=device) - starts[keys]

    packed_p = probs.new_zeros(total_groups, k_max)
    packed_s = torch.zeros(total_groups, k_max, dtype=torch.long, device=device)
    packed_valid = torch.zeros(total_groups, k_max, dtype=torch.bool, device=device)
    packed_p.index_put_((keys, pos), p, accumulate=False)
    packed_s.index_put_((keys, pos), s, accumulate=False)
    packed_valid.index_put_((keys, pos), torch.ones_like(pos, dtype=torch.bool), accumulate=False)

    support_width = 2 * k_max + 1
    center = k_max
    pmf = probs.new_zeros(1, total_groups, support_width)
    pmf[:, :, center] = 1.0

    for step in range(k_max):
        p_step = packed_p[:, step]
        valid_step = packed_valid[:, step]
        sign_step = packed_s[:, step]
        k_center = torch.where(valid_step, 1.0 - p_step, torch.ones_like(p_step))
        # conv1d performs cross-correlation. With padding=1, kernel[0]
        # pulls mass from the previous index and therefore represents a +1
        # shift; kernel[2] represents a -1 shift.
        k_plus = torch.where(valid_step & (sign_step > 0), p_step, torch.zeros_like(p_step))
        k_minus = torch.where(valid_step & (sign_step < 0), p_step, torch.zeros_like(p_step))
        kernel = torch.stack([k_plus, k_center, k_minus], dim=-1).unsqueeze(1)
        pmf = F.conv1d(pmf, kernel, groups=total_groups, padding=1)

    return AggregatePMF(pmf.squeeze(0).reshape(batch, num_groups, support_width), -center)


def aggregate_nll(aggregate: AggregatePMF, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Negative log likelihood for integer aggregate targets."""

    targets = targets.to(device=aggregate.probs.device, dtype=torch.long)
    idx = aggregate.value_to_index(targets)
    valid = (idx >= 0) & (idx < aggregate.probs.shape[-1])
    safe_idx = idx.clamp(0, aggregate.probs.shape[-1] - 1)
    gathered = torch.gather(aggregate.probs, -1, safe_idx.unsqueeze(-1)).squeeze(-1)
    gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    return -torch.log(gathered.clamp_min(eps))
