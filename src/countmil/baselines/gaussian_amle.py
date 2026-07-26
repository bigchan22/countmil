"""Generalized Gaussian approximate maximum likelihood for aggregate sums."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_GAUSSIAN_AMLE_EPS = 1e-4


@dataclass(frozen=True)
class GaussianMoments:
    mean: torch.Tensor
    variance: torch.Tensor


def categorical_sum_moments(
    probs: torch.Tensor,
    support_values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> GaussianMoments:
    """Compute independent finite-support sum mean and variance.

    Args:
        probs: Tensor of shape ``(..., N, S)`` with atomic probabilities.
        support_values: Tensor of shape ``(S,)`` containing contribution values.
        mask: Optional boolean tensor of shape ``(..., N)``. Masked atoms
            contribute zero mean and zero variance.
    """

    if probs.ndim < 2:
        raise ValueError("probs must have shape (..., N, S)")
    if support_values.ndim != 1 or support_values.shape[0] != probs.shape[-1]:
        raise ValueError("support_values must have shape (S,)")
    work_dtype = torch.float64 if probs.dtype == torch.float64 else probs.dtype
    p = probs.to(dtype=work_dtype)
    values = support_values.to(device=probs.device, dtype=work_dtype)
    mean_i = (p * values).sum(dim=-1)
    second_i = (p * values.square()).sum(dim=-1)
    var_i = (second_i - mean_i.square()).clamp_min(0.0)
    if mask is not None:
        if mask.shape != probs.shape[:-1]:
            raise ValueError("mask must have shape (..., N)")
        valid = mask.to(device=probs.device, dtype=work_dtype)
        mean_i = mean_i * valid
        var_i = var_i * valid
    return GaussianMoments(mean=mean_i.sum(dim=-1), variance=var_i.sum(dim=-1))


def signed_bernoulli_sum_moments(
    probs: torch.Tensor,
    signs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> GaussianMoments:
    """Mean and variance for ``sum_i sign_i z_i`` with Bernoulli ``z_i``."""

    if probs.shape != signs.shape:
        raise ValueError("probs and signs must have the same shape")
    work_dtype = torch.float64 if probs.dtype == torch.float64 else probs.dtype
    p = probs.to(dtype=work_dtype)
    s = signs.to(device=probs.device, dtype=work_dtype)
    mean_i = s * p
    var_i = (p * (1.0 - p)).clamp_min(0.0)
    if mask is not None:
        if mask.shape != probs.shape:
            raise ValueError("mask must have the same shape as probs")
        valid = mask.to(device=probs.device, dtype=work_dtype)
        mean_i = mean_i * valid
        var_i = var_i * valid
    return GaussianMoments(mean=mean_i.sum(dim=-1), variance=var_i.sum(dim=-1))


def gaussian_amle_loss_from_moments(
    mean: torch.Tensor,
    variance: torch.Tensor,
    targets: torch.Tensor,
    eps: float = DEFAULT_GAUSSIAN_AMLE_EPS,
) -> torch.Tensor:
    """Per-bag Gaussian-AMLE negative log likelihood up to a constant.

    The epsilon is variance flooring for numerical stability. It is deliberately
    exposed and logged by training scripts because deterministic atoms otherwise
    lead to zero variance.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    targets = targets.to(device=mean.device, dtype=mean.dtype)
    var = variance.clamp_min(0.0) + float(eps)
    loss = 0.5 * ((targets - mean).square() / var + var.log())
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Gaussian-AMLE produced non-finite loss")
    return loss


def categorical_gaussian_amle_loss(
    probs: torch.Tensor,
    support_values: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = DEFAULT_GAUSSIAN_AMLE_EPS,
) -> torch.Tensor:
    moments = categorical_sum_moments(probs, support_values, mask)
    return gaussian_amle_loss_from_moments(moments.mean, moments.variance, targets, eps)


def signed_bernoulli_gaussian_amle_loss(
    probs: torch.Tensor,
    signs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = DEFAULT_GAUSSIAN_AMLE_EPS,
) -> torch.Tensor:
    moments = signed_bernoulli_sum_moments(probs, signs, mask)
    return gaussian_amle_loss_from_moments(moments.mean, moments.variance, targets, eps)


def _log_sub_exp(log_hi: torch.Tensor, log_lo: torch.Tensor) -> torch.Tensor:
    """Stable ``log(exp(log_hi) - exp(log_lo))`` for ``log_hi >= log_lo``."""

    return log_hi + torch.log1p(-torch.exp(log_lo - log_hi).clamp_max(1.0))


def gaussian_integer_bin_nll(
    mean: torch.Tensor,
    variance: torch.Tensor,
    targets: torch.Tensor,
    eps: float = DEFAULT_GAUSSIAN_AMLE_EPS,
) -> torch.Tensor:
    """Discrete integer-bin Gaussian NLL using stable log CDF differences.

    The event for integer target ``y`` is ``[y-0.5, y+0.5]`` under the
    Gaussian approximation. This is an evaluation metric, not the AMLE training
    objective.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    work_dtype = torch.float64 if mean.dtype == torch.float64 or variance.dtype == torch.float64 else mean.dtype
    mu = mean.to(dtype=work_dtype)
    var = variance.to(device=mean.device, dtype=work_dtype).clamp_min(0.0) + float(eps)
    y = targets.to(device=mean.device, dtype=work_dtype)
    sigma = torch.sqrt(var)
    inv_sqrt2 = torch.tensor(2.0, device=mean.device, dtype=work_dtype).sqrt().reciprocal()
    upper = (y + 0.5 - mu) / sigma
    lower = (y - 0.5 - mu) / sigma
    log_phi_upper = torch.special.log_ndtr(upper)
    log_phi_lower = torch.special.log_ndtr(lower)
    log_mass = _log_sub_exp(log_phi_upper, log_phi_lower)
    # For extreme right-tail intervals, use symmetry to avoid subtracting two
    # CDF values both numerically equal to one.
    right_tail = lower > 0
    if right_tail.any():
        log_sf_lower = torch.special.log_ndtr(-lower[right_tail])
        log_sf_upper = torch.special.log_ndtr(-upper[right_tail])
        log_mass = log_mass.clone()
        log_mass[right_tail] = _log_sub_exp(log_sf_lower, log_sf_upper)
    nll = -log_mass
    if not torch.isfinite(nll).all():
        raise FloatingPointError("Gaussian integer-bin NLL produced non-finite values")
    return nll
