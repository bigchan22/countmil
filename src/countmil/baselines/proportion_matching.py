"""Simple LLP proportion matching baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_proportion_matching_loss(probs: torch.Tensor, proportions: torch.Tensor) -> torch.Tensor:
    pred = probs.mean(dim=-1)
    return F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), proportions.float())


def multiclass_proportion_matching_loss(
    probs: torch.Tensor,
    target_proportions: torch.Tensor,
    mask: torch.Tensor,
    loss: str = "kl",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match predicted and observed class proportions for multiclass LLP."""

    if probs.ndim != 3:
        raise ValueError("probs must have shape (B,N,C)")
    if mask.shape != probs.shape[:2]:
        raise ValueError("mask must have shape (B,N)")
    if target_proportions.shape != (probs.shape[0], probs.shape[2]):
        raise ValueError("target_proportions must have shape (B,C)")
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
    pred = (probs * mask.unsqueeze(-1).to(probs.dtype)).sum(dim=1) / denom
    pred = pred.clamp_min(eps)
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target_proportions.to(device=probs.device, dtype=probs.dtype).clamp_min(0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    if loss == "kl":
        return F.kl_div(pred.clamp_min(eps).log(), target, reduction="batchmean")
    if loss == "mse":
        return F.mse_loss(pred, target)
    if loss == "ce":
        return -(target * pred.clamp_min(eps).log()).sum(dim=-1).mean()
    raise ValueError(f"unknown multiclass LLP loss: {loss}")
