"""Faithful LLP-PVC method components adapted from the official code.

The upstream project is pinned as a submodule at
``third_party/ICLR2026_LLP-PVC``.  This module keeps the method-defining
pieces used by ``LLP-PVC.py`` while exposing them to our fixed-bag feature
protocol:

* sigmoid probabilities for the classwise count likelihood;
* Poisson-binomial count likelihood with per-class averaging;
* SGD with Nesterov momentum;
* official warmup-cosine learning-rate rule;
* softmax argmax prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution_fft_tree


@dataclass(frozen=True)
class OfficialLLPPVCConfig:
    lr: float = 2.5e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_frac: float = 0.08
    warmup_lr: float = 5e-5
    eps: float = 1e-30


def init_sigmoid_bias_to_one_over_k(layer: nn.Linear, num_classes: int) -> None:
    """Match upstream ``init_fc_bias_sigmoid_to_1_over_k`` for a linear head."""

    if num_classes <= 1 or layer.bias is None:
        return
    with torch.no_grad():
        layer.bias.fill_(-math.log(num_classes - 1))


def proportions_to_counts_exact(proportions: torch.Tensor, bag_size: int) -> torch.Tensor:
    """Upstream count conversion: round proportions and fix any sum mismatch."""

    counts = torch.round(proportions * float(bag_size)).to(torch.long)
    diff = int(bag_size - int(counts.sum().item()))
    if diff:
        counts[torch.argmax(proportions)] += diff
    return counts.clamp_(0, bag_size)


def _class_count_pmfs(probs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if probs.dim() != 3:
        raise ValueError(f"expected probs [B, n, C], got {tuple(probs.shape)}")
    if mask is None:
        mask = torch.ones(probs.shape[:2], dtype=torch.bool, device=probs.device)
    p = probs.transpose(1, 2)
    atoms = torch.stack([1.0 - p, p], dim=-1)
    zero = probs.new_tensor([1.0, 0.0])
    atoms = torch.where(mask[:, None, :, None], atoms, zero)
    return finite_support_convolution_fft_tree(atoms, support_min=0).probs


def official_count_loss(
    logits: torch.Tensor,
    proportions: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-30,
    reduce: str | None = "mean",
) -> torch.Tensor:
    """Official LLP-PVC classwise count likelihood on a batch of bags.

    This mirrors the training path in upstream ``LLP-PVC.py``:

    ``probs = torch.sigmoid(logits_u_w)``
    ``loss = compute_CC_loss_fft_precise_batched(..., reduce=\"mean\")``

    The upstream function sums NLL over classes for each bag, then averages
    over bags.  It uses proportions converted to integer class counts.
    """

    if logits.dim() != 3:
        raise ValueError(f"expected logits [B, n, C], got {tuple(logits.shape)}")
    if proportions.dim() != 2:
        raise ValueError(f"expected proportions [B, C], got {tuple(proportions.shape)}")
    if logits.shape[0] != proportions.shape[0] or logits.shape[2] != proportions.shape[1]:
        raise ValueError("logits/proportions batch or class dimensions do not match")
    probs = torch.sigmoid(logits)
    pmfs = _class_count_pmfs(probs, mask).clamp_min(eps)
    bag_size = logits.shape[1] if mask is None else int(mask[0].sum().item())
    losses = []
    for b in range(logits.shape[0]):
        n_b = int(mask[b].sum().item()) if mask is not None else bag_size
        counts = proportions_to_counts_exact(proportions[b].to(logits.device), n_b)
        per_class = [
            aggregate_nll(AggregatePMF(pmfs[b, c], 0), counts[c]) for c in range(logits.shape[2])
        ]
        losses.append(torch.stack(per_class).sum())
    loss_b = torch.stack(losses)
    if reduce is None:
        return loss_b.to(logits.dtype)
    if reduce == "mean":
        return loss_b.mean().to(logits.dtype)
    if reduce == "sum":
        return loss_b.sum().to(logits.dtype)
    raise ValueError("reduce must be None|'mean'|'sum'")


def official_predict(logits: torch.Tensor) -> torch.Tensor:
    """Official evaluation rule: softmax scores, argmax class."""

    return torch.softmax(logits, dim=-1).argmax(dim=-1)


class OfficialWarmupCosineLrScheduler(_LRScheduler):
    """Copy of the effective upstream ``WarmupCosineLrScheduler`` rule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iter: int,
        warmup_iter: int,
        warmup_ratio: float = 5e-4,
        warmup: str = "linear",
        last_epoch: int = -1,
    ) -> None:
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self) -> float:
        if self.last_epoch < self.warmup_iter:
            return self.get_warmup_ratio()
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = max(1, self.max_iter - self.warmup_iter)
        return float(np.cos((np.pi * real_iter) / (4 * real_max_iter * 0.5)))

    def get_warmup_ratio(self) -> float:
        t = (self.last_epoch + 1) / max(1, self.warmup_iter)
        if self.warmup == "linear":
            return float(self.warmup_ratio + (1 - self.warmup_ratio) * t)
        if self.warmup == "exp":
            return float(self.warmup_ratio ** (1 - t))
        raise ValueError("warmup must be 'linear' or 'exp'")
