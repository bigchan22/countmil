"""Common metrics for CountMIL experiments."""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.float() - target.float()).abs().mean().item()


def expected_value_from_pmf(probs: torch.Tensor, support_min: int) -> torch.Tensor:
    values = torch.arange(
        support_min,
        support_min + probs.shape[-1],
        device=probs.device,
        dtype=probs.dtype,
    )
    return (probs * values).sum(dim=-1)


def multiclass_ece_from_pmf(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error for aggregate PMF top-1 predictions."""

    if probs.ndim != 2:
        raise ValueError("probs must have shape (N,T)")
    targets = targets.to(device=probs.device, dtype=torch.long)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).to(probs.dtype)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)
    ece = probs.new_tensor(0.0)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf >= boundaries[i]) & (conf <= boundaries[i + 1])
        else:
            mask = (conf >= boundaries[i]) & (conf < boundaries[i + 1])
        if mask.any():
            weight = mask.to(probs.dtype).mean()
            ece = ece + weight * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.detach().cpu().item())


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary ROC AUC without sklearn.

    Returns NaN if only one class is present.
    """

    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().long().cpu()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)

    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg = ranks[order[start:end]].mean()
            ranks[order[start:end]] = avg
        start = end

    rank_sum_pos = ranks[pos].sum().item()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
