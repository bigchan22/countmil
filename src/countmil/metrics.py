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
