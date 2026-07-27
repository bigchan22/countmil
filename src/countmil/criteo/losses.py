"""Aggregate objectives and common Criteo evaluation metrics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from countmil.aggregators import AggregatePMF, aggregate_nll, binary_count_dp


def poisson_binomial_pmfs(probs: torch.Tensor) -> torch.Tensor:
    """Return exact count PMFs for a `(B,N)` Bernoulli-probability tensor."""

    return binary_count_dp(probs).probs


def fsconv_nll_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pmf = binary_count_dp(probs)
    return aggregate_nll(pmf, counts).mean()


def dllp_bce_loss(logits: torch.Tensor, counts: torch.Tensor, bag_size: int) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pbar = probs.mean(dim=1).clamp(1e-7, 1.0 - 1e-7)
    q = counts.float() / float(bag_size)
    return F.binary_cross_entropy(pbar, q)


def dllp_mse_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    pred_count = torch.sigmoid(logits).sum(dim=1)
    return F.mse_loss(pred_count, counts.float())


def easyllp_loss(logits: torch.Tensor, counts: torch.Tensor, *, bag_size: int, global_prior: float) -> torch.Tensor:
    """EasyLLP surrogate from the Google Research LLP-Bench implementation.

    The upstream code samples one instance per bag and applies a weighted BCE.
    Here the expectation over the uniformly sampled instance is used; this is
    deterministic and has the same objective in expectation.
    """

    probs = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    q = counts.float() / float(bag_size)
    prior = probs.new_tensor(float(global_prior))
    w_pos = float(bag_size) * (q - prior) + prior
    w_neg = float(bag_size) * (prior - q) + (1.0 - prior)
    pos = -torch.log(probs).mean(dim=1)
    neg = -torch.log1p(-probs).mean(dim=1)
    return (w_pos * pos + w_neg * neg).mean()


def expected_count_mae(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits).sum(dim=1) - counts.float()).abs()


def ece_binary(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = probs.detach().float().cpu()
    labels = labels.detach().float().cpu()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.any():
            ece += mask.float().mean() * (probs[mask].mean() - labels[mask].mean()).abs()
    return float(ece.item())


def instance_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    p = probs.detach().float().cpu().numpy()
    y = labels.detach().long().cpu().numpy()
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(((probs.detach().cpu() - labels.detach().cpu().float()) ** 2).mean().item()),
        "ece_15": ece_binary(probs, labels, n_bins=15),
        "accuracy_0p5": float(((probs.detach().cpu() >= 0.5).long() == labels.detach().cpu().long()).float().mean().item()),
    }


def aggregate_metrics(probs: torch.Tensor, counts: torch.Tensor) -> dict[str, torch.Tensor]:
    pmf = poisson_binomial_pmfs(probs)
    expected = probs.sum(dim=1)
    rounded = expected.round().long()
    mode = pmf.argmax(dim=1)
    nll = aggregate_nll(AggregatePMF(pmf, 0), counts)
    return {
        "expected_count_mae": (expected - counts.float()).abs(),
        "rounded_expected_count_acc": (rounded == counts.long()).float(),
        "pmf_mode_count_acc": (mode == counts.long()).float(),
        "poisson_binomial_nll": nll,
    }

