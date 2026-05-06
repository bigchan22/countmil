"""Evaluation metrics for A1 multi-class verification — see spec §10.4 / §11."""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def top1_acc_from_log_pmf(log_P: Tensor, bag_y: Tensor) -> float:
    """Top-1 accuracy of bag-sum prediction = fraction with argmax(log_P) == bag_y.

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth bag sum (already shift-corrected).
    Returns:
        scalar float in [0, 1].
    """
    pred = log_P.argmax(dim=1)
    return float((pred == bag_y.to(log_P.device)).float().mean().item())


def mae_from_log_pmf(log_P: Tensor, bag_y: Tensor) -> float:
    """MAE of expected bag sum vs ground truth: E_log_P[Y] = Σ_k k * exp(log_P[k]).

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth bag sum.
    Returns:
        scalar float (mean absolute error).
    """
    T = log_P.shape[1]
    k_grid = torch.arange(T, device=log_P.device).float()
    expected = (log_P.exp() * k_grid).sum(dim=1)
    return float((expected - bag_y.float().to(log_P.device)).abs().mean().item())


def ece_from_log_pmf(log_P: Tensor, bag_y: Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error over predicted bag-sum confidences.

    For each sample, take p_pred = max_k exp(log_P[b, k]) (top-1 confidence)
    and acc_b = 1[argmax(log_P[b]) == bag_y[b]]. Bin samples by p_pred into
    n_bins equal-width bins on [0, 1]; ECE = Σ_bin (|bin| / N) * |acc_bin - conf_bin|.

    Args:
        log_P: (B, T) log probability over bag-sum support.
        bag_y: (B,)   integer ground truth.
        n_bins: number of equal-width bins.
    Returns:
        scalar float (ECE).
    """
    pred = log_P.argmax(dim=1)
    confidence = log_P.exp().max(dim=1).values
    correct = (pred == bag_y.to(log_P.device)).float()
    confidence_np = confidence.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidence_np > lo) & (confidence_np <= hi) if b > 0 \
                 else (confidence_np >= lo) & (confidence_np <= hi)
        if not in_bin.any():
            continue
        acc_bin = correct_np[in_bin].mean()
        conf_bin = confidence_np[in_bin].mean()
        ece += abs(acc_bin - conf_bin) * in_bin.sum() / len(confidence_np)
    return float(ece)


def reliability_diagram_data(
    log_P: Tensor, bag_y: Tensor, n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, accuracies, confidences, weights) for reliability plot.

    Bins with no samples report 0.0 for accuracy and confidence; weight is
    fraction of total samples in that bin (sums to 1).
    """
    pred = log_P.argmax(dim=1)
    confidence = log_P.exp().max(dim=1).values
    correct = (pred == bag_y.to(log_P.device)).float()
    confidence_np = confidence.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    N = len(confidence_np)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidence_np > lo) & (confidence_np <= hi) if b > 0 \
                 else (confidence_np >= lo) & (confidence_np <= hi)
        if not in_bin.any():
            continue
        accs[b] = correct_np[in_bin].mean()
        confs[b] = confidence_np[in_bin].mean()
        weights[b] = in_bin.sum() / N
    return centers, accs, confs, weights
