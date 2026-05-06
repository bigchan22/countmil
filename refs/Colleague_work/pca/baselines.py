"""Baselines for A1 multi-class verification — see spec §8.

All baselines are nn.Modules with forward signature:
    forward(features, bag_y, mask) -> (loss, log_P)
where:
  features: (B, N_max, F) backbone-output features.
  bag_y:    (B,) integer bag sum (already shifted for signed atoms).
  mask:     (B, N_max) bool — True for active instances.
Returns:
  loss:  scalar training loss.
  log_P: (B, NK+1) log-probability over bag-sum support; None if baseline lacks
         a native bag distribution and uses Gaussian wrap in eval (1a, 2a).
"""
from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from pca.losses import LOG_ZERO, atomic_conv, multiclass_marginal_nll_loss


def _gaussian_log_pmf(mu: Tensor, var: Tensor, T: int) -> Tensor:
    """Discretized Gaussian log-PMF over k = 0, 1, ..., T-1.

    Used by methods that train on continuous targets but need a bag PMF for
    NLL/ECE evaluation. Each bin is normalized so log_P sums to 1 in prob space.
    """
    k = torch.arange(T, device=mu.device).float().unsqueeze(0)  # (1, T)
    log_unnorm = -0.5 * ((k - mu.unsqueeze(1)) ** 2) / var.unsqueeze(1).clamp(min=1e-3)
    return log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)


class MeanPoolBaseline(nn.Module):
    """1a — Mean-pool regression. Per-instance scalar prediction; bag sum = Σ_i z_i.

    Loss: MSE(Σ ẑ_i, Y). Eval log_P: Gaussian wrap with σ²_bag = N * K^2 / 12
    (uniform-on-[0,K] per-instance variance × N).
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        z_hat = self.head(features).squeeze(-1)     # (B, N_max)
        z_hat = z_hat * mask.float()
        bag_pred = z_hat.sum(dim=1)                  # (B,)
        loss = F.mse_loss(bag_pred, bag_y.float())
        # Gaussian-wrap bag PMF over k = 0..N_max*K
        T = self.N_max * self.K + 1
        n_active = mask.sum(dim=1).float().clamp(min=1.0)
        var = n_active * (self.K ** 2) / 12.0
        log_P = _gaussian_log_pmf(bag_pred, var, T)
        return loss, log_P


class PLMulticlassBaseline(nn.Module):
    """2a — PL-multiclass (sum-only adaptation).

    Per-instance softmax over {0..K}; bag mean prediction = (1/N) Σ_i E[z_i].
    Loss: MSE(bag_mean_pred, bag_y / N). Eval log_P: Gaussian wrap with
    per-instance variance σ²_i = E[z²]-E[z]² aggregated over active instances.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        logits = self.head(features)                 # (B, N_max, K+1)
        probs = F.softmax(logits, dim=-1)            # (B, N_max, K+1)
        k_grid = torch.arange(self.K + 1, device=features.device).float()
        mu_i = (probs * k_grid).sum(dim=-1)          # (B, N_max)
        var_i = (probs * (k_grid ** 2)).sum(dim=-1) - mu_i ** 2
        mu_i = mu_i * mask.float()
        var_i = var_i * mask.float()
        n_active = mask.sum(dim=1).float().clamp(min=1.0)
        bag_mean_pred = mu_i.sum(dim=1) / n_active
        loss = F.mse_loss(bag_mean_pred, bag_y.float() / n_active)
        T = self.N_max * self.K + 1
        log_P = _gaussian_log_pmf(mu_i.sum(dim=1), var_i.sum(dim=1).clamp(min=1e-3), T)
        return loss, log_P


class CLTGaussianBaseline(nn.Module):
    """2b — CLT Gaussian. Per-instance softmax → moments; bag = N(Σμ, Σσ²).

    Loss: -log N(Y; μ_bag, σ²_bag). Eval log_P: discretized Gaussian over
    k = 0..N_max*K. Native bag distribution (not Gaussian wrap on top of MSE).
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        logits = self.head(features)                # (B, N_max, K+1)
        probs = F.softmax(logits, dim=-1)
        k_grid = torch.arange(self.K + 1, device=features.device).float()
        mu_i = (probs * k_grid).sum(dim=-1)
        var_i = (probs * (k_grid ** 2)).sum(dim=-1) - mu_i ** 2
        mu_i = mu_i * mask.float()
        var_i = var_i * mask.float()
        mu_bag = mu_i.sum(dim=1)
        var_bag = var_i.sum(dim=1).clamp(min=1e-3)
        # Continuous Gaussian NLL on integer Y
        loss = 0.5 * (torch.log(2 * math.pi * var_bag)
                       + (bag_y.float() - mu_bag) ** 2 / var_bag).mean()
        T = self.N_max * self.K + 1
        log_P = _gaussian_log_pmf(mu_bag, var_bag, T)
        return loss, log_P


class AttentionPoolingBaseline(nn.Module):
    """3a — Ilse 2018 gated attention pooling.

    a_i = softmax_i(w^T (tanh(V h_i) ⊙ sigmoid(U h_i))) over masked instances;
    bag embedding z = Σ_i a_i h_i; bag head MLP → Linear(NK+1) → softmax → CE.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int, attn_dim: int = 64):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.V = nn.Linear(feature_dim, attn_dim)
        self.U = nn.Linear(feature_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        T = N_max * K + 1
        self.bag_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, T),
        )

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        gate = torch.tanh(self.V(features)) * torch.sigmoid(self.U(features))
        scores = self.w(gate).squeeze(-1)                # (B, N_max)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=1)                  # (B, N_max)
        z = (attn.unsqueeze(-1) * features).sum(dim=1)   # (B, F)
        bag_logits = self.bag_head(z)                    # (B, T)
        log_P = F.log_softmax(bag_logits, dim=1)
        loss = F.nll_loss(log_P, bag_y)
        return loss, log_P


class DeepSetsBaseline(nn.Module):
    """3b — DeepSets (Zaheer 2017). φ → sum → ρ.

    Per-instance φ(h_i) MLP, mask-aware sum aggregation, bag head ρ → Linear(NK+1)
    → softmax CE. Sum (not mean) to preserve N-information.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int, hidden: int = 64):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        T = N_max * K + 1
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T),
        )

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        phi_h = self.phi(features) * mask.unsqueeze(-1).float()    # (B, N_max, hidden)
        z = phi_h.sum(dim=1)                                         # (B, hidden)
        bag_logits = self.rho(z)                                     # (B, T)
        log_P = F.log_softmax(bag_logits, dim=1)
        loss = F.nll_loss(log_P, bag_y)
        return loss, log_P


class PCABaseline(nn.Module):
    """PCA — exact bag PMF via 1D log-space conv over per-instance atomic PMF.

    Per-instance head: Linear(F, K+1). Atom logits → log_softmax → atomic_conv.
    Loss: multiclass_marginal_nll_loss (handles inactive instances via mask =
    delta_0 forcing). Native exact bag PMF in log space. For signed atoms
    (S=3 → K=2), caller pre-shifts bag_y by +N before passing.
    """

    def __init__(self, feature_dim: int, K: int, N_max: int):
        super().__init__()
        self.K = K
        self.N_max = N_max
        self.head = nn.Linear(feature_dim, K + 1)

    def forward(
        self, features: Tensor, bag_y: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert features.shape[1] <= self.N_max, (
            f"per-batch N={features.shape[1]} > self.N_max={self.N_max}; "
            f"bag PMF support would mismatch")
        logits = self.head(features)             # (B, N_max, K+1)
        # Log-PMF over bag-sum support T = N_max*K+1
        log_atoms = F.log_softmax(logits, dim=-1)
        if mask is not None:
            S = self.K + 1
            delta_0 = log_atoms.new_full((S,), LOG_ZERO)
            delta_0[0] = 0.0
            log_atoms = torch.where(mask.unsqueeze(-1), log_atoms, delta_0)
        log_P = atomic_conv(log_atoms)
        # Pad to global T_max so cross-batch concat is consistent (variable_n_collate_fn
        # pads to per-batch N_max which may be < self.N_max).
        T_max = self.N_max * self.K + 1
        if log_P.shape[1] < T_max:
            pad = T_max - log_P.shape[1]
            log_P = F.pad(log_P, (0, pad), value=LOG_ZERO)
        log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
        loss = -log_P_y.mean()
        return loss, log_P
