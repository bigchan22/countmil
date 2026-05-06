"""Training utilities for B1 verification.

Functions:
  select_device()       — single-source device selection (MPS/CUDA/CPU).
  per_bag_forward()     — bag-level reshape wrapper around any backbone.
  train()               — train loop returning epoch-level metrics dict.
  run_seeds()           — multi-seed sweep, persists per-seed JSON.

See spec §8.
"""
from __future__ import annotations
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sklearn.metrics
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from pca.losses import forward_conv
from pca.metrics import (
    top1_acc_from_log_pmf, mae_from_log_pmf, ece_from_log_pmf,
)


def select_device() -> torch.device:
    """Mac MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def per_bag_forward(model: nn.Module, batch: Tensor) -> Tensor:
    """Reshape (B, N, *features) → backbone → (B, N) logits.

    Args:
        model: backbone returning (B*N, 1) logits.
        batch: (B, N, *features) — (1, 28, 28) for MNIST or (D,) for LLP.
    """
    B, N = batch.shape[:2]
    flat = batch.reshape(B * N, *batch.shape[2:])
    logits_flat = model(flat).squeeze(-1)
    return logits_flat.view(B, N)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """Train + per-epoch eval. Returns metrics dict per spec §12.1.

    Metric definitions:
      - epoch_train_loss: mean per-sample loss over the epoch.
      - epoch_inst_auc:   ROC-AUC of test per-instance probabilities vs ground-truth.
      - epoch_bag_acc:    test bag-level top-1 count accuracy via PMF argmax.
      - best_inst_auc:    mean of last 5 epoch_inst_auc values.
      - best_bag_acc:     mean of last 5 epoch_bag_acc values.
    """
    model = model.to(device)
    epoch_train_loss = []
    epoch_inst_auc = []
    epoch_bag_acc = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for images, bag_count, _gt in train_loader:
            images = images.to(device)
            bag_count = bag_count.to(device)
            logits = per_bag_forward(model, images)
            loss = loss_fn(logits, bag_count)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
            running_count += images.shape[0]
        epoch_train_loss.append(running_loss / max(running_count, 1))

        # Test eval
        model.eval()
        all_p, all_z = [], []
        bag_correct = 0
        bag_total = 0
        with torch.no_grad():
            for images, bag_count, gt in test_loader:
                images = images.to(device)
                logits = per_bag_forward(model, images)
                p = torch.sigmoid(logits)
                all_p.append(p.cpu().flatten())
                all_z.append(gt.flatten())
                log_P = forward_conv(p)
                pred = log_P.argmax(dim=1).cpu()
                bag_correct += (pred == bag_count).sum().item()
                bag_total += bag_count.shape[0]
        inst_auc = float(sklearn.metrics.roc_auc_score(
            torch.cat(all_z).numpy(), torch.cat(all_p).numpy()
        ))
        epoch_inst_auc.append(inst_auc)
        epoch_bag_acc.append(bag_correct / max(bag_total, 1))

    n_tail = min(5, num_epochs)
    return {
        'epoch_train_loss': epoch_train_loss,
        'epoch_inst_auc': epoch_inst_auc,
        'epoch_bag_acc': epoch_bag_acc,
        'best_inst_auc': float(np.mean(epoch_inst_auc[-n_tail:])),
        'best_bag_acc': float(np.mean(epoch_bag_acc[-n_tail:])),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_seeds(
    seeds: list[int],
    build_fn: Callable[[int], tuple],
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    config_extra: dict | None = None,
) -> None:
    """Sequential per-seed training, persists results to save_dir/seed{i}.json.

    Args:
        build_fn(seed) -> (model, train_loader, test_loader, loss_fn, optimizer).
            Must produce identical data and model init for the same seed; method
            differs only via loss_fn for paired comparison validity (spec §8.3).
        config_extra: extra fields to merge into the persisted config dict.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, tr_loader, te_loader, loss_fn, optimizer = build_fn(seed)
        t0 = time.time()
        metrics = train(model, tr_loader, te_loader, loss_fn, optimizer, num_epochs, device)
        wall = time.time() - t0

        out = {
            'config': {**(config_extra or {}), 'seed': seed, 'epochs': num_epochs,
                       'device': str(device), 'git_sha': sha,
                       'torch_version': torch.__version__},
            **metrics,
            'wall_clock_sec': wall,
        }
        out_path = save_dir / f'seed{seed}.json'
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  [seed {seed}] inst_auc={metrics['best_inst_auc']:.4f} "
              f"bag_acc={metrics['best_bag_acc']:.4f} wall={wall:.1f}s -> {out_path}")


def per_bag_features_masked(
    backbone: nn.Module, batch_features: Tensor, mask: Tensor,
) -> Tensor:
    """Reshape (B, N_max, *feature_shape) → backbone → (B, N_max, F).

    Inactive (mask=False) instances are still fed through the backbone (uses
    padding values from collate); their features are zeroed out in the result
    so downstream layers (mean-pool, attention-after-mask, etc.) cannot
    accidentally use them. The baseline's mask handling is the source of truth
    for how inactive instances affect loss.
    """
    B, N_max = batch_features.shape[:2]
    flat = batch_features.reshape(B * N_max, *batch_features.shape[2:])
    feat_flat = backbone(flat)                       # (B*N_max, F)
    F_dim = feat_flat.shape[-1]
    feat = feat_flat.view(B, N_max, F_dim)
    feat = feat * mask.unsqueeze(-1).float()
    return feat


def train_a1(
    backbone: nn.Module,
    baseline: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """A1 train loop with variable-N + mask + 4-metric eval.

    Per-epoch test eval reports:
      epoch_test_nll: mean -log P(bag_y) under bag PMF (uses log_P from baseline).
      epoch_test_acc: top-1 accuracy of argmax(log_P) == bag_y.
      epoch_test_mae: |E[Y_pred] - bag_y|.
      epoch_test_ece: 15-bin Expected Calibration Error.
    Returns dict with epoch arrays and last-5-epoch means under best_test_*.
    """
    backbone = backbone.to(device)
    baseline = baseline.to(device)
    epoch_train_loss = []
    epoch_test_nll = []
    epoch_test_acc = []
    epoch_test_mae = []
    epoch_test_ece = []

    for epoch in range(num_epochs):
        backbone.train(); baseline.train()
        running_loss = 0.0
        running_count = 0
        for features, bag_y, _gt, mask in train_loader:
            features = features.to(device)
            bag_y = bag_y.to(device)
            mask = mask.to(device)
            feat = per_bag_features_masked(backbone, features, mask)
            loss, _log_P = baseline(feat, bag_y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.shape[0]
            running_count += features.shape[0]
        epoch_train_loss.append(running_loss / max(running_count, 1))

        # Test eval
        backbone.eval(); baseline.eval()
        all_log_P, all_y = [], []
        with torch.no_grad():
            for features, bag_y, _gt, mask in test_loader:
                features = features.to(device)
                bag_y = bag_y.to(device)
                mask = mask.to(device)
                feat = per_bag_features_masked(backbone, features, mask)
                _loss, log_P = baseline(feat, bag_y, mask)
                all_log_P.append(log_P.cpu())
                all_y.append(bag_y.cpu())
        log_P_all = torch.cat(all_log_P, dim=0)
        y_all = torch.cat(all_y, dim=0)
        nll = -log_P_all.gather(1, y_all.unsqueeze(1)).squeeze(1).mean().item()
        epoch_test_nll.append(float(nll))
        epoch_test_acc.append(top1_acc_from_log_pmf(log_P_all, y_all))
        epoch_test_mae.append(mae_from_log_pmf(log_P_all, y_all))
        epoch_test_ece.append(ece_from_log_pmf(log_P_all, y_all, n_bins=15))

    n_tail = min(5, num_epochs)
    return {
        'epoch_train_loss': epoch_train_loss,
        'epoch_test_nll': epoch_test_nll,
        'epoch_test_acc': epoch_test_acc,
        'epoch_test_mae': epoch_test_mae,
        'epoch_test_ece': epoch_test_ece,
        'best_test_nll': float(np.mean(epoch_test_nll[-n_tail:])),
        'best_test_acc': float(np.mean(epoch_test_acc[-n_tail:])),
        'best_test_mae': float(np.mean(epoch_test_mae[-n_tail:])),
        'best_test_ece': float(np.mean(epoch_test_ece[-n_tail:])),
    }


def run_seeds_a1(
    seeds: list[int],
    build_fn: Callable[[int], tuple],
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    config_extra: dict | None = None,
) -> None:
    """Sequential per-seed A1 training, persists results to save_dir/seed{i}.json.

    build_fn(seed) -> (backbone, baseline, train_loader, test_loader, optimizer).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        backbone, baseline, tr_loader, te_loader, optimizer = build_fn(seed)
        t0 = time.time()
        metrics = train_a1(backbone, baseline, tr_loader, te_loader, optimizer,
                            num_epochs, device)
        wall = time.time() - t0

        out = {
            'config': {**(config_extra or {}), 'seed': seed, 'epochs': num_epochs,
                        'device': str(device), 'git_sha': sha,
                        'torch_version': torch.__version__},
            **metrics,
            'wall_clock_sec': wall,
        }
        out_path = save_dir / f'seed{seed}.json'
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  [seed {seed}] nll={metrics['best_test_nll']:.4f} "
              f"acc={metrics['best_test_acc']:.4f} ece={metrics['best_test_ece']:.4f} "
              f"wall={wall:.1f}s -> {out_path}")
