#!/usr/bin/env python
"""Train signed MNIST Count-MIL with exact finite-support convolution.

Each instance is a latent Bernoulli target-digit indicator. A known sign
`s_i in {-1,+1}` turns that hidden indicator into contribution `s_i z_i`, and
the observed bag label is the signed sum.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution
from countmil.datasets import SignedMNISTBags, collate_mnist_bags
from countmil.metrics import binary_auc
from countmil.models import ShuklaMNISTSelector
from countmil.training.run import make_run_dir, write_run_metadata
from countmil.training.seed import set_seed


def _parse_simple_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    out: dict[str, Any] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        elif value.startswith("[") and value.endswith("]"):
            parsed = [x.strip() for x in value[1:-1].split(",") if x.strip()]
            parsed = [int(x) if x.lstrip("-").isdigit() else x for x in parsed]
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        out[key.strip()] = parsed
    return out


def signed_count_pmf(probs: torch.Tensor, signs: torch.Tensor, mask: torch.Tensor) -> AggregatePMF:
    """Return exact PMF for `sum_i signs_i z_i`.

    Args:
        probs: `(B,N)` target-digit probabilities.
        signs: `(B,N)` signs in `{-1,+1}`.
        mask: `(B,N)` valid instance mask.
    """

    if probs.shape != signs.shape or probs.shape != mask.shape:
        raise ValueError("probs, signs, and mask must have the same shape")
    if probs.ndim != 2:
        raise ValueError("signed_count_pmf expects (B,N) tensors")

    zero = torch.zeros_like(probs)
    atoms = torch.stack([zero, 1.0 - probs, zero], dim=-1)
    atoms[..., 0] = torch.where(signs < 0, probs, atoms[..., 0])
    atoms[..., 2] = torch.where(signs > 0, probs, atoms[..., 2])

    zero_atom = probs.new_tensor([0.0, 1.0, 0.0])
    atoms = torch.where(mask.unsqueeze(-1), atoms, zero_atom)
    return finite_support_convolution(atoms, support_min=-1)


def _mode_values(pmf: AggregatePMF) -> torch.Tensor:
    return pmf.probs.argmax(dim=-1) + pmf.support_min


def _expected_value(pmf: AggregatePMF) -> torch.Tensor:
    values = torch.arange(pmf.support_min, pmf.support_max + 1, device=pmf.probs.device, dtype=pmf.probs.dtype)
    return (pmf.probs * values).sum(dim=-1)


def expected_signed_count(probs: torch.Tensor, signs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return E[sum_i signs_i z_i] for Bernoulli target probabilities."""

    if probs.shape != signs.shape or probs.shape != mask.shape:
        raise ValueError("probs, signs, and mask must have the same shape")
    return (probs * signs.to(probs.dtype) * mask.to(probs.dtype)).sum(dim=1)


def _evaluate(model: ShuklaMNISTSelector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_bags = 0
    pred_counts = []
    exp_counts = []
    true_counts = []
    inst_scores = []
    inst_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            signed_counts = batch["signed_count"].to(device)
            hidden = batch["instance_labels"].to(device)

            probs = model.predict_proba(x)
            pmf = signed_count_pmf(probs, signs, mask)
            nll = aggregate_nll(pmf, signed_counts)
            total_nll += nll.sum().item()
            total_bags += x.shape[0]

            pred_counts.append(_mode_values(pmf).cpu())
            exp_counts.append(_expected_value(pmf).cpu())
            true_counts.append(signed_counts.cpu())
            inst_scores.append(probs[mask].cpu())
            inst_labels.append(hidden[mask].cpu())

    pred_counts_t = torch.cat(pred_counts)
    exp_counts_t = torch.cat(exp_counts)
    true_counts_t = torch.cat(true_counts)
    inst_scores_t = torch.cat(inst_scores)
    inst_labels_t = torch.cat(inst_labels)
    zero_mask = true_counts_t == 0

    metrics = {
        "nll": total_nll / max(total_bags, 1),
        "signed_count_acc": (pred_counts_t == true_counts_t).float().mean().item(),
        "signed_count_mae": (pred_counts_t.float() - true_counts_t.float()).abs().mean().item(),
        "expected_signed_count_mae": (exp_counts_t - true_counts_t.float()).abs().mean().item(),
        "instance_acc": ((inst_scores_t >= 0.5).long() == inst_labels_t).float().mean().item(),
        "instance_auc": binary_auc(inst_scores_t, inst_labels_t),
        "zero_fraction": zero_mask.float().mean().item(),
    }
    if zero_mask.any():
        metrics["zero_signed_count_acc"] = (pred_counts_t[zero_mask] == 0).float().mean().item()
        metrics["zero_expected_abs_error"] = exp_counts_t[zero_mask].abs().mean().item()
    else:
        metrics["zero_signed_count_acc"] = math.nan
        metrics["zero_expected_abs_error"] = math.nan
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--objective", choices=["nll", "mse"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target-digit", type=int, default=None)
    parser.add_argument("--bag-size-mean", type=int, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--cancellation-heavy", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results-dir", default="results/signed_mnist")
    args = parser.parse_args()

    cfg = {
        "dataset_root": "data",
        "objective": "nll",
        "dataset": "MNIST",
        "target_digit": 9,
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 1000,
        "cancellation_heavy": True,
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    if "bag_size" in cfg and "bag_size_mean" not in cfg:
        cfg["bag_size_mean"] = cfg["bag_size"]
    if "cancellation_split" in cfg and "cancellation_heavy" not in cfg:
        cfg["cancellation_heavy"] = cfg["cancellation_split"]
    for key, value in {
        "dataset_root": args.dataset_root,
        "objective": args.objective,
        "dataset": args.dataset,
        "target_digit": args.target_digit,
        "bag_size_mean": args.bag_size_mean,
        "bag_size_std": args.bag_size_std,
        "train_bags": args.train_bags,
        "seed": args.seed,
    }.items():
        if value is not None:
            cfg[key] = value
    if args.cancellation_heavy:
        cfg["cancellation_heavy"] = True

    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    objective = str(cfg["objective"])
    run_dir = make_run_dir(args.run_root, "signed_mnist", f"conv_{objective}", str(cfg["dataset"]), seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_ds = SignedMNISTBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="train",
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        cancellation_heavy=bool(cfg["cancellation_heavy"]),
        seed=seed,
    )
    test_ds = SignedMNISTBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="test",
        num_bags=int(args.test_bags),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        cancellation_heavy=bool(cfg["cancellation_heavy"]),
        seed=seed + 10_000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mnist_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)

    model = ShuklaMNISTSelector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    best = {"instance_auc": -math.inf}
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in train_loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            signed_counts = batch["signed_count"].to(device)
            probs = model.predict_proba(x)
            if objective == "nll":
                pmf = signed_count_pmf(probs, signs, mask)
                loss = aggregate_nll(pmf, signed_counts).mean()
            elif objective == "mse":
                expected = expected_signed_count(probs, signs, mask)
                loss = F.mse_loss(expected, signed_counts.to(probs.dtype))
            else:
                raise ValueError(f"unknown objective: {objective}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_seen += x.shape[0]

        metrics = _evaluate(model, test_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)})
        if objective == "nll":
            metrics["train_nll"] = metrics["train_loss"]
        metrics.update({"epoch_seconds": time.perf_counter() - epoch_start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={metrics['train_loss']:.4f} "
            f"test_nll={metrics['nll']:.4f} signed_mae={metrics['signed_count_mae']:.3f} "
            f"zero_acc={metrics['zero_signed_count_acc']:.3f} "
            f"inst_auc={metrics['instance_auc']:.4f} "
            f"time={metrics['epoch_seconds']:.2f}s mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )
        if metrics["instance_auc"] > best["instance_auc"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": cfg, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    suffix = "conv" if objective == "nll" else f"conv_{objective}"
    summary_path = results_dir / f"signed_mnist_{suffix}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
