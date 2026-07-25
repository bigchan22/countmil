#!/usr/bin/env python
"""Train expected-value baselines for signed MNIST Count-MIL.

The model predicts per-instance target probabilities p_i. Given known signs
s_i in {-1,+1}, the baseline regresses the expected signed aggregate
sum_i s_i p_i to the observed signed count. Instance labels are not used for
training.
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


def expected_signed_count(probs: torch.Tensor, signs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probs.shape != signs.shape or probs.shape != mask.shape:
        raise ValueError("probs, signs, and mask must have the same shape")
    return (probs * signs.to(probs.dtype) * mask.to(probs.dtype)).sum(dim=1)


def _evaluate(model: ShuklaMNISTSelector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
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
            expected = expected_signed_count(probs, signs, mask)
            exp_counts.append(expected.cpu())
            true_counts.append(signed_counts.cpu())
            inst_scores.append(probs[mask].cpu())
            inst_labels.append(hidden[mask].cpu())

    exp_counts_t = torch.cat(exp_counts)
    true_counts_t = torch.cat(true_counts).float()
    rounded = exp_counts_t.round().long()
    true_long = true_counts_t.long()
    inst_scores_t = torch.cat(inst_scores)
    inst_labels_t = torch.cat(inst_labels)
    zero_mask = true_long == 0
    err = exp_counts_t - true_counts_t
    round_err = rounded.float() - true_counts_t

    metrics = {
        "expected_signed_count_mse": err.square().mean().item(),
        "expected_signed_count_mae": err.abs().mean().item(),
        "rounded_signed_count_mse": round_err.square().mean().item(),
        "rounded_signed_count_mae": round_err.abs().mean().item(),
        "rounded_signed_count_acc": (rounded == true_long).float().mean().item(),
        "instance_acc": ((inst_scores_t >= 0.5).long() == inst_labels_t).float().mean().item(),
        "instance_auc": binary_auc(inst_scores_t, inst_labels_t),
        "zero_fraction": zero_mask.float().mean().item(),
    }
    if zero_mask.any():
        metrics["zero_rounded_signed_count_acc"] = (rounded[zero_mask] == 0).float().mean().item()
        metrics["zero_expected_abs_error"] = exp_counts_t[zero_mask].abs().mean().item()
    else:
        metrics["zero_rounded_signed_count_acc"] = math.nan
        metrics["zero_expected_abs_error"] = math.nan
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
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
    parser.add_argument("--results-dir", default="results/signed_mnist_baseline")
    args = parser.parse_args()

    cfg = {
        "dataset_root": "data",
        "dataset": "MNIST",
        "target_digit": 9,
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 1000,
        "cancellation_heavy": False,
        "objective": "mse",
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    if "bag_size" in cfg and "bag_size_mean" not in cfg:
        cfg["bag_size_mean"] = cfg["bag_size"]
    if "cancellation_split" in cfg and "cancellation_heavy" not in cfg:
        cfg["cancellation_heavy"] = cfg["cancellation_split"]
    for key, value in {
        "dataset_root": args.dataset_root,
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

    run_dir = make_run_dir(args.run_root, "signed_mnist_baseline", "mse", str(cfg["dataset"]), seed)
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

    best = {"expected_signed_count_mse": math.inf}
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
            signed_counts = batch["signed_count"].to(device).float()
            probs = model.predict_proba(x)
            pred = expected_signed_count(probs, signs, mask)
            loss = F.mse_loss(pred, signed_counts)

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
        metrics.update({"epoch": epoch, "train_mse": total_loss / max(total_seen, 1)})
        metrics.update({"epoch_seconds": time.perf_counter() - epoch_start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"epoch={epoch:03d} train_mse={metrics['train_mse']:.4f} "
            f"exp_mse={metrics['expected_signed_count_mse']:.4f} "
            f"round_acc={metrics['rounded_signed_count_acc']:.3f} "
            f"inst_auc={metrics['instance_auc']:.4f} "
            f"time={metrics['epoch_seconds']:.2f}s mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )
        if metrics["expected_signed_count_mse"] < best["expected_signed_count_mse"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": cfg, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"signed_mnist_expected_mse_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
