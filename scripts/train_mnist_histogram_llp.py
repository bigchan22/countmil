#!/usr/bin/env python
"""Train multiclass MNIST LLP on observed bag digit histograms."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from countmil.baselines.proportion_matching import multiclass_proportion_matching_loss
from countmil.datasets import MNISTDigitHistogramBags, collate_mnist_bags
from countmil.models import MNISTDigitClassifier
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


def _evaluate(model: MNISTDigitClassifier, loader: DataLoader, device: torch.device, loss_name: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    count_mae = []
    prop_mae = []
    pred_digits = []
    true_digits = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            proportions = batch["digit_proportions"].to(device)
            counts = batch["digit_counts"].to(device)
            digits = batch["digits"].to(device)
            probs = model.predict_proba(x)
            loss = multiclass_proportion_matching_loss(probs, proportions, mask, loss=loss_name)
            pred_props = (probs * mask.unsqueeze(-1).to(probs.dtype)).sum(dim=1)
            pred_props = pred_props / mask.sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
            pred_counts = pred_props * mask.sum(dim=1, keepdim=True).to(probs.dtype)

            total_loss += loss.item() * x.shape[0]
            total_bags += x.shape[0]
            count_mae.append((pred_counts - counts.to(probs.dtype)).abs().mean(dim=-1).cpu())
            prop_mae.append((pred_props - proportions).abs().mean(dim=-1).cpu())
            pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
            true_digits.append(digits[mask].cpu())

    count_mae_t = torch.cat(count_mae)
    prop_mae_t = torch.cat(prop_mae)
    pred_digits_t = torch.cat(pred_digits)
    true_digits_t = torch.cat(true_digits)
    return {
        "loss": total_loss / max(total_bags, 1),
        "hist_count_mae": count_mae_t.mean().item(),
        "hist_proportion_mae": prop_mae_t.mean().item(),
        "instance_digit_acc": (pred_digits_t == true_digits_t).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--loss", choices=["kl", "mse", "ce"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bag-size-mean", type=int, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results-dir", default="results/mnist_histogram_llp")
    args = parser.parse_args()

    cfg = {
        "dataset_root": "data",
        "dataset": "MNIST",
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 1000,
        "loss": "kl",
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    if "bag_size" in cfg and "bag_size_mean" not in cfg:
        cfg["bag_size_mean"] = cfg["bag_size"]
    for key, value in {
        "loss": args.loss,
        "dataset_root": args.dataset_root,
        "dataset": args.dataset,
        "bag_size_mean": args.bag_size_mean,
        "bag_size_std": args.bag_size_std,
        "train_bags": args.train_bags,
        "seed": args.seed,
    }.items():
        if value is not None:
            cfg[key] = value

    seed = int(cfg["seed"])
    loss_name = str(cfg["loss"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(args.run_root, "mnist_histogram_llp", loss_name, str(cfg["dataset"]), seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_ds = MNISTDigitHistogramBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="train",
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=seed,
    )
    test_ds = MNISTDigitHistogramBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="test",
        num_bags=int(args.test_bags),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=seed + 10_000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mnist_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)

    model = MNISTDigitClassifier(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    best = {"hist_count_mae": math.inf}
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in train_loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            proportions = batch["digit_proportions"].to(device)
            probs = model.predict_proba(x)
            loss = multiclass_proportion_matching_loss(probs, proportions, mask, loss=loss_name)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_seen += x.shape[0]

        metrics = _evaluate(model, test_loader, device, loss_name)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)})
        metrics.update({"epoch_seconds": time.perf_counter() - start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={metrics['train_loss']:.4f} "
            f"hist_mae={metrics['hist_count_mae']:.3f} "
            f"prop_mae={metrics['hist_proportion_mae']:.4f} "
            f"digit_acc={metrics['instance_digit_acc']:.4f}"
        )
        if metrics["hist_count_mae"] < best["hist_count_mae"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": cfg, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"mnist_histogram_llp_{loss_name}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
