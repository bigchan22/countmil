#!/usr/bin/env python
"""Train Attention/Gated-Attention MIL baselines on MNIST-Bags."""

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

from countmil.datasets import MNISTBags, collate_mnist_bags
from countmil.metrics import binary_auc
from countmil.models import AttentionMILMNIST
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


def _evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    bag_scores = []
    bag_labels = []
    inst_scores = []
    inst_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["bag_label"].float().to(device)
            hidden = batch["instance_labels"].to(device)
            logits, weights = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_bags += x.shape[0]
            bag_scores.append(torch.sigmoid(logits).cpu())
            bag_labels.append(labels.long().cpu())

            # Attention weights are a weak proxy for instance ranking. Rescale by
            # valid bag length so a uniform attention baseline is comparable.
            scores = weights * mask.float().sum(dim=1, keepdim=True).clamp_min(1)
            inst_scores.append(scores[mask].cpu())
            inst_labels.append(hidden[mask].cpu())

    bag_scores_t = torch.cat(bag_scores)
    bag_labels_t = torch.cat(bag_labels)
    inst_scores_t = torch.cat(inst_scores)
    inst_labels_t = torch.cat(inst_labels)
    return {
        "bag_bce": total_loss / max(total_bags, 1),
        "bag_acc": ((bag_scores_t >= 0.5).long() == bag_labels_t).float().mean().item(),
        "bag_auc": binary_auc(bag_scores_t, bag_labels_t),
        "instance_auc": binary_auc(inst_scores_t, inst_labels_t),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--target-digit", type=int, default=None)
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
    parser.add_argument("--results-dir", default="results/mnist_bags")
    args = parser.parse_args()

    cfg = {
        "method": "attention",
        "dataset_root": "data",
        "target_digit": 9,
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 500,
        "balanced_binary": True,
        "seed": 0,
        "gated": False,
    }
    cfg.update(_parse_simple_yaml(args.config))
    for key, value in {
        "dataset_root": args.dataset_root,
        "target_digit": args.target_digit,
        "bag_size_mean": args.bag_size_mean,
        "bag_size_std": args.bag_size_std,
        "train_bags": args.train_bags,
        "seed": args.seed,
    }.items():
        if value is not None:
            cfg[key] = value
    if args.gated:
        cfg["gated"] = True
        cfg["method"] = "gated_attention"

    seed = int(cfg["seed"])
    method = "gated_attention" if bool(cfg["gated"]) else "attention"
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(args.run_root, "mnist_bags", method, "MNIST", seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_ds = MNISTBags(
        root=cfg["dataset_root"],
        split="train",
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        balanced_binary=True,
        seed=seed,
    )
    test_ds = MNISTBags(
        root=cfg["dataset_root"],
        split="test",
        num_bags=int(args.test_bags),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        balanced_binary=True,
        seed=seed + 10_000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mnist_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)

    model = AttentionMILMNIST(gated=bool(cfg["gated"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    best = {"bag_auc": -math.inf}
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
            labels = batch["bag_label"].float().to(device)
            logits, _ = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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
        metrics.update({"epoch": epoch, "train_bce": total_loss / max(total_seen, 1)})
        metrics.update({"epoch_seconds": time.perf_counter() - epoch_start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"epoch={epoch:03d} train_bce={metrics['train_bce']:.4f} "
            f"bag_auc={metrics['bag_auc']:.4f} inst_auc={metrics['instance_auc']:.4f} "
            f"time={metrics['epoch_seconds']:.2f}s mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )
        if metrics["bag_auc"] > best["bag_auc"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": {**cfg, "method": method}, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"mnist_bags_{method}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
