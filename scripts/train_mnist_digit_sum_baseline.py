#!/usr/bin/env python
"""Train simple expected-sum baselines for MNIST digit-sum bags."""

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

from countmil.aggregators import aggregate_nll, finite_support_convolution
from countmil.datasets import MNISTDigitSumBags, collate_mnist_bags
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


def expected_digit_sum(class_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.arange(class_probs.shape[-1], device=class_probs.device, dtype=class_probs.dtype)
    instance_means = (class_probs * values).sum(dim=-1)
    return (instance_means * mask.to(class_probs.dtype)).sum(dim=-1)


def digit_sum_pmf(class_probs: torch.Tensor, mask: torch.Tensor):
    zero_atom = class_probs.new_zeros(class_probs.shape[-1])
    zero_atom[0] = 1.0
    atoms = torch.where(mask.unsqueeze(-1), class_probs, zero_atom)
    return finite_support_convolution(atoms, support_min=0)


def expected_sum_loss(pred: torch.Tensor, target: torch.Tensor, objective: str) -> torch.Tensor:
    target_f = target.to(pred.dtype)
    if objective == "mse":
        return F.mse_loss(pred, target_f)
    if objective == "mae":
        return F.l1_loss(pred, target_f)
    if objective == "huber":
        return F.smooth_l1_loss(pred, target_f)
    raise ValueError(f"unknown objective: {objective}")


def _evaluate(model: MNISTDigitClassifier, loader: DataLoader, device: torch.device, objective: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_bags = 0
    exp_sums = []
    true_sums = []
    pred_digits = []
    true_digits = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            digits = batch["digits"].to(device)
            probs = model.predict_proba(x)
            pred = expected_digit_sum(probs, mask)
            pmf = digit_sum_pmf(probs, mask)
            loss = expected_sum_loss(pred, sums, objective)
            nll = aggregate_nll(pmf, sums)

            total_loss += loss.item() * x.shape[0]
            total_nll += nll.sum().item()
            total_bags += x.shape[0]
            exp_sums.append(pred.cpu())
            true_sums.append(sums.cpu())
            pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
            true_digits.append(digits[mask].cpu())

    exp_sums_t = torch.cat(exp_sums)
    true_sums_t = torch.cat(true_sums)
    rounded = exp_sums_t.round().long()
    pred_digits_t = torch.cat(pred_digits)
    true_digits_t = torch.cat(true_digits)
    return {
        "loss": total_loss / max(total_bags, 1),
        "diagnostic_nll": total_nll / max(total_bags, 1),
        "rounded_sum_acc": (rounded == true_sums_t).float().mean().item(),
        "rounded_sum_mae": (rounded.float() - true_sums_t.float()).abs().mean().item(),
        "expected_sum_mae": (exp_sums_t - true_sums_t.float()).abs().mean().item(),
        "instance_digit_acc": (pred_digits_t == true_digits_t).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--objective", choices=["mse", "mae", "huber"], default=None)
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
    parser.add_argument("--results-dir", default="results/mnist_digit_sum_baseline")
    args = parser.parse_args()

    cfg = {
        "dataset_root": "data",
        "dataset": "MNIST",
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 1000,
        "objective": "mse",
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    if "bag_size" in cfg and "bag_size_mean" not in cfg:
        cfg["bag_size_mean"] = cfg["bag_size"]
    for key, value in {
        "objective": args.objective,
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
    objective = str(cfg["objective"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(args.run_root, "mnist_digit_sum_baseline", objective, str(cfg["dataset"]), seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_ds = MNISTDigitSumBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="train",
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=seed,
    )
    test_ds = MNISTDigitSumBags(
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

    best = {"expected_sum_mae": math.inf}
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
            sums = batch["sum"].to(device)
            probs = model.predict_proba(x)
            pred = expected_digit_sum(probs, mask)
            loss = expected_sum_loss(pred, sums, objective)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_seen += x.shape[0]

        metrics = _evaluate(model, test_loader, device, objective)
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
            f"exp_mae={metrics['expected_sum_mae']:.3f} "
            f"round_mae={metrics['rounded_sum_mae']:.3f} "
            f"digit_acc={metrics['instance_digit_acc']:.4f}"
        )
        if metrics["expected_sum_mae"] < best["expected_sum_mae"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": cfg, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"mnist_digit_sum_expected_{objective}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
