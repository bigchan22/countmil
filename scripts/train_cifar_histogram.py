#!/usr/bin/env python
"""Train CIFAR histogram-supervision baselines.

Supports:
- `pvc`: per-class OVR count likelihood with FFT-tree convolution.
- `kl`/`mse`/`ce`: ordinary proportion matching baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution_fft_tree
from countmil.baselines.proportion_matching import multiclass_proportion_matching_loss
from countmil.datasets import CIFARHistogramBags, collate_cifar_bags
from countmil.models import CIFARSmallClassifier
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


def class_count_pmf_fft(class_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = class_probs.transpose(1, 2)
    atoms = torch.stack([1.0 - probs, probs], dim=-1)
    zero_atom = class_probs.new_tensor([1.0, 0.0])
    atoms = torch.where(mask[:, None, :, None], atoms, zero_atom)
    return finite_support_convolution_fft_tree(atoms, support_min=0).probs


def pvc_loss(class_probs: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pmfs = class_count_pmf_fft(class_probs, mask)
    losses = [aggregate_nll(AggregatePMF(pmfs[:, k], 0), counts[:, k]) for k in range(pmfs.shape[1])]
    return torch.stack(losses, dim=-1).mean()


def _evaluate(model: CIFARSmallClassifier, loader: DataLoader, device: torch.device, objective: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    count_mae = []
    prop_mae = []
    exact_hist = []
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            counts = batch["class_counts"].to(device)
            props = batch["class_proportions"].to(device)
            labels = batch["labels"].to(device)
            probs = model.predict_proba(x)
            if objective == "pvc":
                pmfs = class_count_pmf_fft(probs, mask)
                loss = torch.stack(
                    [aggregate_nll(AggregatePMF(pmfs[:, k], 0), counts[:, k]) for k in range(pmfs.shape[1])],
                    dim=-1,
                ).mean()
                pred_counts = pmfs.argmax(dim=-1)
                pred_props = pred_counts.to(probs.dtype) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
            else:
                loss = multiclass_proportion_matching_loss(probs, props, mask, loss=objective)
                pred_props = (probs * mask.unsqueeze(-1).to(probs.dtype)).sum(dim=1)
                pred_props = pred_props / mask.sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
                pred_counts = (pred_props * mask.sum(dim=1, keepdim=True).to(probs.dtype)).round().long()

            total_loss += float(loss.item()) * x.shape[0]
            total_bags += x.shape[0]
            count_mae.append((pred_counts.to(probs.dtype) - counts.to(probs.dtype)).abs().mean(dim=-1).cpu())
            prop_mae.append((pred_props - props).abs().mean(dim=-1).cpu())
            exact_hist.append((pred_counts == counts).all(dim=-1).float().cpu())
            pred_labels.append(probs.argmax(dim=-1)[mask].cpu())
            true_labels.append(labels[mask].cpu())

    count_mae_t = torch.cat(count_mae)
    prop_mae_t = torch.cat(prop_mae)
    exact_hist_t = torch.cat(exact_hist)
    pred_labels_t = torch.cat(pred_labels)
    true_labels_t = torch.cat(true_labels)
    return {
        "loss": total_loss / max(total_bags, 1),
        "hist_count_mae": count_mae_t.mean().item(),
        "hist_proportion_mae": prop_mae_t.mean().item(),
        "hist_exact_acc": exact_hist_t.mean().item(),
        "instance_acc": (pred_labels_t == true_labels_t).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--objective", choices=["pvc", "kl", "mse", "ce"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default=None)
    parser.add_argument("--label-level", choices=["coarse", "fine"], default=None)
    parser.add_argument("--bag-size-mean", type=int, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results-dir", default="results/cifar_histogram")
    args = parser.parse_args()

    cfg = {
        "dataset_root": "data",
        "dataset": "CIFAR100",
        "label_level": "coarse",
        "bag_size_mean": 50,
        "bag_size_std": 10.0,
        "train_bags": 5000,
        "objective": "pvc",
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    for key, value in {
        "objective": args.objective,
        "dataset_root": args.dataset_root,
        "dataset": args.dataset,
        "label_level": args.label_level,
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

    train_ds = CIFARHistogramBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="train",
        label_level=str(cfg["label_level"]),
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=seed,
    )
    test_ds = CIFARHistogramBags(
        root=cfg["dataset_root"],
        dataset=str(cfg["dataset"]),
        split="test",
        label_level=str(cfg["label_level"]),
        num_bags=int(args.test_bags),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=seed + 10_000,
    )
    num_classes = train_ds.num_classes
    run_dir = make_run_dir(args.run_root, "cifar_histogram", objective, f"{cfg['dataset']}_{cfg['label_level']}", seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size, "num_classes": num_classes}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cifar_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cifar_bags)
    model = CIFARSmallClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    best = {"instance_acc": -math.inf}
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
            counts = batch["class_counts"].to(device)
            props = batch["class_proportions"].to(device)
            probs = model.predict_proba(x)
            if objective == "pvc":
                loss = pvc_loss(probs, counts, mask)
            else:
                loss = multiclass_proportion_matching_loss(probs, props, mask, loss=objective)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
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
            f"test_loss={metrics['loss']:.4f} hist_mae={metrics['hist_count_mae']:.3f} "
            f"inst_acc={metrics['instance_acc']:.4f} time={metrics['epoch_seconds']:.2f}s "
            f"mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )
        if metrics["instance_acc"] > best["instance_acc"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": {**cfg, "num_classes": num_classes}, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"cifar_histogram_{objective}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
