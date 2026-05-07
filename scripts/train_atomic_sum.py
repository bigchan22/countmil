#!/usr/bin/env python
"""Train A1-style ordinal-sum bags with the main CountMIL aggregator.

This reproduces the core colleague PCA experiment inside the main code path:
instances emit categorical atomic PMFs over `{0,...,K-1}`, the bag PMF is the
finite-support convolution of those atoms, and training uses only the observed
scalar sum.
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
from countmil.datasets import (
    MNISTOrdinalSumBags,
    SVHNOrdinalSumBags,
    UltraMNISTOrdinalSumBags,
    collate_ordinal_sum_bags,
)
from countmil.metrics import expected_value_from_pmf, multiclass_ece_from_pmf
from countmil.models import MNISTDigitClassifier, PatchOrdinalClassifier, make_cifar_classifier
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
        elif value.lower() in {"none", "null"}:
            parsed = None
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


def ordinal_sum_pmf(class_probs: torch.Tensor, mask: torch.Tensor) -> AggregatePMF:
    """Return exact PMF for `sum_i z_i`, where `z_i in {0,...,K-1}`."""

    if class_probs.ndim != 3:
        raise ValueError("class_probs must have shape (B,N,K)")
    if mask.shape != class_probs.shape[:2]:
        raise ValueError("mask must have shape (B,N)")
    zero_atom = class_probs.new_zeros(class_probs.shape[-1])
    zero_atom[0] = 1.0
    atoms = torch.where(mask.unsqueeze(-1), class_probs, zero_atom)
    return finite_support_convolution(atoms, support_min=0)


def expected_sum_from_instance_probs(class_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return E[sum_i z_i] from per-instance ordinal PMFs."""

    support = torch.arange(class_probs.shape[-1], device=class_probs.device, dtype=class_probs.dtype)
    expected_instances = (class_probs * support).sum(dim=-1)
    return (expected_instances * mask.to(class_probs.dtype)).sum(dim=1)


def _tail_mean(values: list[float], tail: int = 5) -> float:
    if not values:
        return math.nan
    n = min(tail, len(values))
    return float(sum(values[-n:]) / n)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_bags = 0
    pmf_chunks = []
    target_chunks = []
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            labels = batch["labels"].to(device)
            probs = model.predict_proba(x)
            pmf = ordinal_sum_pmf(probs, mask)
            nll = aggregate_nll(pmf, sums)
            total_nll += nll.sum().item()
            total_bags += x.shape[0]
            pmf_chunks.append(pmf.probs.cpu())
            target_chunks.append(sums.cpu())
            pred_labels.append(probs.argmax(dim=-1)[mask].cpu())
            true_labels.append(labels[mask].cpu())

    max_width = max(chunk.shape[-1] for chunk in pmf_chunks)
    pmfs = torch.cat(
        [torch.nn.functional.pad(chunk, (0, max_width - chunk.shape[-1])) for chunk in pmf_chunks],
        dim=0,
    )
    targets = torch.cat(target_chunks)
    modes = pmfs.argmax(dim=-1)
    expected = expected_value_from_pmf(pmfs, 0)
    pred_labels_t = torch.cat(pred_labels)
    true_labels_t = torch.cat(true_labels)
    return {
        "nll": total_nll / max(total_bags, 1),
        "sum_acc": (modes == targets).float().mean().item(),
        "sum_mae": (modes.float() - targets.float()).abs().mean().item(),
        "expected_sum_mae": (expected - targets.float()).abs().mean().item(),
        "aggregate_ece": multiclass_ece_from_pmf(pmfs, targets),
        "instance_label_acc": (pred_labels_t == true_labels_t).float().mean().item(),
    }


def _build_dataset(cfg: dict[str, Any], split: str, seed: int):
    common = {
        "num_bags": int(cfg["train_bags"] if split == "train" else cfg["test_bags"]),
        "bag_size_mean": float(cfg["bag_size_mean"]),
        "bag_size_std": cfg["bag_size_std"],
        "bag_size_min": int(cfg["bag_size_min"]),
        "bag_size_max": int(cfg["bag_size_max"]),
        "per_class_cap": cfg["per_class_cap"],
        "noise_sigma": float(cfg["noise_sigma"]),
        "seed": seed,
    }
    experiment = str(cfg["experiment"])
    if experiment == "mnist_sum":
        return MNISTOrdinalSumBags(root=cfg["dataset_root"], dataset=cfg["dataset"], split=split, **common)
    if experiment == "svhn_sum":
        return SVHNOrdinalSumBags(
            root=cfg["dataset_root"],
            split=split,
            download=bool(cfg["download"]),
            train_split_fallback=bool(cfg["svhn_train_split_fallback"]),
            augment=bool(cfg["augment"]),
            **common,
        )
    if experiment == "ultramnist":
        common["bag_size_mean"] = (int(cfg["bag_size_min"]) + int(cfg["bag_size_max"])) / 2.0
        common["bag_size_std"] = None
        return UltraMNISTOrdinalSumBags(root=cfg["dataset_root"], dataset=cfg["dataset"], split=split, **common)
    raise ValueError(f"unknown experiment: {experiment}")


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_name = str(cfg["model"])
    if model_name == "mnist_cnn":
        return MNISTDigitClassifier(num_classes=int(cfg["num_classes"]))
    if model_name == "small_cnn":
        return make_cifar_classifier("small_cnn", num_classes=int(cfg["num_classes"]))
    if model_name == "resnet18":
        return make_cifar_classifier("resnet18", num_classes=int(cfg["num_classes"]), pretrained=bool(cfg["pretrained"]))
    if model_name == "patch_cnn":
        return PatchOrdinalClassifier(num_classes=int(cfg["num_classes"]))
    raise ValueError(f"unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--objective", choices=["pca", "mse"], default=None)
    parser.add_argument("--experiment", choices=["mnist_sum", "svhn_sum", "ultramnist"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--svhn-train-split-fallback", action="store_true")
    parser.add_argument("--model", choices=["mnist_cnn", "small_cnn", "resnet18", "patch_cnn"], default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--bag-size-mean", type=float, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--bag-size-min", type=int, default=None)
    parser.add_argument("--bag-size-max", type=int, default=None)
    parser.add_argument("--per-class-cap", type=int, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results-dir", default="results/atomic_sum")
    args = parser.parse_args()

    cfg: dict[str, Any] = {
        "experiment": "mnist_sum",
        "objective": "pca",
        "dataset_root": "data",
        "dataset": "MNIST",
        "download": False,
        "svhn_train_split_fallback": False,
        "model": "mnist_cnn",
        "pretrained": False,
        "augment": False,
        "num_classes": 10,
        "bag_size_mean": 10.0,
        "bag_size_std": 2.0,
        "bag_size_min": 5,
        "bag_size_max": 15,
        "per_class_cap": 100,
        "noise_sigma": 0.0,
        "train_bags": 1500,
        "test_bags": 600,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    for key, value in {
        "experiment": args.experiment,
        "objective": args.objective,
        "dataset_root": args.dataset_root,
        "dataset": args.dataset,
        "model": args.model,
        "bag_size_mean": args.bag_size_mean,
        "bag_size_std": args.bag_size_std,
        "bag_size_min": args.bag_size_min,
        "bag_size_max": args.bag_size_max,
        "per_class_cap": args.per_class_cap,
        "noise_sigma": args.noise_sigma,
        "train_bags": args.train_bags,
        "test_bags": args.test_bags,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }.items():
        if value is not None:
            cfg[key] = value
    if args.download:
        cfg["download"] = True
    if args.svhn_train_split_fallback:
        cfg["svhn_train_split_fallback"] = True
    if args.pretrained:
        cfg["pretrained"] = True
    if args.augment:
        cfg["augment"] = True

    if cfg["experiment"] == "svhn_sum":
        if args.model is None:
            cfg["model"] = "resnet18"
        if args.per_class_cap is None:
            cfg["per_class_cap"] = None
        cfg["lr"] = float(cfg["lr"] if args.lr is not None else 1e-4)
        cfg["weight_decay"] = float(cfg["weight_decay"] if args.weight_decay is not None else 5e-4)
    if cfg["experiment"] == "ultramnist":
        cfg["model"] = "patch_cnn" if args.model is None else cfg["model"]
        if args.bag_size_min is None:
            cfg["bag_size_min"] = 3
        if args.bag_size_max is None:
            cfg["bag_size_max"] = 5
        if args.train_bags is None:
            cfg["train_bags"] = 800
        if args.test_bags is None:
            cfg["test_bags"] = 300
        if args.lr is None:
            cfg["lr"] = 5e-4
        if args.per_class_cap is None:
            cfg["per_class_cap"] = None
    if cfg.get("per_class_cap") == "none":
        cfg["per_class_cap"] = None

    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = _build_dataset(cfg, "train", seed)
    test_ds = _build_dataset(cfg, "test", seed + 10_000)
    cfg["num_classes"] = int(train_ds.num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ordinal_sum_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ordinal_sum_bags)
    model = _build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    objective = str(cfg["objective"])
    run_dir = make_run_dir(args.run_root, "atomic_sum", objective, str(cfg["experiment"]), seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    history: dict[str, list[float]] = {"nll": [], "sum_acc": [], "expected_sum_mae": [], "aggregate_ece": [], "instance_label_acc": []}
    final: dict[str, float] = {}
    best = {"nll": math.inf, "expected_sum_mae": math.inf}
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
            if objective == "pca":
                pmf = ordinal_sum_pmf(probs, mask)
                loss = aggregate_nll(pmf, sums).mean()
            elif objective == "mse":
                expected_sum = expected_sum_from_instance_probs(probs, mask)
                loss = F.mse_loss(expected_sum, sums.to(probs.dtype))
            else:
                raise ValueError(f"unknown objective: {objective}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_seen += x.shape[0]

        metrics = _evaluate(model, test_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        metrics.update({"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)})
        if objective == "pca":
            metrics["train_nll"] = metrics["train_loss"]
        metrics.update({"epoch_seconds": time.perf_counter() - start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        for key in history:
            history[key].append(float(metrics[key]))
        final = metrics
        selection_value = metrics["nll"] if objective == "pca" else metrics["expected_sum_mae"]
        best_value = best["nll"] if objective == "pca" else best["expected_sum_mae"]
        if selection_value < best_value:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")
        print(
            f"epoch={epoch:03d} train_loss={metrics['train_loss']:.4f} "
            f"test_nll={metrics['nll']:.4f} acc={metrics['sum_acc']:.4f} "
            f"exp_mae={metrics['expected_sum_mae']:.3f} inst_acc={metrics['instance_label_acc']:.4f} "
            f"time={metrics['epoch_seconds']:.2f}s mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )

    tail5 = {f"tail5_{key}": _tail_mean(values) for key, values in history.items()}
    summary = {
        "best": best,
        "final": final,
        "tail5": tail5,
        "selection_metric": "test_nll" if objective == "pca" else "expected_sum_mae",
        "config": cfg,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_path = results_dir / f"{cfg['experiment']}_{objective}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
