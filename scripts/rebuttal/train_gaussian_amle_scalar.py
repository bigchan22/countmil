#!/usr/bin/env python
"""Train Gaussian-AMLE baselines for scalar aggregate MNIST tasks."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from countmil.baselines.gaussian_amle import (
    DEFAULT_GAUSSIAN_AMLE_EPS,
    categorical_gaussian_amle_loss,
    categorical_sum_moments,
    signed_bernoulli_gaussian_amle_loss,
    signed_bernoulli_sum_moments,
)
from countmil.datasets import MNISTDigitSumBags, SignedMNISTBags, collate_mnist_bags
from countmil.metrics import binary_auc
from countmil.models import MNISTDigitClassifier, ShuklaMNISTSelector
from countmil.training.run import git_commit
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


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    try:
        import torchvision

        metadata["torchvision_version"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover
        metadata["torchvision_version"] = f"unavailable: {exc}"
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        metadata["gpu_index"] = device.index
    return metadata


def _make_run_dir(root: str | Path, cfg: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    commit = git_commit()[:8]
    signs = "cancel" if cfg.get("cancellation_heavy") else "random"
    parts = [
        str(cfg["task"]),
        "gaussian_amle",
        f"n{cfg['bag_size_mean']}",
        f"train{cfg['train_bags']}",
        signs if cfg["task"] == "signed" else "sum",
        f"s{cfg['seed']}",
        f"lr{float(cfg['lr']):.0e}",
        f"eps{float(cfg['eps']):.0e}",
        commit,
        stamp,
    ]
    path = Path(root) / "_".join(parts)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _check_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")


def _build_loaders(cfg: dict[str, Any], batch_size: int):
    common = {
        "root": cfg["dataset_root"],
        "dataset": cfg["dataset"],
        "bag_size": int(cfg["bag_size_mean"]),
        "bag_size_std": float(cfg["bag_size_std"]),
    }
    if cfg["task"] == "digit_sum":
        cls = MNISTDigitSumBags
        extra: dict[str, Any] = {}
    elif cfg["task"] == "signed":
        cls = SignedMNISTBags
        extra = {
            "target_digit": int(cfg["target_digit"]),
            "cancellation_heavy": bool(cfg["cancellation_heavy"]),
        }
    else:
        raise ValueError("task must be digit_sum or signed")
    train_ds = cls(split="train", num_bags=int(cfg["train_bags"]), seed=int(cfg["seed"]), **common, **extra)
    val_ds = cls(split="test", num_bags=int(cfg["val_bags"]), seed=int(cfg["seed"]) + 20_000, **common, **extra)
    test_ds = cls(split="test", num_bags=int(cfg["test_bags"]), seed=int(cfg["seed"]) + 10_000, **common, **extra)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mnist_bags),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mnist_bags),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mnist_bags),
    )


def _evaluate_digit(model: MNISTDigitClassifier, loader: DataLoader, device: torch.device, eps: float, raw_path: Path | None):
    model.eval()
    support = torch.arange(10, device=device, dtype=torch.float32)
    rows = []
    losses = []
    expected = []
    truth = []
    pred_digits = []
    true_digits = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            digits = batch["digits"].to(device)
            probs = model.predict_proba(x)
            moments = categorical_sum_moments(probs, support, mask)
            loss = categorical_gaussian_amle_loss(probs, support, sums, mask, eps=eps)
            pred_sum = moments.mean.round().long()
            losses.append(loss.cpu())
            expected.append(moments.mean.cpu())
            truth.append(sums.cpu())
            pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
            true_digits.append(digits[mask].cpu())
            if raw_path is not None:
                for j in range(x.shape[0]):
                    rows.append(
                        {
                            "batch": batch_id,
                            "row": j,
                            "target": int(sums[j].detach().cpu()),
                            "expected": float(moments.mean[j].detach().cpu()),
                            "variance": float(moments.variance[j].detach().cpu()),
                            "rounded": int(pred_sum[j].detach().cpu()),
                            "loss": float(loss[j].detach().cpu()),
                        }
                    )
    exp_t = torch.cat(expected)
    true_t = torch.cat(truth)
    pred_digits_t = torch.cat(pred_digits)
    true_digits_t = torch.cat(true_digits)
    if raw_path is not None:
        with raw_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "gaussian_nll": torch.cat(losses).mean().item(),
        "expected_sum_mae": (exp_t - true_t.float()).abs().mean().item(),
        "rounded_sum_acc": (exp_t.round().long() == true_t).float().mean().item(),
        "rounded_sum_mae": (exp_t.round() - true_t.float()).abs().mean().item(),
        "instance_digit_acc": (pred_digits_t == true_digits_t).float().mean().item(),
    }


def _evaluate_signed(model: ShuklaMNISTSelector, loader: DataLoader, device: torch.device, eps: float, raw_path: Path | None):
    model.eval()
    rows = []
    losses = []
    expected = []
    truth = []
    scores = []
    labels = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            targets = batch["signed_count"].to(device)
            hidden = batch["instance_labels"].to(device)
            probs = model.predict_proba(x)
            moments = signed_bernoulli_sum_moments(probs, signs, mask)
            loss = signed_bernoulli_gaussian_amle_loss(probs, signs, targets, mask, eps=eps)
            pred = moments.mean.round().long()
            losses.append(loss.cpu())
            expected.append(moments.mean.cpu())
            truth.append(targets.cpu())
            scores.append(probs[mask].cpu())
            labels.append(hidden[mask].cpu())
            if raw_path is not None:
                for j in range(x.shape[0]):
                    rows.append(
                        {
                            "batch": batch_id,
                            "row": j,
                            "target": int(targets[j].detach().cpu()),
                            "expected": float(moments.mean[j].detach().cpu()),
                            "variance": float(moments.variance[j].detach().cpu()),
                            "rounded": int(pred[j].detach().cpu()),
                            "loss": float(loss[j].detach().cpu()),
                        }
                    )
    exp_t = torch.cat(expected)
    true_t = torch.cat(truth)
    scores_t = torch.cat(scores)
    labels_t = torch.cat(labels)
    zero = true_t == 0
    if raw_path is not None:
        with raw_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    out = {
        "gaussian_nll": torch.cat(losses).mean().item(),
        "expected_signed_count_mae": (exp_t - true_t.float()).abs().mean().item(),
        "expected_signed_count_mse": (exp_t - true_t.float()).square().mean().item(),
        "rounded_signed_count_acc": (exp_t.round().long() == true_t).float().mean().item(),
        "rounded_signed_count_mae": (exp_t.round() - true_t.float()).abs().mean().item(),
        "instance_acc": ((scores_t >= 0.5).long() == labels_t).float().mean().item(),
        "instance_auc": binary_auc(scores_t, labels_t),
        "zero_fraction": zero.float().mean().item(),
    }
    out["zero_rounded_signed_count_acc"] = (exp_t.round().long()[zero] == 0).float().mean().item() if zero.any() else math.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["digit_sum", "signed"], default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target-digit", type=int, default=None)
    parser.add_argument("--bag-size-mean", type=int, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--val-bags", type=int, default=1000)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--cancellation-heavy", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=DEFAULT_GAUSSIAN_AMLE_EPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="results/rebuttal/overnight_4090/runs")
    parser.add_argument("--results-dir", default="results/rebuttal/overnight_4090/scalar")
    args = parser.parse_args()

    cfg = {
        "task": "digit_sum",
        "method": "gaussian_amle",
        "dataset_root": "data",
        "dataset": "MNIST",
        "target_digit": 9,
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 1000,
        "val_bags": args.val_bags,
        "test_bags": args.test_bags,
        "cancellation_heavy": False,
        "seed": 0,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eps": args.eps,
    }
    cfg.update(_parse_simple_yaml(args.config))
    for key, value in {
        "task": args.task,
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
    cfg["val_bags"] = args.val_bags
    cfg["test_bags"] = args.test_bags
    cfg["lr"] = args.lr
    cfg["weight_decay"] = args.weight_decay
    cfg["eps"] = args.eps

    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    run_dir = _make_run_dir(args.run_root, cfg)
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, indent=2, sort_keys=True))
    (run_dir / "metadata.json").write_text(json.dumps(_runtime_metadata(device), indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    train_loader, val_loader, test_loader = _build_loaders(cfg, args.batch_size)
    if cfg["task"] == "digit_sum":
        model: torch.nn.Module = MNISTDigitClassifier(num_classes=10).to(device)
        support = torch.arange(10, device=device, dtype=torch.float32)
    else:
        model = ShuklaMNISTSelector().to(device)
        support = None
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]), betas=(0.9, 0.999))
    metrics_path = run_dir / "metrics.jsonl"
    best: dict[str, Any] = {"validation_loss": math.inf}

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            if cfg["task"] == "digit_sum":
                targets = batch["sum"].to(device)
                probs = model.predict_proba(x)
                loss = categorical_gaussian_amle_loss(probs, support, targets, mask, eps=float(cfg["eps"])).mean()
            else:
                targets = batch["signed_count"].to(device)
                signs = batch["signs"].to(device)
                probs = model.predict_proba(x)
                loss = signed_bernoulli_gaussian_amle_loss(probs, signs, targets, mask, eps=float(cfg["eps"])).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            _check_gradients(model)
            optimizer.step()
            total += float(loss.item()) * x.shape[0]
            seen += x.shape[0]

        if cfg["task"] == "digit_sum":
            val_metrics = _evaluate_digit(model, val_loader, device, float(cfg["eps"]), None)
        else:
            val_metrics = _evaluate_signed(model, val_loader, device, float(cfg["eps"]), None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        row = {
            "epoch": epoch,
            "train_gaussian_nll": total / max(seen, 1),
            "validation_loss": val_metrics["gaussian_nll"],
            "epoch_seconds": time.perf_counter() - start,
            "peak_cuda_mem_mb": peak_mb,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        if row["validation_loss"] < best["validation_loss"]:
            best = row
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": row}, run_dir / "checkpoint_best.pt")

    torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "checkpoint_final.pt")
    state = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(state["model"])
    if cfg["task"] == "digit_sum":
        test_metrics = _evaluate_digit(model, test_loader, device, float(cfg["eps"]), run_dir / "raw_test_predictions.jsonl")
    else:
        test_metrics = _evaluate_signed(model, test_loader, device, float(cfg["eps"]), run_dir / "raw_test_predictions.jsonl")
    summary = {
        "best": best,
        "test": test_metrics,
        "selection_metric": "validation Gaussian-AMLE loss",
        "config": {**cfg, "epochs": args.epochs, "batch_size": args.batch_size},
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    name_bits = [str(cfg["task"]), "gaussian_amle", f"n{cfg['bag_size_mean']}", f"train{cfg['train_bags']}"]
    if cfg["task"] == "signed":
        name_bits.append("cancel" if cfg["cancellation_heavy"] else "random")
    name_bits.append(f"lr{float(cfg['lr']):.0e}")
    name_bits.append(f"eps{float(cfg['eps']):.0e}")
    name_bits.append(f"s{seed}.json")
    out = result_dir / "_".join(name_bits)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
