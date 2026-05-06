#!/usr/bin/env python
"""Eval-only squared-error metrics for existing CountMIL checkpoints.

This script is intentionally separate from the training scripts so paper tables
can distinguish logged MAE metrics from recomputed MSE metrics. It rebuilds the
deterministic test-bag sampler from the checkpoint config and writes one JSON
record per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countmil.datasets import (  # noqa: E402
    MNISTBags,
    MNISTDigitHistogramBags,
    MNISTDigitSumBags,
    SignedMNISTBags,
    collate_mnist_bags,
)
from countmil.models import MNISTDigitClassifier, ShuklaMNISTSelector  # noqa: E402
from scripts.train_mnist_bags import _count_pmf  # noqa: E402
from scripts.train_mnist_digit_sum import _expected_sum, digit_sum_pmf  # noqa: E402
from scripts.train_mnist_histogram_pvc import class_count_pmf  # noqa: E402
from scripts.train_signed_mnist import _expected_value, _mode_values, signed_count_pmf  # noqa: E402


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"{path} is not a supported CountMIL checkpoint")
    return checkpoint


def _task_from_checkpoint(path: Path, checkpoint: dict[str, Any], override: str) -> str:
    if override != "auto":
        return override
    cfg = checkpoint["config"]
    experiment = str(cfg.get("experiment", ""))
    metrics = checkpoint.get("metrics", {})
    path_s = str(path)
    if experiment == "mnist_digit_sum" or "sum_mae" in metrics:
        return "digit_sum"
    if experiment == "signed_mnist" or "signed_count_mae" in metrics:
        return "signed"
    if experiment == "mnist_histogram_llp" or "mnist_histogram_llp" in path_s:
        return "histogram_llp"
    if experiment == "mnist_histogram_pvc" or "mnist_histogram_pvc" in path_s:
        return "histogram_pvc"
    if experiment == "mnist_bags" or "count_mae" in metrics:
        return "binary_count"
    raise ValueError(f"could not infer task for {path}")


def _test_loader(cfg: dict[str, Any], dataset_cls: type, test_bags: int, batch_size: int, **kwargs: Any) -> DataLoader:
    dataset = dataset_cls(
        root=cfg.get("dataset_root", "data"),
        dataset=str(cfg.get("dataset", "MNIST")),
        split="test",
        num_bags=test_bags,
        bag_size=int(cfg.get("bag_size_mean", 10)),
        bag_size_std=float(cfg.get("bag_size_std", 0.0)),
        seed=int(cfg.get("seed", 0)) + 10_000,
        **kwargs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mnist_bags)


def _eval_binary(path: Path, checkpoint: dict[str, Any], device: torch.device, test_bags: int, batch_size: int) -> dict[str, Any]:
    cfg = checkpoint["config"]
    model = ShuklaMNISTSelector().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = _test_loader(
        cfg,
        MNISTBags,
        test_bags,
        batch_size,
        target_digit=int(cfg.get("target_digit", 9)),
        balanced_binary=bool(cfg.get("balanced_binary", False)),
    )
    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            counts = batch["count"].to(device)
            probs = model.predict_proba(x)
            pmf = _count_pmf(probs, mask, str(cfg.get("method", "conv")))
            pred = pmf.probs.argmax(dim=-1).to(counts.dtype)
            errors.append((pred.float() - counts.float()).cpu())
    err = torch.cat(errors)
    return {"checkpoint": str(path), "task": "binary_count", "count_mse": err.square().mean().item(), "count_mae": err.abs().mean().item()}


def _eval_digit_sum(path: Path, checkpoint: dict[str, Any], device: torch.device, test_bags: int, batch_size: int) -> dict[str, Any]:
    cfg = checkpoint["config"]
    model = MNISTDigitClassifier(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = _test_loader(cfg, MNISTDigitSumBags, test_bags, batch_size)
    mode_errors = []
    expected_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            pmf = digit_sum_pmf(model.predict_proba(x), mask)
            mode = pmf.probs.argmax(dim=-1).float()
            expected = _expected_sum(pmf)
            mode_errors.append((mode - sums.float()).cpu())
            expected_errors.append((expected - sums.float()).cpu())
    mode_err = torch.cat(mode_errors)
    exp_err = torch.cat(expected_errors)
    return {
        "checkpoint": str(path),
        "task": "digit_sum",
        "sum_mse": mode_err.square().mean().item(),
        "sum_mae": mode_err.abs().mean().item(),
        "expected_sum_mse": exp_err.square().mean().item(),
        "expected_sum_mae": exp_err.abs().mean().item(),
    }


def _eval_signed(path: Path, checkpoint: dict[str, Any], device: torch.device, test_bags: int, batch_size: int) -> dict[str, Any]:
    cfg = checkpoint["config"]
    model = ShuklaMNISTSelector().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = _test_loader(
        cfg,
        SignedMNISTBags,
        test_bags,
        batch_size,
        target_digit=int(cfg.get("target_digit", 9)),
        cancellation_heavy=bool(cfg.get("cancellation_heavy", False)),
    )
    mode_errors = []
    expected_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            counts = batch["signed_count"].to(device)
            pmf = signed_count_pmf(model.predict_proba(x), signs, mask)
            mode = _mode_values(pmf).float()
            expected = _expected_value(pmf)
            mode_errors.append((mode - counts.float()).cpu())
            expected_errors.append((expected - counts.float()).cpu())
    mode_err = torch.cat(mode_errors)
    exp_err = torch.cat(expected_errors)
    return {
        "checkpoint": str(path),
        "task": "signed",
        "signed_count_mse": mode_err.square().mean().item(),
        "signed_count_mae": mode_err.abs().mean().item(),
        "expected_signed_count_mse": exp_err.square().mean().item(),
        "expected_signed_count_mae": exp_err.abs().mean().item(),
    }


def _eval_histogram(
    path: Path, checkpoint: dict[str, Any], device: torch.device, test_bags: int, batch_size: int, task: str
) -> dict[str, Any]:
    cfg = checkpoint["config"]
    model = MNISTDigitClassifier(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = _test_loader(cfg, MNISTDigitHistogramBags, test_bags, batch_size)
    count_errors = []
    prop_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            counts = batch["digit_counts"].to(device)
            probs = model.predict_proba(x)
            bag_sizes = mask.sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
            if task == "histogram_pvc":
                pred_counts = class_count_pmf(probs, mask).argmax(dim=-1).to(probs.dtype)
            else:
                pred_props = (probs * mask.unsqueeze(-1).to(probs.dtype)).sum(dim=1) / bag_sizes
                pred_counts = pred_props * bag_sizes
            target_counts = counts.to(probs.dtype)
            pred_props = pred_counts / bag_sizes
            target_props = target_counts / bag_sizes
            count_errors.append((pred_counts - target_counts).cpu())
            prop_errors.append((pred_props - target_props).cpu())
    count_err = torch.cat(count_errors)
    prop_err = torch.cat(prop_errors)
    return {
        "checkpoint": str(path),
        "task": task,
        "hist_count_mse": count_err.square().mean().item(),
        "hist_count_mae": count_err.abs().mean().item(),
        "hist_proportion_mse": prop_err.square().mean().item(),
        "hist_proportion_mae": prop_err.abs().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--task", choices=["auto", "binary_count", "digit_sum", "signed", "histogram_llp", "histogram_pvc"], default="auto")
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = []
    for path in args.checkpoints:
        checkpoint = _load_checkpoint(path, device)
        task = _task_from_checkpoint(path, checkpoint, args.task)
        if task == "binary_count":
            record = _eval_binary(path, checkpoint, device, args.test_bags, args.batch_size)
        elif task == "digit_sum":
            record = _eval_digit_sum(path, checkpoint, device, args.test_bags, args.batch_size)
        elif task == "signed":
            record = _eval_signed(path, checkpoint, device, args.test_bags, args.batch_size)
        elif task in {"histogram_llp", "histogram_pvc"}:
            record = _eval_histogram(path, checkpoint, device, args.test_bags, args.batch_size, task)
        else:
            raise ValueError(f"unsupported task: {task}")
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")


if __name__ == "__main__":
    main()
