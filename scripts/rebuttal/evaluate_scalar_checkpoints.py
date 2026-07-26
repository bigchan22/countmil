#!/usr/bin/env python
"""Recompute scalar aggregate metrics from saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import stdev
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countmil.aggregators import aggregate_nll
from countmil.datasets import MNISTDigitSumBags, SignedMNISTBags, collate_mnist_bags
from countmil.metrics import binary_auc
from countmil.models import MNISTDigitClassifier, ShuklaMNISTSelector
from scripts.train_mnist_digit_sum import _expected_sum as fs_expected_sum
from scripts.train_mnist_digit_sum import digit_sum_pmf
from scripts.train_mnist_digit_sum_baseline import expected_digit_sum
from scripts.train_signed_mnist import _expected_value as signed_expected_value
from scripts.train_signed_mnist import signed_count_pmf
from scripts.train_signed_mnist_baseline import expected_signed_count


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _checkpoint(run_dir: str | Path) -> Path:
    p = Path(run_dir) / "checkpoint_best.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _digit_loader(cfg: dict[str, Any], test_bags: int = 1000) -> DataLoader:
    ds = MNISTDigitSumBags(
        root=cfg.get("dataset_root", "data"),
        dataset=cfg.get("dataset", "MNIST"),
        split="test",
        num_bags=test_bags,
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        seed=int(cfg["seed"]) + 10_000,
    )
    return DataLoader(ds, batch_size=192, shuffle=False, collate_fn=collate_mnist_bags)


def _signed_loader(cfg: dict[str, Any], test_bags: int = 1000) -> DataLoader:
    ds = SignedMNISTBags(
        root=cfg.get("dataset_root", "data"),
        dataset=cfg.get("dataset", "MNIST"),
        split="test",
        num_bags=test_bags,
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg.get("target_digit", 9)),
        cancellation_heavy=bool(cfg.get("cancellation_heavy", False)),
        seed=int(cfg["seed"]) + 10_000,
    )
    return DataLoader(ds, batch_size=192, shuffle=False, collate_fn=collate_mnist_bags)


def eval_digit_fsconv(summary: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = summary["config"]
    state = torch.load(_checkpoint(summary["run_dir"]), map_location=device)
    model = MNISTDigitClassifier(num_classes=10).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    rows = []
    nlls, mode_preds, rounded_preds, expected, true, pred_digits, true_digits = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch_id, batch in enumerate(_digit_loader(cfg)):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            digits = batch["digits"].to(device)
            probs = model.predict_proba(x)
            pmf = digit_sum_pmf(probs, mask)
            exp = fs_expected_sum(pmf)
            mode = pmf.probs.argmax(dim=-1) + pmf.support_min
            nll = aggregate_nll(pmf, sums)
            nlls.append(nll.cpu())
            mode_preds.append(mode.cpu())
            rounded_preds.append(exp.round().long().cpu())
            expected.append(exp.cpu())
            true.append(sums.cpu())
            pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
            true_digits.append(digits[mask].cpu())
    mode_t = torch.cat(mode_preds)
    rounded_t = torch.cat(rounded_preds)
    exp_t = torch.cat(expected)
    true_t = torch.cat(true)
    pred_digit_t = torch.cat(pred_digits)
    true_digit_t = torch.cat(true_digits)
    return {
        "task": "digit_sum",
        "method": "FS-Conv",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "expected_aggregate_mae": (exp_t - true_t.float()).abs().mean().item(),
        "mode_aggregate_acc": (mode_t == true_t).float().mean().item(),
        "mode_aggregate_mae": (mode_t.float() - true_t.float()).abs().mean().item(),
        "rounded_expected_acc": (rounded_t == true_t).float().mean().item(),
        "rounded_expected_mae": (rounded_t.float() - true_t.float()).abs().mean().item(),
        "instance_acc": (pred_digit_t == true_digit_t).float().mean().item(),
        "aggregate_nll": torch.cat(nlls).mean().item(),
        "source": summary["run_dir"],
    }


def eval_digit_mse(summary: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = summary["config"]
    state = torch.load(_checkpoint(summary["run_dir"]), map_location=device)
    model = MNISTDigitClassifier(num_classes=10).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    expected, true, pred_digits, true_digits = [], [], [], []
    with torch.no_grad():
        for batch in _digit_loader(cfg):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            sums = batch["sum"].to(device)
            digits = batch["digits"].to(device)
            probs = model.predict_proba(x)
            pred = expected_digit_sum(probs, mask)
            expected.append(pred.cpu())
            true.append(sums.cpu())
            pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
            true_digits.append(digits[mask].cpu())
    exp_t = torch.cat(expected)
    true_t = torch.cat(true)
    rounded_t = exp_t.round().long()
    pred_digit_t = torch.cat(pred_digits)
    true_digit_t = torch.cat(true_digits)
    return {
        "task": "digit_sum",
        "method": "MSE",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "expected_aggregate_mae": (exp_t - true_t.float()).abs().mean().item(),
        "mode_aggregate_acc": math.nan,
        "mode_aggregate_mae": math.nan,
        "rounded_expected_acc": (rounded_t == true_t).float().mean().item(),
        "rounded_expected_mae": (rounded_t.float() - true_t.float()).abs().mean().item(),
        "instance_acc": (pred_digit_t == true_digit_t).float().mean().item(),
        "aggregate_nll": math.nan,
        "source": summary["run_dir"],
    }


def eval_digit_gaussian(summary: dict[str, Any]) -> dict[str, Any]:
    cfg, test = summary["config"], summary["test"]
    return {
        "task": "digit_sum",
        "method": "Gaussian-AMLE",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "expected_aggregate_mae": test["expected_sum_mae"],
        "mode_aggregate_acc": test["rounded_sum_acc"],
        "mode_aggregate_mae": test["rounded_sum_mae"],
        "rounded_expected_acc": test["rounded_sum_acc"],
        "rounded_expected_mae": test["rounded_sum_mae"],
        "instance_acc": test["instance_digit_acc"],
        "aggregate_nll": math.nan,
        "gaussian_nll": test["gaussian_nll"],
        "source": summary["run_dir"],
    }


def eval_signed_fsconv(summary: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = summary["config"]
    state = torch.load(_checkpoint(summary["run_dir"]), map_location=device)
    model = ShuklaMNISTSelector().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    nlls, mode_preds, rounded_preds, expected, true, scores, labels = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in _signed_loader(cfg):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            targets = batch["signed_count"].to(device)
            hidden = batch["instance_labels"].to(device)
            probs = model.predict_proba(x)
            pmf = signed_count_pmf(probs, signs, mask)
            exp = signed_expected_value(pmf)
            mode = pmf.probs.argmax(dim=-1) + pmf.support_min
            nlls.append(aggregate_nll(pmf, targets).cpu())
            mode_preds.append(mode.cpu())
            rounded_preds.append(exp.round().long().cpu())
            expected.append(exp.cpu())
            true.append(targets.cpu())
            scores.append(probs[mask].cpu())
            labels.append(hidden[mask].cpu())
    mode_t = torch.cat(mode_preds)
    rounded_t = torch.cat(rounded_preds)
    exp_t = torch.cat(expected)
    true_t = torch.cat(true)
    scores_t = torch.cat(scores)
    labels_t = torch.cat(labels)
    return {
        "task": "signed",
        "method": "FS-Conv",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "cancellation_heavy": bool(cfg.get("cancellation_heavy", False)),
        "expected_aggregate_mae": (exp_t - true_t.float()).abs().mean().item(),
        "mode_aggregate_acc": (mode_t == true_t).float().mean().item(),
        "mode_aggregate_mae": (mode_t.float() - true_t.float()).abs().mean().item(),
        "rounded_expected_acc": (rounded_t == true_t).float().mean().item(),
        "rounded_expected_mae": (rounded_t.float() - true_t.float()).abs().mean().item(),
        "instance_acc": ((scores_t >= 0.5).long() == labels_t).float().mean().item(),
        "instance_auc": binary_auc(scores_t, labels_t),
        "aggregate_nll": torch.cat(nlls).mean().item(),
        "source": summary["run_dir"],
    }


def eval_signed_mse(summary: dict[str, Any]) -> dict[str, Any]:
    cfg, best = summary["config"], summary["best"]
    return {
        "task": "signed",
        "method": "MSE",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "cancellation_heavy": bool(cfg.get("cancellation_heavy", False)),
        "expected_aggregate_mae": best["expected_signed_count_mae"],
        "mode_aggregate_acc": best["rounded_signed_count_acc"],
        "mode_aggregate_mae": best["rounded_signed_count_mae"],
        "rounded_expected_acc": best["rounded_signed_count_acc"],
        "rounded_expected_mae": best["rounded_signed_count_mae"],
        "instance_acc": best["instance_acc"],
        "instance_auc": best["instance_auc"],
        "aggregate_nll": math.nan,
        "source": summary["run_dir"],
    }


def eval_signed_gaussian(summary: dict[str, Any]) -> dict[str, Any]:
    cfg, test = summary["config"], summary["test"]
    return {
        "task": "signed",
        "method": "Gaussian-AMLE",
        "seed": int(cfg["seed"]),
        "bag_size_mean": int(cfg["bag_size_mean"]),
        "train_bags": int(cfg["train_bags"]),
        "cancellation_heavy": bool(cfg.get("cancellation_heavy", False)),
        "expected_aggregate_mae": test["expected_signed_count_mae"],
        "mode_aggregate_acc": test["rounded_signed_count_acc"],
        "mode_aggregate_mae": test["rounded_signed_count_mae"],
        "rounded_expected_acc": test["rounded_signed_count_acc"],
        "rounded_expected_mae": test["rounded_signed_count_mae"],
        "instance_acc": test["instance_acc"],
        "instance_auc": test["instance_auc"],
        "aggregate_nll": math.nan,
        "gaussian_nll": test["gaussian_nll"],
        "source": summary["run_dir"],
    }


def _agg(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["task"], r["method"], r["bag_size_mean"], r["train_bags"], r.get("cancellation_heavy", ""))
        groups.setdefault(key, []).append(r)
    out = []
    metrics = [
        "expected_aggregate_mae",
        "mode_aggregate_acc",
        "mode_aggregate_mae",
        "rounded_expected_acc",
        "rounded_expected_mae",
        "instance_acc",
        "instance_auc",
        "aggregate_nll",
        "gaussian_nll",
    ]
    for key, vals in sorted(groups.items()):
        row = {
            "task": key[0],
            "method": key[1],
            "bag_size_mean": key[2],
            "train_bags": key[3],
            "cancellation_heavy": key[4],
            "seeds": len(vals),
        }
        for m in metrics:
            xs = [float(v[m]) for v in vals if m in v and math.isfinite(float(v[m]))]
            if xs:
                row[f"{m}_mean"] = sum(xs) / len(xs)
                row[f"{m}_sd"] = stdev(xs) if len(xs) > 1 else 0.0
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--gaussian-results-dir",
        default=None,
        help="Directory of Gaussian-AMLE summary JSONs. Defaults to selected-tuning runs if complete, else overnight runs.",
    )
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    agg = _read_csv("results/neurips_pilot_preleave_20260504_1619/aggregate.csv")
    for r in agg:
        if r.get("config.train_bags") != "1000" or r.get("config.bag_size_mean") != "10":
            continue
        summary = _load(r["source"])
        if r.get("config.experiment") == "mnist_digit_sum":
            rows.append(eval_digit_fsconv(summary, device))
        elif r.get("config.experiment") == "signed_mnist":
            rows.append(eval_signed_fsconv(summary, device))

    if args.gaussian_results_dir is not None:
        gaussian_dir = Path(args.gaussian_results_dir)
    else:
        selected = Path("results/rebuttal/gaussian_selected/scalar")
        gaussian_dir = selected if len(list(selected.glob("*.json"))) >= 9 else Path("results/rebuttal/overnight_4090/scalar")
    for p in sorted(gaussian_dir.glob("*.json")):
        summary = _load(p)
        if summary["config"]["task"] == "digit_sum":
            rows.append(eval_digit_gaussian(summary))
        elif summary["config"]["task"] == "signed":
            rows.append(eval_signed_gaussian(summary))

    for p in sorted(Path("results/rebuttal/mse_digit_sum_n10_train1000").glob("*.json")):
        rows.append(eval_digit_mse(_load(p), device))

    for p in sorted(Path("results/server4090_signed_mse_baseline").glob("signed_mse_n10_train1000_*/*.json")):
        rows.append(eval_signed_mse(_load(p)))

    _write_csv(Path("results/rebuttal/scalar_audit_per_seed.csv"), rows)
    _write_csv(Path("results/rebuttal/scalar_audit_summary.csv"), _agg(rows))
    print(f"wrote {len(rows)} per-seed rows")


if __name__ == "__main__":
    main()
