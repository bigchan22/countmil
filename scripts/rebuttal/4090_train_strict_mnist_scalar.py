#!/usr/bin/env python
"""Strict-v3 MNIST scalar rebuttal runs with disjoint validation/test pools."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Dataset

from countmil.aggregators import aggregate_nll
from countmil.baselines.gaussian_amle import (
    DEFAULT_GAUSSIAN_AMLE_EPS,
    categorical_gaussian_amle_loss,
    categorical_sum_moments,
    gaussian_integer_bin_nll,
    signed_bernoulli_gaussian_amle_loss,
    signed_bernoulli_sum_moments,
)
from countmil.datasets import collate_mnist_bags, load_mnist_family
from countmil.masked_forward import forward_valid_instances
from countmil.metrics import binary_auc
from countmil.models import MNISTDigitClassifier, ShuklaMNISTSelector
from countmil.strict_protocol import (
    STRICTV3_TAG,
    BagManifestSpec,
    canonical_hash,
    sample_variable_mnist_manifest,
    save_manifest,
    save_split_payload,
    stratified_train_val_split,
)
from countmil.training.run import git_commit
from countmil.training.seed import set_seed
from scripts.train_mnist_digit_sum import digit_sum_pmf
from scripts.train_signed_mnist import _expected_value as signed_expected_value
from scripts.train_signed_mnist import _mode_values as signed_mode_values
from scripts.train_signed_mnist import signed_count_pmf


class ManifestMNISTBags(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, manifest: dict[str, Any], task: str) -> None:
        self.images = images
        self.labels = labels
        self.manifest = manifest
        self.task = task

    def __len__(self) -> int:
        return int(self.manifest["indices"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mask = self.manifest["mask"][idx].bool()
        indices = self.manifest["indices"][idx][mask].long()
        digits = self.labels[indices]
        out: dict[str, torch.Tensor] = {"instances": self.images[indices], "digits": digits}
        if self.task == "digit_sum":
            out["sum"] = digits.sum().long()
        else:
            signs = self.manifest["signs"][idx][mask].long()
            hidden = (digits == 9).long()
            out["signs"] = signs
            out["instance_labels"] = hidden
            out["signed_count"] = (hidden * signs).sum().long()
        return out


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    meta = {
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "protocol_tag": STRICTV3_TAG,
    }
    if device.type == "cuda":
        meta["gpu_name"] = torch.cuda.get_device_name(device)
        meta["gpu_index"] = device.index
    return meta


def _prepare_manifests(args: argparse.Namespace, task_key: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_images, train_labels = load_mnist_family(args.dataset_root, "MNIST", "train")
    test_images, test_labels = load_mnist_family(args.dataset_root, "MNIST", "test")
    split_dir = Path(args.manifest_root) / "splits"
    split_path = split_dir / f"mnist_split_seed{args.split_seed}.pt"
    if split_path.exists():
        split_payload = torch.load(split_path, map_location="cpu")
    else:
        splits = stratified_train_val_split(train_labels, val_fraction=0.15, seed=args.split_seed)
        split_payload = save_split_payload(
            split_path,
            train_labels,
            splits,
            10,
            {"dataset": "MNIST", "protocol_tag": STRICTV3_TAG, "split_seed": args.split_seed, "val_fraction": 0.15},
        )
    train_idx = split_payload["train_indices"]
    val_idx = split_payload["val_indices"]
    if torch.isin(train_idx, val_idx).any():
        raise RuntimeError("MNIST train/validation image split overlap")

    run_manifest_dir = Path(args.manifest_root) / task_key / f"seed{args.seed}_n{args.bag_size_mean}_train{args.train_bags}_{STRICTV3_TAG}"
    run_manifest_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "train": BagManifestSpec("train", args.seed, args.train_bags, args.bag_size_mean, args.bag_size_std, task_key, cancellation_heavy=args.cancellation_heavy),
        "val": BagManifestSpec("val", args.seed + 20_000, args.val_bags, args.bag_size_mean, args.bag_size_std, task_key, cancellation_heavy=args.cancellation_heavy),
        "test": BagManifestSpec("test", args.seed + 10_000, args.test_bags, args.bag_size_mean, args.bag_size_std, task_key, cancellation_heavy=args.cancellation_heavy),
    }
    manifests: dict[str, Any] = {}
    pools = {"train": train_idx, "val": val_idx, "test": torch.arange(test_labels.numel())}
    labels_by_split = {"train": train_labels, "val": train_labels, "test": test_labels}
    for split, spec in specs.items():
        path = run_manifest_dir / f"{split}_manifest.pt"
        if path.exists():
            manifest = torch.load(path, map_location="cpu")
        else:
            manifest = sample_variable_mnist_manifest(labels_by_split[split], pools[split], spec)
            save_manifest(path, manifest)
        manifests[split] = manifest
    return train_images, train_labels, test_images, test_labels, split_payload, manifests


def _digit_eval(model: MNISTDigitClassifier, loader: DataLoader, device: torch.device, include_hidden: bool, raw_path: Path | None) -> dict[str, float]:
    model.eval()
    support = torch.arange(10, device=device, dtype=torch.float32)
    rows = []
    exp_vals = []
    rounded_vals = []
    mode_vals = []
    targets = []
    nlls = []
    bin_nlls = []
    pred_digits = []
    true_digits = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            y = batch["sum"].to(device)
            logits = forward_valid_instances(model, x, mask)
            probs = torch.softmax(logits, dim=-1)
            pmf = digit_sum_pmf(probs, mask)
            values = torch.arange(pmf.support_min, pmf.support_max + 1, device=device, dtype=probs.dtype)
            expected = (pmf.probs * values).sum(dim=-1)
            mode = pmf.probs.argmax(dim=-1) + pmf.support_min
            moments = categorical_sum_moments(probs, support, mask)
            nll = aggregate_nll(pmf, y)
            bin_nll = gaussian_integer_bin_nll(moments.mean, moments.variance, y)
            exp_vals.append(expected.cpu())
            rounded_vals.append(expected.round().long().cpu())
            mode_vals.append(mode.cpu())
            targets.append(y.cpu())
            nlls.append(nll.cpu())
            bin_nlls.append(bin_nll.cpu())
            if include_hidden:
                digits = batch["digits"].to(device)
                pred_digits.append(probs.argmax(dim=-1)[mask].cpu())
                true_digits.append(digits[mask].cpu())
            if raw_path is not None:
                for j in range(y.numel()):
                    rows.append({"batch": batch_id, "row": j, "target": int(y[j].cpu()), "expected": float(expected[j].cpu()), "rounded": int(expected[j].round().cpu()), "mode": int(mode[j].cpu()), "nll": float(nll[j].cpu())})
    exp_t = torch.cat(exp_vals)
    rounded_t = torch.cat(rounded_vals)
    mode_t = torch.cat(mode_vals)
    target_t = torch.cat(targets)
    out = {
        "expected_sum_mae": (exp_t - target_t.float()).abs().mean().item(),
        "rounded_expected_sum_acc": (rounded_t == target_t).float().mean().item(),
        "pmf_mode_sum_acc": (mode_t == target_t).float().mean().item(),
        "posthoc_discrete_aggregate_nll": torch.cat(nlls).mean().item(),
        "posthoc_gaussian_integer_bin_nll": torch.cat(bin_nlls).mean().item(),
    }
    if include_hidden:
        out["instance_digit_acc"] = (torch.cat(pred_digits) == torch.cat(true_digits)).float().mean().item()
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def _signed_eval(model: ShuklaMNISTSelector, loader: DataLoader, device: torch.device, include_hidden: bool, raw_path: Path | None) -> dict[str, float]:
    model.eval()
    rows = []
    expected_vals = []
    rounded_vals = []
    mode_vals = []
    targets = []
    nlls = []
    bin_nlls = []
    scores = []
    labels = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            signs = batch["signs"].to(device)
            y = batch["signed_count"].to(device)
            logits = forward_valid_instances(model, x, mask)
            probs = torch.sigmoid(logits)
            pmf = signed_count_pmf(probs, signs, mask)
            expected = signed_expected_value(pmf)
            mode = signed_mode_values(pmf)
            moments = signed_bernoulli_sum_moments(probs, signs, mask)
            nll = aggregate_nll(pmf, y)
            bin_nll = gaussian_integer_bin_nll(moments.mean, moments.variance, y)
            expected_vals.append(expected.cpu())
            rounded_vals.append(expected.round().long().cpu())
            mode_vals.append(mode.cpu())
            targets.append(y.cpu())
            nlls.append(nll.cpu())
            bin_nlls.append(bin_nll.cpu())
            if include_hidden:
                hidden = batch["instance_labels"].to(device)
                scores.append(probs[mask].cpu())
                labels.append(hidden[mask].cpu())
            if raw_path is not None:
                for j in range(y.numel()):
                    rows.append({"batch": batch_id, "row": j, "target": int(y[j].cpu()), "expected": float(expected[j].cpu()), "rounded": int(expected[j].round().cpu()), "mode": int(mode[j].cpu()), "nll": float(nll[j].cpu())})
    exp_t = torch.cat(expected_vals)
    rounded_t = torch.cat(rounded_vals)
    mode_t = torch.cat(mode_vals)
    target_t = torch.cat(targets)
    out = {
        "expected_signed_sum_mae": (exp_t - target_t.float()).abs().mean().item(),
        "rounded_signed_sum_mae": (rounded_t.float() - target_t.float()).abs().mean().item(),
        "rounded_signed_sum_acc": (rounded_t == target_t).float().mean().item(),
        "pmf_mode_signed_sum_acc": (mode_t == target_t).float().mean().item(),
        "posthoc_discrete_aggregate_nll": torch.cat(nlls).mean().item(),
        "posthoc_gaussian_integer_bin_nll": torch.cat(bin_nlls).mean().item(),
    }
    if include_hidden:
        score_t = torch.cat(scores)
        label_t = torch.cat(labels)
        out["instance_acc"] = ((score_t >= 0.5).long() == label_t).float().mean().item()
        out["instance_auc"] = binary_auc(score_t, label_t)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def _train_loss(args: argparse.Namespace, model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    x = batch["instances"].to(device)
    mask = batch["mask"].to(device)
    logits = forward_valid_instances(model, x, mask)
    if args.task == "digit_sum":
        y = batch["sum"].to(device)
        probs = torch.softmax(logits, dim=-1)
        values = torch.arange(10, device=device, dtype=probs.dtype)
        if args.method == "fsconv":
            return aggregate_nll(digit_sum_pmf(probs, mask), y).mean()
        expected = (probs * values).sum(dim=-1)
        expected = (expected * mask.to(expected.dtype)).sum(dim=1)
        if args.method == "mse":
            return (expected - y.float()).square().mean()
        return categorical_gaussian_amle_loss(probs, values, y, mask, eps=args.gaussian_eps).mean()
    signs = batch["signs"].to(device)
    y = batch["signed_count"].to(device)
    probs = torch.sigmoid(logits)
    if args.method == "fsconv":
        return aggregate_nll(signed_count_pmf(probs, signs, mask), y).mean()
    moments = signed_bernoulli_sum_moments(probs, signs, mask)
    if args.method == "mse":
        return (moments.mean - y.float()).square().mean()
    return signed_bernoulli_gaussian_amle_loss(probs, signs, y, mask, eps=args.gaussian_eps).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["digit_sum", "signed_random", "signed_cancellation"], required=True)
    p.add_argument("--method", choices=["mse", "gaussian_amle", "fsconv"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--manifest-root", default="results/rebuttal/4090_strictv3/manifests/mnist")
    p.add_argument("--run-root", default="results/rebuttal/4090_strictv3")
    p.add_argument("--split-seed", type=int, default=314159)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bag-size-mean", type=int, default=10)
    p.add_argument("--bag-size-std", type=float, default=2.0)
    p.add_argument("--train-bags", type=int, default=1000)
    p.add_argument("--val-bags", type=int, default=1000)
    p.add_argument("--test-bags", type=int, default=1000)
    p.add_argument("--gaussian-eps", type=float, default=DEFAULT_GAUSSIAN_AMLE_EPS)
    args = p.parse_args()
    args.cancellation_heavy = args.task == "signed_cancellation"
    task_key = "digit_sum" if args.task == "digit_sum" else ("signed_cancellation" if args.cancellation_heavy else "signed_random")

    if git_commit() == "unknown":
        raise RuntimeError("strict-v3 jobs require a Git SHA")
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_images, train_labels, test_images, test_labels, split_payload, manifests = _prepare_manifests(args, task_key)
    train_ds = ManifestMNISTBags(train_images, train_labels, manifests["train"], "digit_sum" if args.task == "digit_sum" else "signed")
    val_ds = ManifestMNISTBags(train_images, train_labels, manifests["val"], "digit_sum" if args.task == "digit_sum" else "signed")
    test_ds = ManifestMNISTBags(test_images, test_labels, manifests["test"], "digit_sum" if args.task == "digit_sum" else "signed")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mnist_bags)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)
    model: torch.nn.Module = MNISTDigitClassifier(10).to(device) if args.task == "digit_sum" else ShuklaMNISTSelector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = vars(args) | {
        "protocol_tag": STRICTV3_TAG,
        "git_commit": git_commit(),
        "split_hash": split_payload["split_hash"],
        "train_manifest_hash": manifests["train"]["manifest_hash"],
        "val_manifest_hash": manifests["val"]["manifest_hash"],
        "test_manifest_hash": manifests["test"]["manifest_hash"],
        "validation_hidden_metrics_logged": False,
        "checkpoint_selection": "aggregate validation training-objective loss",
    }
    cfg_hash = canonical_hash({"config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool)) or v is None}})
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.run_root) / task_key / f"{task_key}_{args.method}_s{args.seed}_{STRICTV3_TAG}_{cfg_hash[:12]}_{git_commit()[:8]}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg | {"config_hash": cfg_hash}, indent=2, sort_keys=True))
    (run_dir / "metadata.json").write_text(json.dumps(_runtime_metadata(device), indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    expected_manifest_hash = manifests["train"]["manifest_hash"]
    best = {"validation_loss": math.inf}
    metrics_path = run_dir / "metrics.jsonl"
    eval_fn = _digit_eval if args.task == "digit_sum" else _signed_eval
    for epoch in range(1, args.epochs + 1):
        if manifests["train"]["manifest_hash"] != expected_manifest_hash:
            raise RuntimeError("training manifest changed across epochs")
        model.train()
        total = 0.0
        seen = 0
        start = time.perf_counter()
        for batch in train_loader:
            loss = _train_loss(args, model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite strict MNIST loss")
            opt.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name}")
            opt.step()
            total += float(loss.item()) * batch["instances"].shape[0]
            seen += batch["instances"].shape[0]
        val = eval_fn(model, val_loader, device, False, None)
        validation_loss = val["posthoc_discrete_aggregate_nll"] if args.method == "fsconv" else (
            val["posthoc_gaussian_integer_bin_nll"] if args.method == "gaussian_amle" else (
                val["expected_sum_mae"] if args.task == "digit_sum" else val["expected_signed_sum_mae"]
            )
        )
        row = {"epoch": epoch, "train_loss": total / max(seen, 1), "validation_loss": validation_loss, "epoch_seconds": time.perf_counter() - start, **{f"val_{k}": v for k, v in val.items()}}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        if validation_loss < best["validation_loss"]:
            best = row
            torch.save({"model": model.state_dict(), "config": cfg, "best": best}, run_dir / "checkpoint_best.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "checkpoint_final.pt")
    state = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(state["model"])
    test = eval_fn(model, test_loader, device, True, run_dir / "raw_test_predictions.jsonl")
    summary = {
        "config": cfg | {"config_hash": cfg_hash},
        "best": best,
        "test": test,
        "selection_metric": "aggregate validation metric only; no hidden validation metrics computed",
        "split_stats": {"train": split_payload["train_stats"], "val": split_payload["val_stats"]},
        "manifest_hashes": {k: manifests[k]["manifest_hash"] for k in ["train", "val", "test"]},
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    out = Path(args.run_root) / task_key / f"{task_key}_{args.method}_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "COMPLETED").write_text("completed\n")


if __name__ == "__main__":
    main()
