#!/usr/bin/env python
"""Strict-v3 fixed-bag CIFAR-10 runs on frozen ResNet-18 features."""

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
from torch import nn

from baselines.llp_pvc.official import OfficialWarmupCosineLrScheduler, official_count_loss, official_predict
from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution_fft_tree
from countmil.baselines.proportion_matching import multiclass_proportion_matching_loss
from countmil.datasets.cifar import load_cifar_family
from countmil.strict_protocol import STRICTV3_TAG, canonical_hash, sample_cifar_count_manifest, save_manifest, save_split_payload, stratified_train_val_split
from countmil.training.run import git_commit
from countmil.training.seed import set_seed
from scripts.rebuttal.train_fixed_cifar10_features import _extract_features


def _class_count_pmfs(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = probs.transpose(1, 2)
    atoms = torch.stack([1.0 - p, p], dim=-1)
    zero = probs.new_tensor([1.0, 0.0])
    atoms = torch.where(mask[:, None, :, None], atoms, zero)
    return finite_support_convolution_fft_tree(atoms, support_min=0).probs


def _count_loss_softmax(logits: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    pmfs = _class_count_pmfs(probs, mask)
    losses = [aggregate_nll(AggregatePMF(pmfs[:, c], 0), counts[:, c]) for c in range(pmfs.shape[1])]
    return torch.stack(losses, dim=-1).mean()


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    vals = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        vals.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
    return float(sum(vals) / len(vals))


def _iter_batches(features: torch.Tensor, labels: torch.Tensor, manifest: dict[str, Any], batch_size: int, shuffle: bool, seed: int):
    order = torch.arange(manifest["indices"].shape[0])
    if shuffle:
        order = order[torch.randperm(order.numel(), generator=torch.Generator().manual_seed(seed))]
    for start in range(0, order.numel(), batch_size):
        rows = order[start : start + batch_size]
        idx = manifest["indices"][rows].long()
        yield {
            "features": features[idx],
            "labels": labels[idx],
            "counts": manifest["counts"][rows].long(),
            "proportions": manifest["counts"][rows].float() / idx.shape[1],
            "mask": torch.ones_like(idx, dtype=torch.bool),
        }


def _evaluate_bags(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, manifest: dict[str, Any], args: argparse.Namespace, include_instances: bool, raw_path: Path | None) -> dict[str, float]:
    head.eval()
    device = next(head.parameters()).device
    count_mae = []
    prop_mae = []
    comp_nll = []
    rows = []
    with torch.no_grad():
        for batch_id, batch in enumerate(_iter_batches(features, labels, manifest, args.batch_size, False, args.seed)):
            x = batch["features"].to(device)
            counts = batch["counts"].to(device)
            props = batch["proportions"].to(device)
            mask = batch["mask"].to(device)
            logits = head(x)
            eval_probs = torch.softmax(logits, dim=-1)
            pmfs = _class_count_pmfs(eval_probs, mask)
            nll = torch.stack([aggregate_nll(AggregatePMF(pmfs[:, c], 0), counts[:, c]) for c in range(pmfs.shape[1])], dim=-1)
            pred_counts = pmfs.argmax(dim=-1)
            pred_props = pred_counts.float() / mask.sum(dim=1, keepdim=True).float()
            per_mae = (pred_counts.float() - counts.float()).abs().mean(dim=-1)
            per_nll = nll.mean(dim=-1)
            count_mae.append(per_mae.cpu())
            prop_mae.append((pred_props - props).abs().mean(dim=-1).cpu())
            comp_nll.append(per_nll.cpu())
            if raw_path is not None:
                for j in range(x.shape[0]):
                    rows.append({"batch": batch_id, "row": j, "count_mae": float(per_mae[j].cpu()), "composite_nll": float(per_nll[j].cpu())})
    out = {
        "hist_count_mae": torch.cat(count_mae).mean().item(),
        "hist_proportion_mae": torch.cat(prop_mae).mean().item(),
        "composite_count_nll": torch.cat(comp_nll).mean().item(),
    }
    if include_instances:
        unique = torch.unique(manifest["indices"].flatten()).long()
        with torch.no_grad():
            logits = head(features[unique].to(device))
            pred = logits.softmax(dim=-1).argmax(dim=-1).cpu()
        target = labels[unique].cpu()
        out["unique_image_instance_acc"] = (pred == target).float().mean().item()
        out["unique_image_macro_f1"] = _macro_f1(pred, target, args.num_classes)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def _load_features(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_root = Path(args.feature_root)
    train_path = feature_root / "cifar10_resnet18_imagenet_train_features.pt"
    test_path = feature_root / "cifar10_resnet18_imagenet_test_features.pt"
    if train_path.exists() and test_path.exists():
        return torch.load(train_path, map_location="cpu"), torch.load(test_path, map_location="cpu")
    return (
        _extract_features(Path(args.dataset_root), feature_root, "train", device, args.batch_size),
        _extract_features(Path(args.dataset_root), feature_root, "test", device, args.batch_size),
    )


def _prepare(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_feat, test_feat = _load_features(args, device)
    split_root = Path(args.manifest_root) / "splits"
    split_path = split_root / f"cifar10_train_holdout_seed{args.split_seed}.pt"
    if split_path.exists():
        split_payload = torch.load(split_path, map_location="cpu")
    else:
        splits = stratified_train_val_split(train_feat["labels"], val_fraction=0.10, seed=args.split_seed)
        split_payload = save_split_payload(
            split_path,
            train_feat["labels"],
            splits,
            args.num_classes,
            {"dataset": "CIFAR10", "protocol_tag": STRICTV3_TAG, "split_seed": args.split_seed, "train_images": 45000, "val_images": 5000},
        )
    if split_payload["train_indices"].numel() != 45_000 or split_payload["val_indices"].numel() != 5_000:
        raise RuntimeError("CIFAR strict split should be 45k/5k")
    manifest_root = Path(args.manifest_root) / f"seed{args.seed}_n{args.bag_size}_train{args.train_bags}_{STRICTV3_TAG}"
    pools = {
        "train": (train_feat["labels"], split_payload["train_indices"], args.seed),
        "val": (train_feat["labels"], split_payload["val_indices"], args.seed + 20_000),
        "test": (test_feat["labels"], torch.arange(test_feat["labels"].numel()), args.seed + 10_000),
    }
    manifests = {}
    for split, (labels, pool, seed) in pools.items():
        path = manifest_root / f"{split}_manifest.pt"
        if path.exists():
            manifest = torch.load(path, map_location="cpu")
        else:
            manifest = sample_cifar_count_manifest(labels, pool, split=split, seed=seed, num_bags=getattr(args, f"{split}_bags"), bag_size=args.bag_size, alpha=args.alpha, num_classes=args.num_classes)
            save_manifest(path, manifest)
        manifests[split] = manifest
    return train_feat, test_feat, split_payload, manifests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["ce_kl", "fsconv_count", "official_llp_pvc"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--feature-root", default="results/rebuttal/fixed_bag_cifar10/features")
    p.add_argument("--manifest-root", default="results/rebuttal/4090_strictv3/cifar_fixed/manifests")
    p.add_argument("--run-root", default="results/rebuttal/4090_strictv3/cifar_fixed/runs")
    p.add_argument("--summary-root", default="results/rebuttal/4090_strictv3/cifar_fixed")
    p.add_argument("--split-seed", type=int, default=271828)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-frac", type=float, default=0.08)
    p.add_argument("--warmup-lr", type=float, default=5e-5)
    p.add_argument("--official-eps", type=float, default=1e-30)
    p.add_argument("--bag-size", type=int, default=64)
    p.add_argument("--train-bags", type=int, default=250)
    p.add_argument("--val-bags", type=int, default=250)
    p.add_argument("--test-bags", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--num-classes", type=int, default=10)
    args = p.parse_args()
    if git_commit() == "unknown":
        raise RuntimeError("strict-v3 jobs require a Git SHA")
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_feat, test_feat, split_payload, manifests = _prepare(args, device)
    if train_feat["feature_hash"] == test_feat["feature_hash"]:
        raise RuntimeError("train and test feature hashes unexpectedly match")
    cfg = vars(args) | {
        "protocol_tag": STRICTV3_TAG,
        "git_commit": git_commit(),
        "feature_hash_train": train_feat["feature_hash"],
        "feature_hash_test": test_feat["feature_hash"],
        "label_hash_train": train_feat["label_hash"],
        "label_hash_test": test_feat["label_hash"],
        "split_hash": split_payload["split_hash"],
        "train_manifest_hash": manifests["train"]["manifest_hash"],
        "val_manifest_hash": manifests["val"]["manifest_hash"],
        "test_manifest_hash": manifests["test"]["manifest_hash"],
        "checkpoint_selection": "aggregate validation composite_count_nll primary, hist_count_mae tie-break",
        "validation_hidden_metrics_logged": False,
    }
    cfg_hash = canonical_hash({"config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool)) or v is None}})
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.run_root) / f"cifar10_{args.method}_s{args.seed}_{STRICTV3_TAG}_{cfg_hash[:12]}_{git_commit()[:8]}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg | {"config_hash": cfg_hash}, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    metadata = {
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    head = nn.Linear(train_feat["features"].shape[1], args.num_classes).to(device)
    if args.method == "official_llp_pvc":
        with torch.no_grad():
            head.bias.fill_(-math.log(args.num_classes - 1))
        opt = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        warmup_iter = max(1, int(round(float(args.epochs) * float(args.warmup_frac))))
        scheduler = OfficialWarmupCosineLrScheduler(
            opt,
            max_iter=args.epochs,
            warmup_iter=warmup_iter,
            warmup_ratio=float(args.warmup_lr) / float(args.lr),
            warmup="linear",
        )
    else:
        opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    best = {"validation_composite_count_nll": math.inf, "validation_hist_count_mae": math.inf}
    metrics_path = run_dir / "metrics.jsonl"
    expected_train_hash = manifests["train"]["manifest_hash"]
    train_features = train_feat["features"]
    train_labels = train_feat["labels"]
    test_features = test_feat["features"]
    test_labels = test_feat["labels"]
    for epoch in range(1, args.epochs + 1):
        if manifests["train"]["manifest_hash"] != expected_train_hash:
            raise RuntimeError("training manifest changed across epochs")
        head.train()
        total = 0.0
        seen = 0
        start = time.perf_counter()
        for batch in _iter_batches(train_features, train_labels, manifests["train"], args.batch_size, True, args.seed + epoch):
            x = batch["features"].to(device)
            counts = batch["counts"].to(device)
            props = batch["proportions"].to(device)
            mask = batch["mask"].to(device)
            logits = head(x)
            if args.method == "ce_kl":
                loss = multiclass_proportion_matching_loss(torch.softmax(logits, dim=-1), props, mask, loss="ce")
            elif args.method == "fsconv_count":
                loss = _count_loss_softmax(logits, counts, mask)
            else:
                loss = official_count_loss(logits, props, mask=mask, eps=args.official_eps)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CIFAR strict loss")
            opt.zero_grad()
            loss.backward()
            for name, param in head.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name}")
            opt.step()
            total += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        if scheduler is not None:
            scheduler.step(epoch)
        val = _evaluate_bags(head, train_features, train_labels, manifests["val"], args, include_instances=False, raw_path=None)
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_composite_count_nll": val["composite_count_nll"],
            "validation_hist_count_mae": val["hist_count_mae"],
            "lr": opt.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - start,
            **{f"val_{k}": v for k, v in val.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        candidate = (row["validation_composite_count_nll"], row["validation_hist_count_mae"])
        current = (best["validation_composite_count_nll"], best["validation_hist_count_mae"])
        if candidate < current:
            best = row
            torch.save({"model": head.state_dict(), "config": cfg, "best": best}, run_dir / "checkpoint_best.pt")
    torch.save({"model": head.state_dict(), "config": cfg}, run_dir / "checkpoint_final.pt")
    state = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    head.load_state_dict(state["model"])
    test = _evaluate_bags(head, test_features, test_labels, manifests["test"], args, include_instances=True, raw_path=run_dir / "raw_per_bag_test_metrics.jsonl")
    if args.method == "official_llp_pvc":
        # Explicitly exercise the official prediction rule on unique test images.
        unique = torch.unique(manifests["test"]["indices"].flatten()).long()
        with torch.no_grad():
            official_pred = official_predict(head(test_features[unique].to(device))).cpu()
        test["official_prediction_rule_unique_acc"] = (official_pred == test_labels[unique]).float().mean().item()
    summary = {
        "config": cfg | {"config_hash": cfg_hash},
        "best": best,
        "test": test,
        "selection_metric": "aggregate validation composite_count_nll primary; hist_count_mae tie-break; no hidden validation metrics",
        "split_stats": {"train": split_payload["train_stats"], "val": split_payload["val_stats"]},
        "train_bag_stats": manifests["train"]["stats"],
        "val_bag_stats": manifests["val"]["stats"],
        "test_bag_stats": manifests["test"]["stats"],
        "manifest_hashes": {k: manifests[k]["manifest_hash"] for k in ["train", "val", "test"]},
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cifar10_{args.method}_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "COMPLETED").write_text("completed\n")


if __name__ == "__main__":
    main()
