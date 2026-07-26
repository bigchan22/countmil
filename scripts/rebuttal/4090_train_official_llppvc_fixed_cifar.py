#!/usr/bin/env python
"""Official LLP-PVC adapter on fixed-bag frozen CIFAR-10 features."""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.llp_pvc import (  # noqa: E402
    OfficialWarmupCosineLrScheduler,
    init_sigmoid_bias_to_one_over_k,
    official_count_loss,
    official_predict,
)
from countmil.training.run import git_commit  # noqa: E402
from countmil.training.seed import set_seed  # noqa: E402
from scripts.rebuttal.train_fixed_cifar10_features import (  # noqa: E402
    _evaluate,
    _manifest_hash,
    _sha256_tensor,
)


UPSTREAM_DIR = Path("third_party/ICLR2026_LLP-PVC")


def _upstream_sha() -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"], text=True).strip()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def _load_features(feature_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train = torch.load(feature_root / "cifar10_resnet18_imagenet_train_features.pt", map_location="cpu")
    test = torch.load(feature_root / "cifar10_resnet18_imagenet_test_features.pt", map_location="cpu")
    if train["feature_hash"] == test["feature_hash"] or train["label_hash"] == test["label_hash"]:
        raise RuntimeError("train/test feature or label hashes unexpectedly match")
    return train, test


def _load_manifests(manifest_root: Path, seed: int, bag_size: int, train_bags: int, alpha: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = manifest_root / f"seed{seed}_n{bag_size}_train{train_bags}_alpha{alpha}"
    out = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_bags.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        stats_path = root / f"{split}_bag_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        payload["path"] = str(path)
        payload["stats"] = stats
        observed = _manifest_hash(payload)
        expected = stats.get("manifest_sha256")
        if expected and observed != expected:
            raise RuntimeError(f"{split} manifest hash mismatch: {observed} != {expected}")
        out.append(payload)
    return tuple(out)  # type: ignore[return-value]


def iter_training_batches(features: torch.Tensor, bags: dict[str, torch.Tensor], batch_size: int, seed: int):
    """Yield training batches without hidden instance labels."""

    order = torch.randperm(bags["indices"].shape[0], generator=torch.Generator().manual_seed(seed))
    for start in range(0, order.numel(), batch_size):
        idx = order[start : start + batch_size]
        bag_idx = bags["indices"][idx]
        counts = bags["counts"][idx]
        yield {
            "features": features[bag_idx],
            "counts": counts,
            "proportions": counts.float() / bag_idx.shape[1],
            "mask": torch.ones_like(bag_idx, dtype=torch.bool),
        }


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    vals = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        vals.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
    return float(sum(vals) / len(vals))


def _write_unique_predictions(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    bags: dict[str, torch.Tensor],
    path: Path,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    unique = torch.unique(bags["indices"]).cpu()
    preds = []
    truth = []
    head.eval()
    with torch.no_grad(), path.open("w") as f:
        for start in range(0, unique.numel(), 4096):
            idx = unique[start : start + 4096]
            logits = head(features[idx].to(device))
            pred = official_predict(logits).cpu()
            y = labels[idx].cpu()
            preds.append(pred)
            truth.append(y)
            probs = torch.softmax(logits, dim=-1).cpu()
            for image_idx, target, cls, conf in zip(idx.tolist(), y.tolist(), pred.tolist(), probs.max(dim=-1).values.tolist()):
                f.write(json.dumps({"image_index": image_idx, "target": target, "prediction": cls, "confidence": conf}, sort_keys=True) + "\n")
    pred_t = torch.cat(preds)
    true_t = torch.cat(truth)
    return {
        "unique_image_instance_acc": (pred_t == true_t).float().mean().item(),
        "unique_image_macro_f1": _macro_f1(pred_t, true_t, num_classes),
        "num_unique_test_images": int(unique.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-root", default="results/rebuttal/fixed_bag_cifar10/features")
    parser.add_argument("--manifest-root", default="results/rebuttal/fixed_bag_cifar10/manifests")
    parser.add_argument("--run-root", default="results/rebuttal/4090_official_llppvc/runs")
    parser.add_argument("--summary-root", default="results/rebuttal/4090_official_llppvc/summaries")
    parser.add_argument("--bag-size", type=int, default=64)
    parser.add_argument("--train-bags", type=int, default=250)
    parser.add_argument("--val-bags", type=int, default=250)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.08)
    parser.add_argument("--warmup-lr", type=float, default=5e-5)
    parser.add_argument("--eps", type=float, default=1e-30)
    parser.add_argument("--run-label", default="primary")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_feat, test_feat = _load_features(Path(args.feature_root))
    train_bags, val_bags, test_bags = _load_manifests(
        Path(args.manifest_root), args.seed, args.bag_size, args.train_bags, args.alpha
    )
    train_hash = _manifest_hash(train_bags)
    val_hash = _manifest_hash(val_bags)
    test_hash = _manifest_hash(test_bags)

    cfg = {
        "task": "fixed_cifar10_official_llppvc",
        "method": "full_official_llp_pvc_adapter",
        "run_label": args.run_label,
        "seed": args.seed,
        "bag_size": args.bag_size,
        "train_bags": args.train_bags,
        "val_bags": args.val_bags,
        "test_bags": args.test_bags,
        "alpha": args.alpha,
        "num_classes": 10,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "warmup_lr": args.warmup_lr,
        "eps": args.eps,
        "probability_for_count_loss": "sigmoid(logits)",
        "prediction_rule": "argmax softmax(logits)",
        "selection_metric": "validation composite_count_nll, tie-break validation hist_count_mae",
        "upstream_sha": _upstream_sha(),
        "feature_hash_train": train_feat["feature_hash"],
        "feature_hash_test": test_feat["feature_hash"],
        "feature_label_hash_train": train_feat["label_hash"],
        "feature_label_hash_test": test_feat["label_hash"],
        "train_manifest_hash": train_hash,
        "val_manifest_hash": val_hash,
        "test_manifest_hash": test_hash,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = (
        f"official_llppvc_n{args.bag_size}_train{args.train_bags}_{args.run_label}_"
        f"s{args.seed}_lr{args.lr:.0e}_{git_commit()[:8]}_{stamp}"
    )
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "RUNNING").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    metadata = {
        "git_commit": git_commit(),
        "upstream_llp_pvc_sha": cfg["upstream_sha"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    try:
        import torchvision

        metadata["torchvision_version"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover
        metadata["torchvision_version"] = f"unavailable: {exc}"
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    head = nn.Linear(train_feat["features"].shape[1], 10).to(device)
    init_sigmoid_bias_to_one_over_k(head, 10)
    param_list = [{"params": [p for _, p in head.named_parameters()]}]
    opt = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    n_iters_all = math.ceil(args.train_bags / args.batch_size) * args.epochs
    warmup_iter = int(args.warmup_frac * n_iters_all)
    sched = OfficialWarmupCosineLrScheduler(
        opt,
        n_iters_all,
        warmup_iter=warmup_iter,
        warmup_ratio=args.warmup_lr / args.lr,
        warmup="linear",
    )

    train_features = train_feat["features"]
    test_features = test_feat["features"]
    train_labels = train_feat["labels"]
    test_labels = test_feat["labels"]
    expected_train_hash = train_hash
    best: dict[str, Any] = {"validation_composite_count_nll": math.inf, "validation_hist_count_mae": math.inf}
    metrics_path = run_dir / "metrics.jsonl"
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_hash = _manifest_hash(train_bags)
            if epoch_hash != expected_train_hash:
                raise RuntimeError("fixed training manifest changed across epochs")
            head.train()
            total = 0.0
            seen = 0
            start = time.perf_counter()
            for batch in iter_training_batches(train_features, train_bags, args.batch_size, args.seed + epoch):
                if "labels" in batch:
                    raise RuntimeError("hidden labels leaked into training batch")
                x = batch["features"].to(device)
                props = batch["proportions"].to(device)
                mask = batch["mask"].to(device)
                logits = head(x)
                loss = official_count_loss(logits, props, mask, eps=args.eps, reduce="mean")
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite LLP-PVC loss")
                opt.zero_grad()
                loss.backward()
                for name, param in head.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f"non-finite gradient in {name}")
                opt.step()
                sched.step()
                total += float(loss.item()) * x.shape[0]
                seen += x.shape[0]
            val = _evaluate(head, test_features, test_labels, val_bags, {**cfg, "batch_size": args.batch_size}, None)
            row = {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "validation_metric": val["composite_count_nll"],
                "validation_tiebreak_hist_count_mae": val["hist_count_mae"],
                "lr": sum(pg["lr"] for pg in opt.param_groups) / len(opt.param_groups),
                "epoch_seconds": time.perf_counter() - start,
                "train_manifest_hash_epoch_check": epoch_hash,
                **{f"val_{k}": v for k, v in val.items()},
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            better = (
                row["val_composite_count_nll"] < best["validation_composite_count_nll"]
                or (
                    row["val_composite_count_nll"] == best["validation_composite_count_nll"]
                    and row["val_hist_count_mae"] < best["validation_hist_count_mae"]
                )
            )
            if better:
                best = {
                    **row,
                    "validation_composite_count_nll": row["val_composite_count_nll"],
                    "validation_hist_count_mae": row["val_hist_count_mae"],
                }
                torch.save({"model": head.state_dict(), "config": cfg, "metrics": row}, run_dir / "checkpoint_best.pt")
        torch.save({"model": head.state_dict(), "config": cfg}, run_dir / "checkpoint_final.pt")
        state = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
        head.load_state_dict(state["model"])
        test = _evaluate(head, test_features, test_labels, test_bags, {**cfg, "batch_size": args.batch_size}, run_dir / "raw_per_bag_test_metrics.jsonl")
        unique = _write_unique_predictions(
            head, test_features, test_labels, test_bags, run_dir / "unique_image_predictions.jsonl", device, 10
        )
        summary = {
            "best": best,
            "test": {**test, **unique},
            "selection_metric": cfg["selection_metric"],
            "config": cfg,
            "run_dir": str(run_dir),
            "train_bag_stats": train_bags["stats"],
            "val_bag_stats": val_bags["stats"],
            "test_bag_stats": test_bags["stats"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
        out_dir = Path(args.summary_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"official_llppvc_n{args.bag_size}_train{args.train_bags}_{args.run_label}_s{args.seed}.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
        (run_dir / "RUNNING").unlink(missing_ok=True)
        (run_dir / "COMPLETED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        print(f"wrote {out}", flush=True)
    except Exception as exc:
        (run_dir / "RUNNING").unlink(missing_ok=True)
        (run_dir / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
