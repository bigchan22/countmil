#!/usr/bin/env python
"""Fixed-bag CIFAR-10 aggregate-supervision experiments on frozen features."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, TensorDataset

from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution_fft_tree
from countmil.baselines.proportion_matching import multiclass_proportion_matching_loss
from countmil.datasets.cifar import load_cifar_family
from countmil.training.run import git_commit
from countmil.training.seed import set_seed


def _sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def _extract_features(root: Path, split: str, device: torch.device, batch_size: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"cifar10_resnet18_imagenet_{split}_features.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")

    images, labels, num_classes = load_cifar_family("data", "CIFAR10", split, "coarse")
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # pragma: no cover
        raise ImportError("fixed CIFAR feature extraction requires torchvision") from exc

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    model.eval().to(device)
    mean = torch.tensor(weights.transforms().mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std, device=device).view(1, 3, 1, 1)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    feats = torch.empty(images.shape[0], feature_dim, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x = (x - mean) / std
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            f = model(x).detach().cpu()
            feats[offset : offset + f.shape[0]] = f
            offset += f.shape[0]
    payload = {
        "features": feats,
        "labels": labels,
        "num_classes": num_classes,
        "feature_hash": _sha256_tensor(feats),
        "label_hash": _sha256_tensor(labels),
        "weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "torchvision_transform_mean": list(weights.transforms().mean),
        "torchvision_transform_std": list(weights.transforms().std),
    }
    torch.save(payload, path)
    return payload


def _sample_fixed_bags(
    labels: torch.Tensor,
    num_bags: int,
    bag_size: int,
    alpha: float,
    seed: int,
    num_classes: int = 10,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    class_indices = [torch.nonzero(labels == c, as_tuple=False).flatten() for c in range(num_classes)]
    if any(idx.numel() == 0 for idx in class_indices):
        raise ValueError("every class needs at least one image")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        priors = torch.distributions.Dirichlet(torch.full((num_classes,), float(alpha))).sample((num_bags,))
    counts = torch.empty(num_bags, num_classes, dtype=torch.long)
    indices = torch.empty(num_bags, bag_size, dtype=torch.long)
    for b in range(num_bags):
        cnt = torch.multinomial(priors[b], bag_size, replacement=True, generator=gen).bincount(minlength=num_classes)
        counts[b] = cnt
        pos = 0
        for c, n_c in enumerate(cnt.tolist()):
            if n_c:
                pool = class_indices[c]
                draw = pool[torch.randint(0, pool.numel(), (n_c,), generator=gen)]
                indices[b, pos : pos + n_c] = draw
                pos += n_c
        perm = torch.randperm(bag_size, generator=gen)
        indices[b] = indices[b, perm]
    return {"indices": indices, "counts": counts, "priors": priors}


def _manifest_stats(bags: dict[str, torch.Tensor], labels: torch.Tensor, num_classes: int) -> dict[str, Any]:
    indices = bags["indices"]
    counts = bags["counts"].float()
    priors = bags["priors"]
    props = counts / indices.shape[1]
    global_prior = torch.bincount(labels, minlength=num_classes).float()
    global_prior = global_prior / global_prior.sum()
    entropy = -(props.clamp_min(1e-12) * props.clamp_min(1e-12).log()).sum(dim=-1)
    dist = (priors - global_prior).abs().sum(dim=-1)
    return {
        "num_unique_bags": int(torch.unique(indices, dim=0).shape[0]),
        "total_instance_slots": int(indices.numel()),
        "num_unique_underlying_images": int(torch.unique(indices).numel()),
        "per_class_count_mean": counts.mean(dim=0),
        "per_class_count_sample_std": counts.std(dim=0, unbiased=True),
        "bag_composition_entropy_mean": float(entropy.mean()),
        "bag_composition_entropy_sample_std": float(entropy.std(unbiased=True)),
        "min_class_proportion": float(props.min()),
        "max_class_proportion": float(props.max()),
        "l1_distance_to_global_prior_mean": float(dist.mean()),
        "l1_distance_to_global_prior_sample_std": float(dist.std(unbiased=True)),
        "manifest_sha256": _sha256_tensor(indices) + ":" + _sha256_tensor(counts.long()),
    }


def _load_or_create_manifest(root: Path, split: str, labels: torch.Tensor, cfg: dict[str, Any], offset: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{split}_bags.pt"
    stats_path = root / f"{split}_bag_stats.json"
    if path.exists():
        payload = torch.load(path, map_location="cpu")
    else:
        payload = _sample_fixed_bags(
            labels,
            int(cfg[f"{split}_bags"]),
            int(cfg["bag_size"]),
            float(cfg["alpha"]),
            int(cfg["seed"]) + offset,
            int(cfg["num_classes"]),
        )
        torch.save(payload, path)
    stats = _manifest_stats(payload, labels, int(cfg["num_classes"]))
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True, default=_json_default))
    payload["stats"] = stats
    payload["path"] = str(path)
    return payload


def _iter_batches(features: torch.Tensor, labels: torch.Tensor, bags: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int):
    order = torch.arange(bags["indices"].shape[0])
    if shuffle:
        order = order[torch.randperm(order.numel(), generator=torch.Generator().manual_seed(seed))]
    for start in range(0, order.numel(), batch_size):
        idx = order[start : start + batch_size]
        bag_idx = bags["indices"][idx]
        yield {
            "features": features[bag_idx],
            "labels": labels[bag_idx],
            "counts": bags["counts"][idx],
            "proportions": bags["counts"][idx].float() / bag_idx.shape[1],
            "mask": torch.ones_like(bag_idx, dtype=torch.bool),
        }


def _class_count_pmfs(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = probs.transpose(1, 2)
    atoms = torch.stack([1.0 - p, p], dim=-1)
    zero = probs.new_tensor([1.0, 0.0])
    atoms = torch.where(mask[:, None, :, None], atoms, zero)
    return finite_support_convolution_fft_tree(atoms, support_min=0).probs


def _count_loss(probs: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pmfs = _class_count_pmfs(probs, mask)
    losses = [aggregate_nll(AggregatePMF(pmfs[:, c], 0), counts[:, c]) for c in range(pmfs.shape[1])]
    return torch.stack(losses, dim=-1).mean()


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    vals = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        denom = 2 * tp + fp + fn
        vals.append((2 * tp / denom.clamp_min(1)).item())
    return float(sum(vals) / len(vals))


def _evaluate(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, bags: dict[str, torch.Tensor], cfg: dict[str, Any], raw_path: Path | None):
    head.eval()
    device = next(head.parameters()).device
    count_mae = []
    prop_mae = []
    comp_nll = []
    preds = []
    truth = []
    raw = []
    with torch.no_grad():
        for batch_id, batch in enumerate(_iter_batches(features, labels, bags, int(cfg["batch_size"]), False, int(cfg["seed"]))):
            x = batch["features"].to(device)
            counts = batch["counts"].to(device)
            props = batch["proportions"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)
            logits = head(x)
            eval_probs = torch.softmax(logits, dim=-1)
            pmfs = _class_count_pmfs(eval_probs, mask)
            nll = torch.stack([aggregate_nll(AggregatePMF(pmfs[:, c], 0), counts[:, c]) for c in range(pmfs.shape[1])], dim=-1)
            pred_counts = pmfs.argmax(dim=-1)
            pred_props = pred_counts.float() / mask.sum(dim=1, keepdim=True).float()
            count_mae.append((pred_counts.float() - counts.float()).abs().mean(dim=-1).cpu())
            prop_mae.append((pred_props - props).abs().mean(dim=-1).cpu())
            comp_nll.append(nll.mean(dim=-1).cpu())
            preds.append(eval_probs.argmax(dim=-1)[mask].cpu())
            truth.append(y[mask].cpu())
            if raw_path is not None:
                for j in range(x.shape[0]):
                    raw.append(
                        {
                            "batch": batch_id,
                            "row": j,
                            "count_mae": float((pred_counts[j].float() - counts[j].float()).abs().mean().cpu()),
                            "composite_nll": float(nll[j].mean().cpu()),
                        }
                    )
    pred_t = torch.cat(preds)
    true_t = torch.cat(truth)
    if raw_path is not None:
        with raw_path.open("w") as f:
            for row in raw:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "instance_acc": (pred_t == true_t).float().mean().item(),
        "macro_f1": _macro_f1(pred_t, true_t, int(cfg["num_classes"])),
        "hist_count_mae": torch.cat(count_mae).mean().item(),
        "hist_proportion_mae": torch.cat(prop_mae).mean().item(),
        "composite_count_nll": torch.cat(comp_nll).mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ce", "kl", "fsconv_count", "official_pvc_count_component"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-root", default="results/rebuttal/overnight_4090/cifar10_features")
    parser.add_argument("--manifest-root", default="results/rebuttal/overnight_4090/fixed_cifar10_manifests")
    parser.add_argument("--run-root", default="results/rebuttal/overnight_4090/runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bag-size", type=int, default=64)
    parser.add_argument("--train-bags", type=int, default=250)
    parser.add_argument("--val-bags", type=int, default=250)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    cfg: dict[str, Any] = {
        "task": "fixed_cifar10",
        "method": args.method,
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
        "weight_decay": args.weight_decay,
    }
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    feature_root = Path(args.feature_root)
    train_feat = _extract_features(feature_root, "train", device, args.batch_size)
    test_feat = _extract_features(feature_root, "test", device, args.batch_size)

    manifest_root = Path(args.manifest_root) / f"seed{args.seed}_n{args.bag_size}_train{args.train_bags}_alpha{args.alpha}"
    train_bags = _load_or_create_manifest(manifest_root, "train", train_feat["labels"], cfg, 0)
    val_bags = _load_or_create_manifest(manifest_root, "val", test_feat["labels"], cfg, 20_000)
    test_bags = _load_or_create_manifest(manifest_root, "test", test_feat["labels"], cfg, 10_000)
    cfg["train_manifest_hash"] = train_bags["stats"]["manifest_sha256"]
    cfg["val_manifest_hash"] = val_bags["stats"]["manifest_sha256"]
    cfg["test_manifest_hash"] = test_bags["stats"]["manifest_sha256"]
    cfg["feature_hash_train"] = train_feat["feature_hash"]
    cfg["feature_hash_test"] = test_feat["feature_hash"]

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.run_root) / f"fixed_cifar10_{args.method}_n{args.bag_size}_train{args.train_bags}_s{args.seed}_{git_commit()[:8]}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
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
    try:
        import torchvision

        metadata["torchvision_version"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover
        metadata["torchvision_version"] = f"unavailable: {exc}"
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    head = nn.Linear(train_feat["features"].shape[1], 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"validation_metric": math.inf}
    metrics_path = run_dir / "metrics.jsonl"
    train_features = train_feat["features"]
    train_labels = train_feat["labels"]
    test_features = test_feat["features"]
    test_labels = test_feat["labels"]
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        seen = 0
        start = time.perf_counter()
        for batch in _iter_batches(train_features, train_labels, train_bags, args.batch_size, True, args.seed + epoch):
            x = batch["features"].to(device)
            counts = batch["counts"].to(device)
            props = batch["proportions"].to(device)
            mask = batch["mask"].to(device)
            logits = head(x)
            if args.method in {"ce", "kl"}:
                probs = torch.softmax(logits, dim=-1)
                loss = multiclass_proportion_matching_loss(probs, props, mask, loss=args.method)
            elif args.method == "fsconv_count":
                loss = _count_loss(torch.softmax(logits, dim=-1), counts, mask)
            else:
                loss = _count_loss(torch.sigmoid(logits), counts, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite fixed-CIFAR loss")
            opt.zero_grad()
            loss.backward()
            for name, param in head.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name}")
            opt.step()
            total += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        val = _evaluate(head, test_features, test_labels, val_bags, cfg, None)
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_metric": val["hist_count_mae"],
            "epoch_seconds": time.perf_counter() - start,
            **{f"val_{k}": v for k, v in val.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        if row["validation_metric"] < best["validation_metric"]:
            best = row
            torch.save({"model": head.state_dict(), "config": cfg, "metrics": row}, run_dir / "checkpoint_best.pt")
    torch.save({"model": head.state_dict(), "config": cfg}, run_dir / "checkpoint_final.pt")
    state = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    head.load_state_dict(state["model"])
    test = _evaluate(head, test_features, test_labels, test_bags, cfg, run_dir / "raw_test_predictions.jsonl")
    summary = {
        "best": best,
        "test": test,
        "selection_metric": "validation hist_count_mae",
        "config": cfg,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    out_dir = Path("results/rebuttal/overnight_4090/fixed_cifar10")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fixed_cifar10_{args.method}_n{args.bag_size}_train{args.train_bags}_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
