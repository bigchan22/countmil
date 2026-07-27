#!/usr/bin/env python
"""A5000 rebuttal SVHN scalar-sum training.

This script reruns the submitted SVHN digit-sum protocol with aggregate-only
validation selection and complete rebuttal artifacts. Hidden instance digit
labels are used only for final evaluation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution
from countmil.baselines.gaussian_amle import categorical_gaussian_amle_loss, categorical_sum_moments
from countmil.datasets import SVHNOrdinalSumBags, TensorOrdinalSumBags, collate_ordinal_sum_bags, load_svhn_family
from countmil.masked_forward import forward_valid_instances
from countmil.models import make_cifar_classifier
from countmil.training.seed import set_seed


MASKED_SVHN_PROTOCOL_VERSION = "strictv3_train_holdout_val_official_test_no_hidden_val_masked_resnet_bag_forward"
MASKED_SVHN_OUTPUT_ROOT = "results/rebuttal/4090_svhn_masked_strictv3"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_sum_pmf(class_probs: torch.Tensor, mask: torch.Tensor) -> AggregatePMF:
    zero_atom = class_probs.new_zeros(class_probs.shape[-1])
    zero_atom[0] = 1.0
    atoms = torch.where(mask.unsqueeze(-1), class_probs, zero_atom)
    return finite_support_convolution(atoms, support_min=0)


def expected_sum_from_probs(class_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    support = torch.arange(class_probs.shape[-1], device=class_probs.device, dtype=class_probs.dtype)
    instance_means = (class_probs * support).sum(dim=-1)
    return (instance_means * mask.to(class_probs.dtype)).sum(dim=-1)


def masked_bag_class_probs(model: torch.nn.Module, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Class probabilities for padded bags without forwarding padded images."""

    logits = forward_valid_instances(model, images, mask)
    return torch.softmax(logits, dim=-1)


def gaussian_bin_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    var = variance.clamp_min(0.0) + float(eps)
    sigma = var.sqrt()
    target_f = target.to(mean.dtype)
    root2 = math.sqrt(2.0)
    upper = (target_f + 0.5 - mean) / (sigma * root2)
    lower = (target_f - 0.5 - mean) / (sigma * root2)
    prob = 0.5 * (torch.erf(upper) - torch.erf(lower))
    return -prob.clamp_min(1e-12).log()


def gaussian_density_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    var = variance.clamp_min(0.0) + float(eps)
    target_f = target.to(mean.dtype)
    return 0.5 * (((target_f - mean).square() / var) + var.log() + math.log(2.0 * math.pi))


def svhn_train_transform(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    out = torch.empty_like(images)
    for i in range(images.shape[0]):
        top = int(torch.randint(0, 9, (1,), generator=gen).item())
        left = int(torch.randint(0, 9, (1,), generator=gen).item())
        crop = padded[i, :, top : top + 32, left : left + 32]
        brightness = 0.8 + 0.4 * torch.rand((), generator=gen).item()
        contrast = 0.8 + 0.4 * torch.rand((), generator=gen).item()
        mean = crop.mean(dim=(-2, -1), keepdim=True)
        crop = ((crop - mean) * contrast + mean) * brightness
        out[i] = crop.clamp(0.0, 1.0)
    return out


def split_svhn_train_holdout(root: str | Path, val_fraction: float, split_seed: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    images, labels = load_svhn_family(root=root, split="train", download=False, train_split_fallback=False)
    gen = torch.Generator()
    gen.manual_seed(int(split_seed))
    order = torch.randperm(images.shape[0], generator=gen)
    val_count = max(1, int(round(images.shape[0] * float(val_fraction))))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def make_dataset(cfg: dict[str, Any], split_name: str) -> TensorOrdinalSumBags:
    if split_name == "train":
        (images, labels), _ = split_svhn_train_holdout(
            cfg["dataset_root"], float(cfg["val_image_fraction"]), int(cfg["val_image_split_seed"])
        )
        seed, bags, transform = int(cfg["seed"]), int(cfg["train_bags"]), svhn_train_transform if bool(cfg["augment"]) else None
        return TensorOrdinalSumBags(
            images,
            labels,
            num_bags=bags,
            bag_size_mean=float(cfg["bag_size_mean"]),
            bag_size_std=float(cfg["bag_size_std"]),
            bag_size_min=int(cfg["bag_size_min"]),
            bag_size_max=int(cfg["bag_size_max"]),
            per_class_cap=None,
            noise_sigma=0.0,
            seed=seed,
            transform=transform,
        )
    elif split_name == "val":
        _, (images, labels) = split_svhn_train_holdout(
            cfg["dataset_root"], float(cfg["val_image_fraction"]), int(cfg["val_image_split_seed"])
        )
        seed, bags = int(cfg["seed"]) + 20_000, int(cfg["val_bags"])
        return TensorOrdinalSumBags(
            images,
            labels,
            num_bags=bags,
            bag_size_mean=float(cfg["bag_size_mean"]),
            bag_size_std=float(cfg["bag_size_std"]),
            bag_size_min=int(cfg["bag_size_min"]),
            bag_size_max=int(cfg["bag_size_max"]),
            per_class_cap=None,
            noise_sigma=0.0,
            seed=seed,
            transform=None,
        )
    elif split_name == "test":
        split, seed, bags, augment = "test", int(cfg["seed"]) + 10_000, int(cfg["test_bags"]), False
    else:
        raise ValueError(split_name)
    return SVHNOrdinalSumBags(
        root=cfg["dataset_root"],
        split=split,
        download=False,
        train_split_fallback=False,
        augment=augment,
        num_bags=bags,
        bag_size_mean=float(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        bag_size_min=int(cfg["bag_size_min"]),
        bag_size_max=int(cfg["bag_size_max"]),
        per_class_cap=None,
        noise_sigma=0.0,
        seed=seed,
    )


def write_manifest(ds: SVHNOrdinalSumBags, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(len(ds)):
            gen = ds._generator(i)
            bag_size = ds._sample_bag_size(gen)
            indices = ds._sample_indices(bag_size, gen)
            labels = ds.labels[indices]
            row = {
                "bag_id": i,
                "indices": [int(x) for x in indices.tolist()],
                "sum": int(labels.sum().item()),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256(path)


def runtime_metadata(device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["dataset_root"])
    train = root / "train_32x32.mat"
    test = root / "test_32x32.mat"
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "command": " ".join(sys.argv),
        "svhn_train_md5": file_md5(train) if train.exists() else None,
        "svhn_test_md5": file_md5(test) if test.exists() else None,
        "svhn_train_sha256": sha256(train) if train.exists() else None,
        "svhn_test_sha256": sha256(test) if test.exists() else None,
        "pretrained_weight_identifier": "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
    }
    try:
        import torchvision

        meta["torchvision_version"] = torchvision.__version__
    except Exception as exc:
        meta["torchvision_version"] = f"unavailable: {exc}"
    if device.type == "cuda":
        meta["gpu_index"] = device.index
        meta["gpu_name"] = torch.cuda.get_device_name(device)
    torch_home = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
    ckpt = torch_home / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if ckpt.exists():
        meta["pretrained_checkpoint_path"] = str(ckpt)
        meta["pretrained_checkpoint_sha256"] = sha256(ckpt)
    return meta


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    eps: float,
    raw_path: Path | None = None,
    include_instance_metrics: bool = True,
) -> dict[str, float]:
    model.eval()
    support = torch.arange(10, device=device, dtype=torch.float32)
    total_train_loss = []
    total_fs_nll = []
    total_gauss_bin = []
    total_gauss_density = []
    expected_chunks = []
    target_chunks = []
    rounded_chunks = []
    mode_chunks = []
    pred_digit_chunks = []
    true_digit_chunks = []
    raw_rows = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["sum"].to(device)
            labels = batch["labels"].to(device) if include_instance_metrics else None
            probs = masked_bag_class_probs(model, x, mask)
            expected = expected_sum_from_probs(probs, mask)
            pmf = atomic_sum_pmf(probs, mask)
            fs_nll = aggregate_nll(pmf, targets)
            moments = categorical_sum_moments(probs, support, mask)
            g_density = gaussian_density_nll(moments.mean, moments.variance, targets, eps)
            g_bin = gaussian_bin_nll(moments.mean, moments.variance, targets, eps)
            if method == "fsconv":
                selection_loss = fs_nll
            elif method == "mse":
                selection_loss = (expected - targets.to(expected.dtype)).square()
            elif method == "gaussian_amle":
                selection_loss = categorical_gaussian_amle_loss(probs, support, targets, mask, eps=eps)
            else:
                raise ValueError(method)
            mode = pmf.probs.argmax(dim=-1) + pmf.support_min
            rounded = expected.round().long()
            total_train_loss.append(selection_loss.cpu())
            total_fs_nll.append(fs_nll.cpu())
            total_gauss_bin.append(g_bin.cpu())
            total_gauss_density.append(g_density.cpu())
            expected_chunks.append(expected.cpu())
            target_chunks.append(targets.cpu())
            rounded_chunks.append(rounded.cpu())
            mode_chunks.append(mode.cpu())
            if include_instance_metrics:
                pred_digit_chunks.append(probs.argmax(dim=-1)[mask].cpu())
                true_digit_chunks.append(labels[mask].cpu())
            if raw_path is not None:
                for j in range(x.shape[0]):
                    raw_rows.append(
                        {
                            "batch": batch_id,
                            "row": j,
                            "target_sum": int(targets[j].cpu()),
                            "expected_sum": float(expected[j].cpu()),
                            "rounded_expected_sum": int(rounded[j].cpu()),
                            "pmf_mode_sum": int(mode[j].cpu()),
                            "gaussian_variance": float(moments.variance[j].cpu()),
                            "fsconv_nll": float(fs_nll[j].cpu()),
                            "gaussian_bin_nll": float(g_bin[j].cpu()),
                            "gaussian_density_nll": float(g_density[j].cpu()),
                            "selection_loss": float(selection_loss[j].cpu()),
                        }
                    )
    if raw_path is not None:
        with raw_path.open("w") as f:
            for row in raw_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    expected_t = torch.cat(expected_chunks)
    target_t = torch.cat(target_chunks)
    rounded_t = torch.cat(rounded_chunks)
    mode_t = torch.cat(mode_chunks)
    out = {
        "selection_loss": torch.cat(total_train_loss).mean().item(),
        "expected_sum_mae": (expected_t - target_t.float()).abs().mean().item(),
        "rounded_expected_sum_acc": (rounded_t == target_t).float().mean().item(),
        "rounded_expected_sum_mae": (rounded_t.float() - target_t.float()).abs().mean().item(),
        "pmf_mode_sum_acc": (mode_t == target_t).float().mean().item(),
        "pmf_mode_sum_mae": (mode_t.float() - target_t.float()).abs().mean().item(),
        "fsconv_discrete_nll": torch.cat(total_fs_nll).mean().item(),
        "gaussian_bin_nll": torch.cat(total_gauss_bin).mean().item(),
        "gaussian_density_nll": torch.cat(total_gauss_density).mean().item(),
    }
    if include_instance_metrics:
        pred_digit_t = torch.cat(pred_digit_chunks)
        true_digit_t = torch.cat(true_digit_chunks)
        out["instance_digit_acc"] = (pred_digit_t == true_digit_t).float().mean().item()
    return out


def unique_manifest_indices(ds: SVHNOrdinalSumBags) -> list[int]:
    indices: set[int] = set()
    for i in range(len(ds)):
        gen = ds._generator(i)
        bag_size = ds._sample_bag_size(gen)
        bag_indices = ds._sample_indices(bag_size, gen)
        indices.update(int(x) for x in bag_indices.tolist())
    return sorted(indices)


def write_unique_instance_predictions(model: torch.nn.Module, ds: SVHNOrdinalSumBags, device: torch.device, path: Path) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 512
    indices = unique_manifest_indices(ds)
    with path.open("w") as f, torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            index_t = torch.tensor(batch_indices, dtype=torch.long)
            x = ds.images[index_t].to(device)
            probs = model.predict_proba(x).cpu()
            pred = probs.argmax(dim=-1)
            labels = ds.labels[index_t]
            for offset, image_index in enumerate(batch_indices):
                row = {
                    "image_index": int(image_index),
                    "true_digit": int(labels[offset]),
                    "predicted_digit": int(pred[offset]),
                    "probs": [float(v) for v in probs[offset].tolist()],
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")


def build_run_dir(root: Path, cfg: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short = git_commit()[:8]
    run_id = f"svhn_sum_{cfg['method']}_strictv3_masked_n{cfg['bag_size_mean']}_train{cfg['train_bags']}_s{cfg['seed']}_{short}_{stamp}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fsconv", "mse", "gaussian_amle"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output-root", default=MASKED_SVHN_OUTPUT_ROOT)
    parser.add_argument("--train-bags", type=int, default=5000)
    parser.add_argument("--val-bags", type=int, default=600)
    parser.add_argument("--test-bags", type=int, default=600)
    parser.add_argument("--bag-size-mean", type=float, default=10.0)
    parser.add_argument("--bag-size-std", type=float, default=2.0)
    parser.add_argument("--bag-size-min", type=int, default=5)
    parser.add_argument("--bag-size-max", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gaussian-eps", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--val-image-fraction", type=float, default=0.15)
    parser.add_argument("--val-image-split-seed", type=int, default=1729)
    args = parser.parse_args()

    cfg: dict[str, Any] = {
        "task": "svhn_digit_sum",
        "method": args.method,
        "seed": args.seed,
        "dataset_root": args.dataset_root,
        "train_bags": args.train_bags,
        "val_bags": args.val_bags,
        "test_bags": args.test_bags,
        "bag_size_mean": args.bag_size_mean,
        "bag_size_std": args.bag_size_std,
        "bag_size_min": args.bag_size_min,
        "bag_size_max": args.bag_size_max,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gaussian_eps": args.gaussian_eps,
        "augment": bool(args.augment),
        "protocol_version": MASKED_SVHN_PROTOCOL_VERSION,
        "masked_bag_forward": "forward_valid_instances(model, instances, mask) before softmax; padded images are never passed through ResNet-18/BatchNorm",
        "val_image_fraction": args.val_image_fraction,
        "val_image_split_seed": args.val_image_split_seed,
        "validation_split": "deterministic 15% holdout from official SVHN train archive",
        "test_split": "official SVHN test archive",
        "model": "resnet18",
        "pretrained": True,
        "num_classes": 10,
        "selection_metric": {
            "fsconv": "validation exact aggregate NLL",
            "mse": "validation expected-sum MSE",
            "gaussian_amle": "validation Gaussian-AMLE training loss",
        }[args.method],
    }
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    run_dir = build_run_dir(output_root, cfg)
    for name in ["manifests", "raw", "checkpoints"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    train_ds = make_dataset(cfg, "train")
    val_ds = make_dataset(cfg, "val")
    test_ds = make_dataset(cfg, "test")
    manifest_hashes = {
        "train": write_manifest(train_ds, run_dir / "manifests" / "train_bags.jsonl"),
        "val": write_manifest(val_ds, run_dir / "manifests" / "val_bags.jsonl"),
        "test": write_manifest(test_ds, run_dir / "manifests" / "test_bags.jsonl"),
    }
    metadata = runtime_metadata(device, cfg)
    metadata["manifest_hashes"] = manifest_hashes
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ordinal_sum_bags)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ordinal_sum_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ordinal_sum_bags)
    model = make_cifar_classifier("resnet18", num_classes=10, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    support = torch.arange(10, device=device, dtype=torch.float32)
    metrics_path = run_dir / "metrics.jsonl"
    best: dict[str, Any] = {"val_selection_loss": math.inf}
    final: dict[str, Any] = {}

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
            targets = batch["sum"].to(device)
            probs = masked_bag_class_probs(model, x, mask)
            if args.method == "fsconv":
                loss = aggregate_nll(atomic_sum_pmf(probs, mask), targets).mean()
            elif args.method == "mse":
                loss = F.mse_loss(expected_sum_from_probs(probs, mask), targets.to(probs.dtype))
            else:
                loss = categorical_gaussian_amle_loss(probs, support, targets, mask, eps=args.gaussian_eps).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        val_metrics = evaluate(model, val_loader, device, args.method, args.gaussian_eps, include_instance_metrics=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_selection_loss": val_metrics["selection_loss"],
            "epoch_seconds": time.perf_counter() - start,
            "peak_cuda_mem_mb": peak_mb,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        final = row
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["val_selection_loss"] < best["val_selection_loss"]:
            best = row
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": row}, run_dir / "checkpoints" / "checkpoint_best.pt")

    torch.save({"model": model.state_dict(), "config": cfg, "metrics": final}, run_dir / "checkpoints" / "checkpoint_final.pt")
    state = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, args.method, args.gaussian_eps, run_dir / "raw" / "test_bag_predictions.jsonl")
    write_unique_instance_predictions(model, test_ds, device, run_dir / "raw" / "test_unique_instance_predictions.jsonl")
    summary = {
        "status": "COMPLETED",
        "config": cfg,
        "metadata": metadata,
        "best": best,
        "final": final,
        "test": test_metrics,
        "run_dir": str(run_dir),
        "manifest_hashes": manifest_hashes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summaries = output_root / "svhn_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    out = summaries / f"svhn_sum_{args.method}_strictv3_masked_n{int(args.bag_size_mean)}_train{args.train_bags}_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
