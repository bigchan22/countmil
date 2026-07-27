#!/usr/bin/env python
"""Train one strict-v1 Criteo feature-bag aggregate method."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from countmil.criteo.constants import BAG_SIZE, PROTOCOL_TAG
from countmil.criteo.data import artifact_record
from countmil.criteo.losses import (
    aggregate_metrics,
    dllp_bce_loss,
    dllp_mse_loss,
    easyllp_loss,
    expected_count_mae,
    fsconv_nll_loss,
    genbags_loss,
    instance_metrics,
    ot_llp_loss,
)
from countmil.criteo.model import CriteoInstanceModel, infer_categorical_layout
from countmil.strict_protocol import canonical_hash
from countmil.training.run import git_commit
from countmil.training.seed import set_seed


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _flatten(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    num = batch["numeric"].to(device)
    cat = batch["categorical"].to(device)
    return num.reshape(-1, num.shape[-1]), cat.reshape(-1, cat.shape[-1])


def _loss(method: str, logits: torch.Tensor, counts: torch.Tensor, global_prior: float, *, training: bool) -> torch.Tensor:
    if method == "dllp_bce":
        return dllp_bce_loss(logits, counts, BAG_SIZE)
    if method == "dllp_mse":
        return dllp_mse_loss(logits, counts)
    if method == "easyllp":
        return easyllp_loss(logits, counts, bag_size=BAG_SIZE, global_prior=global_prior)
    if method == "genbags":
        return genbags_loss(logits, counts, stochastic=training)
    if method == "ot_llp":
        return ot_llp_loss(logits, counts)
    if method == "supervised_oracle":
        return dllp_bce_loss(logits, counts, BAG_SIZE)
    if method == "fsconv":
        return fsconv_nll_loss(logits, counts)
    raise ValueError(method)


def _validate(model: CriteoInstanceModel, loader: DataLoader, device: torch.device, method: str, global_prior: float, max_batches: Optional[int] = None) -> dict[str, float]:
    model.eval()
    maes = []
    native = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            if max_batches is not None and batch_id >= max_batches:
                break
            num, cat = _flatten(batch, device)
            counts = batch["counts"].to(device)
            logits = model(num, cat).reshape(counts.shape[0], BAG_SIZE)
            maes.append(expected_count_mae(logits, counts).detach().cpu())
            native.append(_loss(method, logits, counts, global_prior, training=False).detach().cpu().reshape(1))
    return {
        "val_expected_count_mae": float(torch.cat(maes).mean().item()),
        "val_native_objective": float(torch.cat(native).mean().item()),
    }


def _test_eval(model: CriteoInstanceModel, shard: dict[str, Any], device: torch.device, batch_size: int, raw_dir: Path) -> dict[str, float]:
    model.eval()
    n_bags = int(shard["counts"].shape[0])
    bag_rows = []
    all_probs = []
    all_labels = []
    agg = {"expected_count_mae": [], "rounded_expected_count_acc": [], "pmf_mode_count_acc": [], "poisson_binomial_nll": []}
    with torch.no_grad():
        for start in range(0, n_bags, batch_size):
            end = min(start + batch_size, n_bags)
            num = shard["numeric"][start:end].to(device)
            cat = shard["categorical"][start:end].to(device)
            counts = shard["counts"][start:end].to(device)
            logits = model(num.reshape(-1, num.shape[-1]), cat.reshape(-1, cat.shape[-1])).reshape(end - start, BAG_SIZE)
            probs = torch.sigmoid(logits)
            metrics = aggregate_metrics(probs, counts)
            for key, val in metrics.items():
                agg[key].append(val.detach().cpu())
            labels = shard["hidden_labels"][start:end]
            all_probs.append(probs.detach().cpu().reshape(-1))
            all_labels.append(labels.reshape(-1))
            for j in range(end - start):
                bag_rows.append(
                    {
                        "bag_id": start + j,
                        "count": int(counts[j].cpu().item()),
                        "expected_count": float(probs[j].sum().cpu().item()),
                        "expected_count_mae": float(metrics["expected_count_mae"][j].cpu().item()),
                        "rounded_expected_count_acc": float(metrics["rounded_expected_count_acc"][j].cpu().item()),
                        "pmf_mode_count_acc": float(metrics["pmf_mode_count_acc"][j].cpu().item()),
                        "poisson_binomial_nll": float(metrics["poisson_binomial_nll"][j].cpu().item()),
                    }
                )
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "raw_per_bag_predictions.jsonl").open("w") as f:
        for row in bag_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels).long()
    torch.save({"probs": probs, "labels": labels}, raw_dir / "raw_test_probs_labels.pt")
    out = {k: float(torch.cat(v).mean().item()) for k, v in agg.items()}
    out.update(instance_metrics(probs, labels))
    return out


def _make_loaders(train: dict[str, Any], valid: dict[str, Any], batch_size: int, seed: int, *, expose_train_labels: bool = False) -> tuple[DataLoader, DataLoader]:
    def assert_no_hidden(shard: dict[str, Any], name: str) -> None:
        if "hidden_labels" in shard:
            raise RuntimeError(f"{name} shard exposes hidden labels to train/validation")
    assert_no_hidden(train, "train")
    assert_no_hidden(valid, "valid")
    g = torch.Generator().manual_seed(seed)
    train_ds = []
    for i in range(train["counts"].shape[0]):
        item = {"numeric": train["numeric"][i], "categorical": train["categorical"][i], "counts": train["counts"][i]}
        if expose_train_labels:
            item["hidden_labels"] = train["hidden_labels_external_only"][i]
        train_ds.append(item)
    val_ds = [{"numeric": valid["numeric"][i], "categorical": valid["categorical"][i], "counts": valid["counts"][i]} for i in range(valid["counts"].shape[0])]
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dllp_bce", "dllp_mse", "easyllp", "fsconv", "genbags", "ot_llp", "supervised_oracle"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--artifact-root", default="/home/chanhomin/datasets/criteo_x1/strictv1")
    p.add_argument("--result-root", default="results/rebuttal/4090_criteo_extra_baselines")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--skip-final-test", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    sha = git_commit()
    if sha == "unknown":
        raise RuntimeError("Criteo jobs require a Git SHA")
    root = Path(args.artifact_root) / f"seed{args.seed}"
    train = _load(root / f"train_seed{args.seed}_shard.pt")
    valid = _load(root / f"valid_seed{args.seed}_shard.pt")
    test = _load(root / f"test_seed{args.seed}_shard.pt")
    if "hidden_labels" not in test:
        raise RuntimeError("test shard must contain hidden labels for final evaluation")
    train_hash = train["metadata"]["manifest_hash"]
    val_hash = valid["metadata"]["manifest_hash"]
    test_hash = test["metadata"]["manifest_hash"]
    cat_min = torch.amin(torch.cat([train["categorical"].reshape(-1, 26), valid["categorical"].reshape(-1, 26), test["categorical"].reshape(-1, 26)]), dim=0).tolist()
    cat_max = torch.amax(torch.cat([train["categorical"].reshape(-1, 26), valid["categorical"].reshape(-1, 26), test["categorical"].reshape(-1, 26)]), dim=0).tolist()
    shared, cardinalities = infer_categorical_layout(cat_min, cat_max)
    cfg = {
        **vars(args),
        "protocol_tag": PROTOCOL_TAG,
        "git_commit": sha,
        "train_manifest_hash": train_hash,
        "valid_manifest_hash": val_hash,
        "test_manifest_hash": test_hash,
        "train_input_shard_hash": train["metadata"]["input_shard_hash"],
        "valid_input_shard_hash": valid["metadata"]["input_shard_hash"],
        "test_input_shard_hash": test["metadata"]["input_shard_hash"],
        "shared_categorical_embedding": shared,
        "categorical_cardinalities": cardinalities,
        "checkpoint_metric": "validation expected-count MAE only",
        "hidden_validation_metrics_logged": False,
        "supervised_training_labels_used": args.method == "supervised_oracle",
    }
    cfg_hash = canonical_hash({"config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list)) or v is None}})
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{PROTOCOL_TAG}_{args.method}_s{args.seed}_{cfg_hash[:12]}_{sha[:8]}_{stamp}"
    run_dir = Path(args.result_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "RUNNING").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfg | {"config_hash": cfg_hash, "run_id": run_id}, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu",
    }
    (run_dir / "environment.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader, val_loader = _make_loaders(train, valid, args.batch_size, args.seed, expose_train_labels=args.method == "supervised_oracle")
    global_prior = float(train["counts"].sum().item() / (train["counts"].numel() * BAG_SIZE))
    model = CriteoInstanceModel(categorical_cardinalities=cardinalities, shared_categorical=shared).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"val_expected_count_mae": math.inf, "val_native_objective": math.inf, "epoch": -1}
    bad = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch_id, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_id >= args.max_train_batches:
                    break
                num, cat = _flatten(batch, device)
                counts = batch["counts"].to(device)
                logits = model(num, cat).reshape(counts.shape[0], BAG_SIZE)
                if args.method == "supervised_oracle":
                    labels = batch["hidden_labels"].to(device).float()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    loss = _loss(args.method, logits, counts, global_prior, training=True)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            val = _validate(model, val_loader, device, args.method, global_prior, max_batches=args.max_val_batches)
            row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), **val}
            history.append(row)
            with (run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            improved = (val["val_expected_count_mae"], val["val_native_objective"]) < (best["val_expected_count_mae"], best["val_native_objective"])
            if improved:
                best = {**row}
                torch.save({"model": model.state_dict(), "config": cfg | {"config_hash": cfg_hash}, "best": best}, run_dir / "checkpoint_best.pt")
                bad = 0
            else:
                bad += 1
            if bad >= args.patience:
                break
        ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        test_metrics = {} if args.skip_final_test else _test_eval(model, test, device, args.batch_size, run_dir)
        summary = {
            "run_id": run_id,
            "status": "COMPLETED",
            "config": cfg | {"config_hash": cfg_hash},
            "global_prior_from_aggregate_train_counts": global_prior,
            "best": best,
            "test": test_metrics,
            "run_dir": str(run_dir),
            "selection_metric": "aggregate validation expected-count MAE, tie-break native aggregate validation objective; no hidden validation metrics",
        }
        out = Path(args.result_root) / f"{args.method}_s{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        (run_dir / "COMPLETED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    except Exception as exc:
        (run_dir / "FAILED").write_text(repr(exc) + "\n")
        raise
    finally:
        running = run_dir / "RUNNING"
        if running.exists():
            running.unlink()


if __name__ == "__main__":
    main()
