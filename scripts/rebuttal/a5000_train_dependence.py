#!/usr/bin/env python
"""A5000 conditional-independence stress test for binary counts."""

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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countmil.aggregators import aggregate_nll, binary_count_dp
from countmil.baselines.gaussian_amle import gaussian_amle_loss_from_moments
from experiments.dependence_stress import DependenceData, generate_dependence_data
from countmil.training.seed import set_seed


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class InstanceMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def moments(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = probs.sum(dim=1)
    variance = (probs * (1.0 - probs)).sum(dim=1).clamp_min(0.0)
    return mean, variance


def gaussian_bin_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    var = variance.clamp_min(0.0) + float(eps)
    sigma = var.sqrt()
    target_f = target.to(mean.dtype)
    root2 = math.sqrt(2.0)
    upper = (target_f + 0.5 - mean) / (sigma * root2)
    lower = (target_f - 0.5 - mean) / (sigma * root2)
    prob = 0.5 * (torch.erf(upper) - torch.erf(lower))
    return -prob.clamp_min(1e-12).log()


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.long().flatten()
    scores = scores.float().flatten()
    pos = int(labels.sum().item())
    neg = int(labels.numel() - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[labels == 1].sum()
    return float(((pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)).item())


def central_interval_from_pmf(pmf: torch.Tensor, alpha: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    cdf = pmf.cumsum(dim=-1)
    lower = (cdf >= alpha / 2.0).float().argmax(dim=-1)
    upper = (cdf >= 1.0 - alpha / 2.0).float().argmax(dim=-1)
    return lower, upper


def evaluate(
    model: InstanceMLP,
    loader: DataLoader,
    device: torch.device,
    method: str,
    eps: float,
    raw_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    ys = []
    means = []
    modes = []
    probs_all = []
    z_all = []
    fs_nlls = []
    gauss_nlls = []
    cover = []
    variances = []
    raw_rows = []
    with torch.no_grad():
        for batch_id, (x, y, z) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            probs = model.predict_proba(x)
            mean, variance = moments(probs)
            pmf = binary_count_dp(probs)
            mode = pmf.probs.argmax(dim=-1)
            fs_nll = aggregate_nll(pmf, y)
            gbin = gaussian_bin_nll(mean, variance, y, eps)
            if method == "fsconv":
                lo, hi = central_interval_from_pmf(pmf.probs)
                covered = (y >= lo.to(y.device)) & (y <= hi.to(y.device))
            else:
                half = 1.6448536269514722 * (variance + eps).sqrt()
                covered = (y.float() >= mean - half) & (y.float() <= mean + half)
            ys.append(y.cpu())
            means.append(mean.cpu())
            modes.append(mode.cpu())
            probs_all.append(probs.cpu())
            z_all.append(z.cpu())
            fs_nlls.append(fs_nll.cpu())
            gauss_nlls.append(gbin.cpu())
            cover.append(covered.float().cpu())
            variances.append(variance.cpu())
            if raw_path is not None:
                for j in range(y.shape[0]):
                    raw_rows.append(
                        {
                            "batch": batch_id,
                            "row": j,
                            "target_count": int(y[j].cpu()),
                            "expected_count": float(mean[j].cpu()),
                            "rounded_count": int(mean[j].round().cpu()),
                            "pmf_mode_count": int(mode[j].cpu()),
                            "predicted_variance": float(variance[j].cpu()),
                            "fsconv_nll": float(fs_nll[j].cpu()),
                            "gaussian_bin_nll": float(gbin[j].cpu()),
                        }
                    )
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w") as f:
            for row in raw_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    y_t = torch.cat(ys)
    mean_t = torch.cat(means)
    mode_t = torch.cat(modes)
    prob_t = torch.cat(probs_all).flatten()
    z_t = torch.cat(z_all).flatten()
    rounded = mean_t.round().clamp(0, 32).long()
    residual = y_t.float() - mean_t
    inst_pred = (prob_t >= 0.5).long()
    return {
        "count_mae": (mean_t - y_t.float()).abs().mean().item(),
        "expected_count_mse": (mean_t - y_t.float()).square().mean().item(),
        "rounded_count_acc": (rounded == y_t).float().mean().item(),
        "pmf_mode_count_acc": (mode_t == y_t).float().mean().item(),
        "instance_auc": auc_score(prob_t, z_t),
        "instance_acc_threshold_0p5": (inst_pred == z_t).float().mean().item(),
        "fsconv_aggregate_nll": torch.cat(fs_nlls).mean().item(),
        "gaussian_bin_nll": torch.cat(gauss_nlls).mean().item(),
        "coverage_90": torch.cat(cover).mean().item(),
        "mean_predicted_aggregate_variance": torch.cat(variances).mean().item(),
        "empirical_aggregate_residual_variance": residual.var(unbiased=True).item(),
        "coverage_gap_90": abs(torch.cat(cover).mean().item() - 0.90),
    }


def make_run_dir(root: Path, cfg: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short = git_commit()[:8]
    tau = str(cfg["tau"]).replace(".", "p")
    run_id = f"dependence_{cfg['method']}_tau{tau}_n{cfg['bag_size']}_s{cfg['seed']}_{short}_{stamp}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_manifest(data: DependenceData, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": data.x,
            "z": data.z,
            "y": data.y,
            "u": data.u,
            "w": data.w,
            "intercept": data.intercept,
            "tau": data.tau,
            "seed": data.seed,
            "prevalence": data.prevalence,
            "within_bag_correlation": data.within_bag_correlation,
        },
        path,
    )
    return sha256(path)


def build_data(cfg: dict[str, Any]) -> dict[str, DependenceData]:
    dim = int(cfg["dim"])
    gen = torch.Generator()
    gen.manual_seed(int(cfg["seed"]))
    w_raw = torch.randn(dim, generator=gen)
    w = w_raw / w_raw.norm().clamp_min(1e-12)
    tau = float(cfg["tau"])
    intercept = float(cfg["intercept"])
    return {
        "train": generate_dependence_data(bags=int(cfg["train_bags"]), bag_size=int(cfg["bag_size"]), dim=dim, tau=tau, seed=int(cfg["seed"]) + 1_000, w=w, intercept=intercept),
        "val": generate_dependence_data(bags=int(cfg["val_bags"]), bag_size=int(cfg["bag_size"]), dim=dim, tau=tau, seed=int(cfg["seed"]) + 2_000, w=w, intercept=intercept),
        "test": generate_dependence_data(bags=int(cfg["test_bags"]), bag_size=int(cfg["bag_size"]), dim=dim, tau=tau, seed=int(cfg["seed"]) + 3_000, w=w, intercept=intercept),
    }


def selection_loss(method: str, probs: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    mean, variance = moments(probs)
    if method == "fsconv":
        return aggregate_nll(binary_count_dp(probs), y)
    if method == "mse":
        return (mean - y.to(mean.dtype)).square()
    if method == "gaussian_amle":
        return gaussian_amle_loss_from_moments(mean, variance, y, eps)
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fsconv", "mse", "gaussian_amle"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="results/rebuttal/a5000_svhn_dependence")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gaussian-eps", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--bag-size", type=int, default=32)
    parser.add_argument("--train-bags", type=int, default=2000)
    parser.add_argument("--val-bags", type=int, default=500)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--intercept", type=float, default=-0.5)
    args = parser.parse_args()
    cfg = vars(args).copy()
    cfg["task"] = "dependence"
    cfg["model"] = "instance_mlp"
    cfg["selection_rule"] = "aggregate validation objective only"

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    run_dir = make_run_dir(output_root, cfg)
    for name in ["manifests", "raw", "checkpoints"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    data = build_data(cfg)
    hashes = {split: save_manifest(obj, run_dir / "manifests" / f"{split}.pt") for split, obj in data.items()}
    dgp_stats = {
        split: {
            "prevalence": obj.prevalence,
            "within_bag_correlation": obj.within_bag_correlation,
            "count_mean": float(obj.y.float().mean().item()),
            "count_std": float(obj.y.float().std(unbiased=True).item()),
        }
        for split, obj in data.items()
    }
    metadata = {
        "status": "RUNNING",
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "manifest_hashes": hashes,
        "dgp_stats": dgp_stats,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    train_loader = DataLoader(TensorDataset(data["train"].x, data["train"].y, data["train"].z), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data["val"].x, data["val"].y, data["val"].z), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(data["test"].x, data["test"].y, data["test"].z), batch_size=args.batch_size, shuffle=False)
    model = InstanceMLP(args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best: dict[str, Any] = {"val_selection_loss": math.inf}
    final: dict[str, Any] = {}
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        model.train()
        total = 0.0
        seen = 0
        for x, y, _z in train_loader:
            x = x.to(device)
            y = y.to(device)
            probs = model.predict_proba(x)
            loss = selection_loss(args.method, probs, y, args.gaussian_eps).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        val_metrics = evaluate(model, val_loader, device, args.method, args.gaussian_eps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_selection_loss": {
                "fsconv": val_metrics["fsconv_aggregate_nll"],
                "mse": val_metrics["expected_count_mse"],
                "gaussian_amle": val_metrics["gaussian_bin_nll"],
            }[args.method],
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
    summary = {
        "status": "COMPLETED",
        "config": cfg,
        "metadata": metadata,
        "best": best,
        "final": final,
        "test": test_metrics,
        "run_dir": str(run_dir),
        "manifest_hashes": hashes,
        "dgp_stats": dgp_stats,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    summaries = output_root / "dependence_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    tau = str(args.tau).replace(".", "p")
    out = summaries / f"dependence_tau{tau}_{args.method}_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
