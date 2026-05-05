#!/usr/bin/env python
"""Train MNIST-Bags with exact Count Loss.

This script supports two equivalent likelihood implementations:

- `dp`: Shukla-style dynamic programming Count Loss.
- `conv`: finite-support convolution Count Loss.

It intentionally uses local IDX files through `countmil.datasets.MNISTBags` and
never downloads data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from countmil.aggregators import aggregate_nll, binary_count_dp, finite_support_convolution
from countmil.datasets import MNISTBags, collate_mnist_bags
from countmil.metrics import binary_auc
from countmil.models import ShuklaMNISTSelector
from countmil.posteriors import binary_count_posterior
from countmil.training.run import make_run_dir, write_run_metadata
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
        elif value.startswith("[") and value.endswith("]"):
            parsed = [x.strip() for x in value[1:-1].split(",") if x.strip()]
            parsed = [int(x) if x.lstrip("-").isdigit() else x for x in parsed]
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


def _count_pmf(probs: torch.Tensor, mask: torch.Tensor, method: str):
    probs = probs * mask.float()
    if method == "dp":
        # Masked positions are deterministic zero atoms.
        return binary_count_dp(probs)
    if method == "conv":
        atoms = torch.stack([1.0 - probs, probs], dim=-1)
        atoms = torch.where(mask.unsqueeze(-1), atoms, torch.tensor([1.0, 0.0], device=probs.device))
        return finite_support_convolution(atoms, support_min=0)
    raise ValueError(f"unknown method: {method}")


def _valid_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_select(mask).mean()


def _bernoulli_entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp(eps, 1.0 - eps)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())


def _tempered_marginals(q: torch.Tensor, temperature: float, eps: float = 1e-6) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("em_temperature must be positive")
    if temperature == 1.0:
        return q
    logits = torch.logit(q.clamp(eps, 1.0 - eps))
    return torch.sigmoid(logits / temperature)


def _hard_marginals(q: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hard = torch.zeros_like(q)
    masked_q = q.masked_fill(~mask, float("-inf"))
    for b in range(q.shape[0]):
        k = int(counts[b].clamp(0, int(mask[b].sum().item())).item())
        if k > 0:
            hard[b, masked_q[b].topk(k).indices] = 1.0
    return hard


def _training_loss(
    logits: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
    counts: torch.Tensor,
    method: str,
    objective: str,
    entropy_weight: float,
    em_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pmf = _count_pmf(probs, mask, method)
    nll = aggregate_nll(pmf, counts).mean()
    extras = {"batch_nll": float(nll.detach().item())}

    if objective == "nll":
        return nll, extras
    if objective == "nll_entropy":
        entropy = _valid_mean(_bernoulli_entropy(probs), mask)
        extras["entropy"] = float(entropy.detach().item())
        return nll + entropy_weight * entropy, extras
    if objective in {"soft_em", "tempered_em", "hard_em"}:
        with torch.no_grad():
            q = binary_count_posterior((probs * mask.float()).detach(), counts)
            if objective == "tempered_em":
                q = _tempered_marginals(q, em_temperature)
            elif objective == "hard_em":
                q = _hard_marginals(q, counts, mask)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, q, reduction="none")
        em_loss = _valid_mean(bce, mask)
        extras["em_loss"] = float(em_loss.detach().item())
        extras["posterior_mean"] = float(q.masked_select(mask).mean().detach().item())
        return em_loss, extras
    raise ValueError(f"unknown objective: {objective}")


def _evaluate(model, loader, device: torch.device, method: str) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_bags = 0
    bag_scores = []
    bag_labels = []
    count_preds = []
    count_targets = []
    inst_scores = []
    inst_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            counts = batch["count"].to(device)
            labels = batch["bag_label"].to(device)
            hidden = batch["instance_labels"].to(device)

            probs = model.predict_proba(x)
            pmf = _count_pmf(probs, mask, method)
            nll = aggregate_nll(pmf, counts)
            total_nll += nll.sum().item()
            total_bags += x.shape[0]

            p_zero = pmf.probs[..., 0]
            bag_prob = 1.0 - p_zero
            bag_scores.append(bag_prob.cpu())
            bag_labels.append(labels.cpu())
            count_preds.append(pmf.probs.argmax(dim=-1).cpu())
            count_targets.append(counts.cpu())
            inst_scores.append(probs[mask].cpu())
            inst_labels.append(hidden[mask].cpu())

    bag_scores_t = torch.cat(bag_scores)
    bag_labels_t = torch.cat(bag_labels)
    count_preds_t = torch.cat(count_preds)
    count_targets_t = torch.cat(count_targets)
    inst_scores_t = torch.cat(inst_scores)
    inst_labels_t = torch.cat(inst_labels)

    return {
        "nll": total_nll / max(total_bags, 1),
        "bag_acc": ((bag_scores_t >= 0.5).long() == bag_labels_t).float().mean().item(),
        "bag_auc": binary_auc(bag_scores_t, bag_labels_t),
        "count_acc": (count_preds_t == count_targets_t).float().mean().item(),
        "count_mae": (count_preds_t.float() - count_targets_t.float()).abs().mean().item(),
        "instance_acc": ((inst_scores_t >= 0.5).long() == inst_labels_t).float().mean().item(),
        "instance_auc": binary_auc(inst_scores_t, inst_labels_t),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", choices=["dp", "conv"], default=None)
    parser.add_argument("--objective", choices=["nll", "nll_entropy", "soft_em", "tempered_em", "hard_em"], default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--em-temperature", type=float, default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target-digit", type=int, default=None)
    parser.add_argument("--bag-size-mean", type=int, default=None)
    parser.add_argument("--bag-size-std", type=float, default=None)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=1000)
    parser.add_argument("--balanced-binary", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results-dir", default="results/mnist_bags")
    args = parser.parse_args()

    cfg = {
        "method": "conv",
        "objective": "nll",
        "entropy_weight": 0.01,
        "em_temperature": 0.5,
        "dataset_root": "data",
        "dataset": "MNIST",
        "target_digit": 9,
        "bag_size_mean": 10,
        "bag_size_std": 2.0,
        "train_bags": 500,
        "balanced_binary": True,
        "seed": 0,
    }
    cfg.update(_parse_simple_yaml(args.config))
    for key, value in {
        "method": args.method,
        "objective": args.objective,
        "entropy_weight": args.entropy_weight,
        "em_temperature": args.em_temperature,
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
    if args.balanced_binary:
        cfg["balanced_binary"] = True

    method = str(cfg["method"])
    objective = str(cfg["objective"])
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset_name = str(cfg["dataset"])
    run_dir = make_run_dir(args.run_root, "mnist_bags", f"{method}_{objective}", dataset_name, seed)
    write_run_metadata(run_dir, {**cfg, "epochs": args.epochs, "batch_size": args.batch_size}, seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_ds = MNISTBags(
        root=cfg["dataset_root"],
        dataset=dataset_name,
        split="train",
        num_bags=int(cfg["train_bags"]),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        balanced_binary=bool(cfg["balanced_binary"]),
        seed=seed,
    )
    test_ds = MNISTBags(
        root=cfg["dataset_root"],
        dataset=dataset_name,
        split="test",
        num_bags=int(args.test_bags),
        bag_size=int(cfg["bag_size_mean"]),
        bag_size_std=float(cfg["bag_size_std"]),
        target_digit=int(cfg["target_digit"]),
        balanced_binary=True,
        seed=seed + 10_000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mnist_bags)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mnist_bags)

    model = ShuklaMNISTSelector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    best = {"bag_auc": -math.inf}
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in train_loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            counts = batch["count"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss, _ = _training_loss(
                logits=logits,
                probs=probs,
                mask=mask,
                counts=counts,
                method=method,
                objective=objective,
                entropy_weight=float(cfg["entropy_weight"]),
                em_temperature=float(cfg["em_temperature"]),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_seen += x.shape[0]

        metrics = _evaluate(model, test_loader, device, method)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mb = 0.0
        metrics.update({"epoch": epoch, "train_nll": total_loss / max(total_seen, 1)})
        metrics.update({"epoch_seconds": time.perf_counter() - epoch_start, "peak_cuda_mem_mb": peak_mb})
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"epoch={epoch:03d} train_nll={metrics['train_nll']:.4f} "
            f"test_nll={metrics['nll']:.4f} bag_auc={metrics['bag_auc']:.4f} "
            f"inst_auc={metrics['instance_auc']:.4f} "
            f"time={metrics['epoch_seconds']:.2f}s mem={metrics['peak_cuda_mem_mb']:.1f}MB"
        )
        if metrics["bag_auc"] > best["bag_auc"]:
            best = metrics
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, run_dir / "checkpoint_best.pt")

    summary = {"best": best, "config": cfg, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    suffix = method if objective == "nll" else f"{method}_{objective}"
    summary_path = results_dir / f"mnist_bags_{suffix}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
