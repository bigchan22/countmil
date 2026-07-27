#!/usr/bin/env python
"""Evaluate the deterministic C4+C11 aggregate group-prior baseline."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from countmil.criteo.constants import BAG_SIZE, PROTOCOL_TAG
from countmil.criteo.losses import aggregate_metrics, instance_metrics
from countmil.strict_protocol import canonical_hash
from countmil.training.run import git_commit


C4_INDEX = 3
C11_INDEX = 10


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _group_key_tensor(shard: dict[str, Any]) -> torch.Tensor:
    cats = shard["categorical"]
    keys = cats[:, 0, [C4_INDEX, C11_INDEX]].long()
    if not torch.equal(cats[:, :, C4_INDEX], keys[:, 0:1].expand(-1, cats.shape[1])):
        raise RuntimeError("C4 is not constant within at least one fixed group bag")
    if not torch.equal(cats[:, :, C11_INDEX], keys[:, 1:2].expand(-1, cats.shape[1])):
        raise RuntimeError("C11 is not constant within at least one fixed group bag")
    return keys


def _fit_group_rates(train: dict[str, Any]) -> tuple[dict[tuple[int, int], float], float]:
    sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    keys = _group_key_tensor(train)
    counts = train["counts"]
    for key, count in zip(keys.tolist(), counts.tolist()):
        k = (int(key[0]), int(key[1]))
        sums[k][0] += float(count)
        sums[k][1] += float(BAG_SIZE)
    rates = {k: click_sum / slots for k, (click_sum, slots) in sums.items()}
    global_rate = float(counts.sum().item() / (counts.numel() * BAG_SIZE))
    return rates, global_rate


def _predict_bag_probs(shard: dict[str, Any], rates: dict[tuple[int, int], float], global_rate: float) -> torch.Tensor:
    keys = _group_key_tensor(shard)
    bag_rates = [rates.get((int(k[0]), int(k[1])), global_rate) for k in keys.tolist()]
    return torch.tensor(bag_rates, dtype=torch.float32).reshape(-1, 1).expand(-1, BAG_SIZE).contiguous()


def _evaluate(shard: dict[str, Any], probs: torch.Tensor, raw_dir: Path) -> dict[str, float]:
    counts = shard["counts"].long()
    labels = shard["hidden_labels"].long()
    metrics = aggregate_metrics(probs, counts)
    bag_rows = []
    for i in range(counts.shape[0]):
        bag_rows.append(
            {
                "bag_id": i,
                "count": int(counts[i].item()),
                "expected_count": float(probs[i].sum().item()),
                "expected_count_mae": float(metrics["expected_count_mae"][i].item()),
                "rounded_expected_count_acc": float(metrics["rounded_expected_count_acc"][i].item()),
                "pmf_mode_count_acc": float(metrics["pmf_mode_count_acc"][i].item()),
                "poisson_binomial_nll": float(metrics["poisson_binomial_nll"][i].item()),
            }
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "raw_per_bag_predictions.jsonl").open("w") as f:
        for row in bag_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({"probs": probs.reshape(-1), "labels": labels.reshape(-1)}, raw_dir / "raw_test_probs_labels.pt")
    out = {k: float(v.mean().item()) for k, v in metrics.items()}
    out.update(instance_metrics(probs.reshape(-1), labels.reshape(-1)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--artifact-root", default="/home/chanhomin/datasets/criteo_x1/strictv1")
    p.add_argument("--result-root", default="results/rebuttal/4090_criteo_extra_baselines")
    args = p.parse_args()

    sha = git_commit()
    root = Path(args.artifact_root) / f"seed{args.seed}"
    train = _load(root / f"train_seed{args.seed}_shard.pt")
    valid = _load(root / f"valid_seed{args.seed}_shard.pt")
    test = _load(root / f"test_seed{args.seed}_shard.pt")
    if "hidden_labels" in train or "hidden_labels" in valid:
        raise RuntimeError("train/valid shards expose hidden labels")
    rates, global_rate = _fit_group_rates(train)
    val_probs = _predict_bag_probs(valid, rates, global_rate)
    val_mae = float((val_probs.sum(dim=1) - valid["counts"].float()).abs().mean().item())
    cfg = {
        "method": "group_prior",
        "seed": int(args.seed),
        "protocol_tag": PROTOCOL_TAG,
        "git_commit": sha,
        "train_manifest_hash": train["metadata"]["manifest_hash"],
        "valid_manifest_hash": valid["metadata"]["manifest_hash"],
        "test_manifest_hash": test["metadata"]["manifest_hash"],
        "train_input_shard_hash": train["metadata"]["input_shard_hash"],
        "valid_input_shard_hash": valid["metadata"]["input_shard_hash"],
        "test_input_shard_hash": test["metadata"]["input_shard_hash"],
        "checkpoint_metric": "validation expected-count MAE only",
        "hidden_validation_metrics_logged": False,
        "group_key": "C4+C11",
        "groups_fit": len(rates),
        "global_prior_from_aggregate_train_counts": global_rate,
    }
    cfg_hash = canonical_hash({"config": cfg})
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{PROTOCOL_TAG}_group_prior_s{args.seed}_{cfg_hash[:12]}_{sha[:8]}_{stamp}"
    run_dir = Path(args.result_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg | {"config_hash": cfg_hash, "run_id": run_id}, indent=2, sort_keys=True))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": "cpu",
            },
            indent=2,
            sort_keys=True,
        )
    )
    test_metrics = _evaluate(test, _predict_bag_probs(test, rates, global_rate), run_dir)
    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "config": cfg | {"config_hash": cfg_hash},
        "best": {"epoch": 0, "val_expected_count_mae": val_mae, "val_native_objective": val_mae},
        "test": test_metrics,
        "run_dir": str(run_dir),
        "selection_metric": "deterministic aggregate-only validation expected-count MAE; no hidden validation metrics",
    }
    Path(args.result_root).mkdir(parents=True, exist_ok=True)
    (Path(args.result_root) / f"group_prior_s{args.seed}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "COMPLETED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


if __name__ == "__main__":
    main()
