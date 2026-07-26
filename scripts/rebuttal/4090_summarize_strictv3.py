#!/usr/bin/env python
"""Summarize strict-v3 4090 rebuttal outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/rebuttal/4090_strictv3")


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, sd


def _pm(vals: list[float]) -> str:
    mean, sd = _mean_sd(vals)
    if not math.isfinite(mean):
        return "-"
    return f"{mean:.3f} +/- {sd:.3f}"


def _load(pattern: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(ROOT.glob(pattern)):
        if "/smoke/" in str(path):
            continue
        try:
            js = json.loads(path.read_text())
        except Exception:
            continue
        js["_path"] = str(path)
        out.append(js)
    return out


def _validate_shared(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    groups: dict[tuple[str, int], dict[str, set[str]]] = {}
    for row in rows:
        cfg = row["config"]
        task = cfg.get("task") or cfg.get("method")
        seed = int(cfg["seed"])
        key = (task, seed)
        groups.setdefault(key, {"train": set(), "val": set(), "test": set()})
        hashes = row.get("manifest_hashes", {})
        for split in ["train", "val", "test"]:
            groups[key][split].add(hashes.get(split) or cfg.get(f"{split}_manifest_hash", ""))
        if row.get("selection_metric", "").lower().find("hidden") >= 0:
            if "no hidden" not in row.get("selection_metric", "").lower():
                errors.append(f"hidden-looking selection text in {row['_path']}")
    for key, split_hashes in groups.items():
        for split, hashes in split_hashes.items():
            hashes.discard("")
            if len(hashes) > 1:
                errors.append(f"manifest mismatch for {key} split {split}: {sorted(hashes)}")
    return errors


def _write_scalar() -> None:
    rows = _load("digit_sum/digit_sum_*_s*.json")
    errors = _validate_shared(rows)
    lines = [
        "# Strict-v3 Digit-Sum Table",
        "",
        "MNIST digit sum, mean bag size 10, 1000 fixed training bags. Mean +/- sample standard deviation over seeds 0,1,2.",
        "Validation used aggregate metrics only; hidden digit accuracy is final-test only.",
        "",
        "| Method | Seeds | Expected-sum MAE | Rounded expected acc. | PMF-mode acc. | Discrete aggregate NLL | Instance digit acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["mse", "gaussian_amle", "fsconv"]:
        vals = [r for r in rows if r["config"]["method"] == method]
        if not vals:
            lines.append(f"| {method} | 0 | - | - | - | - | - |")
            continue
        t = [r["test"] for r in vals]
        lines.append(
            f"| {method} | {len(vals)} | {_pm([x['expected_sum_mae'] for x in t])} | "
            f"{_pm([x['rounded_expected_sum_acc'] for x in t])} | {_pm([x['pmf_mode_sum_acc'] for x in t])} | "
            f"{_pm([x['posthoc_discrete_aggregate_nll'] for x in t])} | {_pm([x['instance_digit_acc'] for x in t])} |"
        )
    if errors:
        lines += ["", "Validation warnings:"] + [f"- {e}" for e in errors]
    (ROOT / "scalar_table.md").write_text("\n".join(lines) + "\n")


def _write_signed() -> None:
    rows = _load("signed_random/signed_random_*_s*.json") + _load("signed_cancellation/signed_cancellation_*_s*.json")
    errors = _validate_shared(rows)
    lines = [
        "# Strict-v3 Signed MNIST Table",
        "",
        "Signed target-digit MNIST, mean bag size 10, 1000 fixed training bags. Mean +/- sample standard deviation over seeds 0,1,2.",
        "",
        "| Protocol | Method | Seeds | Expected signed MAE | Rounded MAE | Rounded acc. | PMF-mode acc. | Discrete NLL | Instance acc. | Instance AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ["signed_random", "signed_cancellation"]:
        for method in ["mse", "gaussian_amle", "fsconv"]:
            vals = [r for r in rows if r["config"]["task"] == task and r["config"]["method"] == method]
            if not vals:
                lines.append(f"| {task} | {method} | 0 | - | - | - | - | - | - | - |")
                continue
            t = [r["test"] for r in vals]
            lines.append(
                f"| {task} | {method} | {len(vals)} | {_pm([x['expected_signed_sum_mae'] for x in t])} | "
                f"{_pm([x['rounded_signed_sum_mae'] for x in t])} | {_pm([x['rounded_signed_sum_acc'] for x in t])} | "
                f"{_pm([x['pmf_mode_signed_sum_acc'] for x in t])} | {_pm([x['posthoc_discrete_aggregate_nll'] for x in t])} | "
                f"{_pm([x['instance_acc'] for x in t])} | {_pm([x['instance_auc'] for x in t])} |"
            )
    if errors:
        lines += ["", "Validation warnings:"] + [f"- {e}" for e in errors]
    (ROOT / "signed_table.md").write_text("\n".join(lines) + "\n")


def _write_cifar() -> None:
    rows = _load("cifar_fixed/cifar10_*_s*.json")
    errors = _validate_shared(rows)
    lines = [
        "# Strict-v3 Fixed-Bag CIFAR-10 Table",
        "",
        "Frozen ImageNet ResNet-18 features; 45k/5k stratified train/validation split from official train; official test split final-only.",
        "Bag size 64, 250 train bags, 250 validation bags, 1000 test bags. Mean +/- sample standard deviation over seeds 0,1,2.",
        "",
        "| Method | Seeds | Unique-image acc. | Macro-F1 | Count MAE | Classwise composite NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"ce_kl": "CE/KL proportion matching", "fsconv_count": "FS-Conv classwise count likelihood", "official_llp_pvc": "Full official LLP-PVC"}
    for method in ["ce_kl", "official_llp_pvc", "fsconv_count"]:
        vals = [r for r in rows if r["config"]["method"] == method]
        if not vals:
            lines.append(f"| {labels[method]} | 0 | - | - | - | - |")
            continue
        t = [r["test"] for r in vals]
        lines.append(
            f"| {labels[method]} | {len(vals)} | {_pm([x['unique_image_instance_acc'] for x in t])} | "
            f"{_pm([x['unique_image_macro_f1'] for x in t])} | {_pm([x['hist_count_mae'] for x in t])} | "
            f"{_pm([x['composite_count_nll'] for x in t])} |"
        )
    if errors:
        lines += ["", "Validation warnings:"] + [f"- {e}" for e in errors]
    (ROOT / "cifar_table.md").write_text("\n".join(lines) + "\n")


def _paired_stats() -> None:
    out_dir = ROOT / "paired_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["ce_kl", "fsconv_count", "official_llp_pvc"]
    by_method = {m: _load(f"cifar_fixed/cifar10_{m}_s*.json") for m in methods}
    lines = ["# Strict-v3 CIFAR Paired Statistics", ""]
    if not all(len(by_method[m]) == 3 for m in methods):
        lines.append("Paired bootstrap not run because at least one CIFAR method lacks three completed seeds.")
        (ROOT / "paired_statistics.md").write_text("\n".join(lines) + "\n")
        return

    def raw_rows(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        path = Path(row["run_dir"]) / "raw_per_bag_test_metrics.jsonl"
        vals = {"count_mae": [], "composite_nll": []}
        with path.open() as f:
            for line in f:
                js = json.loads(line)
                vals["count_mae"].append(float(js["count_mae"]))
                vals["composite_nll"].append(float(js["composite_nll"]))
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in vals.items()}

    rows_by_method_seed = {
        method: {int(row["config"]["seed"]): row for row in rows}
        for method, rows in by_method.items()
    }
    seeds = [0, 1, 2]
    raw = {
        method: {seed: raw_rows(rows_by_method_seed[method][seed]) for seed in seeds}
        for method in methods
    }
    errors = []
    for seed in seeds:
        hashes = {
            method: rows_by_method_seed[method][seed]["manifest_hashes"]["test"]
            for method in methods
        }
        if len(set(hashes.values())) != 1:
            errors.append(f"seed {seed} test manifest mismatch: {hashes}")
        lengths = {
            method: int(raw[method][seed]["count_mae"].numel())
            for method in methods
        }
        if len(set(lengths.values())) != 1:
            errors.append(f"seed {seed} raw bag-count mismatch: {lengths}")
    if errors:
        lines += ["Bootstrap not run because paired inputs failed validation:"] + [f"- {e}" for e in errors]
        (ROOT / "paired_statistics.md").write_text("\n".join(lines) + "\n")
        (out_dir / "paired_statistics.json").write_text(json.dumps({"errors": errors}, indent=2, sort_keys=True))
        return

    def interval(vals: torch.Tensor) -> dict[str, float]:
        sorted_vals = vals.sort().values
        lo = sorted_vals[int(0.025 * (len(sorted_vals) - 1))].item()
        hi = sorted_vals[int(0.975 * (len(sorted_vals) - 1))].item()
        return {"mean": vals.mean().item(), "ci95_low": lo, "ci95_high": hi}

    def hierarchical(a: str, b: str, metric: str, reps: int = 10_000) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(8675309 + sum(ord(c) for c in a + b + metric))
        per_seed = []
        for seed in seeds:
            per_seed.append(raw[a][seed][metric] - raw[b][seed][metric])
        boot = torch.empty(reps, dtype=torch.float64)
        for r in range(reps):
            seed_pick = torch.randint(0, len(seeds), (len(seeds),), generator=gen)
            pieces = []
            for idx in seed_pick.tolist():
                arr = per_seed[idx]
                bag_pick = torch.randint(0, arr.numel(), (arr.numel(),), generator=gen)
                pieces.append(arr[bag_pick].mean())
            boot[r] = torch.stack(pieces).mean()
        seed_diag = {}
        for seed, arr in zip(seeds, per_seed):
            vals = torch.empty(reps, dtype=torch.float64)
            for r in range(reps):
                bag_pick = torch.randint(0, arr.numel(), (arr.numel(),), generator=gen)
                vals[r] = arr[bag_pick].mean()
            seed_diag[str(seed)] = interval(vals)
        return {"hierarchical": interval(boot), "per_seed": seed_diag}

    comparisons = [
        ("fsconv_count", "ce_kl"),
        ("official_llp_pvc", "ce_kl"),
        ("fsconv_count", "official_llp_pvc"),
    ]
    metrics = ["count_mae", "composite_nll"]
    stats: dict[str, Any] = {
        "description": "Paired differences are method_a - method_b. Bootstrap resamples seeds, then paired test bags within seed.",
        "seeds": seeds,
        "bootstrap_replicates": 10_000,
        "comparisons": {},
    }
    for a, b in comparisons:
        key = f"{a}_minus_{b}"
        stats["comparisons"][key] = {metric: hierarchical(a, b, metric) for metric in metrics}

    (out_dir / "paired_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    lines += [
        "Paired differences are method A - method B; negative is better for Count MAE and Composite NLL.",
        "Bootstrap uses 10,000 replicates, resampling seeds and then paired test bags within each sampled seed.",
        "",
        "| Comparison | Metric | Mean diff. | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for key, by_metric in stats["comparisons"].items():
        label = key.replace("_", " ")
        for metric, result in by_metric.items():
            h = result["hierarchical"]
            lines.append(f"| {label} | {metric} | {h['mean']:.4f} | [{h['ci95_low']:.4f}, {h['ci95_high']:.4f}] |")
    (ROOT / "paired_statistics.md").write_text("\n".join(lines) + "\n")


def _inventory() -> None:
    rows = []
    for p in sorted(ROOT.glob("**/*.json")):
        if "/smoke/" in str(p):
            continue
        try:
            js = json.loads(p.read_text())
        except Exception:
            continue
        cfg = js.get("config", {})
        rows.append(
            {
                "path": str(p),
                "task": cfg.get("task", ""),
                "method": cfg.get("method", ""),
                "seed": cfg.get("seed", ""),
                "protocol_tag": cfg.get("protocol_tag", ""),
                "git_commit": cfg.get("git_commit", ""),
                "config_hash": cfg.get("config_hash", ""),
                "train_manifest_hash": cfg.get("train_manifest_hash", ""),
                "val_manifest_hash": cfg.get("val_manifest_hash", ""),
                "test_manifest_hash": cfg.get("test_manifest_hash", ""),
                "run_dir": js.get("run_dir", ""),
            }
        )
    with (ROOT / "run_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_scalar()
    _write_signed()
    _write_cifar()
    _paired_stats()
    _inventory()
    print("wrote strict-v3 summaries")


if __name__ == "__main__":
    main()
