#!/usr/bin/env python
"""Summarize rebuttal overnight runs into CSV and Markdown tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any


ROOT = Path("results/rebuttal/overnight_4090")


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    return sum(vals) / len(vals), stdev(vals) if len(vals) > 1 else 0.0


def _load_jsons(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    return rows


def _collect_new_rows() -> list[dict[str, Any]]:
    completed = {p.stem for p in (ROOT / "job_status").glob("*.COMPLETED")}
    rows = []
    for p in sorted((ROOT / "scalar").glob("*.json")):
        obj = json.loads(p.read_text())
        cfg = obj["config"]
        test = obj["test"]
        key = p.name.removesuffix(".json")
        is_completed = any(key in marker for marker in completed)
        if not completed or is_completed:
            rows.append({"source": str(p), **cfg, "task": cfg["task"], "method": "Gaussian-AMLE", **test})
    for p in sorted((ROOT / "fixed_cifar10").glob("*.json")):
        obj = json.loads(p.read_text())
        cfg = obj["config"]
        test = obj["test"]
        key = p.name.removesuffix(".json")
        is_completed = any(key in marker for marker in completed)
        if not completed or is_completed:
            rows.append({"source": str(p), "task": "fixed_cifar10", "method": cfg["method"], **cfg, **test})
    return rows


def _write_summary_csv(rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    out = ROOT / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_group(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in group_keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        seeds = [v.get("seed") for v in vals]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {group_keys}: {key}")
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row["seeds"] = len(vals)
        for metric in metrics:
            mean, sd = _mean_sd([v.get(metric) for v in vals])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        out.append(row)
    return out


def _fmt(mean: float, sd: float, n: int) -> str:
    if not math.isfinite(mean):
        return "--"
    if n > 1:
        return f"{mean:.4f} ± {sd:.4f}"
    return f"{mean:.4f}"


def _old_digit_rows() -> list[dict[str, Any]]:
    # Existing FS-Conv and MSE summaries are local; use them as comparator rows.
    rows = []
    agg = Path("results/neurips_pilot_preleave_20260504_1619/aggregate.csv")
    if agg.exists():
        with agg.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("config.experiment") == "mnist_digit_sum" and r.get("config.train_bags") == "1000":
                    rows.append(
                        {
                            "task": "digit_sum",
                            "method": "FS-Conv",
                            "bag_size_mean": r["config.bag_size_mean"],
                            "train_bags": r["config.train_bags"],
                            "seed": r["config.seed"],
                            "expected_sum_mae": float(r["best.expected_sum_mae"]),
                            "rounded_sum_acc": float(r["best.sum_acc"]),
                            "rounded_sum_mae": float(r["best.sum_mae"]),
                            "instance_digit_acc": float(r["best.instance_digit_acc"]),
                        }
                    )
    return rows


def _old_signed_rows() -> list[dict[str, Any]]:
    rows = []
    for agg, method in [
        (Path("results/neurips_pilot_preleave_20260504_1619/aggregate.csv"), "FS-Conv"),
        (Path("results/server4090_signed_mse_baseline/aggregate.csv"), "MSE"),
    ]:
        if not agg.exists():
            continue
        with agg.open(newline="") as f:
            for r in csv.DictReader(f):
                if method == "FS-Conv" and r.get("config.experiment") != "signed_mnist":
                    continue
                if r.get("config.train_bags") != "1000":
                    continue
                cancel = r.get("config.cancellation_heavy")
                rows.append(
                    {
                        "task": "signed",
                        "method": method,
                        "bag_size_mean": r["config.bag_size_mean"],
                        "train_bags": r["config.train_bags"],
                        "cancellation_heavy": cancel,
                        "seed": r["config.seed"],
                        "expected_signed_count_mae": float(r.get("best.expected_signed_count_mae") or r.get("best.signed_count_mae")),
                        "rounded_signed_count_acc": float(r.get("best.rounded_signed_count_acc") or r.get("best.signed_count_acc")),
                        "instance_acc": float(r["best.instance_acc"]),
                        "instance_auc": float(r["best.instance_auc"]),
                    }
                )
    return rows


def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    new_rows = _collect_new_rows()
    _write_summary_csv(new_rows)

    failures = []
    for p in sorted((ROOT / "job_status").glob("*.FAILED")):
        failures.append(json.loads(p.read_text()))
    (ROOT / "failures.md").write_text(
        "# Failures\n\n"
        + ("\n".join(f"- `{f['job']['name']}` return code {f['returncode']} stderr `{f['stderr']}`" for f in failures) if failures else "No failed jobs recorded.\n")
    )

    inv = ["# Run Inventory\n"]
    for p in sorted((ROOT / "job_status").glob("*.*")):
        inv.append(f"- `{p.name}`")
    (ROOT / "run_inventory.md").write_text("\n".join(inv) + "\n")

    digit = _summarize_group(
        _old_digit_rows() + [r for r in new_rows if r["task"] == "digit_sum"],
        ["task", "method", "bag_size_mean", "train_bags"],
        ["expected_sum_mae", "rounded_sum_acc", "rounded_sum_mae", "instance_digit_acc"],
    )
    signed = _summarize_group(
        _old_signed_rows() + [r for r in new_rows if r["task"] == "signed"],
        ["task", "method", "bag_size_mean", "train_bags", "cancellation_heavy"],
        ["expected_signed_count_mae", "rounded_signed_count_acc", "instance_acc", "instance_auc"],
    )
    cifar = _summarize_group(
        [r for r in new_rows if r["task"] == "fixed_cifar10"],
        ["task", "method", "bag_size", "train_bags", "alpha"],
        ["hist_count_mae", "hist_proportion_mae", "composite_count_nll", "instance_acc", "macro_f1"],
    )

    def compact(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
        out = []
        metric_fields = {f"{m}_mean" for m in metrics} | {f"{m}_sd" for m in metrics}
        for r in rows:
            row = {k: r[k] for k in r if k not in metric_fields}
            n = int(r["seeds"])
            for m in metrics:
                row[m] = _fmt(r[f"{m}_mean"], r[f"{m}_sd"], n)
            out.append(row)
        return out

    eq = _load_jsons([ROOT / "pvc_equivalence.json"])
    md = [
        "# Rebuttal Tables",
        "",
        "All `±` values are sample standard deviations with `ddof=1`. Tables only include completed runs and compatible protocols.",
        "",
        "## Table R1: Digit Sum",
        _table(compact(digit, ["expected_sum_mae", "rounded_sum_acc", "rounded_sum_mae", "instance_digit_acc"]), ["method", "bag_size_mean", "train_bags", "seeds", "expected_sum_mae", "rounded_sum_acc", "rounded_sum_mae", "instance_digit_acc"]),
        "",
        "## Table R2: Signed Count",
        _table(compact(signed, ["expected_signed_count_mae", "rounded_signed_count_acc", "instance_acc", "instance_auc"]), ["method", "bag_size_mean", "train_bags", "cancellation_heavy", "seeds", "expected_signed_count_mae", "rounded_signed_count_acc", "instance_acc", "instance_auc"]),
        "",
        "## Table R3: Fixed-Bag CIFAR-10",
        _table(compact(cifar, ["hist_count_mae", "hist_proportion_mae", "composite_count_nll", "instance_acc", "macro_f1"]), ["method", "bag_size", "train_bags", "alpha", "seeds", "hist_count_mae", "hist_proportion_mae", "composite_count_nll", "instance_acc", "macro_f1"]),
        "",
        "## Table R4: FS-Conv vs LLP-PVC Count Component Equivalence",
    ]
    if eq:
        md.append(_table(eq, ["batch_size", "bag_size", "classes", "max_abs_forward_loss_diff", "max_abs_per_class_probability_diff", "max_abs_logit_gradient_diff"]))
    else:
        md.append("No completed equivalence result yet.")
    (ROOT / "rebuttal_tables.md").write_text("\n\n".join(md) + "\n")


if __name__ == "__main__":
    main()
