#!/usr/bin/env python
"""Summarize strict-v1 Criteo runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/rebuttal/4090_criteo_strictv1")
METHODS = ["dllp_bce", "dllp_mse", "easyllp", "fsconv"]
LABELS = {"dllp_bce": "DLLP-BCE", "dllp_mse": "DLLP-MSE", "easyllp": "EasyLLP", "fsconv": "FS-Conv"}


def _load() -> list[dict[str, Any]]:
    out = []
    for p in sorted(ROOT.glob("*_s*.json")):
        try:
            js = json.loads(p.read_text())
        except Exception:
            continue
        js["_path"] = str(p)
        out.append(js)
    return out


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, sd


def _pm(vals: list[float]) -> str:
    m, s = _mean_sd(vals)
    return "-" if not math.isfinite(m) else f"{m:.4f} +/- {s:.4f}"


def main() -> None:
    rows = _load()
    ROOT.mkdir(parents=True, exist_ok=True)
    inv_path = ROOT / "run_inventory.csv"
    with inv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "status", "run_id", "summary_path", "selected_epoch", "val_expected_count_mae", "test_roc_auc", "test_count_mae"])
        for r in rows:
            cfg, best, test = r["config"], r["best"], r["test"]
            w.writerow([cfg["method"], cfg["seed"], r["status"], r["run_id"], r["_path"], best["epoch"], best["val_expected_count_mae"], test["roc_auc"], test["expected_count_mae"]])

    instance_lines = [
        "# Criteo Strict-v1 Instance Metrics",
        "",
        "LLP-Bench-style Criteo feature-bag experiment. Mean +/- sample standard deviation over seeds 0,1,2.",
        "",
        "| Method | Seeds | ROC-AUC | PR-AUC | Log loss | Brier | ECE-15 | Acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregate_lines = [
        "# Criteo Strict-v1 Aggregate Metrics",
        "",
        "All aggregate metrics are common post-hoc evaluations using each method's instance probabilities.",
        "",
        "| Method | Seeds | Count MAE | Rounded count acc. | PMF-mode acc. | Exact count NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    instance_csv = [["method", "seeds", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "pr_auc_sd", "log_loss_mean", "log_loss_sd", "brier_mean", "brier_sd", "ece_15_mean", "ece_15_sd"]]
    aggregate_csv = [["method", "seeds", "count_mae_mean", "count_mae_sd", "rounded_acc_mean", "rounded_acc_sd", "pmf_mode_acc_mean", "pmf_mode_acc_sd", "nll_mean", "nll_sd"]]
    failures = ["# Criteo Strict-v1 Failures", ""]
    for method in METHODS:
        vals = [r for r in rows if r["config"]["method"] == method and r["status"] == "COMPLETED"]
        t = [r["test"] for r in vals]
        instance_lines.append(
            f"| {LABELS[method]} | {len(vals)} | {_pm([x['roc_auc'] for x in t])} | {_pm([x['pr_auc'] for x in t])} | "
            f"{_pm([x['log_loss'] for x in t])} | {_pm([x['brier'] for x in t])} | {_pm([x['ece_15'] for x in t])} | {_pm([x['accuracy_0p5'] for x in t])} |"
        )
        aggregate_lines.append(
            f"| {LABELS[method]} | {len(vals)} | {_pm([x['expected_count_mae'] for x in t])} | {_pm([x['rounded_expected_count_acc'] for x in t])} | "
            f"{_pm([x['pmf_mode_count_acc'] for x in t])} | {_pm([x['poisson_binomial_nll'] for x in t])} |"
        )
        if len(vals) != 3:
            failures.append(f"- {method}: expected 3 completed seeds, found {len(vals)}.")
        def ms(key: str) -> tuple[float, float]:
            return _mean_sd([x[key] for x in t])
        roc = ms("roc_auc"); pr = ms("pr_auc"); ll = ms("log_loss"); br = ms("brier"); ece = ms("ece_15")
        instance_csv.append([method, len(vals), roc[0], roc[1], pr[0], pr[1], ll[0], ll[1], br[0], br[1], ece[0], ece[1]])
        mae = ms("expected_count_mae"); racc = ms("rounded_expected_count_acc"); macc = ms("pmf_mode_count_acc"); nll = ms("poisson_binomial_nll")
        aggregate_csv.append([method, len(vals), mae[0], mae[1], racc[0], racc[1], macc[0], macc[1], nll[0], nll[1]])
    (ROOT / "instance_table.md").write_text("\n".join(instance_lines) + "\n")
    (ROOT / "aggregate_table.md").write_text("\n".join(aggregate_lines) + "\n")
    with (ROOT / "instance_table.csv").open("w", newline="") as f:
        csv.writer(f).writerows(instance_csv)
    with (ROOT / "aggregate_table.csv").open("w", newline="") as f:
        csv.writer(f).writerows(aggregate_csv)
    (ROOT / "failures.md").write_text("\n".join(failures) + "\n")
    report = [
        "# Criteo Strict-v1 Final Report",
        "",
        "This is an LLP-Bench-style Criteo feature-bag experiment, not an exact full LLP-Bench reproduction.",
        "Checkpoint selection used aggregate validation expected-count MAE only; hidden validation metrics were not computed.",
        "",
        "## Instance Metrics",
        "",
        *instance_lines[4:],
        "",
        "## Aggregate Metrics",
        "",
        *aggregate_lines[4:],
    ]
    Path("docs/rebuttal/criteo_STRICTV1_FINAL_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

