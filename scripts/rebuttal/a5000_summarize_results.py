#!/usr/bin/env python
"""Write A5000 rebuttal inventories, tables, failures, and final reports."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rebuttal/a5000_svhn_dependence"
DOCS = ROOT / "docs/rebuttal"


def load_jsons(path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def fmt(values: list[float], digits: int = 3) -> str:
    m, s = mean_sd(values)
    if m != m:
        return "missing"
    if s != s:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svhn_outputs(summaries: list[dict[str, Any]]) -> None:
    summaries = [
        s
        for s in summaries
        if s.get("config", {}).get("protocol_version") == "strictv2_train_holdout_val_no_hidden_val_metrics"
    ]
    rows = []
    for s in summaries:
        cfg = s["config"]
        test = s["test"]
        meta = s["metadata"]
        rows.append(
            {
                "method": cfg["method"],
                "seed": cfg["seed"],
                "status": s["status"],
                "git_commit": meta["git_commit"],
                "run_dir": s["run_dir"],
                "train_manifest_sha256": s["manifest_hashes"]["train"],
                "val_manifest_sha256": s["manifest_hashes"]["val"],
                "test_manifest_sha256": s["manifest_hashes"]["test"],
                "expected_sum_mae": test["expected_sum_mae"],
                "rounded_sum_acc": test["rounded_expected_sum_acc"],
                "pmf_mode_acc": test["pmf_mode_sum_acc"],
                "fsconv_discrete_nll": test["fsconv_discrete_nll"],
                "gaussian_bin_nll": test["gaussian_bin_nll"],
                "instance_acc": test["instance_digit_acc"],
            }
        )
    write_csv(OUT / "svhn_run_inventory.csv", rows)

    methods = [("mse", "MSE"), ("gaussian_amle", "Gaussian-AMLE"), ("fsconv", "FS-Conv")]
    table_rows = []
    for method, label in methods:
        group = [r for r in rows if r["method"] == method and int(r["seed"]) in {0, 1, 2}]
        table_rows.append(
            {
                "Method": label,
                "Expected-sum MAE": fmt([float(r["expected_sum_mae"]) for r in group]),
                "Rounded sum acc.": fmt([float(r["rounded_sum_acc"]) for r in group]),
                "PMF mode acc.": "—" if method == "mse" else fmt([float(r["pmf_mode_acc"]) for r in group]),
                "Instance acc.": fmt([float(r["instance_acc"]) for r in group]),
            }
        )
    write_csv(OUT / "svhn_rebuttal_table.csv", table_rows)
    md = ["| Method | Expected-sum MAE | Rounded sum acc. | PMF mode acc. | Instance acc. |", "|---|---:|---:|---:|---:|"]
    md += [f"| {r['Method']} | {r['Expected-sum MAE']} | {r['Rounded sum acc.']} | {r['PMF mode acc.']} | {r['Instance acc.']} |" for r in table_rows]
    md.append("")
    md.append("`+/-` denotes sample standard deviation over completed seeds 0, 1, and 2. Missing entries are not inferred.")
    (OUT / "svhn_rebuttal_table.md").write_text("\n".join(md) + "\n")

    DOCS.mkdir(parents=True, exist_ok=True)
    complete = all(len([r for r in rows if r["method"] == method and int(r["seed"]) in {0, 1, 2}]) == 3 for method, _ in methods)
    report = [
        "# A5000 SVHN Final Report",
        "",
        f"Status: {'complete' if complete else 'incomplete'}; no missing results are fabricated.",
        "",
        "Protocol: pretrained SVHN digit-sum, bag size distribution mean 10/std 2/min 5/max 15, 5000 training bags, seeds 0/1/2, aggregate-only validation checkpoint selection.",
        "",
        "Metrics distinguish expected-sum rounding from FS-Conv PMF mode. Gaussian-AMLE reports integer-bin Gaussian NLL for discrete evaluation.",
        "",
        (OUT / "svhn_rebuttal_table.md").read_text(),
    ]
    (DOCS / "a5000_SVHN_FINAL_REPORT.md").write_text("\n".join(report))


def dependence_outputs(summaries: list[dict[str, Any]]) -> None:
    rows = []
    for s in summaries:
        cfg = s["config"]
        test = s["test"]
        stats = s["dgp_stats"]["test"]
        rows.append(
            {
                "tau": cfg["tau"],
                "method": cfg["method"],
                "seed": cfg["seed"],
                "status": s["status"],
                "git_commit": s["metadata"]["git_commit"],
                "run_dir": s["run_dir"],
                "test_manifest_sha256": s["manifest_hashes"]["test"],
                "prevalence": stats["prevalence"],
                "empirical_correlation": stats["within_bag_correlation"],
                "count_mae": test["count_mae"],
                "rounded_count_acc": test["rounded_count_acc"],
                "auc": test["instance_auc"],
                "instance_acc_threshold_0p5": test["instance_acc_threshold_0p5"],
                "fsconv_aggregate_nll": test["fsconv_aggregate_nll"],
                "gaussian_bin_nll": test["gaussian_bin_nll"],
                "coverage_90": test["coverage_90"],
                "mean_predicted_aggregate_variance": test["mean_predicted_aggregate_variance"],
                "empirical_aggregate_residual_variance": test["empirical_aggregate_residual_variance"],
                "coverage_gap_90": test["coverage_gap_90"],
            }
        )
    write_csv(OUT / "dependence_run_inventory.csv", rows)

    methods = [("mse", "MSE"), ("gaussian_amle", "Gaussian-AMLE"), ("fsconv", "FS-Conv")]
    taus = [0.0, 0.5, 1.0, 2.0]
    table_rows = []
    for tau in taus:
        tau_rows = [r for r in rows if abs(float(r["tau"]) - tau) < 1e-9]
        corr = fmt([float(r["empirical_correlation"]) for r in tau_rows], 3)
        for method, label in methods:
            group = [r for r in tau_rows if r["method"] == method and int(r["seed"]) in {0, 1, 2}]
            table_rows.append(
                {
                    "tau": tau,
                    "Empirical correlation": corr,
                    "Method": label,
                    "Count MAE": fmt([float(r["count_mae"]) for r in group]),
                    "AUC": fmt([float(r["auc"]) for r in group]),
                    "NLL": fmt([float(r["fsconv_aggregate_nll"] if method == "fsconv" else r["gaussian_bin_nll"]) for r in group]),
                    "90% coverage": fmt([float(r["coverage_90"]) for r in group]),
                }
            )
    write_csv(OUT / "dependence_table.csv", table_rows)
    md = ["| tau | Empirical correlation | Method | Count MAE | AUC | NLL | 90% coverage |", "|---:|---:|---|---:|---:|---:|---:|"]
    md += [f"| {r['tau']:.1f} | {r['Empirical correlation']} | {r['Method']} | {r['Count MAE']} | {r['AUC']} | {r['NLL']} | {r['90% coverage']} |" for r in table_rows]
    md.append("")
    md.append("`+/-` denotes sample standard deviation over completed seeds 0, 1, and 2. Missing entries are not inferred.")
    (OUT / "dependence_table.md").write_text("\n".join(md) + "\n")

    complete = all(len([r for r in rows if abs(float(r["tau"]) - tau) < 1e-9 and r["method"] == method and int(r["seed"]) in {0, 1, 2}]) == 3 for tau in taus for method, _ in methods)
    report = [
        "# A5000 Dependence Final Report",
        "",
        f"Status: {'complete' if complete else 'incomplete'}; no missing results are fabricated.",
        "",
        "Synthetic setup: d=10, n=32, fixed train/validation/test manifests, tau in {0.0, 0.5, 1.0, 2.0}, seeds 0/1/2.",
        "",
        "The purpose is sensitivity characterization under increasing hidden bag-level dependence. The natural extension for failures is conditioning on sufficient observed bag context or integrating a bag-level latent variable; that extension is not implemented here.",
        "",
        (OUT / "dependence_table.md").read_text(),
    ]
    (DOCS / "a5000_DEPENDENCE_FINAL_REPORT.md").write_text("\n".join(report))


def failures_output() -> None:
    rows = []
    status_dir = OUT / "scheduler/status"
    for path in sorted(status_dir.glob("*.json")):
        status = json.loads(path.read_text())
        if status.get("status") != "COMPLETED":
            rows.append(status)
    lines = ["# A5000 Failures", ""]
    if not rows:
        lines.append("No scheduler failures recorded.")
    for row in rows:
        lines.extend(
            [
                f"## {row.get('job_key', path.stem)}",
                "",
                f"- status: {row.get('status')}",
                f"- returncode: {row.get('returncode')}",
                f"- log: {row.get('log_path')}",
                "",
            ]
        )
    (OUT / "failures.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svhn_outputs(load_jsons(OUT / "svhn_summaries"))
    dependence_outputs(load_jsons(OUT / "dependence_summaries"))
    failures_output()


if __name__ == "__main__":
    main()
