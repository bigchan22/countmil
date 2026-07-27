#!/usr/bin/env python
"""Reconstruct masked strict-v3 SVHN tables from raw prediction files."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rebuttal/4090_svhn_masked_strictv3"
DOCS = ROOT / "docs/rebuttal"
PROTOCOL = "strictv3_train_holdout_val_official_test_no_hidden_val_masked_resnet_bag_forward"
METHODS = [("mse", "MSE"), ("gaussian_amle", "Gaussian-AMLE"), ("fsconv", "FS-Conv")]


def load_jsons(path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(values: list[float], digits: int = 3) -> str:
    m, s = mean_sd(values)
    if not math.isfinite(m):
        return "missing"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def read_bag_metrics(run_dir: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (run_dir / "raw/test_bag_predictions.jsonl").read_text().splitlines()]
    n = len(rows)
    if n != 600:
        raise RuntimeError(f"expected 600 raw test bags in {run_dir}, found {n}")
    return {
        "expected_sum_mae": sum(abs(float(r["expected_sum"]) - float(r["target_sum"])) for r in rows) / n,
        "rounded_expected_sum_acc": sum(int(r["rounded_expected_sum"]) == int(r["target_sum"]) for r in rows) / n,
        "pmf_mode_sum_acc": sum(int(r["pmf_mode_sum"]) == int(r["target_sum"]) for r in rows) / n,
        "fsconv_discrete_nll": sum(float(r["fsconv_nll"]) for r in rows) / n,
        "gaussian_bin_nll": sum(float(r["gaussian_bin_nll"]) for r in rows) / n,
    }


def read_unique_instance_acc(run_dir: Path) -> float:
    path = run_dir / "raw/test_unique_instance_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise RuntimeError(f"empty unique-instance prediction file: {path}")
    return sum(int(r["predicted_digit"]) == int(r["true_digit"]) for r in rows) / len(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summaries = load_jsons(OUT / "svhn_summaries")
    rows = []
    failures = ["# 4090 SVHN Masked Strict-v3 Failures", ""]
    for s in summaries:
        cfg = s["config"]
        if cfg.get("protocol_version") != PROTOCOL:
            continue
        run_dir = Path(s["run_dir"])
        try:
            bag = read_bag_metrics(run_dir)
            inst_acc = read_unique_instance_acc(run_dir)
        except Exception as exc:
            failures.append(f"- `{run_dir}` reconstruction failed: `{exc}`")
            continue
        rows.append(
            {
                "method": cfg["method"],
                "seed": int(cfg["seed"]),
                "status": s["status"],
                "git_commit": s["metadata"]["git_commit"],
                "run_dir": s["run_dir"],
                "train_manifest_sha256": s["manifest_hashes"]["train"],
                "val_manifest_sha256": s["manifest_hashes"]["val"],
                "test_manifest_sha256": s["manifest_hashes"]["test"],
                "svhn_train_md5": s["metadata"]["svhn_train_md5"],
                "svhn_test_md5": s["metadata"]["svhn_test_md5"],
                "svhn_train_sha256": s["metadata"]["svhn_train_sha256"],
                "svhn_test_sha256": s["metadata"]["svhn_test_sha256"],
                "pretrained_checkpoint_sha256": s["metadata"].get("pretrained_checkpoint_sha256"),
                "selected_epoch": s["best"]["epoch"],
                "val_selection_loss": s["best"]["val_selection_loss"],
                "expected_sum_mae": bag["expected_sum_mae"],
                "rounded_expected_sum_acc": bag["rounded_expected_sum_acc"],
                "pmf_mode_sum_acc": bag["pmf_mode_sum_acc"],
                "fsconv_discrete_nll": bag["fsconv_discrete_nll"],
                "gaussian_bin_nll": bag["gaussian_bin_nll"],
                "instance_digit_acc": inst_acc,
            }
        )
    write_csv(OUT / "svhn_masked_run_inventory.csv", rows)

    table_rows = []
    for method, label in METHODS:
        group = sorted([r for r in rows if r["method"] == method and r["seed"] in {0, 1, 2}], key=lambda r: r["seed"])
        if len(group) != 3:
            failures.append(f"- `{method}` expected 3 seeds, found {len(group)}.")
        table_rows.append(
            {
                "Method": label,
                "Seeds": len(group),
                "Expected-sum MAE": fmt([float(r["expected_sum_mae"]) for r in group]),
                "Rounded sum acc.": fmt([float(r["rounded_expected_sum_acc"]) for r in group]),
                "PMF mode acc.": fmt([float(r["pmf_mode_sum_acc"]) for r in group]),
                "FS-Conv NLL": fmt([float(r["fsconv_discrete_nll"]) for r in group]),
                "Gaussian-bin NLL": fmt([float(r["gaussian_bin_nll"]) for r in group]),
                "Instance acc.": fmt([float(r["instance_digit_acc"]) for r in group]),
            }
        )
    write_csv(OUT / "svhn_masked_rebuttal_table.csv", table_rows)
    md = [
        "# 4090 SVHN Masked Strict-v3 Final Table",
        "",
        "`+/-` denotes sample standard deviation over seeds 0, 1, and 2. Metrics are reconstructed from raw final-test predictions.",
        "",
        "| Method | Seeds | Expected-sum MAE | Rounded sum acc. | PMF mode acc. | FS-Conv NLL | Gaussian-bin NLL | Instance acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    md += [
        f"| {r['Method']} | {r['Seeds']} | {r['Expected-sum MAE']} | {r['Rounded sum acc.']} | {r['PMF mode acc.']} | {r['FS-Conv NLL']} | {r['Gaussian-bin NLL']} | {r['Instance acc.']} |"
        for r in table_rows
    ]
    (OUT / "svhn_masked_rebuttal_table.md").write_text("\n".join(md) + "\n")

    if len(rows) == 9 and len(failures) == 2:
        failures.append("No failures recorded.")
    (OUT / "failures.md").write_text("\n".join(failures) + "\n")

    docs = [
        "# 4090 SVHN Masked Strict-v3 Final Audit",
        "",
        f"- Protocol version: `{PROTOCOL}`.",
        "- Defect repaired: padded zero images in 5-D variable-length bags are not forwarded through ResNet-18/BatchNorm.",
        "- Training and bag-level validation/test evaluation use `forward_valid_instances(model, instances, mask)` followed by softmax.",
        "- Flat unpadded unique-instance final-test evaluation remains unchanged.",
        "- Validation metrics are aggregate-only; hidden instance labels are evaluated only after checkpoint selection on the final test split.",
        "- No old checkpoints or old result summaries are reused in this table.",
        "- All rows are reconstructed from raw final-test prediction files.",
        "",
        "## Final Table",
        "",
        *md[4:],
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "4090_SVHN_MASKED_STRICTV3_FINAL_AUDIT.md").write_text("\n".join(docs) + "\n")


if __name__ == "__main__":
    main()
