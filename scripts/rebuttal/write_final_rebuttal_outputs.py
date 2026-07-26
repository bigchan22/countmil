#!/usr/bin/env python
"""Write compact final rebuttal result tables from completed artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path("results/rebuttal")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(x: str | float | int | None, digits: int = 3) -> str:
    if x in {None, ""}:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _pm(row: dict[str, str], metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean")
    sd = row.get(f"{metric}_sd")
    if mean in {None, ""}:
        return "-"
    if sd in {None, ""}:
        return _fmt(mean, digits)
    return f"{_fmt(mean, digits)} +/- {_fmt(sd, digits)}"


def write_scalar() -> None:
    rows = _read_csv(ROOT / "scalar_audit_summary.csv")
    per_seed = _read_csv(ROOT / "scalar_audit_per_seed.csv")
    lines = [
        "# Final Scalar Baselines",
        "",
        "All entries are mean +/- sample standard deviation over seeds 0,1,2.",
        "FS-Conv reports both PMF-mode aggregate accuracy and rounded expected-aggregate accuracy.",
        "Gaussian-AMLE and MSE have no discrete PMF mode, so mode accuracy is the rounded mean/mode surrogate.",
        "Continuous Gaussian NLL is not compared directly with exact discrete FS-Conv NLL.",
        "",
        "## Digit Sum, MNIST, n=10, train bags=1000",
        "",
        "| Method | Expected aggregate MAE | PMF/aggregate-mode acc. | Rounded expected acc. | Instance acc. | Exact discrete NLL | Gaussian NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r.get("task") == "digit_sum" and r.get("bag_size_mean") == "10" and r.get("train_bags") == "1000":
            lines.append(
                "| {method} | {mae} | {mode} | {rounded} | {inst} | {nll} | {gnll} |".format(
                    method=r["method"],
                    mae=_pm(r, "expected_aggregate_mae"),
                    mode=_pm(r, "mode_aggregate_acc"),
                    rounded=_pm(r, "rounded_expected_acc"),
                    inst=_pm(r, "instance_acc"),
                    nll=_pm(r, "aggregate_nll"),
                    gnll=_pm(r, "gaussian_nll"),
                )
            )
    lines += [
        "",
        "## Signed MNIST, n=10, train bags=1000",
        "",
        "| Sign protocol | Method | Expected aggregate MAE | Aggregate-mode acc. | Rounded expected acc. | Instance acc. | Instance AUC | Exact discrete NLL | Gaussian NLL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r.get("task") == "signed" and r.get("bag_size_mean") == "10" and r.get("train_bags") == "1000":
            sign = "cancellation" if r.get("cancellation_heavy") == "True" else "random"
            lines.append(
                "| {sign} | {method} | {mae} | {mode} | {rounded} | {inst} | {auc} | {nll} | {gnll} |".format(
                    sign=sign,
                    method=r["method"],
                    mae=_pm(r, "expected_aggregate_mae"),
                    mode=_pm(r, "mode_aggregate_acc"),
                    rounded=_pm(r, "rounded_expected_acc"),
                    inst=_pm(r, "instance_acc"),
                    auc=_pm(r, "instance_auc"),
                    nll=_pm(r, "aggregate_nll"),
                    gnll=_pm(r, "gaussian_nll"),
                )
            )
    lines += [
        "",
        "## Raw Sources",
        "",
    ]
    for r in per_seed:
        lines.append(f"- {r.get('task')} {r.get('method')} seed {r.get('seed')}: `{r.get('source')}`")
    (ROOT / "final_scalar_baselines.md").write_text("\n".join(lines) + "\n")


def _collect_fixed_cifar() -> list[dict[str, Any]]:
    rows = []
    for p in sorted((ROOT / "fixed_bag_cifar10" / "summaries").glob("*.json")):
        js = json.loads(p.read_text())
        cfg = js["config"]
        rows.append(
            {
                "method": cfg["method"],
                "seed": cfg["seed"],
                "bag_size": cfg["bag_size"],
                "train_bags": cfg["train_bags"],
                "val_bags": cfg["val_bags"],
                "test_bags": cfg["test_bags"],
                "instance_acc": js["test"]["instance_acc"],
                "macro_f1": js["test"]["macro_f1"],
                "hist_count_mae": js["test"]["hist_count_mae"],
                "composite_count_nll": js["test"]["composite_count_nll"],
                "selection_metric": js["selection_metric"],
                "train_manifest_hash": cfg["train_manifest_hash"],
                "test_manifest_hash": cfg["test_manifest_hash"],
                "run_dir": js["run_dir"],
                "train_bag_stats": js["train_bag_stats"],
            }
        )
    return rows


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def write_fixed_cifar() -> None:
    rows = _collect_fixed_cifar()
    lines = [
        "# Final Fixed-Bag CIFAR-10",
        "",
        "Protocol: frozen ImageNet-pretrained ResNet-18 features; n=64; 250 fixed training bags, 250 fixed validation bags, 1000 fixed test bags; Dirichlet alpha=0.3; seeds 0,1,2.",
        "Checkpoint selection uses aggregate validation count MAE, not hidden instance labels.",
        "",
    ]
    if not rows:
        lines += [
            "No completed fixed-bag CIFAR-10 summaries are available yet.",
            "The local CIFAR-10 archive was incomplete at audit time, so dataset acquisition/integrity verification is required before this table can be filled.",
        ]
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["method"], []).append(r)
        lines += [
            "| Method | Seeds | Instance acc. | Macro-F1 | Count MAE | Composite count NLL |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for method, vals in sorted(groups.items()):
            def cell(metric: str) -> str:
                mean, sd = _mean_sd([float(v[metric]) for v in vals])
                return f"{mean:.3f} +/- {sd:.3f}"
            lines.append(
                f"| {method} | {len(vals)} | {cell('instance_acc')} | {cell('macro_f1')} | {cell('hist_count_mae')} | {cell('composite_count_nll')} |"
            )
        lines += ["", "## Bag Manifests", ""]
        for r in rows:
            stats = r["train_bag_stats"]
            lines.append(
                f"- {r['method']} seed {r['seed']}: train hash `{r['train_manifest_hash']}`, "
                f"test hash `{r['test_manifest_hash']}`, unique train images {stats['num_unique_underlying_images']}, "
                f"entropy mean {stats['bag_composition_entropy_mean']:.3f}; source `{r['run_dir']}`"
            )
    (ROOT / "final_fixed_bag_cifar.md").write_text("\n".join(lines) + "\n")


def write_all_runs() -> None:
    rows: list[dict[str, Any]] = []
    for r in _read_csv(ROOT / "scalar_audit_per_seed.csv"):
        rows.append({"family": "scalar", **r})
    for r in _read_csv(ROOT / "gaussian_tuning" / "digit_sum_grid_summary.csv"):
        rows.append({"family": "gaussian_tuning", **r})
    for r in _collect_fixed_cifar():
        rows.append({"family": "fixed_cifar10", **{k: v for k, v in r.items() if k != "train_bag_stats"}})
    verdict = ROOT / "pvc_verdict.json"
    if verdict.exists():
        js = json.loads(verdict.read_text())
        rows.append(
            {
                "family": "equivalence",
                "classification": js["classification"],
                "recommended_label": js["recommended_label"],
                **{f"equiv_{k}": v for k, v in js["numerical_equivalence"].items()},
            }
        )
    keys = sorted({k for r in rows for k in r})
    with (ROOT / "all_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_audit() -> None:
    scalar_rows = _read_csv(ROOT / "scalar_audit_summary.csv")
    fixed_rows = _collect_fixed_cifar()
    verdict_path = ROOT / "pvc_verdict.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
    tuning_rows = _read_csv(ROOT / "gaussian_tuning" / "digit_sum_grid_summary.csv")
    best_tuning = tuning_rows[0] if tuning_rows else {}
    lines = [
        "# Final Experiment Audit",
        "",
        "## PVC Verdict",
        "",
        f"- Classification: {verdict.get('classification', 'pending')}",
        f"- Recommended label: {verdict.get('recommended_label', 'pending')}",
        "- Existing local `PVC` rows must not be called full official LLP-PVC unless a separate faithful official run is completed.",
        "- Equivalence result is numerical verification of the shared count-likelihood component, not a competitive baseline.",
        "",
        "## Scalar Metrics",
        "",
        f"- Scalar summary rows available: {len(scalar_rows)}",
        "- FS-Conv digit-sum aggregate accuracy is split into PMF-mode accuracy and rounded expected-sum accuracy.",
        "- Gaussian-AMLE NLL is continuous-density NLL and is not directly compared with exact discrete FS-Conv NLL.",
        "",
        "## Gaussian-AMLE Tuning",
        "",
    ]
    if best_tuning:
        lines.append(
            f"- Selected by dev seed aggregate validation loss: lr={best_tuning['lr']}, eps={best_tuning['eps']}, "
            f"validation loss={float(best_tuning['best_val_loss']):.6f}."
        )
    else:
        lines.append("- Tuning grid summary is not available yet.")
    lines += [
        "",
        "## Fixed-Bag CIFAR-10",
        "",
    ]
    if fixed_rows:
        lines.append(f"- Completed fixed-bag CIFAR-10 runs: {len(fixed_rows)}")
    else:
        lines.append("- Fixed-bag CIFAR-10 results are not complete yet; do not make performance claims from this table.")
    lines += [
        "",
        "## Artifact Checks",
        "",
        "- Manuscript files were intentionally not edited by these scripts.",
        "- Missing values are left absent/pending rather than fabricated.",
    ]
    Path("docs/rebuttal").mkdir(parents=True, exist_ok=True)
    Path("docs/rebuttal/FINAL_EXPERIMENT_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    write_scalar()
    write_fixed_cifar()
    write_all_runs()
    write_audit()
    print("wrote final rebuttal outputs")


if __name__ == "__main__":
    main()
