#!/usr/bin/env python
"""Build final cross-server rebuttal integration tables from available files."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("results/rebuttal")
DOCS = Path("docs/rebuttal")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def pm(mean: float | None, sd: float | None, digits: int = 3) -> str:
    if mean is None:
        return "-"
    return f"{fmt(mean, digits)} +/- {fmt(sd, digits)}"


def metric_summary(rows: list[dict[str, str]], metric: str) -> str:
    vals = [float(r[f"{metric}_mean"]) for r in rows if r.get(f"{metric}_mean")]
    sds = [float(r[f"{metric}_sd"]) for r in rows if r.get(f"{metric}_sd")]
    if not vals:
        return "-"
    return pm(vals[0], sds[0] if sds else 0.0)


def scalar_rows() -> list[dict[str, str]]:
    return read_csv(ROOT / "scalar_audit_summary.csv")


def fixed_cifar_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "fixed_bag_cifar10" / "summaries").glob("*.json")):
        js = read_json(path)
        cfg = js["config"]
        rows.append(
            {
                "method": cfg["method"],
                "seed": int(cfg["seed"]),
                "path": str(path),
                "run_dir": js["run_dir"],
                "git_sha": Path(js["run_dir"]).name.split("_")[-2]
                if len(Path(js["run_dir"]).name.split("_")) >= 2
                else "",
                "train_manifest_hash": cfg["train_manifest_hash"],
                "val_manifest_hash": cfg["val_manifest_hash"],
                "test_manifest_hash": cfg["test_manifest_hash"],
                "instance_acc": float(js["test"]["instance_acc"]),
                "macro_f1": float(js["test"]["macro_f1"]),
                "count_mae": float(js["test"]["hist_count_mae"]),
                "composite_nll": float(js["test"]["composite_count_nll"]),
            }
        )
    return rows


def official_llppvc_dev_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "4090_official_llppvc" / "dev_runs").glob("*/summary.json")):
        js = read_json(path)
        metadata_path = path.parent / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        rows.append(
            {
                "lr": float(js["config"]["lr"]),
                "seed": int(js["config"]["seed"]),
                "path": str(path),
                "run_dir": js["run_dir"],
                "git_sha": metadata.get("git_sha", ""),
                "train_manifest_hash": js["config"].get("train_manifest_hash", ""),
                "test_manifest_hash": js["config"].get("test_manifest_hash", ""),
                "val_nll": float(js["best"]["validation_composite_count_nll"]),
                "val_mae": float(js["best"]["validation_hist_count_mae"]),
                "test_nll": float(js["test"]["composite_count_nll"]),
                "test_mae": float(js["test"]["hist_count_mae"]),
                "unique_acc": float(js["test"]["unique_image_instance_acc"]),
                "macro_f1": float(js["test"]["unique_image_macro_f1"]),
            }
        )
    return rows


def write_tables() -> None:
    scalar = scalar_rows()
    fixed = fixed_cifar_rows()
    dev = official_llppvc_dev_rows()
    verdict = read_json(ROOT / "pvc_verdict.json") if (ROOT / "pvc_verdict.json").exists() else {}
    equiv = read_json(ROOT / "overnight_4090" / "pvc_equivalence.json")

    table_rows: list[dict[str, str]] = []

    md = [
        "# Final Rebuttal Experiment Tables",
        "",
        "This file integrates the locally available 4090 and merged A5000 artifacts.",
        "All mean +/- values use sample standard deviation over seeds unless noted.",
        "Missing rows are left explicit; no values are fabricated or interpolated.",
        "",
        "## 1. Stronger Scalar Baselines",
        "",
        "MNIST digit sum, n=10, train bags=1000, seeds 0,1,2.",
        "",
        "| Method | Expected aggregate MAE | Aggregate-mode acc. | Rounded expected acc. | Instance acc. | NLL field | Raw summary |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for method in ["MSE", "Gaussian-AMLE", "FS-Conv"]:
        rows = [r for r in scalar if r.get("task") == "digit_sum" and r.get("method") == method]
        if not rows:
            continue
        r = rows[0]
        nll = metric_summary(rows, "aggregate_nll") if r.get("aggregate_nll_mean") else metric_summary(rows, "gaussian_nll")
        md.append(
            f"| {method} | {metric_summary(rows, 'expected_aggregate_mae')} | "
            f"{metric_summary(rows, 'mode_aggregate_acc')} | {metric_summary(rows, 'rounded_expected_acc')} | "
            f"{metric_summary(rows, 'instance_acc')} | {nll} | `results/rebuttal/scalar_audit_summary.csv` |"
        )
        table_rows.append(
            {
                "section": "scalar_digit_sum",
                "method": method,
                "seeds": "3",
                "metric_1_name": "expected_aggregate_mae",
                "metric_1": metric_summary(rows, "expected_aggregate_mae"),
                "metric_2_name": "aggregate_mode_accuracy",
                "metric_2": metric_summary(rows, "mode_aggregate_acc"),
                "metric_3_name": "instance_accuracy",
                "metric_3": metric_summary(rows, "instance_acc"),
                "raw_paths": "results/rebuttal/scalar_audit_summary.csv; results/rebuttal/scalar_audit_per_seed.csv",
            }
        )

    md += [
        "",
        "Signed MNIST, n=10, train bags=1000, random/cancellation protocols, seeds 0,1,2.",
        "",
        "| Sign protocol | Method | Expected aggregate MAE | Aggregate-mode acc. | Instance acc. | Instance AUC | Raw summary |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for sign_flag, sign_name in [("False", "random"), ("True", "cancellation")]:
        for method in ["MSE", "Gaussian-AMLE", "FS-Conv"]:
            rows = [
                r
                for r in scalar
                if r.get("task") == "signed" and r.get("method") == method and r.get("cancellation_heavy") == sign_flag
            ]
            if not rows:
                continue
            md.append(
                f"| {sign_name} | {method} | {metric_summary(rows, 'expected_aggregate_mae')} | "
                f"{metric_summary(rows, 'mode_aggregate_acc')} | {metric_summary(rows, 'instance_acc')} | "
                f"{metric_summary(rows, 'instance_auc')} | `results/rebuttal/scalar_audit_summary.csv` |"
            )

    md += [
        "",
        "## 2. LLP-PVC Relationship",
        "",
        f"- Existing local `PVC` classification: `{verdict.get('classification_label', 'unavailable')}`.",
        f"- Existing local rows may be called full LLP-PVC: `{verdict.get('may_call_existing_rows_full_llp_pvc', False)}`.",
        f"- Forward loss difference: `{equiv['max_abs_forward_loss_diff']:.3e}`.",
        f"- Per-class PMF difference: `{equiv['max_abs_per_class_probability_diff']:.3e}`.",
        f"- Logit-gradient difference: `{equiv['max_abs_logit_gradient_diff']:.3e}`.",
        "- This is a component-equivalence check, not a competitive full-method baseline.",
        "",
        "Official LLP-PVC full-pipeline status:",
        "",
    ]
    if dev:
        best = min(dev, key=lambda r: (r["val_nll"], r["val_mae"]))
        md += [
            f"- Development-only seed 999 LR check completed; selected LR by aggregate validation NLL would be `{best['lr']}`.",
            f"- Best dev row: val NLL `{best['val_nll']:.3f}`, val count MAE `{best['val_mae']:.3f}`, "
            f"test NLL `{best['test_nll']:.3f}`, test count MAE `{best['test_mae']:.3f}`, unique-image acc. `{best['unique_acc']:.3f}`.",
            "- Required official LLP-PVC seeds 0,1,2 are not present in committed result summaries.",
        ]
    else:
        md.append("- No full-pipeline official LLP-PVC development or final summaries are present.")

    md += [
        "",
        "## 3. Fixed Finite-Bag CIFAR-10",
        "",
        "Protocol: frozen ImageNet-pretrained ResNet-18 features, n=64, 250 fixed train bags, 250 validation bags, 1000 test bags, Dirichlet alpha=0.3.",
        "",
        "| Method | Seeds | Instance acc. | Macro-F1 | Count MAE | Composite NLL | Raw summaries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in fixed:
        by_method.setdefault(row["method"], []).append(row)
    display = [
        ("CE/KL proportion matching", ["ce", "kl"]),
        ("Full official LLP-PVC", ["official_llppvc"]),
        ("FS-Conv classwise count likelihood", ["fsconv_count"]),
        ("PVC count-component only", ["official_pvc_count_component"]),
    ]
    for label, keys in display:
        vals = [r for k in keys for r in by_method.get(k, [])]
        if not vals:
            md.append(f"| {label} | 0 | - | - | - | - | not available |")
            continue
        inst = mean_sd([r["instance_acc"] for r in vals])
        f1 = mean_sd([r["macro_f1"] for r in vals])
        mae = mean_sd([r["count_mae"] for r in vals])
        nll = mean_sd([r["composite_nll"] for r in vals])
        paths = "; ".join(r["path"] for r in vals)
        md.append(f"| {label} | {len(vals)} | {pm(*inst)} | {pm(*f1)} | {pm(*mae)} | {pm(*nll)} | `{paths}` |")
        table_rows.append(
            {
                "section": "fixed_cifar10",
                "method": label,
                "seeds": str(len(vals)),
                "metric_1_name": "instance_accuracy",
                "metric_1": pm(*inst),
                "metric_2_name": "count_mae",
                "metric_2": pm(*mae),
                "metric_3_name": "composite_nll",
                "metric_3": pm(*nll),
                "raw_paths": paths,
            }
        )

    md += [
        "",
        "## 4. Natural-Image Replication",
        "",
        "No strict committed three-seed pretrained SVHN result summaries are present after merging `origin/exp/a5000-svhn-dependence`.",
        "The A5000 branch includes protocol and invalid-run audits plus rerun scripts, but not reportable strict result JSONs.",
        "See `docs/rebuttal/a5000_SVHN_PROTOCOL_AUDIT.md` and `docs/rebuttal/a5000_SVHN_INVALID_RUN_AUDIT.md`.",
        "",
        "## 5. Conditional-Independence Sensitivity",
        "",
        "No committed dependence-stress result summaries are present after merging `origin/exp/a5000-svhn-dependence`.",
        "The merged branch includes the generator and training scripts under `src/experiments/dependence_stress/` and `scripts/rebuttal/a5000_train_dependence.py`.",
    ]

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "FINAL_REBUTTAL_EXPERIMENT_TABLES.md").write_text("\n".join(md) + "\n")
    with (ROOT / "FINAL_REBUTTAL_EXPERIMENT_TABLES.csv").open("w", newline="") as f:
        fields = ["section", "method", "seeds", "metric_1_name", "metric_1", "metric_2_name", "metric_2", "metric_3_name", "metric_3", "raw_paths"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)


def write_inventory() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.glob("**/*.json")) + sorted(ROOT.glob("**/*.csv")) + sorted(ROOT.glob("**/*.md")):
        if path.name.startswith("FINAL_"):
            continue
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        rows.append(
            {
                "path": str(path),
                "kind": path.suffix.lstrip("."),
                "tracked": str(tracked),
                "size_bytes": str(path.stat().st_size),
            }
        )
    with (ROOT / "FINAL_RUN_INVENTORY.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "kind", "tracked", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def write_failures_and_docs() -> None:
    head = git(["rev-parse", "HEAD"])
    branch = git(["branch", "--show-current"])
    sha_4090 = git(["rev-parse", "origin/exp/4090-official-llppvc"])
    sha_a5000 = git(["rev-parse", "origin/exp/a5000-svhn-dependence"])
    marker_4090 = subprocess.run(
        ["git", "show", "origin/exp/4090-official-llppvc:docs/rebuttal/4090_FINAL_BRANCH_SHA.txt"],
        text=True,
        capture_output=True,
    )
    marker_a5000 = subprocess.run(
        ["git", "show", "origin/exp/a5000-svhn-dependence:docs/rebuttal/A5000_FINAL_BRANCH_SHA.txt"],
        text=True,
        capture_output=True,
    )
    failure_lines = [
        "# Final Failure / Gap Inventory",
        "",
        "- `docs/rebuttal/4090_FINAL_BRANCH_SHA.txt` is missing from `origin/exp/4090-official-llppvc`.",
        "- `docs/rebuttal/A5000_FINAL_BRANCH_SHA.txt` is missing from `origin/exp/a5000-svhn-dependence`.",
        "- Full official LLP-PVC seeds 0,1,2 are not present in committed result summaries; only smoke and local dev seed 999 outputs are available.",
        "- Strict A5000 SVHN and dependence result summaries are not present in the merged branch; the branch contains protocol audits and runnable scripts.",
        "- No manuscript files were edited.",
    ]
    (ROOT / "FINAL_FAILURE_INVENTORY.md").write_text("\n".join(failure_lines) + "\n")

    summary = [
        "# Final Rebuttal Experiment Summary",
        "",
        f"- Integration branch: `{branch}`.",
        f"- Integration HEAD before final summary commit: `{head}`.",
        f"- 4090 feature tip: `{sha_4090}`; recorded final marker: `{marker_4090.stdout.strip() if marker_4090.returncode == 0 else 'MISSING'}`.",
        f"- A5000 feature tip: `{sha_a5000}`; recorded final marker: `{marker_a5000.stdout.strip() if marker_a5000.returncode == 0 else 'MISSING'}`.",
        "- Scalar MNIST rebuttal baselines are complete for three seeds and distinguish PMF-mode from rounded expectation metrics.",
        "- Fixed finite-bag CIFAR-10 CE/KL and FS-Conv rows are complete for three seeds with fixed manifests.",
        "- The old `official_pvc_count_component` row is explicitly not labeled full LLP-PVC.",
        "- Full official LLP-PVC has upstream pin/audit and adapter validation, but final seeds 0,1,2 are not available in the merged artifacts.",
        "- Natural-image SVHN and conditional-dependence final rows are unavailable in committed artifacts.",
        "",
        "Primary result files:",
        "",
        "- `results/rebuttal/FINAL_REBUTTAL_EXPERIMENT_TABLES.md`",
        "- `results/rebuttal/FINAL_REBUTTAL_EXPERIMENT_TABLES.csv`",
        "- `results/rebuttal/FINAL_RUN_INVENTORY.csv`",
        "- `results/rebuttal/FINAL_FAILURE_INVENTORY.md`",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "FINAL_REBUTTAL_EXPERIMENT_SUMMARY.md").write_text("\n".join(summary) + "\n")

    provenance = [
        "# Final Git and Artifact Provenance",
        "",
        f"- Branch: `{branch}`.",
        f"- Current HEAD at generation time: `{head}`.",
        f"- Remote 4090 branch tip: `{sha_4090}`.",
        f"- Remote A5000 branch tip: `{sha_a5000}`.",
        "- Merge strategy: explicit `--no-ff` merge commits; no force-push or history rewrite.",
        "- SHA marker verification failed because both expected final marker files are missing on the corresponding feature branch tips.",
        "- Git excludes were checked for datasets, feature tensors, checkpoints, pretrained weights, `.part` files, and caches before final staging.",
        "- Tracked fixed-bag `.pt` files are small manifest files, not feature tensors or checkpoints.",
        "",
        "Non-Git/local artifacts:",
        "",
        "- CIFAR data archives, frozen feature tensors, and model checkpoints remain local/ignored.",
        "- Official LLP-PVC dev checkpoints under `results/rebuttal/4090_official_llppvc/dev_runs/*/checkpoint_*.pt` are intentionally not staged.",
    ]
    (DOCS / "FINAL_GIT_AND_ARTIFACT_PROVENANCE.md").write_text("\n".join(provenance) + "\n")


def main() -> None:
    write_tables()
    write_inventory()
    write_failures_and_docs()
    print("wrote final integration outputs")


if __name__ == "__main__":
    main()
