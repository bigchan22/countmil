#!/usr/bin/env python
"""Recompute and summarize strict Criteo final comparisons from raw predictions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from countmil.criteo.constants import BAG_SIZE, PROTOCOL_TAG
from countmil.criteo.losses import aggregate_metrics, instance_metrics


STRICT_ROOT = Path("results/rebuttal/4090_criteo_strictv1")
EXTRA_ROOT = Path("results/rebuttal/4090_criteo_extra_baselines")
ARTIFACT_ROOT = Path("/home/chanhomin/datasets/criteo_x1/strictv1")
METHODS = ["group_prior", "dllp_bce", "dllp_mse", "easyllp", "genbags", "ot_llp", "fsconv", "supervised_oracle"]
LABELS = {
    "group_prior": "Group prior",
    "dllp_bce": "DLLP-BCE",
    "dllp_mse": "DLLP-MSE",
    "easyllp": "EasyLLP",
    "genbags": "GenBags",
    "ot_llp": "OT-LLP",
    "fsconv": "FS-Conv",
    "supervised_oracle": "Supervised oracle",
}
OLD_METHODS = {"dllp_bce", "dllp_mse", "easyllp", "fsconv"}


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, sd


def _pm(vals: list[float]) -> str:
    mean, sd = _mean_sd(vals)
    return "-" if not math.isfinite(mean) else f"{mean:.4f} +/- {sd:.4f}"


def _summary_path(method: str, seed: int) -> Path:
    root = STRICT_ROOT if method in OLD_METHODS else EXTRA_ROOT
    return root / f"{method}_s{seed}.json"


def _load_summary(method: str, seed: int) -> dict[str, Any]:
    path = _summary_path(method, seed)
    js = json.loads(path.read_text())
    js["_summary_path"] = str(path)
    return js


def _test_counts(seed: int) -> torch.Tensor:
    shard = torch.load(ARTIFACT_ROOT / f"seed{seed}" / f"test_seed{seed}_shard.pt", map_location="cpu")
    return shard["counts"].long()


def _recompute(summary: dict[str, Any]) -> dict[str, Any]:
    seed = int(summary["config"]["seed"])
    run_dir = Path(summary["run_dir"])
    raw = run_dir / "raw_test_probs_labels.pt"
    if not raw.exists():
        raise FileNotFoundError(f"missing raw prediction tensor for recomputation: {raw}")
    payload = torch.load(raw, map_location="cpu")
    probs = payload["probs"].float()
    labels = payload["labels"].long()
    counts = _test_counts(seed)
    bag_probs = probs.reshape(counts.numel(), BAG_SIZE)
    agg = aggregate_metrics(bag_probs, counts)
    instance = instance_metrics(probs, labels)
    return {
        "summary": summary,
        "seed": seed,
        "method": summary["config"]["method"],
        "instance": instance,
        "aggregate": {k: float(v.mean().item()) for k, v in agg.items()},
        "per_bag": {k: v.detach().cpu().numpy().astype(np.float64) for k, v in agg.items()},
        "raw_prediction_path": str(raw),
    }


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _bootstrap(rows: list[dict[str, Any]], *, reference: str = "fsconv", n_boot: int = 10000, seed: int = 202605) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_method_seed = {(r["method"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if r["method"] == reference})
    out: dict[str, Any] = {"reference": reference, "n_boot": n_boot, "resampling_unit": "whole test bags within seed"}
    for method in METHODS:
        if method == reference:
            continue
        if any((method, s) not in by_method_seed or (reference, s) not in by_method_seed for s in seeds):
            continue
        method_out = {}
        for metric in ["expected_count_mae", "poisson_binomial_nll"]:
            diffs = {
                s: by_method_seed[(method, s)]["per_bag"][metric] - by_method_seed[(reference, s)]["per_bag"][metric]
                for s in seeds
            }
            observed = float(np.mean([d.mean() for d in diffs.values()]))
            boots = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                vals = []
                for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                    d = diffs[int(sampled_seed)]
                    idx = rng.integers(0, d.shape[0], size=d.shape[0])
                    vals.append(float(d[idx].mean()))
                boots[b] = float(np.mean(vals))
            method_out[metric] = {
                "mean_difference_method_minus_reference": observed,
                "ci95_percentile": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            }
        out[method] = method_out
    return out


def main() -> None:
    EXTRA_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = ["# Criteo Extra Baselines Failures", ""]
    for method in METHODS:
        for seed in [0, 1, 2]:
            path = _summary_path(method, seed)
            if not path.exists():
                failures.append(f"- missing summary: `{path}`")
                continue
            try:
                rows.append(_recompute(_load_summary(method, seed)))
            except Exception as exc:
                failures.append(f"- failed recomputation for {method} seed {seed}: `{exc}`")

    inv_rows = [["method", "seed", "status", "run_id", "summary_path", "raw_prediction_path", "selected_epoch", "val_expected_count_mae", "roc_auc", "count_mae"]]
    for r in rows:
        s = r["summary"]
        inv_rows.append(
            [
                r["method"],
                r["seed"],
                s["status"],
                s["run_id"],
                s["_summary_path"],
                r["raw_prediction_path"],
                s["best"]["epoch"],
                s["best"]["val_expected_count_mae"],
                r["instance"]["roc_auc"],
                r["aggregate"]["expected_count_mae"],
            ]
        )
    _write_csv(EXTRA_ROOT / "run_inventory.csv", inv_rows)

    inst_lines = [
        "# Criteo Final Instance Metrics",
        "",
        "LLP-Bench-style Criteo C4+C11 feature-bag experiment. Metrics are recomputed from final test predictions. Mean +/- sample standard deviation over seeds 0,1,2.",
        "",
        "| Method | Seeds | ROC-AUC | PR-AUC | Log loss | Brier | ECE-15 | Acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    agg_lines = [
        "# Criteo Final Aggregate Metrics",
        "",
        "Aggregate metrics are common post-hoc evaluations using exact Poisson-binomial PMFs from each method's instance probabilities.",
        "",
        "| Method | Seeds | Count MAE | Rounded count acc. | PMF-mode acc. | Exact count NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    inst_csv = [["method", "seeds", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "pr_auc_sd", "log_loss_mean", "log_loss_sd", "brier_mean", "brier_sd", "ece_15_mean", "ece_15_sd", "accuracy_0p5_mean", "accuracy_0p5_sd"]]
    agg_csv = [["method", "seeds", "count_mae_mean", "count_mae_sd", "rounded_acc_mean", "rounded_acc_sd", "pmf_mode_acc_mean", "pmf_mode_acc_sd", "nll_mean", "nll_sd"]]
    for method in METHODS:
        vals = [r for r in rows if r["method"] == method]
        inst = [r["instance"] for r in vals]
        agg = [r["aggregate"] for r in vals]
        inst_lines.append(
            f"| {LABELS[method]} | {len(vals)} | {_pm([x['roc_auc'] for x in inst])} | {_pm([x['pr_auc'] for x in inst])} | "
            f"{_pm([x['log_loss'] for x in inst])} | {_pm([x['brier'] for x in inst])} | {_pm([x['ece_15'] for x in inst])} | {_pm([x['accuracy_0p5'] for x in inst])} |"
        )
        agg_lines.append(
            f"| {LABELS[method]} | {len(vals)} | {_pm([x['expected_count_mae'] for x in agg])} | {_pm([x['rounded_expected_count_acc'] for x in agg])} | "
            f"{_pm([x['pmf_mode_count_acc'] for x in agg])} | {_pm([x['poisson_binomial_nll'] for x in agg])} |"
        )
        if len(vals) != 3:
            failures.append(f"- {method}: expected 3 completed/recomputed seeds, found {len(vals)}.")

        def ms(metric_rows: list[dict[str, float]], key: str) -> tuple[float, float]:
            return _mean_sd([x[key] for x in metric_rows])

        roc = ms(inst, "roc_auc"); pr = ms(inst, "pr_auc"); ll = ms(inst, "log_loss"); br = ms(inst, "brier"); ece = ms(inst, "ece_15"); acc = ms(inst, "accuracy_0p5")
        inst_csv.append([method, len(vals), roc[0], roc[1], pr[0], pr[1], ll[0], ll[1], br[0], br[1], ece[0], ece[1], acc[0], acc[1]])
        mae = ms(agg, "expected_count_mae"); racc = ms(agg, "rounded_expected_count_acc"); macc = ms(agg, "pmf_mode_count_acc"); nll = ms(agg, "poisson_binomial_nll")
        agg_csv.append([method, len(vals), mae[0], mae[1], racc[0], racc[1], macc[0], macc[1], nll[0], nll[1]])

    (EXTRA_ROOT / "instance_table.md").write_text("\n".join(inst_lines) + "\n")
    (EXTRA_ROOT / "aggregate_table.md").write_text("\n".join(agg_lines) + "\n")
    _write_csv(EXTRA_ROOT / "instance_table.csv", inst_csv)
    _write_csv(EXTRA_ROOT / "aggregate_table.csv", agg_csv)
    (EXTRA_ROOT / "failures.md").write_text("\n".join(failures) + "\n")

    boot = _bootstrap(rows)
    (EXTRA_ROOT / "paired_cluster_bootstrap.json").write_text(json.dumps(boot, indent=2, sort_keys=True))
    boot_lines = [
        "# Criteo Paired Cluster Bootstrap",
        "",
        "Differences are method minus FS-Conv. Negative values favor the method for error/NLL metrics. Whole test bags are resampled within seed; 10,000 bootstrap replicates.",
        "",
        "| Method | Metric | Mean diff. | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for method in METHODS:
        if method == "fsconv" or method not in boot:
            continue
        for metric, val in boot[method].items():
            if not isinstance(val, dict):
                continue
            ci = val["ci95_percentile"]
            boot_lines.append(f"| {LABELS[method]} | {metric} | {val['mean_difference_method_minus_reference']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    (EXTRA_ROOT / "paired_cluster_bootstrap.md").write_text("\n".join(boot_lines) + "\n")

    audit = [
        "# Criteo Extra Baseline Audit",
        "",
        f"- Protocol tag: `{PROTOCOL_TAG}`.",
        "- Existing DLLP-BCE, DLLP-MSE, EasyLLP, and FS-Conv predictions are reused from strict-v1 without modifying their result directories.",
        "- New GenBags, OT-LLP, group-prior, and supervised-oracle outputs are written only under `results/rebuttal/4090_criteo_extra_baselines/`.",
        "- GenBags follows the Google Research LLP-Bench Gaussian combining-weight definition with block size 4 and 60 generated bags per block; therefore 8 original bags produce 120 generalized bags.",
        "- OT-LLP uses a nonregularized disjoint-bag hard assignment with exactly the observed count positives in each bag.",
        "- Group-prior estimates C4+C11 click rates from aggregate training counts only and backs off to the global aggregate training click rate for unseen groups.",
        "- The supervised oracle uses instance BCE on training labels and is labeled only as an upper reference.",
        "- Validation checkpointing uses aggregate expected-count MAE; validation loaders do not expose hidden instance labels.",
        "- Final instance metrics are computed only on the final test split after checkpoint selection.",
        "- Google Research LLP-Bench source commit recorded for method definitions: `ec7c3d346277b737bc2decffcd1b533d4b7ec105`.",
    ]
    Path("docs/rebuttal/CRITEO_EXTRA_BASELINE_AUDIT.md").write_text("\n".join(audit) + "\n")
    report = [
        "# Criteo Final Comparison",
        "",
        "This comparison uses the same strict C4+C11 bag-size-128 manifests, feature shards, preprocessing, architecture, batch size, training budget, aggregate-only validation protocol, and final-test evaluation code.",
        "",
        "## Instance Metrics",
        "",
        *inst_lines[4:],
        "",
        "## Aggregate Metrics",
        "",
        *agg_lines[4:],
        "",
        "## Paired Cluster Bootstrap",
        "",
        *boot_lines[4:],
    ]
    Path("docs/rebuttal/CRITEO_FINAL_COMPARISON.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
