#!/usr/bin/env python
"""Aggregate per-seed JSONs across A1 datasets → summary.md + final verdict.

Per spec §11. Usage:
  python scripts/analyze_a1.py                     # full verdict (all 4 datasets)
  python scripts/analyze_a1.py --partial           # subset of datasets present
  python scripts/analyze_a1.py --datasets mnist_sum mnist_signed
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.stats


HARD_CAP = {
    'mnist_sum':     50,
    'mnist_signed':  50,
    'svhn_sum':      1000,
    'ultramnist':    300,
}
ALL_DATASETS = ['mnist_sum', 'svhn_sum', 'ultramnist', 'mnist_signed']
MULTICLASS_DATASETS = ['mnist_sum', 'svhn_sum', 'ultramnist']
DISTRIBUTION_NATIVE = ['2b', '3a', '3b']


def _select_tail5(d: dict) -> dict:
    """Default — `mean(epoch_test_*[-5:])`, the value already stored in JSON."""
    return {'nll': d['best_test_nll'], 'acc': d['best_test_acc'],
            'mae': d['best_test_mae'], 'ece': d['best_test_ece']}


def _select_min_test_nll(d: dict) -> dict:
    """DIAGNOSTIC ONLY — test-set leakage. Picks the epoch with min test NLL."""
    nll_arr = d.get('epoch_test_nll')
    if not nll_arr:
        return _select_tail5(d)
    idx = int(np.argmin(nll_arr))
    return {'nll': float(nll_arr[idx]),
            'acc': float(d['epoch_test_acc'][idx]),
            'mae': float(d['epoch_test_mae'][idx]),
            'ece': float(d['epoch_test_ece'][idx])}


def _select_argmin_train_loss(d: dict) -> dict:
    """Methodologically clean: argmin over training loss → that epoch's test metrics.
    Empirically near-equivalent to tail5 since training loss is monotonically
    decreasing within a fixed budget."""
    train_arr = d.get('epoch_train_loss')
    if not train_arr:
        return _select_tail5(d)
    idx = int(np.argmin(train_arr))
    return {'nll': float(d['epoch_test_nll'][idx]),
            'acc': float(d['epoch_test_acc'][idx]),
            'mae': float(d['epoch_test_mae'][idx]),
            'ece': float(d['epoch_test_ece'][idx])}


SELECTORS = {
    'tail5':              _select_tail5,
    'min_test_nll':       _select_min_test_nll,
    'argmin_train_loss':  _select_argmin_train_loss,
}


@dataclass
class CellStat:
    mean: float
    std: float
    seeds: list[float] = field(default_factory=list)


def aggregate(results_root: Path, datasets: Iterable[str],
              selector: str = 'tail5') -> dict:
    """summary[dataset][method][metric] = CellStat for metric ∈ {nll, acc, mae, ece}.

    `selector` controls how each seed JSON's per-epoch arrays are reduced to a
    single scalar per metric. Default `tail5` reads the already-stored
    `best_test_*` (mean of last 5 epochs). See SELECTORS for alternatives.
    """
    select_fn = SELECTORS[selector]
    summary: dict = {}
    for ds in datasets:
        ds_dir = results_root / ds
        if not ds_dir.exists():
            continue
        per_method: dict = {}
        for setting_dir in sorted(ds_dir.glob('N*_*')):
            method = setting_dir.name.rsplit('_', 1)[1]
            seed_jsons = sorted(setting_dir.glob('seed*.json'))
            if not seed_jsons:
                continue
            metrics = {'nll': [], 'acc': [], 'mae': [], 'ece': []}
            for f in seed_jsons:
                d = json.loads(f.read_text())
                m = select_fn(d)
                for k in metrics:
                    metrics[k].append(m[k])
            cells = {}
            for k, vs in metrics.items():
                cells[k] = CellStat(
                    mean=float(np.mean(vs)),
                    std=float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                    seeds=vs,
                )
            per_method[method] = cells
        if per_method:
            summary[ds] = per_method
    return summary


def evaluate_h1(summary, multiclass_datasets) -> dict:
    """H1: PCA NLL < min distribution-native baseline NLL,
       paired t p<0.05, |Δ|≥0.02 nats, on ≥2 of 3 datasets."""
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        if ds not in summary or 'pca' not in summary[ds]:
            per_dataset[ds] = {'available': False}
            continue
        pca_seeds = np.array(summary[ds]['pca']['nll'].seeds)
        baselines = [m for m in DISTRIBUTION_NATIVE if m in summary[ds]]
        if not baselines:
            per_dataset[ds] = {'available': False}
            continue
        baseline_seeds = np.stack([np.array(summary[ds][m]['nll'].seeds) for m in baselines])
        best_per_seed = baseline_seeds.min(axis=0)
        deltas = best_per_seed - pca_seeds
        delta_mean = float(np.mean(deltas))
        if len(pca_seeds) > 1:
            _, p_value = scipy.stats.ttest_rel(best_per_seed, pca_seeds)
        else:
            p_value = 1.0
        passes = bool(delta_mean >= 0.02 and p_value < 0.05)
        per_dataset[ds] = {'available': True, 'delta': delta_mean,
                            'p': float(p_value), 'pass': passes,
                            'baselines_used': baselines}
        if passes:
            pass_count += 1
    available_count = sum(1 for v in per_dataset.values() if v.get('available'))
    # "≥2 of 3" rule on full data; loosened to "≥ min(2, available)" in partial.
    threshold = min(2, available_count) if available_count > 0 else 1
    return {
        'pass': pass_count >= threshold,
        'pass_count': pass_count,
        'available_count': available_count,
        'threshold': threshold,
        'per_dataset': per_dataset,
    }


def _h2_pca_top1_check(summary, ds: str) -> dict:
    """H2 sub-check: PCA acc strictly greater than max baseline acc, finite NLL.

    Relative threshold replaces the previous absolute `acc > 0.30` (MNIST-
    calibrated, see docs/notes/2026-05-05-a1-verdict-reframing.md). Random-
    chance differs across datasets (1/19 for MNIST/ultramnist, 1/91 for SVHN
    sum-range), so a single absolute floor is mis-calibrated. Top-1 among
    methods is the natural definition of "PCA learns this atom shape."
    """
    pca_acc = summary[ds]['pca']['acc'].mean
    pca_nll = summary[ds]['pca']['nll'].mean
    baselines = [m for m in summary[ds] if m != 'pca']
    if not baselines:
        return {'pass': False,
                'detail': f'PCA acc={pca_acc:.3f} but no baselines available'}
    max_baseline_acc = max(summary[ds][m]['acc'].mean for m in baselines)
    passes = bool(pca_acc > max_baseline_acc and np.isfinite(pca_nll))
    return {
        'pass': passes,
        'detail': f'PCA acc={pca_acc:.3f} vs max baseline={max_baseline_acc:.3f}, '
                   f'nll={pca_nll:.3f}',
    }


def evaluate_h2(summary, results_root: Path, partial: bool = False) -> dict:
    """H2: atom-shape generality must-pass.

    With partial=True, datasets not in summary are skipped (not recorded as
    failures) — used for Day 4 partial verdict per spec §11.7.
    """
    checks = {}
    # Binary B1 reuse — read mnist_mil/summary.json's NLL @ N=50 instance AUC > 0.9.
    # B1 summary schema: {summary: {<N>: {<method>: {mean, std, seeds}}}}.
    # NB: B1's 'nll' key is a method label (the NLL training method), and the
    # mean value is best_inst_auc, not a negative log-likelihood.
    b1_summary_path = results_root / 'mnist_mil' / 'summary.json'
    if b1_summary_path.exists():
        b1 = json.loads(b1_summary_path.read_text())
        try:
            b1_nll_n50_auc = b1['summary']['50']['nll']['mean']
            checks['binary_b1_reuse'] = {
                'pass': bool(b1_nll_n50_auc > 0.9),
                'detail': f'B1 NLL@N=50 inst_auc = {b1_nll_n50_auc:.4f}',
            }
        except (KeyError, TypeError) as e:
            checks['binary_b1_reuse'] = {
                'pass': False, 'detail': f'B1 summary unparseable: {e}'}
    else:
        checks['binary_b1_reuse'] = {'pass': False,
                                       'detail': f'missing {b1_summary_path}'}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        if ds in summary and 'pca' in summary[ds]:
            checks[f'multiclass_{ds}'] = _h2_pca_top1_check(summary, ds)
        elif not partial:
            checks[f'multiclass_{ds}'] = {'pass': False, 'detail': 'absent'}
        # else (partial=True and ds absent): skip silently
    if 'mnist_signed' in summary and 'pca' in summary['mnist_signed']:
        checks['signed_mnist_signed'] = _h2_pca_top1_check(summary, 'mnist_signed')
    elif not partial:
        checks['signed_mnist_signed'] = {'pass': False, 'detail': 'absent'}
    all_pass = all(c['pass'] for c in checks.values()) if checks else False
    return {'pass': all_pass, 'checks': checks}


def evaluate_h3(summary, multiclass_datasets) -> dict:
    """H3: PCA ECE ≤ 0.7 × min distribution-native baseline ECE on ≥2 of 3."""
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        if ds not in summary or 'pca' not in summary[ds]:
            per_dataset[ds] = {'available': False}
            continue
        pca_ece = summary[ds]['pca']['ece'].mean
        baselines = [m for m in DISTRIBUTION_NATIVE if m in summary[ds]]
        if not baselines:
            per_dataset[ds] = {'available': False}
            continue
        baseline_eces = [summary[ds][m]['ece'].mean for m in baselines]
        min_baseline_ece = float(min(baseline_eces))
        ratio = pca_ece / min_baseline_ece if min_baseline_ece > 0 else float('inf')
        passes = bool(ratio <= 0.7)
        per_dataset[ds] = {'available': True, 'pca_ece': float(pca_ece),
                            'min_baseline_ece': min_baseline_ece,
                            'ratio': float(ratio), 'pass': passes,
                            'baselines_used': baselines}
        if passes:
            pass_count += 1
    available_count = sum(1 for v in per_dataset.values() if v.get('available'))
    threshold = min(2, available_count) if available_count > 0 else 1
    return {'pass': pass_count >= threshold, 'pass_count': pass_count,
            'available_count': available_count, 'threshold': threshold,
            'per_dataset': per_dataset}


def evaluate_h4(summary, multiclass_datasets,
                acc_margin: float = 0.05, mae_margin: float = 0.05) -> dict:
    """H4: PCA wins point prediction (acc + MAE) by margin on ≥2/3 multi-class.

    Robust under any selector — captures the case where A1's atomic-PMF
    architecture produces sharper / more accurate predictions than baselines
    even when NLL/calibration trade off (H1/H3 fail).

    Pass criterion per dataset: PCA acc > max(baseline acc) + acc_margin
    AND PCA mae < min(baseline mae) - mae_margin.
    """
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        if ds not in summary or 'pca' not in summary[ds]:
            per_dataset[ds] = {'available': False}
            continue
        pca_acc = summary[ds]['pca']['acc'].mean
        pca_mae = summary[ds]['pca']['mae'].mean
        baselines = [m for m in summary[ds] if m != 'pca']
        if not baselines:
            per_dataset[ds] = {'available': False}
            continue
        max_baseline_acc = max(summary[ds][m]['acc'].mean for m in baselines)
        min_baseline_mae = min(summary[ds][m]['mae'].mean for m in baselines)
        acc_win = bool(pca_acc > max_baseline_acc + acc_margin)
        mae_win = bool(pca_mae < min_baseline_mae - mae_margin)
        passes = bool(acc_win and mae_win)
        per_dataset[ds] = {
            'available': True,
            'pca_acc': float(pca_acc),
            'max_baseline_acc': float(max_baseline_acc),
            'pca_mae': float(pca_mae),
            'min_baseline_mae': float(min_baseline_mae),
            'acc_win': acc_win, 'mae_win': mae_win, 'pass': passes,
        }
        if passes:
            pass_count += 1
    available = sum(1 for v in per_dataset.values() if v.get('available'))
    threshold = min(2, available) if available > 0 else 1
    return {'pass': pass_count >= threshold, 'pass_count': pass_count,
            'available_count': available, 'threshold': threshold,
            'per_dataset': per_dataset}


def detect_saturation(summary) -> dict:
    """For each multi-class dataset: True iff distribution-native NLL spread <0.05
       OR any method's top-1 acc > 0.97."""
    out = {}
    for ds in MULTICLASS_DATASETS:
        if ds not in summary:
            continue
        nll_means = [summary[ds][m]['nll'].mean for m in DISTRIBUTION_NATIVE
                      if m in summary[ds]]
        acc_means = [summary[ds][m]['acc'].mean for m in summary[ds]]
        if not nll_means or not acc_means:
            out[ds] = False
            continue
        out[ds] = bool((max(nll_means) - min(nll_means) < 0.05)
                        or (max(acc_means) > 0.97))
    return out


def hard_preset_already_run(results_root: Path, saturation: dict) -> dict:
    """Per-dataset bool: True iff at least one seed*.json under results/<ds>/
       has config.per_class_cap == HARD_CAP[ds]. Datasets with sat=False return
       False (rule does not apply).
    """
    out = {}
    for ds, sat in saturation.items():
        if not sat:
            out[ds] = False
            continue
        ds_dir = results_root / ds
        if not ds_dir.exists():
            out[ds] = False
            continue
        out[ds] = any(
            json.loads(p.read_text()).get('config', {}).get('per_class_cap') == HARD_CAP.get(ds)
            for p in ds_dir.rglob('seed*.json')
        )
    return out


def hard_preset_clis(saturation: dict, already: dict | None = None) -> list[str]:
    clis = []
    already = already or {}
    for ds, sat in saturation.items():
        if not sat:
            continue
        if already.get(ds, False):
            continue   # already run hard preset for this dataset
        cap = HARD_CAP[ds]
        if ds == 'mnist_sum':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_mnist_sum.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.1 "
                    f"--output-dir results/mnist_sum")
        elif ds == 'mnist_signed':
            for m in ['pca', '1a']:
                clis.append(
                    f"uv run python scripts/run_mnist_signed.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.1 "
                    f"--output-dir results/mnist_signed")
        elif ds == 'svhn_sum':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_svhn_sum.py --method {m} --epochs 80 "
                    f"--seeds 0,1,2,3,4 --per-class-cap {cap} --noise-sigma 0.05 "
                    f"--output-dir results/svhn_sum")
        elif ds == 'ultramnist':
            for m in ['pca', '1a', '2a', '2b', '3a', '3b']:
                clis.append(
                    f"uv run python scripts/run_ultramnist.py --method {m} --epochs 60 "
                    f"--seeds 0,1,2 --per-class-cap {cap} "
                    f"--output-dir results/ultramnist")
    return clis


def verdict(h1, h2, h3, h4, saturation, results_root) -> str:
    if not h2['pass']:
        return 'ALGEBRA_BROKEN'
    if h1['pass']:
        return 'A1_CONFIRMED'
    already = hard_preset_already_run(results_root, saturation)
    sat_unprocessed = any(
        saturation.get(ds, False) and not already.get(ds, False)
        for ds in saturation
    )
    if sat_unprocessed:
        return 'RUN_HARD_PRESET'
    if h3['pass']:
        return 'A1_CALIBRATION_FOCUS'
    if h4['pass']:
        return 'A1_POINT_PREDICTION_WIN'
    return 'DEMOTE_A1'


def render_dataset_summary(ds: str, methods: dict) -> str:
    lines = [f"# Results — {ds}", "", "## Metrics (mean ± std)", ""]
    lines.append("| Method | Top-1 acc | MAE | NLL | ECE |")
    lines.append("|--------|-----------|-----|-----|-----|")
    method_order = ['1a', '2a', '2b', '3a', '3b', 'pca']
    for m in method_order:
        if m not in methods:
            continue
        c = methods[m]
        lines.append(
            f"| {m} | {c['acc'].mean:.3f} ± {c['acc'].std:.3f} "
            f"| {c['mae'].mean:.3f} ± {c['mae'].std:.3f} "
            f"| {c['nll'].mean:.3f} ± {c['nll'].std:.3f} "
            f"| {c['ece'].mean:.3f} ± {c['ece'].std:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_overall(h1, h2, h3, h4, saturation, v: str,
                   selector: str = 'tail5') -> str:
    lines = ["# A1 Verification — Final Summary", "",
             f"## Verdict: **{v}**", "",
             f"_Selector: `{selector}`_", "", "## H1 — NLL primary", ""]
    lines.append(f"- pass={h1['pass']}; pass_count={h1['pass_count']} "
                 f"(of {h1['available_count']} available)")
    for ds, d in h1['per_dataset'].items():
        if not d.get('available'):
            lines.append(f"  - {ds}: not available")
        else:
            lines.append(f"  - {ds}: Δ={d['delta']:+.3f} nats, p={d['p']:.3f}, "
                         f"{'PASS' if d['pass'] else 'FAIL'}")
    lines.extend(["", "## H2 — atom-shape generality"])
    for k, c in h2['checks'].items():
        lines.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} ({c['detail']})")
    lines.extend(["", "## H3 — calibration"])
    lines.append(f"- pass={h3['pass']}; pass_count={h3['pass_count']}")
    for ds, d in h3['per_dataset'].items():
        if not d.get('available'):
            lines.append(f"  - {ds}: not available")
        else:
            lines.append(f"  - {ds}: ratio={d['ratio']:.2f}, "
                         f"{'PASS' if d['pass'] else 'FAIL'}")
    lines.extend(["", "## H4 — point-prediction supremacy (acc + MAE)"])
    lines.append(f"- pass={h4['pass']}; pass_count={h4['pass_count']} "
                 f"(of {h4['available_count']} available)")
    for ds, d in h4['per_dataset'].items():
        if not d.get('available'):
            lines.append(f"  - {ds}: not available")
        else:
            lines.append(
                f"  - {ds}: PCA acc={d['pca_acc']:.3f} vs max_baseline="
                f"{d['max_baseline_acc']:.3f}; PCA mae={d['pca_mae']:.3f} vs "
                f"min_baseline={d['min_baseline_mae']:.3f}; "
                f"{'PASS' if d['pass'] else 'FAIL'}"
            )
    lines.extend(["", "## Saturation detection"])
    for ds, sat in saturation.items():
        lines.append(f"- {ds}: {'detected' if sat else 'clear'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', type=Path, default=Path('results'))
    p.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    p.add_argument('--partial', action='store_true',
                   help='Allow subset of datasets; verdict scoped to available.')
    p.add_argument('--selector', choices=list(SELECTORS.keys()), default='tail5',
                   help='Per-seed metric selector. Default tail5 reads stored '
                        'best_test_* (mean of last 5 epochs). min_test_nll is '
                        'a TEST-LEAKAGE diagnostic — do not report as primary. '
                        'argmin_train_loss reads test metrics at the epoch '
                        'with min training loss (clean, but typically '
                        'near-equivalent to tail5 within a fixed epoch budget).')
    args = p.parse_args()

    summary = aggregate(args.results_root, args.datasets, selector=args.selector)
    if not summary:
        print("[error] no datasets aggregated; nothing to analyze.")
        raise SystemExit(1)

    multiclass = [ds for ds in MULTICLASS_DATASETS if ds in summary]
    h1 = evaluate_h1(summary, multiclass)
    h2 = evaluate_h2(summary, args.results_root, partial=args.partial)
    h3 = evaluate_h3(summary, multiclass)
    h4 = evaluate_h4(summary, multiclass)
    saturation = detect_saturation(summary)
    v = verdict(h1, h2, h3, h4, saturation, args.results_root)

    # Per-dataset summaries
    for ds, methods in summary.items():
        md = render_dataset_summary(ds, methods)
        out = args.results_root / ds / 'summary.md'
        out.write_text(md)
        out_json = args.results_root / ds / 'summary.json'
        out_json.write_text(json.dumps({
            m: {k: {'mean': c.mean, 'std': c.std, 'seeds': c.seeds}
                for k, c in cells.items()}
            for m, cells in methods.items()
        }, indent=2))

    # Overall summary
    overall = render_overall(h1, h2, h3, h4, saturation, v,
                              selector=args.selector)
    (args.results_root / 'summary_overall.md').write_text(overall)
    print(overall)

    if v == 'RUN_HARD_PRESET':
        already = hard_preset_already_run(args.results_root, saturation)
        print("\n--- Hard preset CLIs to execute next ---")
        for cli in hard_preset_clis(saturation, already):
            print(cli)


if __name__ == "__main__":
    main()
