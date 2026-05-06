#!/usr/bin/env python
"""Aggregate per-seed JSONs → markdown summary + H1/H2/H3 verdict.

Per spec §11. Usage:
  python scripts/analyze.py --results-dir results/mnist_mil/
  python scripts/analyze.py --results-dir results/llp_adult/
"""
from __future__ import annotations
import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats


@dataclass
class CellStat:
    mean: float
    std: float
    seeds: list[float] = field(default_factory=list)


def aggregate(results_dir: Path) -> dict[int, dict[str, CellStat]]:
    """Returns summary[bag_size][method] = CellStat over best_inst_auc."""
    summary: dict[int, dict[str, CellStat]] = {}
    for setting_dir in sorted(results_dir.glob('N*_*')):
        name = setting_dir.name           # e.g., N50_em
        N_str, method = name.split('_', 1)
        N = int(N_str[1:])
        seed_jsons = sorted(setting_dir.glob('seed*.json'))
        aucs = [json.loads(f.read_text())['best_inst_auc'] for f in seed_jsons]
        cell = CellStat(mean=float(np.mean(aucs)), std=float(np.std(aucs, ddof=1)),
                        seeds=aucs)
        summary.setdefault(N, {})[method] = cell
    return summary


def evaluate_h1(summary, target_N: int) -> dict:
    """H1: Δ at target_N ≥ 1.0 pp, paired t-test p < 0.05."""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    delta_pp = (em.mean - nl.mean) * 100
    diffs = np.array(em.seeds) - np.array(nl.seeds)
    _, p_value = scipy.stats.ttest_rel(em.seeds, nl.seeds)
    return {'pass': delta_pp >= 1.0 and p_value < 0.05,
            'delta_pp': delta_pp, 'p_value': float(p_value),
            'paired_diffs_pp': (diffs * 100).tolist()}


def evaluate_h2(summary, sizes: list[int]) -> dict:
    """H2: Δ monotone non-decreasing across bag sizes (0.2 pp slack)."""
    deltas_pp = [(summary[N]['em'].mean - summary[N]['nll'].mean) * 100 for N in sizes]
    monotone = all(deltas_pp[i] <= deltas_pp[i + 1] + 0.2 for i in range(len(deltas_pp) - 1))
    return {'pass': monotone, 'deltas_pp': deltas_pp}


def evaluate_h3(summary, target_N: int) -> dict:
    """H3: std(EM) ≤ 0.7 × std(NLL) at target_N."""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    ratio = em.std / nl.std if nl.std > 0 else float('inf')
    return {'pass': ratio <= 0.7, 'std_ratio': ratio,
            'em_std': em.std, 'nll_std': nl.std}


def verdict(h1, h2, h3) -> str:
    delta = h1['delta_pp']
    if delta < 0.5:
        return 'DEMOTE_B1'
    if delta < 1.0:
        return 'F2_ABLATION_NEEDED'
    if not h1['pass']:
        return 'F2_ABLATION_NEEDED'
    if h2['pass'] or h3['pass']:
        return 'PROCEED_TO_LLP'
    return 'PROCEED_BUT_WEAK'


def render_markdown(summary, h1, h2, h3, target_N, sizes, exp_name) -> str:
    lines = [f"# Results — {exp_name}", "", "## Instance-level AUC (mean ± std over 5 seeds)", "",
             "| N   | NLL              | EM               | Δ (pp)        |",
             "|-----|------------------|------------------|---------------|"]
    for N in sizes:
        nl = summary[N]['nll']; em = summary[N]['em']
        delta = (em.mean - nl.mean) * 100
        mark = '✓' if delta >= 1.0 else ('~' if delta >= 0.5 else '✗')
        lines.append(f"| {N:<3} | {nl.mean:.4f} ± {nl.std:.4f}  "
                     f"| {em.mean:.4f} ± {em.std:.4f}  | {delta:+.2f} {mark}      |")
    v = verdict(h1, h2, h3)
    lines += ["", "## Hypothesis verdicts", "",
              f"- H1 (Δ @ N={target_N} ≥ 1pp, p<0.05): "
              f"**{'PASS' if h1['pass'] else 'FAIL'}**  Δ = {h1['delta_pp']:+.2f} pp, "
              f"p = {h1['p_value']:.3f}",
              f"- H2 (monotone in N): "
              f"**{'PASS' if h2['pass'] else 'FAIL'}**  deltas_pp = "
              f"[{', '.join(f'{d:+.2f}' for d in h2['deltas_pp'])}]",
              f"- H3 (std reduction ≥ 30%): "
              f"**{'PASS' if h3['pass'] else 'FAIL'}**  ratio = {h3['std_ratio']:.2f}",
              "", f"## Verdict: {v}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, required=True)
    args = p.parse_args()

    summary = aggregate(args.results_dir)
    sizes = sorted(summary.keys())
    target_N = 50 if 50 in sizes else (128 if 128 in sizes else sizes[len(sizes) // 2])
    h1 = evaluate_h1(summary, target_N)
    h2 = evaluate_h2(summary, sizes)
    h3 = evaluate_h3(summary, target_N)
    md = render_markdown(summary, h1, h2, h3, target_N, sizes, args.results_dir.name)
    print(md)
    (args.results_dir / 'summary.md').write_text(md)
    (args.results_dir / 'summary.json').write_text(json.dumps({
        'h1': h1, 'h2': h2, 'h3': h3, 'verdict': verdict(h1, h2, h3),
        'summary': {N: {m: {'mean': c.mean, 'std': c.std, 'seeds': c.seeds}
                        for m, c in d.items()} for N, d in summary.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
