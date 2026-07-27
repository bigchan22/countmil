#!/usr/bin/env python
"""Construct strict fixed-size Criteo_x1 feature-bag shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from countmil.criteo.constants import PROTOCOL_TAG, SEEDS, SPLIT_BAGS
from countmil.criteo.data import artifact_record, build_split_shard, locate_csvs
from countmil.training.run import git_commit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/chanhomin/datasets/criteo_x1")
    p.add_argument("--artifact-root", default="/home/chanhomin/datasets/criteo_x1/strictv1")
    p.add_argument("--out-doc", default="docs/rebuttal/criteo_BAG_CONSTRUCTION_AUDIT.md")
    p.add_argument("--artifact-index", default="artifacts/rebuttal/criteo_STRICTV1_EXTERNAL_ARTIFACTS.json")
    args = p.parse_args()
    csvs = locate_csvs(Path(args.data_root) / "extracted")
    root = Path(args.artifact_root)
    sha = git_commit()
    command = " ".join(sys.argv)
    rows = []
    artifact_rows = []
    for seed in SEEDS:
        for split in ["train", "valid", "test"]:
            info = build_split_shard(
                csv_path=csvs[split],
                out_dir=root / f"seed{seed}",
                split=split,
                seed=seed,
                requested_bags=SPLIT_BAGS[split],
                git_sha=sha,
            )
            rows.append(info)
            for key in ["shard_path", "parquet_path"]:
                artifact_rows.append(
                    artifact_record(
                        Path(info[key]),
                        producing_command=command,
                        run_id=f"{PROTOCOL_TAG}_{split}_seed{seed}",
                        git_sha=sha,
                        required_to_retrain=True,
                        required_to_reconstruct_metrics=(split == "test"),
                    )
                )
    Path(args.artifact_index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.artifact_index).write_text(json.dumps(artifact_rows, indent=2, sort_keys=True))
    lines = [
        "# Criteo Strict-v1 Bag Construction Audit",
        "",
        f"- Protocol tag: `{PROTOCOL_TAG}`",
        "- Grouping key: `C4`, `C11`",
        "- Bag size: `128`",
        "- Selection is label-blind and based on deterministic MD5 ordering.",
        "",
        "| Seed | Split | Candidate full bags | Selected bags | Retained instances | Click prevalence | Count mean | Count std | Zero-click % | Max count | Manifest hash | Input hash |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if r["selected_bags"] < r["requested_bags"]:
            print(f"WARNING: {r['split']} seed {r['seed']} shortfall {r['shortfall']}", file=sys.stderr)
        lines.append(
            f"| {r['seed']} | {r['split']} | {r['candidate_full_bags']} | {r['selected_bags']} | "
            f"{r['retained_instances']} | {r['click_prevalence']:.6f} | {r['click_count_mean']:.3f} | "
            f"{r['click_count_sample_std']:.3f} | {r['zero_click_bag_percentage']:.2f} | {r['max_click_count']} | "
            f"`{r['manifest_hash'][:12]}` | `{r['input_shard_hash'][:12]}` |"
        )
    Path(args.out_doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_doc).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

