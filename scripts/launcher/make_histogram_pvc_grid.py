#!/usr/bin/env python
"""Generate OVR PVC/count-likelihood jobs for digit histograms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


COMMON = """dataset_root: data
dataset: {dataset}
bag_size_mean: {bag_mean}
bag_size_std: {bag_std}
train_bags: {train_bags}
seed: {seed}
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-root", default="configs/histogram_pvc")
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-bags", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--bag-settings", nargs="+", default=["10:2", "50:10"])
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / tag
    rows = ["name\tscript\tconfig\textra_args\n"]

    for setting in args.bag_settings:
        mean_s, std_s = setting.split(":", 1)
        bag_mean = int(mean_s)
        bag_std = int(std_s)
        for train_bags in args.train_bags:
            for seed in args.seeds:
                common = COMMON.format(
                    dataset=args.dataset,
                    bag_mean=bag_mean,
                    bag_std=bag_std,
                    train_bags=train_bags,
                    seed=seed,
                )
                name = f"{args.dataset.lower()}_histogram_pvc_n{bag_mean}_train{train_bags}_s{seed}"
                cfg = out_dir / "histogram_pvc" / f"{name}.yaml"
                _write(cfg, "experiment: mnist_histogram_pvc\nmethod: ovr_count_likelihood\n" + common)
                rows.append(f"{name}\tscripts/train_mnist_histogram_pvc.py\t{cfg}\t\n")

    manifest = out_dir / "manifest.tsv"
    _write(manifest, "".join(rows))
    print(f"wrote {len(rows) - 1} jobs to {manifest}")


if __name__ == "__main__":
    main()
