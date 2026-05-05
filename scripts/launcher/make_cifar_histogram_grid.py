#!/usr/bin/env python
"""Generate CIFAR histogram-supervision jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


COMMON = """dataset_root: data
dataset: {dataset}
label_level: {label_level}
bag_size_mean: {bag_mean}
bag_size_std: {bag_std}
train_bags: {train_bags}
seed: {seed}
backbone: {backbone}
pretrained: {pretrained}
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-root", default="configs/cifar_histogram")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
    parser.add_argument("--label-level", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-bags", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--bag-settings", nargs="+", default=["50:10", "100:20"])
    parser.add_argument("--objectives", nargs="+", default=["pvc", "kl", "mse"])
    parser.add_argument("--backbone", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--pretrained", action="store_true")
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
                    label_level=args.label_level,
                    bag_mean=bag_mean,
                    bag_std=bag_std,
                    train_bags=train_bags,
                    seed=seed,
                    backbone=args.backbone,
                    pretrained=str(args.pretrained).lower(),
                )
                for objective in args.objectives:
                    name = f"{args.dataset.lower()}_{args.label_level}_hist_{objective}_n{bag_mean}_train{train_bags}_s{seed}"
                    cfg = out_dir / "cifar_histogram" / f"{name}.yaml"
                    _write(cfg, f"experiment: cifar_histogram\nobjective: {objective}\n" + common)
                    rows.append(f"{name}\tscripts/train_cifar_histogram.py\t{cfg}\t\n")

    manifest = out_dir / "manifest.tsv"
    _write(manifest, "".join(rows))
    print(f"wrote {len(rows) - 1} jobs to {manifest}")


if __name__ == "__main__":
    main()
