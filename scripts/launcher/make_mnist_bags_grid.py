#!/usr/bin/env python
"""Generate MNIST-Bags config grid matching Shukla-style settings."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """experiment: mnist_bags
method: {method}
dataset_root: data
dataset: MNIST
target_digit: 9
bag_size_mean: {bag_mean}
bag_size_std: {bag_std}
train_bags: {train_bags}
balanced_binary: true
seed: {seed}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="configs/mnist_bags/shukla_grid")
    parser.add_argument("--methods", nargs="+", default=["conv", "dp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-bags", type=int, nargs="+", default=[50, 100, 150, 200, 300, 400, 500])
    parser.add_argument("--bag-settings", nargs="+", default=["10:2", "50:10", "100:20"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for setting in args.bag_settings:
        mean_s, std_s = setting.split(":", 1)
        bag_mean = int(mean_s)
        bag_std = int(std_s)
        for method in args.methods:
            for train_bags in args.train_bags:
                for seed in args.seeds:
                    path = out_dir / f"{method}_n{bag_mean}_train{train_bags}_s{seed}.yaml"
                    path.write_text(
                        TEMPLATE.format(
                            method=method,
                            bag_mean=bag_mean,
                            bag_std=bag_std,
                            train_bags=train_bags,
                            seed=seed,
                        )
                    )
                    written.append(path)
    print(f"wrote {len(written)} configs to {out_dir}")


if __name__ == "__main__":
    main()

