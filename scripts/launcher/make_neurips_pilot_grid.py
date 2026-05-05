#!/usr/bin/env python
"""Generate a medium NeurIPS pilot grid and launch manifest.

The manifest is TSV to keep the shell launcher dependency-free:

    name    script    config    extra_args

`extra_args` is parsed with shell quoting by the launcher, so this generator
only writes values it controls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


COMMON = """dataset_root: data
dataset: MNIST
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
    parser.add_argument("--out-root", default="configs/neurips_pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-bags", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--bag-settings", nargs="+", default=["10:2", "50:10"])
    parser.add_argument("--posterior-objectives", nargs="+", default=["nll", "hard_em", "tempered_em"])
    parser.add_argument("--include-digit-sum", action="store_true", default=True)
    parser.add_argument("--include-signed", action="store_true", default=True)
    parser.add_argument("--include-posterior", action="store_true", default=True)
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / tag
    manifest = out_dir / "manifest.tsv"
    rows = ["name\tscript\tconfig\textra_args\n"]

    for setting in args.bag_settings:
        mean_s, std_s = setting.split(":", 1)
        bag_mean = int(mean_s)
        bag_std = int(std_s)
        for train_bags in args.train_bags:
            for seed in args.seeds:
                common = COMMON.format(bag_mean=bag_mean, bag_std=bag_std, train_bags=train_bags, seed=seed)

                if args.include_posterior:
                    for objective in args.posterior_objectives:
                        name = f"posterior_{objective}_n{bag_mean}_train{train_bags}_s{seed}"
                        cfg = out_dir / "mnist_bags" / f"{name}.yaml"
                        extra = ""
                        if objective == "tempered_em":
                            extra = "--em-temperature 0.5"
                        _write(
                            cfg,
                            "experiment: mnist_bags\n"
                            "method: conv\n"
                            f"objective: {objective}\n"
                            "target_digit: 9\n"
                            "balanced_binary: true\n"
                            + common,
                        )
                        rows.append(f"{name}\tscripts/train_mnist_bags.py\t{cfg}\t{extra}\n")

                if args.include_digit_sum:
                    name = f"digit_sum_n{bag_mean}_train{train_bags}_s{seed}"
                    cfg = out_dir / "digit_sum" / f"{name}.yaml"
                    _write(cfg, "experiment: mnist_digit_sum\nmethod: finite_support_conv\n" + common)
                    rows.append(f"{name}\tscripts/train_mnist_digit_sum.py\t{cfg}\t\n")

                if args.include_signed:
                    for cancellation in [False, True]:
                        suffix = "cancel" if cancellation else "random"
                        name = f"signed_{suffix}_n{bag_mean}_train{train_bags}_s{seed}"
                        cfg = out_dir / "signed" / f"{name}.yaml"
                        _write(
                            cfg,
                            "experiment: signed_mnist\n"
                            "method: signed_conv_countmil\n"
                            "target_digit: 9\n"
                            f"cancellation_heavy: {str(cancellation).lower()}\n"
                            + common,
                        )
                        rows.append(f"{name}\tscripts/train_signed_mnist.py\t{cfg}\t\n")

    _write(manifest, "".join(rows))
    print(f"wrote {len(rows) - 1} jobs to {manifest}")


if __name__ == "__main__":
    main()
