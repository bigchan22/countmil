#!/usr/bin/env python
"""Generate a conservative FashionMNIST robustness grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


COMMON = """dataset_root: data
dataset: FashionMNIST
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
    parser.add_argument("--out-root", default="configs/fashionmnist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-bags", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--bag-settings", nargs="+", default=["10:2", "50:10"])
    parser.add_argument("--target-digit", type=int, default=9)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["binary", "attention", "digit_sum", "histogram_llp", "signed"],
        choices=["binary", "attention", "digit_sum", "histogram_llp", "signed"],
    )
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
                common = COMMON.format(bag_mean=bag_mean, bag_std=bag_std, train_bags=train_bags, seed=seed)
                suffix = f"n{bag_mean}_train{train_bags}_s{seed}"

                if "binary" in args.experiments:
                    name = f"fashion_binary_countmil_{suffix}"
                    cfg = out_dir / "mnist_bags" / f"{name}.yaml"
                    _write(
                        cfg,
                        "experiment: fashionmnist_bags\n"
                        "method: conv\n"
                        "objective: nll\n"
                        f"target_digit: {args.target_digit}\n"
                        "balanced_binary: true\n"
                        + common,
                    )
                    rows.append(f"{name}\tscripts/train_mnist_bags.py\t{cfg}\t\n")

                if "attention" in args.experiments:
                    for gated in [False, True]:
                        method = "gated_attention" if gated else "attention"
                        name = f"fashion_{method}_{suffix}"
                        cfg = out_dir / "attention" / f"{name}.yaml"
                        _write(
                            cfg,
                            "experiment: fashionmnist_attention\n"
                            f"method: {method}\n"
                            f"target_digit: {args.target_digit}\n"
                            "balanced_binary: true\n"
                            f"gated: {str(gated).lower()}\n"
                            + common,
                        )
                        rows.append(f"{name}\tscripts/train_mnist_bags_attention.py\t{cfg}\t\n")

                if "digit_sum" in args.experiments:
                    name = f"fashion_digit_sum_{suffix}"
                    cfg = out_dir / "digit_sum" / f"{name}.yaml"
                    _write(cfg, "experiment: fashionmnist_digit_sum\nmethod: finite_support_conv\n" + common)
                    rows.append(f"{name}\tscripts/train_mnist_digit_sum.py\t{cfg}\t\n")

                if "histogram_llp" in args.experiments:
                    name = f"fashion_histogram_llp_kl_{suffix}"
                    cfg = out_dir / "histogram_llp" / f"{name}.yaml"
                    _write(cfg, "experiment: fashionmnist_histogram_llp\nloss: kl\n" + common)
                    rows.append(f"{name}\tscripts/train_mnist_histogram_llp.py\t{cfg}\t\n")

                if "signed" in args.experiments:
                    for cancellation in [False, True]:
                        mode = "cancel" if cancellation else "random"
                        name = f"fashion_signed_{mode}_{suffix}"
                        cfg = out_dir / "signed" / f"{name}.yaml"
                        _write(
                            cfg,
                            "experiment: fashionmnist_signed\n"
                            "method: signed_conv_countmil\n"
                            f"target_digit: {args.target_digit}\n"
                            f"cancellation_heavy: {str(cancellation).lower()}\n"
                            + common,
                        )
                        rows.append(f"{name}\tscripts/train_signed_mnist.py\t{cfg}\t\n")

    manifest = out_dir / "manifest.tsv"
    _write(manifest, "".join(rows))
    print(f"wrote {len(rows) - 1} jobs to {manifest}")


if __name__ == "__main__":
    main()
