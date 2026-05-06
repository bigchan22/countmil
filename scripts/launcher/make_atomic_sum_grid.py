#!/usr/bin/env python
"""Write manifests for A1-style ordinal-sum experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def _parse_seeds(raw: list[str]) -> list[int]:
    out: list[int] = []
    for item in raw:
        out.extend(int(x) for x in item.replace(",", " ").split())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--experiments", nargs="+", choices=["mnist_sum", "svhn_sum", "ultramnist"], default=["mnist_sum"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-bags", type=int, default=None)
    parser.add_argument("--test-bags", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output-root", default="configs/atomic_sum")
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.tsv"
    seeds = _parse_seeds(args.seeds)

    rows = []
    for experiment in args.experiments:
        for seed in seeds:
            cmd = [
                "PYTHONPATH=src",
                ".venv/bin/python",
                "scripts/train_atomic_sum.py",
                "--experiment",
                experiment,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--dataset-root",
                args.dataset_root,
                "--results-dir",
                f"results/atomic_sum_{tag}",
                "--run-root",
                f"runs/atomic_sum_{tag}",
            ]
            if args.train_bags is not None:
                cmd.extend(["--train-bags", str(args.train_bags)])
            if args.test_bags is not None:
                cmd.extend(["--test-bags", str(args.test_bags)])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            rows.append((f"{experiment}_pca_s{seed}", " ".join(cmd)))

    with manifest.open("w") as f:
        f.write("job\tcommand\n")
        for job, cmd in rows:
            f.write(f"{job}\t{cmd}\n")
    print(f"wrote {len(rows)} jobs to {manifest}")


if __name__ == "__main__":
    main()
