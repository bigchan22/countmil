#!/usr/bin/env python
"""UltraMNIST experiment runner.

Default config (spec §13 Day 6):
  6 methods × 3 seeds × 60 epochs ≈ ~13.5h overnight. UltraMNIST has natural
  variable N ∈ {3,4,5} (no per-class-cap or noise modulation by default).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import (
    MeanPoolBaseline, PLMulticlassBaseline, CLTGaussianBaseline,
    AttentionPoolingBaseline, DeepSetsBaseline, PCABaseline,
)
from pca.data import UltraMNISTBagDataset, variable_n_collate_fn
from pca.models import PatchEncoder
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 800
DEFAULT_TEST_BAGS = 300
DEFAULT_EPOCHS = 60
K = 9
N_MAX = 5
LR_DEFAULT = 5e-4   # spec §15: UltraMNIST lr=5e-4 (lower than MNIST/SVHN)

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
    '2a':  PLMulticlassBaseline,
    '2b':  CLTGaussianBaseline,
    '3a':  AttentionPoolingBaseline,
    '3b':  DeepSetsBaseline,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(BASELINE_REGISTRY.keys()), required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2])
    p.add_argument('--bag-size-min', type=int, default=3)
    p.add_argument('--bag-size-max', type=int, default=5)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=None)
    p.add_argument('--lr', type=float, default=LR_DEFAULT)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=Path, default=Path('results/ultramnist'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = UltraMNISTBagDataset(
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            train=True, seed=seed)
        test_ds = UltraMNISTBagDataset(
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            train=False, seed=seed)
        backbone = PatchEncoder()
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()), lr=args.lr)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    cap_str = args.per_class_cap if args.per_class_cap is not None else 'none'
    save_dir = (args.output_dir / f"N4_cap{cap_str}_{args.method}")
    config_extra = {
        'experiment': 'ultramnist', 'method': args.method, 'atom_support': K + 1,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': 0.0,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
