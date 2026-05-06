#!/usr/bin/env python
"""SVHN-sum experiment runner.

Default config: 6 methods × 5 seeds × 80 epochs on ResNet18FromScratch.

Hyperparameter note (2026-05-05): the original spec §13 Day 5 defaults
(`lr=1e-3`, no augmentation, no weight decay) caused catastrophic overfitting
on a 1500-bag train pool. New defaults: `lr=1e-4`, `weight_decay=5e-4`, plus
RandomCrop(32, padding=4) + ColorJitter augmentation in SVHNSumBagDataset
when train=True. See docs/notes/2026-05-05-a1-verdict-reframing.md.

Smoke usage:
  python scripts/run_svhn_sum.py --method pca --epochs 2 --num-bags 50 --seeds 0
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
from pca.data import SVHNSumBagDataset, variable_n_collate_fn
from pca.models import ResNet18FromScratch
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 1500
DEFAULT_TEST_BAGS = 600
DEFAULT_EPOCHS = 80
K = 9
N_MAX = 15

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
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--bag-size-mean', type=float, default=10.0)
    p.add_argument('--bag-size-std', type=float, default=2.0)
    p.add_argument('--bag-size-min', type=int, default=5)
    p.add_argument('--bag-size-max', type=int, default=15)
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--num-test-bags', type=int, default=DEFAULT_TEST_BAGS)
    p.add_argument('--per-class-cap', type=int, default=None)
    p.add_argument('--noise-sigma', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=Path, default=Path('results/svhn_sum'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}, σ={args.noise_sigma}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = SVHNSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=True, seed=seed)
        test_ds = SVHNSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=False, seed=seed)
        backbone = ResNet18FromScratch(in_channels=3)
        baseline = BaselineCls(feature_dim=backbone.feature_dim, K=K, N_max=N_MAX)
        opt = Adam(list(backbone.parameters()) + list(baseline.parameters()),
                   lr=args.lr, weight_decay=args.weight_decay)
        return (
            backbone, baseline,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=variable_n_collate_fn),
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=variable_n_collate_fn),
            opt,
        )

    cap_str = args.per_class_cap if args.per_class_cap is not None else 'none'
    save_dir = (args.output_dir / f"N{int(args.bag_size_mean)}_"
                                   f"cap{cap_str}_"
                                   f"sig{args.noise_sigma}_{args.method}")
    config_extra = {
        'experiment': 'svhn_sum', 'method': args.method, 'atom_support': K + 1,
        'bag_size_mean': args.bag_size_mean, 'bag_size_std': args.bag_size_std,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': args.noise_sigma,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'augment': True,  # tied to dataset's train flag
        'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
