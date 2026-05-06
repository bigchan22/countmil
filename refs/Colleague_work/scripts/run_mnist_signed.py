#!/usr/bin/env python
"""MNIST-signed-sum experiment runner.

Signed atom case: K=2 (S=3 over {-1,0,+1}), bag-sum support T = 2*N+1 (shifted).
Default config (spec §13 Day 4): 5 seeds × 80 epochs × 2 methods (PCA + 1a baseline).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.baselines import MeanPoolBaseline, PCABaseline
from pca.data import MNISTSignedSumBagDataset, variable_n_collate_fn
from pca.models import SmallCNNMulticlass
from pca.train import run_seeds_a1, select_device

DEFAULT_TRAIN_BAGS = 1500
DEFAULT_TEST_BAGS = 600
DEFAULT_EPOCHS = 80
K = 2     # support {-1, 0, +1} with caller-side shift → K=2 in atomic_conv
N_MAX = 15

BASELINE_REGISTRY = {
    'pca': PCABaseline,
    '1a':  MeanPoolBaseline,
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
    p.add_argument('--per-class-cap', type=int, default=100)
    p.add_argument('--noise-sigma', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output-dir', type=Path, default=Path('results/mnist_signed'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, method={args.method}, seeds={args.seeds}, "
          f"epochs={args.epochs}, cap={args.per_class_cap}, σ={args.noise_sigma}")
    BaselineCls = BASELINE_REGISTRY[args.method]

    def build_fn(seed: int):
        train_ds = MNISTSignedSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=True, seed=seed)
        test_ds = MNISTSignedSumBagDataset(
            bag_size_mean=args.bag_size_mean, bag_size_std=args.bag_size_std,
            bag_size_min=args.bag_size_min, bag_size_max=args.bag_size_max,
            num_bags=args.num_test_bags, per_class_cap=args.per_class_cap,
            noise_sigma=args.noise_sigma, train=False, seed=seed)
        backbone = SmallCNNMulticlass()
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

    save_dir = (args.output_dir / f"N{int(args.bag_size_mean)}_"
                                   f"cap{args.per_class_cap}_"
                                   f"sig{args.noise_sigma}_{args.method}")
    config_extra = {
        'experiment': 'mnist_signed', 'method': args.method, 'atom_support': K + 1,
        'bag_size_mean': args.bag_size_mean, 'bag_size_std': args.bag_size_std,
        'bag_size_min': args.bag_size_min, 'bag_size_max': args.bag_size_max,
        'per_class_cap': args.per_class_cap, 'noise_sigma': args.noise_sigma,
        'lr': args.lr, 'batch_size': args.batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': args.num_test_bags,
        'K': K, 'N_max': N_MAX,
    }
    run_seeds_a1(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
