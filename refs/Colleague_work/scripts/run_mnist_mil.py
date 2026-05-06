#!/usr/bin/env python
"""MNIST-MIL experiment runner.

Per spec §9.2: --bag-size {10,50,100} --method {nll,em} --epochs 100 \\
               --seeds 0,1,2,3,4 --output-dir results/mnist_mil/

Smoke usage:
  python scripts/run_mnist_mil.py --bag-size 10 --method nll \\
    --epochs 2 --num-bags 50 --seeds 0
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make `pca` importable when this file is run directly via `python scripts/run_mnist_mil.py`.
# (Python 3.13's site.py skips `_`-prefixed editable .pth files, so the editable install
# created by uv/hatchling doesn't register the project root on sys.path.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from functools import partial
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.data import MNISTBagDataset
from pca.losses import em_joint_loss, marginal_nll_loss
from pca.models import SmallCNN
from pca.train import run_seeds, select_device

# Default training pool sizes per spec §8.4.
DEFAULT_TRAIN_BAGS = 1000
DEFAULT_TEST_BAGS = 500
DEFAULT_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bag-size', type=int, required=True, choices=[10, 50, 100])
    p.add_argument('--method', choices=['nll', 'em'], required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS,
                   help='Training-pool bag count (test pool fixed at half).')
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', type=Path, default=Path('results/mnist_mil'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, bag_size={args.bag_size}, method={args.method}, "
          f"seeds={args.seeds}, epochs={args.epochs}")

    batch_size = 32 if args.bag_size <= 50 else 16

    def build_fn(seed: int):
        train_ds = MNISTBagDataset(bag_size=args.bag_size, num_bags=args.num_bags,
                                   positive_digit=9, train=True, seed=seed)
        # Test pool: half training count, fixed across methods (paired comparison).
        test_ds = MNISTBagDataset(bag_size=args.bag_size, num_bags=DEFAULT_TEST_BAGS,
                                  positive_digit=9, train=False, seed=seed)
        model = SmallCNN()
        opt = Adam(model.parameters(), lr=args.lr)
        loss_fn = (marginal_nll_loss if args.method == 'nll'
                   else partial(em_joint_loss, lam=args.lam))
        return (
            model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            loss_fn, opt,
        )

    save_dir = args.output_dir / f"N{args.bag_size}_{args.method}"
    config_extra = {
        'experiment': 'mnist_mil', 'bag_size': args.bag_size, 'method': args.method,
        'lam': args.lam if args.method == 'em' else None,
        'lr': args.lr, 'batch_size': batch_size,
        'num_train_bags': args.num_bags, 'num_test_bags': DEFAULT_TEST_BAGS,
    }
    run_seeds(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
