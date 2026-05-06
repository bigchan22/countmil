#!/usr/bin/env python
"""LLP Adult experiment runner.

Per spec §9.3: --bag-size {32,128,512} --method {nll,em} --epochs 200 \\
               --seeds 0,1,2,3,4 --output-dir results/llp_adult/
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make `pca` importable when this file is run directly via `python scripts/run_llp_adult.py`.
# (Python 3.13's site.py skips `_`-prefixed editable .pth files, so the editable install
# created by uv/hatchling doesn't register the project root on sys.path.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from functools import partial
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader

from pca.data import LLPBagDataset
from pca.losses import em_joint_loss, marginal_nll_loss
from pca.models import MLP
from pca.train import run_seeds, select_device


DEFAULT_TRAIN_BAGS = 2000
DEFAULT_TEST_BAGS = 1000
DEFAULT_EPOCHS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bag-size', type=int, required=True, choices=[32, 128, 512])
    p.add_argument('--method', choices=['nll', 'em'], required=True)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')],
                   default=[0, 1, 2, 3, 4])
    p.add_argument('--num-bags', type=int, default=DEFAULT_TRAIN_BAGS)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', type=Path, default=Path('results/llp_adult'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"device: {device}, bag_size={args.bag_size}, method={args.method}, "
          f"seeds={args.seeds}, epochs={args.epochs}")

    if args.bag_size <= 128:
        batch_size = 32
    else:
        batch_size = 8

    # Inspect feature dim once to pin MLP in_dim.
    probe = LLPBagDataset('adult', bag_size=args.bag_size, num_bags=1, train=True, seed=0)
    feats, _, _ = probe[0]
    in_dim = feats.shape[1]

    def build_fn(seed: int):
        train_ds = LLPBagDataset('adult', bag_size=args.bag_size,
                                 num_bags=args.num_bags, train=True, seed=seed)
        test_ds = LLPBagDataset('adult', bag_size=args.bag_size,
                                num_bags=DEFAULT_TEST_BAGS, train=False, seed=seed)
        model = MLP(in_dim=in_dim)
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
        'experiment': 'llp_adult', 'bag_size': args.bag_size, 'method': args.method,
        'lam': args.lam if args.method == 'em' else None,
        'lr': args.lr, 'batch_size': batch_size, 'in_dim': in_dim,
        'num_train_bags': args.num_bags, 'num_test_bags': DEFAULT_TEST_BAGS,
    }
    run_seeds(args.seeds, build_fn, args.epochs, device, save_dir, config_extra)
    print(f"[done] results in {save_dir}")


if __name__ == "__main__":
    main()
