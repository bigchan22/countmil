#!/usr/bin/env python
"""Numerically compare FS-Conv classwise counts to the PVC count component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from countmil.aggregators import AggregatePMF, aggregate_nll, binary_count_dp, finite_support_convolution_fft_tree


def fsconv_classwise_loss(logits: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    atoms = torch.stack([1.0 - probs.transpose(1, 2), probs.transpose(1, 2)], dim=-1)
    pmfs = finite_support_convolution_fft_tree(atoms, support_min=0).probs
    losses = [aggregate_nll(AggregatePMF(pmfs[:, c], 0), counts[:, c]) for c in range(pmfs.shape[1])]
    return torch.stack(losses, dim=-1).mean(), pmfs


def pvc_count_component_loss(logits: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference Poisson-binomial count component with the same softmax probs."""

    probs = torch.softmax(logits, dim=-1)
    pieces = []
    losses = []
    for c in range(probs.shape[-1]):
        pmf = binary_count_dp(probs[:, :, c]).probs
        pieces.append(pmf)
        losses.append(aggregate_nll(AggregatePMF(pmf, 0), counts[:, c]))
    return torch.stack(losses, dim=-1).mean(), torch.stack(pieces, dim=1)


def make_valid_counts(batch: int, bag_size: int, classes: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (batch, bag_size), generator=gen)
    return torch.stack([torch.bincount(row, minlength=classes) for row in labels], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--bag-size", type=int, default=13)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/rebuttal/overnight_4090/pvc_equivalence.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    logits_a = torch.randn(args.batch_size, args.bag_size, args.classes, dtype=torch.float64, requires_grad=True)
    counts = make_valid_counts(args.batch_size, args.bag_size, args.classes, args.seed + 99)
    logits_b = logits_a.detach().clone().requires_grad_(True)

    fs_loss, fs_pmfs = fsconv_classwise_loss(logits_a, counts)
    pvc_loss, pvc_pmfs = pvc_count_component_loss(logits_b, counts)
    fs_loss.backward()
    pvc_loss.backward()

    result = {
        "batch_size": args.batch_size,
        "bag_size": args.bag_size,
        "classes": args.classes,
        "seed": args.seed,
        "fsconv_loss": float(fs_loss.detach()),
        "pvc_count_component_loss": float(pvc_loss.detach()),
        "max_abs_forward_loss_diff": float((fs_loss.detach() - pvc_loss.detach()).abs()),
        "max_abs_per_class_probability_diff": float((fs_pmfs.detach() - pvc_pmfs.detach()).abs().max()),
        "max_abs_logit_gradient_diff": float((logits_a.grad - logits_b.grad).abs().max()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
