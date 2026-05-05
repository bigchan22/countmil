"""Binary Count Loss baseline matching Shukla et al.'s DP likelihood."""

from __future__ import annotations

import torch

from countmil.aggregators import aggregate_nll, binary_count_dp


def shukla_count_loss(probs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    pmf = binary_count_dp(probs)
    return aggregate_nll(pmf, counts).mean()

