from __future__ import annotations

import itertools

import numpy as np
import torch

from countmil.aggregators import brute_force_binary_count
from countmil.criteo.losses import (
    dllp_bce_loss,
    dllp_mse_loss,
    easyllp_loss,
    fsconv_nll_loss,
    genbags_covariance,
    genbags_loss,
    ot_llp_loss,
    ot_llp_pseudo_labels,
    poisson_binomial_pmfs,
)


def test_poisson_binomial_matches_bruteforce():
    probs = torch.tensor([[0.2, 0.4, 0.7]], dtype=torch.float64)
    got = poisson_binomial_pmfs(probs)[0]
    expected = brute_force_binary_count(probs[0]).probs
    assert torch.allclose(got, expected, atol=1e-12)
    assert torch.allclose(got.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_fsconv_gradients_are_finite():
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([0, 2, 5])
    loss = fsconv_nll_loss(logits, counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_dllp_bce_and_mse_reference_values():
    logits = torch.zeros(2, 4)
    counts = torch.tensor([1, 3])
    bce = dllp_bce_loss(logits, counts, 4)
    mse = dllp_mse_loss(logits, counts)
    assert torch.allclose(bce, torch.tensor(0.69314718), atol=1e-6)
    assert torch.allclose(mse, torch.tensor(1.0))


def test_easyllp_matches_numpy_reference():
    logits = torch.tensor([[0.0, 0.5, -0.25, 0.1]], dtype=torch.float64)
    counts = torch.tensor([2])
    prior = 0.3
    got = easyllp_loss(logits, counts, bag_size=4, global_prior=prior)
    p = torch.sigmoid(logits).numpy()[0]
    q = 2 / 4
    w_pos = 4 * (q - prior) + prior
    w_neg = 4 * (prior - q) + (1 - prior)
    ref = w_pos * (-np.log(p).mean()) + w_neg * (-np.log1p(-p).mean())
    assert np.allclose(float(got), ref)


def test_genbags_deterministic_matches_numpy_reference_for_two_blocks():
    logits = torch.tensor(
        [
            [-1.0, 0.0, 1.0, 0.5],
            [0.2, 0.1, -0.3, 1.5],
            [-0.7, 0.4, 0.6, -0.2],
            [1.2, -1.1, 0.3, 0.9],
            [0.1, 0.2, 0.3, 0.4],
            [-0.5, -0.4, 0.8, 1.0],
            [0.6, 0.7, -0.2, -0.1],
            [1.1, -0.9, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    counts = torch.tensor([1, 2, 1, 3, 2, 1, 2, 2])
    got = genbags_loss(logits, counts, stochastic=False)
    diffs = (torch.sigmoid(logits).sum(dim=1) - counts.float()).numpy()
    cov = genbags_covariance(block_size=4, dtype=torch.float64).numpy()
    ref = np.mean([d @ cov @ d for d in diffs.reshape(2, 4)])
    assert np.allclose(float(got), ref)


def test_genbags_stochastic_eight_bags_uses_120_generalized_bags():
    logits = torch.zeros(8, 4, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([2] * 8)
    loss = genbags_loss(logits, counts, block_size=4, num_gen_bags_per_block=60, stochastic=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_ot_llp_pseudo_labels_match_counts_exactly_and_train():
    logits = torch.tensor([[0.1, 2.0, 0.3, 1.5], [1.0, 0.0, -1.0, 0.5]], requires_grad=True)
    counts = torch.tensor([2, 0])
    pseudo = ot_llp_pseudo_labels(logits, counts)
    assert pseudo.sum(dim=1).tolist() == [2.0, 0.0]
    assert pseudo[0].tolist() == [0.0, 1.0, 0.0, 1.0]
    loss = ot_llp_loss(logits, counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
