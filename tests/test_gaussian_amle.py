import unittest

import torch

from countmil.aggregators import finite_support_convolution
from countmil.baselines.gaussian_amle import (
    categorical_gaussian_amle_loss,
    categorical_sum_moments,
    signed_bernoulli_gaussian_amle_loss,
    signed_bernoulli_sum_moments,
)


def _pmf_moments(pmf):
    values = torch.arange(pmf.support_min, pmf.support_max + 1, dtype=pmf.probs.dtype)
    mean = (pmf.probs * values).sum(dim=-1)
    var = (pmf.probs * values.square()).sum(dim=-1) - mean.square()
    return mean, var.clamp_min(0.0)


class GaussianAMLETests(unittest.TestCase):
    def test_categorical_moments_match_exact_convolution(self):
        atoms = torch.tensor(
            [[[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]],
            dtype=torch.float64,
        )
        support = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        moments = categorical_sum_moments(atoms, support)
        pmf = finite_support_convolution(atoms, support_min=0)
        mean, var = _pmf_moments(pmf)
        self.assertTrue(torch.allclose(moments.mean, mean, atol=1e-12))
        self.assertTrue(torch.allclose(moments.variance, var, atol=1e-12))

    def test_signed_moments_are_correct(self):
        p = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float64)
        s = torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float64)
        moments = signed_bernoulli_sum_moments(p, s)
        self.assertTrue(torch.allclose(moments.mean, torch.tensor([-0.1], dtype=torch.float64)))
        self.assertTrue(torch.allclose(moments.variance, torch.tensor([0.61], dtype=torch.float64)))

    def test_gradients_are_finite(self):
        logits = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        support = torch.arange(5, dtype=torch.float64)
        loss = categorical_gaussian_amle_loss(
            probs,
            support,
            torch.tensor([5.0, 8.0], dtype=torch.float64),
        ).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_nearly_deterministic_atoms_remain_finite(self):
        probs = torch.tensor([[[1.0 - 1e-12, 1e-12], [1e-12, 1.0 - 1e-12]]], dtype=torch.float64)
        support = torch.tensor([0.0, 1.0], dtype=torch.float64)
        loss = categorical_gaussian_amle_loss(probs, support, torch.tensor([1.0], dtype=torch.float64), eps=1e-8)
        self.assertTrue(torch.isfinite(loss).all())

    def test_padding_contributes_zero_moments(self):
        probs = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
        mask = torch.tensor([[True, False]])
        support = torch.tensor([0.0, 1.0], dtype=torch.float64)
        moments = categorical_sum_moments(probs, support, mask)
        self.assertTrue(torch.allclose(moments.mean, torch.tensor([1.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(moments.variance, torch.tensor([0.0], dtype=torch.float64)))

    def test_batch_and_per_bag_agree(self):
        logits = torch.randn(3, 4, 2, dtype=torch.float64)
        probs = torch.softmax(logits, dim=-1)
        support = torch.tensor([0.0, 1.0], dtype=torch.float64)
        batch = categorical_sum_moments(probs, support)
        means = []
        vars_ = []
        for i in range(probs.shape[0]):
            one = categorical_sum_moments(probs[i : i + 1], support)
            means.append(one.mean[0])
            vars_.append(one.variance[0])
        self.assertTrue(torch.allclose(batch.mean, torch.stack(means)))
        self.assertTrue(torch.allclose(batch.variance, torch.stack(vars_)))

    def test_signed_loss_has_finite_gradients(self):
        logits = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        probs = torch.sigmoid(logits)
        signs = torch.tensor([[1, -1, 1, -1, 1, -1], [-1, -1, 1, 1, -1, 1]], dtype=torch.float64)
        loss = signed_bernoulli_gaussian_amle_loss(
            probs,
            signs,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
