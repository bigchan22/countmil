import unittest

import torch

from countmil.aggregators import (
    aggregate_nll,
    binary_count_dp,
    brute_force_binary_count,
    brute_force_finite_support,
    finite_support_convolution,
    grouped_signed_binary_convolution,
)


class AggregatorTests(unittest.TestCase):
    def test_binary_bruteforce_dp_and_convolution_agree(self):
        probs = torch.tensor([0.2, 0.7, 0.4, 0.9], dtype=torch.float64)
        brute = brute_force_binary_count(probs)
        dp = binary_count_dp(probs)
        atoms = torch.stack([1.0 - probs, probs], dim=-1)
        conv = finite_support_convolution(atoms, support_min=0)

        self.assertTrue(torch.allclose(brute.probs, dp.probs, atol=1e-12))
        self.assertTrue(torch.allclose(dp.probs, conv.probs, atol=1e-12))
        self.assertEqual(conv.support_min, 0)
        self.assertEqual(conv.support_max, 4)

    def test_ordinal_convolution_support_and_mass(self):
        atoms = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]],
                [[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]],
            ],
            dtype=torch.float64,
        )
        agg = finite_support_convolution(atoms, support_min=0)
        self.assertEqual(tuple(agg.probs.shape), (2, 5))
        self.assertTrue(torch.allclose(agg.probs.sum(dim=-1), torch.ones(2, dtype=torch.float64)))

    def test_ordinal_convolution_matches_bruteforce(self):
        atoms = torch.tensor(
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]],
            dtype=torch.float64,
        )
        brute = brute_force_finite_support(atoms, support_min=0)
        conv = finite_support_convolution(atoms, support_min=0)
        self.assertTrue(torch.allclose(brute.probs, conv.probs, atol=1e-12))
        self.assertEqual(conv.support_min, 0)
        self.assertEqual(conv.support_max, 6)

    def test_negative_support_convolution_matches_bruteforce(self):
        atoms = torch.tensor(
            [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.6, 0.1, 0.3]],
            dtype=torch.float64,
        )
        brute = brute_force_finite_support(atoms, support_min=-1)
        conv = finite_support_convolution(atoms, support_min=-1)
        self.assertTrue(torch.allclose(brute.probs, conv.probs, atol=1e-12))
        self.assertEqual(conv.support_min, -3)
        self.assertEqual(conv.support_max, 3)

    def test_grouped_signed_conv_matches_generic(self):
        probs = torch.tensor([[0.2, 0.7, 0.4, 0.3]], dtype=torch.float64)
        signs = torch.tensor([[1, -1, 1, -1]])
        groups = torch.tensor([[0, 0, 1, 1]])

        grouped = grouped_signed_binary_convolution(probs, signs, groups, num_groups=2)
        expected = []
        for g in range(2):
            mask = groups[0] == g
            atoms = []
            for p, s in zip(probs[0, mask], signs[0, mask]):
                if int(s.item()) < 0:
                    atoms.append(torch.stack([p, 1.0 - p, p.new_tensor(0.0)]))
                else:
                    atoms.append(torch.stack([p.new_tensor(0.0), 1.0 - p, p]))
            expected.append(finite_support_convolution(torch.stack(atoms), support_min=-1).probs)
        expected = torch.stack(expected).unsqueeze(0)

        self.assertEqual(grouped.support_min, -2)
        self.assertTrue(torch.allclose(grouped.probs, expected, atol=1e-12))

    def test_grouped_signed_conv_multiple_batches(self):
        probs = torch.tensor(
            [[0.2, 0.7, 0.4, 0.3], [0.6, 0.1, 0.8, 0.5]],
            dtype=torch.float64,
        )
        signs = torch.tensor([[1, -1, 1, -1], [-1, -1, 1, 1]])
        groups = torch.tensor([[0, 0, 1, 1], [0, 1, 1, -1]])
        grouped = grouped_signed_binary_convolution(probs, signs, groups, num_groups=2)

        for b in range(2):
            for g in range(2):
                mask = groups[b] == g
                atoms = []
                for p, s in zip(probs[b, mask], signs[b, mask]):
                    if int(s.item()) < 0:
                        atoms.append(torch.stack([p, 1.0 - p, p.new_tensor(0.0)]))
                    else:
                        atoms.append(torch.stack([p.new_tensor(0.0), 1.0 - p, p]))
                expected = finite_support_convolution(torch.stack(atoms), support_min=-1)
                offset = expected.support_min - grouped.support_min
                got = grouped.probs[b, g, offset : offset + expected.probs.numel()]
                self.assertTrue(torch.allclose(got, expected.probs, atol=1e-12))

    def test_aggregate_nll_indexing(self):
        pmf = binary_count_dp(torch.tensor([0.25, 0.5], dtype=torch.float64))
        nll = aggregate_nll(pmf, torch.tensor(1))
        self.assertTrue(torch.allclose(nll, -torch.log(torch.tensor(0.5, dtype=torch.float64))))

    def test_aggregate_nll_out_of_range_is_large(self):
        pmf = binary_count_dp(torch.tensor([0.25, 0.5], dtype=torch.float64))
        nll = aggregate_nll(pmf, torch.tensor(10))
        self.assertTrue(torch.isfinite(nll))

    def test_gradients_are_finite_for_dp_and_convolution(self):
        logits = torch.tensor([[0.2, -0.7, 1.4]], dtype=torch.float64, requires_grad=True)
        probs = torch.sigmoid(logits)
        dp_loss = aggregate_nll(binary_count_dp(probs), torch.tensor([2])).mean()
        atoms = torch.stack([1.0 - probs, probs], dim=-1)
        conv_loss = aggregate_nll(finite_support_convolution(atoms), torch.tensor([2])).mean()
        loss = dp_loss + conv_loss
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
