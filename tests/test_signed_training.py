import unittest

import torch

from scripts.train_signed_mnist import signed_count_pmf


class SignedTrainingTests(unittest.TestCase):
    def test_signed_count_pmf_deterministic_contributions(self):
        probs = torch.tensor([[1.0, 1.0, 0.0]])
        signs = torch.tensor([[1, -1, 1]])
        mask = torch.tensor([[True, True, True]])

        pmf = signed_count_pmf(probs, signs, mask)

        self.assertEqual(pmf.support_min, -3)
        self.assertEqual(pmf.support_max, 3)
        self.assertAlmostEqual(pmf.probs[0, 3].item(), 1.0)
        self.assertAlmostEqual(pmf.probs.sum().item(), 1.0)

    def test_signed_count_pmf_masks_padding_as_zero(self):
        probs = torch.tensor([[1.0, 1.0]])
        signs = torch.tensor([[1, -1]])
        mask = torch.tensor([[True, False]])

        pmf = signed_count_pmf(probs, signs, mask)

        self.assertAlmostEqual(pmf.probs[0, 3].item(), 1.0)
        self.assertAlmostEqual(pmf.probs.sum().item(), 1.0)

    def test_signed_count_pmf_keeps_gradients(self):
        logits = torch.randn(2, 4, requires_grad=True)
        probs = torch.sigmoid(logits)
        signs = torch.tensor([[1, -1, 1, -1], [-1, 1, 1, -1]])
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

        pmf = signed_count_pmf(probs, signs, mask)
        loss = pmf.probs.square().mean()
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
