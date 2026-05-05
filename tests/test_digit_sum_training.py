import unittest

import torch

from scripts.train_mnist_digit_sum import digit_sum_pmf


class DigitSumTrainingTests(unittest.TestCase):
    def test_digit_sum_pmf_masks_padding_as_zero(self):
        probs = torch.zeros(1, 3, 10)
        probs[0, 0, 2] = 1.0
        probs[0, 1, 5] = 1.0
        probs[0, 2, 9] = 1.0
        mask = torch.tensor([[True, True, False]])

        pmf = digit_sum_pmf(probs, mask)

        self.assertEqual(pmf.support_min, 0)
        self.assertEqual(pmf.support_max, 27)
        self.assertAlmostEqual(pmf.probs[0, 7].item(), 1.0)
        self.assertAlmostEqual(pmf.probs.sum().item(), 1.0)

    def test_digit_sum_pmf_keeps_gradients(self):
        logits = torch.randn(2, 4, 10, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

        pmf = digit_sum_pmf(probs, mask)
        loss = pmf.probs.square().mean()
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
