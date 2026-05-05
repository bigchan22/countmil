import unittest

import torch

from scripts.train_mnist_histogram_pvc import class_count_pmf, histogram_pvc_loss


class HistogramPVCTrainingTests(unittest.TestCase):
    def test_class_count_pmf_matches_deterministic_histogram(self):
        probs = torch.zeros(1, 4, 3)
        probs[0, 0, 0] = 1.0
        probs[0, 1, 1] = 1.0
        probs[0, 2, 1] = 1.0
        probs[0, 3, 2] = 1.0
        mask = torch.tensor([[True, True, True, False]])

        pmf = class_count_pmf(probs, mask)

        self.assertEqual(tuple(pmf.shape), (1, 3, 5))
        self.assertAlmostEqual(pmf[0, 0, 1].item(), 1.0)
        self.assertAlmostEqual(pmf[0, 1, 2].item(), 1.0)
        self.assertAlmostEqual(pmf[0, 2, 0].item(), 1.0)

    def test_histogram_pvc_loss_keeps_gradients(self):
        logits = torch.randn(2, 5, 4, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        counts = torch.tensor([[1, 1, 2, 1], [0, 1, 1, 1]])

        loss = histogram_pvc_loss(probs, counts, mask)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
