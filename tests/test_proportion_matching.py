import unittest

import torch

from countmil.baselines.proportion_matching import multiclass_proportion_matching_loss


class ProportionMatchingTests(unittest.TestCase):
    def test_multiclass_proportion_loss_is_small_when_matching(self):
        probs = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ]
        )
        mask = torch.tensor([[True, True], [True, False]])
        target = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])

        loss = multiclass_proportion_matching_loss(probs, target, mask, loss="kl")

        self.assertLess(loss.item(), 1e-4)

    def test_multiclass_proportion_loss_keeps_gradients(self):
        logits = torch.randn(2, 4, 3, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        target = torch.tensor([[0.25, 0.25, 0.5], [0.5, 0.5, 0.0]])

        loss = multiclass_proportion_matching_loss(probs, target, mask, loss="mse")
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
