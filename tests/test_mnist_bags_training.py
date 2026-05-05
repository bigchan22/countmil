import unittest

import torch

from scripts.train_mnist_bags import _hard_marginals, _tempered_marginals, _training_loss


class MNISTBagsTrainingTests(unittest.TestCase):
    def test_hard_marginals_respect_counts_and_mask(self):
        q = torch.tensor([[0.1, 0.8, 0.4, 0.9], [0.7, 0.6, 0.5, 0.4]])
        counts = torch.tensor([2, 1])
        mask = torch.tensor([[True, True, True, True], [True, False, False, False]])

        hard = _hard_marginals(q, counts, mask)

        self.assertTrue(torch.equal(hard[0], torch.tensor([0.0, 1.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(hard[1], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(hard.sum(dim=1), torch.tensor([2.0, 1.0])))

    def test_tempered_marginals_sharpen_when_temperature_below_one(self):
        q = torch.tensor([0.2, 0.8])
        sharpened = _tempered_marginals(q, temperature=0.5)

        self.assertLess(sharpened[0].item(), q[0].item())
        self.assertGreater(sharpened[1].item(), q[1].item())

    def test_training_loss_supports_posterior_objectives(self):
        logits = torch.tensor([[0.2, -0.4, 0.8], [0.1, 0.3, -0.7]], requires_grad=True)
        probs = torch.sigmoid(logits)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        counts = torch.tensor([2, 1])

        for objective in ["nll", "nll_entropy", "soft_em", "tempered_em", "hard_em"]:
            loss, extras = _training_loss(
                logits=logits,
                probs=probs,
                mask=mask,
                counts=counts,
                method="conv",
                objective=objective,
                entropy_weight=0.01,
                em_temperature=0.5,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("batch_nll", extras)

        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
