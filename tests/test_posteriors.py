import unittest

import torch

from countmil.posteriors import binary_count_posterior


class PosteriorTests(unittest.TestCase):
    def test_binary_posteriors_sum_to_observed_count(self):
        probs = torch.tensor(
            [[0.2, 0.7, 0.4, 0.9], [0.5, 0.1, 0.8, 0.3]],
            dtype=torch.float64,
        )
        counts = torch.tensor([2, 1])
        q = binary_count_posterior(probs, counts)
        self.assertTrue(torch.allclose(q.sum(dim=1), counts.to(torch.float64), atol=1e-10))
        self.assertTrue(((q >= 0) & (q <= 1)).all())


if __name__ == "__main__":
    unittest.main()

