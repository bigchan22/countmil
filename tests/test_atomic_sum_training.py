import unittest

import torch
from torch.utils.data import DataLoader

from countmil.datasets import TensorOrdinalSumBags, TensorSignedBags, collate_mnist_bags, collate_ordinal_sum_bags
from countmil.metrics import multiclass_ece_from_pmf
from scripts.train_atomic_sum import _evaluate, expected_sum_from_instance_probs, ordinal_sum_pmf


class AtomicSumTrainingTests(unittest.TestCase):
    def test_ordinal_sum_pmf_masks_padding_as_zero_atom(self):
        probs = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        mask = torch.tensor([[True, True, False]])
        pmf = ordinal_sum_pmf(probs, mask)
        self.assertEqual(pmf.support_min, 0)
        self.assertEqual(int(pmf.probs.argmax(dim=-1).item()), 3)

    def test_ordinal_sum_pmf_keeps_gradients(self):
        logits = torch.randn(2, 4, 5, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.ones(2, 4, dtype=torch.bool)
        pmf = ordinal_sum_pmf(probs, mask)
        loss = -pmf.probs[:, 3].clamp_min(1e-12).log().mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_expected_sum_from_instance_probs_masks_padding(self):
        probs = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        mask = torch.tensor([[True, True, False]])
        expected = expected_sum_from_instance_probs(probs, mask)
        self.assertTrue(torch.allclose(expected, torch.tensor([3.0])))

    def test_expected_sum_from_instance_probs_keeps_gradients(self):
        logits = torch.randn(2, 4, 5, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.ones(2, 4, dtype=torch.bool)
        expected = expected_sum_from_instance_probs(probs, mask)
        loss = expected.square().mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_tensor_ordinal_sum_collate(self):
        images = torch.arange(6 * 1 * 2 * 2, dtype=torch.float32).reshape(6, 1, 2, 2)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        ds = TensorOrdinalSumBags(
            images,
            labels,
            num_bags=2,
            bag_size_mean=3,
            bag_size_std=None,
            bag_size_min=3,
            bag_size_max=3,
            seed=7,
        )
        batch = collate_ordinal_sum_bags([ds[0], ds[1]])
        self.assertEqual(tuple(batch["instances"].shape), (2, 3, 1, 2, 2))
        self.assertEqual(tuple(batch["labels"].shape), (2, 3))
        self.assertTrue(batch["mask"].all())
        self.assertTrue(torch.equal(batch["sum"], batch["labels"].sum(dim=1)))

    def test_tensor_signed_bags_match_signed_count_contract(self):
        images = torch.arange(10 * 3 * 4 * 4, dtype=torch.float32).reshape(10, 3, 4, 4)
        labels = torch.arange(10)
        ds = TensorSignedBags(
            images,
            labels,
            num_bags=2,
            bag_size=5,
            bag_size_std=0,
            target_label=9,
            seed=11,
        )
        item = ds[0]
        self.assertTrue(torch.equal(item["signed_instance_labels"], item["signs"] * item["instance_labels"]))
        self.assertEqual(item["signed_count"].item(), item["signed_instance_labels"].sum().item())
        batch = collate_mnist_bags([item, ds[1]])
        self.assertEqual(tuple(batch["instances"].shape), (2, 5, 3, 4, 4))
        self.assertTrue(batch["mask"].all())

    def test_tensor_signed_bags_accept_multiple_target_labels(self):
        images = torch.arange(10 * 1 * 2 * 2, dtype=torch.float32).reshape(10, 1, 2, 2)
        labels = torch.arange(10)
        ds = TensorSignedBags(
            images,
            labels,
            num_bags=1,
            bag_size=10,
            bag_size_std=0,
            target_label=[5, 6, 7, 8, 9],
            seed=5,
        )
        item = ds[0]
        expected = torch.isin(item["labels"], torch.tensor([5, 6, 7, 8, 9])).long()
        self.assertTrue(torch.equal(item["instance_labels"], expected))

    def test_multiclass_ece_perfect_predictions(self):
        probs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        targets = torch.tensor([1, 0])
        self.assertEqual(multiclass_ece_from_pmf(probs, targets), 0.0)

    def test_evaluate_pads_variable_width_pmfs(self):
        images = torch.rand(8, 1, 2, 2)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        ds = TensorOrdinalSumBags(
            images,
            labels,
            num_bags=4,
            bag_size_mean=2,
            bag_size_std=1.0,
            bag_size_min=1,
            bag_size_max=3,
            seed=3,
        )
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_ordinal_sum_bags)

        class ConstantModel(torch.nn.Module):
            def predict_proba(self, x):
                shape = (*x.shape[:2], 3)
                return torch.full(shape, 1.0 / 3.0, dtype=x.dtype, device=x.device)

        metrics = _evaluate(ConstantModel(), loader, torch.device("cpu"))
        self.assertIn("sum_acc", metrics)
        self.assertIn("expected_sum_mae", metrics)


if __name__ == "__main__":
    unittest.main()
