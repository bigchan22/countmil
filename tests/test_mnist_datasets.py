import unittest
from pathlib import Path

import torch

from countmil.datasets import MNISTBags, MNISTDigitHistogramBags, MNISTDigitSumBags, SignedMNISTBags, collate_mnist_bags, load_mnist_family


@unittest.skipUnless(Path("data/MNIST/raw/train-images-idx3-ubyte").exists(), "local MNIST raw files not present")
class MNISTDatasetTests(unittest.TestCase):
    def test_load_mnist_family_train(self):
        images, labels = load_mnist_family(root="data", dataset="MNIST", split="train")
        self.assertEqual(tuple(images.shape), (60000, 1, 28, 28))
        self.assertEqual(tuple(labels.shape), (60000,))
        self.assertTrue((images >= 0).all())
        self.assertTrue((images <= 1).all())

    def test_binary_mnist_bag_fields_are_consistent(self):
        ds = MNISTBags(root="data", split="train", num_bags=4, bag_size=8, seed=123, balanced_binary=True)
        item = ds[0]
        self.assertEqual(tuple(item["instances"].shape), (8, 1, 28, 28))
        self.assertEqual(tuple(item["digits"].shape), (8,))
        self.assertEqual(tuple(item["instance_labels"].shape), (8,))
        self.assertEqual(item["count"].item(), item["instance_labels"].sum().item())
        self.assertEqual(item["bag_label"].item(), int(item["count"].item() > 0))

    def test_digit_sum_bag_fields_are_consistent(self):
        ds = MNISTDigitSumBags(root="data", split="test", num_bags=3, bag_size=6, seed=456)
        item = ds[1]
        self.assertEqual(tuple(item["instances"].shape), (6, 1, 28, 28))
        self.assertEqual(item["sum"].item(), item["digits"].sum().item())

    def test_digit_histogram_bag_fields_are_consistent(self):
        ds = MNISTDigitHistogramBags(root="data", split="test", num_bags=3, bag_size=6, seed=456)
        item = ds[1]
        self.assertEqual(tuple(item["instances"].shape), (6, 1, 28, 28))
        self.assertEqual(tuple(item["digit_counts"].shape), (10,))
        self.assertEqual(item["digit_counts"].sum().item(), item["digits"].numel())
        self.assertTrue(torch.equal(item["digit_counts"], torch.bincount(item["digits"], minlength=10)))
        self.assertTrue(torch.allclose(item["digit_proportions"].sum(), torch.tensor(1.0)))

    def test_collate_keeps_digit_histogram_vectors(self):
        ds = MNISTDigitHistogramBags(root="data", split="test", num_bags=2, bag_size=6, bag_size_std=1, seed=456)
        batch = collate_mnist_bags([ds[0], ds[1]])
        self.assertEqual(tuple(batch["digit_counts"].shape), (2, 10))
        self.assertEqual(tuple(batch["digit_proportions"].shape), (2, 10))

    def test_signed_mnist_bag_fields_are_consistent(self):
        ds = SignedMNISTBags(
            root="data",
            split="train",
            num_bags=3,
            bag_size=7,
            seed=789,
            cancellation_heavy=True,
        )
        item = ds[2]
        self.assertEqual(tuple(item["instances"].shape), (7, 1, 28, 28))
        self.assertTrue(torch.isin(item["signs"], torch.tensor([-1, 1])).all())
        self.assertTrue(torch.equal(item["signed_instance_labels"], item["signs"] * item["instance_labels"]))
        self.assertEqual(item["signed_count"].item(), item["signed_instance_labels"].sum().item())


if __name__ == "__main__":
    unittest.main()
