import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from countmil.datasets import CIFARHistogramBags, CIFARSignedBags, collate_cifar_bags, collate_mnist_bags, load_cifar_family


def _write_pickle(path: Path, obj: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


class CIFARDatasetTests(unittest.TestCase):
    def test_cifar100_tar_loader_and_bags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "cifar-100-python"
            extracted.mkdir()
            data = torch.arange(6 * 3 * 32 * 32, dtype=torch.uint8).reshape(6, -1).numpy()
            obj = {
                "data": data,
                "coarse_labels": [0, 1, 1, 2, 2, 2],
                "fine_labels": [0, 3, 4, 7, 8, 9],
            }
            _write_pickle(extracted / "train", obj)
            _write_pickle(extracted / "test", obj)
            archive = root / "cifar-100-python.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(extracted, arcname="cifar-100-python")

            images, labels, num_classes = load_cifar_family(root, "CIFAR100", "train", "coarse")
            self.assertEqual(tuple(images.shape), (6, 3, 32, 32))
            self.assertEqual(labels.tolist(), [0, 1, 1, 2, 2, 2])
            self.assertEqual(num_classes, 20)

            ds = CIFARHistogramBags(root=root, dataset="CIFAR100", split="train", num_bags=2, bag_size=4, bag_size_std=0, seed=1)
            item = ds[0]
            self.assertEqual(tuple(item["instances"].shape), (4, 3, 32, 32))
            self.assertEqual(int(item["class_counts"].sum().item()), 4)
            self.assertEqual(item["class_counts"].tolist(), torch.bincount(item["labels"], minlength=20).tolist())
            self.assertTrue(torch.allclose(item["class_proportions"], item["class_counts"].float() / 4))
            batch = collate_cifar_bags([ds[0], ds[1]])
            self.assertEqual(tuple(batch["instances"].shape), (2, 4, 3, 32, 32))
            self.assertTrue(batch["mask"].all())

            aug = CIFARHistogramBags(
                root=root,
                dataset="CIFAR100",
                split="train",
                num_bags=1,
                bag_size=4,
                bag_size_std=0,
                seed=1,
                augment=True,
            )
            aug_item = aug[0]
            self.assertEqual(tuple(aug_item["instances"].shape), (4, 3, 32, 32))
            self.assertEqual(aug_item["class_counts"].tolist(), item["class_counts"].tolist())

            signed = CIFARSignedBags(
                root=root,
                dataset="CIFAR100",
                split="train",
                num_bags=1,
                bag_size=4,
                bag_size_std=0,
                target_label=2,
                seed=1,
            )
            signed_item = signed[0]
            self.assertTrue(torch.equal(signed_item["signed_instance_labels"], signed_item["signs"] * signed_item["instance_labels"]))
            self.assertEqual(signed_item["signed_count"].item(), signed_item["signed_instance_labels"].sum().item())
            signed_batch = collate_mnist_bags([signed_item])
            self.assertEqual(tuple(signed_batch["instances"].shape), (1, 4, 3, 32, 32))
            self.assertTrue(signed_batch["mask"].all())

            multi_signed = CIFARSignedBags(
                root=root,
                dataset="CIFAR100",
                split="train",
                num_bags=1,
                bag_size=4,
                bag_size_std=0,
                target_label=[1, 2],
                seed=1,
            )
            multi_item = multi_signed[0]
            expected = torch.isin(multi_item["labels"], torch.tensor([1, 2])).long()
            self.assertTrue(torch.equal(multi_item["instance_labels"], expected))


if __name__ == "__main__":
    unittest.main()
