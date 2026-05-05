import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from countmil.datasets import CIFARHistogramBags, collate_cifar_bags, load_cifar_family


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
            batch = collate_cifar_bags([ds[0], ds[1]])
            self.assertEqual(tuple(batch["instances"].shape), (2, 4, 3, 32, 32))
            self.assertTrue(batch["mask"].all())


if __name__ == "__main__":
    unittest.main()
