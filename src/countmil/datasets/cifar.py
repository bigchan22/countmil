"""CIFAR archive readers and on-the-fly histogram bag samplers."""

from __future__ import annotations

from pathlib import Path
import pickle
import tarfile
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset


Split = Literal["train", "test"]


def _find_archive(root: str | Path, dataset: str) -> Path:
    root = Path(root)
    if dataset == "CIFAR100":
        candidates = [root / "cifar-100-python.tar.gz", root / "cifar-100-python"]
    elif dataset == "CIFAR10":
        candidates = [root / "cifar-10-python.tar.gz", root / "cifar-10-batches-py"]
    else:
        raise ValueError("dataset must be CIFAR10 or CIFAR100")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing {dataset} archive/extracted directory under {root}")


def _read_pickle_from_tar(archive: Path, member_suffix: str) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(member_suffix)), None)
        if member is None:
            raise FileNotFoundError(f"missing {member_suffix} in {archive}")
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"could not extract {member.name} from {archive}")
        return pickle.load(f, encoding="latin1")


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar_family(
    root: str | Path = "data",
    dataset: str = "CIFAR100",
    split: Split = "train",
    label_level: str = "coarse",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load CIFAR images as `(N,3,32,32)` float tensor and labels.

    For CIFAR-100, `label_level` can be `coarse` (20 classes) or `fine` (100).
    For CIFAR-10, labels are always the 10 ordinary classes.
    """

    dataset = dataset.upper().replace("-", "")
    if dataset == "CIFAR100":
        archive = _find_archive(root, "CIFAR100")
        filename = "train" if split == "train" else "test"
        obj = _read_pickle_from_tar(archive, f"/{filename}") if archive.suffixes[-2:] == [".tar", ".gz"] else _read_pickle(archive / filename)
        labels_key = "coarse_labels" if label_level == "coarse" else "fine_labels"
        num_classes = 20 if label_level == "coarse" else 100
    elif dataset == "CIFAR10":
        archive = _find_archive(root, "CIFAR10")
        if archive.suffixes[-2:] == [".tar", ".gz"]:
            if split == "train":
                batches = [_read_pickle_from_tar(archive, f"/data_batch_{i}") for i in range(1, 6)]
                data = torch.cat([torch.as_tensor(b["data"], dtype=torch.uint8) for b in batches], dim=0)
                labels = torch.tensor([y for b in batches for y in b["labels"]], dtype=torch.long)
                return data.reshape(-1, 3, 32, 32).float().div_(255.0), labels, 10
            obj = _read_pickle_from_tar(archive, "/test_batch")
        else:
            if split == "train":
                batches = [_read_pickle(archive / f"data_batch_{i}") for i in range(1, 6)]
                data = torch.cat([torch.as_tensor(b["data"], dtype=torch.uint8) for b in batches], dim=0)
                labels = torch.tensor([y for b in batches for y in b["labels"]], dtype=torch.long)
                return data.reshape(-1, 3, 32, 32).float().div_(255.0), labels, 10
            obj = _read_pickle(archive / "test_batch")
        labels_key = "labels"
        num_classes = 10
    else:
        raise ValueError("dataset must be CIFAR10 or CIFAR100")

    data = torch.as_tensor(obj["data"], dtype=torch.uint8)
    labels = torch.tensor(obj[labels_key], dtype=torch.long)
    images = data.reshape(-1, 3, 32, 32).float().div_(255.0)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")
    return images, labels, num_classes


class CIFARHistogramBags(Dataset):
    """CIFAR bags with observed class-count histograms."""

    def __init__(
        self,
        root: str | Path = "data",
        dataset: str = "CIFAR100",
        split: Split = "train",
        label_level: str = "coarse",
        num_bags: int = 1000,
        bag_size: int = 50,
        bag_size_std: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        if bag_size < 1:
            raise ValueError("bag_size must be positive")
        self.images, self.labels, self.num_classes = load_cifar_family(root, dataset, split, label_level)
        self.num_bags = int(num_bags)
        self.bag_size = int(bag_size)
        self.bag_size_std = bag_size_std
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_bags

    def _generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))
        return gen

    def _sample_bag_size(self, gen: torch.Generator) -> int:
        if self.bag_size_std is None or self.bag_size_std <= 0:
            return self.bag_size
        sample = torch.normal(
            mean=torch.tensor(float(self.bag_size)),
            std=torch.tensor(float(self.bag_size_std)),
            generator=gen,
        )
        return max(1, int(round(float(sample.item()))))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = self._generator(idx)
        bag_size = self._sample_bag_size(gen)
        indices = torch.randint(0, self.images.shape[0], (bag_size,), generator=gen)
        labels = self.labels[indices]
        counts = torch.bincount(labels, minlength=self.num_classes).long()
        return {
            "instances": self.images[indices],
            "labels": labels,
            "class_counts": counts,
            "class_proportions": counts.float() / labels.numel(),
        }


def collate_cifar_bags(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["instances"].shape[0] for item in batch)
    batch_size = len(batch)
    instances = batch[0]["instances"].new_zeros(batch_size, max_len, *batch[0]["instances"].shape[1:])
    labels = torch.zeros(batch_size, max_len, dtype=batch[0]["labels"].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["instances"].shape[0]
        instances[i, :n] = item["instances"]
        labels[i, :n] = item["labels"]
        mask[i, :n] = True
    return {
        "instances": instances,
        "labels": labels,
        "mask": mask,
        "class_counts": torch.stack([item["class_counts"] for item in batch]),
        "class_proportions": torch.stack([item["class_proportions"] for item in batch]),
    }
