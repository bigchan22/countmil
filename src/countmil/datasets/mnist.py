"""MNIST-family IDX readers and on-the-fly bag samplers."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
from torch.utils.data import Dataset


Split = Literal["train", "test"]


def _idx_paths(root: str | Path, dataset: str, split: Split) -> tuple[Path, Path]:
    raw = Path(root) / dataset / "raw"
    prefix = "train" if split == "train" else "t10k"
    return raw / f"{prefix}-images-idx3-ubyte", raw / f"{prefix}-labels-idx1-ubyte"


def _read_bytes(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            return f.read()
    raise FileNotFoundError(f"missing IDX file: {path} or {gz_path}")


def load_idx_images(path: str | Path) -> torch.Tensor:
    """Load IDX image file as float tensor `(N,1,H,W)` in `[0,1]`."""

    path = Path(path)
    data = _read_bytes(path)
    if len(data) < 16:
        raise ValueError(f"invalid image IDX file, too short: {path}")
    magic, n_items, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"invalid image IDX magic for {path}: {magic}")
    expected = 16 + n_items * rows * cols
    if len(data) != expected:
        raise ValueError(f"invalid image IDX size for {path}: got {len(data)}, expected {expected}")
    pixels = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return pixels.reshape(n_items, 1, rows, cols).float().div_(255.0)


def load_idx_labels(path: str | Path) -> torch.Tensor:
    """Load IDX label file as long tensor `(N,)`."""

    path = Path(path)
    data = _read_bytes(path)
    if len(data) < 8:
        raise ValueError(f"invalid label IDX file, too short: {path}")
    magic, n_items = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"invalid label IDX magic for {path}: {magic}")
    expected = 8 + n_items
    if len(data) != expected:
        raise ValueError(f"invalid label IDX size for {path}: got {len(data)}, expected {expected}")
    return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).long()


def load_mnist_family(root: str | Path = "data", dataset: str = "MNIST", split: Split = "train") -> tuple[torch.Tensor, torch.Tensor]:
    images_path, labels_path = _idx_paths(root, dataset, split)
    images = load_idx_images(images_path)
    labels = load_idx_labels(labels_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")
    return images, labels


class _BaseMNISTBags(Dataset):
    def __init__(
        self,
        root: str | Path = "data",
        dataset: str = "MNIST",
        split: Split = "train",
        num_bags: int = 1000,
        bag_size: int = 10,
        bag_size_std: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        if bag_size < 1:
            raise ValueError("bag_size must be positive")
        self.images, self.labels = load_mnist_family(root=root, dataset=dataset, split=split)
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

    def _sample_indices(self, idx: int) -> torch.Tensor:
        gen = self._generator(idx)
        bag_size = self._sample_bag_size(gen)
        return torch.randint(0, self.images.shape[0], (bag_size,), generator=gen)


class MNISTBags(_BaseMNISTBags):
    """Binary MIL and exact-count bags for a target digit.

    Returns a dict with:
      - `instances`: `(bag_size,1,28,28)`
      - `instance_labels`: binary hidden target-digit labels
      - `count`: exact number of target-digit instances
      - `bag_label`: binary contains-target label
    """

    def __init__(self, *args, target_digit: int = 9, balanced_binary: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_digit = int(target_digit)
        self.balanced_binary = bool(balanced_binary)
        self.pos_idx = torch.nonzero(self.labels == self.target_digit, as_tuple=False).flatten()
        self.neg_idx = torch.nonzero(self.labels != self.target_digit, as_tuple=False).flatten()
        if self.pos_idx.numel() == 0 or self.neg_idx.numel() == 0:
            raise ValueError("target digit split is empty")

    def _sample_balanced_indices(self, idx: int) -> torch.Tensor:
        gen = self._generator(idx)
        bag_size = self._sample_bag_size(gen)
        positive_bag = idx % 2 == 0
        if positive_bag:
            n_pos = int(torch.randint(1, bag_size + 1, (1,), generator=gen).item())
            n_neg = bag_size - n_pos
        else:
            n_pos = 0
            n_neg = bag_size
        pos = self.pos_idx[torch.randint(0, self.pos_idx.numel(), (n_pos,), generator=gen)]
        neg = self.neg_idx[torch.randint(0, self.neg_idx.numel(), (n_neg,), generator=gen)]
        idxs = torch.cat([pos, neg])
        return idxs[torch.randperm(idxs.numel(), generator=gen)]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indices = self._sample_balanced_indices(idx) if self.balanced_binary else self._sample_indices(idx)
        labels = (self.labels[indices] == self.target_digit).long()
        count = labels.sum()
        return {
            "instances": self.images[indices],
            "digits": self.labels[indices],
            "instance_labels": labels,
            "count": count.long(),
            "bag_label": (count > 0).long(),
        }


class MNISTDigitSumBags(_BaseMNISTBags):
    """Ordinal finite-support sum bags with hidden digit labels."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indices = self._sample_indices(idx)
        digits = self.labels[indices]
        return {
            "instances": self.images[indices],
            "digits": digits,
            "sum": digits.sum().long(),
        }


class MNISTDigitHistogramBags(_BaseMNISTBags):
    """Multiclass LLP bags with observed digit-count histograms."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indices = self._sample_indices(idx)
        digits = self.labels[indices]
        counts = torch.bincount(digits, minlength=10).long()
        return {
            "instances": self.images[indices],
            "digits": digits,
            "digit_counts": counts,
            "digit_proportions": counts.float() / digits.numel(),
        }


class SignedMNISTBags(_BaseMNISTBags):
    """Signed Count-MIL bags for a target digit.

    Each target-digit instance contributes `sign_i`; non-target instances
    contribute zero. Signs are sampled independently unless
    `cancellation_heavy=True`, in which case signs alternate to make zero sums
    more common when target instances appear.
    """

    def __init__(
        self,
        *args,
        target_digit: int | Sequence[int] = 9,
        cancellation_heavy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(target_digit, int):
            target_digits = [target_digit]
        else:
            target_digits = [int(x) for x in target_digit]
        if not target_digits:
            raise ValueError("target_digit must contain at least one digit")
        self.target_digits = torch.tensor(sorted(set(target_digits)), dtype=torch.long)
        self.target_digit = int(self.target_digits[0].item()) if self.target_digits.numel() == 1 else self.target_digits.tolist()
        self.cancellation_heavy = bool(cancellation_heavy)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = self._generator(idx)
        indices = self._sample_indices(idx)
        is_target = torch.isin(self.labels[indices], self.target_digits).long()
        if self.cancellation_heavy:
            signs = torch.ones(indices.numel(), dtype=torch.long)
            signs[1::2] = -1
            signs = signs[torch.randperm(indices.numel(), generator=gen)]
        else:
            signs = torch.randint(0, 2, (indices.numel(),), generator=gen).mul(2).sub(1).long()
        signed_labels = signs * is_target
        return {
            "instances": self.images[indices],
            "digits": self.labels[indices],
            "instance_labels": is_target,
            "signs": signs,
            "signed_instance_labels": signed_labels,
            "signed_count": signed_labels.sum().long(),
        }


def collate_mnist_bags(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length MNIST bags."""

    max_len = max(item["instances"].shape[0] for item in batch)
    batch_size = len(batch)
    instances = batch[0]["instances"].new_zeros(batch_size, max_len, *batch[0]["instances"].shape[1:])
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    out: dict[str, torch.Tensor] = {"instances": instances, "mask": mask}

    keys_1d = ["digits", "instance_labels", "signs", "signed_instance_labels"]
    for key in keys_1d:
        if key in batch[0]:
            out[key] = torch.zeros(batch_size, max_len, dtype=batch[0][key].dtype)

    scalar_keys = ["count", "bag_label", "sum", "signed_count"]
    for key in scalar_keys:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])

    vector_keys = ["digit_counts", "digit_proportions"]
    for key in vector_keys:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])

    for i, item in enumerate(batch):
        n = item["instances"].shape[0]
        instances[i, :n] = item["instances"]
        mask[i, :n] = True
        for key in keys_1d:
            if key in item:
                out[key][i, :n] = item[key]
    return out
