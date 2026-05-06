"""Variable-size ordinal-sum bag datasets for A1-style experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .mnist import load_mnist_family


TensorTransform = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


class TensorOrdinalSumBags(Dataset):
    """Sample variable-size bags from image tensors with scalar ordinal sums.

    Hidden labels are returned for evaluation only. Training scripts should use
    the observed `sum` field as the weak label.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        num_bags: int = 1000,
        bag_size_mean: float = 10.0,
        bag_size_std: Optional[float] = 2.0,
        bag_size_min: int = 5,
        bag_size_max: int = 15,
        per_class_cap: Optional[int] = None,
        noise_sigma: float = 0.0,
        seed: int = 0,
        transform: TensorTransform | None = None,
    ) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same first dimension")
        if bag_size_min < 1 or bag_size_max < bag_size_min:
            raise ValueError("invalid bag-size bounds")
        self.images = images
        self.labels = labels.long()
        self.num_bags = int(num_bags)
        self.bag_size_mean = float(bag_size_mean)
        self.bag_size_std = None if bag_size_std is None else float(bag_size_std)
        self.bag_size_min = int(bag_size_min)
        self.bag_size_max = int(bag_size_max)
        self.per_class_cap = None if per_class_cap is None else int(per_class_cap)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.transform = transform
        self.num_classes = int(self.labels.max().item()) + 1
        self.class_indices = [torch.nonzero(self.labels == k, as_tuple=False).flatten() for k in range(self.num_classes)]
        if any(idx.numel() == 0 for idx in self.class_indices):
            raise ValueError("all ordinal classes must be present")

    def __len__(self) -> int:
        return self.num_bags

    def _generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))
        return gen

    def _sample_bag_size(self, gen: torch.Generator) -> int:
        if self.bag_size_std is None or self.bag_size_std <= 0:
            sample = int(round(self.bag_size_mean))
        else:
            draw = torch.normal(
                mean=torch.tensor(self.bag_size_mean),
                std=torch.tensor(self.bag_size_std),
                generator=gen,
            )
            sample = int(round(float(draw.item())))
        return max(self.bag_size_min, min(self.bag_size_max, sample))

    def _sample_indices(self, bag_size: int, gen: torch.Generator) -> torch.Tensor:
        if self.per_class_cap is None:
            return torch.randint(0, self.images.shape[0], (bag_size,), generator=gen)

        chosen = []
        class_counts = torch.zeros(self.num_classes, dtype=torch.long)
        attempts = 0
        max_attempts = max(100, bag_size * 100)
        while len(chosen) < bag_size and attempts < max_attempts:
            attempts += 1
            cls = int(torch.randint(0, self.num_classes, (1,), generator=gen).item())
            if class_counts[cls] >= self.per_class_cap:
                continue
            idx_pool = self.class_indices[cls]
            chosen.append(idx_pool[torch.randint(0, idx_pool.numel(), (1,), generator=gen)].item())
            class_counts[cls] += 1
        if len(chosen) < bag_size:
            extra = torch.randint(0, self.images.shape[0], (bag_size - len(chosen),), generator=gen).tolist()
            chosen.extend(extra)
        return torch.tensor(chosen, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = self._generator(idx)
        bag_size = self._sample_bag_size(gen)
        indices = self._sample_indices(bag_size, gen)
        labels = self.labels[indices]
        instances = self.images[indices]
        if self.transform is not None:
            instances = self.transform(instances, gen)
        if self.noise_sigma > 0:
            instances = (instances + torch.randn(instances.shape, generator=gen) * self.noise_sigma).clamp(0.0, 1.0)
        return {
            "instances": instances,
            "labels": labels,
            "sum": labels.sum().long(),
        }


class MNISTOrdinalSumBags(TensorOrdinalSumBags):
    """MNIST ordinal-sum bags with colleague A1-style bag-size controls."""

    def __init__(self, root: str | Path = "data", dataset: str = "MNIST", split: str = "train", **kwargs) -> None:
        images, labels = load_mnist_family(root=root, dataset=dataset, split=split)  # type: ignore[arg-type]
        super().__init__(images, labels, **kwargs)


def _ultramnist_transform(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Synthetic UltraMNIST-style stand-in: 28x28 gray digit on 64x64 RGB canvas."""

    del gen
    resized = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
    return resized.repeat(1, 3, 1, 1)


class UltraMNISTOrdinalSumBags(MNISTOrdinalSumBags):
    """Synthetic UltraMNIST-style bags built from MNIST digits.

    This mirrors the colleague reference's license-safe stand-in: it is not the
    external Kaggle UltraMNIST dataset.
    """

    def __init__(
        self,
        root: str | Path = "data",
        dataset: str = "MNIST",
        split: str = "train",
        bag_size_min: int = 3,
        bag_size_max: int = 5,
        **kwargs,
    ) -> None:
        kwargs.setdefault("bag_size_mean", (bag_size_min + bag_size_max) / 2.0)
        kwargs.setdefault("bag_size_std", None)
        super().__init__(
            root=root,
            dataset=dataset,
            split=split,
            bag_size_min=bag_size_min,
            bag_size_max=bag_size_max,
            transform=_ultramnist_transform,
            **kwargs,
        )


def _load_svhn_mat(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import scipy.io
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("loading local SVHN .mat files requires scipy") from exc
    obj = scipy.io.loadmat(path)
    data = torch.as_tensor(obj["X"], dtype=torch.uint8).permute(3, 2, 0, 1).float().div_(255.0)
    labels = torch.as_tensor(obj["y"].reshape(-1), dtype=torch.long).remainder(10)
    return data, labels


def load_svhn_family(
    root: str | Path = "data",
    split: str = "train",
    download: bool = False,
    train_split_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SVHN via torchvision if available.

    Download is opt-in to preserve the repository's no-download-by-default
    behavior on remote servers.
    """

    root = Path(root)
    local_name = "train_32x32.mat" if split == "train" else "test_32x32.mat"
    local_path = root / local_name
    if local_path.exists():
        return _load_svhn_mat(local_path)
    if split != "train" and train_split_fallback and (root / "train_32x32.mat").exists():
        images, labels = _load_svhn_mat(root / "train_32x32.mat")
        # Deterministic holdout from the training archive. This is a fallback
        # for machines where the official test archive is unavailable.
        cutoff = max(1, int(round(images.shape[0] * 0.85)))
        return images[cutoff:], labels[cutoff:]
    if split == "train" and train_split_fallback and (root / "train_32x32.mat").exists():
        images, labels = _load_svhn_mat(root / "train_32x32.mat")
        cutoff = max(1, int(round(images.shape[0] * 0.85)))
        return images[:cutoff], labels[:cutoff]

    try:
        from torchvision.datasets import SVHN
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("SVHN loading requires torchvision") from exc
    tv_split = "train" if split == "train" else "test"
    ds = SVHN(root=str(root), split=tv_split, download=download)
    images = torch.as_tensor(ds.data, dtype=torch.uint8).float().div_(255.0)
    labels = torch.as_tensor(ds.labels, dtype=torch.long)
    labels = labels.remainder(10)
    return images, labels


def _svhn_train_transform(images: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    out = torch.empty_like(images)
    for i in range(images.shape[0]):
        top = int(torch.randint(0, 9, (1,), generator=gen).item())
        left = int(torch.randint(0, 9, (1,), generator=gen).item())
        crop = padded[i, :, top : top + 32, left : left + 32]
        brightness = 0.8 + 0.4 * torch.rand((), generator=gen).item()
        contrast = 0.8 + 0.4 * torch.rand((), generator=gen).item()
        mean = crop.mean(dim=(-2, -1), keepdim=True)
        crop = ((crop - mean) * contrast + mean) * brightness
        out[i] = crop.clamp(0.0, 1.0)
    return out


class SVHNOrdinalSumBags(TensorOrdinalSumBags):
    """SVHN ordinal-sum bags for A1-style cross-domain checks."""

    def __init__(
        self,
        root: str | Path = "data",
        split: str = "train",
        download: bool = False,
        train_split_fallback: bool = False,
        augment: bool = True,
        **kwargs,
    ) -> None:
        images, labels = load_svhn_family(
            root=root,
            split=split,
            download=download,
            train_split_fallback=train_split_fallback,
        )
        transform = _svhn_train_transform if augment and split == "train" else None
        super().__init__(images, labels, transform=transform, **kwargs)


def collate_ordinal_sum_bags(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
        "sum": torch.stack([item["sum"] for item in batch]),
    }
