"""Bag datasets for B1 verification.

MNISTBagDataset: bags of MNIST digits with count-of-positives label.
LLPBagDataset:   bags of UCI tabular instances (Adult, Magic) with count label.

See spec §6 for math/conventions.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.datasets import SVHN as _SVHN
from torchvision import transforms

_MNIST_ROOT = Path.home() / ".cache" / "torch" / "datasets"
_LLP_CACHE_DIR = Path.home() / ".cache" / "pca" / "llp"


class MNISTBagDataset(Dataset):
    """MNIST-MIL bag dataset with count supervision.

    Returns per item: (images, bag_count, gt_labels)
        images:    (N, 1, 28, 28) float in [0, 1]
        bag_count: int — number of positive_digit instances in bag
        gt_labels: (N,) long — 1 iff label == positive_digit (eval only)
    """

    def __init__(
        self,
        bag_size: int,
        num_bags: int,
        positive_digit: int = 9,
        train: bool = True,
        seed: int = 0,
    ):
        self.bag_size = bag_size
        self.num_bags = num_bags
        self.positive_digit = positive_digit

        # Load + normalize underlying MNIST
        tfm = transforms.Compose([transforms.ToTensor()])  # → float [0, 1], shape (1, 28, 28)
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        # Materialize into tensors for fast indexing.
        self._images = torch.stack([mnist[i][0] for i in range(len(mnist))])     # (M, 1, 28, 28)
        self._labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])    # (M,)

        # Pre-sample bag indices deterministically from a seeded RNG.
        rng = np.random.default_rng(seed)
        M = self._images.shape[0]
        self._bag_indices = np.stack([
            rng.choice(M, size=bag_size, replace=False) for _ in range(num_bags)
        ])  # (num_bags, bag_size)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        idxs = self._bag_indices[idx]
        images = self._images[idxs]                                    # (N, 1, 28, 28)
        labels = self._labels[idxs]                                    # (N,)
        gt = (labels == self.positive_digit).long()                    # (N,)
        bag_count = int(gt.sum().item())
        return images, bag_count, gt


def _build_adult_cache(cache_path: Path) -> dict:
    """Fetch Adult, one-hot encode, standardize, split. Cache as torch dict."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_openml('adult', version=2, as_frame=True)
    X_df = raw.data.copy()
    y_raw = raw.target

    # Drop rows with '?' markers in any column
    mask = ~(X_df.astype(str) == '?').any(axis=1)
    X_df = X_df[mask]
    y_raw = y_raw[mask]

    y = (y_raw == '>50K').astype(int).to_numpy()
    cat_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    X_cat = pd.get_dummies(X_df[cat_cols], dummy_na=False).astype(float)
    X_num = X_df[num_cols].astype(float)
    X_full = pd.concat([X_num, X_cat], axis=1).to_numpy().astype('float32')

    # Train/test split: stratified 80/20, fixed seed
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.2, random_state=0, stratify=y,
    )

    # Standardize numerical columns based on train statistics
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    blob = {
        'X_train': torch.from_numpy(X_tr).float(),
        'y_train': torch.from_numpy(y_tr).long(),
        'X_test': torch.from_numpy(X_te).float(),
        'y_test': torch.from_numpy(y_te).long(),
    }
    torch.save(blob, cache_path)
    return blob


class LLPBagDataset(Dataset):
    """LLP bag dataset for tabular UCI datasets (Adult).

    Returns per item: (features, bag_count, gt_labels)
        features:  (N, D) float
        bag_count: int
        gt_labels: (N,) long
    """

    def __init__(
        self,
        dataset_name: str,
        bag_size: int,
        num_bags: int,
        train: bool = True,
        seed: int = 0,
    ):
        assert dataset_name == 'adult', f"only 'adult' supported (got {dataset_name})"
        self.bag_size = bag_size
        self.num_bags = num_bags

        cache_path = _LLP_CACHE_DIR / f"{dataset_name}.pt"
        if cache_path.exists():
            blob = torch.load(cache_path, weights_only=True)
        else:
            blob = _build_adult_cache(cache_path)

        if train:
            self._features = blob['X_train']
            self._labels = blob['y_train']
        else:
            self._features = blob['X_test']
            self._labels = blob['y_test']

        rng = np.random.default_rng(seed)
        M = self._features.shape[0]
        self._bag_indices = np.stack([
            rng.choice(M, size=bag_size, replace=False) for _ in range(num_bags)
        ])

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        idxs = self._bag_indices[idx]
        feats = self._features[idxs]
        gt = self._labels[idxs]
        bag_count = int(gt.sum().item())
        return feats, bag_count, gt


class MNISTSumBagDataset(Dataset):
    """MNIST bag dataset where bag label = sum of digit values.

    Atom support: S=10 (digits 0..9). Bag sum range: [0, 9*N_max].

    Args:
        bag_size_mean, bag_size_std: truncated-normal parameters for N.
        bag_size_min, bag_size_max:  truncation bounds (inclusive).
        num_bags:      number of bags.
        per_class_cap: if set, subsample MNIST train pool to this many per class
                       (deterministic via seed). None → full pool.
        noise_sigma:   if > 0, add N(0, σ) noise to images at access time
                       (training and eval identically — task hardening).
        train:         True → MNIST train split, False → test split.
        seed:          determinism over class subsampling, bag indices, bag sizes.
    """

    def __init__(
        self,
        bag_size_mean: float = 10.0,
        bag_size_std:  float = 2.0,
        bag_size_min:  int = 5,
        bag_size_max:  int = 15,
        num_bags:      int = 1500,
        per_class_cap: int | None = 100,
        noise_sigma:   float = 0.0,
        train:         bool = True,
        seed:          int = 0,
    ):
        self.bag_size_mean = bag_size_mean
        self.bag_size_std = bag_size_std
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max
        self.num_bags = num_bags
        self.noise_sigma = noise_sigma

        tfm = transforms.Compose([transforms.ToTensor()])
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        all_images = torch.stack([mnist[i][0] for i in range(len(mnist))])
        all_labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        # Pre-sample bag sizes + indices.
        sizes = []
        bag_indices = []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(round(rng.normal(bag_size_mean, bag_size_std)))
            n = max(bag_size_min, min(bag_size_max, n))
            sizes.append(n)
            bag_indices.append(rng.choice(M, size=n, replace=False))
        self._bag_sizes = sizes
        self._bag_indices = bag_indices
        self._noise_rng = np.random.default_rng(seed + 10_000)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum, labels.long(), mask


# Pinned signed-digit mapping: 0 → 0, odd-nonzero → -1, even-nonzero → +1.
_SIGNED_DIGIT_MAP = torch.tensor([0, -1, +1, -1, +1, -1, +1, -1, +1, -1])


class MNISTSignedSumBagDataset(MNISTSumBagDataset):
    """MNIST signed-sum bag dataset.

    Per-instance ground truth z_i ∈ {-1, 0, +1} via _SIGNED_DIGIT_MAP.
    Bag sum range: [-N, +N], stored shifted by +N → bag_sum_shifted ∈ [0, 2N].
    Caller (metrics.py) un-shifts for accuracy/MAE.
    """

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        signed = _SIGNED_DIGIT_MAP[labels]                  # (n,) values in {-1,0,1}
        bag_sum_shifted = int(signed.sum().item()) + n      # in [0, 2n]
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum_shifted, signed.long(), mask


class SVHNSumBagDataset(Dataset):
    """SVHN bag dataset (RGB 32x32) with bag label = sum of digit values.

    Atom support S = 10 (digits 0..9, but SVHN uses {1..10} natively → mapped
    to {0..9}). Bag sum range: [0, 9 * N_max].

    `augment=True` (default) applies RandomCrop(32, padding=4) + ColorJitter
    to train-side images per __getitem__ call to mitigate ResNet18-from-scratch
    overfitting on a sparse 1500-bag training set. Test-side images are never
    augmented regardless of this flag.
    """

    def __init__(
        self,
        bag_size_mean: float = 10.0,
        bag_size_std:  float = 2.0,
        bag_size_min:  int = 5,
        bag_size_max:  int = 15,
        num_bags:      int = 1500,
        per_class_cap: int | None = None,
        noise_sigma:   float = 0.0,
        train:         bool = True,
        seed:          int = 0,
        augment:       bool = True,
    ):
        self.bag_size_mean = bag_size_mean
        self.bag_size_std = bag_size_std
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max
        self.num_bags = num_bags
        self.noise_sigma = noise_sigma

        split = 'train' if train else 'test'
        tfm = transforms.Compose([transforms.ToTensor()])
        svhn = _SVHN(root=str(_MNIST_ROOT), split=split, download=True, transform=tfm)
        all_images = torch.stack([svhn[i][0] for i in range(len(svhn))])
        all_labels = torch.tensor([svhn[i][1] for i in range(len(svhn))])
        # SVHN labels: 1..10 in original; torchvision returns 0..9 already
        # (label 10 in raw → 0 here). Verified with torchvision >= 0.17.

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        sizes = []
        bag_indices = []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(round(rng.normal(bag_size_mean, bag_size_std)))
            n = max(bag_size_min, min(bag_size_max, n))
            sizes.append(n)
            bag_indices.append(rng.choice(M, size=n, replace=False))
        self._bag_sizes = sizes
        self._bag_indices = bag_indices
        self._noise_rng = np.random.default_rng(seed + 10_000)

        if train and augment:
            from torchvision.transforms import v2 as _v2
            self._augment = _v2.Compose([
                _v2.RandomCrop(32, padding=4),
                _v2.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self._augment = None

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        images = self._images[idxs]
        labels = self._labels[idxs]
        if self._augment is not None:
            images = self._augment(images)
        if self.noise_sigma > 0:
            noise = torch.from_numpy(self._noise_rng.normal(
                0.0, self.noise_sigma, size=images.shape).astype('float32'))
            images = images + noise
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return images, bag_sum, labels.long(), mask


class UltraMNISTBagDataset(Dataset):
    """Synthetic stand-in for Kaggle UltraMNIST.

    Composes 3-5 MNIST digits at non-overlapping positions on a 4000x4000 RGB
    canvas, then crops 64x64 RGB patches per digit (positions known by
    construction). Per-instance feature is the patch crop. Bag sum = sum of
    digit values. Atom support S=10, K=9.

    Note (binding): this is a stand-in to keep Day 6 executable without the
    Kaggle license blocker. The contract (returns (patches, bag_sum, gt, mask))
    matches what a real UltraMNIST loader would produce; swap is one-file.
    """

    def __init__(
        self,
        num_bags:      int = 600,
        bag_size_min:  int = 3,
        bag_size_max:  int = 5,
        per_class_cap: int | None = None,
        train:         bool = True,
        seed:          int = 0,
    ):
        self.num_bags = num_bags
        self.bag_size_min = bag_size_min
        self.bag_size_max = bag_size_max

        tfm = transforms.Compose([transforms.ToTensor()])
        mnist = MNIST(root=str(_MNIST_ROOT), train=train, download=True, transform=tfm)
        all_images = torch.stack([mnist[i][0] for i in range(len(mnist))])  # (M, 1, 28, 28)
        all_labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])

        rng = np.random.default_rng(seed)
        if per_class_cap is not None:
            keep_idx = []
            for k in range(10):
                cls_idx = (all_labels == k).nonzero(as_tuple=True)[0].numpy()
                if len(cls_idx) > per_class_cap:
                    chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                else:
                    chosen = cls_idx
                keep_idx.append(chosen)
            keep_idx = np.concatenate(keep_idx)
            self._images = all_images[keep_idx]
            self._labels = all_labels[keep_idx]
        else:
            self._images = all_images
            self._labels = all_labels

        # Pre-sample bags: pick N_b digits and 64x64 patch positions.
        sizes, indices, positions = [], [], []
        M = self._images.shape[0]
        for _ in range(num_bags):
            n = int(rng.integers(bag_size_min, bag_size_max + 1))
            idxs = rng.choice(M, size=n, replace=False)
            pos = self._sample_positions(rng, n)
            sizes.append(n)
            indices.append(idxs)
            positions.append(pos)
        self._bag_sizes = sizes
        self._bag_indices = indices
        self._bag_positions = positions

    @staticmethod
    def _sample_positions(rng, n: int, canvas: int = 4000, patch: int = 64,
                          margin: int = 32) -> np.ndarray:
        """Non-overlapping (greedy) random patch top-left coordinates."""
        out = []
        attempts = 0
        while len(out) < n and attempts < 5000:
            x = int(rng.integers(margin, canvas - patch - margin))
            y = int(rng.integers(margin, canvas - patch - margin))
            ok = all(abs(x - ox) > patch or abs(y - oy) > patch
                     for ox, oy in out)
            if ok:
                out.append((x, y))
            attempts += 1
        # Fallback: in the unlikely sparse case, just append remaining as random.
        while len(out) < n:
            x = int(rng.integers(margin, canvas - patch - margin))
            y = int(rng.integers(margin, canvas - patch - margin))
            out.append((x, y))
        return np.array(out, dtype=np.int32)

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int):
        n = self._bag_sizes[idx]
        idxs = self._bag_indices[idx]
        positions = self._bag_positions[idx]
        # Compose 28x28 MNIST digits onto 64x64 RGB patches centered in patch.
        patches = torch.zeros((n, 3, 64, 64))
        offset = (64 - 28) // 2
        for i, didx in enumerate(idxs):
            digit = self._images[didx]                       # (1, 28, 28) ∈ [0,1]
            digit_rgb = digit.repeat(3, 1, 1)                 # (3, 28, 28)
            patches[i, :, offset:offset + 28, offset:offset + 28] = digit_rgb
        labels = self._labels[idxs]
        bag_sum = int(labels.sum().item())
        mask = torch.ones(n, dtype=torch.bool)
        return patches, bag_sum, labels.long(), mask


def variable_n_collate_fn(batch):
    """Collate fn that pads variable-N items to batch's N_max with masking.

    Each item must be a 4-tuple (features, bag_sum, gt_per_instance, mask) where
    features.shape[0] == gt_per_instance.shape[0] == mask.shape[0] == N_item.
    Inactive padding atoms get features = 0 (not used downstream — mask flags
    them) and mask = False; gt is padded with 0 (caller must respect mask).
    """
    n_max = max(item[0].shape[0] for item in batch)
    B = len(batch)
    feature_shape = batch[0][0].shape[1:]
    padded_features = batch[0][0].new_zeros((B, n_max, *feature_shape))
    padded_gt = torch.zeros((B, n_max), dtype=torch.long)
    padded_mask = torch.zeros((B, n_max), dtype=torch.bool)
    bag_sums = torch.zeros((B,), dtype=torch.long)
    for b, (feats, bag_sum, gt, mask) in enumerate(batch):
        n_b = feats.shape[0]
        padded_features[b, :n_b] = feats
        padded_gt[b, :n_b] = gt
        padded_mask[b, :n_b] = mask
        bag_sums[b] = bag_sum
    return padded_features, bag_sums, padded_gt, padded_mask
