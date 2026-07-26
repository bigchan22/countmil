"""Dataset builders for CountMIL experiments."""

from .atomic_sum import (
    MNISTOrdinalSumBags,
    SVHNOrdinalSumBags,
    SVHNSignedBags,
    TensorOrdinalSumBags,
    TensorSignedBags,
    UltraMNISTOrdinalSumBags,
    collate_ordinal_sum_bags,
    load_svhn_family,
)
from .cifar import CIFARHistogramBags, CIFARSignedBags, collate_cifar_bags, load_cifar_family
from .mnist import (
    MNISTBags,
    MNISTDigitHistogramBags,
    MNISTDigitSumBags,
    SignedMNISTBags,
    collate_mnist_bags,
    load_idx_images,
    load_idx_labels,
    load_mnist_family,
)

__all__ = [
    "CIFARHistogramBags",
    "CIFARSignedBags",
    "MNISTOrdinalSumBags",
    "MNISTBags",
    "MNISTDigitHistogramBags",
    "MNISTDigitSumBags",
    "SVHNOrdinalSumBags",
    "SVHNSignedBags",
    "SignedMNISTBags",
    "TensorOrdinalSumBags",
    "TensorSignedBags",
    "UltraMNISTOrdinalSumBags",
    "collate_cifar_bags",
    "collate_mnist_bags",
    "collate_ordinal_sum_bags",
    "load_cifar_family",
    "load_idx_images",
    "load_idx_labels",
    "load_mnist_family",
    "load_svhn_family",
]
