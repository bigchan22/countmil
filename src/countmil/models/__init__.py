"""Selector and instance-level model definitions."""

from .atomic_sum import PatchOrdinalClassifier
from .cifar_cnn import CIFARResNet18Classifier, CIFARSmallClassifier, make_cifar_classifier
from .mnist_cnn import (
    AttentionMILMNIST,
    MNISTCountLossCNN,
    MNISTDigitClassifier,
    MNISTFeatureExtractor,
    ShuklaMNISTSelector,
)

__all__ = [
    "AttentionMILMNIST",
    "CIFARResNet18Classifier",
    "CIFARSmallClassifier",
    "PatchOrdinalClassifier",
    "make_cifar_classifier",
    "MNISTCountLossCNN",
    "MNISTDigitClassifier",
    "MNISTFeatureExtractor",
    "ShuklaMNISTSelector",
]
