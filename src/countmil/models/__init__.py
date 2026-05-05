"""Selector and instance-level model definitions."""

from .cifar_cnn import CIFARSmallClassifier
from .mnist_cnn import (
    AttentionMILMNIST,
    MNISTCountLossCNN,
    MNISTDigitClassifier,
    MNISTFeatureExtractor,
    ShuklaMNISTSelector,
)

__all__ = [
    "AttentionMILMNIST",
    "CIFARSmallClassifier",
    "MNISTCountLossCNN",
    "MNISTDigitClassifier",
    "MNISTFeatureExtractor",
    "ShuklaMNISTSelector",
]
