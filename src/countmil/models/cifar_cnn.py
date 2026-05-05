"""CIFAR instance classifiers for bag experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CIFARSmallClassifier(nn.Module):
    """Compact CNN for CIFAR-10/100 coarse-label bag experiments."""

    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 5:
            batch, bag, channels, height, width = x.shape
            x = x.reshape(batch * bag, channels, height, width)
            restore_shape = (batch, bag)
        elif x.ndim == 4:
            restore_shape = original_shape[:1]
        else:
            raise ValueError("expected input shape (N,3,32,32) or (B,N,3,32,32)")
        logits = self.classifier(self.features(x).flatten(1))
        return logits.reshape(*restore_shape, self.num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


class CIFARResNet18Classifier(nn.Module):
    """Torchvision ResNet-18 wrapper for CIFAR bag experiments.

    `pretrained=True` requests ImageNet weights from torchvision. That may
    download weights if they are not already cached, so experiment launchers
    keep it opt-in.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - depends on optional torchvision
            raise ImportError("CIFARResNet18Classifier requires torchvision") from exc

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        if pretrained:
            mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
            std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
        else:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 5:
            batch, bag, channels, height, width = x.shape
            x = x.reshape(batch * bag, channels, height, width)
            restore_shape = (batch, bag)
        elif x.ndim == 4:
            restore_shape = original_shape[:1]
        else:
            raise ValueError("expected input shape (N,3,32,32) or (B,N,3,32,32)")
        x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
        if self.pretrained:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.model(x)
        return logits.reshape(*restore_shape, self.num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def make_cifar_classifier(backbone: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if backbone == "small_cnn":
        if pretrained:
            raise ValueError("pretrained is only supported for resnet18")
        return CIFARSmallClassifier(num_classes=num_classes)
    if backbone == "resnet18":
        return CIFARResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"unknown CIFAR backbone: {backbone}")
