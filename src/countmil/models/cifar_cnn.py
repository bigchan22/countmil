"""Small CNN instance classifier for CIFAR bag experiments."""

from __future__ import annotations

import torch
from torch import nn


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
