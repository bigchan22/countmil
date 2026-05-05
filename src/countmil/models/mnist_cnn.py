"""MNIST instance selectors.

The default architecture follows the MNIST MIL network reported by Shukla et
al. for Count Loss experiments, itself derived from Ilse et al.:

conv(5)-20 + ReLU, maxpool(2), conv(5)-50 + ReLU, maxpool(2),
fc-500 + ReLU, fc-1.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


OutputMode = Literal["logits", "probs", "log_probs"]


class ShuklaMNISTSelector(nn.Module):
    """LeNet-style binary instance selector for MNIST bags.

    Input can be either `(N,1,28,28)` or `(B,N,1,28,28)`.
    """

    def __init__(self, output_mode: OutputMode = "logits") -> None:
        super().__init__()
        if output_mode not in {"logits", "probs", "log_probs"}:
            raise ValueError(f"unsupported output_mode: {output_mode}")
        self.output_mode = output_mode
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 5:
            batch, bag, channels, height, width = x.shape
            x = x.reshape(batch * bag, channels, height, width)
            restore_shape = (batch, bag)
        elif x.ndim == 4:
            restore_shape = original_shape[:1]
        else:
            raise ValueError("expected input shape (N,1,28,28) or (B,N,1,28,28)")

        logits = self.classifier(self.features(x)).squeeze(-1)
        logits = logits.reshape(*restore_shape)

        if self.output_mode == "logits":
            return logits
        if self.output_mode == "probs":
            return torch.sigmoid(logits)
        return F.logsigmoid(logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.output_mode
        self.output_mode = "logits"
        try:
            return self.forward(x)
        finally:
            self.output_mode = mode


MNISTCountLossCNN = ShuklaMNISTSelector


class MNISTDigitClassifier(nn.Module):
    """LeNet-style 10-class instance classifier for MNIST digit-sum bags.

    Input can be either `(N,1,28,28)` or `(B,N,1,28,28)`. The output appends a
    class dimension of size `num_classes`.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 5:
            batch, bag, channels, height, width = x.shape
            x = x.reshape(batch * bag, channels, height, width)
            restore_shape = (batch, bag)
        elif x.ndim == 4:
            restore_shape = original_shape[:1]
        else:
            raise ValueError("expected input shape (N,1,28,28) or (B,N,1,28,28)")
        logits = self.classifier(self.features(x))
        return logits.reshape(*restore_shape, self.num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


class MNISTFeatureExtractor(nn.Module):
    """Feature trunk matching the Shukla/Ilse MNIST MIL CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected input shape (N,1,28,28)")
        return self.features(x)


class AttentionMILMNIST(nn.Module):
    """Ilse-style attention or gated-attention MIL baseline for MNIST-Bags."""

    def __init__(self, attention_dim: int = 128, gated: bool = False) -> None:
        super().__init__()
        self.gated = bool(gated)
        self.encoder = MNISTFeatureExtractor()
        self.attention_v = nn.Linear(500, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        if self.gated:
            self.attention_u = nn.Linear(500, attention_dim)
        self.classifier = nn.Linear(500, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(bag_logits, attention_weights)`.

        Args:
            x: `(B,N,1,28,28)` padded bag tensor.
            mask: `(B,N)` valid-instance mask. If omitted, all instances are valid.
        """

        if x.ndim != 5:
            raise ValueError("expected input shape (B,N,1,28,28)")
        batch, bag, channels, height, width = x.shape
        if mask is None:
            mask = torch.ones(batch, bag, dtype=torch.bool, device=x.device)
        h = self.encoder(x.reshape(batch * bag, channels, height, width)).reshape(batch, bag, 500)
        a = torch.tanh(self.attention_v(h))
        if self.gated:
            a = a * torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(a).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask.bool(), weights, torch.zeros_like(weights))
        bag_repr = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        logits = self.classifier(bag_repr).squeeze(-1)
        return logits, weights
