"""Per-instance backbones for B1 verification.

SmallCNN: MNIST → single logit.
MLP:      tabular features → single logit.

See spec §7.
"""
from torch import nn, Tensor


class SmallCNN(nn.Module):
    """Per-instance MNIST → single logit. ~206k params, Ilse 2018 / Shukla scale."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28→14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B*N, 1, 28, 28)
        return self.classifier(self.features(x))   # (B*N, 1)


class MLP(nn.Module):
    """Per-instance tabular feature vector → single logit."""

    def __init__(self, in_dim: int, hidden=(64, 32), dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Dropout(dropout), nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # x: (B*N, D)
        return self.net(x)


class SmallCNNMulticlass(nn.Module):
    """Per-instance MNIST → 128-dim feature vector.

    Identical body to SmallCNN (B1) but the final classification layer is
    OMITTED — this class is a feature extractor; baselines own their own head.
    Output shape: (B*N, 128). For MNIST-sum and (via num_classes parameter on
    the head) MNIST-signed.
    """
    feature_dim = 128

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:    # x: (B*N, 1, 28, 28)
        return self.head(self.features(x))      # (B*N, 128)


class ResNet18FromScratch(nn.Module):
    """ResNet-18 from scratch (no pretrained weights). 32x32 input adaptation.

    Replaces the original 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv
    (CIFAR-style adaptation; preserves spatial resolution for small inputs),
    and removes the final classification layer to expose 512-d features.
    Output shape: (B*N, 512). For SVHN-sum.
    """
    feature_dim = 512

    def __init__(self, in_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(weights=None)
        # CIFAR-style stem
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        # Drop final classifier; .fc → identity, output is the 512-dim avgpool result.
        self.net.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:     # x: (B*N, 3, 32, 32)
        return self.net(x)                       # (B*N, 512)


class PatchEncoder(nn.Module):
    """Small RGB CNN for 64x64 patches → 128-dim feature.

    For UltraMNIST: per-instance feature is a digit-bounding-box patch crop
    resized to 64x64 RGB. Output shape: (B*N, 128). Architecturally similar
    to SmallCNNMulticlass but adapted for 64x64x3 input.
    """
    feature_dim = 128

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32→16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16→8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:    # x: (B*N, 3, 64, 64)
        return self.head(self.features(x))      # (B*N, 128)
