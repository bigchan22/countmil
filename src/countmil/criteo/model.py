"""Common tabular instance model for Criteo aggregate-supervision runs."""

from __future__ import annotations

import torch
from torch import nn


class CriteoInstanceModel(nn.Module):
    """Shared categorical embedding MLP used by every aggregate method."""

    def __init__(
        self,
        *,
        categorical_cardinalities: list[int],
        shared_categorical: bool,
        embedding_dim: int = 8,
        num_numeric: int = 13,
    ) -> None:
        super().__init__()
        self.shared_categorical = bool(shared_categorical)
        self.num_fields = len(categorical_cardinalities)
        self.num_numeric = int(num_numeric)
        self.embedding_dim = int(embedding_dim)
        if self.shared_categorical:
            self.embeddings = nn.Embedding(max(categorical_cardinalities), embedding_dim)
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(n, embedding_dim) for n in categorical_cardinalities])
        width = self.num_fields * embedding_dim + self.num_numeric
        self.net = nn.Sequential(
            nn.Linear(width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if numeric.ndim != 2 or categorical.ndim != 2:
            raise ValueError("numeric and categorical inputs must be 2D")
        if self.shared_categorical:
            emb = self.embeddings(categorical.long())
        else:
            parts = [emb(categorical[:, j].long()) for j, emb in enumerate(self.embeddings)]
            emb = torch.stack(parts, dim=1)
        x = torch.cat([numeric.float(), emb.flatten(start_dim=1)], dim=1)
        return self.net(x).squeeze(-1)


def infer_categorical_layout(mins: list[int], maxs: list[int]) -> tuple[bool, list[int]]:
    """Infer whether Criteo categorical IDs can use one shared embedding table."""

    ranges = sorted((int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    disjoint = True
    prev_hi = -1
    for lo, hi in ranges:
        if lo <= prev_hi:
            disjoint = False
            break
        prev_hi = max(prev_hi, hi)
    if disjoint:
        return True, [max(maxs) + 1 for _ in maxs]
    return False, [int(hi) + 1 for hi in maxs]
