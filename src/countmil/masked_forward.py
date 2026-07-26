"""Utilities for forwarding padded bags without evaluating padded entries."""

from __future__ import annotations

import torch
from torch import nn


def forward_valid_instances(module: nn.Module, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forward only valid bag entries through an image module.

    Args:
        module: Module accepting a flat image tensor ``(M,C,H,W)``.
        images: Padded bag tensor ``(B,N,C,H,W)``.
        mask: Boolean valid-entry tensor ``(B,N)``.

    Returns:
        Tensor ``(B,N,...)`` where padded rows are zeros and valid rows preserve
        the original bag order.
    """

    if images.ndim != 5:
        raise ValueError("images must have shape (B,N,C,H,W)")
    if mask.shape != images.shape[:2]:
        raise ValueError("mask must have shape (B,N)")
    mask = mask.to(device=images.device, dtype=torch.bool)
    valid_images = images[mask]
    if valid_images.numel() == 0:
        raise ValueError("at least one valid image is required")
    valid_out = module(valid_images)
    out = valid_out.new_zeros(*images.shape[:2], *valid_out.shape[1:])
    out[mask] = valid_out
    return out
