"""Strict train/validation/test split and manifest helpers for rebuttal runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


STRICTV3_TAG = "strictv3_train_holdout_val_official_test_no_hidden_val"


def canonical_hash(payload: dict[str, Any]) -> str:
    """Hash tensors and scalar metadata in a stable content-oriented format."""

    h = hashlib.sha256()

    def update_obj(key: str, value: Any) -> None:
        h.update(str(key).encode())
        h.update(b"\0")
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(value, dict):
            for sub_key in sorted(value):
                update_obj(f"{key}.{sub_key}", value[sub_key])
        elif isinstance(value, (list, tuple)):
            h.update(json.dumps(list(value), sort_keys=True).encode())
        elif isinstance(value, (str, int, float, bool)) or value is None:
            h.update(json.dumps(value, sort_keys=True).encode())
        else:
            raise TypeError(f"unsupported hash payload type for {key}: {type(value).__name__}")
        h.update(b"\0")

    for key in sorted(payload):
        update_obj(key, payload[key])
    return h.hexdigest()


def stratified_train_val_split(labels: torch.Tensor, val_fraction: float = 0.15, seed: int = 314159) -> dict[str, torch.Tensor]:
    labels = labels.detach().cpu().long()
    gen = torch.Generator().manual_seed(int(seed))
    train_parts = []
    val_parts = []
    for cls in torch.unique(labels, sorted=True).tolist():
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        n_val = int(round(float(idx.numel()) * float(val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = torch.cat(train_parts).sort().values
    val_idx = torch.cat(val_parts).sort().values
    if torch.isin(train_idx, val_idx).any():
        raise RuntimeError("train/validation split overlap")
    return {"train_indices": train_idx, "val_indices": val_idx}


def split_stats(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> dict[str, Any]:
    subset = labels[indices].long()
    counts = torch.bincount(subset, minlength=num_classes)
    props = counts.float() / counts.sum().clamp_min(1)
    return {
        "num_images": int(indices.numel()),
        "class_counts": counts.tolist(),
        "class_proportions": [float(x) for x in props.tolist()],
        "indices_hash": canonical_hash({"indices": indices.long(), "class_counts": counts.long()}),
    }


def save_split_payload(path: Path, labels: torch.Tensor, splits: dict[str, torch.Tensor], num_classes: int, metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **{k: v.long().cpu() for k, v in splits.items()},
        "metadata": dict(metadata),
    }
    payload["train_stats"] = split_stats(labels, payload["train_indices"], num_classes)
    payload["val_stats"] = split_stats(labels, payload["val_indices"], num_classes)
    payload["split_hash"] = canonical_hash(
        {
            "train_indices": payload["train_indices"],
            "val_indices": payload["val_indices"],
            "metadata": metadata,
        }
    )
    torch.save(payload, path)
    (path.with_suffix(".json")).write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return payload


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass(frozen=True)
class BagManifestSpec:
    split: str
    seed: int
    num_bags: int
    bag_size_mean: int
    bag_size_std: float
    task: str
    target_digit: int = 9
    cancellation_heavy: bool = False


def sample_variable_mnist_manifest(
    labels: torch.Tensor,
    pool_indices: torch.Tensor,
    spec: BagManifestSpec,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    gen = torch.Generator().manual_seed(int(spec.seed))
    labels = labels.detach().cpu().long()
    pool_indices = pool_indices.detach().cpu().long()
    lengths = []
    bags = []
    signs_list = []
    for _ in range(int(spec.num_bags)):
        if spec.bag_size_std > 0:
            n = max(1, int(round(float(torch.normal(torch.tensor(float(spec.bag_size_mean)), torch.tensor(float(spec.bag_size_std)), generator=gen).item()))))
        else:
            n = int(spec.bag_size_mean)
        draw = pool_indices[torch.randint(0, pool_indices.numel(), (n,), generator=gen)]
        lengths.append(n)
        bags.append(draw)
        if spec.task.startswith("signed"):
            if spec.cancellation_heavy:
                signs = torch.ones(n, dtype=torch.long)
                signs[1::2] = -1
                signs = signs[torch.randperm(n, generator=gen)]
            else:
                signs = torch.randint(0, 2, (n,), generator=gen).mul(2).sub(1).long()
            signs_list.append(signs)
    max_len = max(lengths)
    indices = torch.full((spec.num_bags, max_len), -1, dtype=torch.long)
    mask = torch.zeros(spec.num_bags, max_len, dtype=torch.bool)
    signs = torch.zeros(spec.num_bags, max_len, dtype=torch.long)
    for b, draw in enumerate(bags):
        n = draw.numel()
        indices[b, :n] = draw
        mask[b, :n] = True
        if signs_list:
            signs[b, :n] = signs_list[b]
    safe_indices = indices.clamp_min(0)
    digits = labels[safe_indices]
    digits = torch.where(mask, digits, torch.zeros_like(digits))
    payload: dict[str, Any] = {"indices": indices, "mask": mask, "digits": digits}
    if spec.task == "digit_sum":
        payload["aggregate"] = (digits * mask.long()).sum(dim=1).long()
    elif spec.task.startswith("signed"):
        hidden = ((digits == int(spec.target_digit)) & mask).long()
        payload["signs"] = signs
        payload["hidden"] = hidden
        payload["aggregate"] = (hidden * signs).sum(dim=1).long()
        payload["cancellation_zero_fraction"] = float((payload["aggregate"] == 0).float().mean())
    else:
        raise ValueError(f"unknown manifest task: {spec.task}")
    payload["metadata"] = {
        "split": spec.split,
        "seed": int(spec.seed),
        "num_bags": int(spec.num_bags),
        "bag_size_mean": int(spec.bag_size_mean),
        "bag_size_std": float(spec.bag_size_std),
        "task": spec.task,
        "target_digit": int(spec.target_digit),
        "cancellation_heavy": bool(spec.cancellation_heavy),
    }
    payload["manifest_hash"] = canonical_hash({k: v for k, v in payload.items() if k != "metadata"} | {"metadata": payload["metadata"]})
    return payload


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    path.with_suffix(".json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


def sample_cifar_count_manifest(
    labels: torch.Tensor,
    pool_indices: torch.Tensor,
    *,
    split: str,
    seed: int,
    num_bags: int,
    bag_size: int,
    alpha: float,
    num_classes: int,
) -> dict[str, Any]:
    labels = labels.detach().cpu().long()
    pool_indices = pool_indices.detach().cpu().long()
    gen = torch.Generator().manual_seed(int(seed))
    class_indices = []
    for c in range(num_classes):
        cls = pool_indices[labels[pool_indices] == c]
        if cls.numel() == 0:
            raise ValueError(f"class {c} has no images in {split} pool")
        class_indices.append(cls)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        priors = torch.distributions.Dirichlet(torch.full((num_classes,), float(alpha))).sample((num_bags,))
    counts = torch.empty(num_bags, num_classes, dtype=torch.long)
    indices = torch.empty(num_bags, bag_size, dtype=torch.long)
    for b in range(num_bags):
        cnt = torch.multinomial(priors[b], bag_size, replacement=True, generator=gen).bincount(minlength=num_classes)
        counts[b] = cnt
        pos = 0
        for c, n_c in enumerate(cnt.tolist()):
            if n_c:
                pool = class_indices[c]
                draw = pool[torch.randint(0, pool.numel(), (n_c,), generator=gen)]
                indices[b, pos : pos + n_c] = draw
                pos += n_c
        indices[b] = indices[b, torch.randperm(bag_size, generator=gen)]
    props = counts.float() / float(bag_size)
    global_prior = torch.bincount(labels[pool_indices], minlength=num_classes).float()
    global_prior = global_prior / global_prior.sum()
    entropy = -(props.clamp_min(1e-12) * props.clamp_min(1e-12).log()).sum(dim=1)
    dist = (priors - global_prior).abs().sum(dim=1)
    metadata = {
        "split": split,
        "seed": int(seed),
        "num_bags": int(num_bags),
        "bag_size": int(bag_size),
        "alpha": float(alpha),
        "num_classes": int(num_classes),
    }
    stats = {
        "num_unique_bags": int(torch.unique(indices, dim=0).shape[0]),
        "total_instance_slots": int(indices.numel()),
        "num_unique_underlying_images": int(torch.unique(indices).numel()),
        "per_class_count_mean": [float(x) for x in counts.float().mean(dim=0).tolist()],
        "per_class_count_sample_std": [float(x) for x in counts.float().std(dim=0, unbiased=True).tolist()],
        "bag_composition_entropy_mean": float(entropy.mean()),
        "bag_composition_entropy_sample_std": float(entropy.std(unbiased=True)),
        "min_class_proportion": float(props.min()),
        "max_class_proportion": float(props.max()),
        "l1_distance_to_global_prior_mean": float(dist.mean()),
        "l1_distance_to_global_prior_sample_std": float(dist.std(unbiased=True)),
    }
    payload = {"indices": indices, "counts": counts, "priors": priors, "metadata": metadata, "stats": stats}
    payload["manifest_hash"] = canonical_hash({"indices": indices, "counts": counts, "priors": priors, "metadata": metadata})
    return payload
