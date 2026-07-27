import math

import torch
from torch import nn

from countmil.baselines.gaussian_amle import gaussian_integer_bin_nll
from countmil.masked_forward import forward_valid_instances
from countmil.strict_protocol import canonical_hash, sample_variable_mnist_manifest, stratified_train_val_split, BagManifestSpec
from scripts.rebuttal.a5000_train_svhn_scalar import masked_bag_class_probs


class BatchSensitiveBackbone(nn.Module):
    def forward(self, x):
        return x.flatten(1) - x.flatten(1).mean(dim=0, keepdim=True)


def test_forward_valid_instances_independent_of_padding_count():
    model = BatchSensitiveBackbone()
    valid = torch.randn(2, 3, 1, 2, 2, requires_grad=True)
    short = torch.zeros(2, 5, 1, 2, 2)
    long = torch.zeros(2, 9, 1, 2, 2)
    short[:, :3] = valid
    long[:, :3] = valid
    short_mask = torch.zeros(2, 5, dtype=torch.bool)
    long_mask = torch.zeros(2, 9, dtype=torch.bool)
    short_mask[:, :3] = True
    long_mask[:, :3] = True
    out_short = forward_valid_instances(model, short, short_mask)
    out_long = forward_valid_instances(model, long, long_mask)
    assert torch.allclose(out_short[short_mask], out_long[long_mask])
    loss = out_short[short_mask].square().sum()
    loss.backward()
    assert valid.grad is not None
    assert torch.isfinite(valid.grad).all()


def test_svhn_masked_bag_probs_do_not_forward_padding():
    model = BatchSensitiveBackbone()
    valid = torch.randn(2, 3, 1, 2, 2, requires_grad=True)
    short = torch.zeros(2, 5, 1, 2, 2)
    long = torch.zeros(2, 8, 1, 2, 2)
    short[:, :3] = valid
    long[:, :3] = valid
    short_mask = torch.zeros(2, 5, dtype=torch.bool)
    long_mask = torch.zeros(2, 8, dtype=torch.bool)
    short_mask[:, :3] = True
    long_mask[:, :3] = True
    short_probs = masked_bag_class_probs(model, short, short_mask)
    long_probs = masked_bag_class_probs(model, long, long_mask)
    assert torch.allclose(short_probs[short_mask], long_probs[long_mask])
    assert torch.all(short_probs[~short_mask] == torch.softmax(torch.zeros(4), dim=0))
    loss = short_probs[short_mask].square().sum()
    loss.backward()
    assert valid.grad is not None
    assert torch.isfinite(valid.grad).all()


def test_stratified_split_is_disjoint_and_reproducible():
    labels = torch.arange(100) % 10
    a = stratified_train_val_split(labels, val_fraction=0.15, seed=7)
    b = stratified_train_val_split(labels, val_fraction=0.15, seed=7)
    assert torch.equal(a["train_indices"], b["train_indices"])
    assert torch.equal(a["val_indices"], b["val_indices"])
    assert not torch.isin(a["train_indices"], a["val_indices"]).any()
    assert a["val_indices"].numel() == 20  # round(10 * .15) per class = 2


def test_manifest_hash_changes_with_signs():
    labels = torch.arange(200) % 10
    pool = torch.arange(200)
    spec = BagManifestSpec("train", 1, 8, 10, 2.0, "signed_random", cancellation_heavy=False)
    first = sample_variable_mnist_manifest(labels, pool, spec)
    second = sample_variable_mnist_manifest(labels, pool, spec)
    assert first["manifest_hash"] == second["manifest_hash"]
    changed = dict(first)
    changed["signs"] = first["signs"].clone()
    changed["signs"][0, 0] *= -1
    assert canonical_hash({"indices": changed["indices"], "mask": changed["mask"], "signs": changed["signs"]}) != canonical_hash(
        {"indices": first["indices"], "mask": first["mask"], "signs": first["signs"]}
    )


def test_gaussian_integer_bin_nll_stable_central_and_tails():
    mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    var = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    targets = torch.tensor([0.0, 8.0, -15.0], dtype=torch.float64)
    nll = gaussian_integer_bin_nll(mean, var, targets, eps=1e-12)
    assert torch.isfinite(nll).all()
    assert nll[0].item() < nll[1].item() < nll[2].item()
    nll.sum().backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad[1].item() < 0
    assert mean.grad[2].item() > 0


def test_gaussian_integer_bin_nll_matches_erf_reference_near_center():
    mean = torch.tensor([1.2], dtype=torch.float64)
    var = torch.tensor([2.3], dtype=torch.float64)
    target = torch.tensor([2.0], dtype=torch.float64)
    got = gaussian_integer_bin_nll(mean, var, target, eps=0.0 + 1e-12)
    sigma = math.sqrt(2.3 + 1e-12)
    upper = (2.5 - 1.2) / (sigma * math.sqrt(2.0))
    lower = (1.5 - 1.2) / (sigma * math.sqrt(2.0))
    mass = 0.5 * (math.erf(upper) - math.erf(lower))
    assert torch.allclose(got, torch.tensor([-math.log(mass)], dtype=torch.float64), atol=1e-10)
