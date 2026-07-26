import torch

from scripts.rebuttal.train_fixed_cifar10_features import _iter_batches, _manifest_hash, _sample_fixed_bags


def test_fixed_bag_manifest_hash_is_epoch_invariant():
    labels = torch.arange(100) % 10
    bags = _sample_fixed_bags(labels, num_bags=12, bag_size=8, alpha=0.3, seed=7, num_classes=10)
    original = _manifest_hash(bags)

    for epoch in range(1, 4):
        list(_iter_batches(torch.randn(100, 4), labels, bags, batch_size=3, shuffle=True, seed=100 + epoch))
        assert _manifest_hash(bags) == original


def test_fixed_bag_sampling_is_seed_reproducible():
    labels = torch.arange(100) % 10
    a = _sample_fixed_bags(labels, num_bags=12, bag_size=8, alpha=0.3, seed=11, num_classes=10)
    b = _sample_fixed_bags(labels, num_bags=12, bag_size=8, alpha=0.3, seed=11, num_classes=10)
    c = _sample_fixed_bags(labels, num_bags=12, bag_size=8, alpha=0.3, seed=12, num_classes=10)

    assert _manifest_hash(a) == _manifest_hash(b)
    assert _manifest_hash(a) != _manifest_hash(c)
