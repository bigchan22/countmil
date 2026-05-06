"""Unit tests for pca/data.py — see spec §10.2."""
import torch

from pca.data import LLPBagDataset, MNISTBagDataset


def test_mnist_bag_count_matches_gt():
    ds = MNISTBagDataset(bag_size=10, num_bags=20, positive_digit=9, train=True, seed=0)
    for i in range(len(ds)):
        images, bag_count, gt_labels = ds[i]
        assert bag_count == int(gt_labels.sum().item())
        assert images.shape == (10, 1, 28, 28)
        assert gt_labels.shape == (10,)
        assert gt_labels.dtype == torch.long


def test_mnist_seed_determinism():
    ds1 = MNISTBagDataset(bag_size=10, num_bags=5, positive_digit=9, train=True, seed=42)
    ds2 = MNISTBagDataset(bag_size=10, num_bags=5, positive_digit=9, train=True, seed=42)
    for i in range(5):
        i1, c1, g1 = ds1[i]
        i2, c2, g2 = ds2[i]
        assert torch.equal(i1, i2)
        assert c1 == c2
        assert torch.equal(g1, g2)


def test_llp_adult_loads_and_caches():
    """First load fetches+caches; second load is fast (uses cache)."""
    import time
    t0 = time.time()
    ds = LLPBagDataset('adult', bag_size=8, num_bags=10, train=True, seed=0)
    t1 = time.time()
    # Second instantiation should hit cache (under 5s)
    ds2 = LLPBagDataset('adult', bag_size=8, num_bags=10, train=True, seed=0)
    t2 = time.time()
    assert t2 - t1 < 5.0, f"second load took {t2 - t1:.2f}s — cache not used?"


def test_llp_adult_feature_dim():
    ds = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=0)
    feats, bag_count, gt = ds[0]
    assert feats.shape[0] == 8
    assert feats.shape[1] >= 100  # Adult D ≈ 108 after one-hot
    assert gt.shape == (8,)
    assert bag_count == int(gt.sum().item())


def test_llp_seed_determinism():
    ds1 = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=7)
    ds2 = LLPBagDataset('adult', bag_size=8, num_bags=5, train=True, seed=7)
    for i in range(5):
        f1, c1, g1 = ds1[i]
        f2, c2, g2 = ds2[i]
        assert torch.equal(f1, f2)
        assert c1 == c2
