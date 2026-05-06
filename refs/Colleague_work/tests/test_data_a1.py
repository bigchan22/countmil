"""Unit tests for A1 datasets and collate — see spec §10.3."""
import torch

from pca.data import variable_n_collate_fn


def test_collate_pads_to_n_max():
    """Items with N=3 and N=5 → batch padded to N_max=5 with mask."""
    item_a = (torch.randn(3, 1, 28, 28), 7, torch.tensor([2, 3, 2]),
              torch.ones(3, dtype=torch.bool))
    item_b = (torch.randn(5, 1, 28, 28), 11, torch.tensor([1, 4, 2, 3, 1]),
              torch.ones(5, dtype=torch.bool))
    feats, bag_sums, gts, mask = variable_n_collate_fn([item_a, item_b])
    assert feats.shape == (2, 5, 1, 28, 28), f"feats shape {tuple(feats.shape)}"
    assert mask.shape == (2, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True] * 5
    assert bag_sums.tolist() == [7, 11]
    assert gts.shape == (2, 5)


def test_collate_mask_active_count_matches_n():
    """mask.sum(dim=1) equals each bag's actual N."""
    items = [
        (torch.randn(n, 1, 28, 28), n, torch.zeros(n, dtype=torch.long),
         torch.ones(n, dtype=torch.bool))
        for n in [3, 5, 4, 7]
    ]
    _, _, _, mask = variable_n_collate_fn(items)
    assert mask.sum(dim=1).tolist() == [3, 5, 4, 7]


import numpy as np
from pca.data import MNISTSumBagDataset


def test_mnist_sum_bag_count_matches_gt():
    ds = MNISTSumBagDataset(num_bags=20, per_class_cap=100, seed=0)
    for i in range(len(ds)):
        images, bag_sum, gt, mask = ds[i]
        n_active = mask.sum().item()
        assert images.shape == (n_active, 1, 28, 28)
        assert gt.shape == (n_active,)
        assert mask.all()                     # MNISTSumBagDataset returns mask=all True
        assert bag_sum == int(gt.sum().item())


def test_mnist_sum_variable_n_in_range():
    ds = MNISTSumBagDataset(num_bags=200, bag_size_min=5, bag_size_max=15, seed=1)
    sizes = [ds[i][0].shape[0] for i in range(len(ds))]
    assert min(sizes) >= 5
    assert max(sizes) <= 15
    # Truncated normal with std=2: mean should be near 10
    assert 8.5 < float(np.mean(sizes)) < 11.5, f"mean={np.mean(sizes)}"


def test_mnist_sum_per_class_cap():
    ds = MNISTSumBagDataset(num_bags=10, per_class_cap=50, seed=2)
    label_counts = {}
    for k in range(10):
        label_counts[k] = (ds._labels == k).sum().item()
    for k, count in label_counts.items():
        assert count <= 50, f"class {k}: count={count} > cap=50"


def test_mnist_sum_seed_determinism():
    ds1 = MNISTSumBagDataset(num_bags=5, per_class_cap=100, seed=42)
    ds2 = MNISTSumBagDataset(num_bags=5, per_class_cap=100, seed=42)
    for i in range(5):
        f1, s1, g1, m1 = ds1[i]
        f2, s2, g2, m2 = ds2[i]
        assert torch.equal(f1, f2)
        assert s1 == s2
        assert torch.equal(g1, g2)


def test_mnist_sum_noise_increases_variance():
    ds_no = MNISTSumBagDataset(num_bags=50, per_class_cap=100, noise_sigma=0.0, seed=3)
    ds_n = MNISTSumBagDataset(num_bags=50, per_class_cap=100, noise_sigma=0.1, seed=3)
    var_no = torch.cat([ds_no[i][0].flatten() for i in range(20)]).var().item()
    var_n = torch.cat([ds_n[i][0].flatten() for i in range(20)]).var().item()
    assert var_n > var_no, f"noisy var={var_n} not > clean={var_no}"


from pca.data import MNISTSignedSumBagDataset


def test_mnist_signed_bag_sum_matches_shifted_gt():
    """Signed mapping: 0→0, odd nonzero→-1, even nonzero→+1.
       bag_sum stored = true_sum + N (caller un-shifts via metrics).
    """
    ds = MNISTSignedSumBagDataset(num_bags=20, per_class_cap=100, seed=0)
    for i in range(20):
        images, bag_sum_shifted, gt_signed, mask = ds[i]
        n = mask.sum().item()
        true_sum = int(gt_signed.sum().item())
        assert bag_sum_shifted == true_sum + n
        # gt values must be in {-1, 0, +1}
        assert set(gt_signed.tolist()).issubset({-1, 0, 1})


def test_mnist_signed_atom_support_size_3():
    """For N=10 signed, support length = 2*N+1 = 21 (after shifting)."""
    ds = MNISTSignedSumBagDataset(num_bags=10, bag_size_min=10, bag_size_max=10, seed=1)
    images, bag_sum, gt, mask = ds[0]
    assert mask.sum().item() == 10
    assert 0 <= bag_sum <= 20         # shifted range [0, 2N]


from pca.data import SVHNSumBagDataset


def test_svhn_sum_bag_count_matches_gt():
    """SVHN-sum bag dataset returns RGB 32x32 patches and matching bag sums."""
    ds = SVHNSumBagDataset(num_bags=8, seed=0)
    for i in range(8):
        images, bag_sum, gt, mask = ds[i]
        n = mask.sum().item()
        assert images.shape == (n, 3, 32, 32)
        assert gt.shape == (n,)
        assert bag_sum == int(gt.sum().item())


def test_svhn_sum_natural_difficulty_no_cap():
    """Default per_class_cap=None → uses full pool."""
    ds = SVHNSumBagDataset(num_bags=4, per_class_cap=None, seed=0)
    M = ds._images.shape[0]
    assert M > 50_000        # SVHN train ~73k


from pca.data import UltraMNISTBagDataset


def test_ultramnist_synthetic_bag_count_matches_gt():
    """UltraMNIST stand-in: 3-5 MNIST digits scattered, returns 64x64 patch crops."""
    ds = UltraMNISTBagDataset(num_bags=5, seed=0)
    for i in range(5):
        patches, bag_sum, gt, mask = ds[i]
        n = mask.sum().item()
        assert 3 <= n <= 5
        assert patches.shape == (n, 3, 64, 64)
        assert gt.shape == (n,)
        assert bag_sum == int(gt.sum().item())


def test_ultramnist_seed_determinism():
    ds1 = UltraMNISTBagDataset(num_bags=3, seed=42)
    ds2 = UltraMNISTBagDataset(num_bags=3, seed=42)
    for i in range(3):
        p1, s1, g1, _ = ds1[i]
        p2, s2, g2, _ = ds2[i]
        assert torch.equal(p1, p2)
        assert s1 == s2
        assert torch.equal(g1, g2)
