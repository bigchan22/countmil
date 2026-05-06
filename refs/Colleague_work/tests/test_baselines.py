"""Smoke tests for pca/baselines.py — see spec §10.2.

Per-baseline: forward returns finite loss + finite gradient on backbone params;
log_P (when not None) is a valid log-probability vector.
"""
import torch
import torch.nn.functional as F
import pytest


def _smoke_inputs(B=2, N=10, F_dim=128, K=9, T=None):
    """Random features + bag_y + mask for baseline smoke tests."""
    if T is None:
        T = N * K + 1
    torch.manual_seed(0)
    features = torch.randn(B, N, F_dim, requires_grad=True)
    bag_y = torch.randint(0, T, (B,))
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -2:] = False           # mark bag 0 as having only N-2 active
    return features, bag_y, mask


def _assert_log_p_valid(log_P, expected_T):
    assert log_P.shape[1] == expected_T, f"shape {tuple(log_P.shape)}"
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), \
        f"log_P does not sum to 1: got {sums.tolist()}"


def test_mean_pool_baseline_smoke():
    from pca.baselines import MeanPoolBaseline
    features, bag_y, mask = _smoke_inputs()
    model = MeanPoolBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_pl_multiclass_baseline_smoke():
    from pca.baselines import PLMulticlassBaseline
    features, bag_y, mask = _smoke_inputs()
    model = PLMulticlassBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_clt_gaussian_baseline_smoke():
    from pca.baselines import CLTGaussianBaseline
    features, bag_y, mask = _smoke_inputs()
    model = CLTGaussianBaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_attention_pooling_baseline_smoke():
    from pca.baselines import AttentionPoolingBaseline
    features, bag_y, mask = _smoke_inputs()
    model = AttentionPoolingBaseline(feature_dim=128, K=9, N_max=10, attn_dim=64)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_deepsets_baseline_smoke():
    from pca.baselines import DeepSetsBaseline
    features, bag_y, mask = _smoke_inputs()
    model = DeepSetsBaseline(feature_dim=128, K=9, N_max=10, hidden=64)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_pca_baseline_smoke():
    from pca.baselines import PCABaseline
    features, bag_y, mask = _smoke_inputs()
    model = PCABaseline(feature_dim=128, K=9, N_max=10)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, features)
    assert torch.isfinite(g).all()
    _assert_log_p_valid(log_P, expected_T=10 * 9 + 1)


def test_pca_baseline_signed_atom():
    """PCA with S=3 signed atom + caller-shifted bag_y works."""
    from pca.baselines import PCABaseline
    torch.manual_seed(0)
    B, N, F_dim = 2, 8, 128
    features = torch.randn(B, N, F_dim, requires_grad=True)
    # signed range: bag sum ∈ [-N, N], shifted to [0, 2N]
    bag_y = torch.randint(0, 2 * N + 1, (B,))
    mask = torch.ones(B, N, dtype=torch.bool)
    model = PCABaseline(feature_dim=128, K=2, N_max=8)   # K=2 → S=3 (signed support)
    loss, log_P = model(features, bag_y, mask)
    assert torch.isfinite(loss)
    _assert_log_p_valid(log_P, expected_T=8 * 2 + 1)
