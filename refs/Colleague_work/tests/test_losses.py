"""Unit tests for pca/losses.py — see spec §10.1."""
import torch
import torch.nn.functional as F

from pca.losses import forward_conv
from pca.losses import leave_one_out_posterior
from pca.losses import marginal_nll_loss
from pca.losses import em_joint_loss


def test_forward_conv_n2_hand():
    """
    p = [0.3, 0.4]
    P(Σ=0) = 0.7 * 0.6           = 0.42
    P(Σ=1) = 0.3*0.6 + 0.7*0.4   = 0.46
    P(Σ=2) = 0.3 * 0.4           = 0.12
    """
    p = torch.tensor([[0.3, 0.4]])
    log_P = forward_conv(p)
    P = log_P.exp()
    expected = torch.tensor([[0.42, 0.46, 0.12]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"


def test_forward_conv_n3_hand():
    """
    p = [0.5, 0.5, 0.5]
    Independent fair coins → Binomial(3, 0.5):
      P(0) = 1/8, P(1) = 3/8, P(2) = 3/8, P(3) = 1/8.
    """
    p = torch.tensor([[0.5, 0.5, 0.5]])
    log_P = forward_conv(p)
    P = log_P.exp()
    expected = torch.tensor([[0.125, 0.375, 0.375, 0.125]])
    assert torch.allclose(P, expected, atol=1e-6)


def test_forward_conv_sums_to_one():
    """For any valid p, sum_k P(Σ=k) ≈ 1."""
    torch.manual_seed(0)
    p = torch.rand(8, 50)
    log_P = forward_conv(p)
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), f"got {sums.tolist()}"


def test_posterior_n2_hand():
    """
    p = [0.3, 0.4], y = 1.
    P_{j≠0}(Σ=0) = 1-p_1 = 0.6,  P_{j≠0}(Σ=1) = p_1 = 0.4
    P_{j≠1}(Σ=0) = 1-p_0 = 0.7,  P_{j≠1}(Σ=1) = p_0 = 0.3
    P(Σ=1) = 0.46
    q_0 = p_0 * P_{j≠0}(Σ=0) / P(Σ=1) = 0.3 * 0.6 / 0.46 ≈ 0.39130
    q_1 = p_1 * P_{j≠1}(Σ=0) / P(Σ=1) = 0.4 * 0.7 / 0.46 ≈ 0.60870
    """
    p = torch.tensor([[0.3, 0.4]])
    bag_y = torch.tensor([1])
    q = leave_one_out_posterior(p, bag_y)
    expected = torch.tensor([[0.391304, 0.608696]])
    assert torch.allclose(q, expected, atol=1e-4), f"got {q.tolist()}"


def test_posterior_sum_equals_count():
    """Σ_u q_u should equal bag_y (posterior expected count = observed count)."""
    torch.manual_seed(1)
    p = torch.rand(8, 20)
    bag_y = torch.randint(0, 21, (8,))
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q.sum(dim=1), bag_y.float(), atol=1e-3), f"got {q.sum(dim=1).tolist()} vs {bag_y.tolist()}"


def test_posterior_y_zero():
    """If bag_y == 0, no positives → q ≡ 0."""
    p = torch.rand(4, 10)
    bag_y = torch.zeros(4, dtype=torch.long)
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q, torch.zeros_like(q))


def test_posterior_y_n():
    """If bag_y == N, all positive → q ≡ 1."""
    p = torch.rand(4, 10)
    bag_y = torch.full((4,), 10, dtype=torch.long)
    q = leave_one_out_posterior(p, bag_y)
    assert torch.allclose(q, torch.ones_like(q))


def test_marginal_nll_returns_scalar_finite():
    torch.manual_seed(2)
    logits = torch.randn(4, 10)
    bag_y = torch.randint(0, 11, (4,))
    loss = marginal_nll_loss(logits, bag_y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_em_loss_lam_zero_equals_nll():
    """lam=0 must be numerically equivalent to marginal_nll_loss within 1e-6."""
    torch.manual_seed(3)
    logits = torch.randn(4, 10)
    bag_y = torch.randint(0, 11, (4,))
    nll = marginal_nll_loss(logits, bag_y)
    em0 = em_joint_loss(logits, bag_y, lam=0.0)
    assert torch.allclose(nll, em0, atol=1e-6), f"nll={nll.item()} em0={em0.item()}"


def test_em_loss_lam_zero_grad_cosine():
    """Gradient direction equivalence: cos(grad NLL, grad EM(lam=0)) ≥ 0.9999."""
    torch.manual_seed(4)
    logits = torch.randn(4, 10, requires_grad=True)
    bag_y = torch.randint(0, 11, (4,))

    g_nll, = torch.autograd.grad(marginal_nll_loss(logits, bag_y), logits, retain_graph=False)
    g_em0, = torch.autograd.grad(em_joint_loss(logits, bag_y, lam=0.0), logits, retain_graph=False)
    cos = (g_nll * g_em0).sum() / (g_nll.norm() * g_em0.norm() + 1e-12)
    assert cos.item() > 0.9999, f"cos={cos.item()}"


def test_em_loss_no_nan_at_extremes():
    """Saturated probabilities + boundary bag_y must remain finite."""
    torch.manual_seed(5)
    # near-saturated logits (large positive → p≈1)
    logits = torch.full((4, 10), 8.0, requires_grad=True)
    for y in [0, 5, 10]:
        bag_y = torch.full((4,), y, dtype=torch.long)
        loss = em_joint_loss(logits, bag_y, lam=1.0)
        assert torch.isfinite(loss), f"loss not finite at y={y}: {loss.item()}"
        g, = torch.autograd.grad(loss, logits, retain_graph=False)
        assert torch.isfinite(g).all(), f"grad not finite at y={y}"
