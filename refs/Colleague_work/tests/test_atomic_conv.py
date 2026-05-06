"""Unit tests for atomic_conv and multiclass_marginal_nll_loss — see spec §10.1."""
import torch
import torch.nn.functional as F

from pca.losses import forward_conv, atomic_conv, multiclass_marginal_nll_loss


def test_atomic_conv_bernoulli_equiv_forward_conv():
    """Bernoulli case (S=2): atomic_conv(log_atoms) ≈ forward_conv(p)."""
    torch.manual_seed(0)
    p = torch.rand(8, 12).clamp(min=1e-3, max=1 - 1e-3)
    log_atoms = torch.stack([(1 - p).log(), p.log()], dim=-1)   # (B, N, 2)
    log_P_atomic = atomic_conv(log_atoms)
    log_P_forward = forward_conv(p)
    assert torch.allclose(log_P_atomic, log_P_forward, atol=1e-6), \
        f"max diff: {(log_P_atomic - log_P_forward).abs().max().item()}"


def test_atomic_conv_n2_multiclass_hand():
    """
    N=2, K=2, support {0, 1, 2}. Atoms d_1 = [0.5, 0.3, 0.2], d_2 = [0.4, 0.4, 0.2].
    Conv:
      P(0) = 0.5*0.4                                = 0.20
      P(1) = 0.5*0.4 + 0.3*0.4                      = 0.32
      P(2) = 0.5*0.2 + 0.3*0.4 + 0.2*0.4            = 0.30
      P(3) = 0.3*0.2 + 0.2*0.4                      = 0.14
      P(4) = 0.2*0.2                                = 0.04
    """
    log_atoms = torch.tensor([[
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2],
    ]]).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    expected = torch.tensor([[0.20, 0.32, 0.30, 0.14, 0.04]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"


def test_atomic_conv_n3_multiclass_hand():
    """N=3 uniform atoms over {0,1,2}: bag PMF = trinomial 1/27 of compositions."""
    log_atoms = torch.full((1, 3, 3), 1.0 / 3).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    # Compositions of s using {0,1,2} with 3 instances:
    # P(s=0) = C(3,3,0,0)*(1/3)^3 = 1/27
    # P(s=1) = 3/27, P(s=2)=6/27, P(s=3)=7/27, P(s=4)=6/27, P(s=5)=3/27, P(s=6)=1/27
    expected = torch.tensor([[1, 3, 6, 7, 6, 3, 1]]) / 27.0
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"


def test_atomic_conv_sums_to_one():
    """For any valid atom logits, sum_k P(Σ=k) ≈ 1."""
    torch.manual_seed(0)
    logits = torch.randn(8, 10, 4)              # (B, N, S=4) random
    log_atoms = F.log_softmax(logits, dim=-1)
    log_P = atomic_conv(log_atoms)
    sums = log_P.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), f"got {sums.tolist()}"


def test_atomic_conv_s3_symmetric_pmf():
    """S=3 symmetric atoms produce the expected triangle bag PMF.

    For N=2 with d_1 = d_2 = [0.25, 0.5, 0.25]:
      P(s=0) = 0.0625, P(s=1) = 0.25, P(s=2) = 0.375, P(s=3) = 0.25, P(s=4) = 0.0625.

    Note: this verifies S=3 PMF correctness only. The caller-side +N shift
    convention used by signed atoms (e.g., {-1, 0, +1} → bag_y_shifted = true_sum + N
    in MNISTSignedSumBagDataset, Task 21) is exercised end-to-end in Task 16
    (PCABaseline signed-atom test), not here.
    """
    log_atoms = torch.tensor([[[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]]).log()
    log_P = atomic_conv(log_atoms)
    P = log_P.exp()
    expected = torch.tensor([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
    assert torch.allclose(P, expected, atol=1e-6), f"got {P.tolist()}"


def test_multiclass_nll_finite():
    """Forward + grad finite on random batch."""
    torch.manual_seed(2)
    logits = torch.randn(4, 10, 5, requires_grad=True)   # B=4, N=10, S=5 (K=4)
    bag_y = torch.randint(0, 10 * 4 + 1, (4,))            # bag sum in [0, NK]
    loss = multiclass_marginal_nll_loss(logits, bag_y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    g, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(g).all()


def test_multiclass_nll_mask_handles_variable_n():
    """Loss with N=4 + mask=[T,T,F,F] equals loss on identical first-2 logits with N=2."""
    torch.manual_seed(7)
    full = torch.randn(2, 4, 4)                              # (B=2, N_max=4, S=4)
    mask = torch.tensor([[True, True, False, False],
                         [True, True, False, False]])
    bag_y = torch.tensor([3, 5])                              # bag sum in [0, 2*3]=6
    loss_masked = multiclass_marginal_nll_loss(full, bag_y, mask=mask)
    # Equivalent N=2 case: pad bag PMF to T=N_max*(S-1)+1=13. Use first-2 logits;
    # bag_y stays the same. Without mask the bag-sum range is [0, 2*3]=6,
    # but atomic_conv yields T=2*3+1=7 there vs T=4*3+1=13 here. With masking,
    # inactive atoms = delta_0 keep bag_y at the same offset → loss should match
    # 'unpadded' N=2 evaluated at the same bag_y.
    half = full[:, :2]
    loss_n2 = multiclass_marginal_nll_loss(half, bag_y)
    assert torch.allclose(loss_masked, loss_n2, atol=1e-5), \
        f"masked={loss_masked.item()} vs n2={loss_n2.item()}"


def test_small_cnn_multiclass_output_shape():
    """Forward returns (B*N, 128) features."""
    from pca.models import SmallCNNMulticlass
    model = SmallCNNMulticlass()
    x = torch.randn(8, 1, 28, 28)
    feat = model(x)
    assert feat.shape == (8, 128), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()


def test_resnet18_from_scratch_output_shape():
    """Forward returns (B*N, 512) features for 32x32 RGB input."""
    from pca.models import ResNet18FromScratch
    model = ResNet18FromScratch()
    x = torch.randn(4, 3, 32, 32)
    feat = model(x)
    assert feat.shape == (4, 512), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()


def test_patch_encoder_output_shape():
    from pca.models import PatchEncoder
    model = PatchEncoder()
    x = torch.randn(6, 3, 64, 64)
    feat = model(x)
    assert feat.shape == (6, 128), f"got {tuple(feat.shape)}"
    assert torch.isfinite(feat).all()
