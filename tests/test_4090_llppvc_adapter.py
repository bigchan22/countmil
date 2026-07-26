import importlib

import torch

from baselines.llp_pvc import (
    OfficialWarmupCosineLrScheduler,
    official_count_loss,
    official_predict,
)
from countmil.aggregators import AggregatePMF, aggregate_nll, finite_support_convolution_fft_tree
from scripts.rebuttal.train_fixed_cifar10_features import _manifest_hash, _sample_fixed_bags

iter_training_batches = importlib.import_module(
    "scripts.rebuttal.4090_train_official_llppvc_fixed_cifar"
).iter_training_batches


def reference_sigmoid_count_loss(logits, proportions, mask=None):
    probs = torch.sigmoid(logits)
    if mask is None:
        mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    p = probs.transpose(1, 2)
    atoms = torch.stack([1 - p, p], dim=-1)
    zero = probs.new_tensor([1.0, 0.0])
    atoms = torch.where(mask[:, None, :, None], atoms, zero)
    pmfs = finite_support_convolution_fft_tree(atoms, support_min=0).probs
    losses = []
    for b in range(logits.shape[0]):
        n = int(mask[b].sum().item())
        counts = torch.round(proportions[b] * n).long()
        diff = n - int(counts.sum())
        if diff:
            counts[torch.argmax(proportions[b])] += diff
        losses.append(torch.stack([aggregate_nll(AggregatePMF(pmfs[b, c], 0), counts[c]) for c in range(logits.shape[2])]).sum())
    return torch.stack(losses).mean()


def test_official_loss_and_gradient_agree_with_reference():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([[2, 1, 1, 1], [0, 2, 2, 1], [1, 1, 2, 1]], dtype=torch.float64)
    props = counts / 5.0
    got = official_count_loss(logits, props)
    ref = reference_sigmoid_count_loss(logits, props)
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)
    g1 = torch.autograd.grad(got, logits, retain_graph=True)[0]
    g2 = torch.autograd.grad(ref, logits)[0]
    assert (g1 - g2).abs().max().item() < 1e-12


def test_official_predict_uses_softmax_argmax():
    logits = torch.tensor([[[0.0, 2.0, -1.0], [3.0, 1.0, 2.0]]])
    assert official_predict(logits).tolist() == [[1, 0]]
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 2))


def test_warmup_scheduler_moves_lr_without_nan():
    layer = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    sched = OfficialWarmupCosineLrScheduler(opt, max_iter=10, warmup_iter=2, warmup_ratio=0.05, warmup="linear")
    vals = []
    for _ in range(5):
        opt.step()
        sched.step()
        vals.append(opt.param_groups[0]["lr"])
    assert all(v > 0 for v in vals)
    assert all(torch.isfinite(torch.tensor(vals)))


def test_fixed_manifest_stability_and_no_training_labels():
    labels = torch.arange(100) % 10
    bags = _sample_fixed_bags(labels, num_bags=8, bag_size=6, alpha=0.3, seed=3, num_classes=10)
    features = torch.randn(100, 4)
    original = _manifest_hash(bags)
    for epoch in range(3):
        for batch in iter_training_batches(features, bags, batch_size=3, seed=100 + epoch):
            assert "labels" not in batch
            assert set(batch) == {"features", "counts", "proportions", "mask"}
        assert _manifest_hash(bags) == original
