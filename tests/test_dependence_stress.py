import torch

from experiments.dependence_stress import generate_dependence_data, within_bag_label_correlation


def test_dependence_data_shapes_and_count_consistency():
    data = generate_dependence_data(bags=12, bag_size=5, dim=3, tau=0.5, seed=7)

    assert data.x.shape == (12, 5, 3)
    assert data.z.shape == (12, 5)
    assert torch.equal(data.y, data.z.sum(dim=1))
    assert 0.05 < data.prevalence < 0.95


def test_within_bag_correlation_increases_with_shared_effect():
    base = generate_dependence_data(bags=2000, bag_size=8, dim=4, tau=0.0, seed=11)
    dep = generate_dependence_data(bags=2000, bag_size=8, dim=4, tau=2.0, seed=11)

    assert abs(within_bag_label_correlation(base.z)) < 0.08
    assert within_bag_label_correlation(dep.z) > within_bag_label_correlation(base.z) + 0.05
