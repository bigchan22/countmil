from scripts.train_cifar_histogram import _is_better_summary


def test_cifar_summary_selection_uses_observed_histogram_metric():
    best = {"hist_count_mae": 0.4, "instance_acc": 0.99}
    candidate = {"hist_count_mae": 0.3, "instance_acc": 0.10}

    assert _is_better_summary(candidate, best)


def test_cifar_summary_selection_ignores_hidden_instance_accuracy():
    best = {"hist_count_mae": 0.4, "instance_acc": 0.10}
    candidate = {"hist_count_mae": 0.5, "instance_acc": 0.99}

    assert not _is_better_summary(candidate, best)
