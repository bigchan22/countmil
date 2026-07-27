from __future__ import annotations

import torch

from countmil.criteo.data import CriteoBagDataset
from countmil.criteo.model import CriteoInstanceModel, infer_categorical_layout


def _toy_shard(include_test_labels: bool = False):
    shard = {
        "numeric": torch.randn(2, 3, 13),
        "categorical": torch.randint(0, 20, (2, 3, 26)),
        "counts": torch.tensor([1, 2]),
        "row_ids": torch.arange(6).reshape(2, 3),
        "metadata": {"manifest_hash": "abc", "input_shard_hash": "def"},
    }
    if include_test_labels:
        shard["hidden_labels"] = torch.tensor([[0, 1, 0], [1, 1, 0]])
    return shard


def test_train_dataset_hides_labels():
    ds = CriteoBagDataset(_toy_shard(), expose_hidden_labels=False)
    item = ds[0]
    assert "hidden_labels" not in item
    assert set(item) == {"numeric", "categorical", "counts", "row_ids"}


def test_test_dataset_can_expose_labels():
    ds = CriteoBagDataset(_toy_shard(True), expose_hidden_labels=True)
    assert "hidden_labels" in ds[0]


def test_categorical_layout_shared_and_field_specific():
    shared, cards = infer_categorical_layout([0, 10], [9, 19])
    assert shared
    assert cards == [20, 20]
    shared, cards = infer_categorical_layout([0, 0], [9, 19])
    assert not shared
    assert cards == [10, 20]


def test_batch_and_single_forward_agree_eval_mode():
    torch.manual_seed(0)
    model = CriteoInstanceModel(categorical_cardinalities=[50] * 26, shared_categorical=True)
    model.eval()
    numeric = torch.randn(4, 13)
    categorical = torch.randint(0, 50, (4, 26))
    batch = model(numeric, categorical)
    single = torch.stack([model(numeric[i : i + 1], categorical[i : i + 1])[0] for i in range(4)])
    assert torch.allclose(batch, single)


def test_permuting_bags_does_not_change_eval_probabilities():
    torch.manual_seed(1)
    model = CriteoInstanceModel(categorical_cardinalities=[40] * 26, shared_categorical=True)
    model.eval()
    numeric = torch.randn(2, 5, 13)
    categorical = torch.randint(0, 40, (2, 5, 26))
    flat = model(numeric.reshape(-1, 13), categorical.reshape(-1, 26)).reshape(2, 5)
    perm = torch.tensor([1, 0])
    permuted = model(numeric[perm].reshape(-1, 13), categorical[perm].reshape(-1, 26)).reshape(2, 5)
    assert torch.allclose(flat[perm], permuted)

