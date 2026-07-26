# A5000 SVHN scalar-sum protocol audit

This audit covers the submitted-protocol SVHN integer digit-sum experiment requested for the NeurIPS 2026 rebuttal.

## Submitted-protocol configuration found locally

Existing A5000 run metadata/logs identify the submitted SVHN scalar-sum protocol:

- Dataset/task: SVHN integer digit sum (`experiment: svhn_sum`).
- Data root: `data/`.
- Local data files:
  - `data/train_32x32.mat`
  - `data/test_32x32.mat`
- SVHN train MD5: `e26dedcc434d2e4c54c9b2d4a06d8373`.
- SVHN test MD5: `eb5a983be6a315427106f1b164d9cef3`.
- SVHN train SHA256: `435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8`.
- SVHN test SHA256: `cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b`.
- Bag sampler: `SVHNOrdinalSumBags`, sampled on the fly from official split tensors.
- Bag size: `bag_size_mean=10.0`, `bag_size_std=2.0`, clipped by `bag_size_min=5`, `bag_size_max=15`.
- Training bags: `5000`.
- Test bags in existing submitted-protocol runs: `600`.
- Validation bags: not present in older runs; rebuttal reruns add deterministic aggregate-only validation bags.
- Split seeds:
  - Train bags: `seed`.
  - Test bags in old scripts: `seed + 10000`.
  - New rebuttal validation bags: `seed + 20000`.
- Model: `resnet18`.
- Pretraining: `pretrained=True`, torchvision `ResNet18_Weights.IMAGENET1K_V1`.
- Input normalization/resizing: `CIFARResNet18Classifier` uses ImageNet transforms and resizes to `224x224` when pretrained.
- Augmentation: train split only, random reflected crop plus brightness/contrast jitter from `_svhn_train_transform`; disabled for validation/test.
- Optimizer: Adam.
- Learning rate: `1e-4`.
- Weight decay: `5e-4`.
- Epochs: `40`.
- Batch size: `16`.
- Original old FS-Conv command shape:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/train_atomic_sum.py \
    --experiment svhn_sum --seed SEED --epochs 40 \
    --train-bags 5000 --test-bags 600 --batch-size 16 \
    --augment --pretrained --device cuda \
    --results-dir RESULTS --run-root RUNS
  ```
- Original old MSE command shape was the same protocol with MSE objective support from the earlier A5000 branch.

## Existing local results

Existing local submitted-protocol-style results:

- FS-Conv pretrained seed 0/1/2:
  - `results/serverA5000_svhn_a1_train5000_pretrained_20260506_2129/svhn_sum_pca_s0.json`
  - `results/serverA5000_svhn_a1_train5000_pretrained_20260506_2129/svhn_sum_pca_s1.json`
  - `results/serverA5000_svhn_a1_train5000_pretrained_20260506_2129/svhn_sum_pca_s2.json`
- MSE pretrained seed 0/1:
  - `results/serverA5000_svhn_a1_train5000_mse_pretrained_20260507_0429/svhn_sum_mse_s0.json`
  - `results/serverA5000_svhn_a1_train5000_mse_pretrained_20260507_0429/svhn_sum_mse_s1.json`

Checkpoints exist under the corresponding `runs/serverA5000_*` directories, but these old results are not reused for the strict rebuttal table.

## Reuse decision

Existing seed-0 FS-Conv/MSE runs are **not reused** for the mandatory rebuttal comparison.

Reasons:

- The old scripts selected checkpoints using test-set aggregate metrics, not an independent validation set.
- The old summaries did not save the rebuttal-required raw per-bag aggregate predictions.
- The old summaries did not save the rebuttal-required raw per-instance predictions on unique test images.
- The old summaries did not save fixed train/validation/test bag manifests and manifest hashes.
- The old runs were produced from an earlier Git SHA (`56df92a...`), not the synchronized rebuttal baseline.

Therefore seeds `0,1,2` for FS-Conv, MSE, and Gaussian-AMLE will all be rerun on the synchronized A5000 branch with aggregate-only validation selection and fixed manifest hashes.

Hidden instance digit labels will be used only for final evaluation metrics, never for training, validation checkpoint selection, or early stopping.
