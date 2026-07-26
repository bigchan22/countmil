# 4090 Strict-v3 Final Audit

Strict-v3 completion status: complete on the 4x RTX 4090 server.

The strict-v3 source state contains the strict protocol, masked-backbone
utility, stable Gaussian integer-bin evaluation, strict MNIST/CIFAR trainers,
and launch/summarization scripts. The frozen training SHA for reported runs is:

`be75f6f1e9a68411e9014c665ba0201e830c70ee`

Strict-v3 summaries include only runs whose config records:

- protocol tag `strictv3_train_holdout_val_official_test_no_hidden_val`;
- full Git SHA;
- clean-run config hash;
- train/validation/test split hashes;
- train/validation/test manifest hashes;
- aggregate-only validation selection metric.

Historical rows invalidated by test-set checkpoint selection or validation/test
image-pool overlap did not enter the strict-v3 tables.

Validation and checkpointing:

- validation computes aggregate metrics only;
- no hidden instance accuracy, AUC, or digit accuracy is logged on validation;
- final test evaluation is run only after training ends, aggregate validation
  checkpoint selection is complete, and the selected checkpoint is loaded;
- all methods share train/validation/test manifests within each seed.

Completed result evidence:

- MNIST digit sum: MSE, Gaussian-AMLE, FS-Conv; seeds 0,1,2.
- signed MNIST random signs: MSE, Gaussian-AMLE, FS-Conv; seeds 0,1,2.
- signed MNIST cancellation-heavy signs: MSE, Gaussian-AMLE, FS-Conv; seeds
  0,1,2.
- corrected fixed-bag CIFAR-10: CE/KL, FS-Conv count likelihood, full official
  LLP-PVC; seeds 0,1,2.
- official LLP-PVC seed-999 LR grid: `5e-4`, `1e-3`, `2.5e-3`; selected `5e-4`
  by aggregate validation composite count NLL.

Summary files:

- `results/rebuttal/4090_strictv3/scalar_table.md`
- `results/rebuttal/4090_strictv3/signed_table.md`
- `results/rebuttal/4090_strictv3/cifar_table.md`
- `results/rebuttal/4090_strictv3/paired_statistics.md`
- `results/rebuttal/4090_strictv3/run_inventory.csv`

Validation run:

- `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src .venv/bin/pytest -q`
- result: 66 passed.
