# CountMIL Experiment Report

Generated: 2026-05-06 UTC.

This file records the current experiment status, main hyperparameters, and aggregated results used for the NeurIPS draft. Instance labels are not used for training in these experiments; they are used only for evaluation.

## Common Setup

- Code root: `~/countmil`
- Dataset root: `data/`
- Bag sampling: on the fly; bags are not pregenerated to disk.
- Seeds: usually `0, 1, 2` unless otherwise noted.
- Test bags: generally `1000`.
- MNIST-family model: `MNISTDigitClassifier` or Shukla-style LeNet selector from `src/countmil/models/mnist_cnn.py`.
- MNIST-family optimizer: Adam, `lr=5e-4`, `weight_decay=1e-4`, betas `(0.9, 0.999)`.
- MNIST-family epochs for main grids: `50`.
- 4090 large-bag batch size: `192` for several MNIST histogram jobs.
- CIFAR small-CNN/resnet histogram optimizer defaults: Adam unless Ma-style options are enabled.
- Ma-style CIFAR protocol options now supported: augmentation, SGD, momentum, cosine or Ma cosine schedule, pretrained ResNet-18.

## Runtime and Exactness

Purpose: verify finite-support convolution exactness and demonstrate GPU/FFT speed for large one-vs-rest count PMFs.

Files:

- `scripts/bench_exactness_runtime.py`
- `scripts/bench_multiclass_ovr_fft.py`
- `results/bench_multiclass_ovr_fft_cuda_20260505/multiclass_ovr_fft_cuda_s0.csv`

Key CUDA benchmark: batch size `256`, classes `10`.

| Bag size | CPU DP seconds | GPU FFT-tree seconds | Max abs error |
|---:|---:|---:|---:|
| 64 | 8.246 | 0.00097 | 3.4e-7 |
| 128 | 17.230 | 0.00097 | 4.2e-7 |
| 256 | 36.785 | 0.00121 | 5.1e-7 |
| 512 | 54.040 | 0.00134 | 6.6e-7 |

Interpretation: the GPU FFT-tree backend is numerically close to CPU DP and far faster in this batched multiclass one-vs-rest setting. This supports the computational story, but claims should be phrased as practical GPU/FFT speedup rather than a universal end-to-end training speedup.

## Binary MNIST CountMIL

Purpose: connect to Shukla-style binary count loss and test hidden instance recovery.

Training:

- Dataset: MNIST bags.
- Label: count of target digit positives.
- Objective: exact Bernoulli count negative log likelihood.
- Model: Shukla-style LeNet selector.
- Epochs: `50`.
- Train bags: `1000` or `5000`.
- Bag sizes: mean `10` std `2`; mean `50` std `10`.
- Seeds: `0,1,2`.

Results:

| Bag/train | Bag AUC | Instance AUC | Count MAE |
|---|---:|---:|---:|
| 10/1000 | 1.000 | 0.999 | 0.115 |
| 10/5000 | 1.000 | 0.999 | 0.090 |
| 50/1000 | 1.000 | 0.998 | 0.686 |
| 50/5000 | 1.000 | 0.999 | 0.460 |

Interpretation: binary count likelihood is a strong sanity check and baseline connection, not the novelty claim. It recovers excellent bag and instance ranking, while count calibration is harder for larger bags.

## MNIST Digit-Sum Likelihood

Purpose: test scalar ordinal aggregate labels beyond binary counts.

Training:

- Dataset: MNIST digit-sum bags.
- Label: scalar sum of hidden digit values.
- Model: MNIST digit classifier.
- Objective: exact finite-support sum likelihood over digits `0..9`.
- Epochs: `50`.
- Train bags: `1000` or `5000`.
- Bag sizes: mean `10` std `2`; mean `50` std `10`.
- Seeds: `0,1,2`.

Results:

| Bag/train | Instance digit accuracy | Sum MAE |
|---|---:|---:|
| 10/1000 | 0.982 | 0.612 |
| 10/5000 | 0.992 | 0.277 |
| 50/1000 | 0.693 | 4.901 |
| 50/5000 | 0.887 | 2.229 |

Interpretation: exact scalar-sum likelihood is strong for small bags and remains useful for larger bags, but large-bag scalar sums are substantially harder.

## Digit-Sum Expected-Value Baselines

Purpose: test whether matching only `sum_i E[z_i]` is enough.

Training:

- Server: 3090.
- Dataset: MNIST digit-sum bags.
- Objectives: MSE, MAE, Huber on expected sum.
- Model: MNIST digit classifier.
- Epochs: `50`.
- Status as of 2026-05-05 14:00 UTC: partial; long run still expected to take days.

Completed partial results:

| Bag/train | Loss | Seeds done | Expected MAE | Rounded acc. | Rounded MAE | Digit acc. |
|---|---:|---:|---:|---:|---:|---:|
| 10/1000 | Huber | 3 | 2.330 | 0.136 | 2.323 | 0.278 |
| 10/1000 | MAE | 3 | 2.337 | 0.141 | 2.323 | 0.286 |
| 10/1000 | MSE | 3 | 2.535 | 0.133 | 2.527 | 0.226 |
| 10/5000 | Huber | 1 | 0.548 | 0.784 | 0.474 | 0.988 |
| 10/5000 | MAE | 1 | 0.430 | 0.835 | 0.412 | 0.989 |
| 10/5000 | MSE | 1 | 1.022 | 0.386 | 0.981 | 0.730 |

Interpretation: expected-value baselines are weak in the low-data scalar-sum setting, but MAE/Huber become competitive for small bags with more training bags. The defensible claim is data efficiency and likelihood fidelity, not universal dominance.

## Atomic-Support Generality: UltraMNIST-Style Bags

Purpose: check that the same finite-support convolutional likelihood works for a variable-size RGB patch setting, not only ordinary MNIST digit-sum bags.

Server: 4090.

Training:

- Dataset: UltraMNIST-style synthetic RGB digit-patch bags from `UltraMNISTOrdinalSumBags`.
- Label: scalar sum of hidden digit labels.
- Objective: exact finite-support sum NLL over atoms `{0,...,9}`.
- Model: `PatchOrdinalClassifier`.
- Bag size range: `3` to `5`.
- Train bags: `800`.
- Test bags: `300`.
- Epochs: `60`.
- Seeds: `0,1,2,3,4`.
- Result directory: `results/atomic_sum_server4090_ultramnist_20260506_1758/`.
- Aggregated CSV: `results/atomic_sum_server4090_ultramnist_20260506_1758/aggregate.csv`.

Completed status: `5/5` seeds, no tracebacks/OOM/killed errors. GPUs were idle after completion.

Best-checkpoint means over five seeds:

| Metric | Mean | Std. dev. |
|---|---:|---:|
| Sum accuracy | 0.9007 | 0.0092 |
| Instance accuracy | 0.9743 | 0.0031 |
| Sum MAE | 0.3407 | 0.0379 |
| Expected-sum MAE | 0.4626 | 0.0317 |
| NLL | 0.3270 | 0.0155 |

Tail-5 means over five seeds:

| Metric | Mean | Std. dev. |
|---|---:|---:|
| Sum accuracy | 0.8944 | 0.0078 |
| Instance accuracy | 0.9726 | 0.0024 |
| Expected-sum MAE | 0.4877 | 0.0251 |
| NLL | 0.3546 | 0.0143 |

Interpretation: this is a strong positive robustness result for the A1/UltraMNIST-style aggregate-sum story. The model recovers hidden instance labels at about `97%` accuracy while seeing only aggregate sums during training. The result should be described as an UltraMNIST-style synthetic patch benchmark unless a real UltraMNIST dataset is substituted.

## Signed CountMIL

Purpose: test signed aggregates and cancellation.

Training:

- Dataset: MNIST signed count bags.
- Label: `Y=sum_i s_i z_i`, with known signs `s_i in {-1,+1}`.
- Model: Shukla-style LeNet selector.
- Objective: exact signed count likelihood via finite-support convolution.
- Epochs: `50`.
- Train bags: `1000` or `5000`.
- Bag sizes: mean `10` std `2`; mean `50` std `10`.
- Sign modes: random and cancellation-heavy.
- Seeds: `0,1,2`.

Results:

| Bag/train | Signs | Instance AUC | Signed MAE | Zero-count accuracy |
|---|---|---:|---:|---:|
| 10/1000 | random | 0.995 | 0.095 | 0.933 |
| 10/1000 | cancellation | 0.998 | 0.071 | 0.965 |
| 10/5000 | random | 0.999 | 0.045 | 0.969 |
| 10/5000 | cancellation | 0.999 | 0.046 | 0.979 |
| 50/1000 | random | 0.999 | 0.243 | 0.772 |
| 50/1000 | cancellation | 0.969 | 1.689 | 0.420 |
| 50/5000 | random | 0.999 | 0.198 | 0.814 |
| 50/5000 | cancellation | 0.993 | 1.118 | 0.591 |

Interpretation: signed aggregates are a useful novelty case. Cancellation-heavy large bags are genuinely ambiguous, but instance ranking remains strong with enough data.

## FashionMNIST Robustness

Purpose: test robustness outside MNIST digits and compare hidden instance recovery against attention MIL.

Server: A5000.

Training:

- Dataset: FashionMNIST.
- Grid size: 28 jobs.
- Seeds: `0,1`.
- Bag sizes: mean `10` std `2`; mean `50` std `10`.
- Models: count-likelihood convolution, attention MIL, gated attention MIL, digit/category sum and histogram variants.

Completed status: `28/28` jobs, no errors.

Binary CountMIL vs attention:

| Bag size | Method | Bag AUC | Bag acc. | Count MAE | Instance AUC |
|---:|---|---:|---:|---:|---:|
| 10 | conv | 0.9988 | 0.9605 | 0.183 | 0.9973 |
| 10 | attention | 0.9980 | 0.9725 | - | 0.7034 |
| 10 | gated attention | 0.9979 | 0.9775 | - | 0.7221 |
| 50 | conv | 0.9987 | 0.7720 | 0.886 | 0.9980 |
| 50 | attention | 0.9983 | 0.9880 | - | 0.7434 |
| 50 | gated attention | 0.9984 | 0.9850 | - | 0.7146 |

Other FashionMNIST results:

| Experiment | Bag size | Main result |
|---|---:|---|
| Digit/category sum | 10 | sum MAE 3.362, instance category acc. 0.4718 |
| Digit/category sum | 50 | sum MAE 7.869, instance category acc. 0.4211 |
| Histogram LLP | 10 | count MAE 0.321, instance category acc. 0.8184 |
| Histogram LLP | 50 | count MAE 0.903, instance category acc. 0.7808 |
| Signed CountMIL random | 10 | signed MAE 0.113, signed acc. 0.8875, instance AUC 0.9977 |
| Signed CountMIL cancellation | 10 | signed MAE 0.104, signed acc. 0.8970, instance AUC 0.9979 |
| Signed CountMIL random | 50 | signed MAE 0.381, signed acc. 0.6545, instance AUC 0.9979 |
| Signed CountMIL cancellation | 50 | signed MAE 0.951, signed acc. 0.4355, instance AUC 0.9961 |

Interpretation: FashionMNIST is best used as a robustness/instance-recovery result, not as the headline ordinal-sum result.

## MNIST Histogram LLP and PVC

Purpose: compare ordinary proportion matching against one-vs-rest count likelihood/PVC under multiclass histogram supervision.

Training:

- Server: 4090.
- Dataset: MNIST digit histogram bags.
- Hidden labels: digit labels used only for evaluation.
- LLP objectives: CE, KL, MSE proportion matching.
- PVC objective: one-vs-rest Bernoulli count likelihood per digit class.
- Model: MNIST digit classifier.
- Optimizer: Adam, `lr=5e-4`, `weight_decay=1e-4`.
- Epochs: `50`.
- Batch size: `192` on 4090 grid.
- Test bags: `1000`.
- Bag sizes: mean `10` std `2`; mean `50` std `10`.
- Train bags: `1000`, `5000`.
- Seeds: `0,1,2`.
- Configs:
  - `configs/histogram_llp/server4090_llp_20260505_0733/manifest.tsv`
  - `configs/histogram_pvc/server4090_pvc_20260505_0751/manifest.tsv`
- Results:
  - `results/server4090_histogram_llp/`
  - `results/server4090_histogram_pvc/`

Completed status:

- Histogram LLP: `36/36`.
- Histogram PVC: `12/12`.
- Logs: no tracebacks, OOMs, killed jobs, or missing-file errors found.
- GPUs idle after completion.

Aggregated results:

| Bag/train | Objective | Seeds | Count MAE | Prop. MAE | Digit acc. | Hist exact |
|---|---|---:|---:|---:|---:|---:|
| 10/1000 | CE | 3 | 0.0804 | 0.00807 | 0.9725 | - |
| 10/1000 | KL | 3 | 0.0804 | 0.00807 | 0.9725 | - |
| 10/1000 | MSE | 3 | 0.1814 | 0.01846 | 0.9548 | - |
| 10/1000 | PVC | 3 | 0.0320 | 0.00316 | 0.9827 | 0.838 |
| 10/5000 | CE | 3 | 0.0303 | 0.00300 | 0.9890 | - |
| 10/5000 | KL | 3 | 0.0303 | 0.00300 | 0.9890 | - |
| 10/5000 | MSE | 3 | 0.1292 | 0.01312 | 0.9743 | - |
| 10/5000 | PVC | 3 | 0.0159 | 0.00150 | 0.9917 | 0.918 |
| 50/1000 | CE | 3 | 0.3756 | 0.00768 | 0.9640 | - |
| 50/1000 | KL | 3 | 0.3756 | 0.00768 | 0.9640 | - |
| 50/1000 | MSE | 3 | 1.6825 | 0.03453 | 0.1156 | - |
| 50/1000 | PVC | 3 | 0.1370 | 0.00276 | 0.9847 | 0.436 |
| 50/5000 | CE | 3 | 0.1838 | 0.00373 | 0.9880 | - |
| 50/5000 | KL | 3 | 0.1838 | 0.00373 | 0.9880 | - |
| 50/5000 | MSE | 3 | 1.6837 | 0.03455 | 0.1156 | - |
| 50/5000 | PVC | 3 | 0.0801 | 0.00161 | 0.9915 | 0.659 |

Interpretation:

- CE/KL proportion matching are strong baselines.
- PVC is consistently better on count MAE, proportion MAE, and hidden digit accuracy.
- MSE proportion matching collapses for large bags.
- The result supports the claim that likelihood fidelity matters even for rich histogram labels.

## CIFAR-100 Coarse Histogram Pilot

Purpose: exploratory multiclass histogram setting beyond MNIST-family images.

Server: A5000.

Training:

- Dataset: CIFAR-100 coarse labels.
- Classes: 20 coarse classes.
- Backbone: small CNN.
- Bag size: mean `50`.
- Train bags: `1000`.
- Seeds: `0,1`.
- Objectives: PVC, KL, MSE.

Completed status: `6/6`, no errors.

| Objective | Count MAE | Prop. MAE | Instance acc. |
|---|---:|---:|---:|
| KL | 1.242 | 0.0259 | 0.299 |
| MSE | 1.200 | 0.0251 | 0.108 |
| PVC | 1.238 | 0.0253 | 0.308 |

Interpretation: above chance and useful as a pipeline check, but not strong enough as a headline result.

## CIFAR-10 ResNet-18 From Scratch

Purpose: direct LLP/PVC-style CIFAR-10 comparison without pretrained features.

Server: A5000.

Training:

- Dataset: CIFAR-10.
- Backbone: ResNet-18 from scratch.
- Objectives: KL, MSE, PVC.
- Bag sizes: `16`, `64`, `128`.
- Train bags: `1000`.
- Seeds: `0,1`.
- Status: `18/18` complete.

Results:

| Bag size | Objective | Count MAE | Prop. MAE | Instance acc. |
|---:|---|---:|---:|---:|
| 16 | KL | 0.806 | 0.0522 | 0.513 |
| 16 | MSE | 0.825 | 0.0533 | 0.476 |
| 16 | PVC | 0.838 | 0.0524 | 0.527 |
| 64 | KL | 1.856 | 0.0292 | 0.490 |
| 64 | MSE | 2.013 | 0.0316 | 0.270 |
| 64 | PVC | 1.855 | 0.0290 | 0.496 |
| 128 | KL | 2.860 | 0.0224 | 0.457 |
| 128 | MSE | 2.829 | 0.0222 | 0.226 |
| 128 | PVC | 2.707 | 0.0211 | 0.488 |

Interpretation: from-scratch ResNet-18 is far weaker than Ma/LLP-PVC reported results, showing the backbone initialization matters.

## CIFAR-10 Ma-Style Pretrained ResNet-18 PVC

Purpose: match Ma/LLP-PVC CIFAR-10 backbone protocol more closely.

Server: A5000.

Training:

- Dataset: CIFAR-10.
- Backbone: ImageNet-pretrained ResNet-18.
- Objective: PVC / one-vs-rest count likelihood.
- Augmentation: random crop with padding and horizontal flip.
- Optimizer: SGD.
- Momentum: `0.9`.
- Weight decay: `1e-4`.
- Learning rate: `5e-4`.
- Scheduler: Ma-style cosine `eta_k = eta_0 cos(7 pi k / 16K)`.
- Epochs: `500`.
- Bag sizes completed: `16`, `64`.
- Bag size not completed: `128`.
- Train bags: `1000`.
- Seeds: `0,1`.
- Checkpoint-selection caveat: source commit `c189355` now selects CIFAR best checkpoints by observed `hist_count_mae`, not hidden `instance_acc`. Completed JSONs produced before this fix should be interpreted using final epoch metrics or by recomputing best-by-hist from `metrics.jsonl`.
- Caveat: current code scales CIFAR images to `[0,1]`; exact normalization used by Ma should be verified before claiming exact protocol match.

Completed pretrained PVC results:

| Setting | Final inst. acc. | Final hist MAE | Best-by-hist epoch | Best-by-hist inst. acc. | Best-by-hist hist MAE |
|---|---:|---:|---:|---:|---:|
| n16 seed 0 | 0.8996 | 0.2662 | 16 | 0.8994 | 0.2594 |
| n16 seed 1 | 0.9052 | 0.2539 | 25 | 0.9083 | 0.2418 |
| n64 seed 0 | 0.9126 | 0.6878 | 18 | 0.9118 | 0.6611 |
| n64 seed 1 | 0.9163 | 0.6691 | 22 | 0.9150 | 0.6469 |

Mean final accuracies:

| Bag size | Train bags | Final inst. acc. | Best-by-hist inst. acc. | Best-by-hist hist MAE |
|---:|---:|---:|---:|---:|
| 16 | 1000 | 0.9024 | 0.9038 | 0.2506 |
| 64 | 1000 | 0.9145 | 0.9134 | 0.6540 |

Ma/LLP-PVC reported CIFAR-10 reference accuracies:

| Bag size | Reported instance acc. |
|---:|---:|
| 16 | 0.845 |
| 64 | 0.789 |
| 128 | 0.760 |

Interpretation:

- The matched pretrained PVC protocol is strong on CIFAR-10 and exceeds the reported LLP-PVC reference accuracies for `n=16` and `n=64`.
- The earlier apparent `n64 > n16` advantage is not a fair fixed-instance-budget conclusion: `n16/train1000` sees `16,000` image instances per epoch, while `n64/train1000` sees `64,000`.
- Therefore, the safe claim is that count/histogram supervision benefits from aggregate-labeled image volume under a strong pretrained representation.
- Do not claim larger bags are intrinsically better or more scalable than fully supervised learning without matched annotation-budget baselines.

### CIFAR-10 Fixed-Instance-Budget Diagnostic

Purpose: compare `n16/train1000` against `n64/train250`, so both settings expose `16,000` sampled image instances per epoch.

Server: A5000.

Run:

- tmux: `countmil_a5000_cifar10_ma_pvc_n64_train250`
- Manifest: `configs/cifar_histogram/serverA5000_cifar10_ma_pvc_n64_train250_20260506_0430/manifest.tsv`
- Results: `results/serverA5000_cifar10_ma_pvc_n64_train250`
- Logs: `logs/serverA5000_cifar10_ma_pvc_n64_train250`
- GPUs: `0,1`
- Epochs: `500`
- Batch size: `8`
- Test bags: `1000`

Completed status at 2026-05-06:

| Setting | Selection | Seed 0 inst. acc. | Seed 1 inst. acc. | Mean inst. acc. | Mean hist MAE |
|---|---|---:|---:|---:|---:|
| n64 train250 | final epoch | 0.8033 | 0.8035 | 0.8034 | 1.115 |
| n64 train250 | best by hist MAE | 0.8091 | 0.8035 | 0.8063 | 1.013 |

Comparison at matched or unmatched sampled image exposure:

| Setting | Total train instances/epoch | Mean final inst. acc. |
|---|---:|---:|
| n16 train1000 | 16,000 | 0.9024 |
| n64 train250 | 16,000 | 0.8034 |
| n64 train1000 | 64,000 | 0.9145 |

This completed diagnostic is far below the completed `n16/train1000` final accuracy despite matched sampled image instances per epoch.
It also separates the two effects that were confounded in the earlier comparison: large bags can work well when they expose more total images/aggregate observations, but larger bag size alone is not better under fixed total instance exposure.
The diagnostic therefore argues against claiming intrinsic large-bag superiority.

## Current Recommended Paper Claims

- Do not claim binary count loss is new.
- State that binary Bernoulli CountMIL recovers Shukla-style count probability.
- Use signed and ordinal aggregates as the main novelty beyond binary counts.
- Use posterior marginals as an inference/diagnostic tool, not as a central EM training method.
- Say exact scalar-sum likelihood is more data-efficient than expected-value matching in low-data settings.
- Say histogram LLP labels are rich and ordinary CE/KL baselines are strong.
- Say PVC/count likelihood improves over CE/KL proportion matching in completed MNIST histogram experiments, especially for larger bags.
- Treat CIFAR-10 pretrained ResNet-18 as strong for completed `n=16/train1000` and `n=64/train1000`, but the completed `n=64/train250` fixed-instance-budget diagnostic argues against claiming larger bags are intrinsically better. Avoid scalability claims until `n=128` and matched-budget baselines finish.
