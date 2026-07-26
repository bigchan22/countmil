# Criteo LLP-Bench Protocol Audit

Status: **protocol partially recoverable; experiment not launched**

This branch did not run a Criteo experiment. The exact LLP-Bench Criteo code
path and fixed-size feature-bag construction are recoverable from the official
Google Research source, but the required Criteo input files are not present on
this server and the official training script uses instance-labeled test AUC for
early stopping, which is incompatible with our rebuttal protocol.

## Sources Inspected

- Paper: Brahmbhatt, Pokala, Saket, and Raghuveer, "LLP-Bench: A Large Scale
  Tabular Benchmark for Learning from Label Proportions", arXiv:2310.10096,
  version 2.
- Google Research publication page for the CIKM 2024 version.
- Official source tree:
  `google-research/google-research/LLP_Bench`
- Official source SHA inspected:
  `ec7c3d346277b737bc2decffcd1b533d4b7ec105`

## Recoverable Published Criteo CTR Setup

The official README states that Criteo CTR preprocessing depends on the
DeepGraphLearning `featureRec` Criteo preprocessing script and requires these
three files to be copied into `data/raw_dataset/`:

- `train_x.txt`
- `train_y.txt`
- `train_i.txt`

The official LLP-Bench preprocessing then builds:

- `data/preprocessed_dataset/preprocessed_criteo.csv`

from 13 dense integer features `I1` to `I13`, 26 categorical features `C1` to
`C26`, and binary click label `label`.

The fixed-size Criteo CTR feature-bag script is:

- `LLP_Bench/bag_ds_creation/fixed_size_feature_bag_ds_creation.py`
- launcher: `LLP_Bench/bag_ds_creation/fixed_size_feature_bag_ds_creation.sh`

It uses:

- source dataset: Criteo Kaggle CTR after the above preprocessing;
- feature-pair grouping list from the launcher;
- candidate bag sizes: `64`, `128`, `256`, `512`;
- five splits from `KFold(n_splits=5, shuffle=True, random_state=42)`;
- training fold only for fixed-size bag construction;
- features offset into a shared multi-hot vocabulary;
- bag label: `label_count = sum(label)`;
- fixed-size feature-random construction: group rows by `(C<c1>, C<c2>)`,
  shuffle the groups, concatenate them, then chunk into consecutive fixed-size
  bags.

One exact bounded configuration that would be valid to predeclare is:

- dataset: Criteo CTR;
- feature pair: `C1_C7`;
- split: `0`;
- fixed bag size: `64`;
- train bag file expected by LLP-Bench:
  `data/bag_ds/split_0/train/feature_random_64_C1_C7.ftr`;
- instance test CSV expected by LLP-Bench:
  `data/bag_ds/split_0/test/random.csv` according to the official dataset
  loader for fixed-size feature-random bags.

The last item is an implementation inconsistency: the fixed-size feature-bag
creator writes only the training `.ftr` file, while the training loader expects
the test CSV from the random-bag path for `feature_random_bags=True`.

## Training Protocol in Official Code

The official Criteo CTR training code uses:

- model: multi-hot encoding layer, dense 128 ReLU, dense 64 ReLU, sigmoid output;
- optimizer: Adam with learning rate `1e-5`;
- default epochs: `50`;
- batch size: `8` bags;
- validation batch size: `1024`;
- methods include `dllp_bce`, `dllp_mse`, `easy_llp`, `genbags`, `ot_llp`,
  `soft_erot_llp`, `hard_erot_llp`, `sim_llp`, and `mean_map`.

However, the official trainer monitors `val_auc` and restores best weights using
the instance-labeled validation/test stream. That is not acceptable for the
rebuttal protocol, which requires aggregate-only validation and final
instance-label evaluation only after checkpoint selection.

## Local Data Audit

I searched the repository worktree and `/home/chanhomin` for:

- `*criteo*`
- `train_x.txt`
- `train_y.txt`
- `train_i.txt`

No required Criteo raw, preprocessed, or LLP-Bench bag files were found.

The Criteo worktree also has no `data/` directory. The main strict-v3 worktree
contains image datasets but no Criteo files.

## Decision

No Criteo experiment was launched.

Reasons:

1. The required Criteo CTR files are absent locally.
2. Downloading/recreating the Criteo Kaggle CTR preprocessing pipeline would be
   a large external-data operation and may require Kaggle credentials.
3. The official LLP-Bench training implementation uses instance-labeled
   validation/test AUC for early stopping, so it cannot be used directly under
   the rebuttal leakage policy.
4. The fixed-size feature-bag loader path for test data is ambiguous in the
   official code and would need a documented strict adaptation before training.

This branch therefore records the recoverable protocol and the blocker. It does
not create a custom substitute and does not call any result LLP-Bench.
