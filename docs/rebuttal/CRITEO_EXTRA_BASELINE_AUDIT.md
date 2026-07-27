# Criteo Extra Baseline Audit

- Protocol tag: `criteo_x1_c4_c11_k128_strictv1`.
- Existing DLLP-BCE, DLLP-MSE, EasyLLP, and FS-Conv predictions are reused from strict-v1 without modifying their result directories.
- New GenBags, OT-LLP, group-prior, and supervised-oracle outputs are written only under `results/rebuttal/4090_criteo_extra_baselines/`.
- GenBags follows the Google Research LLP-Bench Gaussian combining-weight definition with block size 4 and 60 generated bags per block; therefore 8 original bags produce 120 generalized bags.
- OT-LLP uses a nonregularized disjoint-bag hard assignment with exactly the observed count positives in each bag.
- Group-prior estimates C4+C11 click rates from aggregate training counts only and backs off to the global aggregate training click rate for unseen groups.
- The supervised oracle uses instance BCE on training labels and is labeled only as an upper reference.
- Validation checkpointing uses aggregate expected-count MAE; validation loaders do not expose hidden instance labels.
- Final instance metrics are computed only on the final test split after checkpoint selection.
- Google Research LLP-Bench source commit recorded for method definitions: `ec7c3d346277b737bc2decffcd1b533d4b7ec105`.
