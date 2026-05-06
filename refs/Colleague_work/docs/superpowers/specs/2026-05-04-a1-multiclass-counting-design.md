# A1: Atomic PMF Generality (Multi-class Ordinal Counting) — Design

| | |
|---|---|
| **Date** | 2026-05-04 |
| **Status** | Draft (awaiting approval) |
| **Owner** | Byung-Hak Hwang (KIAS) |
| **Project** | Probabilistic Convolutional Aggregator (PCA) — NeurIPS 2026 |
| **Track** | `03_evaluation_strategy.md` §4 Track 1 (Empirical core, week 2) |
| **Related notes** | `docs/notes/2026-05-04-a1-analysis.md`; `01_pca_strengthening.md §A1`; `README.md §2 (priority Main)` |
| **Depends on** | B1 verification artifacts (`pca/losses.py:forward_conv`, `pca/models.py:SmallCNN`, `tests/test_losses.py`) committed at `fc1c9da` |

---

## 1. Context and motivation

### 1.1 Where this sits in the project

The Probabilistic Convolutional Aggregator (PCA) computes the exact PMF of a sum of independent atomic PMFs via 1D convolution and trains under aggregate-only supervision via marginal NLL on $-\log P(\sum z_i = Y)$. The pure-methodology pivot for NeurIPS 2026 has chosen A1 as the **Main** contribution after B1 was demoted to Low (week 1 verdict, 2026-05-04).

A1's claim: the convolutional view exposes an **algebraic uniformity** across atom shapes — Bernoulli, multi-class ordinal categorical, signed, and multiplicity atoms are unified by a single algorithm parametrized by kernel choice. Shukla 2023's binary DP (imported from SIMPLE Prop. 1, Ahmed et al. ICLR 2023) is the special case $S=2$. This spec verifies the multi-class ordinal and signed cases empirically.

### 1.2 The A1 idea — Framing (B): algebraic unification

| Atom shape | Support $\{...\}$ | Support size $S$ | Bag sum range |
|---|---|---|---|
| Bernoulli | $\{0, 1\}$ | 2 | $[0, N]$ |
| Multi-class ordinal (categorical) | $\{0, 1, \ldots, K\}$ | $K{+}1$ | $[0, NK]$ |
| Signed | $\{-1, 0, +1\}$ | 3 | $[-N, +N]$ |
| Multiplicity (per-instance) | $\{0, m_i\}$ | $m_i{+}1$ | varies |

The conv view: bag PMF = $\mathbf d_1 * \cdots * \mathbf d_N$ where $\mathbf d_i$ is the per-instance atomic PMF. **Atom shape change requires only kernel-length change**, not algorithm change. PCA's contribution is this algebraic abstraction, not a multi-class DP rederivation.

### 1.3 Lineage — SIMPLE, not SPL

Shukla 2023 imports the binary count DP $P(\sum z_i = k)$ in $O(nk)$ from **SIMPLE** (Ahmed/Zeng/Niepert/Van den Broeck, ICLR 2023, arXiv 2210.01941), not SPL. SPL (Ahmed/Yousri/Niepert/Vergari/Van den Broeck, NeurIPS 2022, arXiv 2206.00426) is a separate framework for generic propositional-logic-constraint conditioning over binary label vectors and does not natively support multi-class atomic PMFs. PCA generalizes the SIMPLE/Shukla binary count primitive by atomic-PMF algebra; SPL stands as a broader related framework. See `docs/notes/2026-05-04-a1-analysis.md §3` for SPL relationship and Related Work treatment.

### 1.4 Multi-class scope — ordinal counting only

Throughout this spec and the paper, "multi-class" denotes **ordinal counting variables** $z_i \in \{0, \ldots, K\}$ where $\sum_i z_i$ has numeric meaning (e.g., digit-sum). Nominal multi-class targets (ImageNet, CIFAR raw labels) are out of scope — the conv view requires the support set to admit a sum operation. CIFAR re-configurations (re-labeling by ordinal attribute, CIFAR-stack synthesis) were considered and excluded; see `docs/notes/2026-05-04-a1-analysis.md §7` for justification.

### 1.5 Why this verification is necessary

The algebraic claim is mathematically clean (atom shape change = kernel-length change). The empirical claim is that PCA's exact bag PMF gives a measurable advantage over baselines that approximate the bag distribution differently (mean-pool, label-proportion average, CLT Gaussian, attention pooling, DeepSets). Without empirical evidence, A1 reduces to "we have a cleaner notation" — insufficient for NeurIPS 2026.

Falsifier: if PCA cannot statistically beat the strongest of 5 baselines on NLL or ECE on at least 2 of 3 multi-class image datasets (after saturation insurance), A1 is demoted. See Section 2.3.

---

## 2. Hypotheses and falsifier

### 2.1 Hypotheses (saturation-aware)

NLL and ECE are the **primary judging metrics**; top-1 accuracy on bag sum and MAE are reported but not used for verdict (saturation regime concern; see §2.4).

**H1 (primary, NLL):**
> PCA's test NLL is below the best of {2b CLT Gaussian, 3a Attention, 3b DeepSets} on **at least 2 of 3** multi-class datasets {MNIST-sum, SVHN-sum, UltraMNIST}, paired t-test $p < 0.05$ over 5 seeds (3 for UltraMNIST), absolute $|\Delta\text{NLL}| \ge 0.02$ nats.

**H2 (atom-shape generality, must-pass):**
> PCA produces a valid PMF and outperforms random on every atom shape:
> - Binary (B1 reuse): instance AUC > 0.9 (already verified, citation only).
> - Multi-class $K{=}9$ on each of MNIST-sum, SVHN-sum, UltraMNIST: top-1 acc on bag sum > 0.30, NLL finite.
> - Signed (MNIST-signed-sum toy): top-1 acc on bag sum > 0.20, NLL finite.

**H3 (calibration, secondary):**
> PCA ECE $\le 0.7 \times \min$(distribution-native baseline ECE) on **at least 2 of 3** multi-class datasets, paired comparison.

**H4 (supporting, not gating):**
> Top-1 accuracy on bag sum and MAE are reported. Used in narrative only when H1 is borderline.

### 2.2 Statistical considerations

- 5 seeds for MNIST-sum, SVHN-sum, MNIST-signed; 3 seeds for UltraMNIST (wall-clock budget).
- Paired t-test (`scipy.stats.ttest_rel`): same seed → same data and model init, only loss/architecture differs across methods.
- $|\Delta\text{NLL}| \ge 0.02$ nats threshold: B1 retrospective showed 0.01 is noise level; 0.02 is detectable with 5-seed paired test.
- ECE bin count: 15 (CIFAR/ImageNet calibration standard).
- Multi-hypothesis correction: H1 defined as "≥ 2 of 3 datasets" controls family-wise error implicitly.

### 2.3 Falsifier — verdict logic

| Condition | Verdict | Action |
|---|---|---|
| H2 fail anywhere | `ALGEBRA_BROKEN` | Code/spec bug. Stop, debug, do not interpret as scientific result. |
| H2 pass + H1 pass (default) | **`A1_CONFIRMED`** | Paper main: "atom-shape generality + NLL advantage." H3 reinforces. Proceed to Phase 2 (B2 bolt-on, separate spec). |
| H2 pass + H1 fail (default) + saturation signal | `RUN_HARD_PRESET` | Apply hard preset (per-class-cap=50 + noise σ=0.1 for MNIST/signed; cap=1000 + noise σ=0.05 for SVHN) to affected dataset(s). Re-evaluate H1. ~5–15 h additional. |
| H2 pass + H1 fail (hard) + H3 pass | `A1_CALIBRATION_FOCUS` | Pivot paper headline to "atom-shape generality + calibration advantage." H1 moved to appendix. (Pre-registered pivot — interpretation re-discussed when results arrive.) |
| H2 pass + H1 fail (hard) + H3 fail | **`DEMOTE_A1`** | A1 → Low priority. Update `README.md` priority table. Begin alternative track immediately. |

### 2.4 Saturation auto-detection (`analyze_a1.py`)

After default-config results arrive, `analyze_a1.py` flags saturation if:
```
saturation_signal = (
    max(distribution_native_baseline_nll_means) - min(...) < 0.05
    OR max(top1_acc_means_across_methods) > 0.97
)
```
On signal: emit `RUN_HARD_PRESET`, print exact CLI for hard-preset rerun. Hard preset per Section 4.

### 2.5 Signed toy verdict (separate)

MNIST-signed-sum is a toy demonstration of the signed atom case. Verdict scope:
- H2 signed sub-condition only (top-1 acc > 0.20, NLL finite).
- Compared against mean-pool baseline only (1 baseline).
- Reported as 1 table + 1 atom-shape figure.
- Does not enter `DEMOTE_A1` logic — its purpose is algebraic-claim demonstration.

---

## 3. Approach and method choices

### 3.1 Multi-class DP ≡ conv equivalence — body proposition (not baseline)

The paper §3 contains a short proposition:

> **Proposition (informal).** *For independent atomic PMFs $\mathbf d_1, \ldots, \mathbf d_N$ over support $\{0, \ldots, K\}$, the bag PMF $P(\sum_i z_i = s)$ computed by recursive convolution and by the natural multi-class generalization of SIMPLE Prop. 1's DP are pointwise equal. Both run in $O(NK \cdot NK)$.*

This proposition is in the body (with proof sketch); full proof in appendix. We do **not** implement multi-class DP as a baseline — the proposition discharges the equivalence; comparing computationally identical methods would dilute the contribution narrative. This pre-empts the "trivial extension of DP" reviewer critique by exposing the equivalence as part of our framing.

### 3.2 SPL — related work paragraph, not baseline

Per `docs/notes/2026-05-04-a1-analysis.md §3`, SPL is treated as related framework in §6 with a 4-axis comparison (target structure, compilation cost, base lineage, scope). Not a baseline: SPL does not natively support multi-class atomic PMFs; encoding our count constraint would require per-$K$ SDD compilation, making the comparison axis "compilation cost vs no compilation" rather than the algebraic uniformity we expose.

### 3.3 Variable bag size — Ilse 2018 protocol

Bag size $N \sim \text{Normal}(10, 2)$ truncated to $[5, 15]$ for MNIST-sum, SVHN-sum, MNIST-signed. UltraMNIST has natural variable $N \in \{3, 4, 5\}$. This makes PCA's "atom count and atom shape are orthogonal" property visible, and matches Shukla 2023 / Ilse 2018 MNIST-MIL conventions for direct comparison.

### 3.4 Per-dataset backbone, head unified

Each dataset uses appropriate-capacity backbone (SmallCNN-multiclass for MNIST, ResNet-18 from scratch for SVHN, patch-based encoder for UltraMNIST). All baselines and PCA share the same backbone per dataset; only the head differs. Method-comparison fairness is the priority over dataset-comparison capacity uniformity.

### 3.5 Phase separation — A1 standalone, B2 separate

This spec is Phase 1 (A1 standalone). B2 (bag mixup / additive consistency) is a separate Phase 2 spec written **after** A1 verdict. Reasons:
- A1's falsifier must be standalone; B2 entanglement risks ablation requests.
- B1 retrospective: standalone falsification is cleaner than coupled.
- B2 may become "A1 paper bonus" or "future work / next paper" depending on A1 outcome.

### 3.6 Saturation insurance — task-difficulty modulation

Two saturation-insurance axes, applied **identically at training and evaluation** (task hardening, not robustness benchmark — see §4.5):

| Axis | CLI flag | Default (MNIST-sum / signed) | Hard preset |
|---|---|---|---|
| Per-class data scarcity | `--per-class-cap K` | **100** (sweet-spot estimate; MNIST instance-level acc target ~92–94%) | **50** (instance-level acc target ~88–90%, floor of safety) |
| Image corruption | `--noise-sigma σ` | 0.0 | **0.1** (light, MNIST-C low-severity range) |

For SVHN-sum and UltraMNIST: defaults are no cap and no noise (natural difficulty sufficient). Hard presets: cap=1000+noise=0.05 (SVHN), cap=300 (UltraMNIST).

**Day 3 sweep guard.** Day 3 includes a sweep: PCA × per-class-cap ∈ {200, 100, 50} × seed=0 × 30 epochs. Auto-decision rules:
- All caps PCA top-1 < 0.30 → cap too strict, raise default to 200.
- caps 100 and 200 differ < 1pp → saturation regime, lower default to 50.
- cap=100 PCA top-1 in [0.50, 0.85] → confirm default=100.

This sweep eliminates the cap-tuning iteration risk inside the 1-week budget.

---

## 4. Module architecture

### 4.1 Directory layout

Items marked ★ are added by this spec; unmarked items are reused from B1 verification.

```
probabilistic-convolutional-aggregator/
├── pca/
│   ├── __init__.py
│   ├── losses.py            ★ atomic_conv, multiclass_marginal_nll_loss added
│   ├── baselines.py         ★ NEW — 5 baselines (1a, 2a, 2b, 3a, 3b)
│   ├── data.py              ★ MNISTSum/SVHNSum/UltraMNIST/MNISTSigned bag datasets added
│   ├── models.py            ★ SmallCNNMulticlass, SmallCNNSigned, ResNet18FromScratch, PatchEncoder added
│   ├── metrics.py           ★ NEW — ECE, MAE, top-1 acc on bag sum, reliability diagram
│   └── train.py             ★ extended (variable-N collate, mask-aware loss)
├── scripts/
│   ├── sanity_a1.py         ★ NEW — atomic_conv binary equivalence + multi-class N=2,3 + signed shift + 5 baselines smoke
│   ├── run_mnist_sum.py     ★ NEW
│   ├── run_svhn_sum.py      ★ NEW
│   ├── run_ultramnist.py    ★ NEW
│   ├── run_mnist_signed.py  ★ NEW
│   └── analyze_a1.py        ★ NEW — H1/H2/H3 verdict + saturation detection + hard-preset CLI emit
├── tests/
│   ├── test_atomic_conv.py  ★ NEW
│   ├── test_baselines.py    ★ NEW
│   ├── test_data_a1.py      ★ NEW
│   └── test_metrics.py      ★ NEW
├── results/
│   ├── mnist_sum/, svhn_sum/, ultramnist/, mnist_signed/   ★ NEW dirs
│   └── (existing mnist_mil/, llp_adult/ untouched)
└── docs/superpowers/specs/
    ├── 2026-05-03-b1-em-verification-design.md   (template reference)
    └── 2026-05-04-a1-multiclass-counting-design.md   (this file)
```

### 4.2 Module dependencies (acyclic)

```
losses.py     (torch only, standalone)
metrics.py    (torch + sklearn for AUC/MAE — standalone)
data.py       (torch + torchvision + sklearn + pandas)
models.py     (torch.nn only)
baselines.py  (torch.nn + losses + models)
train.py      (losses + metrics + models + baselines + data)
scripts/      (train + analyze for analyze_a1.py)
```

`losses.py` and `metrics.py` are pure libraries — testable standalone, zero project dependencies.

### 4.3 Non-goals (explicit, see §5.4 of B1 spec for shared rationale)

- No abstract `Experiment` / `Trainer` base class.
- No Hydra/omegaconf/YAML config; argparse + module constants only.
- No multi-GPU, DDP, accelerate.
- No mixed precision.
- No model checkpointing; last-5-epoch mean for stability.
- No early stopping; fixed epochs.
- No pretrained backbones (ResNet-18 from scratch); fairness across baselines.
- No FFT-based conv; sequential atomic_conv suffices for $S \le 10$, $N \le 15$, support length $T = N(S{-}1){+}1 \le 136$.

---

## 5. Loss functions and atomic conv — `pca/losses.py`

### 5.1 Existing primitives (B1, reused untouched)

`forward_conv(p)`, `leave_one_out_posterior(p, bag_y)`, `marginal_nll_loss(logits, bag_y)`, `em_joint_loss(logits, bag_y, lam)` — see B1 spec §5. Constants `EPS = 1e-7`, `LOG_ZERO = -1e30`.

### 5.2 New: `atomic_conv(log_atoms)`

```python
def atomic_conv(log_atoms: Tensor) -> Tensor:
    """Generic 1D log-space convolution of independent atomic PMFs.

    Args:
        log_atoms: (B, N, S) — per-instance log-PMF over atom support.
                   Each row sums to 1 in prob space (logsumexp over last dim = 0).
                   S = atom support size:
                     S=2 → Bernoulli (matches forward_conv).
                     S=K+1 → multi-class ordinal {0, ..., K}.
                     S=3 → signed {-1, 0, +1} (caller handles ±-shift).
                     S=m_i+1 → multiplicity (variable per instance).
    Returns:
        log_P: (B, T) — log P(sum_i z_i = k) for k = 0, ..., T-1.
               T = N * (S - 1) + 1.
    Complexity: O(B * N * S * T) sequential conv. For N=15, S=10: ~1.4e4 ops/bag.

    Autograd-safe at structurally unreachable positions (uses LOG_ZERO sentinel,
    same rationale as forward_conv — see B1 spec §5.1).
    """
    B, N, S = log_atoms.shape
    T = N * (S - 1) + 1
    log_P = log_atoms.new_full((B, T), LOG_ZERO)
    log_P[:, 0] = 0.0
    for i in range(N):
        new_log_P = log_atoms.new_full((B, T), LOG_ZERO)
        for s in range(S):
            if s == 0:
                shifted = log_P
            else:
                shifted = F.pad(log_P[:, :T - s], (s, 0), value=LOG_ZERO)
            term = log_atoms[:, i, s].unsqueeze(1) + shifted
            new_log_P = torch.logaddexp(new_log_P, term)
        log_P = new_log_P
    return log_P
```

**Sanity invariants** (tests in §10):
- Bernoulli case: `atomic_conv(log_atoms_bernoulli) ≈ forward_conv(p)` (1e-6).
- `log_P.exp().sum(dim=1) ≈ 1` for any valid input (1e-5).
- N=2,3 hand-computed multi-class matches.

### 5.3 New: `multiclass_marginal_nll_loss(logits, bag_y, atom_offset=0)`

```python
def multiclass_marginal_nll_loss(
    logits: Tensor,    # (B, N, S) per-instance pre-softmax logits over atom support
    bag_y:  Tensor,    # (B,) integer bag sum, already shifted to [0, T-1] by caller
    mask:   Tensor | None = None,  # (B, N) boolean — True for active instances
) -> Tensor:
    """Multi-class generalization of marginal_nll_loss.

    Inactive instances (mask=False) have their atom forced to delta_0 = [1, 0, ..., 0]
    in prob space, so they contribute identity element to the convolution and are
    automatically ignored in the bag PMF. This handles variable bag size without
    a separate code path.

    For signed atoms ({-1, 0, +1}): caller passes shifted bag_y (true_y + N) and
    sets atom_offset = N internally during accuracy/MAE computation in metrics.py.
    """
    B, N, S = logits.shape
    log_atoms = F.log_softmax(logits, dim=-1)
    if mask is not None:
        # Force inactive atoms to delta_0
        delta_0 = log_atoms.new_full((S,), LOG_ZERO)
        delta_0[0] = 0.0
        log_atoms = torch.where(mask.unsqueeze(-1), log_atoms, delta_0)
    log_P = atomic_conv(log_atoms)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()
```

### 5.4 What's not in `losses.py`

- B2 bag-mixup loss — Phase 2.
- Multi-class leave-one-out posterior — A1 spec does not need it (B1's EM was demoted; only NLL needed for A1 verification).
- FFT-based conv — A2, separate spec.

---

## 6. Datasets — `pca/data.py`

### 6.1 Common contract

All A1 dataset classes return per item:
```python
(features, bag_sum, gt_per_instance, mask) = dataset[idx]
# features:        (N, *feature_shape) — variable N per item
# bag_sum:         int — already shifted for signed (true sum + N for signed)
# gt_per_instance: (N,) long — true z_i values (eval only)
# mask:            (N,) bool — all True for fixed N, mixed for variable N (handled by collate)
```

A custom `variable_n_collate_fn` pads to `N_max` in batch, attaches mask:
```python
def variable_n_collate_fn(batch):
    N_max = max(item[0].shape[0] for item in batch)
    # pad features to N_max, build mask
    ...
    return padded_features, bag_sums, padded_gts, mask
```

### 6.2 `MNISTSumBagDataset`

```python
class MNISTSumBagDataset(Dataset):
    def __init__(
        self,
        bag_size_mean: float = 10.0,
        bag_size_std:  float = 2.0,
        bag_size_min:  int   = 5,
        bag_size_max:  int   = 15,
        num_bags:      int   = 1500,
        per_class_cap: int | None = 100,    # default; None for SVHN/UltraMNIST
        noise_sigma:   float = 0.0,
        train:         bool  = True,
        seed:          int   = 0,
    ):
        ...
```

**Bag construction.**
1. Load MNIST; if `per_class_cap` set, subsample train pool to that many examples per class (deterministic via `seed`).
2. For each bag $b$: sample $N_b \sim \mathcal{N}(\mu, \sigma)$, clamp to $[\text{min}, \text{max}]$, round to int.
3. Sample $N_b$ MNIST indices without replacement.
4. `gt_per_instance[i] = digit_label[i]`, `bag_sum = gt_per_instance.sum()`.
5. If `noise_sigma > 0`: add Gaussian noise to images at access time (training and eval identically).

**Atom support** $S = 10$, $K = 9$, bag sum range $[0, 9 \cdot 15] = [0, 135]$.

### 6.3 `SVHNSumBagDataset`

Same contract as MNIST-sum, with:
- `torchvision.datasets.SVHN`, RGB 32×32.
- Default `per_class_cap=None` (SVHN train ~73k, naturally harder).
- Default `noise_sigma=0.0`. Hard preset: `noise_sigma=0.05`.

### 6.4 `UltraMNISTBagDataset`

```python
class UltraMNISTBagDataset(Dataset):
    """4000×4000 images with 3-5 MNIST digits scattered.

    Native variable N: bag_size depends on the number of digits in the image.
    No per-class-cap or noise modulation by default (natural difficulty).
    """
```

Uses Kaggle UltraMNIST corpus (manual download to `~/.cache/pca/ultramnist/`; spec includes URL and verification hash). "Per-instance feature" is a patch crop; `PatchEncoder` model partitions the 4000×4000 image into a fixed grid of patches and predicts per-patch class. Patches without digits → atom = $\delta_0$.

### 6.5 `MNISTSignedSumBagDataset`

```python
class MNISTSignedSumBagDataset(MNISTSumBagDataset):
    """Signed mapping: digit d → z_i ∈ {-1, 0, +1}.

    Mapping (pinned in code):
      d == 0       → z_i = 0
      d in {1,3,5,7,9}  → z_i = -1   (odd-nonzero)
      d in {2,4,6,8}    → z_i = +1   (even-nonzero)

    Bag sum range: [-N, +N]. Stored in `bag_sum` after shifting by +N so
    bag_sum ∈ [0, 2N]. Caller (metrics.py) un-shifts for accuracy/MAE.
    """
```

Inherits MNIST-sum's `per_class_cap` and `noise_sigma` (default 100, 0.0). Atom support $S = 3$.

### 6.6 Tests (§10.3)

- `bag_sum == gt.sum()` (unshifted for signed) for every bag, every dataset.
- Variable-N distribution matches truncated normal (KS test, weak threshold).
- Mask alignment: `mask.sum(dim=1)` == actual `N` per bag.
- `per_class_cap`: actual per-class instance count ≤ cap.
- `noise_sigma`: image variance increase consistent with σ.

---

## 7. Models — `pca/models.py`

All new models output **categorical logits over atom support** (last layer = `Linear(features, S)`); per-instance softmax happens inside loss/baseline modules.

### 7.1 `SmallCNNMulticlass(num_classes=10)`

Identical body to `SmallCNN` (B1) but with `Linear(128, num_classes)` final layer instead of `Linear(128, 1)`. ~210k params. For MNIST-sum, MNIST-signed (`num_classes=3`).

### 7.2 `ResNet18FromScratch(in_channels=3, num_classes=10)`

`torchvision.models.resnet18(weights=None)` with first conv adapted to 32×32 input (3×3 conv stride 1, no maxpool — CIFAR-style adaptation), final FC `→ num_classes`. ~11M params. For SVHN-sum.

### 7.3 `PatchEncoder(patch_size=64, num_classes=10, image_size=4000)`

Sliding-window CNN. Image partitioned into $\lceil 4000/64 \rceil^2$ patches; each patch passed through a small CNN (similar to SmallCNN but RGB input); output is `(num_patches, num_classes)` per-patch logit. Patches with no digit get atom $\approx \delta_0$ via training signal (no special handling). For UltraMNIST. ~1–2M params.

### 7.4 `SmallCNNSigned()`

Alias for `SmallCNNMulticlass(num_classes=3)`.

### 7.5 Per-bag forward (variable N + mask)

```python
def per_bag_forward(model, batch_features, mask):
    """
    batch_features: (B, N_max, *feature_shape) — padded
    mask:           (B, N_max) bool — True for active instances
    Returns: logits (B, N_max, S)
    """
    B, N_max = batch_features.shape[:2]
    flat = batch_features.reshape(B * N_max, *batch_features.shape[2:])
    logits_flat = model(flat)            # (B*N_max, S)
    return logits_flat.view(B, N_max, -1)
```

Mask is consumed by loss/baseline (forces inactive atoms to $\delta_0$ for PCA, ignores in averaging for baselines).

---

## 8. Baselines — `pca/baselines.py`

All baselines share signature:
```python
def baseline_forward(
    backbone_features: Tensor,  # (B, N_max, F) or shared logits (B, N_max, S)
    bag_y:             Tensor,  # (B,) bag sum
    mask:              Tensor,  # (B, N_max) bool
    K:                 int,     # max instance value (= S-1 for categorical)
) -> tuple[Tensor, Tensor | None]:
    """
    Returns:
        loss:     scalar — training loss for this method
        log_P:    (B, NK+1) or None — bag PMF for NLL/ECE evaluation.
                  None for methods that lack native bag distribution
                  (1a, 2a use Gaussian wrap in eval).
    """
```

### 8.1 1a Mean-pool regression

- Per-instance head: `Linear(features, 1)` → scalar $\hat z_i$.
- $\hat Y = \sum_i \text{mask}_i \cdot \hat z_i$.
- Loss: MSE($\hat Y$, $Y$).
- Bag PMF: None (Gaussian wrap in eval, $\sigma^2 = N \cdot K^2/12$).

### 8.2 2a PL-multiclass (sum-only adaptation)

- Per-instance head: `Linear(features, K+1)` → softmax $p_i \in \Delta^{K+1}$.
- $\mathbb E[z_i] = \sum_k k \cdot p_i[k]$.
- Bag mean $\bar Y / N = \frac{1}{N}\sum_i \mathbb E[z_i]$.
- Loss: MSE($\bar Y / N$, $Y / N$).
- Bag PMF: None (Gaussian wrap as above).

> Note: classical PL (Tsai 2020) targets bag-level proportion vector; sum-only target requires this expected-sum adaptation. Spec note in paper §4.

### 8.3 2b CLT Gaussian

- Per-instance head: `Linear(features, K+1)` → softmax $p_i$.
- Moments: $\mu_i = \sum_k k\, p_i[k]$, $\sigma_i^2 = \sum_k k^2 p_i[k] - \mu_i^2$.
- Bag prediction: $\hat Y \sim \mathcal N(\sum_i \text{mask}_i \mu_i, \sum_i \text{mask}_i \sigma_i^2)$.
- Loss: $-\log \mathcal N(Y; \mu_\text{bag}, \sigma_\text{bag}^2)$.
- Bag PMF: continuous Gaussian (discretized for ECE binning).

### 8.4 3a Attention pooling (Ilse 2018 gated)

- Backbone outputs feature $h_i \in \mathbb R^F$ (no per-instance head).
- Attention weights: $a_i = \text{softmax}_i(\mathbf w^\top \tanh(\mathbf V h_i) \odot \text{sigmoid}(\mathbf U h_i))$ over masked instances.
- Bag embedding: $z = \sum_i a_i h_i$.
- Bag head: $\text{MLP}(z) \to \text{Linear}(NK+1) \to \text{softmax}$.
- Loss: CE(softmax_bag, $Y$).
- Bag PMF: direct softmax.

> $NK+1$ is fixed to $\max(N) \cdot K + 1 = 15 \cdot 9 + 1 = 136$ for MNIST-sum/SVHN-sum/signed; $5 \cdot 9 + 1 = 46$ for UltraMNIST. Padded with $-\infty$ logits for shorter actual ranges.

### 8.5 3b DeepSets

- Backbone outputs feature $h_i$.
- Instance encoder: $\phi(h_i) = \text{MLP}_1(h_i)$.
- Aggregation: $z = \sum_i \text{mask}_i \cdot \phi(h_i)$ (sum, not mean — preserves $N$ information).
- Bag head: $\rho(z) = \text{MLP}_2(z) \to \text{Linear}(NK+1) \to \text{softmax}$.
- Loss: CE(softmax_bag, $Y$).
- Bag PMF: direct softmax.

### 8.6 PCA (ours)

- Per-instance head: `Linear(features, K+1)` → log-softmax → $\log p_i$.
- For inactive instances: atom = $\delta_0$ (handled inside `multiclass_marginal_nll_loss` via mask).
- Bag PMF: `atomic_conv(log_atoms)` → exact $\log P(\sum z = k)$.
- Loss: `multiclass_marginal_nll_loss`.
- Bag PMF: native exact PMF.

### 8.7 Tests (§10.4)

For each baseline: forward pass produces finite loss + finite gradient on a random N=10 batch with random labels. `log_P` (when not None) is a valid log-probability vector (logsumexp = 0).

---

## 9. Training, scripts — `pca/train.py`, `scripts/`

### 9.1 `train(...)` extension

Reuses B1 `train()` skeleton with extensions:
- Accepts `loss_fn(features, bag_y, mask, K)` taking 4 args (vs B1's 2).
- Per-epoch test eval computes 4 metrics: NLL, top-1 acc on bag sum (argmax of `log_P`), MAE (expected bag sum from $\log P$), ECE (15-bin reliability over bag-sum predictions).
- `last_5_epoch_mean` for each metric → `best_test_*`.

### 9.2 `run_seeds(...)`

Identical to B1: deterministic seeding, sequential per seed, persist JSON per seed. Schema in §12.1.

### 9.3 Hyperparameters (per dataset)

See Section 4.1 of brainstorming output (Design §4) for full table. Pinned in `scripts/run_*.py` via argparse defaults.

### 9.4 Scripts

Common CLI for `run_*.py`:
```
python scripts/run_<dataset>.py \
  --method {pca,1a,2a,2b,3a,3b} \
  --epochs <int> --seeds <comma-list> \
  --bag-size-mean <float> --bag-size-std <float> \
  --per-class-cap <int|None> --noise-sigma <float> \
  --output-dir results/<dataset>/
```

Magic numbers absent from code: per-class-cap, noise-sigma, bag-size are CLI flags from Day 1.

`scripts/sanity_a1.py`: see §10.2 for required checks.

`scripts/analyze_a1.py`:
1. Glob `results/<dataset>/N*_*/seed*.json` for each dataset.
2. Aggregate per (dataset, method) → mean, std, per-seed list.
3. Compute H1, H2, H3 (Section 11).
4. Detect saturation (Section 2.4).
5. Emit `summary.md` per dataset + `summary_overall.md`.
6. Print verdict + (if `RUN_HARD_PRESET`) exact CLI commands.

---

## 10. Testing strategy

### 10.1 `tests/test_atomic_conv.py`

| Test | Asserts |
|---|---|
| `test_atomic_conv_bernoulli_equiv_forward_conv` | `atomic_conv(log_atoms_S2) ≈ forward_conv(p)` (1e-6) on random batch |
| `test_atomic_conv_n2_multiclass_hand` | Hand-computed N=2, K=2 case matches |
| `test_atomic_conv_n3_multiclass_hand` | Hand-computed N=3, K=2 case matches |
| `test_atomic_conv_sums_to_one` | `log_P.exp().sum(1) ≈ 1` for random 8×10×4 (1e-5) |
| `test_atomic_conv_signed_shift` | Signed atom (S=3, supported as {-1,0,+1}) when caller shifts bag_y by +N → equivalent to S=3 atom over {0,1,2} |
| `test_multiclass_nll_finite` | `multiclass_marginal_nll_loss` finite + finite grad |
| `test_multiclass_nll_mask_handles_variable_n` | With mask=[T,T,F,F]: result equals N=2 case |

### 10.2 `tests/test_baselines.py` + `scripts/sanity_a1.py`

Per-baseline unit tests:
- Forward returns finite loss.
- Backward produces finite gradients on backbone params.
- `log_P` (when returned) is valid log-probability (logsumexp ≈ 0).

`scripts/sanity_a1.py` end-to-end:
1. `atomic_conv` Bernoulli equivalence.
2. Multi-class N=2,3 hand cases.
3. Signed atom shift correctness.
4. `multiclass_marginal_nll_loss` + Adam step → no NaN, params change.
5. Each of 5 baselines + PCA: 1 forward + backward + Adam step on N=10 batch.
6. Variable-N collate with N ∈ {5, 10, 15} in same batch → all losses finite.

### 10.3 `tests/test_data_a1.py`

- `bag_sum == gt.sum()` (unshifted for signed).
- Variable-N distribution: $N$ truncated to $[\text{min}, \text{max}]$, mean/std consistent with config (weak threshold).
- Mask: `mask.sum(dim=1)` == per-bag actual $N$.
- `per_class_cap`: actual per-class count ≤ cap.
- `noise_sigma=0.1`: image variance increase ≈ $0.1^2$ relative to baseline.
- Determinism: same seed → byte-identical bags.

### 10.4 `tests/test_metrics.py`

- ECE on toy (10 predictions, hand-computed bins) matches reference value.
- MAE on integer support.
- `top1_acc` from log_P matches argmax-vs-truth count.

### 10.5 Test gates

- **Day 1 end**: `pytest tests/test_atomic_conv.py tests/test_metrics.py` 100% pass.
- **Day 2 end**: `pytest tests/test_baselines.py` + `scripts/sanity_a1.py` 100% pass.
- **Day 3 entry**: `pytest tests/test_data_a1.py` 100% pass; smoke runs produce schema-valid JSONs.
- **Day 7**: `analyze_a1.py` produces a verdict for each dataset.

---

## 11. Hypothesis evaluation logic — `analyze_a1.py`

### 11.1 Aggregation

```python
@dataclass
class CellStat:
    mean: float
    std:  float
    seeds: list[float]   # raw per-seed best_test_<metric>

# summary[dataset][method][metric] = CellStat   for metric in {nll, acc, mae, ece}
def aggregate(results_root: Path) -> dict:
    ...
```

### 11.2 H1 — NLL primary

```python
def evaluate_h1(summary, multiclass_datasets=['mnist_sum', 'svhn_sum', 'ultramnist']) -> dict:
    """
    H1: PCA NLL < best baseline NLL, paired t p<0.05, |Δ|≥0.02 nats,
        on at least 2 of 3 datasets.
    """
    distribution_native = ['2b', '3a', '3b']
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        pca_seeds = summary[ds]['pca']['nll'].seeds
        best_baseline_per_seed = np.minimum.reduce([
            summary[ds][m]['nll'].seeds for m in distribution_native
        ])
        deltas = np.array(best_baseline_per_seed) - np.array(pca_seeds)
        delta_mean = float(np.mean(deltas))
        t_stat, p_value = scipy.stats.ttest_rel(best_baseline_per_seed, pca_seeds)
        passes = (delta_mean >= 0.02 and p_value < 0.05)
        per_dataset[ds] = dict(delta=delta_mean, p=p_value, pass_=passes)
        if passes: pass_count += 1
    return dict(pass_=(pass_count >= 2), per_dataset=per_dataset, pass_count=pass_count)
```

### 11.3 H2 — atom-shape generality

```python
def evaluate_h2(summary) -> dict:
    """All atom shapes produce valid PMF + above-random performance."""
    # Binary case: read B1 verification summary (committed artifact, NLL row at N=50).
    b1_summary_path = Path('results/mnist_mil/summary.json')
    b1 = json.loads(b1_summary_path.read_text())
    binary_pca_n50_auc = b1['per_size'][50]['nll']['mean']  # NLL is the PCA equivalent in B1
    checks = {
        'binary_b1_reuse':       binary_pca_n50_auc > 0.9,
        'multiclass_mnist_sum':  summary['mnist_sum']['pca']['acc'].mean > 0.30,
        'multiclass_svhn_sum':   summary['svhn_sum']['pca']['acc'].mean > 0.30,
        'multiclass_ultramnist': summary['ultramnist']['pca']['acc'].mean > 0.30,
        'signed_mnist_signed':   summary['mnist_signed']['pca']['acc'].mean > 0.20,
        'all_nll_finite':        all(np.isfinite(summary[ds]['pca']['nll'].mean)
                                     for ds in ['mnist_sum','svhn_sum','ultramnist','mnist_signed']),
    }
    return dict(pass_=all(checks.values()), checks=checks)
```

### 11.4 H3 — calibration

```python
def evaluate_h3(summary, multiclass_datasets=['mnist_sum','svhn_sum','ultramnist']) -> dict:
    """PCA ECE ≤ 0.7 × min(distribution-native baseline ECE) on ≥ 2 of 3."""
    distribution_native = ['2b', '3a', '3b']
    pass_count = 0
    per_dataset = {}
    for ds in multiclass_datasets:
        pca_ece = summary[ds]['pca']['ece'].mean
        min_baseline_ece = min(summary[ds][m]['ece'].mean for m in distribution_native)
        ratio = pca_ece / min_baseline_ece if min_baseline_ece > 0 else float('inf')
        passes = ratio <= 0.7
        per_dataset[ds] = dict(pca_ece=pca_ece, min_baseline_ece=min_baseline_ece, 
                                ratio=ratio, pass_=passes)
        if passes: pass_count += 1
    return dict(pass_=(pass_count >= 2), per_dataset=per_dataset, pass_count=pass_count)
```

### 11.5 Saturation detection

```python
def detect_saturation(summary, dataset: str) -> bool:
    nlls = [summary[dataset][m]['nll'].mean for m in ['2b','3a','3b']]
    accs = [summary[dataset][m]['acc'].mean for m in ['1a','2a','2b','3a','3b','pca']]
    return (max(nlls) - min(nlls) < 0.05) or (max(accs) > 0.97)
```

### 11.6 Verdict

```python
def verdict(h1, h2, h3, saturation_per_dataset, results_root: Path) -> str:
    if not h2['pass_']:
        return 'ALGEBRA_BROKEN'
    if h1['pass_']:
        return 'A1_CONFIRMED'
    # H1 failed
    # Hard preset detection: presence of any seed*.json with config.per_class_cap == hard value
    # under results/<dataset>/ (e.g., MNIST cap=50 vs default 100).
    hard_preset_already = any(
        (results_root / ds).rglob('seed*.json') and
        any(json.loads(p.read_text())['config'].get('per_class_cap') == HARD_CAP[ds]
            for p in (results_root / ds).rglob('seed*.json'))
        for ds in saturation_per_dataset
    )
    if any(saturation_per_dataset.values()) and not hard_preset_already:
        return 'RUN_HARD_PRESET'
    # H1 failed even on hard preset (or no saturation signal)
    if h3['pass_']:
        return 'A1_CALIBRATION_FOCUS'
    return 'DEMOTE_A1'

# Hard-preset cap sentinel per dataset
HARD_CAP = {
    'mnist_sum':     50,
    'mnist_signed':  50,
    'svhn_sum':      1000,
    'ultramnist':    300,
}
```

### 11.7 Partial verdict (Day 4)

`analyze_a1.py --partial` accepts a subset of datasets (default: only those with all expected JSONs present). On Day 4, this runs over `{mnist_sum, mnist_signed}` only:
- H1 evaluated on `multiclass_datasets=['mnist_sum']` (single dataset; "≥2 of 3" rule loosened to "pass on this one").
- H2 evaluated on the binary B1 reuse + multiclass MNIST-sum + signed sub-conditions only.
- H3 evaluated on `mnist_sum` only.
- Saturation detection on `mnist_sum`.

Partial verdict's purpose is operational decision support (e.g., adjust SVHN-sum/UltraMNIST hyperparameters before Day 5–6 overnight runs), not final paper claim. Final verdict on Day 7 supersedes.

### 11.8 Reporting

`results/<dataset>/summary.md` per dataset:
```markdown
# Results — <dataset>
**Generated**: 2026-05-XX  **Config**: default  **Seeds**: 5

## Metrics (mean ± std)

| Method | Top-1 acc       | MAE         | NLL         | ECE         |
|--------|-----------------|-------------|-------------|-------------|
| 1a     | ...             | ...         | ...*        | ...*        |
| 2a     | ...             | ...         | ...*        | ...*        |
| 2b     | ...             | ...         | ...         | ...         |
| 3a     | ...             | ...         | ...         | ...         |
| 3b     | ...             | ...         | ...         | ...         |
| PCA    | ...             | ...         | ...         | ...         |

(*Gaussian wrap)

## H1 (NLL): {pass/fail}  Δ = ... nats  p = ...
## H3 (ECE): {pass/fail}  ratio = ...
## Saturation: {detected/clear}
```

`results/summary_overall.md`:
- H1 across datasets (pass count, per-dataset breakdown).
- H2 atom-shape generality table.
- H3 across datasets.
- Saturation detection result.
- Final verdict.
- Reliability diagram (PCA vs Attention, dataset chosen by H1 strongest).

---

## 12. Result storage

### 12.1 Per-run JSON schema

```json
{
  "config": {
    "experiment": "mnist_sum",
    "method": "pca",
    "atom_support": 10,
    "bag_size_mean": 10.0, "bag_size_std": 2.0,
    "bag_size_min": 5, "bag_size_max": 15,
    "per_class_cap": 100, "noise_sigma": 0.0,
    "lr": 1e-3, "epochs": 80, "batch_size": 32,
    "num_train_bags": 1500, "num_test_bags": 600,
    "seed": 3, "device": "mps"
  },
  "epoch_train_loss": [...],
  "epoch_test_nll":   [...],
  "epoch_test_acc":   [...],
  "epoch_test_mae":   [...],
  "epoch_test_ece":   [...],
  "best_test_nll":    1.234,
  "best_test_acc":    0.612,
  "best_test_mae":    1.85,
  "best_test_ece":    0.067,
  "wall_clock_sec":   612.5,
  "git_sha":          "...",
  "torch_version":    "..."
}
```

### 12.2 Git policy

- `results/<dataset>/<setting>/seed*.json`: gitignored (~1MB×88 = 88MB).
- `results/<dataset>/summary.md`, `summary.json`: committed.
- `results/summary_overall.md`: committed.
- All code (pca/, scripts/, tests/) committed.
- This spec committed at design approval time.

`.gitignore` additions (extending B1's):
```
results/mnist_sum/N*_*/
results/svhn_sum/N*_*/
results/ultramnist/N*_*/
results/mnist_signed/N*_*/
~/.cache/pca/
```

---

## 13. Day-by-day plan

| Day | Deliverable | Gate |
|---|---|---|
| **1** | `pca/losses.py` (`atomic_conv`, `multiclass_marginal_nll_loss`); `pca/metrics.py` (ECE, MAE, top-1 acc); `tests/test_atomic_conv.py`, `tests/test_metrics.py` | All tests pass; Bernoulli equivalence verified |
| **2** | `pca/baselines.py` (5 baselines); `pca/models.py` extensions (SmallCNNMulticlass, ResNet18FromScratch, PatchEncoder, SmallCNNSigned); `pca/train.py` extension; `scripts/sanity_a1.py` | sanity_a1 passes |
| **3** | `pca/data.py` MNIST-sum + MNIST-signed; `scripts/run_mnist_sum.py`, `run_mnist_signed.py`; **per-class-cap sweep smoke** (PCA × cap∈{200,100,50} × seed=0 × 30 ep) | Sweep guard rules confirm default cap |
| **4** | MNIST-sum 30 runs (~5h) + MNIST-signed 10 runs (~1.7h); `analyze_a1.py` partial verdict (MNIST-sum + signed) | All 40 JSONs schema-valid |
| **5** | `pca/data.py` SVHN-sum; `scripts/run_svhn_sum.py`; SVHN-sum smoke + 30 runs (overnight, ~15h) | SVHN runs JSON-valid |
| **6** | `pca/data.py` UltraMNIST + `pca/models.py` PatchEncoder finalization; `scripts/run_ultramnist.py`; smoke + 18 runs (overnight, ~13.5h) | UltraMNIST runs JSON-valid |
| **7** | `analyze_a1.py` final verdict; (if `RUN_HARD_PRESET`) hard preset rerun on flagged dataset; `summary_overall.md` commit | Final verdict emitted; `summary_*` files committed |

**Compute schedule.** Day 4–6 heavy runs are overnight. Day 7 morning: collect JSONs. Day 7 afternoon: analyze. Day 7 evening: verdict + README priority update if needed.

---

## 14. Decision points

### 14.1 Day 1 → 2

`pytest tests/test_atomic_conv.py tests/test_metrics.py` exits 0 with all tests passing. Wall-clock < 60s.

### 14.2 Day 2 → 3

`python scripts/sanity_a1.py` prints `[OK] All A1 sanity checks passed.` and returns 0.

### 14.3 Day 3 → 4

Smoke sweep results: PCA top-1 acc on MNIST-sum at cap ∈ {200, 100, 50} satisfies one of:
- All caps → top-1 < 0.30: cap too strict → raise default to 200, re-smoke (~1h).
- caps 200 vs 100 differ < 1pp: saturation regime → lower default to 50, re-smoke.
- cap=100 → top-1 in [0.50, 0.85]: confirm default=100 ✓.

### 14.4 Day 4 → 5

40 JSONs (MNIST-sum 30 + signed 10) present and schema-valid. Partial `summary.md` produced for both datasets.

### 14.5 Day 5 → 6

SVHN-sum 30 JSONs present and schema-valid. SVHN-sum `summary.md` produced.

### 14.6 Day 6 → 7

UltraMNIST 18 JSONs present and schema-valid. UltraMNIST `summary.md` produced.

### 14.7 Day 7 (the critical gate)

`analyze_a1.py` emits one of:
- `A1_CONFIRMED` → `README.md` priority remains A1 Main; Phase 2 (B2 bolt-on spec) starts.
- `RUN_HARD_PRESET` → exact CLI for hard preset emitted; ~5–15h additional runs; Day 8 verdict.
- `A1_CALIBRATION_FOCUS` → README updated noting calibration-focused pivot; design re-discussion with user.
- `DEMOTE_A1` → `README.md` priority A1 Main → Low; alternative track (B2 standalone or cross-domain validation) starts.
- `ALGEBRA_BROKEN` → STOP. Code/spec debug. No scientific interpretation.

---

## 15. Out of scope

This spec does **not** cover:

- **B2 bag mixup / additive consistency** — Phase 2 spec, written after `A1_CONFIRMED` (or alternative).
- **A4 cancellation lemma / risk-consistency theorem** — Track 2 theory work, code not in scope.
- **A2 FFT/tree reduction** — sequential atomic_conv is sufficient for $N \le 15$, $S \le 10$, $T \le 136$. Separate spec only if UltraMNIST wall-clock becomes bottleneck.
- **LLP / PU benchmarks for A1** — cross-domain validation Phase 2 (Track 3 follow-up).
- **Atom support size sweep ablation ($K=10$ vs $K=99$ etc.)** — paper appendix candidate, not main verification.
- **Hyperparameter tuning** — lr, batch, optimizer fixed (Adam, lr=1e-3 except UltraMNIST lr=5e-4). Saturation insurance is the only hyperparameter axis.
- **Pretrained backbones** — ResNet-18 from scratch; pretrain would unfair-advantage all methods including baselines.
- **Mixed precision / multi-GPU / DDP** — single device, fp32.
- **Multi-class DP baseline implementation** — equivalence handled by §3 body proposition; implementing it would be redundant and dilute contribution narrative.
- **SPL baseline implementation** — Related Work paragraph only; SPL does not natively support multi-class atomic PMFs, encoding the count constraint requires per-$K$ SDD compilation (different comparison axis).
- **CIFAR-style nominal classes** — out of scope; conv view requires sum-defined support. CIFAR re-configuration options considered and excluded; one-line footnote in paper.
- **Multiplicity case empirical experiment** — covered as algebraic claim in body (atom shape $\{0, m_i\}$, length $m_i+1$ trivially supported by `atomic_conv`). Empirical demonstration left to future work.

---

## 16. Dependencies and environment

### 16.1 `pyproject.toml` additions

```toml
[project]
name = "pca"
version = "0.2.0"           # bumped from 0.1.0 (B1)
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "numpy>=1.26",
  "pandas>=2.1",
  "scikit-learn>=1.3",
  "scipy>=1.11",
  "matplotlib>=3.8",     # NEW: reliability diagrams
  "torchmetrics>=1.2",   # NEW: standard ECE
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]
```

### 16.2 Setup commands

```sh
cd /Users/bhwang/Documents/Projects/probabilistic-convolutional-aggregator
uv sync                                # picks up new deps
uv run pytest tests/                   # all tests pass
uv run python scripts/sanity_a1.py     # sanity passes
```

### 16.3 Compute environment

- Mac (Darwin), Apple Silicon → MPS device for PyTorch.
- CPU fallback available.
- Estimated wall-clock for full default-config A1 verification: **~35 hours** sequential (5h MNIST-sum + 1.7h signed + 15h SVHN + 13.5h UltraMNIST). Plus Day 1–3 implementation (~12–18h human-time).

### 16.4 UltraMNIST data acquisition

UltraMNIST corpus (4000×4000 multi-digit images) from Kaggle competition / Scientific Data 2024. Manual download required (license accept). Cache to `~/.cache/pca/ultramnist/` after first download. Spec includes verification hash on first run; mismatched hash → fail-fast to prompt re-download.

---

## Appendix A: SIMPLE Prop. 1 — multi-class generalization sketch

(Body §3 proposition's appendix proof.)

**Proposition.** For $N$ independent atomic PMFs $\mathbf d_1, \ldots, \mathbf d_N$ over support $\{0, 1, \ldots, K\}$, the bag PMF
$$P_\text{bag}[s] = P\!\Big(\sum_{i=1}^N z_i = s\Big), \quad s \in \{0, 1, \ldots, NK\},$$
can be computed in $O(N \cdot K \cdot NK)$ time by both:
1. **Recursive convolution**: $P_\text{bag}^{(0)} = \delta_0$; $P_\text{bag}^{(i)}[s] = \sum_{k=0}^{K} \mathbf d_i[k] \cdot P_\text{bag}^{(i-1)}[s - k]$.
2. **Forward DP** (multi-class generalization of SIMPLE Prop. 1): Same recursion, indexed differently; the inner loop iterates over $k \in \{0, \ldots, K\}$ identical to step (i).

The two computations are pointwise equal; they are the same algorithm under different notational conventions.

**Implication.** Implementing multi-class DP separately as a baseline would compute the same values as PCA's `atomic_conv`, providing zero comparison signal. Hence we include the equivalence as proposition and exclude the DP baseline.

---

## Appendix B: References

- **Shukla et al. 2023** — "A Unified Approach to Count-Based Weakly Supervised Learning", NeurIPS 2023. `sources/Shukla2023/camera_ready.tex`. Binary base.
- **SIMPLE** — Ahmed/Zeng/Niepert/Van den Broeck, ICLR 2023. arXiv 2210.01941. Base reference for binary count DP.
- **SPL** — Ahmed/Yousri/Niepert/Vergari/Van den Broeck, NeurIPS 2022. arXiv 2206.00426. Related framework, not baseline.
- **Tsai & Lin 2020** — PL (Learning from Label Proportions), ICML 2020. Baseline 2a's binary origin.
- **Ilse/Tomczak/Welling 2018** — Attention-based Deep MIL, ICML 2018. Baseline 3a.
- **Zaheer et al. 2017** — Deep Sets, NeurIPS 2017. Baseline 3b.
- **Hendrycks & Dietterich 2019** — Common Corruptions, ICLR 2019. Saturation insurance noise framework reference.
- **Mu & Gilmer 2019** — MNIST-C. Standard MNIST corruption benchmark.
- **Ovadia et al. 2019** — Calibration under distributional shift, NeurIPS 2019. ECE-with-noise reference.
- **B1 verification artifacts** — `pca/losses.py`, `pca/models.py`, `tests/test_losses.py`, `results/mnist_mil/summary.md`, `results/llp_adult/summary.md`. Committed at `fc1c9da`.
- **Project notes** — `docs/notes/2026-05-04-a1-analysis.md` (brainstorming input); `01_pca_strengthening.md §A1`; `03_evaluation_strategy.md §4 Track 1`; `README.md §2 (priority Main)`.

---
