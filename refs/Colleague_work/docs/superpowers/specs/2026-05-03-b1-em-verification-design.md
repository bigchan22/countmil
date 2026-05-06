# B1: Count-conditioned EM verification — Design

| | |
|---|---|
| **Date** | 2026-05-03 |
| **Status** | Draft (awaiting approval) |
| **Owner** | Byung-Hak Hwang (KIAS) |
| **Project** | Probabilistic Convolutional Aggregator (PCA) — NeurIPS 2026 |
| **Track** | `03_evaluation_strategy.md` §4 Track 1 (Empirical core, week 1–3) |
| **Related notes** | `02_training_strengthening.md §B1`, `01_pca_strengthening.md §A1` |

---

## 1. Context and motivation

### 1.1 Where this sits in the project

The Probabilistic Convolutional Aggregator (PCA) computes the exact PMF of a Bernoulli sum via convolution and trains under aggregate-only supervision via marginal NLL on $-\log P(\sum z_i = Y)$. The pure-methodology pivot for NeurIPS 2026 has chosen five contribution candidates:

- **A1** (Main): Atomic PMF generality (multi-class / signed / mixture).
- **B1** (High): Count-conditioned EM / posterior matching — *this spec*.
- **B2** (High): Bag mixup / additive consistency.
- **A4** (High, contingent): Signed counting risk-consistency theorem.
- **Cross-domain validation** (High).

B1 is scheduled for week 1 verification because (i) the implementation is small (~120 lines of core math), and (ii) it is the contribution most at risk of being framing-only — its claim must be falsified or substantiated empirically before further investment.

### 1.2 The B1 idea

For a bag of $N$ instances with predicted Bernoulli probabilities $p_1, \dots, p_N$ and observed count $Y = \sum z_i$, the posterior over instance assignment is exactly computable:

$$
q_u := P(z_u = 1 \mid \textstyle\sum_i z_i = Y) = \frac{p_u \cdot P_{j \neq u}(\sum = Y - 1)}{P(\sum = Y)}.
$$

Both numerator and denominator are bag PMFs, computable by convolution. With prefix·suffix conv this can be done for all $u$ in $O(N^2)$ total — same order as one forward conv.

The **EM joint loss** (Approach A) augments marginal NLL with a soft cross-entropy against the detached posterior:

$$
\mathcal L_\text{EM} = -\log P(\textstyle\sum z = Y) + \lambda \cdot \mathrm{CE}(p, q_\text{stop\_grad}).
$$

### 1.3 Why this verification is necessary

A purely theoretical observation: **marginal NLL gradient already encodes the same posterior**. Differentiating $\log P(\sum = Y)$ with respect to $p_u$:

$$
\frac{\partial \log P(\sum=Y)}{\partial p_u} = \frac{P_{j \neq u}(\sum = Y-1) - P_{j \neq u}(\sum = Y)}{P(\sum = Y)}.
$$

Together with $P(\sum=Y) = (1 - p_u) P_{j \neq u}(\sum = Y) + p_u P_{j \neq u}(\sum = Y - 1)$, the posterior $q_u$ can be recovered in closed form from this gradient (see Appendix A). Therefore:

> "B1 is more informative than NLL" is **mathematically false**. NLL gradient descent and EM both use the same posterior; they differ only in *optimization trajectory*, not in *information content*.

B1's contribution, if any, must come from one of three optimization-side framings:

- **(F1) Optimization geometry**: M-step is a fixed-target supervised classification problem, well-conditioned per iteration. Marginal NLL has a moving-target gradient that may exhibit different local-optimum / saddle-escape behavior.
- **(F2) Auxiliary objective compositionality**: The explicit pseudo-label $q$ allows clean addition of entropy / sparsity / calibration regularizers on the M-step that don't compose well with marginal NLL directly.
- **(F3) Variance / stability**: M-step gradients are deterministic given $q$; NLL gradients aggregate the implicit posterior estimation noise.

If none of (F1), (F2), (F3) manifest empirically, B1 is framing-only and is **demoted from main contribution**.

---

## 2. Hypotheses under test and falsifier

### 2.1 Concrete hypotheses (all evaluated on test-set instance-level binary AUC)

- **H1 (primary)**: On MNIST-MIL with bag size $N = 50$ (standard Ilse 2018 / Shukla scale), averaged over 5 seeds, the EM joint loss yields instance-level AUC at least **1.0 percentage points** above plain marginal NLL, with paired t-test $p < 0.05$.

- **H2 (bag-size scaling)**: The AUC delta $\Delta(N) := \mathrm{AUC}_\text{EM}(N) - \mathrm{AUC}_\text{NLL}(N)$ is **monotone non-decreasing** across $N \in \{10, 50, 100\}$ (with 0.2 pp tolerance for noise). This corresponds to (F1)-style argument: marginal NLL gradient noise grows with bag size.

- **H3 (variance reduction)**: The cross-seed standard deviation of EM at $N = 50$ is at most **0.7 ×** that of NLL. This corresponds to (F3).

### 2.2 Falsifier (explicit demote criteria)

| Outcome at end of Day 5 | Verdict | Action |
|---|---|---|
| H1 passes AND (H2 OR H3) passes | `PROCEED_TO_LLP` | Day 6–7: LLP Adult experiment |
| H1 passes AND H2/H3 both fail | `PROCEED_BUT_WEAK` | LLP yes, but B1 is "boost-axis" not "main" in the paper |
| $0.5 \le \Delta(N{=}50) < 1.0$ pp | `F2_ABLATION_NEEDED` | Add entropy reg in M-step, redo MNIST-MIL N=50 only |
| $\Delta(N{=}50) < 0.5$ pp or negative | **`DEMOTE_B1`** | B1 → Low priority; Track 1 step 3 (B2) starts immediately; README priority table updated |

**Statistical considerations**:
- 5 seeds is small; we use **paired** t-test (same seed → same data and model init for both NLL and EM) to control variance.
- All hypotheses are pre-registered in this spec. No fishing.

### 2.3 LLP Adult hypothesis (Day 6–7, only if `PROCEED_*`)

- **H1_LLP**: instance-level AUC delta at $N=128 \ge 1$ pp.
- **H2_LLP**: monotone in $N \in \{32, 128, 512\}$. Strong H2_LLP → "B1's edge grows with bag size" is the paper's main message about B1.
- **H3_LLP**: std reduction ≥ 30% at $N = 128$.

---

## 3. Approach choice — Joint loss (Approach A)

### 3.1 Variants considered

| Variant | E-step frequency | Pros | Cons |
|---|---|---|---|
| **A: Joint loss** | Implicit, every batch | Drop-in (toggle a flag), wall-clock = NLL, lam=0 ≡ NLL gives free sanity | Not "classical" EM |
| B: Periodic E-step | Every $K$ epochs | Faithful to classical EM, cleaner free-energy connection | Extra hparams, slower wall-clock, more code |
| C: Warmup + EM | Switch at fraction $\tau$ | Avoids EM cold-start | Yet another hparam |

### 3.2 Selected: A

Chosen for:
1. **Falsifier velocity**: lam=0 numerical equivalence to NLL gives instant gradient-direction sanity (Section 5.1 test). lam=1 ablation runs wall-clock equivalent to baseline, so 5 seeds finish in the same time budget.
2. **CLAUDE.md alignment**: minimum code (~120 lines core), no new abstractions, no new hyperparameters beyond `lam`.
3. **Sequential decision logic**: B fits later if A passes H1 but H2/H3 fail and we want to test classical-EM as an additional ablation. C fits if A is unstable. Premature inclusion blocks falsifier.

### 3.3 Loss formulation

Let $\sigma$ denote sigmoid, $p_i = \sigma(\ell_i)$ for per-instance logit $\ell_i$, $Y$ the observed bag count.

$$
\mathcal L_{\text{NLL}}(\boldsymbol\ell, Y) = -\log P(\textstyle\sum z = Y \mid \boldsymbol p),
$$

$$
q_u = \frac{p_u \cdot P_{j \neq u}(\sum = Y-1)}{P(\sum = Y)}, \quad q_u \text{ has \texttt{stop\_grad}},
$$

$$
\mathcal L_{\text{EM}}(\boldsymbol\ell, Y; \lambda) = \mathcal L_{\text{NLL}} - \lambda \sum_u \big[q_u \log p_u + (1 - q_u) \log(1 - p_u)\big].
$$

Default $\lambda = 1.0$. Sweep $\{0.1, 0.5, 1.0, 2.0\}$ is stretch (Section 11.4).

---

## 4. Module architecture

### 4.1 Directory layout

```
probabilistic-convolutional-aggregator/
├── pca/
│   ├── __init__.py
│   ├── losses.py           # forward_conv, leave_one_out_posterior, marginal_nll_loss, em_joint_loss
│   ├── data.py             # MNISTBagDataset, LLPBagDataset
│   ├── models.py           # SmallCNN (MNIST), MLP (LLP)
│   └── train.py            # train(...), run_seeds(...), device selection, metrics
├── scripts/
│   ├── sanity_n4.py        # toy gradient + lam=0 equivalence + posterior sum check
│   ├── run_mnist_mil.py    # argparse entrypoint, per-seed training
│   ├── run_llp_adult.py    # argparse entrypoint
│   └── analyze.py          # results aggregation, hypothesis evaluation, markdown emit
├── tests/
│   ├── __init__.py
│   ├── test_losses.py      # numerical correctness vs hand-computed N=2,3
│   └── test_data.py        # bag construction invariants
├── results/                # gitignored: per-run JSON. Summaries are NOT gitignored.
├── docs/superpowers/specs/
│   └── 2026-05-03-b1-em-verification-design.md   # this file
└── pyproject.toml          # uv-managed
```

### 4.2 Module dependencies (acyclic)

```
losses.py  ──── (torch only, standalone)
                │
data.py    ──── (torch, torchvision, sklearn, pandas)
                │
models.py  ──── (torch.nn only)
                │
train.py   ──── losses + models + data
                │
scripts/*  ──── train + (analyze for analyze.py)
```

`losses.py` has zero project dependencies. This is intentional — it must be testable standalone.

### 4.3 Non-goals (explicit)

- No abstract `Experiment` / `Trainer` base class. Scripts directly call `train()`.
- No Hydra, omegaconf, or YAML config. Each script uses argparse + module-level constants.
- No multi-GPU, DDP, or accelerate. Single device.
- No mixed precision. Models small; not warranted.
- No model checkpointing. We retain final-state metrics only (last-5-epoch mean for stability).
- No early stopping. Fixed epochs.

---

## 5. Loss functions — `pca/losses.py`

### 5.1 Constants

```python
EPS = 1e-7         # sigmoid output clamp lower bound
LOG_ZERO = -1e30   # log-zero sentinel; see §5.2 for rationale
```

`LOG_ZERO` replaces `float('-inf')` everywhere `forward_conv` accumulates a value that may participate in autograd. PyTorch's `torch.logaddexp(-inf, -inf) = -inf` is correct in forward but produces `NaN` in backward via `exp(-inf - (-inf)) = exp(NaN)`; the NaN then propagates through chain rule to all output positions. `-1e30` keeps the algebra (`exp(-1e30) ≈ 0` to far below float32 precision) while preserving finite gradients. `leave_one_out_posterior` runs under `torch.no_grad()` and therefore retains `float('-inf')` — the asymmetry is intentional.

### 5.2 `forward_conv(p)`

Compute the bag PMF in log space via sequential convolution.

**Signature**:
```python
def forward_conv(p: torch.Tensor) -> torch.Tensor:
    """
    Args:
        p: (B, N) per-instance Bernoulli probability, expected in (0, 1).
    Returns:
        log_P: (B, N+1) — log_P[b, k] = log P(sum_{i=1..N} z_{b,i} = k).
    Complexity: O(N^2) per bag, O(B * N^2) total.
    """
```

**Algorithm** (log space, vectorized over batch):

```python
def forward_conv(p):
    B, N = p.shape
    log_p1 = p.clamp(min=EPS, max=1-EPS).log()           # (B, N)
    log_p0 = (1 - p).clamp(min=EPS, max=1-EPS).log()      # (B, N)
    log_P = p.new_full((B, N+1), LOG_ZERO)               # use LOG_ZERO, not -inf
    log_P[:, 0] = 0.0
    for i in range(N):
        # log_P_new[k] = logaddexp(log_p0[i] + log_P[k], log_p1[i] + log_P[k-1])
        stay  = log_p0[:, i:i+1] + log_P
        # shift along last axis: log_P[k-1], with LOG_ZERO at k=0
        shifted = F.pad(log_P[:, :-1], (1, 0), value=LOG_ZERO)
        shift = log_p1[:, i:i+1] + shifted
        log_P = torch.logaddexp(stay, shift)
    return log_P
```

**Sanity invariant** (test in Section 10.1):
- `log_P.exp().sum(dim=1) ≈ 1` for any valid `p`.

### 5.3 `leave_one_out_posterior(p, bag_y)`

**Signature**:
```python
def leave_one_out_posterior(p: torch.Tensor, bag_y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        p:     (B, N) per-instance Bernoulli probability.
        bag_y: (B,)   integer count target in [0, N].
    Returns:
        q: (B, N) detached posterior  q_u = P(z_u=1 | sum=bag_y).
    Complexity: O(N^2) per bag (prefix + suffix + N combines).
    """
```

**Math contract**:

$$
\text{prefix}[u][k] = P\Big(\sum_{i < u} z_i = k\Big), \quad \text{suffix}[u][k] = P\Big(\sum_{i > u} z_i = k\Big),
$$

$$
P_{j \neq u}(\sum = s) = (\text{prefix}[u] * \text{suffix}[u+1])[s] = \sum_k \text{prefix}[u][k] \cdot \text{suffix}[u+1][s-k],
$$

$$
q_u = \frac{p_u \cdot P_{j \neq u}(\sum = Y - 1)}{P(\sum = Y)}.
$$

All operations in log space.

**Algorithm**:

```python
def leave_one_out_posterior(p, bag_y):
    with torch.no_grad():
        B, N = p.shape
        log_p1 = p.clamp(min=EPS, max=1-EPS).log()           # (B, N)
        log_p0 = (1 - p).clamp(min=EPS, max=1-EPS).log()     # (B, N)
        
        # ---- Step 1: build prefix table.  prefix[:, i, :] = log PMF of d_0..d_{i-1}
        prefix = p.new_full((B, N+1, N+1), float('-inf'))
        prefix[:, 0, 0] = 0.0
        for i in range(N):
            stay  = log_p0[:, i:i+1] + prefix[:, i, :]
            shift = log_p1[:, i:i+1] + F.pad(prefix[:, i, :-1], (1, 0), value=float('-inf'))
            prefix[:, i+1, :] = torch.logaddexp(stay, shift)
        
        # ---- Step 2: build suffix table.  suffix[:, i, :] = log PMF of d_i..d_{N-1}
        suffix = p.new_full((B, N+1, N+1), float('-inf'))
        suffix[:, N, 0] = 0.0
        for i in reversed(range(N)):
            stay  = log_p0[:, i:i+1] + suffix[:, i+1, :]
            shift = log_p1[:, i:i+1] + F.pad(suffix[:, i+1, :-1], (1, 0), value=float('-inf'))
            suffix[:, i, :] = torch.logaddexp(stay, shift)
        
        # ---- Step 3: log P(sum = bag_y) — recovered from prefix[N], same as forward_conv(p)
        log_P_full = prefix[:, N, :]                                     # (B, N+1)
        log_P_y    = log_P_full.gather(1, bag_y.unsqueeze(1)).squeeze(1) # (B,)
        
        # ---- Step 4: for each u, compute log P_{j != u}(sum = bag_y - 1)
        #
        # Math: log_P_excl_at[b, u] = logsumexp_k [ prefix[b, u, k] + suffix[b, u+1, target - k] ]
        # where target = bag_y[b] - 1, and only k with 0 <= target-k <= N contribute.
        #
        # One implementation (loop over u, vectorized over batch and k):
        log_P_excl_at = p.new_full((B, N), float('-inf'))
        for u in range(N):
            # prefix_u: (B, N+1), suffix_u1: (B, N+1)
            prefix_u  = prefix[:, u, :]
            suffix_u1 = suffix[:, u+1, :]
            # 1D log-space convolution at index `target = bag_y - 1` for each bag b.
            # For each batch element, we need logsumexp over k of (prefix_u[b, k] + suffix_u1[b, target_b - k]).
            # Vectorize by indexing suffix_u1 with (target_b - k) per batch element:
            #   k_grid: (1, N+1)  → values 0..N
            #   target: (B, 1)    → bag_y - 1
            #   sk_idx = target - k_grid: (B, N+1); valid when in [0, N]
            k_grid = torch.arange(N+1, device=p.device).unsqueeze(0)        # (1, N+1)
            target = (bag_y - 1).unsqueeze(1)                                # (B, 1)
            sk_idx = target - k_grid                                          # (B, N+1)
            valid  = (sk_idx >= 0) & (sk_idx <= N)
            sk_idx_clamped = sk_idx.clamp(min=0, max=N)
            suffix_term = suffix_u1.gather(1, sk_idx_clamped)                 # (B, N+1)
            terms = torch.where(valid, prefix_u + suffix_term, torch.full_like(prefix_u, float('-inf')))
            log_P_excl_at[:, u] = torch.logsumexp(terms, dim=1)
        
        # ---- Step 5: posterior in log space
        log_q = log_p1 + log_P_excl_at - log_P_y.unsqueeze(1)               # (B, N)
        q = log_q.exp().clamp(min=0.0, max=1.0)
        
        # ---- Step 6: edge cases (numerical safety at boundaries)
        q = torch.where(bag_y.unsqueeze(1) == 0, torch.zeros_like(q), q)
        q = torch.where(bag_y.unsqueeze(1) == N, torch.ones_like(q),  q)
    
    return q  # already detached via torch.no_grad
```

The `for u in range(N)` loop is $O(N)$ iterations of $O(N)$-work each → $O(N^2)$ total. Implementation may further vectorize over $u$, but correctness is what tests assert (Section 10.1).

**Sanity invariants** (tests in Section 10.1):
- $\sum_u q_u \approx Y$ for all bags (posterior expected count = observed count).
- For $Y = 0$: $q \equiv 0$. For $Y = N$: $q \equiv 1$.
- N=2 hand-computed case matches.

### 5.4 `marginal_nll_loss(logits, bag_y)`

```python
def marginal_nll_loss(logits: torch.Tensor, bag_y: torch.Tensor) -> torch.Tensor:
    """
    Shukla-style aggregate NLL.
    Args:
        logits: (B, N)
        bag_y:  (B,) long
    Returns: scalar mean over batch.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()
```

### 5.5 `em_joint_loss(logits, bag_y, lam=1.0)` — Approach A

```python
def em_joint_loss(
    logits: torch.Tensor,
    bag_y: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        logits: (B, N)
        bag_y:  (B,)
        lam:    weight on the soft cross-entropy. lam=0.0 must be numerically equivalent
                to marginal_nll_loss within 1e-6.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    nll = -log_P_y.mean()
    
    if lam == 0.0:
        return nll  # exact equivalence, no posterior computation
    
    q = leave_one_out_posterior(p, bag_y)  # detached
    log_p1 = p.clamp(min=EPS, max=1-EPS).log()
    log_p0 = (1 - p).clamp(min=EPS, max=1-EPS).log()
    ce = -(q * log_p1 + (1 - q) * log_p0).mean()
    
    return nll + lam * ce
```

### 5.6 What's *not* in `losses.py`

- Signed counting (atom length 3) — A1 future work.
- Multi-class atoms (length K+1) — A1 future work.
- FFT-based conv — A2 future work; sequential is sufficient for $N \le 100$.
- M-step regularizers (entropy, sparsity) — added later only if F2 ablation is triggered.

---

## 6. Data — `pca/data.py`

### 6.1 `MNISTBagDataset`

**Signature**:
```python
class MNISTBagDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bag_size: int,           # N
        num_bags: int,           # number of bags to generate
        positive_digit: int = 9, # Shukla / Ilse convention
        train: bool = True,      # MNIST train (60k) vs test (10k) pool
        seed: int = 0,           # for reproducibility
    ):
        ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> tuple[Tensor, int, Tensor]:
        """
        Returns:
            images:    (N, 1, 28, 28) float, normalized [0, 1]
            bag_count: int — number of positive_digit instances in bag (training target)
            gt_labels: (N,) int 0/1 — per-instance ground truth (eval only, never for loss)
        """
```

**Bag construction**:
- For each bag index `b` (with reproducible per-bag seed derived from `seed + b`):
  - Sample $N$ MNIST indices **without replacement** from the chosen pool.
  - `images[i]`: the i-th sampled image.
  - `gt_labels[i] = 1` iff sampled label == `positive_digit` else 0.
  - `bag_count = gt_labels.sum()`.
- Bags are **independent** (sampling with replacement across bags is fine).
- Use `torchvision.datasets.MNIST(root=DEFAULT_TORCHVISION_ROOT, train=train, download=True)`.

**Defaults used by scripts**:
- Train pool: 1000 bags per (N, method, seed) combination.
- Test pool: 500 bags per (N, seed) combination — shared across methods for paired comparison.

### 6.2 `LLPBagDataset`

**Signature**:
```python
class LLPBagDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: Literal['adult', 'magic'],
        bag_size: int,
        num_bags: int,
        train: bool = True,
        seed: int = 0,
    ):
        ...
    def __getitem__(self, idx) -> tuple[Tensor, int, Tensor]:
        """
        Returns:
            features:  (N, D) — D ≈ 108 for Adult, D = 10 for Magic
            bag_count: int
            gt_labels: (N,) int — eval only
        """
```

**Adult preprocessing**:
1. `sklearn.datasets.fetch_openml('adult', version=2, as_frame=True)`.
2. Drop rows with `?` markers in raw features.
3. Categorical columns → one-hot via `pd.get_dummies`.
4. Numerical columns → z-score standardize using **training-pool statistics only**.
5. Target: `>50K` → 1, `<=50K` → 0.
6. Cache the preprocessed `(X_train, y_train, X_test, y_test)` tensors at `~/.cache/pca/llp/adult.pt`. Subsequent loads bypass preprocessing.

**Magic preprocessing**:
1. `sklearn.datasets.fetch_openml('MagicTelescope', version=1, as_frame=True)`.
2. All features numerical, z-score standardize.
3. Target: `g` → 1 (gamma), `h` → 0 (hadron). Or swap; convention pinned in code.
4. Cache at `~/.cache/pca/llp/magic.pt`.

**Train/test split**: Adult has fixed split. Magic uses 80/20 stratified split with fixed seed (independent of bag-construction seed).

**Bag construction**: identical to MNIST — sample N indices without replacement from instance pool.

### 6.3 Data tests (Section 10.2)

Required invariants:
- `bag_count == gt_labels.sum().item()` for every bag.
- Image shapes / feature dims match expectation.
- Same `seed` → same bags (determinism).

---

## 7. Models — `pca/models.py`

### 7.1 `SmallCNN` for MNIST-MIL

```python
class SmallCNN(nn.Module):
    """
    Per-instance MNIST → single logit. Shukla / Ilse-2018 scale.
    ~206k params total (the Linear(1568, 128) layer dominates).
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 → 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x):  # x: (B*N, 1, 28, 28)
        return self.classifier(self.features(x))   # (B*N, 1)
```

### 7.2 `MLP` for LLP

```python
class MLP(nn.Module):
    """Per-instance tabular feature vector → single logit."""
    def __init__(self, in_dim: int, hidden: list[int] = (64, 32), dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Dropout(dropout), nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B*N, D)
        return self.net(x)
```

### 7.3 Per-bag forward pattern (used by `train.py`)

```python
# images: (B, N, 1, 28, 28)
B, N = images.shape[:2]
flat = images.reshape(B * N, *images.shape[2:])
logits_flat = backbone(flat)        # (B*N, 1)
logits = logits_flat.view(B, N)     # (B, N) — fed to losses
```

---

## 8. Training — `pca/train.py`

### 8.1 Device selection (single source of truth)

```python
def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
```

### 8.2 `train(...)`

```python
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],   # marginal_nll_loss or partial(em_joint_loss, lam=...)
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """
    Returns metrics dict with schema in Section 12.1.
    """
```

**Per-bag forward helper** (defined once at top of `train.py`, reused by train and eval phases):

```python
def per_bag_forward(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    batch: (B, N, *features)   — features may be (1, 28, 28) for MNIST or (D,) for LLP.
    Returns: logits (B, N).
    """
    B, N = batch.shape[:2]
    flat = batch.reshape(B * N, *batch.shape[2:])
    logits_flat = model(flat).squeeze(-1)
    return logits_flat.view(B, N)
```

**Per-epoch flow**:
```python
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for images, bag_count, _gt in train_loader:
        images, bag_count = images.to(device), bag_count.to(device)
        logits = per_bag_forward(model, images)            # (B, N)
        loss = loss_fn(logits, bag_count)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.shape[0]
    epoch_train_loss.append(epoch_loss / len(train_loader.dataset))
    
    # Evaluation on test
    model.eval()
    with torch.no_grad():
        all_p, all_z = [], []
        all_bag_correct, total_bags = 0, 0
        for images, bag_count, gt in test_loader:
            images = images.to(device)
            logits = per_bag_forward(model, images)
            p = torch.sigmoid(logits)
            all_p.append(p.cpu().flatten())
            all_z.append(gt.flatten())
            
            log_P = forward_conv(p)
            pred_count = log_P.argmax(dim=1).cpu()
            all_bag_correct += (pred_count == bag_count).sum().item()
            total_bags += bag_count.shape[0]
    
    inst_auc = sklearn.metrics.roc_auc_score(
        torch.cat(all_z).numpy(),
        torch.cat(all_p).numpy()
    )
    bag_acc = all_bag_correct / total_bags
    epoch_inst_auc.append(inst_auc)
    epoch_bag_acc.append(bag_acc)
```

**`best_*` is the mean of the last 5 epochs** (smooths single-epoch noise without requiring early stopping).

### 8.3 `run_seeds(...)`

```python
def run_seeds(
    seeds: list[int],
    build_fn: Callable[[int], tuple],   # returns (model, train_loader, test_loader, loss_fn, optimizer)
    num_epochs: int,
    device: torch.device,
    save_dir: pathlib.Path,
    config_extra: dict = None,
) -> None:
    """
    For each seed:
      1. Set torch / numpy / random seeds.
      2. build_fn(seed) → fresh model + dataloaders + loss + optimizer.
      3. train(...).
      4. Persist metrics + config to save_dir / f'seed{seed}.json'.
    Sequential. No multiprocessing.
    """
```

**Seeding policy**:
- `torch.manual_seed(seed)`
- `np.random.seed(seed)`
- `random.seed(seed)`
- For paired comparison validity: `build_fn(seed)` must produce **identical data** for both NLL and EM with same `seed`. Method differs only via `loss_fn`. Model init also identical (same seed, same architecture).

### 8.4 Hyperparameters (pinned)

| Parameter | MNIST-MIL | LLP Adult/Magic | Rationale |
|---|---|---|---|
| Optimizer | Adam | Adam | Shukla convention |
| Learning rate | 1e-3 | 1e-3 | CellSelection.tex line 630 |
| Batch size | 32 (N≤50), 16 (N=100) | 32 (N≤128), 8 (N=512) | Memory bound on MPS |
| Epochs | 100 | 200 | LLP needs more (weaker per-bag signal) |
| Dropout | 0.5 | 0.5 | Standard |
| `lam` (EM only) | 1.0 | 1.0 | Default; sweep is stretch (Section 11.4) |
| Train bags | 1000 | 2000 | Per (N, method, seed) |
| Test bags | 500 | 1000 | Per (N, seed), shared across methods |
| Seeds | {0, 1, 2, 3, 4} | {0, 1, 2, 3, 4} | 5 for paired t-test |

---

## 9. Scripts

### 9.1 `scripts/sanity_n4.py`

Run after `pytest tests/` passes. Tests end-to-end gradient/loss equivalence and one optimizer step.

**Output on success**: `[OK] All sanity checks passed.`
**Output on failure**: prints which check failed with diagnostic values, exits with code 1.

**Checks**:
1. `forward_conv` on hand-computed N=2 case (matches `test_forward_conv_n2_hand`).
2. `leave_one_out_posterior` on hand-computed N=2 case.
3. `marginal_nll_loss(logits, y) == em_joint_loss(logits, y, lam=0)` numerically (1e-6 tol) on a random N=4 batch.
4. Gradient cosine similarity between NLL and EM(lam=0) ≥ 0.9999, on a random N=4 batch with backbone params.
5. `q.sum(dim=1) ≈ bag_y.float()` on N=10 batch (1e-3 tol).
6. `lam=1.0` loss + one Adam step → no NaN, no Inf, parameters changed.

### 9.2 `scripts/run_mnist_mil.py`

```
Usage:
  python scripts/run_mnist_mil.py \
    --bag-size {10,50,100} \
    --method {nll,em} \
    --epochs 100 \
    --seeds 0,1,2,3,4 \
    --output-dir results/mnist_mil/
```

**Behavior**:
1. Build `MNISTBagDataset` train + test for each seed.
2. `loss_fn`: `marginal_nll_loss` if method='nll' else `partial(em_joint_loss, lam=1.0)`.
3. `run_seeds` over the 5 seeds, sequential.
4. Save to `output-dir/N{bag-size}_{method}/seed{i}.json`.

**Smoke-test mode**:
```
python scripts/run_mnist_mil.py --bag-size 10 --method nll --epochs 2 --num-bags 50 --seeds 0
```
Should complete in under 5 minutes on MPS. Use this on Day 3 entry to detect breakage.

### 9.3 `scripts/run_llp_adult.py`

```
Usage:
  python scripts/run_llp_adult.py \
    --bag-size {32,128,512} \
    --method {nll,em} \
    --epochs 200 \
    --seeds 0,1,2,3,4 \
    --output-dir results/llp_adult/
```

Identical pattern, with `LLPBagDataset('adult', ...)` and `MLP(in_dim=108)`.

### 9.4 `scripts/analyze.py`

```
Usage:
  python scripts/analyze.py --results-dir results/mnist_mil/
  python scripts/analyze.py --results-dir results/llp_adult/
```

**Behavior**:
1. Glob `results-dir / 'N*_*' / 'seed*.json'`.
2. Aggregate per (N, method) into mean, std, list of per-seed `best_inst_auc`.
3. Compute H1, H2, H3 (Section 11).
4. Emit markdown table to stdout AND save to `results-dir / 'summary.md'`.
5. Save machine-readable summary to `results-dir / 'summary.json'`.
6. Print final verdict (`PROCEED_TO_LLP` / `PROCEED_BUT_WEAK` / `F2_ABLATION_NEEDED` / `DEMOTE_B1`).

---

## 10. Testing strategy

### 10.1 Unit tests — `tests/test_losses.py`

These tests are the spine of correctness. Run with `pytest tests/`. Total runtime under 5 seconds.

| Test name | Asserts |
|---|---|
| `test_forward_conv_n2_hand` | `forward_conv([[0.3, 0.4]]) → exp ≈ [0.42, 0.46, 0.12]` (1e-6) |
| `test_forward_conv_n3_hand` | Hand-computed N=3 case matches |
| `test_forward_conv_sums_to_one` | `log_P.exp().sum(1) ≈ 1` for random 8×50 input (1e-5) |
| `test_posterior_n2_hand` | `leave_one_out_posterior([[0.3, 0.4]], [1]) ≈ [[0.391, 0.609]]` (1e-4) |
| `test_posterior_sum_equals_count` | $\sum_u q_u \approx Y$ on random 8×20 (1e-3) |
| `test_posterior_y_zero` | `bag_y == 0` → `q ≡ 0` |
| `test_posterior_y_n` | `bag_y == N` → `q ≡ 1` |
| `test_em_loss_lam_zero_equals_nll` | Numerical equivalence on random 4×10 (1e-6) |
| `test_em_loss_lam_zero_grad_cosine` | Gradient cosine sim ≥ 0.9999 on random 4×10 |
| `test_em_loss_no_nan_at_extremes` | `lam=1, p=0.999`, bag_y in {0, N} → finite loss, finite grad |

### 10.2 Data tests — `tests/test_data.py`

| Test | Asserts |
|---|---|
| `test_mnist_bag_count_matches_gt` | `bag_count == gt.sum()` for all bags |
| `test_mnist_image_shape` | `images.shape == (N, 1, 28, 28)` |
| `test_mnist_seed_determinism` | Same seed → byte-identical images |
| `test_llp_adult_loads_and_caches` | First call fetches+caches; second call uses cache |
| `test_llp_feature_dim_correct` | Adult D ≈ 108, Magic D = 10 |

### 10.3 Smoke test (script-level)

`scripts/run_mnist_mil.py --epochs 2 --num-bags 50 --seeds 0` must finish without error, produce a valid JSON. Run on Day 3 entry.

### 10.4 Test gates (enforced)

- **Day 1 end**: `pytest tests/` must pass before running `sanity_n4.py`.
- **Day 2 end**: `sanity_n4.py` must pass before any MNIST-MIL run.
- **Day 3 entry**: smoke test passes before full MNIST-MIL.
- **Day 5**: `analyze.py` produces a verdict.

---

## 11. Hypothesis evaluation logic — `analyze.py`

### 11.1 Aggregation

```python
@dataclass
class CellStat:
    mean: float
    std: float
    seeds: list[float]   # raw per-seed best_inst_auc

# summary[N][method] = CellStat
def aggregate(results_dir: Path) -> dict[int, dict[str, CellStat]]:
    ...
```

### 11.2 Hypothesis evaluation

```python
def evaluate_h1(summary, target_N: int) -> dict:
    """H1: AUC(EM, target_N) - AUC(NLL, target_N) >= 1.0 pp, paired t-test p < 0.05"""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    delta_pp = (em.mean - nl.mean) * 100
    
    diffs = np.array(em.seeds) - np.array(nl.seeds)
    t_stat, p_value = scipy.stats.ttest_rel(em.seeds, nl.seeds)
    
    return {
        'pass': delta_pp >= 1.0 and p_value < 0.05,
        'delta_pp': delta_pp,
        'p_value': p_value,
        'paired_diffs_pp': (diffs * 100).tolist(),
    }

def evaluate_h2(summary, sizes: list[int]) -> dict:
    """H2: deltas are monotone non-decreasing across sizes (0.2 pp slack)."""
    deltas_pp = [(summary[N]['em'].mean - summary[N]['nll'].mean) * 100 for N in sizes]
    monotone = all(deltas_pp[i] <= deltas_pp[i+1] + 0.2 for i in range(len(deltas_pp) - 1))
    return {'pass': monotone, 'deltas_pp': deltas_pp}

def evaluate_h3(summary, target_N: int) -> dict:
    """H3: std(EM, target_N) <= 0.7 * std(NLL, target_N)."""
    em = summary[target_N]['em']
    nl = summary[target_N]['nll']
    ratio = em.std / nl.std if nl.std > 0 else float('inf')
    return {'pass': ratio <= 0.7, 'std_ratio': ratio, 'em_std': em.std, 'nll_std': nl.std}
```

### 11.3 Verdict

```python
def verdict(h1, h2, h3) -> str:
    delta = h1['delta_pp']
    
    # Tier 1: clear demote
    if delta < 0.5:
        return 'DEMOTE_B1'
    
    # Tier 2: marginal mean delta — try variance reduction via F2
    if delta < 1.0:
        return 'F2_ABLATION_NEEDED'
    
    # Tier 3: delta >= 1.0
    # If high mean but high variance (paired t-test fails): try F2 to tighten
    if not h1['pass']:
        return 'F2_ABLATION_NEEDED'
    
    # Tier 4: H1 passes outright (delta >= 1.0 AND p < 0.05)
    if h2['pass'] or h3['pass']:
        return 'PROCEED_TO_LLP'
    return 'PROCEED_BUT_WEAK'
```

**Truth table** (all 4 outcomes covered):

| `delta` | `h1['pass']` | (h2 or h3) | Verdict |
|---|---|---|---|
| < 0.5 | — | — | `DEMOTE_B1` |
| [0.5, 1.0) | False | — | `F2_ABLATION_NEEDED` |
| ≥ 1.0 | False (p ≥ 0.05) | — | `F2_ABLATION_NEEDED` |
| ≥ 1.0 | True | False | `PROCEED_BUT_WEAK` |
| ≥ 1.0 | True | True | `PROCEED_TO_LLP` |

### 11.4 F2 ablation (only if triggered)

If verdict is `F2_ABLATION_NEEDED`, add an entropy regularizer to the M-step CE term:

```python
# In em_joint_loss, after computing ce:
entropy = -(p * log_p1 + (1 - p) * log_p0).mean()
return nll + lam * ce + lam_ent * entropy   # lam_ent = 0.01
```

Rerun `run_mnist_mil.py --bag-size 50 --method em` only (5 seeds). Re-evaluate H1.

If F2 ablation also fails to clear 1.0 pp at N=50: B1 is demoted.

### 11.5 Reporting format

`summary.md` content:
```markdown
# Results — <experiment-name>
**Generated**: 2026-05-XX

## Instance-level AUC (mean ± std over 5 seeds)

| N   | NLL              | EM               | Δ (pp)       |
|-----|------------------|------------------|--------------|
| 10  | 0.8423 ± 0.0089  | 0.8467 ± 0.0061  | +0.44        |
| 50  | 0.9012 ± 0.0102  | 0.9151 ± 0.0078  | +1.39 ✓      |
| 100 | 0.9123 ± 0.0114  | 0.9298 ± 0.0089  | +1.75 ✓      |

## Hypothesis verdicts

- H1 (AUC delta @ N=50 ≥ 1pp, p<0.05): **PASS**  Δ = +1.39 pp, p = 0.012
- H2 (monotone in N):                  **PASS**  deltas_pp = [+0.44, +1.39, +1.75]
- H3 (std reduction ≥ 30%):            **FAIL**  ratio = 0.77

## Verdict: PROCEED_TO_LLP
```

---

## 12. Result storage

### 12.1 Per-run JSON schema

```json
{
  "config": {
    "experiment":  "mnist_mil",
    "bag_size":    50,
    "method":      "em",
    "lam":         1.0,
    "lr":          1e-3,
    "epochs":      100,
    "batch_size":  32,
    "num_train_bags": 1000,
    "num_test_bags":  500,
    "seed":        3,
    "device":      "mps"
  },
  "epoch_train_loss": [...],
  "epoch_inst_auc":   [...],
  "epoch_bag_acc":    [...],
  "best_inst_auc":    0.9421,
  "best_bag_acc":     0.812,
  "wall_clock_sec":   312.5,
  "git_sha":          "abc1234",
  "torch_version":    "2.x.x"
}
```

### 12.2 Git policy

- `results/<exp>/<setting>/seed*.json`: **gitignored** (large, regeneratable).
- `results/<exp>/summary.md` and `summary.json`: **committed** (small, decision-relevant).
- Spec file `docs/superpowers/specs/2026-05-03-b1-em-verification-design.md`: committed at design approval time.

`.gitignore` additions:
```
results/*/N*_*/
__pycache__/
*.pyc
.pytest_cache/
.venv/
~/.cache/pca/
```

---

## 13. Day-by-day plan

| Day | Deliverable | Gate |
|---|---|---|
| 1 | `pca/losses.py` + `tests/test_losses.py` (10 tests passing) | All unit tests pass |
| 2 | `pca/data.py` + `pca/models.py` + `pca/train.py` + `scripts/sanity_n4.py` | sanity_n4 passes |
| 3 | `scripts/run_mnist_mil.py` smoke (epochs=2, 1 seed, N=10) + 1 full run (N=10, NLL, 1 seed) | Smoke test passes; full run produces valid JSON |
| 4 | All 30 MNIST-MIL runs (5 seeds × 3 sizes × 2 methods) | All seed JSONs present |
| 5 | `analyze.py` + verdict | Verdict emitted |
| 6 | If `PROCEED_*`: `scripts/run_llp_adult.py` smoke + first runs | LLP smoke passes |
| 7 | All 30 LLP runs + `analyze.py` on LLP | Final verdict |

---

## 14. Decision points (gate definitions)

### 14.1 Day 1 → Day 2

`pytest tests/test_losses.py tests/test_data.py` returns 0 with all tests passing.
Total wall-clock: under 30 seconds.

### 14.2 Day 2 → Day 3

`python scripts/sanity_n4.py` prints `[OK] All sanity checks passed.` and returns 0.

### 14.3 Day 3 → Day 4

`python scripts/run_mnist_mil.py --bag-size 10 --method nll --epochs 2 --num-bags 50 --seeds 0` completes without error, JSON has expected schema (Section 12.1), `best_inst_auc` is a float in (0.5, 1.0).

### 14.4 Day 4 → Day 5

All 30 JSONs in `results/mnist_mil/N{10,50,100}_{nll,em}/seed{0..4}.json`. Each JSON validates against schema.

### 14.5 Day 5 (the critical gate)

`analyze.py` emits one of:
- `PROCEED_TO_LLP` → Day 6.
- `PROCEED_BUT_WEAK` → Day 6 with a marker that B1 is now positioned as a boost-axis in the paper (priority memo update).
- `F2_ABLATION_NEEDED` → run F2 ablation (Section 11.4), re-evaluate.
- `DEMOTE_B1` → STOP. Update `README.md` priority table (B1 → Low). Begin Track 1 step 3 (B2) instead.

### 14.6 Day 7

LLP analyze emits final week-1 status. README updated with B1 verdict (kept-as-High / boost / demoted).

---

## 15. What this spec does *not* cover (out of scope, deferred)

- B2 (bag mixup / additive consistency) implementation — separate spec when Track 1 step 3 starts.
- A1 multi-class atoms — separate spec for MNIST-sum experiment.
- A2 FFT/tree reduction — separate spec triggered only after digital-pathology benchmark is in scope.
- A4 cancellation lemma proof — separate Track 2 work, not code.
- Pieri-rule discovery / qt-LR application — out of project scope (see `README.md §3 결정 사항`).
- Hyperparameter sweep over `lam` — stretch; only if H1 passes weakly and `F2_ABLATION_NEEDED` is triggered.
- LLP Magic dataset — Adult is the primary; Magic is a secondary check only if Adult is ambiguous.
- PU benchmarks (Binarized MNIST etc.) — Track 3 step 3b, separate spec.

---

## 16. Dependencies and environment

### 16.1 `pyproject.toml` (uv-managed)

```toml
[project]
name = "pca"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "numpy>=1.26",
  "pandas>=2.1",
  "scikit-learn>=1.3",
  "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 16.2 Setup commands

```sh
cd /Users/bhwang/Documents/Projects/probabilistic-convolutional-aggregator
uv venv
uv pip install -e ".[dev]"
# Verify
uv run pytest tests/
uv run python scripts/sanity_n4.py
```

### 16.3 Compute environment

- Mac (Darwin), Apple Silicon → MPS device for PyTorch.
- CPU fallback available.
- Estimated wall-clock for full week-1 runs: 6–8 hours (sequential, no parallelization).

---

## Appendix A: Posterior closed-form via NLL gradient

Claim: $q_u = p_u \cdot (1 + (1 - p_u) \cdot a_u)$, where $a_u = \partial \log P(\sum=Y) / \partial p_u$.

**Derivation**: Let $A = P_{j \neq u}(\sum = Y)$ and $B = P_{j \neq u}(\sum = Y - 1)$. Both are independent of $p_u$.

The bag PMF satisfies:

$$
P(\sum = Y) = (1 - p_u) A + p_u B. \quad (\text{Eq. 1})
$$

Differentiating $\log P$ with respect to $p_u$:

$$
a_u = \frac{1}{P(\sum=Y)} \frac{\partial}{\partial p_u}[(1 - p_u) A + p_u B] = \frac{B - A}{P(\sum = Y)}. \quad (\text{Eq. 2})
$$

Solving (Eq. 1) and (Eq. 2) for $A, B$:

- From (Eq. 2): $B = A + a_u P(\sum = Y)$.
- Substituting into (Eq. 1): $P = (1 - p_u) A + p_u (A + a_u P) = A + p_u a_u P$.
- $\Rightarrow A = P (1 - p_u a_u)$, $B = P (1 + a_u (1 - p_u))$.

Then:
$$
q_u := P(z_u = 1 \mid \sum = Y) = \frac{p_u B}{P(\sum = Y)} = p_u (1 + a_u (1 - p_u)).
$$

**Implication**: Plain marginal NLL gradient descent is implicitly EM-like. Any empirical gain from explicit B1 is therefore a property of *optimization* (trajectory, variance, regularizer composition), not *information*. This grounds the framings (F1)/(F2)/(F3) in Section 1.3.

---

## Appendix B: References

- **Shukla et al. 2023** — "A Unified Approach to Count-Based Weakly Supervised Learning", NeurIPS 2023. Located at `sources/Shukla2023/camera_ready.tex`. Primary baseline; LLP/MIL/PU paradigms; Theorem 2 (LLP risk consistency) at lines `:1006`–`:1071`.
- **Ilse et al. 2018** — "Attention-based Deep Multiple Instance Learning", ICML 2018. Defines the MNIST-MIL benchmark adopted in Section 6.1.
- **Kobayashi 2022** — Reduction technique used in Shukla's risk-consistency proof. Foundation for A4 (separate spec).
- **`CellSelection.tex`** — Prior project draft. PCA method §4.1 (lines 428–493). Numerical results in §5–6 are application-specific and **not** baselines for this verification.
- **Project research notes** —
  - `README.md`: pivot plan, priority table.
  - `02_training_strengthening.md §B1`: this spec implements B1.
  - `03_evaluation_strategy.md §4 Track 1`: this spec is week-1 work of Track 1.
