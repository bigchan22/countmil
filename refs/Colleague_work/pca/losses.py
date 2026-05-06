"""Loss functions for the Probabilistic Convolutional Aggregator (PCA).

All routines work in log space for numerical stability. See
docs/superpowers/specs/2026-05-03-b1-em-verification-design.md §5 for math.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-7

# Log-zero sentinel: avoids logaddexp(-inf, -inf) → NaN gradient
# (exp(-inf - (-inf)) = exp(NaN) = NaN). -1e30 gives finite (0.5, 0.5)
# backward, with exp(-1e30) ≈ 0 to far below float32 precision so forward
# values at structurally unreachable positions are unchanged. Used in
# forward_conv (which is differentiable). leave_one_out_posterior keeps
# float('-inf') because it runs under torch.no_grad().
LOG_ZERO = -1e30


def forward_conv(p: Tensor) -> Tensor:
    """Compute log P(sum z_i = k), k = 0..N, via sequential conv in log space.

    Args:
        p: (B, N) per-instance Bernoulli probability ∈ (0, 1).
    Returns:
        log_P: (B, N+1) — log_P[b, k] = log P(sum z_{b,i} = k).
    Complexity: O(B * N^2).
    Autograd-safe at structurally unreachable positions (uses LOG_ZERO sentinel).
    """
    B, N = p.shape
    log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
    log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()
    log_P = p.new_full((B, N + 1), LOG_ZERO)
    log_P[:, 0] = 0.0
    for i in range(N):
        stay = log_p0[:, i:i + 1] + log_P
        shifted = F.pad(log_P[:, :-1], (1, 0), value=LOG_ZERO)
        shift = log_p1[:, i:i + 1] + shifted
        log_P = torch.logaddexp(stay, shift)
    return log_P


def leave_one_out_posterior(p: Tensor, bag_y: Tensor) -> Tensor:
    """Posterior q_u = P(z_u=1 | sum=bag_y) via prefix·suffix conv in log space.

    Args:
        p:     (B, N) per-instance Bernoulli probability ∈ (0, 1).
        bag_y: (B,)   integer count target ∈ [0, N].
    Returns:
        q: (B, N) detached posterior.
    Complexity: O(B * N^2).
    """
    with torch.no_grad():
        # Note: uses float('-inf') (not LOG_ZERO) because the no_grad
        # context disables autograd, so the logaddexp(-inf,-inf) NaN-gradient
        # bug that motivated LOG_ZERO in forward_conv does not apply here.
        B, N = p.shape
        log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
        log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()

        # prefix[:, i, :] = log PMF of d_0..d_{i-1}
        prefix = p.new_full((B, N + 1, N + 1), float('-inf'))
        prefix[:, 0, 0] = 0.0
        for i in range(N):
            stay = log_p0[:, i:i + 1] + prefix[:, i, :]
            shift = log_p1[:, i:i + 1] + F.pad(prefix[:, i, :-1], (1, 0), value=float('-inf'))
            prefix[:, i + 1, :] = torch.logaddexp(stay, shift)

        # suffix[:, i, :] = log PMF of d_i..d_{N-1}
        suffix = p.new_full((B, N + 1, N + 1), float('-inf'))
        suffix[:, N, 0] = 0.0
        for i in reversed(range(N)):
            stay = log_p0[:, i:i + 1] + suffix[:, i + 1, :]
            shift = log_p1[:, i:i + 1] + F.pad(suffix[:, i + 1, :-1], (1, 0), value=float('-inf'))
            suffix[:, i, :] = torch.logaddexp(stay, shift)

        # log P(Σ=bag_y) — recovered from prefix[N], same as forward_conv(p)
        log_P_full = prefix[:, N, :]                                       # (B, N+1)
        log_P_y = log_P_full.gather(1, bag_y.unsqueeze(1)).squeeze(1)      # (B,)

        # log P_{j≠u}(Σ=bag_y - 1) via 1-D conv in log space
        log_P_excl_at = p.new_full((B, N), float('-inf'))
        k_grid = torch.arange(N + 1, device=p.device).unsqueeze(0)         # (1, N+1)
        target = (bag_y - 1).unsqueeze(1)                                   # (B, 1)
        for u in range(N):
            sk_idx = target - k_grid                                       # (B, N+1)
            valid = (sk_idx >= 0) & (sk_idx <= N)
            sk_clamped = sk_idx.clamp(min=0, max=N)
            suffix_term = suffix[:, u + 1, :].gather(1, sk_clamped)        # (B, N+1)
            terms = torch.where(valid, prefix[:, u, :] + suffix_term,
                                torch.full_like(prefix[:, u, :], float('-inf')))
            log_P_excl_at[:, u] = torch.logsumexp(terms, dim=1)

        log_q = log_p1 + log_P_excl_at - log_P_y.unsqueeze(1)
        q = log_q.exp().clamp(min=0.0, max=1.0)

        # Edge cases
        q = torch.where(bag_y.unsqueeze(1) == 0, torch.zeros_like(q), q)
        q = torch.where(bag_y.unsqueeze(1) == N, torch.ones_like(q), q)

    return q


def marginal_nll_loss(logits: Tensor, bag_y: Tensor) -> Tensor:
    """Shukla-style aggregate NLL: -log P(Σ z = bag_y).

    Args:
        logits: (B, N) per-instance pre-sigmoid scores.
        bag_y:  (B,)  long integer in [0, N].
    Returns:
        scalar mean over batch.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()


def em_joint_loss(logits: Tensor, bag_y: Tensor, lam: float = 1.0) -> Tensor:
    """Approach A joint loss: marginal NLL + λ × CE(p, q.detach()).

    `lam=0.0` returns marginal_nll_loss exactly (numerical equivalence,
    no posterior computation). See spec §3.3 and §5.5.

    Args:
        logits: (B, N) per-instance pre-sigmoid scores.
        bag_y:  (B,)  long integer in [0, N].
        lam:    weight on the soft cross-entropy.
    Returns:
        scalar mean over batch.
    """
    p = torch.sigmoid(logits)
    log_P = forward_conv(p)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    nll = -log_P_y.mean()

    if lam == 0.0:
        return nll

    q = leave_one_out_posterior(p, bag_y)
    log_p1 = p.clamp(min=EPS, max=1 - EPS).log()
    log_p0 = (1 - p).clamp(min=EPS, max=1 - EPS).log()
    ce = -(q * log_p1 + (1 - q) * log_p0).mean()
    return nll + lam * ce


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
    Autograd-safe at structurally unreachable positions (uses LOG_ZERO sentinel).
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


def multiclass_marginal_nll_loss(
    logits: Tensor,
    bag_y: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Multi-class generalization of marginal_nll_loss.

    Args:
        logits: (B, N, S) per-instance pre-softmax logits over atom support.
        bag_y:  (B,) integer bag sum, already shifted to [0, T-1] by caller
                (T = N*(S-1)+1). For signed atoms ({-1,0,+1}) the caller
                passes shifted bag_y (true_y + N).
        mask:   (B, N) bool, optional. True for active instances. Inactive
                instances are forced to atom = delta_0 = [1, 0, ..., 0],
                contributing identity to the convolution and ignored in the
                bag PMF. Handles variable bag size without a separate path.
    Returns:
        scalar — mean NLL over batch.
    """
    B, N, S = logits.shape
    log_atoms = F.log_softmax(logits, dim=-1)
    if mask is not None:
        delta_0 = log_atoms.new_full((S,), LOG_ZERO)
        delta_0[0] = 0.0
        log_atoms = torch.where(mask.unsqueeze(-1), log_atoms, delta_0)
    log_P = atomic_conv(log_atoms)
    log_P_y = log_P.gather(1, bag_y.unsqueeze(1)).squeeze(1)
    return -log_P_y.mean()
