# CountMIL project instructions for Codex

## Project goal
We are preparing experiments for a NeurIPS paper on finite-support convolutional aggregate likelihoods for weakly supervised Count-MIL.

The central idea:
- A bag contains instances x_i.
- Instance labels/contributions are latent.
- The observed label is an aggregate integer value such as a count, signed count, or sum.
- We model the aggregate distribution by convolving finite-support atomic PMFs.

For binary Count-MIL:
    d_i = (1 - p_i, p_i)
    P(Y = .) = d_1 * ... * d_N

For signed Count-MIL:
    Y = sum_i s_i z_i, s_i in {-1, +1}

For ordinal finite-support sums:
    Z_i in {0, ..., K}
    Y = sum_i Z_i

## Important positioning
- Do not claim we invented binary Count Loss.
- For ordinary Bernoulli counts, our likelihood recovers Shukla et al.'s count probability.
- The novelty we want to test is:
  1. convolutional finite-support view,
  2. signed / cancellation aggregates,
  3. ordinal finite-support sums such as MNIST digit-sum,
  4. count-conditioned posterior marginals for instance recovery,
  5. practical grouped GPU implementation.

## Experiments to implement first
Priority 0: exactness/runtime microbenchmark
- Compare convolutional PMF against brute force and DP.
- Test binary, signed, and categorical finite-support atoms.
- Measure forward and forward+backward runtime on CPU/GPU.

Priority 1: MNIST-Bags
- Binary MIL label: contains digit 9.
- Exact count label: number of digit 9s.
- Evaluate bag metrics and hidden instance recovery.

Priority 2: MNIST digit-sum
- Bag label is sum of digit values.
- Instance digit labels are hidden during training.
- Evaluate aggregate sum accuracy/NLL and instance digit accuracy.

Priority 3: signed Count-MIL
- Signed MNIST first.
- Y = sum_i s_i z_i.
- Pay special attention to cancellation-heavy bags where Y=0.

Priority 4: posterior EM / posterior matching
- Compute q_i = P(z_i = 1 | sum_j z_j = Y).
- Test plain NLL, NLL+entropy, soft EM, tempered EM, and hard EM.
- Evaluate whether posterior EM sharpens p_i and improves instance recovery.

Optional:
- PROPOR-style LLP comparison.
- CIFAR-100 multi-constraint counts.
- CelebA attribute-counts.
- LR coefficient case study as a short math/science case.

## Coding rules
- Do not pre-generate millions of bags to disk. Sample bags on the fly.
- Keep dataset root configurable, default `data/`.
- Do not download datasets unless explicitly asked.
- Use clear module names:
  - `src/countmil/aggregators.py`
  - `src/countmil/posteriors.py`
  - `src/countmil/datasets/`
  - `src/countmil/models/`
  - `src/countmil/metrics.py`
- Add tests before large training scripts.
- Write tests for numerical equality:
  - brute force vs DP,
  - DP vs convolution,
  - posterior marginals sum to observed count in binary unsigned case.
- Always report what files were changed and what tests were run.
