# CountMIL Remote Codex Handoff

We are preparing NeurIPS experiments for finite-support convolutional aggregate likelihoods in weakly supervised Count-MIL.

Do not claim binary Count Loss is new. Binary Bernoulli counts recover the Shukla-style count probability. The story is:

- unified finite-support convolution view,
- practical grouped GPU implementation,
- signed/cancellation aggregates,
- ordinal finite-support sums such as digit-sum,
- count-conditioned posterior marginals for instance recovery,
- comparison to attention MIL, LLP/proportion matching, and expected-sum baselines.

## Rules

- Do not download datasets unless explicitly asked.
- Do not overwrite existing `results/`, `runs/`, or checkpoint files.
- Use unique result roots per server.
- Put reusable code in `src/`, long scripts in `scripts/`, configs under `configs/`, and outputs under `results/`.
- Run `PYTHONPATH=src .venv/bin/python -m pytest` after code changes.
- Do not commit or push unless asked by the user.

## Current Priority

Main 4090 server is running the primary 72-job pilot. Extra servers should fill comparison gaps:

- 3090 server: expected-sum baselines and multiclass histogram LLP for MNIST.
- A5000 server: shorter FashionMNIST robustness jobs and benchmark replication.

Use `docs/REMOTE_EXPERIMENTS.md` for concrete commands.

The most important paper comparisons still missing are the MNIST digit-sum expected-sum baselines and multiclass histogram LLP/proportion-matching baselines.
