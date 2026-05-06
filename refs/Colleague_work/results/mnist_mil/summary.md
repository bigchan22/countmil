# Results — mnist_mil

## Instance-level AUC (mean ± std over 5 seeds)

| N   | NLL              | EM               | Δ (pp)        |
|-----|------------------|------------------|---------------|
| 10  | 0.9986 ± 0.0007  | 0.9986 ± 0.0006  | -0.00 ✗      |
| 50  | 0.9985 ± 0.0009  | 0.9985 ± 0.0006  | +0.00 ✗      |
| 100 | 0.9986 ± 0.0016  | 0.9987 ± 0.0017  | +0.00 ✗      |

## Hypothesis verdicts

- H1 (Δ @ N=50 ≥ 1pp, p<0.05): **FAIL**  Δ = +0.00 pp, p = 0.816
- H2 (monotone in N): **PASS**  deltas_pp = [-0.00, +0.00, +0.00]
- H3 (std reduction ≥ 30%): **PASS**  ratio = 0.65

## Verdict: DEMOTE_B1
