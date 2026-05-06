# Results — llp_adult

## Instance-level AUC (mean ± std over 5 seeds)

| N   | NLL              | EM               | Δ (pp)        |
|-----|------------------|------------------|---------------|
| 32  | 0.8635 ± 0.0062  | 0.8645 ± 0.0052  | +0.10 ✗      |
| 128 | 0.8704 ± 0.0050  | 0.8715 ± 0.0037  | +0.11 ✗      |

## Hypothesis verdicts

- H1 (Δ @ N=128 ≥ 1pp, p<0.05): **FAIL**  Δ = +0.11 pp, p = 0.501
- H2 (monotone in N): **PASS**  deltas_pp = [+0.10, +0.11]
- H3 (std reduction ≥ 30%): **FAIL**  ratio = 0.73

## Verdict: DEMOTE_B1
