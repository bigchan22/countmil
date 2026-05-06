# Per-class-cap sweep result (Day 3 gate)

| cap | best_test_acc | best_test_nll |
|---|---|---|
| 200 | 0.635 | 1.567 |
| 100 | 0.618 | 1.946 |
|  50 | 0.580 | 2.659 |

**Decision (spec §3.6):** default cap = 100
**Rationale:** Rule 3 fired — cap=100 PCA top-1 acc (0.618) is in [0.50, 0.85], confirming cap=100 is neither too strict nor in the saturation regime (caps 100 and 200 differ by 1.7pp > 1pp, so saturation rule does not fire).
