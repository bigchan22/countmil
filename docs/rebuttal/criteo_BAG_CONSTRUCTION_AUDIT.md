# Criteo Strict-v1 Bag Construction Audit

- Protocol tag: `criteo_x1_c4_c11_k128_strictv1`
- Grouping key: `C4`, `C11`
- Bag size: `128`
- Selection is label-blind and based on deterministic MD5 ordering.

| Seed | Split | Candidate full bags | Selected bags | Retained instances | Click prevalence | Count mean | Count std | Zero-click % | Max count | Manifest hash | Input hash |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | train | 118667 | 12000 | 1536000 | 0.270087 | 34.571 | 21.116 | 0.21 | 125 | `517ed3c19160` | `7d6feab1c8b7` |
| 0 | valid | 19703 | 3000 | 384000 | 0.274096 | 35.084 | 21.081 | 0.20 | 128 | `e7b114a87dc3` | `afa548019995` |
| 0 | test | 8707 | 5000 | 640000 | 0.284263 | 36.386 | 21.326 | 0.08 | 126 | `f7b1d485ad82` | `c126149c1707` |
| 1 | train | 118667 | 12000 | 1536000 | 0.273061 | 34.952 | 21.528 | 0.19 | 128 | `e7b4a2f6d277` | `fadd5457d815` |
| 1 | valid | 19703 | 3000 | 384000 | 0.281641 | 36.050 | 21.265 | 0.07 | 119 | `c968b55fa6d8` | `d6979c859e32` |
| 1 | test | 8707 | 5000 | 640000 | 0.281672 | 36.054 | 20.934 | 0.14 | 126 | `e3ad2519a961` | `84ac39c6b934` |
| 2 | train | 118667 | 12000 | 1536000 | 0.271942 | 34.809 | 21.372 | 0.16 | 128 | `e8645aa9d40f` | `e4f0f28a39dd` |
| 2 | valid | 19703 | 3000 | 384000 | 0.275620 | 35.279 | 21.141 | 0.13 | 128 | `2f918435b15a` | `6a01875f9909` |
| 2 | test | 8707 | 5000 | 640000 | 0.282438 | 36.152 | 21.146 | 0.12 | 116 | `a4607ccda92d` | `c4c1733ff87c` |
