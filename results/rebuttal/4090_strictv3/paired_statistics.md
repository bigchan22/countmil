# Strict-v3 CIFAR Paired Statistics

Paired differences are method A - method B; negative is better for Count MAE and Composite NLL.
Bootstrap uses 10,000 replicates, resampling seeds and then paired test bags within each sampled seed.

| Comparison | Metric | Mean diff. | 95% CI |
|---|---|---:|---:|
| fsconv count minus ce kl | count_mae | 0.1286 | [0.0610, 0.2432] |
| fsconv count minus ce kl | composite_nll | -0.0939 | [-0.1652, -0.0243] |
| official llp pvc minus ce kl | count_mae | 5.0427 | [4.5093, 5.4341] |
| official llp pvc minus ce kl | composite_nll | 3.7944 | [3.2505, 4.1414] |
| fsconv count minus official llp pvc | count_mae | -4.9137 | [-5.3592, -4.4462] |
| fsconv count minus official llp pvc | composite_nll | -3.8881 | [-4.2313, -3.4166] |
