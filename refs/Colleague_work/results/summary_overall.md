# A1 Verification — Final Summary

## Verdict: **A1_CONFIRMED**

_Selector: `tail5`_

## H1 — NLL primary

- pass=True; pass_count=2 (of 3 available)
  - mnist_sum: Δ=+0.452 nats, p=0.190, FAIL
  - svhn_sum: Δ=+0.969 nats, p=0.001, PASS
  - ultramnist: Δ=+2.070 nats, p=0.005, PASS

## H2 — atom-shape generality
- binary_b1_reuse: PASS (B1 NLL@N=50 inst_auc = 0.9985)
- multiclass_mnist_sum: PASS (PCA acc=0.635 vs max baseline=0.541, nll=3.290)
- multiclass_svhn_sum: PASS (PCA acc=0.245 vs max baseline=0.116, nll=3.653)
- multiclass_ultramnist: PASS (PCA acc=0.899 vs max baseline=0.572, nll=0.606)
- signed_mnist_signed: PASS (PCA acc=0.692 vs max baseline=0.302, nll=3.293)

## H3 — calibration
- pass=False; pass_count=1
  - mnist_sum: ratio=5.08, FAIL
  - svhn_sum: ratio=2.58, FAIL
  - ultramnist: ratio=0.62, PASS

## H4 — point-prediction supremacy (acc + MAE)
- pass=True; pass_count=3 (of 3 available)
  - mnist_sum: PCA acc=0.635 vs max_baseline=0.541; PCA mae=1.360 vs min_baseline=1.513; PASS
  - svhn_sum: PCA acc=0.245 vs max_baseline=0.116; PCA mae=3.081 vs min_baseline=3.665; PASS
  - ultramnist: PCA acc=0.899 vs max_baseline=0.572; PCA mae=0.401 vs min_baseline=0.963; PASS

## Saturation detection
- mnist_sum: clear
- svhn_sum: clear
- ultramnist: clear
