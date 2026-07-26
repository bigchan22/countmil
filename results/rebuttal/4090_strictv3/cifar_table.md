# Strict-v3 Fixed-Bag CIFAR-10 Table

Frozen ImageNet ResNet-18 features; 45k/5k stratified train/validation split from official train; official test split final-only.
Bag size 64, 250 train bags, 250 validation bags, 1000 test bags. Mean +/- sample standard deviation over seeds 0,1,2.

| Method | Seeds | Unique-image acc. | Macro-F1 | Count MAE | Classwise composite NLL |
|---|---:|---:|---:|---:|---:|
| CE/KL proportion matching | 3 | 0.848 +/- 0.001 | 0.847 +/- 0.002 | 1.647 +/- 0.005 | 2.782 +/- 0.065 |
| Full official LLP-PVC | 3 | 0.309 +/- 0.188 | 0.256 +/- 0.178 | 6.685 +/- 0.489 | 6.579 +/- 0.425 |
| FS-Conv classwise count likelihood | 3 | 0.851 +/- 0.005 | 0.851 +/- 0.005 | 1.775 +/- 0.106 | 2.688 +/- 0.059 |
