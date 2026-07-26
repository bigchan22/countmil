# Final Git and Artifact Provenance

- Branch: `neurips26-rebuttal`.
- Current HEAD at generation time: `f34a6737ddc7f41bb20d4e0131065dd674c2b78e`.
- Remote 4090 branch tip: `c9928f33bb42b168800c2e7488b4cfdd51ca516b`.
- Remote A5000 branch tip: `796d0b468b88a0a92deb912abe35bb63074080c3`.
- Merge strategy: explicit `--no-ff` merge commits; no force-push or history rewrite.
- SHA marker verification failed because both expected final marker files are missing on the corresponding feature branch tips.
- Git excludes were checked for datasets, feature tensors, checkpoints, pretrained weights, `.part` files, and caches before final staging.
- Tracked fixed-bag `.pt` files are small manifest files, not feature tensors or checkpoints.

Non-Git/local artifacts:

- CIFAR data archives, frozen feature tensors, and model checkpoints remain local/ignored.
- Official LLP-PVC dev checkpoints under `results/rebuttal/4090_official_llppvc/dev_runs/*/checkpoint_*.pt` are intentionally not staged.
