# Project brief: CountMIL experiments

## Paper framing
Working title:
Finite-Support Convolutional Likelihoods for Aggregate Supervision

Main empirical questions:
1. Does convolution compute the same exact count likelihood as Shukla-style DP?
2. Is convolutional Count Loss GPU-practical in batched settings?
3. Can finite-support convolution handle tasks beyond Bernoulli counts?
4. Can signed Count-MIL handle cancellation?
5. Can count-conditioned posterior EM improve instance-level recovery?

## Core methods
1. Binary convolutional count likelihood
2. Signed convolutional count likelihood
3. Categorical finite-support convolution
4. Count-conditioned posterior marginals
5. EM / posterior matching variants

## Required metrics
Bag-level:
- NLL
- exact aggregate accuracy
- MAE
- calibration if applicable

Instance-level:
- AUROC
- AUPRC
- F1
- accuracy
- entropy of p_i
- fraction of near-binary predictions

Runtime:
- forward time
- forward+backward time
- peak GPU memory if easy
- batch size / bag size / support size scaling

## Baselines
- Sum/MSE regression
- Shukla-style DP Count Loss
- brute-force enumeration for small bags
- Attention MIL / Gated Attention MIL for MNIST-Bags
- supervised oracle using hidden instance labels
- optional PROPOR / LLP-PVC if we run LLP benchmarks

## First implementation target
Build the aggregate likelihood layer and tests before any full training.
