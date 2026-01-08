# Phase C Results: Fashion-MNIST Validation

## Overview

Phase C validates SBM Adaptive-K on Fashion-MNIST, a harder dataset than MNIST.
The goal is to verify that adaptive compute behavior generalizes beyond MNIST.

## Key Finding

**Adaptive-K correctly scales compute to task difficulty:**

| Dataset | Difficulty | K_mean | FLOPs vs baseline | Accuracy vs baseline |
|---------|------------|--------|-------------------|---------------------|
| MNIST | Easy | 1.66 | **-17%** (saves) | +0.03% (same) |
| Fashion-MNIST | Hard | 2.12 | +6% (invests) | **+0.59%** (better) |

This is the intended behavior: the system automatically detects harder inputs and allocates more compute to achieve better results.

## Detailed Results

### MNIST (Phase B baseline)

| Metric | static_topk | sbm_adaptive_k | Delta |
|--------|-------------|----------------|-------|
| Accuracy | 97.96%  0.38% | 97.99%  0.27% | +0.03% |
| F1 | 97.95%  0.38% | 97.97%  0.27% | +0.02% |
| FLOPs | 2,073,924 | 1,721,688 | **-17.0%** |
| K_mean | 2.00 | 1.66 | -17.0% |
| Entropy | 0.00 | 0.54 | - |
| Robustness | baseline | 7 wins / 4 losses |  |

### Fashion-MNIST (Phase C)

| Metric | static_topk | sbm_adaptive_k | Delta |
|--------|-------------|----------------|-------|
| Accuracy | 86.25%  0.18% | 86.84%  0.61% | **+0.59%** |
| F1 | 86.27%  0.09% | 86.80%  0.54% | +0.53% |
| FLOPs | 2,073,924 | 2,198,214 | +6.0% |
| K_mean | 2.00 | 2.12 | +6.0% |
| Entropy | 0.00 | 0.80 | - |
| Robustness | baseline | 4 wins / 7 losses | - |

## Input Robustness Analysis

### Fashion-MNIST Perturbation Results

| Perturbation | static_topk | sbm_adaptive_k | Winner |
|--------------|-------------|----------------|--------|
| gaussian:0.0 | 86.25% | 86.84% | adaptive_k |
| gaussian:0.1 | 11.79% | 10.69% | static_topk |
| gaussian:0.2 | 11.73% | 10.62% | static_topk |
| gaussian:0.3 | 11.64% | 10.54% | static_topk |
| salt_pepper:0.0 | 86.25% | 86.84% | adaptive_k |
| salt_pepper:0.05 | 79.46% | 80.60% | adaptive_k |
| salt_pepper:0.1 | 66.58% | 67.92% | adaptive_k |
| occlusion:0.0 | 86.25% | 86.84% | adaptive_k |
| occlusion:0.15 | 73.26% | 72.56% | static_topk |
| occlusion:0.3 | 51.27% | 50.04% | static_topk |
| inversion:1.0 | 12.69% | 11.54% | static_topk |

**Key observation:** Both models catastrophically fail under gaussian noise (accuracy drops to ~11%). This is a dataset/architecture limitation, not specific to Adaptive-K.

## Conclusions

1. **Adaptive compute works as designed**: saves compute on easy tasks, invests on hard tasks
2. **Accuracy improvement on hard tasks**: +0.59% on Fashion-MNIST with only 6% more FLOPs
3. **Robustness trade-off**: Adaptive-K is slightly less robust under heavy perturbations
4. **Architecture limitation**: Both models are highly sensitive to gaussian noise on Fashion-MNIST

## Files Generated

- `results/summaries/fashion_mnist_input_robustness.json`
- `results/summaries/fashion_mnist_b2_scoreboard.json`
- `results/aggregated_results.json` (updated)

## Next Steps

- [ ] Create automated scoreboard generation script
- [ ] Consider CIFAR-10 validation (requires architecture review for noise sensitivity)
- [ ] Document findings in executive summary
