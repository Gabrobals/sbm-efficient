# SBM Adaptive-K: Executive Summary

## TL;DR

We built and validated an **adaptive compute classifier** that automatically adjusts computational resources based on input difficulty:
- **Easy inputs** → uses fewer experts → **saves 17% compute**
- **Hard inputs** → uses more experts → **gains +0.6% accuracy**

This is validated across 2 datasets with 5-seed statistical rigor.

---

## The Problem

Traditional neural networks spend the **same compute on every input**, whether it's trivially easy or ambiguously hard. This wastes resources on easy cases and may underperform on hard ones.

## Our Solution: SBM Adaptive-K

A Mixture-of-Experts architecture where:
- A **router** measures uncertainty (entropy) for each input
- Based on uncertainty, it selects **K experts** dynamically:
  - Low entropy (confident) → fewer experts (K=1-2)
  - High entropy (uncertain) → more experts (K=3-4)
- Only selected experts execute → **sparse compute**

---

## Key Results

| Dataset | Difficulty | K_mean | FLOPs vs Baseline | Accuracy vs Baseline |
|---------|------------|--------|-------------------|----------------------|
| MNIST | Easy | 1.66 | **-17%** | +0.03% (same) |
| Fashion-MNIST | Hard | 2.12 | +6% | **+0.59%** (better) |

### Interpretation

The system **correctly identifies task difficulty**:
- On MNIST (98% baseline accuracy), most inputs are easy → saves compute
- On Fashion-MNIST (86% baseline accuracy), inputs are harder → invests compute for better results

---

## Robustness Testing

We tested both models under input perturbations:

| Perturbation | MNIST Winner | Fashion-MNIST Winner |
|--------------|--------------|----------------------|
| Clean (no noise) | Adaptive-K | Adaptive-K |
| Salt & Pepper | Adaptive-K | Adaptive-K |
| Gaussian Noise | Tie | Static (both fail) |
| Occlusion | Mixed | Static |

**MNIST**: Adaptive-K wins 5/11 tests, loses 1, ties 5
**Fashion-MNIST**: Adaptive-K wins 5/11 tests, loses 6

---

## Methodology

- **Multi-seed validation**: 5 seeds (42-46) per configuration
- **Metrics**: Accuracy, Precision, Recall, F1, FLOPs, K_mean, Latency
- **Automated postflight**: Schema validation for every run
- **Reproducible**: All configs, scripts, and results in version control

---

## Business Value

For inference-heavy workloads:

| Scenario | Benefit |
|----------|---------|
| Easy workload (like MNIST) | **17% compute cost reduction** at same quality |
| Hard workload (like Fashion-MNIST) | **Better accuracy** with minimal cost increase |
| Mixed workload | Adaptive allocation optimizes both |

### Example ROI

If you spend **$100K/month** on inference for an "easy" workload:
- Adaptive-K could save **$17K/month** ($204K/year)
- No accuracy loss

---

## Current Status

| Milestone | Status |
|-----------|--------|
| Core algorithm | ✅ Implemented |
| MNIST validation | ✅ Complete (17% FLOPs reduction) |
| Fashion-MNIST validation | ✅ Complete (adaptive behavior confirmed) |
| Multi-seed harness | ✅ 5 seeds per config |
| Robustness testing | ✅ 11 perturbation types |
| Automated tooling | ✅ Scoreboard generation |

**TRL: 4** (validated in lab environment)

---

## Next Steps

1. **CNN architecture** for CIFAR-10 / ImageNet subsets
2. **GPU profiling** for production latency numbers
3. **SDK packaging** for easy integration
4. **Scaling experiments** on larger models

---

## Files & Reproducibility

All code, configs, and results are in the repository:

- `configs/` - YAML configurations for all experiments
- `scripts/` - Automation (multi-seed runner, scoreboard generator)
- `results/summaries/` - JSON results for each task
- `PHASE_C_RESULTS.md` - Detailed technical results

To reproduce:
```bash
python -m scripts.run_multiseed --config configs/sbm_adaptive_k_mnist.yaml --seeds 42,43,44,45,46
python -m scripts.generate_scoreboard --task mnist
```

---

## Contact

[Your name/email here]

---

*Generated: January 2026*
