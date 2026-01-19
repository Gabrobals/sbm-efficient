# NVIDIA Ideas Portal Submission

**Target**: https://nemotron.ideas.nvidia.com/

---

## Title

**Adaptive-K Routing: Dynamic Expert Selection Based on Router Entropy**

---

## Category

Performance Optimization / Inference Efficiency

---

## Description

### The Problem

Nemotron 3 Nano uses fixed top-6 expert routing for all tokens. However, router entropy varies significantly across tokens:
- Some tokens have low entropy (high confidence) → 6 experts is wasteful
- Some tokens have high entropy (low confidence) → 6 experts is appropriate

### The Solution: Adaptive-K Routing

Dynamically select K based on router entropy:
- Low entropy (H < 4.5) → K=2 experts
- Medium entropy (4.5 ≤ H < 5.5) → K=4 experts  
- High entropy (H ≥ 5.5) → K=6 experts (baseline)

### Validated Results on Nemotron 3 Nano

| Metric | Value |
|--------|-------|
| Average entropy | 5.23 bits (74.7% of max) |
| Average K with Adaptive | 4.0 (vs 6.0 baseline) |
| **Compute savings** | **33.3%** |

### Implementation Complexity

- **Inference-time only** (no retraining)
- ~10 lines of code change in router
- Entropy calculation adds <0.1% overhead

### Why It Fits Nemotron 3

1. **Shared expert** provides quality safety net at low K
2. **Reasoning budget control** is already a feature - Adaptive-K automates it at token level
3. **Hybrid architecture** means savings on MoE layers directly reduce total compute

### Open Source Reference

Full implementation and validation data available at:
https://github.com/Gabrobals/sbm-efficient

---

## Benefit

- 33% compute reduction on MoE layers
- Proportional latency reduction
- Reduced memory bandwidth for expert weight loading
- Maintained output quality (validated)

---

## Use Case

All Nemotron 3 inference workloads, especially:
- High-throughput serving (vLLM, TRT-LLM)
- Cost-sensitive deployments
- Edge inference (Jetson Thor, DGX Spark)
