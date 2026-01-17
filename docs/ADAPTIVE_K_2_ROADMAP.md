# Adaptive-K 2.0 Roadmap

> **Kaizen Philosophy**: Small continuous improvements + One moonshot

**Start Date**: January 2026
**Author**: Gabriele Ballarani

---

## Executive Summary

Two parallel tracks:
- **Track A (Revenue)**: Adaptive Speculative Decoding - 4 weeks to MVP
- **Track B (Innovation)**: Information Flow Monitor - 12 weeks to paper

Both use the same core principle: **entropy/information → dynamic compute allocation**

---

## Track A: Adaptive Speculative Decoding

### The Problem

Current speculative decoding uses **fixed draft length** (K=4 or K=8):

```
Token "the" (easy)     → K=4 draft → 3 wasted speculations
Token "quantum" (hard) → K=4 draft → not enough, rejection
```

### The Solution

**Entropy-Adaptive Draft Length**:

| Model Confidence | Draft Length | Draft Model |
|------------------|--------------|-------------|
| Very High (H < 0.5) | K=16 | Tiny (68M) |
| High (H < 1.0) | K=8 | Small (160M) |
| Medium (H < 2.0) | K=4 | Medium (1B) |
| Low (H > 2.0) | K=1 or skip | No speculation |

### Expected Impact

- **20-40% speedup** over fixed speculative decoding
- **Stacks with Adaptive-K MoE**: 0.7 × 0.7 = 0.49 (51% savings)

### Implementation Plan

#### Week 1: Research & Setup
- [ ] Fork vLLM repository
- [ ] Study speculative decoding internals (`vllm/spec_decode/`)
- [ ] Profile entropy distribution on Llama-3-8B
- [ ] Document baseline metrics

#### Week 2: Entropy Extraction
- [ ] Add entropy computation hook in draft model
- [ ] Create entropy histogram visualization
- [ ] Identify threshold candidates from data
- [ ] Write entropy calibration script

#### Week 3: Adaptive Logic
- [ ] **⚠️ MOVE TO PRIVATE REPO before vLLM integration**
- [ ] Implement adaptive draft length selector
- [ ] Add threshold configuration
- [ ] Unit tests for edge cases
- [ ] Integration with vLLM pipeline

#### Week 4: Benchmarks & Release
- [ ] Benchmark on HumanEval, MT-Bench
- [ ] Measure tok/s, P50/P99 latency
- [ ] Write blog post
- [ ] Release as vLLM plugin
- [ ] Submit PR to vLLM

### Deliverables

1. `adaptive-speculative` Python package
2. Blog post: "Entropy-Guided Speculative Decoding: 30% Faster LLM Inference"
3. vLLM PR
4. Benchmark dashboard

### Success Metrics

| Metric | Target |
|--------|--------|
| Speedup over fixed spec | >20% |
| Acceptance rate | >85% |
| Memory overhead | <5% |
| Integration complexity | <100 LOC |

---

## Track B: Information Flow Monitor (IFM)

### The Vision

> "What if we could SEE how information flows through a neural network in real-time, and use that to skip unnecessary computation?"

This is **NOT** incremental optimization. This is a **new paradigm**.

### Theoretical Foundation

#### Information Bottleneck Principle (Tishby 2000)

The optimal representation Z of input X for predicting Y minimizes:
```
L = I(X; Z) - β·I(Z; Y)
```

**Key insight**: Most layers are COMPRESSING information, not adding it.

#### Mutual Information for Layer Importance

For each layer L:
```
Importance(L) = I(input_L; final_output) - I(output_L; final_output)
```

If Importance(L) ≈ 0, the layer can be skipped.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LLM Backbone                          │
├─────────┬─────────┬─────────┬─────────┬─────────────────┤
│ Layer 1 │ Layer 2 │ Layer 3 │   ...   │ Layer N         │
│    ↓    │    ↓    │    ↓    │    ↓    │    ↓            │
│ Probe 1 │ Probe 2 │ Probe 3 │   ...   │ Probe N         │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────────────┘
     │         │         │         │         │
     └─────────┴─────────┴─────────┴─────────┘
                         │
              ┌──────────▼──────────┐
              │   Flow Aggregator   │
              │   (MI Estimation)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Skip Controller   │
              │   (Real-time)       │
              └─────────────────────┘
```

### Phase 1: Information Probing (Weeks 1-3)

#### Week 1: Probe Architecture
- [ ] Design lightweight probe network (linear → softmax)
- [ ] Implement probe insertion for Llama architecture
- [ ] Create probe training pipeline
- [ ] Test on Llama-3-8B

#### Week 2: MI Estimation
- [ ] Implement MINE estimator (Mutual Information Neural Estimation)
- [ ] Alternative: InfoNCE bound
- [ ] Calibrate on WikiText-2
- [ ] Validate MI estimates

#### Week 3: Flow Mapping
- [ ] Compute MI(layer_i, final_logits) for all layers
- [ ] Visualize "information flow maps"
- [ ] Identify patterns:
  - Which layers have low MI?
  - Correlation with token position?
  - Correlation with token type?
- [ ] Document findings

### Phase 2: Pattern Discovery (Weeks 4-6)

#### Week 4: Statistical Analysis
- [ ] Cluster layers by information profile
- [ ] Identify "compressor" vs "transformer" layers
- [ ] Analyze per-token variation
- [ ] Create layer importance heatmaps

#### Week 5: Predictor Design
- [ ] Design lightweight predictor: "Will layer L add information?"
- [ ] Features: previous activations, token embeddings, position
- [ ] Train predictor on MI labels
- [ ] Evaluate predictor accuracy

#### Week 6: Predictor Optimization
- [ ] Distill predictor to <1M parameters
- [ ] Optimize for inference speed
- [ ] Quantize predictor to INT8
- [ ] Measure overhead

### Phase 3: Adaptive Execution (Weeks 7-10)

#### Week 7: Skip Mechanism
- [ ] Implement layer skip in forward pass
- [ ] Residual connection preservation
- [ ] Handle attention mask propagation
- [ ] Test numerical stability

#### Week 8: Dynamic Routing
- [ ] Real-time skip decisions
- [ ] Threshold tuning
- [ ] Quality vs speed tradeoff curves
- [ ] A/B comparison with baseline

#### Week 9: Integration
- [ ] Package as standalone library
- [ ] Integration with Transformers library
- [ ] Integration with vLLM
- [ ] Documentation

#### Week 10: Optimization
- [ ] Profile memory usage
- [ ] CUDA kernel optimization (if needed)
- [ ] Batch efficiency
- [ ] Edge cases handling

### Phase 4: Publication & Release (Weeks 11-12)

#### Week 11: Paper Writing
- [ ] Introduction: The information flow perspective
- [ ] Method: IFM architecture and training
- [ ] Experiments: Llama, Mistral, Mixtral
- [ ] Results: Speed vs quality curves
- [ ] Analysis: What did we learn about LLMs?

#### Week 12: Release
- [ ] ArXiv submission
- [ ] GitHub release
- [ ] Blog post
- [ ] Twitter/LinkedIn announcement
- [ ] Submit to ICML/NeurIPS

### Deliverables

1. **Paper**: "Information Flow Monitor: Seeing and Skipping in Large Language Models"
2. **Library**: `information-flow-monitor` package
3. **Visualizations**: Interactive information flow maps
4. **Benchmarks**: Comprehensive speed/quality results
5. **Blog series**: 3-part deep dive

### Success Metrics

| Metric | Target |
|--------|--------|
| Layer skip rate | 20-40% |
| Quality retention | >98% |
| Speedup | 25-50% |
| Paper acceptance | Top venue |

### Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| MI estimation too slow | Medium | Pre-compute, use bounds |
| Predictor not accurate | Medium | Ensemble, confidence threshold |
| Quality degradation | Low | Conservative thresholds |
| Scooped by big lab | Low | Move fast, publish early |

---

## Combined Timeline

```
Jan 2026
├── Week 1-2: Track A (Speculative) + Track B Phase 1 start
├── Week 3-4: Track A MVP release
│
Feb 2026
├── Week 5-6: Track B Pattern Discovery
├── Week 7-8: Track B Adaptive Execution
│
Mar 2026
├── Week 9-10: Track B Integration
├── Week 11-12: Track B Paper + Release
```

### Resource Allocation

| Week | Track A | Track B | Total Hours |
|------|---------|---------|-------------|
| 1-2 | 70% | 30% | 40h |
| 3-4 | 80% | 20% | 40h |
| 5-8 | 20% | 80% | 40h |
| 9-12 | 10% | 90% | 40h |

---

## Revenue Integration

### Track A (Speculative) Monetization

1. **Open source plugin** → Credibility
2. **Consulting**: "We'll implement this for your infra"
3. **Enterprise support**: SLA for production deployments

### Track B (IFM) Monetization

1. **Paper** → Academic credibility
2. **Proprietary optimizations** → Licensing to cloud providers
3. **Consulting**: "We understand LLM information flow"
4. **Training data**: Sell information flow profiles

---

## Synergies with Adaptive-K 1.0

| Component | Adaptive-K MoE | Speculative | IFM |
|-----------|----------------|-------------|-----|
| Core principle | Entropy → K | Entropy → draft len | MI → skip |
| Stackable? | - | ✅ Yes | ✅ Yes |
| Combined savings | 30-50% | +20-30% | +25-40% |

**Total potential**: 1 - (0.6 × 0.75 × 0.65) = **70%+ compute reduction**

---

## Next Actions

### This Week
1. [ ] Fork vLLM
2. [ ] Set up profiling environment
3. [ ] Read speculative decoding code
4. [ ] Read MINE paper

### This Month
1. [ ] Track A MVP
2. [ ] Track B probes working
3. [ ] First information flow visualizations
4. [ ] Blog post on speculative decoding

---

## References

### Speculative Decoding
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)

### Information Theory in Deep Learning
- Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle" (2015)
- Shwartz-Ziv & Tishby, "Opening the Black Box of Deep Neural Networks" (2017)
- Belghazi et al., "MINE: Mutual Information Neural Estimation" (2018)

### Dynamic Computation
- Graves, "Adaptive Computation Time for Recurrent Neural Networks" (2016)
- Schuster et al., "Confident Adaptive Language Modeling" (CALM, 2022)

---

*"The best way to predict the future is to invent it." - Alan Kay*

*"Information is the resolution of uncertainty." - Claude Shannon*
