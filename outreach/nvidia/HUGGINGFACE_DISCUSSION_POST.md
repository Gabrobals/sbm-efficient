# HuggingFace Discussion Post for Nemotron 3 Nano

**Target**: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions

---

## Title

**[Research] Adaptive-K Routing Validation: 33% Compute Savings on Nemotron 3 Nano**

---

## Post Content

Hi NVIDIA team! 👋

I've been working on **Adaptive-K routing** - an entropy-guided method for dynamic expert selection in MoE models. Today I validated it on Nemotron 3 Nano and wanted to share the results.

### TL;DR
- **33.3% compute savings** by dynamically selecting K∈{2,4,6} based on router entropy
- **Zero retraining required** - inference-time only
- Entropy-based selection: confident tokens use fewer experts

### Results

| Test Case | Router Entropy | Effective K | Savings |
|-----------|---------------|-------------|---------|
| Easy tokens | 5.26 bits | 4.1 | 32.4% |
| Code tokens | 5.28 bits | 4.0 | 33.3% |
| Hard tokens | 5.16 bits | 3.9 | 34.4% |
| **Average** | **5.23 bits** | **4.0** | **33.3%** |

### The Insight

Nemotron 3's router entropy (measured via pre-top-k logits) averages 5.23 bits out of 7.0 max (log₂(128)). This means:
- ~75% of max entropy → router is moderately confident
- Many tokens don't need all 6 experts
- The shared expert provides a quality safety net for aggressive K reduction

### Methodology

Since `output_router_logits` isn't supported, I used forward hooks on `backbone.layers.X.mixer.gate` to compute full 128-expert logits:

```python
router_logits = hidden_states @ module.weight.T  # [batch, seq, 128]
probs = softmax(router_logits)
entropy = -sum(probs * log(probs))  # Per-token entropy
```

### Why This Matters for Nemotron 3

1. **Amplifies reasoning budget control**: Users already control reasoning tokens - Adaptive-K automates compute optimization per-token
2. **Shared expert synergy**: The always-active shared expert means quality is maintained even at K=2
3. **No retraining**: Drop-in replacement for the router

### Open Source

- **Full validation**: https://github.com/Gabrobals/sbm-efficient
- **Results JSON**: `results/nemotron3_nano_validation.json`
- **Whitepaper**: `outreach/cerebras/ADAPTIVE_K_CEREBRAS_WHITEPAPER.md`

### Questions for NVIDIA

1. Would you be interested in integrating Adaptive-K as an optional routing mode?
2. Is there a preferred way to contribute to the Nemotron cookbooks?
3. Any plans to expose `output_router_logits` in future versions?

Happy to collaborate on benchmarks or provide a PR to the NeMo repository!

---

**Gabriele Balsamo**  
GitHub: [@Gabrobals](https://github.com/Gabrobals)
