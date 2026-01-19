# Combination Experiments: Adaptive-K + Dynamic Compute Methods

## 🎯 Objective

Empirically validate **Proposition 7.1 (Multiplicative Savings)** from the whitepaper:

$$\eta_{\text{combined}} = \prod_{i=1}^{m} \eta_i$$

By combining Adaptive-K with:
1. **Early Exit** (depth dimension)
2. **Token Pruning** (sequence length dimension)  
3. **Speculative Decoding** (generation steps dimension)

## 📊 Experiment Matrix

| Combination | Expected η | Dimensions | Complexity |
|-------------|-----------|------------|------------|
| Adaptive-K alone | 0.475 | Width | ⭐ Baseline |
| + Early Exit | 0.475 × 0.70 = 0.33 | Width + Depth | ⭐⭐ |
| + Token Pruning | 0.475 × 0.65 = 0.31 | Width + Length | ⭐⭐ |
| + Speculative Decoding | 0.475 × 0.50 = 0.24 | Width + Steps | ⭐⭐⭐ |
| All combined | 0.475 × 0.70 × 0.65 = 0.22 | All | ⭐⭐⭐⭐ |

## 🔬 Experiment 1: Adaptive-K + Early Exit

### Background
Early Exit allows tokens to skip remaining transformer layers when intermediate confidence is high.

### Implementation Options
1. **CALM** (Confident Adaptive Language Modeling) - Google
2. **DeeBERT** - Early exit for BERT
3. **Custom entropy-based exit** - Use layer output entropy

### Setup
```python
# Pseudo-implementation
class AdaptiveKWithEarlyExit:
    def __init__(self, model, exit_threshold=0.9, k_thresholds=[1.275]):
        self.model = model
        self.exit_threshold = exit_threshold  # Confidence for early exit
        self.k_thresholds = k_thresholds       # Entropy for K selection
    
    def forward(self, x):
        for layer_idx, layer in enumerate(self.model.layers):
            # Check early exit condition
            if layer_idx > 0:
                exit_prob = self.compute_exit_confidence(x)
                if exit_prob > self.exit_threshold:
                    return x, layer_idx  # Early exit!
            
            # Apply MoE layer with Adaptive-K
            if hasattr(layer, 'moe'):
                x = self.adaptive_k_moe(x, layer.moe)
            else:
                x = layer(x)
        
        return x, len(self.model.layers)
```

### Metrics to Measure
- **PPL**: Perplexity (quality)
- **Avg Layers**: Mean layers executed per token
- **Avg K**: Mean experts per MoE layer
- **Combined Savings**: 1 - (Avg_Layers/Total_Layers × Avg_K/K_baseline)
- **Latency**: Wall-clock time per token

### Expected Results
| Config | Avg Layers | Avg K | Compute | PPL Δ |
|--------|-----------|-------|---------|-------|
| Baseline | 32/32 | 2.0 | 100% | — |
| Early Exit only | 22/32 | 2.0 | 68.7% | +0.5% |
| Adaptive-K only | 32/32 | 1.38 | 69.0% | +0.8% |
| **Combined** | 22/32 | 1.38 | **47.5%** | +1.2% |

---

## 🔬 Experiment 2: Adaptive-K + Token Pruning (ToMe)

### Background
Token Merging (ToMe) reduces sequence length by merging similar tokens mid-inference.

### Implementation
Use the official ToMe implementation: https://github.com/facebookresearch/ToMe

```python
# Integration approach
import tome

class AdaptiveKWithToMe:
    def __init__(self, model, merge_ratio=0.5, k_thresholds=[1.275]):
        self.model = model
        self.merge_ratio = merge_ratio  # Fraction of tokens to merge
        self.k_thresholds = k_thresholds
        
        # Apply ToMe to transformer layers
        tome.patch.timm(self.model)
    
    def forward(self, x):
        # ToMe merges tokens automatically during attention
        # Adaptive-K selects experts in MoE layers
        for layer in self.model.layers:
            if hasattr(layer, 'moe'):
                x = self.adaptive_k_moe(x, layer.moe)
            else:
                x = layer(x)  # ToMe applied here
        return x
```

### Metrics
- **Sequence Length Ratio**: Avg tokens after merging / Original tokens
- **Combined Compute**: Seq_Ratio × (Avg_K / K_baseline)

### Expected Results
| Config | Seq Ratio | Avg K | Compute | PPL Δ |
|--------|-----------|-------|---------|-------|
| Baseline | 1.0 | 2.0 | 100% | — |
| ToMe only (r=0.5) | 0.65 | 2.0 | 65% | +1.0% |
| Adaptive-K only | 1.0 | 1.38 | 69.0% | +0.8% |
| **Combined** | 0.65 | 1.38 | **44.8%** | +1.5% |

---

## 🔬 Experiment 3: Adaptive-K + Speculative Decoding

### Background
Speculative decoding uses a small "draft" model to propose multiple tokens, verified in parallel by the main model.

### Implementation
Use vLLM's built-in speculative decoding:

```python
from vllm import LLM, SamplingParams

# Speculative decoding setup
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    speculative_model="mistralai/Mistral-7B-v0.1",  # Draft model
    num_speculative_tokens=5,
    use_v2_block_manager=True
)

# Integrate Adaptive-K in the target model's MoE layers
# (Requires vLLM fork or custom integration)
```

### Key Insight
- Speculative decoding accelerates **generation** (multiple tokens per forward pass)
- Adaptive-K reduces compute **per forward pass**
- They are **truly orthogonal** - no interaction effects

### Metrics
- **Acceptance Rate**: Fraction of draft tokens accepted
- **Tokens per Step**: Avg tokens generated per target model forward
- **Combined Speedup**: Tokens_per_step × (K_baseline / Avg_K)

### Expected Results
| Config | Tokens/Step | MoE Compute | Effective Speedup |
|--------|-------------|-------------|-------------------|
| Baseline | 1.0 | 100% | 1.0× |
| Speculative only | 3.2 | 100% | 3.2× |
| Adaptive-K only | 1.0 | 47.5% | 2.1× |
| **Combined** | 3.2 | 47.5% | **6.7×** |

---

## 🔬 Experiment 4: Triple Combination

### The Ultimate Stack
```
┌─────────────────────────────────────────┐
│  Speculative Decoding (3.2× speedup)    │  ← Generation efficiency
├─────────────────────────────────────────┤
│  Early Exit (70% layers)                │  ← Depth efficiency
├─────────────────────────────────────────┤
│  Adaptive-K (47.5% experts)             │  ← Width efficiency
└─────────────────────────────────────────┘
```

### Expected Combined Savings
```
Total Compute = 0.70 (early exit) × 0.475 (adaptive-k) = 0.33
Effective Speedup = 3.2 (speculative) × (1/0.33) = 9.7×
```

**Theoretical: 9.7× speedup with <3% quality loss**

---

## 📋 Implementation Roadmap

### Phase 1: Baseline Measurements (Week 1)
- [ ] Measure Adaptive-K alone on Mixtral 8×7B
- [ ] Measure Early Exit alone (implement CALM-style)
- [ ] Measure ToMe alone
- [ ] Measure Speculative Decoding alone (vLLM)

### Phase 2: Pairwise Combinations (Week 2)
- [ ] Adaptive-K + Early Exit
- [ ] Adaptive-K + ToMe
- [ ] Adaptive-K + Speculative Decoding

### Phase 3: Triple Combination (Week 3)
- [ ] Adaptive-K + Early Exit + Speculative
- [ ] Full validation on multiple benchmarks

### Phase 4: Paper Update (Week 4)
- [ ] Add results to whitepaper Section 7.2
- [ ] Update Proposition 7.1 with empirical validation
- [ ] Create comparison visualizations

---

## 🛠️ Required Resources

### Hardware
- **GPU**: A100 80GB (for Mixtral 8×7B full inference)
- **Alternative**: 2× RTX 4090 with model sharding

### Software Dependencies
```bash
pip install adaptive-k-routing  # Our package
pip install vllm                # For speculative decoding
pip install timm tome           # For token merging
pip install transformers accelerate
```

### Models
- **Target**: Mixtral 8×7B-v0.1 (HuggingFace)
- **Draft** (for speculative): Mistral-7B-v0.1

---

## 📊 Success Criteria

| Metric | Target |
|--------|--------|
| Combined compute savings | >60% vs baseline |
| Quality degradation | <3% PPL increase |
| Speedup validation | Within 20% of theoretical |
| Statistical significance | p < 0.05 for all comparisons |

---

## 🎯 Potential Impact

If successful, this demonstrates:
1. **Adaptive-K is composable** with other efficiency methods
2. **Multiplicative savings are real**, not theoretical
3. **Production viability** of combined optimization stack
4. **New SOTA** for MoE inference efficiency

This could be a **standalone follow-up paper** or major addition to current whitepaper.

---

## 📝 Notes

- Start with **Experiment 2 (ToMe)** - easiest integration
- **Experiment 3 (Speculative)** requires vLLM fork - higher effort
- **Experiment 1 (Early Exit)** needs custom implementation
- Consider publishing intermediate results on HuggingFace

---

*Created: January 17, 2026*
*Status: Planning*
