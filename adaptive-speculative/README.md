# Adaptive Speculative Decoding

> Entropy-guided dynamic draft length for speculative decoding

## The Problem

Current speculative decoding uses **fixed draft length** (e.g., K=5):

```
Token "the" (easy, H=0.2)     → K=5 draft → 4 wasted speculations
Token "quantum" (hard, H=3.1) → K=5 draft → high rejection rate
```

## Our Solution

**Entropy-Adaptive Draft Length**:

```python
def get_adaptive_k(logits, thresholds=[0.5, 1.0, 2.0], k_values=[16, 8, 4, 1]):
    """Select draft length based on model confidence."""
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    for threshold, k in zip(thresholds, k_values[:-1]):
        if entropy < threshold:
            return k
    return k_values[-1]
```

| Model Confidence | Entropy H | Draft Length K |
|------------------|-----------|----------------|
| Very High | H < 0.5 | K=16 |
| High | H < 1.0 | K=8 |
| Medium | H < 2.0 | K=4 |
| Low | H > 2.0 | K=1 (skip) |

## Expected Impact

- **20-40% speedup** over fixed speculative decoding
- **Stacks with Adaptive-K MoE**: multiplicative savings
- **Higher acceptance rate**: right-sized speculation

## Project Structure

```
adaptive-speculative/
├── README.md
├── pyproject.toml
├── src/
│   └── adaptive_speculative/
│       ├── __init__.py
│       ├── entropy.py          # Entropy computation
│       ├── adaptive_proposer.py # Modified proposer
│       └── config.py           # Threshold configuration
├── experiments/
│   ├── profile_entropy.py      # Profile entropy distribution
│   ├── benchmark.py            # Speed benchmarks
│   └── calibrate_thresholds.py # Find optimal thresholds
└── tests/
    └── test_entropy.py
```

## Roadmap

### Week 1: Research & Profile
- [ ] Profile entropy distribution on Llama-3-8B
- [ ] Visualize entropy vs token type
- [ ] Document baseline metrics

### Week 2: Implementation
- [ ] Implement entropy computation hook
- [ ] Create adaptive K selector
- [ ] Integration with vLLM

### Week 3: Benchmarks
- [ ] HumanEval, MT-Bench
- [ ] tok/s, P50/P99 latency
- [ ] Acceptance rate analysis

### Week 4: Release
- [ ] Blog post
- [ ] vLLM PR
- [ ] PyPI package

## Quick Start (coming soon)

```python
from adaptive_speculative import AdaptiveSpeculativeConfig

config = AdaptiveSpeculativeConfig(
    draft_model="facebook/opt-125m",
    thresholds=[0.5, 1.0, 2.0],
    k_values=[16, 8, 4, 1],
)

# Use with vLLM
llm = LLM(
    model="meta-llama/Llama-3-8B",
    speculative_config=config.to_vllm_config(),
)
```

## License

Apache 2.0

## Citation

```bibtex
@software{adaptive_speculative_2026,
  author = {Ballarani, Gabriele},
  title = {Adaptive Speculative Decoding: Entropy-Guided Dynamic Draft Length},
  year = {2026},
  url = {https://github.com/Gabrobals/adaptive-speculative}
}
```
