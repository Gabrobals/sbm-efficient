# Adaptive-K SDK

> **Entropy-guided dynamic expert selection for Mixture-of-Experts models**  
> Reduce inference costs by 30-50% with proven methodology.

[![PyPI](https://img.shields.io/pypi/v/adaptive-k)](https://pypi.org/project/adaptive-k/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-k)](https://pypi.org/project/adaptive-k/)

---

## 🚀 Quick Start

```bash
pip install adaptive-k
```

```python
from adaptive_k import AdaptiveKRouter

# Load pre-calibrated router
router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")

# Route tokens
indices, weights, metrics = router.route(router_logits, return_metrics=True)

print(f"Compute savings: {metrics.compute_savings:.1%}")
# Output: Compute savings: 47.2%
```

---

## 📊 Proven Results

| Model | Savings | Quality Retained |
|-------|---------|------------------|
| Mixtral 8x7B | **52.5%** | 99.8% |
| Qwen-MoE | **32.4%** | 99.9% |
| OLMoE-1B-7B | **24.7%** | 99.7% |

---

## 💡 How It Works

Adaptive-K dynamically selects the number of experts (K) based on **routing entropy**:

```
Low entropy (confident) → K=1 → 87.5% compute saved
Medium entropy         → K=2 → 75% compute saved  
High entropy (uncertain) → K=4 → Full routing
```

The key insight: when the router is confident, fewer experts are needed.

---

## 📖 Usage

### Basic Routing

```python
from adaptive_k import AdaptiveKRouter

router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")

# Your MoE router logits (batch_size, num_experts)
router_logits = model.router(hidden_states)

# Adaptive-K routing
expert_indices, expert_weights, _ = router.route(router_logits)

# Use selected experts
output = execute_experts(hidden_states, expert_indices, expert_weights)
```

### Custom Calibration

```python
from adaptive_k import Calibrator

calibrator = Calibrator(
    target_savings=0.40,      # 40% target savings
    quality_threshold=0.99    # Max 1% quality loss
)

result = calibrator.calibrate(
    model=your_model,
    dataset=calibration_data
)

print(f"Optimal thresholds: {result.optimal_thresholds}")
print(f"Expected savings: {result.expected_savings:.1%}")
```

### Check Statistics

```python
# After processing many tokens
print(router.stats)
# {
#   'tokens_processed': 1_234_567,
#   'average_savings': 0.472,
#   'estimated_cost_reduction': '47.2%'
# }
```

---

## 🔧 Configuration

```python
from adaptive_k import AdaptiveKRouter, RoutingConfig

config = RoutingConfig(
    k_values=[1, 2, 4],           # Available K values
    entropy_thresholds=[0.6, 1.2], # H < 0.6 → K=1, H < 1.2 → K=2, else K=4
    num_experts=8
)

router = AdaptiveKRouter(config=config)
```

---

## 🔌 Integrations

### HuggingFace Transformers

```python
# Coming in v0.2.0
router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")
model = router.patch(model)  # Automatic integration
```

### vLLM

```python
# Coming in v0.3.0
from adaptive_k.integrations import vllm_patch
model = vllm_patch(model, router)
```

### TensorRT-LLM

See our [TensorRT-LLM PR #10672](https://github.com/NVIDIA/TensorRT-LLM/pull/10672) for native integration.

---

## 📈 Benchmarking

```bash
# CLI benchmark
adaptive-k benchmark --model mixtral-8x7b --dataset wikitext-2

# Output:
# Model: mixtral-8x7b
# Dataset: wikitext-2
# Baseline perplexity: 5.42
# Adaptive-K perplexity: 5.44 (+0.4%)
# Compute savings: 47.2%
```

---

## 📄 License

Apache 2.0 - Free for commercial use.

---

## 🔗 Links

- **Website**: https://adaptive-k.vertexdata.it
- **Paper**: [Entropy-Guided Dynamic Expert Selection](https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf)
- **GitHub**: https://github.com/Gabrobals/sbm-efficient

---

## 📞 Support

- **Email**: amministrazione@vertexdata.it
- **Issues**: [GitHub Issues](https://github.com/Gabrobals/sbm-efficient/issues)

---

*Made with ❤️ by [Vertex Data](https://vertexdata.it)*
