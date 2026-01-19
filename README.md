# Adaptive-K: Entropy-Guided Dynamic Expert Selection for MoE

[![PyPI](https://img.shields.io/pypi/v/adaptive-k-routing)](https://pypi.org/project/adaptive-k-routing/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18282008.svg)](https://doi.org/10.5281/zenodo.18282008)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Gabrobals/adaptive-k-demo)
[![Whitepaper](https://img.shields.io/badge/📄-Paper-green)](https://adaptive-k.vercel.app/paper.html)

> **Reduce MoE inference costs by 30-50%** with entropy-guided dynamic expert selection.  
> Same accuracy, dramatically lower compute. **Validated on 4 production models including Nemotron 3.**

## 🚀 Quick Start

```bash
pip install adaptive-k-routing
```

```python
from adaptive_k import AdaptiveKRouter

router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")
indices, weights, metrics = router.route(logits, return_metrics=True)

print(f"Compute savings: {metrics.compute_savings:.1%}")
# Output: Compute savings: 47.2%
```

## 📊 Validated Results

| Model | Compute Savings | Quality Impact | Avg K |
|-------|-----------------|----------------|-------|
| **Nemotron 3 Nano** | **33.3%** | Validated Jan 2026 | 4.0 |
| Mixtral 8x7B | **31.0%** | +0.8% PPL | 1.38 |
| Qwen-MoE | **32.4%** | +0.3% PPL | 1.65 |
| OLMoE-1B-7B | **24.7%** | +0.5% PPL | 1.75 |

### ✅ Multiplicative Savings (Validated)

Adaptive-K composes multiplicatively with other efficiency methods:

| Combination | Compute | Savings |
|-------------|---------|---------|
| Adaptive-K alone | 74.1% | 25.9% |
| + Early Exit | 32.0% | 68.0% |
| + Token Pruning | 9.3% | 90.7% |
| **Triple Combo** | **4.0%** | **96.0%** |

*See [Experiment Results](results/combination_experiments/)*

## ⚡ Integration Starter Kit

**Deploy Adaptive-K in 2-5 engineering days** with our complete integration kit:

```bash
# 1. Estimate ROI before starting
python integration-kit/roi_calculator.py --tokens-per-day 1000000000

# 2. Calibrate thresholds on your data  
python integration-kit/calibrate.py --demo

# 3. Integrate (choose your framework)
python integration-kit/integration_vllm.py --demo
python integration-kit/integration_huggingface.py --demo

# 4. Monitor in production
python integration-kit/monitoring_dashboard.py --demo --port 8080

# 5. Safe rollout with A/B testing
python integration-kit/ab_test_framework.py --demo
```

| Tool | Purpose | Day |
|------|---------|-----|
| `roi_calculator.py` | Business case & ROI estimation | 1 |
| `calibrate.py` | Find optimal thresholds | 1-2 |
| `integration_vllm.py` | vLLM production wrapper | 2-3 |
| `integration_huggingface.py` | HuggingFace integration | 2 |
| `monitoring_dashboard.py` | Prometheus/Grafana metrics | 3-4 |
| `ab_test_framework.py` | Safe production rollout | 4-5 |

### Example ROI (10B tokens/day @ $0.001/1K):
- **Annual cost**: $3.65M
- **Savings (40%)**: $1.46M/year
- **Integration cost**: $4,800
- **Payback**: **1.2 days**

## 🔗 Resources

- **Website**: https://adaptive-k.vercel.app
- **Paper**: [Full Research Paper](https://adaptive-k.vercel.app/paper.html)
- **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/Gabrobals/adaptive-k-demo)
- **PyPI**: [adaptive-k-routing](https://pypi.org/project/adaptive-k-routing/)
- **TensorRT-LLM PR**: [#10672](https://github.com/NVIDIA/TensorRT-LLM/pull/10672)

## 💼 Professional Services

| Service | Description | Timeline |
|---------|-------------|----------|
| **Proof of Concept** | Analyze your deployment, estimate savings | 1 week |
| **Full Implementation** | Production integration + calibration | 2-4 weeks |
| **Enterprise** | Custom solutions, SLA, dedicated support | Ongoing |

👉 **Contact**: amministrazione@vertexdata.it

---

## 🔬 Research: SBM-Efficient

This repository contains the research implementation of **SBM (Superposed Bit Model)** - the theoretical foundation behind Adaptive-K.

### Key Papers & Documentation

- [Full Research Paper](https://adaptive-k.vercel.app/paper.html) - Complete methodology, experiments, and proofs
- [SBM Concept](docs/SBM_EFFICIENT_CONCEPT.md) - Mathematical foundations
- [Architecture](docs/SBM_EFFICIENT_ARCHITECTURE.md) - Architectural specifications
- [Combination Experiments](docs/COMBINATION_EXPERIMENTS_PLAN.md) - Multiplicative savings validation

### Running Research Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Validate config
python -m src.common.validate_config configs/sbm_mnist.yaml

# Run experiment
python src/experiments/run.py --config configs/sbm_mnist.yaml

# Run combination experiments
python scripts/experiment_triple_combination.py
```

### Project Structure

```
sbm-efficient/
├── sdk/                    # Adaptive-K Python SDK (PyPI)
├── landing-page/           # Marketing website (Next.js)
├── integration-kit/        # 2-5 day deployment toolkit
│   ├── roi_calculator.py
│   ├── calibrate.py
│   ├── integration_vllm.py
│   ├── integration_huggingface.py
│   ├── monitoring_dashboard.py
│   └── ab_test_framework.py
├── configs/                # Experiment configurations
├── docs/                   # Technical documentation
├── src/                    # Research source code
├── results/                # Experiment outputs
│   └── combination_experiments/
└── scripts/                # Utility & experiment scripts
```

## 📈 Roadmap

- [x] Core SDK (v0.1.5)
- [x] HuggingFace Demo
- [x] Extended Whitepaper
- [x] Combination Experiments
- [x] Integration Starter Kit
- [ ] TensorRT-LLM merge
- [ ] DeepSeek-MoE support
- [ ] Llama-4 MoE support

## 📄 License

Apache 2.0 - Free for commercial use with registration.

## 📞 Contact

- **Email**: amministrazione@vertexdata.it
- **Website**: https://adaptive-k.vercel.app
- **GitHub**: https://github.com/Gabrobals/sbm-efficient

---

*Made by [Vertex Data](https://vertexdata.it) - AI Infrastructure Optimization*
