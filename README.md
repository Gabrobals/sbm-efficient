# Adaptive-K: Entropy-Guided Dynamic Expert Selection for MoE

[![PyPI](https://img.shields.io/pypi/v/adaptive-k)](https://pypi.org/project/adaptive-k/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Reduce MoE inference costs by 30-50%** with entropy-guided dynamic expert selection.  
> Same accuracy, dramatically lower compute.

## 🚀 Quick Install

```bash
pip install adaptive-k
```

```python
from adaptive_k import AdaptiveKRouter

router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")
indices, weights, metrics = router.route(logits, return_metrics=True)

print(f"Compute savings: {metrics.compute_savings:.1%}")
# Output: Compute savings: 47.2%
```

## 📊 Validated Results

| Model | Compute Savings | Quality Retained |
|-------|-----------------|------------------|
| Mixtral 8x7B | **52.5%** | 99.8% |
| Qwen-MoE | **32.4%** | 99.9% |
| OLMoE-1B-7B | **24.7%** | 99.7% |

## 🔗 Links

- **Website**: https://adaptive-k.vertexdata.it
- **PyPI**: https://pypi.org/project/adaptive-k/
- **Paper**: [Entropy-Guided Dynamic Expert Selection](docs/ARXIV_PAPER_DRAFT.md)
- **TensorRT-LLM PR**: [#10672](https://github.com/NVIDIA/TensorRT-LLM/pull/10672)

## 💼 Professional Services

Need help integrating Adaptive-K into your production pipeline?

| Service | Description |
|---------|-------------|
| **Proof of Concept** | Analyze your MoE deployment, estimate savings |
| **Full Implementation** | Production-ready integration with calibration |
| **Enterprise** | Custom solutions, SLA, dedicated support |

👉 **Contact**: https://adaptive-k.vertexdata.it/#contact

---

## 🔬 Research: SBM-Efficient

This repository also contains the research implementation of **SBM (Superposed Bit Model)** - the theoretical foundation behind Adaptive-K.

### Research Documentation

- [SBM Concept](docs/SBM_EFFICIENT_CONCEPT.md) - Mathematical foundations
- [Architecture](docs/SBM_EFFICIENT_ARCHITECTURE.md) - Architectural specifications
- [Implementation Notes](docs/IMPLEMENTATION_NOTES.md) - Implementation guide

### Running Research Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Validate config
python -m src.common.validate_config configs/sbm_mnist.yaml

# Run experiment
python src/experiments/run.py --config configs/sbm_mnist.yaml
```

### Project Structure

```
sbm-efficient/
├── sdk/              # Adaptive-K Python SDK (PyPI package)
├── landing-page/     # Marketing website (Next.js)
├── configs/          # Experiment configurations
├── docs/             # Theoretical documentation
├── src/              # Research source code
│   ├── common/       # Utilities (seed, device, metrics)
│   ├── data/         # Data loaders
│   ├── models/       # Model implementations
│   ├── training/     # Training loops
│   └── experiments/  # Experiment runner
├── results/          # Experiment outputs
└── scripts/          # Utility scripts
```

```bash
# Create feature branch
git checkout -b feature/baseline

# Implement changes following docs/IMPLEMENTATION_NOTES.md

# Run experiments
python src/experiments/run.py --config configs/baseline_mnist.yaml

# Verify metrics.json is generated
cat results/runs/<run_id>/metrics.json

# Merge to main
git checkout main
git merge feature/baseline
```

## 📄 License

Apache 2.0 - Free for commercial use.

## 📞 Contact

- **Email**: amministrazione@vertexdata.it
- **Website**: https://adaptive-k.vertexdata.it
- **GitHub**: https://github.com/Gabrobals/sbm-efficient

---

*Made by [Vertex Data](https://vertexdata.it) - AI Infrastructure Optimization*
