# SBM-Efficient: Quantum-Inspired Sparse Routing

Implementazione rigorosa del modello **SBM (Superposed Bit Model)** con routing sparso ed efficiente basato su misura quantistica e decoerenza controllata.

## Documenti Teorici

- [docs/SBM_EFFICIENT_CONCEPT.md](docs/quantum_inspired_bloch_models_teoria_matematica.md) - Fondamenti matematici
- [docs/SBM_EFFICIENT_ARCHITECTURE.md](docs/sbm_h_superposed_bit_model_in_hilbert_space.md) - Specifiche architetturali
- [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) - Guida implementativa

## Quick Start

### Validation (do this first)

```bash
# Install dependencies
pip install -r requirements.txt

# Validate a config file
python -m src.common.validate_config configs/sbm_mnist.yaml

# Run full pre-flight checks
python -m src.experiments.preflight configs/baseline_mnist.yaml
```

### Running Experiments (after validation passes)

```bash
# Run baseline on MNIST
python src/experiments/run.py --config configs/baseline_mnist.yaml

# Run SBM on MNIST
python src/experiments/run.py --config configs/sbm_mnist.yaml

# Validate completed run
python -m src.experiments.postflight results/runs/<run_id>/
```

## Project Structure

```
sbm-efficient/
├── configs/          # Experiment configurations
├── docs/             # Theoretical documentation
├── src/              # Source code
│   ├── common/       # Utilities (seed, device, metrics)
│   ├── data/         # Data loaders
│   ├── models/       # Model implementations
│   ├── training/     # Training loops
│   ├── profiling/    # FLOPs and latency measurement
│   └── experiments/  # Experiment runner
├── results/          # Experiment outputs
└── scripts/          # Shell scripts
```

## Key Features

- **Sparse routing**: Only active modules are executed (real FLOPs savings)
- **Decoherence schedule**: Progressive collapse from exploration to exploitation
- **Rigorous profiling**: Real FLOPs and latency measurement
- **Reproducible**: YAML configs + seed control + JSON logging

## Experiment Protocol

1. **Synthetic tasks**: XOR generalization
2. **Vision tasks**: MNIST, Fashion-MNIST, CIFAR-10
3. **Robustness**: Noise, occlusion, inversion

## Mandatory Metrics

Every run produces `results/runs/<run_id>/metrics.json` with:

- Test accuracy
- Training stability (loss variance)
- FLOPs executed (only active modules)
- Latency (p50/p90/p99)
- Active modules mean
- Entropy mean

## Development Workflow

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

## License

Research project - see documentation for details.
