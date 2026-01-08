# SBM-Efficient: Quantum-Inspired Sparse Routing

Quantum-inspired ML research using sparse expert routing with decoherence-based training. **Not quantum computing** - uses quantum mechanics math structures on classical PyTorch.

> **Guardrail**: SBM uses superposition as a compute-selection primitive, not as a representation of quantum states. Theory lives in `docs/`, not here.

## Quick Commands

```bash
# Validate config before running
python -m src.common.validate_config configs/sbm_mnist.yaml

# Run pre-flight checks (git status, PyTorch, device)
python -m src.experiments.preflight configs/sbm_mnist.yaml

# Run single experiment
python -m src.experiments.run --config configs/sbm_mnist.yaml

# Run multi-seed validation (5 seeds × all variants)
python -m scripts.run_multiseed

# Compare results
python scripts/compare_results.py
```

## Architecture Flow

```
Input → FeatureExtractor → Router → ExpertPool (sparse) → Classifier
```

**Key components in** [src/models](src/models):
- `baseline.py`: MLP/CNN reference models
- `routing.py`: `RandomRouting`, `StaticTopK`, `SBMRouting` (learnable with tau)
- `experts.py`: `FeatureExtractor`, `ExpertPool` with sparse execution
- `sbm_model.py`: Full `SBMModel` combining all components

## Model Types & Configs

| Model | Config Pattern | Routing Behavior |
|-------|---------------|------------------|
| `baseline` | `baseline_*.yaml` | All experts active (K=N) |
| `sbm` | `sbm_*.yaml` | Learnable routing with decoherence |
| `random_routing` | `random_routing_*.yaml` | Random K experts per sample |
| `static_topk` | `static_topk_*.yaml` | Fixed K experts always |

**Valid tasks**: `xor`, `mnist`, `fashion_mnist`, `cifar10`

## Config Structure (YAML)

Required sections - see [configs/sbm_mnist.yaml](configs/sbm_mnist.yaml) for full example:
```yaml
run:
  task: "mnist"      # xor|mnist|fashion_mnist|cifar10
  model: "sbm"       # baseline|sbm|random_routing|static_topk
  seed: 42

sbm:                 # Required for non-baseline models
  experts_num: 16    # N modules
  experts_top_k: 2   # K active per sample
  tau_start: 2.0     # Initial temperature
  tau_end: 0.5       # Final temperature (decoherence)
  tau_schedule: "linear"  # linear|cosine|constant
  lambda_entropy: 0.01    # Entropy regularization
```

## Training with Decoherence

The `tau` parameter controls exploration vs exploitation via [src/training/sbm_loops.py](src/training/sbm_loops.py):
- **High tau** (start): Soft routing, more exploration
- **Low tau** (end): Sharp routing, exploitation
- Schedule types: `constant`, `linear`, `cosine`

## Output Structure

Every run creates `results/runs/<run_id>/`:
- `config.yaml` - Experiment config copy
- `metrics.json` - **Mandatory metrics**: accuracy, flops_executed, latency_ms, active_modules_mean, entropy_mean
- `stdout.log` - Training output

## Validation Pipeline

1. **Pre-flight** (`src/experiments/preflight.py`): Config schema, git status, device availability
2. **Training**: With progress bars showing `loss`, `acc`, `τ`, `H` (entropy)
3. **Post-flight** (`src/experiments/postflight.py`): Verify metrics.json completeness

## Code Patterns

**Router interface** - all routers return `(indices, weights, entropy)`:
```python
indices, weights, entropy = self.router(features, tau=tau)
# indices: (batch, K) expert indices
# weights: (batch, K) combination weights  
# entropy: scalar routing entropy
```

**Creating models** - use factory functions:
```python
from src.models.baseline import create_baseline_model
from src.models.sbm_model import create_sbm_model
model = create_sbm_model(task="mnist", config=config)
```

## Development Rules

- **Theory-first**: Update `docs/*.md` before changing architecture
- **Config-driven**: All hyperparameters in YAML, never hardcoded
- **Reproducible**: Fixed seeds, JSON logging, multi-seed validation (5 seeds minimum)
- **Language**: Code, configs, and logs in English. Theory docs may be in Italian.

## Gotchas (DON'T)

| Anti-pattern | Correct approach |
|--------------|------------------|
| Reset τ per seed/batch | Monotonic schedule, logged per epoch |
| Full compute + mask | Execute only Top-K experts |
| `.to(device)` in tight loops | Device decided once, passed to modules |
| Measure theoretical FLOPs | Measure real FLOPs on active modules only |
| Print with emoji/unicode | ASCII-safe logging only |

## Next Steps

See [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for roadmap. Pending: Bloch encoding, Cayley operators, observable heads, phase ablations.
