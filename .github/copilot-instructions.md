# SBM-Efficient: Quantum-Inspired Sparse Routing

Quantum-inspired ML research using sparse expert routing with decoherence-based training. **Not quantum computing** - uses quantum mechanics math structures on classical PyTorch.

> **Guardrail**: SBM uses superposition as a compute-selection primitive, not as a representation of quantum states. Theory lives in `docs/`, not here.

## Quick Commands

```bash
# Validate config before running (ALWAYS do this first)
python -m src.common.validate_config configs/sbm_mnist.yaml

# Run pre-flight checks (git status, PyTorch, device)
python -m src.experiments.preflight configs/sbm_mnist.yaml

# Run single experiment
python -m src.experiments.run --config configs/sbm_mnist.yaml

# Run multi-seed validation (5 seeds × all variants) - REQUIRED for results
python -m scripts.run_multiseed --config configs/sbm_mnist.yaml --seeds 42,43,44,45,46

# Compare and aggregate results
python scripts/compare_results.py
```

## Architecture Flow

```
Input → FeatureExtractor → Router → ExpertPool (sparse) → Classifier
```

**Key components in `src/models/`:**
- `baseline.py`: MLP/CNN reference models (full compute baseline)
- `routing.py`: `RandomRouting`, `StaticTopK`, `SBMRouting` (learnable with tau)
- `experts.py`: `FeatureExtractor`, `ExpertPool` with TRUE sparse execution
- `sbm_model.py`: `SBMModel` and `SBMAdaptiveKModel` combining all components

**Routing module in `src/routing/`:**
- `adaptive_k.py`: `AdaptiveKPolicy` - entropy-based dynamic K selection

## Model Types & Configs

| Model | Config Pattern | Routing Behavior |
|-------|---------------|------------------|
| `baseline` | `baseline_*.yaml` | All experts active (K=N) |
| `sbm` | `sbm_*.yaml` | Learnable routing with decoherence |
| `sbm_adaptive_k` | `sbm_adaptive_k_*.yaml` | Dynamic K based on entropy thresholds |
| `random_routing` | `random_routing_*.yaml` | Random K experts per sample |
| `static_topk` | `static_topk_*.yaml` | Fixed K experts always |

**Valid tasks**: `xor`, `mnist`, `fashion_mnist`, `cifar10`

## Config Structure (YAML)

Required sections - see `configs/sbm_mnist.yaml`:
```yaml
run:
  task: "mnist"      # xor|mnist|fashion_mnist|cifar10
  model: "sbm"       # baseline|sbm|sbm_adaptive_k|random_routing|static_topk
  seed: 42

sbm:                 # Required for non-baseline models
  experts_num: 16
  experts_top_k: 2
  tau_start: 2.0
  tau_end: 0.5
  tau_schedule: "linear"  # linear|cosine|constant
  lambda_entropy: 0.01

adaptive_k:          # Required ONLY for sbm_adaptive_k model
  k_values: [1, 2, 4]
  h_thresholds: [0.5, 1.5]  # len = len(k_values) - 1, ascending
```

## Training with Decoherence

Temperature `tau` controls exploration→exploitation via `src/training/sbm_loops.py`:
- **High tau** (start): Soft routing, exploration
- **Low tau** (end): Sharp routing, exploitation
- Computed by `get_tau_schedule()` - MONOTONIC per epoch, never reset

## Output Structure

Every run creates `results/runs/<run_id>/` where run_id = `{date}_{task}_{model}_seed{N}_{gitsha}`:
- `config.yaml` - Experiment config copy
- `metrics.json` - **Mandatory**: accuracy, flops_executed, latency_ms, active_modules_mean, entropy_mean, noise evaluations
- `stdout.log` - Training output

Aggregated results: `results/aggregated_results.json`

## Code Patterns

**Router interface** - ALL routers return `(indices, weights, entropy)`:
```python
indices, weights, entropy = self.router(features, tau=tau)
# indices: (batch, K) expert indices
# weights: (batch, K) combination weights  
# entropy: scalar routing entropy
```

**Creating models** - ALWAYS use factory functions:
```python
from src.models.sbm_model import create_sbm_model, create_sbm_adaptive_k_model
model = create_sbm_model(task="mnist", config=config)
model = create_sbm_adaptive_k_model(task="mnist", config=config)  # for adaptive K
```

**ExpertPool sparse execution** - executes ONLY selected experts (no masking):
```python
# In experts.py - iterates unique_experts, processes only samples using each
for expert_idx in unique_experts:
    selected_x = x[samples_using_expert]
    expert_output = self.experts[expert_idx](selected_x)  # REAL sparse!
```

## Critical Anti-Patterns (DON'T)

| Anti-pattern | Correct approach |
|--------------|------------------|
| Reset τ per seed/batch | Monotonic schedule via `get_tau_schedule()` |
| Full compute + mask | Execute ONLY Top-K via `ExpertPool.forward()` |
| `.to(device)` in loops | Device set once in `src/common/device.py` |
| Theoretical FLOPs | Real FLOPs from `Expert.count_flops()` |
| Hardcoded hyperparams | ALL config in YAML, validated by `validate_config.py` |
| Print emoji/unicode | ASCII-only logging |
| Single seed results | Multi-seed (5 minimum) via `run_multiseed.py` |

## Development Rules

- **Theory-first**: Update `docs/*.md` BEFORE changing architecture
- **Config-driven**: All hyperparameters in YAML, never hardcoded
- **Reproducible**: Fixed seeds via `src/common/seed.py`, JSON logging
- **Validate before run**: Always `python -m src.common.validate_config <yaml>`

## Key Files for Understanding

- `src/training/sbm_loops.py`: Main training loop, tau scheduling, noise robustness testing
- `src/models/routing.py`: Router implementations (SBMRouting is the learnable one)
- `src/routing/adaptive_k.py`: AdaptiveKPolicy for dynamic K selection
- `src/common/validate_config.py`: Config schema enforcement
- `docs/IMPLEMENTATION_NOTES.md`: Full implementation spec and roadmap
