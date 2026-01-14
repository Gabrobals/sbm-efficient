# SBM-Efficient: Quantum-Inspired Sparse Routing

Sparse expert routing with decoherence-based training on classical PyTorch. Uses quantum mechanics math (superposition → measurement → collapse) as compute-selection primitives, NOT quantum computing.

## Quick Commands

```bash
# 1. Validate config (ALWAYS first)
python -m src.common.validate_config configs/sbm_mnist.yaml

# 2. Pre-flight checks (git status, device, imports)
python -m src.experiments.preflight configs/sbm_mnist.yaml

# 3. Single experiment run
python -m src.experiments.run --config configs/sbm_mnist.yaml

# 4. Multi-seed (REQUIRED for publishable results)
python -m scripts.run_multiseed --config configs/sbm_mnist.yaml --seeds 42,43,44,45,46

# 5. Aggregate and compare
python scripts/compare_results.py
```

## Architecture

```
Input → FeatureExtractor → Router → ExpertPool(sparse) → Classifier → Logits
```

| Component | File | Role |
|-----------|------|------|
| Routers | `src/models/routing.py` | `RandomRouting`, `StaticTopK`, `SBMRouting` (learnable) |
| Experts | `src/models/experts.py` | `ExpertPool` with TRUE sparse execution |
| Model | `src/models/sbm_model.py` | `SBMModel`, `SBMAdaptiveKModel` |
| Adaptive-K | `src/routing/adaptive_k.py` | Entropy-threshold K selection |

## Model Types

| Model | Config | Behavior |
|-------|--------|----------|
| `baseline` | `baseline_*.yaml` | All experts (K=N) |
| `sbm` | `sbm_*.yaml` | Learnable routing + tau schedule |
| `sbm_adaptive_k` | `sbm_adaptive_k_*.yaml` | Dynamic K per sample |
| `random_routing` | `random_routing_*.yaml` | Random K experts |
| `static_topk` | `static_topk_*.yaml` | Fixed K always |

**Tasks**: `xor`, `mnist`, `fashion_mnist`, `cifar10`

## Config Schema

```yaml
run:
  task: "mnist"           # Required
  model: "sbm"            # Required
  seed: 42                # Required

sbm:                      # Required for sbm/sbm_adaptive_k/random/static
  experts_num: 16
  experts_top_k: 2
  tau_start: 2.0
  tau_end: 0.5
  tau_schedule: "linear"  # linear|cosine|constant
  lambda_entropy: 0.01

adaptive_k:               # Only for sbm_adaptive_k
  k_values: [1, 2, 4]
  h_thresholds: [0.6, 1.2]  # len = k_values - 1, ascending

robustness_input:         # Optional: input robustness testing
  enabled: true
  gaussian_sigmas: [0.0, 0.1, 0.2, 0.3]
  salt_pepper_probs: [0.0, 0.05, 0.1]
  occlusion_ratios: [0.0, 0.15, 0.3]
  inversion: true
```

## Code Patterns

**Router interface** - ALL routers return `(indices, weights, entropy)`:
```python
indices, weights, entropy = self.router(features, tau=tau)
# indices: (B, K), weights: (B, K), entropy: scalar
```

**Model creation** - Use factory functions:
```python
from src.models.sbm_model import create_sbm_model
model = create_sbm_model(task="mnist", config=config)
```

**ExpertPool** - TRUE sparse (no mask trick):
```python
for expert_idx in unique_experts:
    selected_x = x[samples_using_expert]
    output = self.experts[expert_idx](selected_x)  # Only selected!
```

**Device/Seed** - Set once via utilities:
```python
from src.common.device import get_device
from src.common.seed import set_seed
```

## Critical Anti-Patterns

| DON'T | DO |
|-------|-----|
| Reset τ per batch | Monotonic via `get_tau_schedule()` |
| Full compute + mask | Execute ONLY Top-K in `ExpertPool` |
| `.to(device)` in loops | `get_device()` once at start |
| Theoretical FLOPs | `Expert.count_flops()` (real) |
| Hardcoded hyperparams | All in YAML, validated |
| Print emoji/unicode | ASCII-only logging |
| Single seed results | Multi-seed (5+) always |

## Output Structure

`results/runs/{date}_{task}_{model}_seed{N}_{gitsha}/`:
- `config.yaml` - Experiment config
- `metrics.json` - **Required**: accuracy, flops_executed, latency_ms, active_modules_mean, entropy_mean
- `stdout.log` - Training log

## Development Rules

1. **Config-first**: `python -m src.common.validate_config <yaml>` before running
2. **Theory-first**: Update `docs/*.md` BEFORE architecture changes
3. **Reproducible**: Seeds via `src/common/seed.py`, JSON metrics
4. **Multi-seed**: Never publish single-seed; 5+ required
