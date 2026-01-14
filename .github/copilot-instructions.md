# SBM-Efficient: Quantum-Inspired Sparse Routing

Sparse expert routing with decoherence-based training on classical PyTorch. Uses quantum mechanics math (superposition → measurement → collapse) as compute-selection primitives, NOT quantum computing.

## Quick Commands

```bash
# 1. Validate config (ALWAYS first, exit 0=OK, 2=invalid)
python -m src.common.validate_config configs/sbm_mnist.yaml

# 2. Pre-flight checks (git, device, imports; exit 0=pass, 1=fail)
python -m src.experiments.preflight configs/sbm_mnist.yaml

# 3. Single experiment run
python -m src.experiments.run --config configs/sbm_mnist.yaml

# 4. Post-flight validation (exit 0=pass, 3=metrics invalid, 5=missing files)
python -m src.experiments.postflight results/runs/<run_id>/

# 5. Multi-seed (REQUIRED for publishable results)
python -m scripts.run_multiseed --config configs/sbm_mnist.yaml --seeds 42,43,44,45,46

# 6. Aggregate and compare
python scripts/compare_results.py
```

## Architecture

```
Input → FeatureExtractor → Router → ExpertPool(sparse) → Classifier → Logits
```

| Component | File | Role |
|-----------|------|------|
| Routers | `src/models/routing.py` | `RandomRouting`, `StaticTopK`, `SBMRouting` |
| Experts | `src/models/experts.py` | `ExpertPool` with TRUE sparse execution |
| Model | `src/models/sbm_model.py` | `SBMModel`, `SBMAdaptiveKModel` |
| Adaptive-K | `src/routing/adaptive_k.py` | Entropy-threshold K selection |
| Tau Schedule | `src/training/sbm_loops.py` | `get_tau_schedule()` for per-epoch τ |

## Model Types

| Model | Config | K Behavior |
|-------|--------|------------|
| `baseline` | `baseline_*.yaml` | All experts (K=N) |
| `sbm` | `sbm_*.yaml` | Fixed K with learnable routing |
| `sbm_adaptive_k` | `sbm_adaptive_k_*.yaml` | Dynamic K per sample via entropy thresholds |
| `random_routing` | `random_routing_*.yaml` | Random K experts |
| `static_topk` | `static_topk_*.yaml` | Same K experts always |

**Tasks**: `xor`, `mnist`, `fashion_mnist`, `cifar10`

## Config Schema

```yaml
run:
  task: "mnist"           # Required: xor|mnist|fashion_mnist|cifar10
  model: "sbm"            # Required: baseline|sbm|sbm_adaptive_k|random_routing|static_topk
  seed: 42                # Required

sbm:                      # Required for non-baseline models
  experts_num: 16
  experts_top_k: 2
  tau_start: 2.0          # High = exploration (soft selection)
  tau_end: 0.5            # Low = exploitation (hard selection)
  tau_schedule: "linear"  # linear|cosine|constant (per EPOCH, not batch!)
  lambda_entropy: 0.01

adaptive_k:               # Only for sbm_adaptive_k
  k_values: [1, 2, 4]
  h_thresholds: [0.6, 1.2]  # len = k_values - 1, ascending
```

## Code Patterns

**Router interface** - ALL routers return `(indices, weights, entropy)`:
```python
indices, weights, entropy = self.router(features, tau=tau)
# indices: (B, K), weights: (B, K), entropy: scalar
```

**ExpertPool** - TRUE sparse execution (NOT full-compute + mask):
```python
# Fixed-K: expert_pool.forward(x, indices, weights) -> (output, flops)
# Adaptive-K: expert_pool.execute_sparse(x, idx_list, weights_list) -> (output, flops)
```

**Tau schedule** - Monotonic per epoch (never reset per batch):
```python
from src.training.sbm_loops import get_tau_schedule
tau = get_tau_schedule("linear", tau_start, tau_end, epoch, total_epochs)
```

**Device/Seed** - Set once at startup:
```python
from src.common.device import get_device  # Handles CUDA fallback
from src.common.seed import set_seed       # Sets random, numpy, torch, cudnn
```

## Critical Anti-Patterns

| DON'T | DO |
|-------|-----|
| Reset τ per batch | τ monotonic per EPOCH via `get_tau_schedule()` |
| Full compute + mask | Execute ONLY selected experts in `ExpertPool` |
| `.to(device)` in loops | `get_device()` once at start |
| Theoretical FLOPs | `Expert.count_flops(batch_size)` for real count |
| Hardcoded hyperparams | All config in YAML, validated before run |
| Print emoji/unicode | ASCII-only logging |
| Single seed results | Multi-seed (5+) for any published result |

## Output Structure

`results/runs/{date}_{task}_{model}_seed{N}_{gitsha}/`:
- `config.yaml` - Frozen experiment config
- `metrics.json` - **Required**: `accuracy`, `flops_executed`, `latency_ms`, `active_modules_mean`, `entropy_mean`
- `stdout.log` - Training log

## Development Rules

1. **Config-first**: Always `validate_config` before running experiments
2. **Theory-first**: Update `docs/*.md` BEFORE implementing architecture changes
3. **Reproducible**: Seeds via `set_seed()`, all outputs as JSON
4. **Multi-seed**: Never publish single-seed; minimum 5 seeds required
