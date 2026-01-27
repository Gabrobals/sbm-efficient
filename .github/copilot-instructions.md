# Adaptive-K: Entropy-Guided Dynamic Expert Selection for MoE

Two halves: **research** (`src/`) and **production SDK** (`sdk/`, `integration-kit/`).  
Core insight: route fewer experts when router is confident → 30-50% compute savings.

## ⚠️ START EVERY SESSION HERE

**Read IN ORDER**:
1. `SESSION_STATE.md` → Current state, what to do NOW
2. `RESEARCH_MASTER_ROADMAP.md` → Research tracks & experiments
3. `BUSINESS_ROADMAP.md` → Commercial status (SEPARATE from research!)

**Current Research Tracks (Jan 2026)**:
| Track | Status | Key Files |
|-------|--------|-----------|
| Adaptive-K v1 | ✅ DONE | `src/routing/adaptive_k.py`, PR #10672 |
| Temporal EMA | 🔄 ACTIVE | `src/models/temporal_router.py` |
| Adaptive Speculative | 📋 NEXT | `adaptive-speculative/` |

## Architecture Overview

```
sbm-efficient/
├── src/                    # Research (PyTorch, MNIST/CIFAR)
│   ├── routing/adaptive_k.py   # AdaptiveKPolicy - core algorithm
│   ├── models/routing.py       # RandomRouting, StaticTopK, SBMRouting
│   ├── models/temporal_router.py # TemporalEMARouter (active track)
│   ├── training/sbm_loops.py   # train_sbm(), get_tau_schedule()
│   └── common/validate_config.py # Config validation (exit 0=OK, 2=invalid)
├── sdk/                    # PyPI package "adaptive-k-routing"
│   └── adaptive_k/router.py    # AdaptiveKRouter (production)
├── integration-kit/        # 2-5 day deployment toolkit
│   ├── calibrate.py            # Find optimal thresholds
│   ├── integration_vllm.py     # vLLM wrapper
│   └── roi_calculator.py       # Business case
├── configs/                # YAML configs - ALWAYS validate first!
└── tensorrt_llm_contribution/  # NVIDIA TensorRT-LLM PR
```

## Essential Commands

```bash
# 1. ALWAYS validate config first
python -m src.common.validate_config configs/sbm_adaptive_k_mnist.yaml

# 2. Run experiment
python src/experiments/run.py --config configs/sbm_adaptive_k_mnist.yaml

# 3. Multi-seed (REQUIRED for publishable results)
python scripts/run_multiseed.py --config configs/sbm_adaptive_k_mnist.yaml --seeds 42,43,44,45,46

# 4. Temporal EMA experiments (active track)
python scripts/run_temporal_experiments.py --experiment all --seeds 42,43,44,45,46
```

## Core Code Patterns

### Router Interface (ALL routers)
```python
indices, weights, entropy = router(features, tau=tau)
# indices: (B, K), weights: (B, K), entropy: scalar
```

### Adaptive-K Threshold Logic
```python
# Config: k_values: [1, 2, 4], h_thresholds: [0.6, 1.2]
# H < 0.6      → K=1 (confident)
# 0.6 ≤ H < 1.2 → K=2 (medium)
# H ≥ 1.2      → K=4 (uncertain)
```

### Tau Schedule (MONOTONIC per EPOCH)
```python
from src.training.sbm_loops import get_tau_schedule
tau = get_tau_schedule("linear", tau_start=2.0, tau_end=0.5, epoch, total_epochs)
# NEVER reset tau per batch!
```

### Temporal EMA Router
```python
# Formula: ema_t = μ * ema_{t-1} + (1-μ) * p_t
# Optimal: μ=0.8, temporal_strength=0.1
router = TemporalEMARouter(d_model=128, n_experts=16, momentum=0.8)
```

## Config Schema (configs/*.yaml)

```yaml
run:
  task: "mnist"           # xor|mnist|fashion_mnist|cifar10
  model: "sbm_adaptive_k" # baseline|sbm|sbm_adaptive_k|random_routing|static_topk
  seed: 42
sbm:
  experts_num: 16
  experts_top_k: 4        # Max K (actual K is dynamic)
  tau_start: 2.0          # High = exploration
  tau_end: 0.5            # Low = exploitation
adaptive_k:               # Only for sbm_adaptive_k model
  k_values: [1, 2, 4]
  h_thresholds: [0.6, 1.2]  # len = k_values - 1, ascending
```

## Critical Anti-Patterns

| ❌ DON'T | ✅ DO |
|---------|------|
| Reset τ per batch | τ monotonic per EPOCH via `get_tau_schedule()` |
| Full compute + mask experts | Execute ONLY selected experts in `ExpertPool` |
| `.to(device)` in loops | `get_device()` once at start |
| Hardcode hyperparams | All config in YAML, validate before run |
| Single seed results | Multi-seed (5+) for publishable results |
| Run without validation | `python -m src.common.validate_config` first |

## Output Structure

`results/runs/{date}_{task}_{model}_seed{N}_{gitsha}/`:
- `config.yaml` - Frozen config
- `metrics.json` - Required: `accuracy`, `flops_executed`, `latency_ms`, `active_modules_mean`, `entropy_mean`

## External Contributions (TensorRT-LLM, vLLM, HF)

**BEFORE pushing to external projects:**

1. **Read CONTRIBUTING.md** of target project
2. **PR title formats**:
   - TensorRT-LLM: `[TICKET][type] Summary` (e.g., `[None][feat] Add Adaptive-K`)
   - vLLM: Conventional commits
3. **Run pre-commit**:
   ```bash
   pre-commit install && pre-commit run --all-files
   ```
4. **Sign commits (DCO)** if required:
   ```bash
   git commit -s -m "message"  # Adds Signed-off-by line
   ```

| Project | Title Format | DCO Required |
|---------|--------------|--------------|
| TensorRT-LLM | `[TICKET][type] Summary` | ✅ Yes |
| vLLM | Conventional commits | ❌ No |
| HuggingFace | `[Component] Description` | ❌ No |
