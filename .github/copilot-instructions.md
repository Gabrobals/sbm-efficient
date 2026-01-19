# Adaptive-K: Entropy-Guided Dynamic Expert Selection for MoE

This repo has two halves: **research** (SBM experiments in `src/`) and **production SDK** (`sdk/`, `integration-kit/`). Both share the core insight: route fewer experts when router is confident.

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Research experiments (PyTorch, MNIST/CIFAR) |
| `sdk/` | PyPI package `adaptive-k-routing` for production MoE |
| `integration-kit/` | 2-5 day deployment toolkit (vLLM, HuggingFace) |
| `configs/` | YAML configs for all experiment variants |
| `landing-page/` | Next.js marketing site |
| `arxiv_paper/` | LaTeX paper source |

## Research Workflow (src/)

```bash
# 1. Validate config (ALWAYS first, exit 0=OK, 2=invalid)
python -m src.common.validate_config configs/sbm_mnist.yaml

# 2. Pre-flight checks (git, device, imports; exit 0=pass, 1=fail)
python -m src.experiments.preflight configs/sbm_mnist.yaml

# 3. Single experiment
python -m src.experiments.run --config configs/sbm_mnist.yaml

# 4. Post-flight validation (exit 0=pass, 3=metrics invalid)
python -m src.experiments.postflight results/runs/<run_id>/

# 5. Multi-seed (REQUIRED for publishable results)
python scripts/run_multiseed.py --config configs/sbm_mnist.yaml --seeds 42,43,44,45,46
```

## Research Architecture

```
Input → FeatureExtractor → Router → ExpertPool(sparse) → Classifier → Logits
```

| Component | File | Role |
|-----------|------|------|
| Routers | `src/models/routing.py` | `RandomRouting`, `StaticTopK`, `SBMRouting` |
| Experts | `src/models/experts.py` | `ExpertPool` with TRUE sparse execution |
| Adaptive-K | `src/routing/adaptive_k.py` | `AdaptiveKPolicy` entropy-threshold K |
| Tau Schedule | `src/training/sbm_loops.py` | `get_tau_schedule()` per-epoch τ |

**Model types**: `baseline`, `sbm`, `sbm_adaptive_k`, `random_routing`, `static_topk`
**Tasks**: `xor`, `mnist`, `fashion_mnist`, `cifar10`

## Integration Kit (integration-kit/)

Production deployment in 2-5 days:
```bash
python integration-kit/roi_calculator.py --tokens-per-day 1B    # Day 1: ROI
python integration-kit/calibrate.py --demo                      # Day 1-2: Thresholds
python integration-kit/integration_vllm.py --demo               # Day 2-3: vLLM
python integration-kit/monitoring_dashboard.py --demo           # Day 3-4: Metrics
python integration-kit/ab_test_framework.py --demo              # Day 4-5: Rollout
```

## Code Patterns

**Router interface** - ALL routers return `(indices, weights, entropy)`:
```python
indices, weights, entropy = self.router(features, tau=tau)
# indices: (B, K), weights: (B, K), entropy: scalar
```

**Adaptive-K threshold policy** - entropy determines K:
```python
# Low entropy (confident) → small K; High entropy → large K
# h_thresholds: [0.6, 1.2], k_values: [1, 2, 4]
# H < 0.6 → K=1, 0.6 ≤ H < 1.2 → K=2, H ≥ 1.2 → K=4
```

**ExpertPool** - TRUE sparse execution (NOT full-compute + mask):
```python
# Only selected experts execute - real compute savings
output, flops = expert_pool.forward(x, indices, weights)
```

**Tau schedule** - Monotonic per EPOCH (never per batch):
```python
from src.training.sbm_loops import get_tau_schedule
tau = get_tau_schedule("linear", tau_start, tau_end, epoch, total_epochs)
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

## Config Schema (configs/*.yaml)

```yaml
run:
  task: "mnist"           # xor|mnist|fashion_mnist|cifar10
  model: "sbm_adaptive_k" # baseline|sbm|sbm_adaptive_k|random_routing|static_topk
  seed: 42

sbm:
  experts_num: 16
  experts_top_k: 4        # Max K (actual is dynamic for adaptive_k)
  tau_start: 2.0          # High = exploration
  tau_end: 0.5            # Low = exploitation
  tau_schedule: "linear"  # linear|cosine|constant

adaptive_k:               # Only for sbm_adaptive_k model
  k_values: [1, 2, 4]
  h_thresholds: [0.6, 1.2]  # len = k_values - 1, ascending
```

## Output Structure

`results/runs/{date}_{task}_{model}_seed{N}_{gitsha}/`:
- `config.yaml` - Frozen config
- `metrics.json` - Required: `accuracy`, `flops_executed`, `latency_ms`, `active_modules_mean`, `entropy_mean`

## Development Rules

1. **Config-first**: Always `validate_config` before experiments
2. **Theory-first**: Update `docs/*.md` BEFORE architecture changes
3. **Multi-seed**: Never publish single-seed; minimum 5 seeds
4. **SDK changes**: Update both `sdk/` and `integration-kit/` examples

---

## Adaptive-K 2.0 (Strategic Evolution)

> Roadmap: [docs/ADAPTIVE_K_2_ROADMAP.md](../docs/ADAPTIVE_K_2_ROADMAP.md)

### Target Integrations

| Target | Architecture | Integration Point |
|--------|--------------|-------------------|
| **Nemotron 3 Nano** | 128 experts, top-6 fixed, sigmoid gating | Replace top-6 → adaptive-K in router MLP |
| **Nemotron 3 Super/Ultra** | Latent MoE, 4x expert expansion | Scale with latent routing |
| **Cerebras WSE-3** | Native sparsity, SLAC cores | Hardware co-design for dynamic routing |

### Two Parallel Tracks

**Track A: Adaptive Speculative Decoding** (`adaptive-speculative/`)
```
Entropy → Draft Length (not expert count)
High confidence → K=16 drafts with tiny model
Low confidence → K=1 or skip speculation
```
- MVP: 4 weeks, target vLLM integration
- Stacks with MoE Adaptive-K: 0.7 × 0.7 = 51% savings

**Track B: Information Flow Monitor** (research)
```
Mutual Information → Layer Skip Decisions
MI(layer_i, final_output) ≈ 0 → skip layer
```
- Paper target: 12 weeks
- Combined potential: 70%+ compute reduction

### Key Differences from Adaptive-K 1.0

| Aspect | Adaptive-K 1.0 | Adaptive-K 2.0 |
|--------|----------------|----------------|
| Scope | MoE expert routing | Speculative decoding + layer skipping |
| Signal | Router entropy | Draft confidence + MI flow |
| Targets | Research models (MNIST) | Production LLMs (Nemotron, Llama) |
| Code | `src/routing/adaptive_k.py` | `adaptive-speculative/` |

### Working with Adaptive-K 2.0

```bash
# Adaptive Speculative Decoding (Track A)
cd adaptive-speculative
pip install -e .
python experiments/profile_entropy.py --model llama-3-8b

# Information Flow Monitor (Track B) - future
# Code will be in separate repo when ready
```

### Strategic Context

- **Cerebras**: $10B OpenAI deal for low-latency inference → Adaptive-K reduces tokens needed
- **Nvidia Nemotron 3**: Open-source MoE with reasoning budget control → Adaptive-K makes it automatic
- **DeepSeek-V3**: 671B MoE, K=8 fixed → Active discussion (Issue #1089, high-priority internal review)
- **Combined value**: Hardware speed × software efficiency = multiplicative gains

### Active Outreach (outreach/)

| Target | Status | Next Step |
|--------|--------|-----------|
| DeepSeek | 🟢 High-priority review | Awaiting feedback (~Feb 2) |
| Nvidia | 🟡 Planned | Fork Nemotron 3 Nano |
| Cerebras | 🟡 Strategy ready | LinkedIn outreach to Hagay Lupesko |
