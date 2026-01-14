# Phase E Strategic Roadmap: Post-Mixtral Consolidation

## Executive Summary

After completing Mixtral validation, this document guides the next strategic phase of SBM Adaptive-K development. It covers paper analysis, SLM testing, TensorRT-LLM integration opportunities, and research consolidation.

---

## 1. Current Status (Post-Validation Complete)

| Achievement | Result | Status |
|-------------|--------|--------|
| Qwen-MoE validation | **32.4% compute reduction** | ✅ Done |
| OLMoE-1B-7B validation | **24.7% compute reduction** | ✅ Done |
| Mixtral 8x7B validation | **52.5% compute reduction** | ✅ Done |
| TensorRT-LLM implementation | `AdaptiveKMoeRoutingMethod` class | ✅ Done |
| arXiv paper draft | `docs/ARXIV_PAPER_DRAFT.md` | ✅ Done |

**All three major MoE architectures validated successfully!**

---

## 2. Paper Analysis & Connections

### 2.1 Recursive Language Models (arXiv:2512.24601)

**Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT)
**Published**: December 2025

**Key Concept**: 
> "RLMs treat long prompts as part of an external environment and allow the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt."

**Relevance to Adaptive-K**:

| RLM Concept | Adaptive-K Parallel |
|-------------|---------------------|
| Decompose complex inputs | Allocate more K for complex tokens |
| Process simpler snippets efficiently | Use fewer experts for easy tokens |
| Inference-time scaling | Dynamic compute allocation |

**Research Connection**:
- RLMs scale **context processing** dynamically
- Adaptive-K scales **expert allocation** dynamically
- **Combined approach**: Use entropy to decide BOTH expert count AND recursion depth

**Potential Paper Title**: "Entropy-Guided Dynamic Compute: Combining Adaptive Expert Selection with Recursive Processing"

---

### 2.2 The Lottery Ticket Hypothesis (arXiv:1803.03635)

**Authors**: Jonathan Frankle, Michael Carbin (MIT)
**Published**: 2018, ICLR 2019

**Key Concept**:
> "Dense networks contain sparse subnetworks ('winning tickets') that—when trained in isolation—reach test accuracy comparable to the original network."

**Core Finding**: 
- Winning tickets are **10-20% of original size**
- They learn **faster** and reach **higher accuracy**

**Relevance to Adaptive-K**:

| Lottery Ticket | Adaptive-K Parallel |
|----------------|---------------------|
| Some weights are "winning tickets" | Some experts are more relevant per input |
| Sparse networks can match dense | K=2 can match K=4 on easy inputs |
| Pruning reveals essential structure | Low entropy reveals expert confidence |

**Research Connection**:

```
Lottery Ticket: Find sparse STATIC subnetwork that works for all inputs
Adaptive-K:     Find sparse DYNAMIC subnetwork per input based on difficulty
```

**Key Insight**: Lottery Ticket finds ONE winning ticket. Adaptive-K finds **per-input winning tickets** dynamically.

**Potential Research Direction**:
1. Combine pruning with adaptive routing
2. "Lottery Tickets" in expert selection
3. Identify which experts are "winning tickets" for which input types

---

## 3. Small Language Model (SLM) Testing Strategy

### 3.1 Why SLMs Matter

| Factor | Benefit |
|--------|---------|
| Faster iteration | Test hypotheses quickly |
| Lower cost | Run on consumer GPUs |
| Edge deployment | Mobile/embedded use cases |
| Growing market | Microsoft Phi, Google Gemma, etc. |

### 3.2 Target SLM Models

| Model | Parameters | MoE? | VRAM (4-bit) | Notes |
|-------|------------|------|--------------|-------|
| **Phi-3-mini** | 3.8B | No | ~2GB | Microsoft, can add MoE layer |
| **Gemma-2B** | 2B | No | ~1.5GB | Google, good baseline |
| **SmolLM** | 1.7B | No | ~1GB | HuggingFace |
| **Qwen2-1.5B** | 1.5B | No | ~1GB | Alibaba |
| **OLMoE-1B-7B** | 1B active/7B total | **Yes** | ~4GB | First open MoE SLM! |

### 3.3 Priority: OLMoE-1B-7B

**OLMoE** (Open Language Model with Mixture of Experts) is ideal:
- Open source (Apache 2.0)
- Small enough for local testing
- True MoE architecture (8 experts, top-2)
- From Allen AI

**Test Plan**:
```bash
# Load OLMoE
pip install transformers accelerate
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('allenai/OLMoE-1B-7B-0924')
# Apply Adaptive-K routing
"
```

### 3.4 SLM Experiments

| Experiment | Description | Success Metric |
|------------|-------------|----------------|
| E1: OLMoE baseline | Measure standard top-2 performance | Baseline metrics |
| E2: Adaptive-K on OLMoE | Apply entropy-based K selection | >15% compute reduction |
| E3: Latency benchmark | Real inference time comparison | <10% latency overhead |
| E4: Edge deployment | Run on RTX 3060 6GB | Successfully runs |

---

## 4. TensorRT-LLM Integration Analysis

### 4.1 Current TensorRT-LLM MoE Architecture

From code analysis:

```
tensorrt_llm/_torch/modules/fused_moe/
├── create_moe.py          # Factory function for MoE backends
├── interface.py           # MoE base interface
├── routing.py             # Routing algorithms (CRITICAL)
├── fused_moe_cutlass.py   # CUTLASS backend
├── fused_moe_wide_ep.py   # Wide Expert Parallelism
└── moe_load_balancer.py   # Load balancing
```

### 4.2 Key Integration Points

**File**: `tensorrt_llm/_torch/modules/fused_moe/routing.py`

Current routing methods:
- `BaseMoeRoutingMethod` - Base interface
- Standard top-k routing
- DeepSeek-specific routing with `noaux_tc_op`

**Current limitation** (from Issue #2497):
> "I want to override router expert selection to use custom routing distributions... Is there any way to do this dynamically?"

**Adaptive-K would solve this!**

### 4.3 Proposed TensorRT-LLM Contribution

```python
# New class in routing.py
class AdaptiveKRoutingMethod(BaseMoeRoutingMethod):
    """
    Entropy-based adaptive top-k routing.
    Selects k dynamically based on router confidence.
    """
    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 4,
        entropy_threshold_low: float = 3.55,
        entropy_threshold_high: float = 3.79,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.threshold_low = entropy_threshold_low
        self.threshold_high = entropy_threshold_high
    
    def apply(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute routing weights
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Compute entropy per token
        entropy = -torch.sum(
            routing_weights * torch.log(routing_weights + 1e-9), 
            dim=-1
        )
        
        # Dynamic k selection
        k_values = torch.where(
            entropy < self.threshold_low, 
            self.k_min,
            torch.where(entropy < self.threshold_high, self.k_mid, self.k_max)
        )
        
        # Select top-k experts per token (variable k)
        # ... implementation
        
        return token_selected_experts, token_final_scales
```

### 4.4 Contribution Strategy

| Step | Action | Timeline |
|------|--------|----------|
| 1 | Fork TensorRT-LLM | Day 1 |
| 2 | Implement AdaptiveKRoutingMethod | Days 2-4 |
| 3 | Add unit tests | Days 5-6 |
| 4 | Benchmark vs standard routing | Days 7-8 |
| 5 | Open Issue proposing feature | Day 9 |
| 6 | Submit Pull Request | Day 10 |

---

## 5. Comparison: Adaptive-K vs TensorRT-LLM Routing

| Feature | TensorRT-LLM (Current) | Adaptive-K (Proposed) |
|---------|------------------------|----------------------|
| K value | **Fixed** (e.g., top-2) | **Dynamic** per token |
| Routing decision | Learned gate only | Gate + entropy analysis |
| Compute allocation | Uniform | Input-adaptive |
| Efficiency gain | Baseline | **+32.4%** on Qwen-MoE |
| Implementation | Production-ready | Research-validated |

---

## 6. Research Synthesis Document

### 6.1 Unified Theory: "Difficulty-Aware Sparse Computation"

```
                    ┌─────────────────────────────────────┐
                    │   DIFFICULTY-AWARE SPARSE COMPUTE   │
                    └─────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ Lottery Ticket │      │  Adaptive-K   │      │     RLMs      │
    │   (Static)     │      │  (Dynamic)    │      │  (Recursive)  │
    └───────────────┘      └───────────────┘      └───────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
    Find sparse            Select K experts        Decompose long
    subnetwork that        per token based         prompts into
    works for ALL          on DIFFICULTY           manageable
    inputs                                         chunks
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │     UNIFIED INSIGHT:                │
                    │     Not all computation is equal.   │
                    │     Allocate resources based on     │
                    │     actual difficulty/complexity.   │
                    └─────────────────────────────────────┘
```

### 6.2 Paper Outline: "Dynamic Sparse Computation in Large Language Models"

**Abstract**: We present a unified framework for adaptive compute allocation in LLMs, combining entropy-based expert selection (Adaptive-K), recursive prompt processing (RLMs), and sparse network theory (Lottery Ticket). Our approach achieves 32% compute reduction on MoE models while maintaining quality.

**Sections**:
1. Introduction: The Compute Efficiency Crisis
2. Background: MoE, Lottery Ticket, RLMs
3. Method: Entropy-Guided Adaptive-K Routing
4. Experiments: MNIST → Fashion-MNIST → Qwen-MoE → Mixtral
5. Analysis: When Does Adaptive-K Help Most?
6. Integration: TensorRT-LLM Implementation
7. Future Work: Combining Adaptive-K with RLMs
8. Conclusion

---

## 7. Immediate Action Plan (Updated 2026-01-06)

### Week 1: Consolidation ✅ COMPLETE
- [x] Complete Mixtral test → **52.5% compute reduction**
- [x] Document results in EXECUTIVE_SUMMARY.md → **TRL 7**
- [x] Test OLMoE-1B-7B → **24.7% compute reduction**

### Week 2: TensorRT-LLM Contribution ✅ COMPLETE
- [x] Implement `AdaptiveKMoeRoutingMethod` class
- [x] Create `workspace/adaptive_k_routing_trtllm.py`
- [x] Test implementation locally
- [x] Write PR documentation (`workspace/TRTLLM_PR_README.md`)

### Week 3: arXiv Paper Draft ✅ COMPLETE
- [x] Draft paper in `docs/ARXIV_PAPER_DRAFT.md`
- [x] Include all three model validations
- [x] Add references and related work

### Week 4: Next Steps (TODO)
- [ ] Fork TensorRT-LLM and submit PR
- [ ] Convert markdown draft to LaTeX
- [ ] Submit to arXiv
- [ ] Post LinkedIn announcement
- [ ] Engage with NVIDIA developer community

---

## 8. Commands for Opus (VS Code)

### 8.1 Complete Mixtral Test (if GPU still available)

```bash
# On Vast.ai terminal
cd /workspace/sbm-efficient

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print('Loading Mixtral 8x7B in 4-bit...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
)

print(f'Model loaded! Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# Analyze MoE structure
for name, module in model.named_modules():
    if 'gate' in name.lower() or 'router' in name.lower():
        print(f'{name}: {type(module).__name__}')
"
```

### 8.2 Test OLMoE Locally (6GB GPU)

```bash
# On local machine
cd "c:\Users\ottic\Desktop\SBM Efficent"

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print('Loading OLMoE-1B-7B...')
model = AutoModelForCausalLM.from_pretrained(
    'allenai/OLMoE-1B-7B-0924',
    torch_dtype=torch.float16,
    device_map='auto'
)
print(f'Loaded! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
"
```

### 8.3 Clone TensorRT-LLM for Analysis

```bash
cd "c:\Users\ottic\Desktop"
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
code tensorrt_llm/_torch/modules/fused_moe/routing.py
```

---

## 9. Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Compute reduction (Qwen) | >25% | ✅ 32.4% |
| Compute reduction (Mixtral) | >25% | ⏳ Pending |
| Compute reduction (OLMoE) | >15% | ⏳ Pending |
| TensorRT-LLM PR submitted | Yes | ⏳ Pending |
| arXiv paper draft | Complete | ⏳ Pending |
| LinkedIn engagement | >50 reactions | ⏳ Pending |

---

## 10. Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Mixtral doesn't show same gains | Low | Already validated on Qwen-MoE |
| TensorRT-LLM PR rejected | Medium | Open Issue first, get feedback |
| Competition publishes first | Low | Our results are unique + open source |
| Cloud GPU costs exceed budget | Medium | Use SLMs for most testing |

---

## 11. Contact & Resources

- **Repository**: https://github.com/Gabrobals/sbm-efficient
- **Author**: Gabriele Balsamo (gabriele.balsamo30@gmail.com)
- **License**: GPL v3 (open source) + Commercial available

---

*Document created: January 2026*
*For use with Claude/Opus in VS Code*
