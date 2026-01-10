# ROADMAP Phase D: MoE Validation on Real Models

## ✅ PHASE D COMPLETE

**Result**: 32.4% expert compute reduction validated on Qwen1.5-MoE-A2.7B.

See [PHASE_D_RESULTS.md](PHASE_D_RESULTS.md) for detailed results.

---

## Objective
Validate SBM Adaptive-K routing on real Mixture-of-Experts language models to demonstrate that the concept works beyond MNIST/Fashion-MNIST.

## Current Status (Completed)
- [x] Phase A: Metrics hardening
- [x] Phase B: MNIST validation (17% FLOPs reduction)
- [x] Phase C: Fashion-MNIST validation (adaptive behavior confirmed)
- [x] Executive Summary + GitHub public repo
- [x] GPL + Commercial dual license
- [x] **Phase D: Real MoE validation (32.4% savings)**

## Phase D: Real MoE Validation

### D1. Local Setup (Cost: 0€) ✅
**Goal**: Verify CUDA works, install dependencies, test with tiny model

Steps:
1. ✅ Verify RTX 3060 CUDA availability
2. ✅ Install transformers, bitsandbytes, accelerate
3. ✅ Test loading a small model (TinyLlama or Phi-2)
4. ✅ Verify inference works

Success criteria:
- [x] torch.cuda.is_available() == True
- [x] Can load and run inference on small model
- [x] Baseline latency/memory measured

### D2. Local MoE Test (Cost: 0€) → Moved to Cloud ✅
**Goal**: Test with smallest MoE model that fits in 12GB VRAM

Note: RTX 3060 (6GB) insufficient for MoE models. Moved to Vast.ai RTX 4090.

Target models (in order of preference):
1. ✅ Qwen1.5-MoE-A2.7B-Chat (4-bit) - ~8GB VRAM

Steps:
1. ✅ Load MoE model in 4-bit quantization
2. ✅ Run baseline inference (standard Top-K routing)
3. ✅ Measure: tokens/sec, memory usage, perplexity on sample text
4. ✅ Identify where routing happens in model architecture

Success criteria:
- [x] MoE model loads on RTX 4090
- [x] Baseline metrics recorded (9.4 tok/s, 8.38GB VRAM)
- [x] Routing mechanism located (60 experts, K=4)

### D3. Implement Adaptive-K for Real MoE (Cost: 0€) ✅
**Goal**: Modify routing to use entropy-based Adaptive-K

Steps:
1. ✅ Create wrapper/hook for MoE routing layer
2. ✅ Implement entropy calculation on router logits
3. ✅ Implement dynamic K selection based on thresholds
4. ✅ Test on same samples as D2

Success criteria:
- [x] Adaptive-K routing implemented (entropy-based K selection)
- [x] Can switch between baseline and adaptive routing
- [x] No crashes, correct output format

### D4. Cloud Benchmark (Cost: ~$2 USD) ✅
**Goal**: Run proper benchmarks on larger GPU

Platform: Vast.ai
GPU: RTX 4090 (24GB VRAM)
Time: ~2 hours

Steps:
1. ✅ Setup cloud instance with CUDA
2. ✅ Clone repo, install dependencies
3. ✅ Run benchmarks on Qwen-MoE
4. ✅ Compare: Baseline Top-K vs Adaptive-K
5. ✅ Measure: Entropy distribution, K distribution, savings

Success criteria:
- [x] MoE model runs with both routing methods
- [x] Adaptive-K shows **32.4% compute reduction**
- [x] Results documented in JSON format

### D5. Documentation & Publication (Cost: 0�)
**Goal**: Document results, update repo, publish

Steps:
1. Create PHASE_D_RESULTS.md with all benchmarks
2. Update EXECUTIVE_SUMMARY.md with MoE results
3. Update README.md with new capabilities
4. Write LinkedIn post with real MoE results
5. (Optional) Write arXiv paper draft

Success criteria:
- [ ] Results documented
- [ ] GitHub updated
- [ ] LinkedIn post published
- [ ] Credibility established for MoE routing

---

## Technical Details

### Adaptive-K Implementation for Transformers MoE

The key modification is in the router forward pass:
```python
# Standard Top-K (baseline)
def forward(self, hidden_states):
    router_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
    # Always selects exactly top_k experts
    
# Adaptive-K (our method)
def forward_adaptive(self, hidden_states):
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=-1)
    
    # Calculate entropy per token
    entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-9), dim=-1)
    
    # Dynamic K based on entropy thresholds
    k_values = torch.where(entropy < self.threshold_low, self.k_min,
               torch.where(entropy < self.threshold_high, self.k_mid, self.k_max))
    
    # Select top-k with variable k per token
    # (implementation varies by framework)
```

### Key Files to Modify

For HuggingFace Transformers:
- `transformers/models/mixtral/modeling_mixtral.py` - MixtralSparseMoeBlock
- `transformers/models/qwen2_moe/modeling_qwen2_moe.py` - Qwen2MoeSparseMoeBlock

### Metrics to Collect

| Metric | How to measure |
|--------|----------------|
| Perplexity | `loss.exp()` on validation set |
| Tokens/sec | `num_tokens / inference_time` |
| Active experts mean | Count non-zero expert activations |
| K_mean | Average K selected per token |
| Memory | `torch.cuda.max_memory_allocated()` |

---

## Timeline

| Phase | Duration | Cost |
|-------|----------|------|
| D1: Local setup | 1-2 hours | 0� |
| D2: Local MoE test | 2-3 hours | 0� |
| D3: Implement Adaptive-K | 4-6 hours | 0� |
| D4: Cloud benchmark | 1-2 days | ~20� |
| D5: Documentation | 2-3 hours | 0� |

**Total: ~2-3 days of work, ~20� cost**

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MoE doesn't fit in 12GB | Use 4-bit quantization, or skip to cloud |
| Adaptive-K breaks model | Test on tiny batches first, compare outputs |
| Cloud GPU issues | Have backup plan (RunPod vs Vast.ai) |
| No improvement shown | Document findings honestly, analyze why |

---

## Success Definition

**Minimum success**: Adaptive-K runs on a real MoE without breaking it
**Good success**: Measurable FLOPs/latency reduction with same perplexity
**Great success**: Paper-worthy results (significant efficiency gain)

---

## Commands Reference
```bash
# Install dependencies
pip install transformers accelerate bitsandbytes scipy

# Load model in 4-bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

*Document created: January 2026*
*Author: Gabriele Balsamo (partita IVA � 18354371009)*
*Contact: gabriele.balsamo30@gmail.com*
