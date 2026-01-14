"""
OLMoE-1B-7B - Adaptive-K Routing Analysis (FIXED)
64 experts, top-8 default routing
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from datetime import datetime

print("="*60)
print("OLMoE-1B-7B - ADAPTIVE-K ROUTING TEST")
print("64 experts, top-8 default")
print("="*60)

# Load model
print("\n[1/4] Loading OLMoE-1B-7B...")
model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Model loaded! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Config: {model.config.num_experts} experts, top-{model.config.num_experts_per_tok}")

# Hook to capture router logits from gate
router_logits_all = []

def capture_gate_hook(module, input, output):
    # Gate output is logits over 64 experts
    router_logits_all.append(output.detach().cpu().float())

# Register hooks on all gate layers
print("\n[2/4] Setting up routing hooks...")
hooks = []
num_layers = len(model.model.layers)
for i in range(num_layers):
    gate = model.model.layers[i].mlp.gate
    hook = gate.register_forward_hook(capture_gate_hook)
    hooks.append(hook)
print(f"Registered hooks on {len(hooks)} layers")

# Test prompts - varied complexity
print("\n[3/4] Running inference with test prompts...")
test_prompts = [
    "2+2=",
    "Hello",
    "The capital of France is",
    "Explain quantum physics",
    "def fibonacci(n):",
    "In the beginning, the universe was",
    "The difference between AI and ML is",
    "Write a poem about the ocean",
]

results = []
for prompt in test_prompts:
    router_logits_all.clear()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - start_time
    
    # Analyze entropy from gate logits
    entropies = []
    for logits in router_logits_all:
        # logits shape: [batch, seq_len, 64] or [batch*seq, 64]
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
        probs = F.softmax(logits, dim=-1)
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0
    
    result = {
        "prompt": prompt,
        "avg_entropy": avg_entropy,
        "latency_ms": latency * 1000,
        "tokens_generated": outputs.shape[1] - inputs.input_ids.shape[1]
    }
    results.append(result)
    print(f"  '{prompt[:35]:<35}' | H={avg_entropy:.3f} | {latency*1000:.0f}ms")

# Remove hooks
for hook in hooks:
    hook.remove()

# Adaptive-K analysis
print("\n[4/4] Adaptive-K Analysis...")
print("="*60)

all_entropies = [r["avg_entropy"] for r in results]
min_h = min(all_entropies)
max_h = max(all_entropies)
mean_h = sum(all_entropies) / len(all_entropies)

print(f"Entropy stats:")
print(f"  Min: {min_h:.3f}")
print(f"  Max: {max_h:.3f}")
print(f"  Mean: {mean_h:.3f}")

# Thresholds based on entropy distribution
h_low = min_h + (max_h - min_h) * 0.33
h_high = min_h + (max_h - min_h) * 0.66

print(f"\nAdaptive-K thresholds:")
print(f"  H < {h_low:.3f} -> K=4 (confident)")
print(f"  {h_low:.3f} <= H < {h_high:.3f} -> K=8 (moderate)")
print(f"  H >= {h_high:.3f} -> K=12 (uncertain)")

# Assign K values
print(f"\nPer-prompt analysis:")
k_assignments = []
for r in results:
    h = r["avg_entropy"]
    if h < h_low:
        k = 4
    elif h < h_high:
        k = 8
    else:
        k = 12
    k_assignments.append(k)
    status = "LOW" if k == 4 else ("MID" if k == 8 else "HIGH")
    print(f"  [{status}] '{r['prompt'][:30]:<30}' H={h:.3f} -> K={k}")

avg_k = sum(k_assignments) / len(k_assignments)
baseline_k = 8  # Default top-k
max_k = 64  # Total experts

# Compute reduction vs baseline
compute_reduction = (1 - avg_k / baseline_k) * 100

print(f"\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Model: OLMoE-1B-7B")
print(f"Architecture: {model.config.num_experts} experts, top-{baseline_k} default")
print(f"")
print(f"Adaptive-K results:")
print(f"  Average K: {avg_k:.2f}")
print(f"  Baseline K: {baseline_k}")
print(f"  Compute change: {compute_reduction:+.1f}%")

# K distribution
k_dist = {}
for k in k_assignments:
    k_dist[k] = k_dist.get(k, 0) + 1

print(f"\nK distribution:")
for k in sorted(k_dist.keys()):
    count = k_dist[k]
    pct = count / len(k_assignments) * 100
    bar = "#" * int(pct / 5)
    print(f"  K={k:2d}: {count}/{len(k_assignments)} ({pct:4.0f}%) {bar}")

print(f"\nVRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Save results
output = {
    "timestamp": datetime.now().isoformat(),
    "model": "OLMoE-1B-7B-0924",
    "architecture": {
        "num_experts": model.config.num_experts,
        "default_top_k": baseline_k,
        "num_layers": num_layers
    },
    "prompts": results,
    "entropy_stats": {
        "min": min_h,
        "max": max_h,
        "mean": mean_h
    },
    "adaptive_k": {
        "thresholds": {"low": h_low, "high": h_high},
        "k_values": [4, 8, 12],
        "k_distribution": k_dist,
        "avg_k": avg_k,
        "baseline_k": baseline_k,
        "compute_change_pct": compute_reduction
    },
    "vram_gb": torch.cuda.memory_allocated()/1024**3
}

with open("olmoe_adaptive_k_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: olmoe_adaptive_k_results.json")
print("="*60)
