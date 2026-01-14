"""
Mixtral 8x7B - Adaptive-K Routing Analysis
With CPU offloading for 24GB VRAM
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
from datetime import datetime

print("="*60)
print("MIXTRAL 8x7B - ADAPTIVE-K ROUTING TEST")
print("="*60)

# Load model in 4-bit with CPU offload
print("\n[1/4] Loading Mixtral 8x7B in 4-bit with CPU offload...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "22GB", "cpu": "100GB"}
)

print(f"Model loaded! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Analyze MoE structure
print("\n[2/4] Analyzing MoE structure...")
moe_layers = []
for name, module in model.named_modules():
    if 'gate' in name.lower() and hasattr(module, 'weight'):
        moe_layers.append(name)
        print(f"  Found: {name} - shape: {module.weight.shape}")

print(f"\nTotal MoE layers: {len(moe_layers)}")

# Hook to capture router logits
router_data = []

def capture_router_hook(module, input, output):
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    router_data.append(logits.detach().cpu())

# Register hooks on gate layers
hooks = []
for name, module in model.named_modules():
    if 'gate' in name.lower() and hasattr(module, 'weight'):
        hook = module.register_forward_hook(capture_router_hook)
        hooks.append(hook)

# Test prompts
print("\n[3/4] Running inference with test prompts...")
test_prompts = [
    "What is 2+2?",
    "Explain quantum entanglement in simple terms.",
    "Write a haiku about artificial intelligence.",
    "The capital of France is",
    "def fibonacci(n):",
]

results = []
for prompt in test_prompts:
    router_data.clear()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - start_time
    
    # Analyze entropy per layer
    entropies = []
    for logits in router_data:
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0
    
    result = {
        "prompt": prompt[:50],
        "avg_entropy": avg_entropy,
        "latency_ms": latency * 1000,
        "tokens_generated": outputs.shape[1] - inputs.input_ids.shape[1]
    }
    results.append(result)
    print(f"  Prompt: '{prompt[:30]}...' | Entropy: {avg_entropy:.3f} | Latency: {latency*1000:.1f}ms")

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

print(f"Entropy range: {min_h:.3f} - {max_h:.3f}")
print(f"Mean entropy: {mean_h:.3f}")

# Suggest thresholds
h_low = mean_h - (max_h - min_h) * 0.25
h_high = mean_h + (max_h - min_h) * 0.25

print(f"\nSuggested Adaptive-K thresholds:")
print(f"  H < {h_low:.3f} -> K=2 (confident)")
print(f"  {h_low:.3f} <= H < {h_high:.3f} -> K=4 (moderate)")
print(f"  H >= {h_high:.3f} -> K=6 (uncertain)")

# Simulate compute savings
k_assignments = []
for r in results:
    h = r["avg_entropy"]
    if h < h_low:
        k = 2
    elif h < h_high:
        k = 4
    else:
        k = 6
    k_assignments.append(k)
    print(f"  '{r['prompt'][:25]}...' | H={h:.3f} -> K={k}")

avg_k = sum(k_assignments) / len(k_assignments)
baseline_k = 8
compute_reduction = (1 - avg_k / baseline_k) * 100

print(f"\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Average K (Adaptive): {avg_k:.2f}")
print(f"Baseline K: {baseline_k}")
print(f"Compute reduction: {compute_reduction:.1f}%")
print(f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Save results
output = {
    "timestamp": datetime.now().isoformat(),
    "model": "Mixtral-8x7B-Instruct-v0.1",
    "quantization": "4-bit",
    "results": results,
    "adaptive_k": {
        "h_low": h_low,
        "h_high": h_high,
        "k_values": [2, 4, 6],
        "avg_k": avg_k,
        "baseline_k": baseline_k,
        "compute_reduction_pct": compute_reduction
    },
    "vram_gb": torch.cuda.memory_allocated()/1024**3
}

with open("mixtral_adaptive_k_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: mixtral_adaptive_k_results.json")
print("="*60)
