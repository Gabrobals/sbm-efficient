"""
Final simulation: Per-layer Adaptive-K vs Baseline
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

router_logits = []
def hook(module, input, output):
    router_logits.append(output.detach().cpu().float())

for i in range(len(model.model.layers)):
    model.model.layers[i].mlp.gate.register_forward_hook(hook)

# Define layer strategy based on our analysis
ADAPTIVE_LAYERS = {2, 3, 8, 9, 10, 11, 12, 13, 14, 15}
STATIC_LAYERS = {0, 1, 4, 5, 6, 7}

# Test prompts - mix of complexities
prompts = [
    "Hi", "1+1=", "Yes", "The", "OK",  # Simple
    "The capital of France is",         # Factual
    "def fibonacci(n):",                 # Code
    "Explain machine learning",          # Medium
    "Write a detailed essay about AI",   # Complex
    "What is quantum entanglement?",     # Science
]

print("="*70)
print("FINAL SIMULATION: Per-Layer Adaptive-K")
print("="*70)
print(f"Adaptive layers: {sorted(ADAPTIVE_LAYERS)}")
print(f"Static layers: {sorted(STATIC_LAYERS)}")
print("="*70)

total_baseline_k = 0
total_adaptive_k = 0
num_layers = 16

results = []

for prompt in prompts:
    router_logits.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    prompt_baseline = 0
    prompt_adaptive = 0
    
    for layer_idx, logits in enumerate(router_logits):
        logits = logits.view(-1, 64)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        
        # Baseline: always K=8
        baseline_k = 8
        
        # Adaptive-K strategy
        if layer_idx in ADAPTIVE_LAYERS:
            # Dynamic K based on entropy
            if entropy < 3.3:
                adaptive_k = 4
            elif entropy < 3.6:
                adaptive_k = 6
            else:
                adaptive_k = 8
        else:
            # Static layers: always K=8
            adaptive_k = 8
        
        prompt_baseline += baseline_k
        prompt_adaptive += adaptive_k
    
    total_baseline_k += prompt_baseline
    total_adaptive_k += prompt_adaptive
    
    savings = (1 - prompt_adaptive / prompt_baseline) * 100
    results.append({
        "prompt": prompt,
        "baseline": prompt_baseline,
        "adaptive": prompt_adaptive,
        "savings": savings
    })
    
    print(f"  '{prompt[:35]:<35}' | Baseline: {prompt_baseline:3d} | Adaptive: {prompt_adaptive:3d} | Savings: {savings:+5.1f}%")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

overall_savings = (1 - total_adaptive_k / total_baseline_k) * 100
avg_baseline = total_baseline_k / len(prompts)
avg_adaptive = total_adaptive_k / len(prompts)

print(f"Total experts called (baseline):  {total_baseline_k}")
print(f"Total experts called (adaptive):  {total_adaptive_k}")
print(f"Average per prompt (baseline):    {avg_baseline:.1f}")
print(f"Average per prompt (adaptive):    {avg_adaptive:.1f}")
print(f"\n>>> OVERALL COMPUTE REDUCTION: {overall_savings:.1f}% <<<")
print("="*70)

# Save results
import json
output = {
    "model": "OLMoE-1B-7B",
    "strategy": "per-layer-adaptive-k",
    "adaptive_layers": sorted(ADAPTIVE_LAYERS),
    "static_layers": sorted(STATIC_LAYERS),
    "results": results,
    "summary": {
        "total_baseline_k": total_baseline_k,
        "total_adaptive_k": total_adaptive_k,
        "compute_reduction_pct": overall_savings
    }
}
with open("final_olmoe_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to: final_olmoe_results.json")
