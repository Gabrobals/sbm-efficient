"""
Adaptive-K applied PER-LAYER based on entropy
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

prompts = [
    "1+1=",
    "Hello",
    "The capital of France is",
    "def quicksort(arr):",
    "Explain the theory of relativity in detail",
    "Write a haiku",
]

print("="*70)
print("ADAPTIVE-K PER-LAYER ANALYSIS")
print("="*70)

# Collect entropy per layer across prompts
layer_entropies = {i: [] for i in range(16)}

for prompt in prompts:
    router_logits.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    for layer_idx, logits in enumerate(router_logits):
        logits = logits.view(-1, 64)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        layer_entropies[layer_idx].append(entropy)

print("\nEntropy variance per layer (high variance = adaptive opportunity):")
print("-"*70)

adaptive_layers = []
static_layers = []

for layer_idx in range(16):
    entropies = layer_entropies[layer_idx]
    mean_h = sum(entropies) / len(entropies)
    variance = sum((h - mean_h)**2 for h in entropies) / len(entropies)
    std_h = variance ** 0.5
    
    # Classify layer
    if std_h > 0.1:  # High variance = adaptive opportunity
        layer_type = "ADAPTIVE"
        adaptive_layers.append(layer_idx)
    else:
        layer_type = "STATIC"
        static_layers.append(layer_idx)
    
    bar = "#" * int(std_h * 50)
    print(f"  Layer {layer_idx:2d}: mean={mean_h:.3f} std={std_h:.3f} [{layer_type:8s}] {bar}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Adaptive layers (high entropy variance): {adaptive_layers}")
print(f"Static layers (low entropy variance):    {static_layers}")
print(f"\nPotential strategy:")
print(f"  - Layers {adaptive_layers}: Use Adaptive-K (2-8 based on entropy)")
print(f"  - Layers {static_layers}: Use fixed K=8")

# Calculate compute savings
adaptive_pct = len(adaptive_layers) / 16 * 100
potential_savings = len(adaptive_layers) / 16 * 0.5 * 100  # Assume 50% reduction on adaptive layers

print(f"\nCompute analysis:")
print(f"  Adaptive layers: {len(adaptive_layers)}/16 ({adaptive_pct:.0f}%)")
print(f"  If we save 50% on adaptive layers: ~{potential_savings:.0f}% total savings")
print("="*70)
