"""
Deep analysis of OLMoE router behavior
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

router_logits = []
def hook(module, input, output):
    router_logits.append(output.detach().cpu().float())

for i in range(len(model.model.layers)):
    model.model.layers[i].mlp.gate.register_forward_hook(hook)

print("="*70)
print("DEEP ROUTER ANALYSIS - OLMoE-1B-7B")
print("="*70)

test_prompts = [
    "1+1=",                                    # Math simple
    "The capital of France is",                # Factual
    "def hello():",                            # Code
    "Explain quantum entanglement briefly",    # Complex
]

for prompt in test_prompts:
    router_logits.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    print(f"\n{'='*70}")
    print(f"PROMPT: '{prompt}'")
    print(f"{'='*70}")
    
    # Analyze first layer (layer 0) and last layer (layer 15)
    for layer_idx in [0, 7, 15]:
        if layer_idx >= len(router_logits):
            continue
            
        logits = router_logits[layer_idx]
        if logits.dim() == 3:
            logits = logits.view(-1, 64)
        
        probs = F.softmax(logits, dim=-1)
        
        # Statistics
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        top1_prob = probs.max(dim=-1).values.mean().item()
        top8_probs = probs.topk(8, dim=-1).values.sum(dim=-1).mean().item()
        
        # Expert selection
        top8_indices = probs.topk(8, dim=-1).indices[0].tolist()
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    Entropy: {entropy:.3f}")
        print(f"    Top-1 probability: {top1_prob*100:.1f}%")
        print(f"    Top-8 cumulative probability: {top8_probs*100:.1f}%")
        print(f"    Top-8 experts selected: {top8_indices}")
        
        # Distribution shape
        sorted_probs = probs.sort(dim=-1, descending=True).values[0]
        print(f"    Prob distribution (top 10): {[f'{p:.2%}' for p in sorted_probs[:10].tolist()]}")

print("\n" + "="*70)
print("ANALYSIS: Checking if experts specialize by prompt type")
print("="*70)

# Check expert overlap between different prompt types
expert_sets = {}
for prompt in test_prompts:
    router_logits.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    # Get top-8 experts from layer 8 (middle)
    logits = router_logits[8].view(-1, 64)
    probs = F.softmax(logits, dim=-1)
    top8 = set(probs.topk(8, dim=-1).indices[0].tolist())
    expert_sets[prompt[:20]] = top8
    
print("\nExpert overlap between prompts:")
prompts_list = list(expert_sets.keys())
for i, p1 in enumerate(prompts_list):
    for p2 in prompts_list[i+1:]:
        overlap = len(expert_sets[p1] & expert_sets[p2])
        print(f"  '{p1}' vs '{p2}': {overlap}/8 experts in common")

print("\n" + "="*70)
print("KEY INSIGHT: High overlap = generic routing, Low overlap = specialized")
print("="*70)
