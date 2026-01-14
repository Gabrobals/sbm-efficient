"""
Per-layer expert selection analysis
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

prompts = ["1+1=", "Write a poem about AI"]

print("="*70)
print("PER-LAYER EXPERT SELECTION")
print("="*70)

all_experts = {}
for prompt in prompts:
    router_logits.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    print(f"\nPrompt: '{prompt}'")
    experts_per_layer = []
    for layer_idx, logits in enumerate(router_logits):
        logits = logits.view(-1, 64)
        probs = F.softmax(logits, dim=-1)
        top8 = probs.topk(8, dim=-1).indices[0].tolist()
        experts_per_layer.append(set(top8))
        if layer_idx in [0, 5, 10, 15]:
            print(f"  Layer {layer_idx:2d}: {sorted(top8)}")
    
    all_experts[prompt] = experts_per_layer

# Compare overlap per layer
print("\n" + "="*70)
print("OVERLAP PER LAYER (prompt1 vs prompt2)")
print("="*70)
p1, p2 = prompts[0], prompts[1]
for layer_idx in range(16):
    overlap = len(all_experts[p1][layer_idx] & all_experts[p2][layer_idx])
    bar = "#" * overlap
    print(f"  Layer {layer_idx:2d}: {overlap}/8 {bar}")

print("\n" + "="*70)
print("CONCLUSIONE:")
print("  Se overlap = 8 ovunque -> routing completamente statico")
print("  Se overlap varia -> alcune layer specializzano")
print("="*70)
