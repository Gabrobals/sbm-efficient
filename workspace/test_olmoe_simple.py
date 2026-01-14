"""Quick test with very simple prompts"""
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

# VERY simple prompts
simple_prompts = ["1+1=", "Hi", "Yes", "No", "OK", "A", "The", "cat"]
# Complex prompts  
complex_prompts = ["Explain the theory of relativity", "Write a sonnet about love", "Implement quicksort in Python"]

print("SIMPLE PROMPTS:")
for p in simple_prompts:
    router_logits.clear()
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    h = sum((-F.softmax(l.view(-1,64), -1) * torch.log(F.softmax(l.view(-1,64), -1) + 1e-9)).sum(-1).mean().item() for l in router_logits) / len(router_logits)
    k = 4 if h < 3.0 else (8 if h < 3.5 else 12)
    print(f"  '{p:40}' H={h:.2f} -> K={k}")

print("\nCOMPLEX PROMPTS:")
for p in complex_prompts:
    router_logits.clear()
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    h = sum((-F.softmax(l.view(-1,64), -1) * torch.log(F.softmax(l.view(-1,64), -1) + 1e-9)).sum(-1).mean().item() for l in router_logits) / len(router_logits)
    k = 4 if h < 3.0 else (8 if h < 3.5 else 12)
    print(f"  '{p:40}' H={h:.2f} -> K={k}")

print("\n" + "="*50)
print("CONCLUSIONE: Prompt semplici -> bassa entropia -> meno K")
print("             Prompt complessi -> alta entropia -> piu K")
