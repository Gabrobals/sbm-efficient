import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Exploring OlmoeSparseMoeBlock structure:")
print("="*60)

# Get first MoE block
moe_block = model.model.layers[0].mlp
print(f"MoE Block type: {type(moe_block).__name__}")
print(f"\nAttributes and children:")
for name in dir(moe_block):
    if not name.startswith('_'):
        attr = getattr(moe_block, name, None)
        if hasattr(attr, 'weight'):
            print(f"  {name}: {type(attr).__name__} - weight {attr.weight.shape}")
        elif isinstance(attr, torch.nn.Module):
            print(f"  {name}: {type(attr).__name__}")

print(f"\nNamed children:")
for name, child in moe_block.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight'):
        print(f"    weight shape: {child.weight.shape}")

# Check config
print(f"\nModel config MoE settings:")
cfg = model.config
for attr in ['num_experts', 'num_experts_per_tok', 'router', 'moe']:
    if hasattr(cfg, attr):
        print(f"  {attr}: {getattr(cfg, attr)}")
print(f"  num_local_experts: {getattr(cfg, 'num_local_experts', 'N/A')}")
print(f"  num_experts_per_tok: {getattr(cfg, 'num_experts_per_tok', 'N/A')}")
