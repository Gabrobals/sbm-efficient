"""
Explore OLMoE structure to find correct router modules
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading OLMoE-1B-7B...")
model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"\nVRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print("\n" + "="*60)
print("MODEL STRUCTURE - Looking for MoE components:")
print("="*60)

for name, module in model.named_modules():
    module_type = type(module).__name__
    # Look for anything related to MoE/routing/gate/expert
    if any(x in name.lower() or x in module_type.lower() 
           for x in ['moe', 'gate', 'router', 'expert', 'switch']):
        print(f"\n{name}")
        print(f"  Type: {module_type}")
        if hasattr(module, 'weight'):
            print(f"  Weight shape: {module.weight.shape}")
        # Print submodules
        for subname, submod in module.named_children():
            print(f"    .{subname}: {type(submod).__name__}")

print("\n" + "="*60)
print("First layer structure:")
print("="*60)
# Look at first transformer layer
for name, module in model.named_modules():
    if 'layers.0' in name and len(name.split('.')) <= 4:
        print(f"{name}: {type(module).__name__}")
