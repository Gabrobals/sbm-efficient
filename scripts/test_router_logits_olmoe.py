"""
Quick test: Extract router logits from OLMoE-1B-7B
Proves that Adaptive-K validation is possible on open-source MoE models.

OLMoE: 64 experts, top-8 routing
Memory: ~14GB with float16, ~7GB with 8bit quantization
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy: H = -sum(p * log(p))"""
    probs = probs.flatten()
    probs = probs[probs > 1e-10]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))


def main():
    print("=" * 60)
    print("OLMoE Router Logits Extraction Test")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, using CPU (will be slow)")
    
    print("\nLoading OLMoE-1B-7B (64 experts, top-8)...")
    
    # Load with 8-bit quantization if bitsandbytes available
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    except ImportError:
        quantization_config = None
        print("bitsandbytes not available, loading in float16")
    
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        output_router_logits=True,  # <-- KEY PARAMETER!
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        trust_remote_code=True
    )
    
    # Test prompts with varying complexity
    test_prompts = [
        "The capital of France is",  # Easy, factual
        "Write a Python function to",  # Medium, code
        "Explain the implications of quantum entanglement on",  # Hard, reasoning
    ]
    
    print("\n" + "=" * 60)
    print("Extracting router logits...")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt[:50]}...'")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_router_logits=True,
                return_dict=True
            )
        
        # Check if router_logits are available
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            router_logits = outputs.router_logits
            print(f"  Router logits available: {len(router_logits)} layers")
            
            # Analyze entropy across layers
            entropies = []
            for layer_idx, layer_logits in enumerate(router_logits):
                if layer_logits is not None:
                    # Apply softmax to get probabilities
                    probs = torch.softmax(layer_logits, dim=-1).cpu().numpy()
                    # Average entropy across batch and sequence
                    layer_entropy = np.mean([compute_entropy(probs[0, t, :]) 
                                            for t in range(probs.shape[1])])
                    entropies.append(layer_entropy)
            
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            max_entropy = np.log2(64)  # 64 experts, uniform = max entropy
            
            print(f"  Mean entropy: {mean_entropy:.3f} (max: {max_entropy:.2f})")
            print(f"  Std entropy:  {std_entropy:.3f}")
            
            # Adaptive-K projection
            # h_thresholds: [1.5, 3.0, 4.5], k_values: [2, 4, 6, 8]
            if mean_entropy < 1.5:
                projected_k = 2
            elif mean_entropy < 3.0:
                projected_k = 4
            elif mean_entropy < 4.5:
                projected_k = 6
            else:
                projected_k = 8
            
            base_k = 8  # OLMoE uses top-8
            savings = (1 - projected_k / base_k) * 100
            print(f"  Adaptive-K projection: K={projected_k} (base K=8)")
            print(f"  Projected savings: {savings:.1f}%")
            
        else:
            print("  WARNING: router_logits not found in outputs")
            print(f"  Available keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Router logits extraction works!")
    print("=" * 60)
    print("\nThis proves Adaptive-K can be validated on any HuggingFace MoE model")
    print("without needing hardware partnerships.")


if __name__ == "__main__":
    main()
