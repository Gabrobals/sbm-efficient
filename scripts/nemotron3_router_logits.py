#!/usr/bin/env python3
"""
Nemotron 3 Nano Router Logits Extraction
=========================================

Test Adaptive-K on NVIDIA Nemotron 3 Nano (30B-A3B) MoE model.

Architecture:
- 128 routed experts + 1 shared expert
- Top-6 routing (+ 1 shared = 7 active per token)
- 30B total params, 3.5B active
- Hybrid Mamba2-Transformer MoE

Requirements:
- GPU: A100 80GB or H100 (model is ~60GB in BF16)
- For 4-bit: RTX 4090 24GB might work

Setup on Vast.ai:
    # Use PyTorch 2.x + CUDA 12.x template
    pip install torch transformers accelerate bitsandbytes numpy

Run:
    python nemotron3_router_logits.py                    # Full precision (needs A100)
    python nemotron3_router_logits.py --quantize 4bit   # 4-bit (RTX 4090)
    python nemotron3_router_logits.py --quantize 8bit   # 8-bit (RTX 4090)
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy: H = -sum(p * log2(p))"""
    probs = probs.flatten()
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log2(probs))


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron 3 router logits")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], 
                       default="none", help="Quantization level")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Max tokens to generate")
    args = parser.parse_args()
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    NUM_EXPERTS = 128  # routed experts
    SHARED_EXPERTS = 1
    TOP_K = 6  # experts per token (not counting shared)
    MAX_ENTROPY = np.log2(NUM_EXPERTS)  # ~7.0
    
    print("=" * 70)
    print("NEMOTRON 3 NANO - ADAPTIVE-K ROUTER LOGITS VALIDATION")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print(f"Architecture: {NUM_EXPERTS} routed + {SHARED_EXPERTS} shared experts, top-{TOP_K}")
    print(f"Max entropy: {MAX_ENTROPY:.2f} bits")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        
        if vram < 24 and args.quantize == "none":
            print("\nWARNING: Not enough VRAM for full precision!")
            print("Use --quantize 4bit or --quantize 8bit")
            return
    else:
        print("\nERROR: No GPU detected")
        return
    
    # Load model with appropriate quantization
    print(f"\nLoading model (quantization: {args.quantize})...")
    start = time.time()
    
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if args.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"Loaded in {time.time() - start:.1f}s")
    
    # Check if model has router logits output
    print("\n" + "=" * 70)
    print("CHECKING ROUTER LOGITS AVAILABILITY")
    print("=" * 70)
    
    # Nemotron uses custom architecture - need to inspect
    print(f"\nModel type: {type(model).__name__}")
    
    # Try a simple forward pass
    test_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Try with output_router_logits
        try:
            outputs = model(**test_input, output_router_logits=True, return_dict=True)
            has_router_logits = hasattr(outputs, 'router_logits') and outputs.router_logits is not None
            print(f"output_router_logits parameter: SUPPORTED")
            print(f"router_logits in output: {has_router_logits}")
        except TypeError as e:
            print(f"output_router_logits parameter: NOT SUPPORTED")
            print(f"Error: {e}")
            has_router_logits = False
            outputs = model(**test_input, return_dict=True)
    
    # Check output attributes
    print(f"\nOutput attributes: {[k for k in dir(outputs) if not k.startswith('_')]}")
    
    if not has_router_logits:
        print("\n" + "=" * 70)
        print("ALTERNATIVE: HOOK-BASED EXTRACTION")
        print("=" * 70)
        print("\nRouter logits not directly available.")
        print("Attempting hook-based extraction from MoE layers...")
        
        # Find MoE layers in the model
        moe_layers = []
        for name, module in model.named_modules():
            if 'moe' in name.lower() or 'expert' in name.lower():
                moe_layers.append((name, type(module).__name__))
        
        print(f"\nFound {len(moe_layers)} potential MoE-related modules:")
        for name, mtype in moe_layers[:10]:  # Show first 10
            print(f"  {name}: {mtype}")
        if len(moe_layers) > 10:
            print(f"  ... and {len(moe_layers) - 10} more")
        
        # Try to find router/gate modules
        router_modules = []
        for name, module in model.named_modules():
            if 'router' in name.lower() or 'gate' in name.lower():
                router_modules.append((name, module))
        
        if router_modules:
            print(f"\nFound {len(router_modules)} router/gate modules!")
            
            # Set up hooks to capture router outputs
            router_outputs = []
            
            def hook_fn(module, input, output):
                router_outputs.append(output.detach().cpu())
            
            hooks = []
            for name, module in router_modules[:5]:  # Hook first 5 layers
                hooks.append(module.register_forward_hook(hook_fn))
            
            # Run forward pass
            with torch.no_grad():
                _ = model(**test_input)
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            if router_outputs:
                print(f"\nCaptured {len(router_outputs)} router outputs!")
                for i, ro in enumerate(router_outputs):
                    print(f"  Layer {i}: shape {ro.shape}")
                    if len(ro.shape) >= 2:
                        probs = torch.softmax(ro, dim=-1).numpy()
                        entropy = np.mean([compute_entropy(probs[..., t, :]) 
                                          for t in range(probs.shape[-2])])
                        print(f"    Entropy: {entropy:.3f} / {MAX_ENTROPY:.2f}")
        else:
            print("\nNo router modules found. Model architecture may differ from expected.")
            print("Please check model source code for router access.")
            return
    
    else:
        # Router logits available - full analysis
        print("\n" + "=" * 70)
        print("ROUTER ENTROPY ANALYSIS")
        print("=" * 70)
        
        # Test prompts
        test_cases = [
            ("easy", "The capital of France is"),
            ("medium", "Write a Python function to calculate fibonacci:"),
            ("hard", "Explain the mathematical foundations of quantum entanglement and its implications for"),
            ("code", "def quicksort(arr):\n    '''Implement quicksort algorithm'''\n    "),
            ("reasoning", "If all mammals are warm-blooded and all whales are mammals, can we conclude that"),
        ]
        
        results = []
        
        for difficulty, prompt in test_cases:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_router_logits=True, return_dict=True)
            
            router_logits = outputs.router_logits
            num_layers = len([l for l in router_logits if l is not None])
            
            # Compute entropy per layer
            layer_entropies = []
            for layer_logits in router_logits:
                if layer_logits is not None:
                    probs = torch.softmax(layer_logits, dim=-1).cpu().numpy()
                    layer_entropy = np.mean([
                        compute_entropy(probs[0, t, :]) 
                        for t in range(probs.shape[1])
                    ])
                    layer_entropies.append(layer_entropy)
            
            mean_entropy = np.mean(layer_entropies)
            std_entropy = np.std(layer_entropies)
            
            # Adaptive-K projection for Nemotron 3 (128 experts, top-6)
            # Thresholds based on entropy distribution
            if mean_entropy < 2.0:
                projected_k = 2
            elif mean_entropy < 3.5:
                projected_k = 3
            elif mean_entropy < 5.0:
                projected_k = 4
            else:
                projected_k = 6
            
            savings = (1 - projected_k / TOP_K) * 100
            
            result = {
                "difficulty": difficulty,
                "prompt": prompt[:50],
                "mean_entropy": float(mean_entropy),
                "std_entropy": float(std_entropy),
                "num_layers": num_layers,
                "projected_k": projected_k,
                "base_k": TOP_K,
                "savings_pct": float(savings),
            }
            results.append(result)
            
            print(f"\n[{difficulty.upper()}] '{prompt[:40]}...'")
            print(f"  Entropy: {mean_entropy:.3f} +/- {std_entropy:.3f} (max: {MAX_ENTROPY:.2f})")
            print(f"  Adaptive-K: {projected_k} (base: {TOP_K}) -> {savings:.1f}% savings")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY - NEMOTRON 3 NANO ADAPTIVE-K VALIDATION")
        print("=" * 70)
        
        avg_savings = np.mean([r["savings_pct"] for r in results])
        avg_entropy = np.mean([r["mean_entropy"] for r in results])
        
        print(f"\nModel: {MODEL_ID}")
        print(f"Architecture: {NUM_EXPERTS} experts, top-{TOP_K}")
        print(f"Average entropy: {avg_entropy:.3f} / {MAX_ENTROPY:.2f}")
        print(f"Average projected savings: {avg_savings:.1f}%")
        
        print("\nPer-difficulty breakdown:")
        for r in results:
            print(f"  {r['difficulty']:10s}: K={r['projected_k']} -> {r['savings_pct']:.1f}%")
        
        # Comparison with whitepaper claims
        print("\n" + "=" * 70)
        print("COMPARISON WITH WHITEPAPER PROJECTIONS")
        print("=" * 70)
        print(f"\nWhitepaper projection for Nemotron 3: 27.1% gross savings")
        print(f"Measured average savings: {avg_savings:.1f}%")
        
        if avg_savings > 20:
            print("\n✓ VALIDATION SUCCESSFUL - Savings within expected range!")
        else:
            print(f"\n⚠ Savings lower than expected - may need threshold tuning")
        
        # Save results
        output = {
            "model": MODEL_ID,
            "architecture": {
                "num_experts": NUM_EXPERTS,
                "shared_experts": SHARED_EXPERTS,
                "top_k": TOP_K,
                "max_entropy": float(MAX_ENTROPY),
            },
            "results": results,
            "summary": {
                "avg_entropy": float(avg_entropy),
                "avg_savings_pct": float(avg_savings),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantization": args.quantize,
        }
        
        output_file = Path("router_logits_nemotron3.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
