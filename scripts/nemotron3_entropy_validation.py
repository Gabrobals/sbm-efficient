#!/usr/bin/env python3
"""
Nemotron 3 Nano Router Entropy Validation Script

This script extracts pre-top-k router logits from NVIDIA Nemotron 3 Nano
and computes entropy to validate Adaptive-K routing potential.

Requirements:
    pip install transformers accelerate mamba-ssm causal-conv1d

Hardware:
    Minimum 2x A100 40GB (80GB total for bf16 weights)

Usage:
    python nemotron3_entropy_validation.py

Author: Gabriele Balsamo
Date: January 19, 2026
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime


def compute_entropy(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute Shannon entropy: H = -sum(p * log(p))"""
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


def compute_adaptive_k_savings(entropy: float, max_entropy: float = 7.0) -> dict:
    """
    Compute Adaptive-K savings based on entropy thresholds.
    
    For Nemotron 3 Nano (128 experts, top-6 baseline):
    - K=2 if H < 4.5 (very confident)
    - K=4 if 4.5 <= H < 5.5 (moderate)
    - K=6 if H >= 5.5 (uncertain, use baseline)
    """
    baseline_k = 6
    
    if entropy < 4.5:
        adaptive_k = 2
    elif entropy < 5.5:
        adaptive_k = 4
    else:
        adaptive_k = 6
    
    savings = (baseline_k - adaptive_k) / baseline_k * 100
    
    return {
        "baseline_k": baseline_k,
        "adaptive_k": adaptive_k,
        "savings_percent": savings,
        "entropy": entropy,
        "entropy_pct_of_max": entropy / max_entropy * 100
    }


def main():
    print("=" * 60)
    print("Nemotron 3 Nano - Adaptive-K Entropy Validation")
    print("=" * 60)
    
    # Model configuration
    model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    num_experts = 128
    max_entropy = torch.log2(torch.tensor(num_experts)).item()  # 7.0 bits
    
    print(f"\nModel: {model_name}")
    print(f"Experts: {num_experts}")
    print(f"Max Entropy: {max_entropy:.2f} bits")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    print("Model loaded!")
    
    # Find router modules
    router_modules = []
    for name, module in model.named_modules():
        if 'mixer.gate' in name and hasattr(module, 'weight'):
            router_modules.append((name, module))
    print(f"Found {len(router_modules)} router modules")
    
    # Test prompts
    test_cases = [
        ("easy", "The capital of France is"),
        ("code", "def fibonacci(n):"),
        ("hard", "Explain the concept of quantum entanglement in terms of"),
    ]
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "architecture": {
            "num_experts": num_experts,
            "top_k": 6,
            "shared_experts": 1,
            "router_modules": len(router_modules),
        },
        "max_entropy": max_entropy,
        "test_cases": [],
    }
    
    # Storage for captured logits
    all_entropies = []
    captured_logits = {}
    
    for case_name, prompt in test_cases:
        print(f"\n{'='*40}")
        print(f"Test case: {case_name}")
        print(f"Prompt: {prompt[:50]}...")
        
        # Clear captured logits
        captured_logits.clear()
        
        # Register hooks
        hooks = []
        for name, module in router_modules:
            def make_hook(module_name, mod):
                def hook_fn(module, input, output):
                    hidden = input[0]  # [batch, seq, hidden_dim]
                    # Compute full router logits
                    logits = hidden.float() @ mod.weight.float().T  # [batch, seq, 128]
                    captured_logits[module_name] = logits.detach()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name, module)))
        
        # Forward pass
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Compute entropy across all layers
        layer_entropies = []
        for name, logits in captured_logits.items():
            probs = F.softmax(logits, dim=-1)
            entropy = compute_entropy(probs)
            mean_entropy = entropy.mean().item()
            layer_entropies.append(mean_entropy)
        
        avg_entropy = sum(layer_entropies) / len(layer_entropies)
        all_entropies.append(avg_entropy)
        
        # Compute savings
        savings_info = compute_adaptive_k_savings(avg_entropy, max_entropy)
        
        case_result = {
            "name": case_name,
            "prompt": prompt,
            "avg_entropy": avg_entropy,
            "entropy_pct_of_max": avg_entropy / max_entropy * 100,
            "adaptive_k": savings_info["adaptive_k"],
            "savings_percent": savings_info["savings_percent"],
            "layers_analyzed": len(layer_entropies),
        }
        results["test_cases"].append(case_result)
        
        print(f"Average Entropy: {avg_entropy:.3f} bits ({avg_entropy/max_entropy*100:.1f}% of max)")
        print(f"Adaptive K: {savings_info['adaptive_k']} (baseline: 6)")
        print(f"Savings: {savings_info['savings_percent']:.1f}%")
    
    # Overall statistics
    overall_entropy = sum(all_entropies) / len(all_entropies)
    overall_savings = compute_adaptive_k_savings(overall_entropy, max_entropy)
    
    results["summary"] = {
        "avg_entropy": overall_entropy,
        "avg_entropy_pct_of_max": overall_entropy / max_entropy * 100,
        "avg_adaptive_k": overall_savings["adaptive_k"],
        "avg_savings_percent": overall_savings["savings_percent"],
    }
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Average Entropy: {overall_entropy:.3f} bits ({overall_entropy/max_entropy*100:.1f}% of max)")
    print(f"Average Savings: {overall_savings['savings_percent']:.1f}%")
    
    # Save results
    output_file = "nemotron3_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
