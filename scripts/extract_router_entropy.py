#!/usr/bin/env python3
"""
Extract Router Logits and Compute Entropy from MoE Models

This script demonstrates that Adaptive-K can be validated on any MoE model
using HuggingFace Transformers with `output_router_logits=True`.

Models supported:
- Mixtral-8x7B (8 experts, top-2)
- Qwen3-30B-A3B (128 experts, top-8) 
- Qwen3-235B-A22B (128 experts, top-8)
- Any HuggingFace MoE model with router logits output

Usage:
    # Small model (fits on single GPU)
    python scripts/extract_router_entropy.py --model Qwen/Qwen3-30B-A3B --quantize 4bit
    
    # Mixtral (requires ~90GB RAM or quantization)
    python scripts/extract_router_entropy.py --model mistralai/Mixtral-8x7B-v0.1 --quantize 4bit
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import json
import os
from datetime import datetime


@dataclass
class EntropyAnalysis:
    """Analysis results for a single input."""
    prompt: str
    tokens: int
    layers: int
    num_experts: int
    top_k: int
    entropy_per_layer: List[float]  # Mean entropy per layer
    entropy_mean: float  # Overall mean
    entropy_std: float  # Overall std
    entropy_min: float
    entropy_max: float
    # Per-token analysis
    tokens_low_entropy: int  # H < theta_1 (would use K_min)
    tokens_mid_entropy: int  # theta_1 <= H < theta_2 (would use K_mid)
    tokens_high_entropy: int  # H >= theta_2 (would use K_max)
    # Adaptive-K projection
    projected_avg_k: float
    projected_compute_saved: float  # Percentage


def compute_entropy(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute Shannon entropy: H = -sum(p * log(p))
    
    Args:
        probs: Probability distribution [batch, seq_len, num_experts]
        eps: Small value for numerical stability
    
    Returns:
        entropy: [batch, seq_len]
    """
    # Ensure probabilities (apply softmax if these are logits)
    if probs.min() < 0:
        probs = F.softmax(probs, dim=-1)
    
    # H = -sum(p * log(p))
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy


def analyze_router_logits(
    router_logits: Tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
    thresholds: List[float],
    k_values: List[int]
) -> Dict:
    """
    Analyze router logits from all layers.
    
    Args:
        router_logits: Tuple of tensors, one per layer, shape [batch, seq_len, num_experts]
        num_experts: Number of experts in the model
        top_k: Baseline top-K value
        thresholds: [theta_1, theta_2] for Adaptive-K
        k_values: [K_min, K_mid, K_max] for Adaptive-K
    
    Returns:
        Analysis dictionary
    """
    all_entropies = []
    entropy_per_layer = []
    
    for layer_idx, logits in enumerate(router_logits):
        # logits shape: [batch, seq_len, num_experts]
        probs = F.softmax(logits.float(), dim=-1)
        entropy = compute_entropy(probs)  # [batch, seq_len]
        
        layer_mean = entropy.mean().item()
        entropy_per_layer.append(layer_mean)
        all_entropies.append(entropy)
    
    # Stack all entropies: [num_layers, batch, seq_len]
    all_entropies = torch.stack(all_entropies)
    
    # Flatten for overall statistics
    flat_entropy = all_entropies.flatten()
    
    # Compute Adaptive-K projections
    theta_1, theta_2 = thresholds
    k_min, k_mid, k_max = k_values
    
    low_entropy_mask = flat_entropy < theta_1
    mid_entropy_mask = (flat_entropy >= theta_1) & (flat_entropy < theta_2)
    high_entropy_mask = flat_entropy >= theta_2
    
    tokens_low = low_entropy_mask.sum().item()
    tokens_mid = mid_entropy_mask.sum().item()
    tokens_high = high_entropy_mask.sum().item()
    total_tokens = flat_entropy.numel()
    
    # Project average K with Adaptive-K
    projected_avg_k = (
        (tokens_low * k_min + tokens_mid * k_mid + tokens_high * k_max) / total_tokens
    )
    
    # Compute projected savings
    baseline_compute = top_k * total_tokens
    adaptive_compute = tokens_low * k_min + tokens_mid * k_mid + tokens_high * k_max
    projected_savings = 1 - (adaptive_compute / baseline_compute)
    
    return {
        'entropy_per_layer': entropy_per_layer,
        'entropy_mean': flat_entropy.mean().item(),
        'entropy_std': flat_entropy.std().item(),
        'entropy_min': flat_entropy.min().item(),
        'entropy_max': flat_entropy.max().item(),
        'tokens_low_entropy': tokens_low,
        'tokens_mid_entropy': tokens_mid,
        'tokens_high_entropy': tokens_high,
        'total_tokens': total_tokens,
        'projected_avg_k': projected_avg_k,
        'projected_compute_saved': projected_savings * 100,
        'thresholds_used': thresholds,
        'k_values_used': k_values,
    }


def load_model(
    model_name: str,
    quantize: Optional[str] = None,
    device_map: str = "auto"
):
    """Load MoE model with optional quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model: {model_name}")
    
    # Configure quantization
    quant_config = None
    if quantize == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("Using 4-bit quantization")
    elif quantize == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with router logits output enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.float16 if not quantize else None,
        output_router_logits=True,  # KEY: Enable router logits output!
    )
    
    # Also set in config for generation
    model.config.output_router_logits = True
    
    return model, tokenizer


def extract_entropy_from_model(
    model,
    tokenizer,
    prompts: List[str],
    thresholds: List[float],
    k_values: List[int],
    max_new_tokens: int = 50
) -> List[Dict]:
    """
    Extract router entropy from model for given prompts.
    """
    results = []
    
    # Get model config
    config = model.config
    num_experts = getattr(config, 'num_local_experts', 
                         getattr(config, 'num_experts', 8))
    top_k = getattr(config, 'num_experts_per_tok', 
                   getattr(config, 'top_k', 2))
    
    print(f"\nModel config: {num_experts} experts, top-{top_k} routing")
    print(f"Thresholds: {thresholds}, K values: {k_values}")
    print("-" * 60)
    
    for prompt in prompts:
        print(f"\nProcessing: {prompt[:50]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Forward pass with router logits
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_router_logits=True,
                return_dict=True
            )
        
        # Extract router logits
        router_logits = outputs.router_logits
        
        if router_logits is None:
            print("  WARNING: router_logits is None - model may not support this")
            continue
        
        # Analyze
        analysis = analyze_router_logits(
            router_logits,
            num_experts=num_experts,
            top_k=top_k,
            thresholds=thresholds,
            k_values=k_values
        )
        
        analysis['prompt'] = prompt
        analysis['num_experts'] = num_experts
        analysis['baseline_top_k'] = top_k
        analysis['num_layers'] = len(router_logits)
        
        results.append(analysis)
        
        # Print summary
        print(f"  Entropy: mean={analysis['entropy_mean']:.3f}, "
              f"std={analysis['entropy_std']:.3f}, "
              f"range=[{analysis['entropy_min']:.3f}, {analysis['entropy_max']:.3f}]")
        print(f"  Token distribution: low={analysis['tokens_low_entropy']}, "
              f"mid={analysis['tokens_mid_entropy']}, high={analysis['tokens_high_entropy']}")
        print(f"  Projected avg K: {analysis['projected_avg_k']:.2f} "
              f"(baseline: {top_k})")
        print(f"  Projected compute savings: {analysis['projected_compute_saved']:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract router entropy from MoE models")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit", None],
        default=None,
        help="Quantization mode"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        default=[1.0, 1.5],
        help="Entropy thresholds [theta_1, theta_2]"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs=3,
        default=[1, 2, 4],
        help="K values [K_min, K_mid, K_max]"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/router_entropy_analysis.json",
        help="Output file"
    )
    
    args = parser.parse_args()
    
    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list.",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was chaos. Then came order.",
    ]
    
    # Load model
    model, tokenizer = load_model(args.model, args.quantize)
    
    # Extract entropy
    results = extract_entropy_from_model(
        model, tokenizer, prompts,
        thresholds=args.thresholds,
        k_values=args.k_values
    )
    
    # Aggregate results
    if results:
        avg_entropy = np.mean([r['entropy_mean'] for r in results])
        avg_savings = np.mean([r['projected_compute_saved'] for r in results])
        avg_k = np.mean([r['projected_avg_k'] for r in results])
        
        summary = {
            'model': args.model,
            'timestamp': datetime.now().isoformat(),
            'thresholds': args.thresholds,
            'k_values': args.k_values,
            'num_prompts': len(results),
            'average_entropy': avg_entropy,
            'average_projected_k': avg_k,
            'average_compute_savings': avg_savings,
            'per_prompt_results': results
        }
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Average entropy: {avg_entropy:.3f}")
        print(f"Average projected K: {avg_k:.2f}")
        print(f"Average compute savings: {avg_savings:.1f}%")
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
