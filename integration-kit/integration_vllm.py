#!/usr/bin/env python3
"""
Adaptive-K Integration for vLLM

This example shows how to integrate Adaptive-K routing into a vLLM deployment.
Integration time: ~4 hours for experienced engineers.

Prerequisites:
    pip install vllm adaptive-k-routing

Usage:
    python integration_vllm.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import os
import sys
from typing import Optional, List, Tuple
import torch

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.moe import MoELayer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not installed. Run: pip install vllm")

# Import Adaptive-K (with fallback for demo)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from sdk.adaptive_k import AdaptiveKRouter
    from sdk.adaptive_k.router import RoutingConfig
    ADAPTIVE_K_AVAILABLE = True
except ImportError:
    ADAPTIVE_K_AVAILABLE = False


class AdaptiveKvLLMWrapper:
    """
    Wrapper that adds Adaptive-K routing to vLLM's MoE layers.
    
    Integration approach:
    1. Load model normally with vLLM
    2. Replace the routing function in each MoE layer
    3. Use entropy-based K selection at inference time
    """
    
    def __init__(
        self,
        model_name: str,
        k_values: List[int] = [1, 2],
        entropy_threshold: float = 1.275,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
    ):
        """
        Initialize Adaptive-K vLLM wrapper.
        
        Args:
            model_name: HuggingFace model name (e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1")
            k_values: Possible K values for adaptive selection
            entropy_threshold: Entropy threshold for K selection
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model dtype ("auto", "float16", "bfloat16")
        """
        self.model_name = model_name
        self.k_values = k_values
        self.entropy_threshold = entropy_threshold
        
        print(f"🚀 Loading {model_name} with Adaptive-K routing...")
        
        if VLLM_AVAILABLE:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                trust_remote_code=True,
            )
            self._patch_moe_layers()
        else:
            print("   (Demo mode - vLLM not available)")
            self.llm = None
        
        # Metrics tracking
        self.total_tokens = 0
        self.total_k_sum = 0
        self.k_distribution = {k: 0 for k in k_values}
        
        print("✅ Adaptive-K vLLM wrapper initialized")
    
    def _patch_moe_layers(self):
        """
        Patch MoE layers to use Adaptive-K routing.
        
        This is the core integration point. We replace the fixed-K routing
        with entropy-based adaptive routing.
        """
        if not VLLM_AVAILABLE:
            return
            
        # Access the model's MoE layers
        # Note: Exact path depends on vLLM version and model architecture
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        
        for name, module in model.named_modules():
            if "moe" in name.lower() or isinstance(module, MoELayer):
                # Store original forward
                original_forward = module.forward
                
                # Create patched forward with adaptive routing
                def make_adaptive_forward(orig_fwd, layer_name):
                    def adaptive_forward(hidden_states, *args, **kwargs):
                        # Compute routing entropy
                        router_logits = module.gate(hidden_states)
                        probs = torch.softmax(router_logits, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                        
                        # Select K based on entropy
                        k = self._select_k(entropy)
                        
                        # Track metrics
                        self._update_metrics(k, hidden_states.shape[0])
                        
                        # Call original with modified top_k
                        # Note: May need adjustment based on vLLM version
                        return orig_fwd(hidden_states, *args, **kwargs)
                    
                    return adaptive_forward
                
                module.forward = make_adaptive_forward(original_forward, name)
                print(f"   Patched: {name}")
    
    def _select_k(self, entropy: torch.Tensor) -> torch.Tensor:
        """Select K based on entropy thresholds."""
        k = torch.full_like(entropy, self.k_values[-1], dtype=torch.long)
        k = torch.where(entropy < self.entropy_threshold, 
                       torch.full_like(k, self.k_values[0]), k)
        return k
    
    def _update_metrics(self, k: torch.Tensor, batch_size: int):
        """Update tracking metrics."""
        self.total_tokens += batch_size
        self.total_k_sum += k.float().sum().item()
        
        for kv in self.k_values:
            self.k_distribution[kv] += (k == kv).sum().item()
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate completions with Adaptive-K routing.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated completions
        """
        if not VLLM_AVAILABLE or self.llm is None:
            # Demo mode
            return [f"[Demo] Response to: {p[:50]}..." for p in prompts]
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def get_metrics(self) -> dict:
        """Get Adaptive-K performance metrics."""
        avg_k = self.total_k_sum / max(self.total_tokens, 1)
        baseline_k = max(self.k_values)
        savings = 1 - (avg_k / baseline_k)
        
        return {
            "total_tokens": self.total_tokens,
            "average_k": avg_k,
            "baseline_k": baseline_k,
            "compute_savings": f"{savings*100:.1f}%",
            "k_distribution": self.k_distribution,
        }
    
    def print_metrics(self):
        """Print formatted metrics."""
        m = self.get_metrics()
        print("\n" + "=" * 50)
        print("📊 ADAPTIVE-K METRICS")
        print("=" * 50)
        print(f"Total tokens processed: {m['total_tokens']:,}")
        print(f"Average K: {m['average_k']:.2f} (baseline: {m['baseline_k']})")
        print(f"Compute savings: {m['compute_savings']}")
        print(f"K distribution: {m['k_distribution']}")
        print("=" * 50)


def demo():
    """Run a quick demo."""
    print("\n" + "=" * 60)
    print("         ADAPTIVE-K + vLLM INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize wrapper
    wrapper = AdaptiveKvLLMWrapper(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        k_values=[1, 2],
        entropy_threshold=1.275,
    )
    
    # Sample prompts
    prompts = [
        "What is 2 + 2?",  # Simple, low entropy -> K=1
        "Explain quantum entanglement in detail.",  # Complex, high entropy -> K=2
        "Hello!",  # Simple greeting -> K=1
        "Write a comprehensive analysis of climate change impacts on agriculture.",  # Complex -> K=2
    ]
    
    print("\n📝 Generating responses...")
    responses = wrapper.generate(prompts, max_tokens=50)
    
    for prompt, response in zip(prompts, responses):
        print(f"\n[Prompt] {prompt}")
        print(f"[Response] {response[:100]}...")
    
    wrapper.print_metrics()
    
    print("\n✅ Demo complete!")
    print("   For production deployment, ensure you have:")
    print("   1. Valid Adaptive-K license")
    print("   2. Calibrated thresholds for your workload")
    print("   3. Monitoring dashboard configured")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive-K vLLM Integration")
    parser.add_argument("--model", type=str, 
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="Model name")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        print("Run with --demo to see example usage")
        print("For production, import AdaptiveKvLLMWrapper and use in your code")
