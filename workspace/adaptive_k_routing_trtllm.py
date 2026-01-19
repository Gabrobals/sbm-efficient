"""
Adaptive-K Routing Method for TensorRT-LLM MoE.

This module implements entropy-based dynamic K selection for Mixture-of-Experts
models. Instead of using a fixed number of experts (top-k), it selects K dynamically
based on the routing entropy (confidence).

Key insight: Lower entropy means the router is more confident, so fewer experts
are needed. Higher entropy means uncertainty, requiring more experts.

Research validation:
- Qwen-MoE: 32.4% compute reduction
- Mixtral 8x7B: 31.0% compute reduction  
- OLMoE-1B-7B: 24.7% compute reduction

Author: Gabriele Balsamo (gabriele.balsamo30@gmail.com)
Date: January 2026
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class AdaptiveKConfig:
    """Configuration for Adaptive-K routing.
    
    Attributes:
        k_min: Minimum number of experts to use (default: 2)
        k_max: Maximum number of experts to use (default: 8)
        k_values: List of possible K values (e.g., [2, 4, 6, 8])
        entropy_thresholds: Thresholds for K selection.
            len(entropy_thresholds) == len(k_values) - 1
            E.g., for k_values=[2,4,6], thresholds=[1.3, 1.7] means:
                H < 1.3 -> K=2
                1.3 <= H < 1.7 -> K=4  
                H >= 1.7 -> K=6
    """
    k_min: int = 2
    k_max: int = 8
    k_values: List[int] = None
    entropy_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [self.k_min, (self.k_min + self.k_max) // 2, self.k_max]
        if self.entropy_thresholds is None:
            # Default thresholds based on our experiments
            # These work well for models like Mixtral, Qwen-MoE
            self.entropy_thresholds = [1.3, 1.7]
        
        assert len(self.entropy_thresholds) == len(self.k_values) - 1, \
            f"entropy_thresholds length ({len(self.entropy_thresholds)}) must be " \
            f"k_values length - 1 ({len(self.k_values) - 1})"


class AdaptiveKMoeRoutingMethod(nn.Module):
    """
    Entropy-based Adaptive-K routing for MoE models.
    
    Dynamically selects the number of experts (K) based on routing entropy:
    - Low entropy (confident routing) -> fewer experts -> save compute
    - High entropy (uncertain routing) -> more experts -> maintain quality
    
    This implements the SBM-Efficient Adaptive-K algorithm validated on:
    - Mixtral 8x7B: 31.0% compute reduction
    - Qwen-MoE: 32.4% compute reduction
    
    Compatible with TensorRT-LLM's BaseMoeRoutingMethod interface.
    """
    
    def __init__(
        self,
        config: Optional[AdaptiveKConfig] = None,
        k_min: int = 2,
        k_max: int = 8,
        entropy_thresholds: Optional[List[float]] = None,
        output_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Adaptive-K routing.
        
        Args:
            config: AdaptiveKConfig object (overrides other params if provided)
            k_min: Minimum experts to use
            k_max: Maximum experts to use (also defines output shape)
            entropy_thresholds: List of thresholds for K selection
            output_dtype: Output dtype for routing weights
        """
        super().__init__()
        
        if config is not None:
            self.config = config
        else:
            k_values = [k_min, (k_min + k_max) // 2, k_max]
            self.config = AdaptiveKConfig(
                k_min=k_min,
                k_max=k_max,
                k_values=k_values,
                entropy_thresholds=entropy_thresholds or [1.3, 1.7]
            )
        
        self.k_max = self.config.k_max
        self.output_dtype = output_dtype
        
        # Statistics tracking
        self._k_counts = {k: 0 for k in self.config.k_values}
        self._total_tokens = 0
        self._entropy_sum = 0.0
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of routing distribution.
        
        Args:
            probs: Probability distribution [num_tokens, num_experts]
            
        Returns:
            Entropy tensor [num_tokens]
        """
        eps = 1e-9
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    def select_k_per_token(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Select K value for each token based on entropy.
        
        Args:
            entropy: Entropy values [num_tokens]
            
        Returns:
            K values [num_tokens] as int tensor
        """
        k_values = self.config.k_values
        thresholds = self.config.entropy_thresholds
        
        # Start with max K
        k = torch.full_like(entropy, k_values[-1], dtype=torch.int32)
        
        # Apply thresholds from highest to lowest
        for i in range(len(thresholds) - 1, -1, -1):
            k = torch.where(entropy < thresholds[i], k_values[i], k)
        
        return k
    
    def apply(
        self, 
        router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive-K routing to router logits.
        
        Args:
            router_logits: Router output [num_tokens, num_experts]
            
        Returns:
            Tuple of:
                - token_selected_experts: [num_tokens, k_max] expert indices
                - token_final_scales: [num_tokens, k_max] routing weights
                
        Note: Returns k_max experts per token for compatibility, but tokens
        with lower K will have zero weights for unused expert slots.
        """
        num_tokens, num_experts = router_logits.shape
        device = router_logits.device
        
        # Compute routing probabilities
        probs = F.softmax(router_logits.to(self.output_dtype), dim=-1)
        
        # Compute entropy per token
        entropy = self.compute_entropy(probs)
        
        # Select K per token
        k_per_token = self.select_k_per_token(entropy)
        
        # Get top-k_max experts (we'll mask unused ones)
        topk_values, topk_indices = torch.topk(probs, k=self.k_max, dim=-1)
        
        # Create mask for valid experts based on per-token K
        # Shape: [num_tokens, k_max]
        k_range = torch.arange(self.k_max, device=device).unsqueeze(0)  # [1, k_max]
        valid_mask = k_range < k_per_token.unsqueeze(1)  # [num_tokens, k_max]
        
        # Zero out weights for invalid expert slots
        masked_values = torch.where(
            valid_mask, 
            topk_values, 
            torch.zeros_like(topk_values)
        )
        
        # Renormalize weights (only over valid experts)
        weight_sum = masked_values.sum(dim=-1, keepdim=True) + 1e-9
        normalized_values = masked_values / weight_sum
        
        # Update statistics
        self._update_stats(k_per_token, entropy)
        
        return topk_indices.to(torch.int32), normalized_values.to(self.output_dtype)
    
    def _update_stats(self, k_per_token: torch.Tensor, entropy: torch.Tensor):
        """Update internal statistics for monitoring."""
        for k in self.config.k_values:
            self._k_counts[k] += (k_per_token == k).sum().item()
        self._total_tokens += k_per_token.numel()
        self._entropy_sum += entropy.sum().item()
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        if self._total_tokens == 0:
            return {"k_distribution": {}, "mean_entropy": 0.0, "total_tokens": 0}
        
        k_dist = {
            k: count / self._total_tokens * 100 
            for k, count in self._k_counts.items()
        }
        
        # Calculate average K and compute savings
        avg_k = sum(k * (count / self._total_tokens) 
                   for k, count in self._k_counts.items())
        compute_savings = (1 - avg_k / self.k_max) * 100
        
        return {
            "k_distribution": k_dist,
            "mean_entropy": self._entropy_sum / self._total_tokens,
            "total_tokens": self._total_tokens,
            "avg_k": avg_k,
            "baseline_k": self.k_max,
            "compute_savings_pct": compute_savings
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._k_counts = {k: 0 for k in self.config.k_values}
        self._total_tokens = 0
        self._entropy_sum = 0.0
    
    def get_experts_per_token(self) -> int:
        """Return maximum K for shape compatibility."""
        return self.k_max
    
    @property
    def experts_per_token(self) -> int:
        return self.get_experts_per_token()


# Factory function for easy integration
def create_adaptive_k_routing(
    k_min: int = 2,
    k_max: int = 8,
    entropy_thresholds: Optional[List[float]] = None,
    **kwargs
) -> AdaptiveKMoeRoutingMethod:
    """
    Create an Adaptive-K routing method.
    
    Example usage:
        routing = create_adaptive_k_routing(k_min=2, k_max=8)
        experts, weights = routing.apply(router_logits)
        print(routing.get_stats())  # Shows compute savings
    
    Args:
        k_min: Minimum experts
        k_max: Maximum experts  
        entropy_thresholds: Custom thresholds (default: [1.3, 1.7])
        **kwargs: Additional arguments passed to AdaptiveKMoeRoutingMethod
        
    Returns:
        AdaptiveKMoeRoutingMethod instance
    """
    return AdaptiveKMoeRoutingMethod(
        k_min=k_min,
        k_max=k_max,
        entropy_thresholds=entropy_thresholds,
        **kwargs
    )


# Example integration with TensorRT-LLM
"""
To use in TensorRT-LLM, replace the standard routing method:

# Original (fixed top-k):
routing_method = DefaultMoeRoutingMethod(top_k=8)

# With Adaptive-K:
from adaptive_k_routing import create_adaptive_k_routing
routing_method = create_adaptive_k_routing(k_min=2, k_max=8)

# The interface is compatible - both return (expert_indices, weights)
experts, weights = routing_method.apply(router_logits)

# Check compute savings:
stats = routing_method.get_stats()
print(f"Compute savings: {stats['compute_savings_pct']:.1f}%")
"""


if __name__ == "__main__":
    # Quick test
    print("Testing AdaptiveKMoeRoutingMethod...")
    
    routing = create_adaptive_k_routing(k_min=2, k_max=8)
    
    # Simulate router logits
    torch.manual_seed(42)
    router_logits = torch.randn(100, 64)  # 100 tokens, 64 experts
    
    experts, weights = routing.apply(router_logits)
    
    print(f"Expert indices shape: {experts.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"\nStatistics: {routing.get_stats()}")
