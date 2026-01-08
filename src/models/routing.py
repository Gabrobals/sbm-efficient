"""
Routing mechanisms for SBM-Efficient.

Provides:
- RandomRouting: Random expert selection (baseline)
- StaticTopK: Fixed expert selection (baseline)
- SBMRouting: Learnable routing with measurement and decoherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RandomRouting(nn.Module):
    """
    Random routing baseline.
    
    Selects K experts randomly for each sample.
    Used to verify that learnable routing provides actual benefit.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Random expert selection.
        
        Args:
            x: Input features (batch_size, feature_dim)
            tau: Temperature (ignored for random routing)
            
        Returns:
            (indices, weights, entropy) tuple
            - indices: Selected expert indices (batch_size, K)
            - weights: Uniform weights (batch_size, K)
            - entropy: Entropy of routing distribution (scalar)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Random selection for each sample
        indices = torch.stack([
            torch.randperm(self.num_experts, device=device)[:self.top_k]
            for _ in range(batch_size)
        ])
        
        # Uniform weights
        weights = torch.ones(batch_size, self.top_k, device=device) / self.top_k
        
        # Maximum entropy (uniform distribution over N experts)
        entropy = math.log(self.num_experts)
        
        return indices, weights, entropy


class StaticTopK(nn.Module):
    """
    Static Top-K routing baseline.
    
    Always selects the same K experts (determined at initialization).
    Used to verify that dynamic routing provides benefit.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        seed: int = 42
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Fixed expert selection (determined once)
        torch.manual_seed(seed)
        self.register_buffer(
            'fixed_indices',
            torch.randperm(num_experts)[:top_k]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Static expert selection.
        
        Args:
            x: Input features (batch_size, feature_dim)
            tau: Temperature (ignored for static routing)
            
        Returns:
            (indices, weights, entropy) tuple
        """
        batch_size = x.size(0)
        device = x.device
        
        # Same indices for all samples
        indices = self.fixed_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Uniform weights
        weights = torch.ones(batch_size, self.top_k, device=device) / self.top_k
        
        # Zero entropy (deterministic)
        entropy = 0.0
        
        return indices, weights, entropy


class SBMRouting(nn.Module):
    """
    SBM (Superposed Bit Model) routing with measurement and decoherence.
    
    This is the core innovation: routing based on quantum-inspired
    measurement with controllable decoherence (temperature schedule).
    
    Key concepts:
    - Router network outputs "measurement scores" for each expert
    - Temperature tau controls sharpness of selection
    - High tau: exploration (soft selection)
    - Low tau: exploitation (hard selection)
    - Entropy regularization encourages/discourages spread
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_experts: int,
        top_k: int
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network: maps features to expert scores
        self.router = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_experts)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Learnable routing with decoherence.
        
        Args:
            x: Input features (batch_size, feature_dim)
            tau: Temperature for softmax (decoherence parameter)
            
        Returns:
            (indices, weights, entropy) tuple
            - indices: Selected expert indices (batch_size, K)
            - weights: Softmax weights for selected experts (batch_size, K)
            - entropy: Mean entropy of routing distribution
        """
        batch_size = x.size(0)
        
        # Get routing scores
        scores = self.router(x)  # (batch_size, num_experts)
        
        # Apply temperature-scaled softmax
        probs = F.softmax(scores / tau, dim=1)  # (batch_size, num_experts)
        
        # Calculate entropy for regularization
        # H(p) = -sum(p * log(p + eps))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        # Top-K selection
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=1)
        
        # Renormalize weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
        return topk_indices, topk_weights, entropy.item()
    
    def get_full_distribution(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> torch.Tensor:
        """Get full probability distribution over experts (for analysis)."""
        scores = self.router(x)
        return F.softmax(scores / tau, dim=1)


class FullRouting(nn.Module):
    """
    Full routing baseline (K=N).
    
    Selects ALL experts with uniform weights.
    This is the "full compute" baseline - same architecture as sparse
    variants but with no sparsity. Used for fair comparison.
    """
    
    def __init__(
        self,
        num_experts: int
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = num_experts  # K = N (all experts)
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Full expert selection (all N experts).
        
        Args:
            x: Input features (batch_size, feature_dim)
            tau: Temperature (ignored for full routing)
            
        Returns:
            (indices, weights, entropy) tuple
        """
        batch_size = x.size(0)
        device = x.device
        
        # All experts selected
        indices = torch.arange(self.num_experts, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Uniform weights
        weights = torch.ones(batch_size, self.num_experts, device=device) / self.num_experts
        
        # Maximum entropy (uniform over all N)
        entropy = math.log(self.num_experts)
        
        return indices, weights, entropy


def create_router(
    routing_type: str,
    feature_dim: int,
    num_experts: int,
    top_k: int,
    seed: int = 42
) -> nn.Module:
    """
    Factory function to create routing module.
    
    Args:
        routing_type: "full", "random", "static", or "sbm"
        feature_dim: Dimension of input features (for SBM)
        num_experts: Number of experts N
        top_k: Number of experts to select K
        seed: Random seed for static routing
        
    Returns:
        Routing module
    """
    if routing_type == "full" or routing_type == "full_routing":
        return FullRouting(num_experts)
    
    elif routing_type == "random" or routing_type == "random_routing":
        return RandomRouting(num_experts, top_k)
    
    elif routing_type == "static" or routing_type == "static_topk":
        return StaticTopK(num_experts, top_k, seed)
    
    elif routing_type == "sbm":
        return SBMRouting(feature_dim, num_experts, top_k)
    
    else:
        raise ValueError(f"Unknown routing type: {routing_type}")


if __name__ == "__main__":
    batch_size = 4
    feature_dim = 64
    num_experts = 8
    top_k = 2
    
    x = torch.randn(batch_size, feature_dim)
    
    print("Testing RandomRouting...")
    router = RandomRouting(num_experts, top_k)
    indices, weights, entropy = router(x)
    print(f"  Indices: {indices}")
    print(f"  Weights: {weights}")
    print(f"  Entropy: {entropy:.4f}")
    
    print("\nTesting StaticTopK...")
    router = StaticTopK(num_experts, top_k)
    indices, weights, entropy = router(x)
    print(f"  Indices: {indices}")
    print(f"  Weights: {weights}")
    print(f"  Entropy: {entropy:.4f}")
    
    print("\nTesting SBMRouting...")
    router = SBMRouting(feature_dim, num_experts, top_k)
    
    # High temperature (exploration)
    indices, weights, entropy = router(x, tau=2.0)
    print(f"  tau=2.0 - Indices: {indices[0]}, Entropy: {entropy:.4f}")
    
    # Low temperature (exploitation)
    indices, weights, entropy = router(x, tau=0.5)
    print(f"  tau=0.5 - Indices: {indices[0]}, Entropy: {entropy:.4f}")
    
    print("\n[OK] All routing modules working!")
