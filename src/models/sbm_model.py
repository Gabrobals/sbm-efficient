"""
Complete SBM-Efficient model with routing.

Combines:
- Feature extractor (task-specific)
- Router (random/static/sbm)
- Expert pool (sparse execution)
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.models.experts import FeatureExtractor, ExpertPool
from src.models.routing import create_router
from src.routing import AdaptiveKPolicy, AdaptiveKConfig
from src.models.routing import SBMRouting


class SBMModel(nn.Module):
    """
    SBM-Efficient model with sparse expert routing.
    
    Architecture:
        input -> FeatureExtractor -> Router -> ExpertPool (sparse) -> Classifier
    
    Supports multiple routing modes:
    - baseline: All experts active (K=N)
    - random_routing: K experts selected randomly
    - static_topk: Same K experts always
    - sbm: Learnable routing with decoherence
    """
    
    def __init__(
        self,
        task: str,
        num_experts: int,
        top_k: int,
        routing_type: str = "sbm",
        feature_dim: int = 64,
        expert_hidden_dim: int = 32,
        num_classes: int = 10,
        seed: int = 42
    ):
        super().__init__()
        
        self.task = task
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_type = routing_type
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Feature extractor (task-specific)
        self.feature_extractor = FeatureExtractor(task, output_dim=feature_dim)
        
        # Router
        self.router = create_router(
            routing_type=routing_type,
            feature_dim=feature_dim,
            num_experts=num_experts,
            top_k=top_k,
            seed=seed
        )
        
        # Expert pool
        self.expert_pool = ExpertPool(
            num_experts=num_experts,
            input_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=feature_dim  # Experts output back to feature space
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Tracking metrics
        self.last_entropy = 0.0
        self.last_flops = 0
        self.last_active_count = 0.0
    
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with sparse routing.
        
        Args:
            x: Input tensor (batch_size, ...)
            tau: Temperature for routing (only used by SBM)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Extract features
        features = self.feature_extractor(x)  # (batch_size, feature_dim)
        
        # Route to experts
        indices, weights, entropy = self.router(features, tau)
        
        # Execute selected experts (sparse!)
        # NOTE: ExpertPool executes ONLY the selected experts - no full compute + masking
        expert_output, flops = self.expert_pool(features, indices, weights)
        
        # Residual connection
        combined = features + expert_output
        
        # Classify
        logits = self.classifier(combined)
        
        # Store metrics for logging
        # For full routing, actual K = N (from router.top_k)
        actual_k = indices.size(1)  # Number of experts actually selected
        self.last_entropy = entropy
        self.last_flops = flops
        self.last_active_count = float(actual_k)
        
        return logits
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics from last forward pass."""
        return {
            "entropy": self.last_entropy,
            "flops_executed": self.last_flops,
            "active_modules": self.last_active_count
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_sbm_model(
    task: str,
    config: Dict[str, Any]
) -> SBMModel:
    """
    Factory function to create SBM model from config.
    
    Args:
        task: Task name
        config: Full experiment config
        
    Returns:
        Initialized SBMModel
    """
    sbm_config = config.get("sbm", {})
    run_config = config.get("run", {})
    
    num_experts = sbm_config.get("experts_num", 16)
    top_k = sbm_config.get("experts_top_k", 2)
    routing_type = run_config.get("model", "sbm")
    seed = run_config.get("seed", 42)
    
    # Determine number of classes
    num_classes = {
        "xor": 2,
        "mnist": 10,
        "fashion_mnist": 10,
        "cifar10": 10
    }.get(task, 10)
    
    # Feature and hidden dims scale with experts for fair comparison
    feature_dim = 64
    expert_hidden_dim = 32
    
    return SBMModel(
        task=task,
        num_experts=num_experts,
        top_k=top_k,
        routing_type=routing_type,
        feature_dim=feature_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_classes=num_classes,
        seed=seed
    )


class SBMAdaptiveKModel(nn.Module):
    """SBM variant with Adaptive-K routing (threshold policy v1)."""

    def __init__(
        self,
        task: str,
        num_experts: int,
        routing_type: str,
        k_values: list[int],
        h_thresholds: list[float],
        feature_dim: int,
        expert_hidden_dim: int,
        num_classes: int,
        seed: int = 42
    ):
        super().__init__()

        self.task = task
        self.num_experts = num_experts
        self.routing_type = routing_type
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k_values = k_values

        # Feature extractor
        self.feature_extractor = FeatureExtractor(task, output_dim=feature_dim)

        # Router network (use learnable SBM router to get logits)
        # Keep top_k as max_k to size buffers, though K is dynamic per sample
        self.max_k = max(k_values)
        self.router = SBMRouting(feature_dim=feature_dim, num_experts=num_experts, top_k=self.max_k)

        # For compatibility with logging/metrics paths expecting top_k
        self.top_k = self.max_k

        # Adaptive-K policy
        self.policy = AdaptiveKPolicy(AdaptiveKConfig(k_values=k_values, h_thresholds=h_thresholds))

        # Expert pool
        self.expert_pool = ExpertPool(
            num_experts=num_experts,
            input_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=feature_dim
        )

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Tracking metrics
        self.last_entropy = 0.0
        self.last_flops = 0
        self.last_active_count = 0.0
        self.last_k_mean = 0.0
        self.last_k_std = 0.0
        self.last_k_histogram: Dict[str, int] = {}
        self.last_k_histogram_pct: Dict[str, float] = {}

    def reset_k_stats(self):
        self.policy.reset_stats()

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        # Extract features
        features = self.feature_extractor(x)

        # Router logits and probabilities
        logits = self.router.router(features)
        probs = F.softmax(logits / tau, dim=1)

        # Entropy per sample
        entropy_per_sample = -(probs * (probs + 1e-9).log()).sum(dim=1)

        # Dynamic K per sample
        K = self.policy.k_from_entropy(entropy_per_sample)

        # Get indices and weights (per-sample lists)
        idx_list, weights_list = self.policy.topk_indices_and_weights(logits, K, tau=tau)

        # Execute selected experts only
        expert_output, flops = self.expert_pool.execute_sparse(features, idx_list, weights_list)

        # Residual combine
        combined = features + expert_output
        logits_out = self.classifier(combined)

        # Stats
        stats = self.policy.get_stats()
        self.last_entropy = float(entropy_per_sample.mean().item())
        self.last_flops = flops
        self.last_active_count = float(K.float().mean().item())
        self.last_k_mean = stats.get("k_mean", 0.0)
        self.last_k_std = stats.get("k_std", 0.0)
        self.last_k_histogram = stats.get("k_histogram", {})
        self.last_k_histogram_pct = stats.get("k_histogram_pct", {})

        return logits_out

    def get_routing_stats(self) -> Dict[str, Any]:
        return {
            "entropy": self.last_entropy,
            "flops_executed": self.last_flops,
            "active_modules": self.last_active_count,
            "k_mean": self.last_k_mean,
            "k_std": self.last_k_std,
            "k_histogram": self.last_k_histogram,
            "k_histogram_pct": self.last_k_histogram_pct
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_sbm_adaptive_k_model(task: str, config: Dict[str, Any]) -> SBMAdaptiveKModel:
    """Factory for Adaptive-K model."""
    sbm_config = config.get("sbm", {})
    adaptive_cfg = config.get("adaptive_k", {})
    run_config = config.get("run", {})

    num_experts = sbm_config.get("experts_num", 16)
    routing_type = run_config.get("model", "sbm_adaptive_k")
    seed = run_config.get("seed", 42)

    k_values = adaptive_cfg.get("k_values", [1, 2, 4])
    h_thresholds = adaptive_cfg.get("h_thresholds", [0.6, 1.2])

    num_classes = {
        "xor": 2,
        "mnist": 10,
        "fashion_mnist": 10,
        "cifar10": 10
    }.get(task, 10)

    feature_dim = 64
    expert_hidden_dim = 32

    return SBMAdaptiveKModel(
        task=task,
        num_experts=num_experts,
        routing_type=routing_type,
        k_values=k_values,
        h_thresholds=h_thresholds,
        feature_dim=feature_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_classes=num_classes,
        seed=seed
    )


if __name__ == "__main__":
    print("Testing SBMModel (XOR, random routing)...")
    model = SBMModel(
        task="xor",
        num_experts=8,
        top_k=2,
        routing_type="random_routing",
        num_classes=2
    )
    x = torch.randn(4, 2)
    out = model(x)
    stats = model.get_routing_stats()
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Stats: {stats}")
    
    print("\nTesting SBMModel (MNIST, sbm routing)...")
    model = SBMModel(
        task="mnist",
        num_experts=16,
        top_k=2,
        routing_type="sbm",
        num_classes=10
    )
    x = torch.randn(4, 1, 28, 28)
    
    # High tau (exploration)
    out = model(x, tau=2.0)
    stats = model.get_routing_stats()
    print(f"  tau=2.0 - Entropy: {stats['entropy']:.4f}, FLOPs: {stats['flops_executed']}")
    
    # Low tau (exploitation)
    out = model(x, tau=0.5)
    stats = model.get_routing_stats()
    print(f"  tau=0.5 - Entropy: {stats['entropy']:.4f}, FLOPs: {stats['flops_executed']}")
    
    print(f"\n  Parameters: {model.count_parameters()}")
    
    print("\n[OK] SBMModel working!")
