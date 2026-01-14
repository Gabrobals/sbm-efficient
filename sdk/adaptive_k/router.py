"""
Core Adaptive-K Router implementation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RoutingConfig:
    """Configuration for Adaptive-K routing."""
    k_values: List[int] = None  # e.g., [1, 2, 4]
    entropy_thresholds: List[float] = None  # e.g., [0.6, 1.2]
    num_experts: int = 8
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 2, 4]
        if self.entropy_thresholds is None:
            self.entropy_thresholds = [0.6, 1.2]


@dataclass 
class RoutingMetrics:
    """Metrics from a routing operation."""
    entropy: float
    selected_k: int
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    compute_savings: float


class AdaptiveKRouter:
    """
    Entropy-guided dynamic expert selection router.
    
    Example:
        >>> router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")
        >>> router.calibrate(dataset="wikitext-2")
        >>> output = router.route(hidden_states)
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self._metrics_history = []
        self._total_tokens = 0
        self._total_savings = 0.0
        
    @classmethod
    def from_pretrained(cls, model_name: str) -> "AdaptiveKRouter":
        """
        Load router with pre-calibrated thresholds for known models.
        
        Args:
            model_name: One of "mixtral-8x7b", "qwen-moe", "olmoe-1b-7b"
        """
        presets = {
            "mixtral-8x7b": RoutingConfig(
                k_values=[1, 2, 4],
                entropy_thresholds=[0.6, 1.2],
                num_experts=8
            ),
            "qwen-moe": RoutingConfig(
                k_values=[1, 2, 4],
                entropy_thresholds=[0.5, 1.0],
                num_experts=8
            ),
            "olmoe-1b-7b": RoutingConfig(
                k_values=[1, 2, 4],
                entropy_thresholds=[0.7, 1.3],
                num_experts=8
            ),
        }
        
        if model_name.lower() not in presets:
            print(f"Warning: No preset for {model_name}, using defaults")
            return cls()
            
        return cls(config=presets[model_name.lower()])
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of routing distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def select_k(self, entropy: torch.Tensor) -> torch.Tensor:
        """Select K based on entropy thresholds."""
        k = torch.full_like(entropy, self.config.k_values[-1], dtype=torch.long)
        
        for i, threshold in enumerate(self.config.entropy_thresholds):
            mask = entropy < threshold
            k[mask] = self.config.k_values[i]
            
        return k
    
    def route(
        self, 
        router_logits: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[RoutingMetrics]]:
        """
        Route tokens to experts using adaptive K selection.
        
        Args:
            router_logits: (batch_size, num_experts) router output
            return_metrics: Whether to return detailed metrics
            
        Returns:
            expert_indices: (batch_size, k) selected expert indices
            expert_weights: (batch_size, k) normalized weights
            metrics: Optional routing metrics
        """
        batch_size = router_logits.shape[0]
        
        # Compute entropy per token
        entropy = self.compute_entropy(router_logits)
        
        # Select K per token
        k_per_token = self.select_k(entropy)
        max_k = k_per_token.max().item()
        
        # Get top-k experts
        probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(probs, k=max_k, dim=-1)
        
        # Normalize weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # Compute savings
        baseline_k = self.config.k_values[-1]
        avg_k = k_per_token.float().mean().item()
        savings = 1.0 - (avg_k / baseline_k)
        
        # Update stats
        self._total_tokens += batch_size
        self._total_savings += savings * batch_size
        
        if return_metrics:
            metrics = RoutingMetrics(
                entropy=entropy.mean().item(),
                selected_k=int(avg_k),
                expert_indices=top_indices,
                expert_weights=top_weights,
                compute_savings=savings
            )
            return top_indices, top_weights, metrics
            
        return top_indices, top_weights, None
    
    def patch(self, model):
        """
        Patch a HuggingFace model to use Adaptive-K routing.
        
        Args:
            model: A HuggingFace MoE model (Mixtral, Qwen, etc.)
            
        Returns:
            Patched model with Adaptive-K routing
        """
        # TODO: Implement model patching for different architectures
        raise NotImplementedError(
            "Model patching coming in v0.2.0. "
            "Use router.route() directly for now."
        )
    
    @property
    def stats(self) -> dict:
        """Get cumulative routing statistics."""
        if self._total_tokens == 0:
            return {"tokens": 0, "avg_savings": 0.0}
            
        return {
            "tokens_processed": self._total_tokens,
            "average_savings": self._total_savings / self._total_tokens,
            "estimated_cost_reduction": f"{(self._total_savings / self._total_tokens) * 100:.1f}%"
        }
    
    def reset_stats(self):
        """Reset cumulative statistics."""
        self._total_tokens = 0
        self._total_savings = 0.0
        self._metrics_history = []
