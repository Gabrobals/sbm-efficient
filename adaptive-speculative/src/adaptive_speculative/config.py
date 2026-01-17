"""
Configuration for adaptive speculative decoding.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdaptiveSpeculativeConfig:
    """
    Configuration for entropy-guided adaptive speculative decoding.
    
    The key parameters are:
    - thresholds: Entropy thresholds that determine K boundaries
    - k_values: Draft lengths for each entropy bucket
    
    Example:
        config = AdaptiveSpeculativeConfig(
            draft_model="facebook/opt-125m",
            thresholds=[0.5, 1.0, 2.0],
            k_values=[16, 8, 4, 1],
        )
    """
    
    # Draft model configuration
    draft_model: str
    draft_model_revision: Optional[str] = None
    draft_model_quantization: Optional[str] = None
    
    # Adaptive K configuration
    thresholds: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    k_values: list[int] = field(default_factory=lambda: [16, 8, 4, 1])
    
    # Fallback for non-adaptive mode
    fallback_k: int = 5
    
    # Enable/disable adaptive mode
    adaptive_enabled: bool = True
    
    # Minimum samples before enabling adaptive (for warmup)
    warmup_tokens: int = 100
    
    # Whether to log K distribution
    log_statistics: bool = False
    log_interval: int = 1000
    
    def __post_init__(self):
        """Validate configuration."""
        assert len(self.thresholds) == len(self.k_values) - 1, \
            f"thresholds ({len(self.thresholds)}) must be len(k_values) - 1 ({len(self.k_values) - 1})"
        
        # Thresholds must be ascending
        for i in range(len(self.thresholds) - 1):
            assert self.thresholds[i] < self.thresholds[i + 1], \
                f"thresholds must be ascending, got {self.thresholds}"
        
        # K values should generally be descending (lower entropy = higher K)
        # but we don't enforce this to allow experimentation
        
    def to_vllm_config(self) -> dict:
        """
        Convert to vLLM speculative_config format.
        
        Note: This returns the base config. The adaptive logic
        must be integrated into vLLM's proposer.
        
        Returns:
            Dict compatible with vLLM's speculative_config parameter
        """
        config = {
            "model": self.draft_model,
            "num_speculative_tokens": max(self.k_values),  # Use max K as upper bound
        }
        
        if self.draft_model_revision:
            config["revision"] = self.draft_model_revision
            
        if self.draft_model_quantization:
            config["quantization"] = self.draft_model_quantization
            
        # Store adaptive config in metadata (for our patched vLLM)
        config["_adaptive_config"] = {
            "enabled": self.adaptive_enabled,
            "thresholds": self.thresholds,
            "k_values": self.k_values,
            "warmup_tokens": self.warmup_tokens,
        }
        
        return config
    
    @classmethod
    def conservative(cls, draft_model: str) -> "AdaptiveSpeculativeConfig":
        """Create conservative config (lower risk, stable speedup)."""
        return cls(
            draft_model=draft_model,
            thresholds=[0.3, 0.7, 1.5],
            k_values=[8, 6, 4, 2],
        )
    
    @classmethod
    def aggressive(cls, draft_model: str) -> "AdaptiveSpeculativeConfig":
        """Create aggressive config (higher potential speedup)."""
        return cls(
            draft_model=draft_model,
            thresholds=[0.7, 1.2, 2.5],
            k_values=[16, 10, 6, 2],
        )
    
    @classmethod
    def balanced(cls, draft_model: str) -> "AdaptiveSpeculativeConfig":
        """Create balanced config (good starting point)."""
        return cls(
            draft_model=draft_model,
            thresholds=[0.5, 1.0, 2.0],
            k_values=[12, 8, 4, 2],
        )


@dataclass
class AdaptiveSpeculativeStats:
    """
    Runtime statistics for adaptive speculative decoding.
    """
    
    total_tokens: int = 0
    total_drafts: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    
    # K distribution counters
    k_counts: dict = field(default_factory=dict)
    
    # Entropy statistics
    entropy_sum: float = 0.0
    entropy_sq_sum: float = 0.0
    
    def update(self, k: int, entropy: float, accepted: int, rejected: int):
        """Update statistics with a new draft round."""
        self.total_tokens += accepted + rejected
        self.total_drafts += 1
        self.accepted_tokens += accepted
        self.rejected_tokens += rejected
        
        self.k_counts[k] = self.k_counts.get(k, 0) + 1
        
        self.entropy_sum += entropy
        self.entropy_sq_sum += entropy ** 2
    
    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens
    
    @property
    def mean_entropy(self) -> float:
        """Mean entropy across all drafts."""
        if self.total_drafts == 0:
            return 0.0
        return self.entropy_sum / self.total_drafts
    
    @property
    def std_entropy(self) -> float:
        """Standard deviation of entropy."""
        if self.total_drafts < 2:
            return 0.0
        mean = self.mean_entropy
        variance = (self.entropy_sq_sum / self.total_drafts) - (mean ** 2)
        return variance ** 0.5 if variance > 0 else 0.0
    
    @property
    def k_distribution(self) -> dict:
        """Normalized K distribution."""
        if self.total_drafts == 0:
            return {}
        return {k: count / self.total_drafts for k, count in self.k_counts.items()}
    
    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_drafts": self.total_drafts,
            "acceptance_rate": self.acceptance_rate,
            "mean_entropy": self.mean_entropy,
            "std_entropy": self.std_entropy,
            "k_distribution": self.k_distribution,
        }
    
    def __str__(self) -> str:
        return (
            f"AdaptiveSpeculativeStats(\n"
            f"  tokens={self.total_tokens}, drafts={self.total_drafts}\n"
            f"  acceptance_rate={self.acceptance_rate:.2%}\n"
            f"  mean_entropy={self.mean_entropy:.3f} +/- {self.std_entropy:.3f}\n"
            f"  k_distribution={self.k_distribution}\n"
            f")"
        )
