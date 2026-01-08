"""
Adaptive-K Policy for SBM-Efficient.

Implements threshold-based policy (v1) for dynamic K selection based on
routing entropy. Lower entropy (confident routing) -> smaller K.
Higher entropy (uncertain routing) -> larger K.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AdaptiveKConfig:
    """Configuration for Adaptive-K policy.
    
    Attributes:
        k_values: List of possible K values, ascending. E.g., [1, 2, 4]
        h_thresholds: Entropy thresholds for K selection.
                      len(h_thresholds) == len(k_values) - 1
                      E.g., [0.6, 1.2] means:
                        H < 0.6 -> K=1
                        0.6 <= H < 1.2 -> K=2
                        H >= 1.2 -> K=4
        eps: Small value for numerical stability in log.
    """
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4])
    h_thresholds: List[float] = field(default_factory=lambda: [0.6, 1.2])
    eps: float = 1e-9
    
    def __post_init__(self):
        """Validate config."""
        if len(self.h_thresholds) != len(self.k_values) - 1:
            raise ValueError(
                f"h_thresholds length ({len(self.h_thresholds)}) must be "
                f"k_values length - 1 ({len(self.k_values) - 1})"
            )
        if self.k_values != sorted(self.k_values):
            raise ValueError(f"k_values must be ascending: {self.k_values}")
        if self.h_thresholds != sorted(self.h_thresholds):
            raise ValueError(f"h_thresholds must be ascending: {self.h_thresholds}")


class AdaptiveKPolicy:
    """Threshold-based Adaptive-K policy (v1).
    
    Selects K based on routing entropy:
    - Low entropy (confident) -> small K (fewer experts)
    - High entropy (uncertain) -> large K (more experts)
    """
    
    def __init__(self, cfg: AdaptiveKConfig):
        """Initialize policy with config.
        
        Args:
            cfg: AdaptiveKConfig with k_values and h_thresholds.
        """
        self.cfg = cfg
        self._k_counts: Dict[int, int] = {k: 0 for k in cfg.k_values}
        self._total_samples = 0
    
    def reset_stats(self):
        """Reset K statistics for new epoch/run."""
        self._k_counts = {k: 0 for k in self.cfg.k_values}
        self._total_samples = 0
    
    def entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution.
        
        Args:
            p: Probability distribution [B, N] (already softmaxed).
        
        Returns:
            Entropy tensor [B].
        """
        eps = self.cfg.eps
        return -(p * (p + eps).log()).sum(dim=-1)
    
    def k_from_entropy(self, H: torch.Tensor) -> torch.Tensor:
        """Determine K for each sample based on entropy.
        
        Uses threshold policy v1:
        - H < h_thresholds[0] -> k_values[0]
        - h_thresholds[i-1] <= H < h_thresholds[i] -> k_values[i]
        - H >= h_thresholds[-1] -> k_values[-1]
        
        Args:
            H: Entropy tensor [B].
        
        Returns:
            K tensor [B] (int64).
        """
        k_vals = self.cfg.k_values
        thresholds = self.cfg.h_thresholds
        device = H.device
        
        # Start with minimum K
        K = torch.full_like(H, fill_value=k_vals[0], dtype=torch.long)
        
        # Apply thresholds
        for i, threshold in enumerate(thresholds):
            K = torch.where(
                H >= threshold,
                torch.tensor(k_vals[i + 1], device=device, dtype=torch.long),
                K
            )
        
        return K
    
    def topk_indices_and_weights(
        self, 
        logits: torch.Tensor, 
        K: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get top-K indices and weights for each sample.
        
        Args:
            logits: Router logits [B, N].
            K: Per-sample K values [B].
            tau: Temperature for softmax weights.
        
        Returns:
            Tuple of:
                - idx_list: List of [K_b] index tensors per sample.
                - weights_list: List of [K_b] weight tensors per sample.
        """
        B = logits.size(0)
        idx_list = []
        weights_list = []
        
        # Compute softmax for weights
        p = F.softmax(logits / tau, dim=-1)
        
        for b in range(B):
            kb = int(K[b].item())
            
            # Get top-K indices
            topk_vals, topk_idx = torch.topk(logits[b], k=kb, dim=-1)
            idx_list.append(topk_idx)
            
            # Get weights (re-normalized softmax over selected experts)
            selected_probs = p[b, topk_idx]
            normalized_weights = selected_probs / (selected_probs.sum() + self.cfg.eps)
            weights_list.append(normalized_weights)
            
            # Update stats
            self._k_counts[kb] = self._k_counts.get(kb, 0) + 1
            self._total_samples += 1
        
        return idx_list, weights_list
    
    def topk_indices_bucketed(
        self, 
        logits: torch.Tensor, 
        K: torch.Tensor,
        tau: float = 1.0
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Get top-K indices grouped by K value (bucketed for efficiency).
        
        Groups samples with same K together for potential batch processing.
        
        Args:
            logits: Router logits [B, N].
            K: Per-sample K values [B].
            tau: Temperature for softmax weights.
        
        Returns:
            Tuple of:
                - buckets: Dict[k_value -> (sample_indices, expert_indices, weights)]
                - K: Original K tensor for reference.
        """
        B = logits.size(0)
        device = logits.device
        
        # Compute softmax for weights
        p = F.softmax(logits / tau, dim=-1)
        
        buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
        for k_val in self.cfg.k_values:
            # Find samples with this K
            mask = (K == k_val)
            if not mask.any():
                continue
            
            sample_indices = mask.nonzero(as_tuple=True)[0]
            batch_logits = logits[sample_indices]  # [num_samples, N]
            batch_p = p[sample_indices]  # [num_samples, N]
            
            # Get top-K for all samples in bucket at once
            _, topk_idx = torch.topk(batch_logits, k=k_val, dim=-1)  # [num_samples, k_val]
            
            # Get weights
            # Gather selected probabilities
            selected_probs = torch.gather(batch_p, dim=1, index=topk_idx)  # [num_samples, k_val]
            normalized_weights = selected_probs / (selected_probs.sum(dim=-1, keepdim=True) + self.cfg.eps)
            
            buckets[k_val] = (sample_indices, topk_idx, normalized_weights)
            
            # Update stats
            num_samples = sample_indices.size(0)
            self._k_counts[k_val] = self._k_counts.get(k_val, 0) + num_samples
            self._total_samples += num_samples
        
        return buckets, K
    
    def get_stats(self) -> Dict[str, float]:
        """Get K statistics.
        
        Returns:
            Dict with k_mean, k_std, k_histogram (counts and percentages).
        """
        if self._total_samples == 0:
            return {
                "k_mean": 0.0,
                "k_std": 0.0,
                "k_histogram": {},
                "k_histogram_pct": {}
            }
        
        # Compute mean
        total_k = sum(k * count for k, count in self._k_counts.items())
        k_mean = total_k / self._total_samples
        
        # Compute std
        variance = sum(
            count * (k - k_mean) ** 2 
            for k, count in self._k_counts.items()
        ) / self._total_samples
        k_std = variance ** 0.5
        
        # Histogram as percentages
        k_histogram_pct = {
            str(k): round(count / self._total_samples * 100, 2)
            for k, count in self._k_counts.items()
        }
        
        return {
            "k_mean": round(k_mean, 4),
            "k_std": round(k_std, 4),
            "k_histogram": {str(k): count for k, count in self._k_counts.items()},
            "k_histogram_pct": k_histogram_pct
        }
