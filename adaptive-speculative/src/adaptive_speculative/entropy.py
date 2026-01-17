"""
Entropy computation utilities for adaptive speculative decoding.

The core insight: model confidence (measured by output entropy) predicts
how many draft tokens will be accepted. Low entropy = confident = more drafts.
"""

import torch
import torch.nn.functional as F
from typing import Union


def compute_entropy(
    logits: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute entropy of logit distribution.
    
    Args:
        logits: Raw logits from model, shape [..., vocab_size]
        dim: Dimension to compute entropy over (default: last)
        eps: Small constant for numerical stability
        
    Returns:
        Entropy values, shape [...] (dim reduced)
        
    Example:
        >>> logits = torch.randn(32, 50257)  # batch of logits
        >>> entropy = compute_entropy(logits)
        >>> entropy.shape
        torch.Size([32])
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def compute_entropy_from_probs(
    probs: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute entropy from probability distribution.
    
    Args:
        probs: Probability distribution, shape [..., vocab_size]
        dim: Dimension to compute entropy over
        eps: Small constant for numerical stability
        
    Returns:
        Entropy values
    """
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def entropy_to_k(
    entropy: Union[float, torch.Tensor],
    thresholds: list[float] = [0.5, 1.0, 2.0],
    k_values: list[int] = [16, 8, 4, 1],
) -> Union[int, torch.Tensor]:
    """
    Map entropy to draft length K.
    
    The mapping logic:
    - Very low entropy (H < 0.5): Model is very confident, use long draft (K=16)
    - Low entropy (H < 1.0): Model is confident, use medium-long draft (K=8)
    - Medium entropy (H < 2.0): Model is uncertain, use short draft (K=4)
    - High entropy (H >= 2.0): Model is very uncertain, skip speculation (K=1)
    
    Args:
        entropy: Entropy value(s), scalar or tensor
        thresholds: Entropy thresholds (ascending), len = len(k_values) - 1
        k_values: Draft lengths for each bucket (descending), len = len(thresholds) + 1
        
    Returns:
        Draft length K, same shape as entropy
        
    Example:
        >>> entropy_to_k(0.3)  # Very confident
        16
        >>> entropy_to_k(0.7)  # Confident
        8
        >>> entropy_to_k(1.5)  # Uncertain
        4
        >>> entropy_to_k(3.0)  # Very uncertain
        1
    """
    assert len(thresholds) == len(k_values) - 1, \
        f"thresholds ({len(thresholds)}) must be len(k_values) - 1 ({len(k_values) - 1})"
    
    if isinstance(entropy, (int, float)):
        # Scalar case
        for threshold, k in zip(thresholds, k_values[:-1]):
            if entropy < threshold:
                return k
        return k_values[-1]
    
    # Tensor case - vectorized
    k_tensor = torch.full_like(entropy, k_values[-1], dtype=torch.long)
    
    # Apply thresholds in reverse order (so higher K overwrites lower K)
    for threshold, k in reversed(list(zip(thresholds, k_values[:-1]))):
        mask = entropy < threshold
        k_tensor[mask] = k
    
    return k_tensor


def batch_entropy_to_k(
    entropies: torch.Tensor,
    thresholds: list[float] = [0.5, 1.0, 2.0],
    k_values: list[int] = [16, 8, 4, 1],
) -> tuple[torch.Tensor, dict]:
    """
    Batch-aware entropy to K mapping with statistics.
    
    Returns both the K values and statistics about the distribution.
    
    Args:
        entropies: Entropy values, shape [batch_size]
        thresholds: Entropy thresholds
        k_values: Draft lengths
        
    Returns:
        Tuple of:
        - k_tensor: Draft lengths, shape [batch_size]
        - stats: Dict with distribution statistics
    """
    k_tensor = entropy_to_k(entropies, thresholds, k_values)
    
    # Compute statistics
    stats = {
        "mean_entropy": entropies.mean().item(),
        "std_entropy": entropies.std().item(),
        "min_entropy": entropies.min().item(),
        "max_entropy": entropies.max().item(),
        "mean_k": k_tensor.float().mean().item(),
        "k_distribution": {},
    }
    
    # K distribution
    for k in k_values:
        count = (k_tensor == k).sum().item()
        stats["k_distribution"][k] = count / len(k_tensor)
    
    return k_tensor, stats


def calibrate_thresholds(
    entropies: torch.Tensor,
    target_k_ratios: list[float] = [0.2, 0.3, 0.3, 0.2],
    k_values: list[int] = [16, 8, 4, 1],
) -> list[float]:
    """
    Calibrate entropy thresholds to achieve target K distribution.
    
    Given a sample of entropies and desired K ratios, find thresholds
    that produce approximately that distribution.
    
    Args:
        entropies: Sample entropy values, shape [N]
        target_k_ratios: Desired fraction for each K bucket (must sum to 1)
        k_values: Draft lengths
        
    Returns:
        Calibrated thresholds
        
    Example:
        >>> entropies = compute_entropy(sample_logits)
        >>> thresholds = calibrate_thresholds(
        ...     entropies, 
        ...     target_k_ratios=[0.25, 0.25, 0.25, 0.25]
        ... )
    """
    assert len(target_k_ratios) == len(k_values), \
        "target_k_ratios must have same length as k_values"
    assert abs(sum(target_k_ratios) - 1.0) < 1e-6, \
        f"target_k_ratios must sum to 1, got {sum(target_k_ratios)}"
    
    # Sort entropies
    sorted_entropies, _ = torch.sort(entropies)
    n = len(sorted_entropies)
    
    # Find quantiles
    thresholds = []
    cumsum = 0.0
    for ratio in target_k_ratios[:-1]:
        cumsum += ratio
        idx = int(cumsum * n)
        idx = min(idx, n - 1)
        thresholds.append(sorted_entropies[idx].item())
    
    return thresholds


# Convenience functions for common configurations

def get_conservative_config() -> tuple[list[float], list[int]]:
    """
    Conservative config: fewer long drafts, lower risk.
    Good for: Production, latency-sensitive applications.
    """
    return [0.3, 0.7, 1.5], [8, 6, 4, 2]


def get_aggressive_config() -> tuple[list[float], list[int]]:
    """
    Aggressive config: more long drafts, higher potential speedup.
    Good for: Batch processing, throughput-focused applications.
    """
    return [0.7, 1.2, 2.5], [16, 10, 6, 2]


def get_balanced_config() -> tuple[list[float], list[int]]:
    """
    Balanced config: middle ground.
    Good for: General use, first experiments.
    """
    return [0.5, 1.0, 2.0], [12, 8, 4, 2]
