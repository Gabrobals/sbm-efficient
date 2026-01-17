"""
Adaptive Speculative Decoding

Entropy-guided dynamic draft length for speculative decoding.
"""

__version__ = "0.1.0"

from .entropy import compute_entropy, entropy_to_k
from .config import AdaptiveSpeculativeConfig

__all__ = [
    "compute_entropy",
    "entropy_to_k", 
    "AdaptiveSpeculativeConfig",
]
