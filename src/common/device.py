"""
Device management utilities.
"""

import torch


def get_device(requested: str = "cuda") -> torch.device:
    """
    Get torch device, falling back to CPU if CUDA not available.
    
    Args:
        requested: Requested device ("cuda" or "cpu")
        
    Returns:
        torch.device object
    """
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
