"""
Expert modules for SBM-Efficient.

Provides modular expert networks that can be selectively activated
based on routing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Expert(nn.Module):
    """
    Single expert module (small MLP).
    
    Each expert is a small feedforward network that processes
    the input independently.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def count_flops(self, batch_size: int = 1) -> int:
        """Count FLOPs for one forward pass."""
        # fc1: 2 * in * out (multiply-add)
        flops_fc1 = 2 * self.fc1.in_features * self.fc1.out_features
        # fc2: 2 * in * out
        flops_fc2 = 2 * self.fc2.in_features * self.fc2.out_features
        return (flops_fc1 + flops_fc2) * batch_size


class ExpertPool(nn.Module):
    """
    Pool of N experts with selective activation.
    
    This is the core module for sparse routing. Only the selected
    experts are executed, providing real computational savings.
    """
    
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create N independent experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass through selected experts only.
        
        IMPORTANT: This executes ONLY the selected experts (no masking trick).
        This is the correct pattern per IMPLEMENTATION_NOTES.md.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            expert_indices: Selected expert indices (batch_size, K)
            expert_weights: Optional weights for combining (batch_size, K)
            
        Returns:
            (output, flops_executed) tuple
        """
        batch_size = x.size(0)
        K = expert_indices.size(1)
        device = x.device
        
        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Track FLOPs
        flops_executed = 0
        
        # Process each expert that is selected by at least one sample
        unique_experts = expert_indices.unique()
        
        for expert_idx in unique_experts:
            expert_idx = expert_idx.item()
            
            # Find which samples use this expert and at which K position
            mask = (expert_indices == expert_idx)  # (batch_size, K)
            samples_using_expert = mask.any(dim=1)  # (batch_size,)
            
            if not samples_using_expert.any():
                continue
            
            # Get samples that use this expert
            selected_x = x[samples_using_expert]  # (num_selected, input_dim)
            
            # Execute expert ONLY on selected samples
            expert_output = self.experts[expert_idx](selected_x)  # (num_selected, output_dim)
            
            # Count FLOPs for this expert execution
            flops_executed += self.experts[expert_idx].count_flops(selected_x.size(0))
            
            # Get weights for this expert (for weighted combination)
            if expert_weights is not None:
                # Sum weights across K positions where this expert appears
                weights_for_expert = (mask.float() * expert_weights).sum(dim=1)  # (batch_size,)
                weights_selected = weights_for_expert[samples_using_expert]  # (num_selected,)
                expert_output = expert_output * weights_selected.unsqueeze(1)
            
            # Add to output
            output[samples_using_expert] += expert_output
        
        return output, flops_executed

    def execute_sparse(
        self,
        x: torch.Tensor,
        idx_list: List[torch.Tensor],
        weights_list: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, int]:
        """Execute experts per-sample with variable K (Adaptive-K).

        Args:
            x: Input tensor [B, input_dim]
            idx_list: List of length B with expert indices per sample
            weights_list: Optional list of weights per sample matching idx_list

        Returns:
            (output, flops_executed)
        """
        batch_size = x.size(0)
        device = x.device
        outputs = []
        flops_executed = 0

        for b, idx in enumerate(idx_list):
            if idx.numel() == 0:
                # No experts selected; append zeros to keep shape consistent
                outputs.append(torch.zeros(1, self.output_dim, device=device))
                continue

            xb = x[b:b + 1]  # Preserve batch dimension
            yb = torch.zeros(1, self.output_dim, device=device)

            # Optional weights for this sample
            wb = None
            if weights_list is not None:
                wb = weights_list[b]
                if wb is not None and wb.numel() != idx.numel():
                    raise ValueError("weights_list entry size must match idx_list entry size")

            for j, e_idx in enumerate(idx.tolist()):
                out_e = self.experts[e_idx](xb)
                flops_executed += self.experts[e_idx].count_flops(batch_size=1)

                if wb is not None:
                    weight = wb[j]
                    out_e = out_e * weight
                yb = yb + out_e

            outputs.append(yb)

        output = torch.cat(outputs, dim=0)
        return output, flops_executed
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureExtractor(nn.Module):
    """
    Feature extractor that prepares input for expert routing.
    
    For vision tasks: CNN backbone
    For tabular tasks: Simple linear projection
    """
    
    def __init__(
        self,
        task: str,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.task = task
        self.output_dim = output_dim
        
        if task == "xor":
            # Simple projection for 2D input
            self.extractor = nn.Linear(2, output_dim)
            self.flatten = nn.Identity()
            
        elif task in ["mnist", "fashion_mnist"]:
            # CNN for 28x28 grayscale
            self.extractor = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((2, 2))
            )
            self.flatten = nn.Flatten()
            self.proj = nn.Linear(32 * 2 * 2, output_dim)
            
        elif task == "cifar10":
            # CNN for 32x32 RGB
            self.extractor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((2, 2))
            )
            self.flatten = nn.Flatten()
            self.proj = nn.Linear(32 * 2 * 2, output_dim)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == "xor":
            return F.relu(self.extractor(x))
        else:
            x = self.extractor(x)
            x = self.flatten(x)
            x = F.relu(self.proj(x))
            return x


if __name__ == "__main__":
    print("Testing Expert...")
    expert = Expert(64, 32, 10)
    x = torch.randn(4, 64)
    out = expert(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  FLOPs per sample: {expert.count_flops(1)}")
    
    print("\nTesting ExpertPool...")
    pool = ExpertPool(num_experts=8, input_dim=64, hidden_dim=32, output_dim=10)
    x = torch.randn(4, 64)
    indices = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]])  # K=2
    weights = torch.softmax(torch.randn(4, 2), dim=1)
    out, flops = pool(x, indices, weights)
    print(f"  Input: {x.shape}, Indices: {indices.shape}")
    print(f"  Output: {out.shape}, FLOPs executed: {flops}")
    print(f"  Parameters: {pool.count_parameters()}")
    
    print("\nTesting FeatureExtractor (MNIST)...")
    fe = FeatureExtractor("mnist", output_dim=64)
    x = torch.randn(4, 1, 28, 28)
    out = fe(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    print("\n[OK] All expert modules working!")
