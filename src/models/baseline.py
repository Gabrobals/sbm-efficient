"""
Baseline models for SBM-Efficient experiments.

Models:
- BaselineMLP: Simple MLP for XOR and tabular tasks
- BaselineCNN: Simple CNN for vision tasks (MNIST, CIFAR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaselineMLP(nn.Module):
    """
    Baseline MLP for XOR and simple classification tasks.
    
    Architecture:
        input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
    
    For fair comparison with SBM, this uses N "expert-like" hidden units
    that can be compared to SBM's N experts with K active.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineCNN(nn.Module):
    """
    Baseline CNN for vision tasks (MNIST, Fashion-MNIST, CIFAR-10).
    
    Architecture:
        Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> FC
    
    Designed to have comparable parameter count to SBM variants.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Adaptive pooling for consistent size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_model(
    task: str,
    config: dict,
    input_shape: Optional[tuple] = None
) -> nn.Module:
    """
    Factory function to create baseline model based on task.
    
    Args:
        task: Task name (xor, mnist, fashion_mnist, cifar10)
        config: Full experiment config dictionary
        input_shape: Optional input shape override
        
    Returns:
        Initialized model
    """
    # Get hidden dim from config (use experts_num * some factor for fair comparison)
    sbm_config = config.get("sbm", {})
    experts_num = sbm_config.get("experts_num", 16)
    
    # Hidden dim scales with number of experts for fair param comparison
    hidden_dim = experts_num * 8  # Each expert ~8 hidden units equivalent
    
    if task == "xor":
        return BaselineMLP(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=2,
            dropout=0.0
        )
    
    elif task == "mnist":
        return BaselineCNN(
            in_channels=1,
            num_classes=10,
            hidden_dim=hidden_dim,
            dropout=0.0
        )
    
    elif task == "fashion_mnist":
        return BaselineCNN(
            in_channels=1,
            num_classes=10,
            hidden_dim=hidden_dim,
            dropout=0.0
        )
    
    elif task == "cifar10":
        return BaselineCNN(
            in_channels=3,
            num_classes=10,
            hidden_dim=hidden_dim,
            dropout=0.0
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Quick test
    print("Testing BaselineMLP (XOR)...")
    mlp = BaselineMLP(input_dim=2, hidden_dim=64, output_dim=2)
    x = torch.randn(4, 2)
    out = mlp(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {mlp.count_parameters()}")
    
    print("\nTesting BaselineCNN (MNIST)...")
    cnn = BaselineCNN(in_channels=1, num_classes=10, hidden_dim=128)
    x = torch.randn(4, 1, 28, 28)
    out = cnn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {cnn.count_parameters()}")
    
    print("\nTesting BaselineCNN (CIFAR)...")
    cnn = BaselineCNN(in_channels=3, num_classes=10, hidden_dim=128)
    x = torch.randn(4, 3, 32, 32)
    out = cnn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {cnn.count_parameters()}")
    
    print("\n[OK] All baseline models working!")
