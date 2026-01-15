"""Tests for AdaptiveKRouter."""

import pytest
import torch
from adaptive_k import AdaptiveKRouter
from adaptive_k.router import RoutingConfig


class TestAdaptiveKRouter:
    """Test suite for AdaptiveKRouter."""
    
    def test_init_default(self):
        """Test default initialization."""
        router = AdaptiveKRouter()
        assert router.config.k_values == [1, 2, 4]
        assert router.config.entropy_thresholds == [0.6, 1.2]
        
    def test_init_custom_config(self):
        """Test custom config initialization."""
        config = RoutingConfig(
            k_values=[1, 2, 8],
            entropy_thresholds=[0.5, 1.0],
            num_experts=16
        )
        router = AdaptiveKRouter(config=config)
        assert router.config.k_values == [1, 2, 8]
        assert router.config.num_experts == 16
        
    def test_from_pretrained_mixtral(self):
        """Test loading Mixtral preset."""
        router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")
        assert router.config.num_experts == 8
        assert router.config.entropy_thresholds == [0.6, 1.2]
        
    def test_from_pretrained_unknown(self):
        """Test loading unknown model uses defaults."""
        router = AdaptiveKRouter.from_pretrained("unknown-model")
        assert router.config.k_values == [1, 2, 4]
        
    def test_compute_entropy(self):
        """Test entropy computation."""
        router = AdaptiveKRouter()
        
        # Uniform distribution = high entropy
        uniform_logits = torch.ones(1, 8)
        entropy_uniform = router.compute_entropy(uniform_logits)
        
        # Peaked distribution = low entropy
        peaked_logits = torch.zeros(1, 8)
        peaked_logits[0, 0] = 10.0
        entropy_peaked = router.compute_entropy(peaked_logits)
        
        assert entropy_uniform > entropy_peaked
        
    def test_select_k(self):
        """Test K selection based on entropy."""
        router = AdaptiveKRouter()
        
        # Low entropy → K=1
        low_entropy = torch.tensor([0.3])
        k_low = router.select_k(low_entropy)
        assert k_low.item() == 1
        
        # Medium entropy → K=2
        med_entropy = torch.tensor([0.8])
        k_med = router.select_k(med_entropy)
        assert k_med.item() == 2
        
        # High entropy → K=4
        high_entropy = torch.tensor([1.5])
        k_high = router.select_k(high_entropy)
        assert k_high.item() == 4
        
    def test_route_basic(self):
        """Test basic routing."""
        router = AdaptiveKRouter()
        
        batch_size = 32
        num_experts = 8
        logits = torch.randn(batch_size, num_experts)
        
        indices, weights, _ = router.route(logits)
        
        # Check shapes
        assert indices.shape[0] == batch_size
        assert weights.shape[0] == batch_size
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        
    def test_route_with_metrics(self):
        """Test routing with metrics."""
        router = AdaptiveKRouter()
        
        logits = torch.randn(16, 8)
        indices, weights, metrics = router.route(logits, return_metrics=True)
        
        assert metrics is not None
        assert 0 <= metrics.compute_savings <= 1
        assert metrics.entropy >= 0
        
    def test_stats_tracking(self):
        """Test statistics tracking."""
        router = AdaptiveKRouter()
        
        # Process some tokens
        for _ in range(10):
            logits = torch.randn(100, 8)
            router.route(logits)
            
        stats = router.stats
        assert stats["tokens_processed"] == 1000
        assert 0 <= stats["average_savings"] <= 1
        
    def test_reset_stats(self):
        """Test stats reset."""
        router = AdaptiveKRouter()
        
        logits = torch.randn(100, 8)
        router.route(logits)
        
        assert router.stats["tokens_processed"] == 100
        
        router.reset_stats()
        assert router.stats["tokens_processed"] == 0


class TestRoutingConfig:
    """Test suite for RoutingConfig."""
    
    def test_default_values(self):
        """Test default config values."""
        config = RoutingConfig()
        assert config.k_values == [1, 2, 4]
        assert config.entropy_thresholds == [0.6, 1.2]
        assert config.num_experts == 8
        
    def test_custom_values(self):
        """Test custom config values."""
        config = RoutingConfig(
            k_values=[1, 4, 8],
            entropy_thresholds=[0.3, 0.9],
            num_experts=16
        )
        assert config.k_values == [1, 4, 8]
        assert len(config.entropy_thresholds) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
