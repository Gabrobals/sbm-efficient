"""
Unit tests for AdaptiveKMoeRoutingMethod.

Tests cover:
- Basic functionality with random logits
- Sparse logits (realistic scenario)
- Edge cases (all low entropy, all high entropy)
- Output shape compatibility
- Statistics accuracy
"""

import pytest
import torch
from typing import Tuple

# Import the routing method (adjust path for actual TensorRT-LLM integration)
import sys
sys.path.insert(0, '.')
from routing import AdaptiveKMoeRoutingMethod, AdaptiveKConfig, create_adaptive_k_routing


class TestAdaptiveKRoutingBasic:
    """Basic functionality tests."""
    
    def test_init_default(self):
        """Test default initialization."""
        routing = AdaptiveKMoeRoutingMethod()
        assert routing.k_max == 8
        assert routing.config.k_min == 2
        assert len(routing.config.k_values) == 3
        
    def test_init_custom(self):
        """Test custom initialization."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=1,
            k_max=4,
            entropy_thresholds=[1.0, 1.5]
        )
        assert routing.k_max == 4
        assert routing.config.k_min == 1
        
    def test_init_with_config(self):
        """Test initialization with AdaptiveKConfig."""
        config = AdaptiveKConfig(
            k_min=2,
            k_max=6,
            k_values=[2, 4, 6],
            entropy_thresholds=[1.2, 1.8]
        )
        routing = AdaptiveKMoeRoutingMethod(config=config)
        assert routing.k_max == 6
        assert routing.config.k_values == [2, 4, 6]


class TestAdaptiveKRoutingApply:
    """Tests for the apply method."""
    
    @pytest.fixture
    def routing(self):
        return AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
    
    def test_output_shape(self, routing):
        """Test output shapes match expectations."""
        router_logits = torch.randn(100, 64)  # 100 tokens, 64 experts
        experts, weights = routing.apply(router_logits)
        
        assert experts.shape == (100, 8)  # (num_tokens, k_max)
        assert weights.shape == (100, 8)
        
    def test_output_dtype(self, routing):
        """Test output dtypes are correct."""
        router_logits = torch.randn(50, 32)
        experts, weights = routing.apply(router_logits)
        
        assert experts.dtype == torch.int32
        assert weights.dtype == torch.float32
        
    def test_weights_sum_to_one(self, routing):
        """Test that non-zero weights sum to ~1 per token."""
        router_logits = torch.randn(100, 64) * 3  # Sparse-ish
        experts, weights = routing.apply(router_logits)
        
        # Each row should sum to 1 (only considering non-zero weights)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
        
    def test_experts_are_valid_indices(self, routing):
        """Test expert indices are valid."""
        num_experts = 64
        router_logits = torch.randn(100, num_experts)
        experts, weights = routing.apply(router_logits)
        
        assert (experts >= 0).all()
        assert (experts < num_experts).all()


class TestAdaptiveKRoutingEntropy:
    """Tests for entropy-based K selection."""
    
    def test_low_entropy_uses_fewer_experts(self):
        """Test that low entropy (confident) routing uses fewer experts."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2,
            k_max=8,
            entropy_thresholds=[1.0, 2.0]
        )
        
        # Create very peaked distribution (low entropy)
        logits = torch.zeros(10, 64)
        logits[:, 0] = 10.0  # Expert 0 dominates
        
        experts, weights = routing.apply(logits)
        stats = routing.get_stats()
        
        # Should use k_min for most tokens
        assert stats['avg_k'] < 4, f"Expected low avg_k, got {stats['avg_k']}"
        
    def test_high_entropy_uses_more_experts(self):
        """Test that high entropy (uncertain) routing uses more experts."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2,
            k_max=8,
            entropy_thresholds=[0.5, 1.0]  # Very low thresholds
        )
        
        # Create uniform distribution (high entropy)
        logits = torch.zeros(10, 8)  # All equal -> max entropy
        
        experts, weights = routing.apply(logits)
        stats = routing.get_stats()
        
        # Should use k_max for all tokens (uniform = max entropy)
        assert stats['avg_k'] == 8, f"Expected k_max=8, got {stats['avg_k']}"


class TestAdaptiveKRoutingStatistics:
    """Tests for statistics tracking."""
    
    def test_stats_initial(self):
        """Test initial statistics."""
        routing = AdaptiveKMoeRoutingMethod()
        stats = routing.get_stats()
        
        assert stats['total_tokens'] == 0
        assert stats['mean_entropy'] == 0.0
        
    def test_stats_after_apply(self):
        """Test statistics update after apply."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        
        logits = torch.randn(100, 64)
        routing.apply(logits)
        
        stats = routing.get_stats()
        assert stats['total_tokens'] == 100
        assert stats['mean_entropy'] > 0
        assert 'k_distribution' in stats
        assert 'compute_savings_pct' in stats
        
    def test_stats_accumulate(self):
        """Test statistics accumulate across multiple calls."""
        routing = AdaptiveKMoeRoutingMethod()
        
        routing.apply(torch.randn(50, 32))
        routing.apply(torch.randn(50, 32))
        
        stats = routing.get_stats()
        assert stats['total_tokens'] == 100
        
    def test_stats_reset(self):
        """Test statistics reset."""
        routing = AdaptiveKMoeRoutingMethod()
        routing.apply(torch.randn(100, 32))
        
        routing.reset_stats()
        stats = routing.get_stats()
        
        assert stats['total_tokens'] == 0
        

class TestAdaptiveKRoutingEdgeCases:
    """Edge case tests."""
    
    def test_single_token(self):
        """Test with single token."""
        routing = AdaptiveKMoeRoutingMethod()
        experts, weights = routing.apply(torch.randn(1, 64))
        
        assert experts.shape == (1, 8)
        assert weights.shape == (1, 8)
        
    def test_single_expert(self):
        """Test with single expert (degenerate case)."""
        routing = AdaptiveKMoeRoutingMethod(k_min=1, k_max=1)
        experts, weights = routing.apply(torch.randn(10, 1))
        
        assert experts.shape == (10, 1)
        assert (experts == 0).all()  # Only expert 0 exists
        
    def test_k_max_equals_num_experts(self):
        """Test when k_max equals number of experts."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        experts, weights = routing.apply(torch.randn(10, 8))  # 8 experts, k_max=8
        
        assert experts.shape == (10, 8)
        
    def test_gpu_if_available(self):
        """Test on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        routing = AdaptiveKMoeRoutingMethod()
        logits = torch.randn(100, 64, device='cuda')
        
        experts, weights = routing.apply(logits)
        
        assert experts.device.type == 'cuda'
        assert weights.device.type == 'cuda'


class TestAdaptiveKRoutingFactory:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Test factory with defaults."""
        routing = create_adaptive_k_routing()
        assert isinstance(routing, AdaptiveKMoeRoutingMethod)
        
    def test_create_custom(self):
        """Test factory with custom params."""
        routing = create_adaptive_k_routing(
            k_min=1,
            k_max=4,
            entropy_thresholds=[0.8, 1.2]
        )
        assert routing.k_max == 4
        assert routing.config.k_min == 1


class TestAdaptiveKConfig:
    """Tests for AdaptiveKConfig."""
    
    def test_default_values(self):
        """Test default config values."""
        config = AdaptiveKConfig()
        assert config.k_min == 2
        assert config.k_max == 8
        assert len(config.k_values) == 3
        
    def test_threshold_validation(self):
        """Test threshold length validation."""
        with pytest.raises(AssertionError):
            AdaptiveKConfig(
                k_values=[2, 4, 6],
                entropy_thresholds=[1.0]  # Should be 2 thresholds
            )
            
    def test_custom_values(self):
        """Test custom config values."""
        config = AdaptiveKConfig(
            k_min=1,
            k_max=16,
            k_values=[1, 4, 8, 16],
            entropy_thresholds=[1.0, 1.5, 2.0]
        )
        assert config.k_min == 1
        assert config.k_max == 16
        assert len(config.k_values) == 4


class TestComputeSavings:
    """Tests for compute savings calculation."""
    
    def test_savings_with_sparse_logits(self):
        """Test realistic compute savings with sparse logits."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        
        # Simulate sparse logits (peaked distributions)
        torch.manual_seed(42)
        logits = torch.randn(1000, 8) * 3
        logits[:, 0] += 2  # Expert 0 preferred
        
        routing.apply(logits)
        stats = routing.get_stats()
        
        # Should achieve some savings with peaked distributions
        assert stats['compute_savings_pct'] > 0, "Expected compute savings with sparse logits"
        
    def test_no_savings_with_uniform(self):
        """Test no savings with uniform distribution."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2, 
            k_max=8,
            entropy_thresholds=[0.1, 0.2]  # Very low thresholds
        )
        
        # Uniform logits -> max entropy -> use k_max
        logits = torch.zeros(100, 8)
        
        routing.apply(logits)
        stats = routing.get_stats()
        
        # With uniform distribution and low thresholds, should use k_max
        assert stats['avg_k'] == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
