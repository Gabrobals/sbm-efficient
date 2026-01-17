"""
Tests for entropy computation.
"""

import pytest
import torch
from adaptive_speculative.entropy import (
    compute_entropy,
    entropy_to_k,
    batch_entropy_to_k,
    calibrate_thresholds,
)


class TestComputeEntropy:
    """Tests for compute_entropy function."""
    
    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution should have high entropy."""
        # Uniform logits -> max entropy = log(vocab_size)
        vocab_size = 1000
        logits = torch.zeros(1, vocab_size)
        entropy = compute_entropy(logits)
        
        expected_max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))
        assert torch.isclose(entropy, expected_max_entropy, atol=0.01)
    
    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution should have low entropy."""
        vocab_size = 1000
        logits = torch.zeros(1, vocab_size)
        logits[0, 0] = 100.0  # Very peaked
        
        entropy = compute_entropy(logits)
        assert entropy < 0.1  # Should be near zero
    
    def test_batch_processing(self):
        """Should handle batched inputs."""
        batch_size = 32
        vocab_size = 50257
        logits = torch.randn(batch_size, vocab_size)
        
        entropy = compute_entropy(logits)
        assert entropy.shape == (batch_size,)
    
    def test_numerical_stability(self):
        """Should not produce NaN or Inf."""
        # Test with extreme values
        logits = torch.tensor([[1e10, -1e10, 0.0, 0.0]])
        entropy = compute_entropy(logits)
        
        assert not torch.isnan(entropy).any()
        assert not torch.isinf(entropy).any()


class TestEntropyToK:
    """Tests for entropy_to_k function."""
    
    def test_scalar_low_entropy(self):
        """Low entropy should map to high K."""
        k = entropy_to_k(0.3)
        assert k == 16
    
    def test_scalar_medium_entropy(self):
        """Medium entropy should map to medium K."""
        k = entropy_to_k(0.7)
        assert k == 8
        
        k = entropy_to_k(1.5)
        assert k == 4
    
    def test_scalar_high_entropy(self):
        """High entropy should map to low K."""
        k = entropy_to_k(3.0)
        assert k == 1
    
    def test_boundary_values(self):
        """Test behavior at threshold boundaries."""
        thresholds = [0.5, 1.0, 2.0]
        k_values = [16, 8, 4, 1]
        
        # Just below threshold
        assert entropy_to_k(0.49, thresholds, k_values) == 16
        # At threshold
        assert entropy_to_k(0.5, thresholds, k_values) == 8
        # Just above threshold
        assert entropy_to_k(0.51, thresholds, k_values) == 8
    
    def test_tensor_input(self):
        """Should handle tensor inputs."""
        entropies = torch.tensor([0.3, 0.7, 1.5, 3.0])
        k_values = entropy_to_k(entropies)
        
        expected = torch.tensor([16, 8, 4, 1])
        assert torch.equal(k_values, expected)
    
    def test_custom_thresholds(self):
        """Should work with custom thresholds."""
        thresholds = [1.0, 2.0]
        k_values = [10, 5, 2]
        
        assert entropy_to_k(0.5, thresholds, k_values) == 10
        assert entropy_to_k(1.5, thresholds, k_values) == 5
        assert entropy_to_k(3.0, thresholds, k_values) == 2


class TestBatchEntropyToK:
    """Tests for batch_entropy_to_k function."""
    
    def test_returns_stats(self):
        """Should return statistics dict."""
        entropies = torch.tensor([0.3, 0.7, 1.5, 3.0])
        k_tensor, stats = batch_entropy_to_k(entropies)
        
        assert "mean_entropy" in stats
        assert "k_distribution" in stats
        assert len(stats["k_distribution"]) > 0
    
    def test_stats_accuracy(self):
        """Statistics should be accurate."""
        entropies = torch.tensor([1.0, 1.0, 1.0, 1.0])
        k_tensor, stats = batch_entropy_to_k(entropies)
        
        assert stats["mean_entropy"] == 1.0
        assert stats["std_entropy"] == 0.0


class TestCalibrateThresholds:
    """Tests for calibrate_thresholds function."""
    
    def test_uniform_target(self):
        """Uniform target should split at quantiles."""
        entropies = torch.arange(0, 100, dtype=torch.float32)
        thresholds = calibrate_thresholds(
            entropies,
            target_k_ratios=[0.25, 0.25, 0.25, 0.25],
            k_values=[16, 8, 4, 1],
        )
        
        assert len(thresholds) == 3
        # Should be approximately at 25, 50, 75
        assert 20 < thresholds[0] < 30
        assert 45 < thresholds[1] < 55
        assert 70 < thresholds[2] < 80
    
    def test_skewed_target(self):
        """Skewed target should have asymmetric thresholds."""
        entropies = torch.arange(0, 100, dtype=torch.float32)
        thresholds = calibrate_thresholds(
            entropies,
            target_k_ratios=[0.5, 0.3, 0.15, 0.05],
            k_values=[16, 8, 4, 1],
        )
        
        assert len(thresholds) == 3
        # First threshold should be higher (50% in first bucket)
        assert 45 < thresholds[0] < 55
    
    def test_invalid_ratios(self):
        """Should raise error for invalid ratios."""
        entropies = torch.arange(0, 100, dtype=torch.float32)
        
        with pytest.raises(AssertionError):
            calibrate_thresholds(
                entropies,
                target_k_ratios=[0.5, 0.5, 0.5],  # Sum > 1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
