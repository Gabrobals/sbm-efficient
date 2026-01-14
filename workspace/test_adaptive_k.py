"""Test Adaptive-K routing with realistic sparse logits."""
import torch
import sys
sys.path.insert(0, 'C:/Users/ottic/Desktop/SBM Efficent/workspace')
from adaptive_k_routing_trtllm import create_adaptive_k_routing

print('Testing with realistic (sparse) router logits...')
routing = create_adaptive_k_routing(k_min=2, k_max=8)

torch.manual_seed(42)
# Simulate sparse logits (some experts much more preferred)
base_logits = torch.randn(1000, 8)  # 1000 tokens, 8 experts
# Make some logits stand out (sparse routing)
sparse_logits = base_logits * 3  # Higher variance = more peaked distributions
sparse_logits[:, 0] += 2  # Expert 0 often preferred
sparse_logits[:, 1] += 1.5  # Expert 1 sometimes preferred

experts, weights = routing.apply(sparse_logits)
stats = routing.get_stats()

print(f'Expert indices shape: {experts.shape}')
print(f'Weights shape: {weights.shape}')
print()
print(f'K Distribution: {stats["k_distribution"]}')
print(f'Average K: {stats["avg_k"]:.2f} vs Baseline K: {stats["baseline_k"]}')
print(f'Mean Entropy: {stats["mean_entropy"]:.3f}')
print()
print(f'** Compute Savings: {stats["compute_savings_pct"]:.1f}% **')
