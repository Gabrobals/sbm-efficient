#!/usr/bin/env python
"""
Adaptive-K Production Demo

Demonstrates Adaptive-K capabilities:
1. Run benchmark on MNIST
2. Show observability features
3. Profile entropy distribution
4. Visualize K selection behavior

Usage:
    python scripts/demo_adaptive_k.py [--config CONFIG] [--verbose]
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.common.device import get_device
from src.common.seed import set_seed
from src.data.loaders import get_data_loaders
from src.models.sbm_model import SBMAdaptiveKModel
from src.profiling import (
    AdaptiveKBenchmark,
    get_metrics,
    get_logger,
    get_tracer,
    get_debugger,
    print_benchmark_summary,
)


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def demo_observability():
    """Demonstrate observability features"""
    print("\n" + "=" * 60)
    print("DEMO: Observability Features")
    print("=" * 60)
    
    # Get logger
    logger = get_logger("demo")
    logger.info("demo_started", feature="observability")
    
    # Get metrics collector
    metrics = get_metrics()
    
    # Simulate some inference traces
    from src.profiling.observability import InferenceTrace
    
    for i in range(10):
        trace = InferenceTrace(
            request_id=f"req_{i:03d}",
            timestamp=time.time(),
            latency_ms=50 + i * 5,
            avg_k=1.5 + (i % 3) * 0.5,
            compute_saved_pct=40 + (i % 4) * 5,
            k_distribution={1: 40, 2: 40, 4: 20},
            used_fallback=(i == 7)  # One fallback
        )
        metrics.record_inference(trace)
        logger.log_inference(trace)
    
    # Print summary
    summary = metrics.get_summary()
    print("\nMetrics Summary:")
    print(f"  Total inferences: {summary['total_inferences']}")
    print(f"  Mean latency: {summary['latency'].get('mean', 0)*1000:.1f}ms")
    print(f"  Mean K: {summary['avg_k'].get('mean', 0):.2f}")
    print(f"  Fallback rate: {summary['fallback_rate']*100:.1f}%")
    
    logger.info("demo_completed", feature="observability")


def demo_debugger():
    """Demonstrate debug tools"""
    print("\n" + "=" * 60)
    print("DEMO: Debug Tools")
    print("=" * 60)
    
    debugger = get_debugger()
    debugger.enable_verbose()
    
    # Simulate entropy values from different inputs
    import random
    random.seed(42)
    
    # Low entropy (confident) -> K=1
    low_entropy = [random.uniform(0.1, 0.5) for _ in range(20)]
    
    # Medium entropy -> K=2
    medium_entropy = [random.uniform(0.7, 1.0) for _ in range(15)]
    
    # High entropy (uncertain) -> K=4
    high_entropy = [random.uniform(1.3, 2.0) for _ in range(10)]
    
    all_entropies = low_entropy + medium_entropy + high_entropy
    random.shuffle(all_entropies)
    
    # Trace K selection
    trace = debugger.trace_k_selection(
        entropies=all_entropies,
        thresholds=[0.6, 1.2],
        k_values=[1, 2, 4]
    )
    
    print(f"\nProcessed {len(all_entropies)} samples")
    print(f"Theoretical compute saving: {(1 - trace['avg_k']/8)*100:.1f}%")


def demo_benchmark(config_path: str, verbose: bool = False):
    """Run benchmark demo"""
    print("\n" + "=" * 60)
    print("DEMO: Benchmark Suite")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    
    # Setup
    device = get_device()
    set_seed(config['run']['seed'])
    
    print(f"Device: {device}")
    print(f"Task: {config['run']['task']}")
    print(f"Model: {config['run']['model']}")
    
    # Get data
    _, test_loader = get_data_loaders(
        task=config['run']['task'],
        batch_size=config['data']['batch_size']
    )
    
    # Create model
    task = config['run']['task']
    
    # Get task-specific dimensions
    task_dims = {
        'mnist': {'feature_dim': 256, 'num_classes': 10},
        'fashion_mnist': {'feature_dim': 256, 'num_classes': 10},
        'cifar10': {'feature_dim': 512, 'num_classes': 10},
        'xor': {'feature_dim': 64, 'num_classes': 2},
    }
    dims = task_dims.get(task, {'feature_dim': 256, 'num_classes': 10})
    
    model = SBMAdaptiveKModel(
        task=task,
        num_experts=config['sbm']['experts_num'],
        routing_type='sbm',
        k_values=config['adaptive_k']['k_values'],
        h_thresholds=config['adaptive_k']['h_thresholds'],
        feature_dim=dims['feature_dim'],
        expert_hidden_dim=dims['feature_dim'],
        num_classes=dims['num_classes'],
        seed=config['run']['seed']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run benchmark
    benchmark = AdaptiveKBenchmark(model, device=str(device))
    
    # Quick accuracy test (subset)
    print("\nRunning accuracy benchmark (subset)...")
    
    # Create subset loader for demo
    from torch.utils.data import Subset, DataLoader
    
    subset_size = min(1000, len(test_loader.dataset))
    subset = Subset(test_loader.dataset, range(subset_size))
    subset_loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    result = benchmark.run_accuracy_benchmark(
        subset_loader,
        benchmark_name=config['run']['task'],
        use_adaptive_k=True
    )
    
    print(f"\nAccuracy: {result.score:.2f}%")
    print(f"Average K: {result.avg_k:.2f}")
    print(f"Compute saved: {result.compute_saved_pct:.1f}%")
    
    # Entropy profiling
    print("\nProfiling entropy distribution...")
    entropy_profile = benchmark.profile_entropy_distribution(subset_loader, num_batches=20)
    
    if 'error' not in entropy_profile:
        print(f"Mean entropy: {entropy_profile['entropy_stats']['mean']:.3f}")
        print(f"Effective K: {entropy_profile['effective_k']:.2f}")
        print("K distribution:")
        for k, pct in entropy_profile.get('k_distribution', {}).items():
            bar = "█" * int(pct / 5)
            print(f"  {k}: {bar} {pct:.1f}%")
    
    # Latency test
    if verbose:
        print("\nRunning latency benchmark...")
        # Input shape depends on task
        input_shapes = {
            'mnist': (1, 28, 28),
            'fashion_mnist': (1, 28, 28),
            'cifar10': (3, 32, 32),
            'xor': (2,)
        }
        input_shape = input_shapes.get(task, (1, 28, 28))
        latency_results = benchmark.run_latency_benchmark(
            input_shape,
            batch_sizes=[1, 8, 32],
            warmup=5,
            iterations=20
        )
        
        print("\nLatency Results:")
        print(f"{'Batch':<10} {'Baseline':<15} {'Adaptive':<15} {'Speedup':<10}")
        print("-" * 50)
        for lr in latency_results:
            print(f"{lr.batch_size:<10} {lr.baseline_latency_ms:>10.2f}ms   "
                  f"{lr.adaptive_latency_ms:>10.2f}ms   {lr.speedup:.2f}x")


def demo_production_simulation():
    """Simulate production deployment"""
    print("\n" + "=" * 60)
    print("DEMO: Production Simulation")
    print("=" * 60)
    
    tracer = get_tracer(enable_detailed=True)
    logger = get_logger("production")
    metrics = get_metrics()
    
    # Reset metrics for clean demo
    metrics.reset()
    
    # Simulate production traffic
    import random
    random.seed(42)
    
    num_requests = 50
    print(f"\nSimulating {num_requests} production requests...")
    
    for i in range(num_requests):
        # Simulate varying input complexity
        complexity = random.choice(['easy', 'medium', 'hard'])
        
        # Different K based on complexity
        if complexity == 'easy':
            avg_k = random.uniform(1.0, 1.5)
            latency = random.uniform(20, 40)
        elif complexity == 'medium':
            avg_k = random.uniform(1.8, 2.5)
            latency = random.uniform(40, 70)
        else:
            avg_k = random.uniform(3.0, 4.0)
            latency = random.uniform(60, 100)
        
        # Occasional fallback
        use_fallback = random.random() < 0.02  # 2% fallback rate
        if use_fallback:
            avg_k = 8
            latency *= 1.5
        
        from src.profiling.observability import InferenceTrace
        trace = InferenceTrace(
            request_id=f"prod_{i:04d}",
            timestamp=time.time(),
            latency_ms=latency,
            avg_k=avg_k,
            compute_saved_pct=(1 - avg_k/8) * 100,
            k_distribution={},
            used_fallback=use_fallback
        )
        
        metrics.record_inference(trace)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_requests} requests")
    
    # Final summary
    summary = metrics.get_summary()
    
    print("\n" + "-" * 40)
    print("Production Metrics Summary")
    print("-" * 40)
    print(f"Total requests: {summary['total_inferences']}")
    print(f"P50 latency: {summary['latency'].get('p50', 0)*1000:.1f}ms")
    print(f"P99 latency: {summary['latency'].get('p99', 0)*1000:.1f}ms")
    print(f"Avg compute saved: {summary['compute_saved'].get('mean', 0)*100:.1f}%")
    print(f"Fallback rate: {summary['fallback_rate']*100:.1f}%")
    
    # Cost estimation
    gpu_cost_per_hour = 2.0  # $/hour for A100
    avg_latency_sec = summary['latency'].get('mean', 0.05)
    baseline_latency = avg_latency_sec / (1 - summary['compute_saved'].get('mean', 0.3))
    
    hours_saved = (baseline_latency - avg_latency_sec) * summary['total_inferences'] / 3600
    cost_saved = hours_saved * gpu_cost_per_hour
    
    print(f"\nEstimated cost savings: ${cost_saved:.4f} for {summary['total_inferences']} requests")
    print(f"  Extrapolated to 1M requests: ${cost_saved * 1000000 / summary['total_inferences']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive-K Production Demo")
    parser.add_argument('--config', type=str, 
                       default='configs/sbm_adaptive_k_mnist.yaml',
                       help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'observability', 'debug', 'benchmark', 'production'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("       ADAPTIVE-K PRODUCTION DEMO")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Demo: {args.demo}")
    
    demos = {
        'observability': demo_observability,
        'debug': demo_debugger,
        'benchmark': lambda: demo_benchmark(args.config, args.verbose),
        'production': demo_production_simulation,
    }
    
    if args.demo == 'all':
        for name, func in demos.items():
            try:
                func()
            except Exception as e:
                print(f"\n[WARNING] Demo '{name}' failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    else:
        demos[args.demo]()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
