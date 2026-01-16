"""
Adaptive-K Benchmark Suite

Comprehensive benchmarks for validating Adaptive-K performance:
- Accuracy benchmarks
- Latency profiling  
- Throughput measurement
- Entropy analysis
"""

import time
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    benchmark_name: str
    score: float
    metric: str
    baseline_score: Optional[float] = None
    degradation_pct: Optional[float] = None
    compute_saved_pct: float = 0
    avg_k: float = 0
    k_distribution: Dict[int, float] = None
    latency_ms: float = 0
    samples_processed: int = 0
    
    def __post_init__(self):
        if self.k_distribution is None:
            self.k_distribution = {}


@dataclass
class LatencyProfile:
    """Latency profiling results"""
    batch_size: int
    sequence_length: int
    baseline_latency_ms: float
    adaptive_latency_ms: float
    speedup: float
    latency_reduction_pct: float
    p50_latency_ms: float
    p99_latency_ms: float


@dataclass
class ThroughputProfile:
    """Throughput profiling results"""
    batch_size: int
    sequence_length: int
    baseline_tokens_per_sec: float
    adaptive_tokens_per_sec: float
    throughput_improvement_pct: float


class AdaptiveKBenchmark:
    """
    Main benchmark class for Adaptive-K evaluation.
    
    Usage:
        benchmark = AdaptiveKBenchmark(model, router)
        results = benchmark.run_accuracy_benchmark(dataset, "mnist")
        latency = benchmark.run_latency_benchmark(batch_sizes=[1, 4, 8, 16])
    """
    
    # Standard benchmark configurations
    BENCHMARKS = {
        # Vision benchmarks (our current focus)
        'mnist': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_full_k': 98.5
        },
        'fashion_mnist': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_full_k': 89.0
        },
        'cifar10': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_full_k': 85.0
        },
        
        # Language modeling (for future MoE integration)
        'wikitext2': {
            'metric': 'perplexity',
            'lower_is_better': True,
            'baseline_mixtral': 3.84
        },
        'wikitext103': {
            'metric': 'perplexity',
            'lower_is_better': True,
            'baseline_mixtral': 3.76
        },
        
        # Reasoning (for future MoE integration)
        'mmlu': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 70.6
        },
        'hellaswag': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 84.2
        }
    }
    
    def __init__(self, model, router=None, device: str = None):
        """
        Initialize benchmark suite.
        
        Args:
            model: The model to benchmark (SBMModel or similar)
            router: Optional separate router (if not part of model)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.router = router
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
    
    def run_accuracy_benchmark(
        self, 
        dataloader, 
        benchmark_name: str,
        use_adaptive_k: bool = True,
        baseline_score: float = None
    ) -> BenchmarkResult:
        """
        Run accuracy benchmark on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader with test data
            benchmark_name: Name of benchmark for reporting
            use_adaptive_k: Whether to use adaptive K selection
            baseline_score: Optional baseline to compare against
        
        Returns:
            BenchmarkResult with accuracy and compute metrics
        """
        self.model.eval()
        self.model.to(self.device)
        
        correct = 0
        total = 0
        total_k = 0
        k_counts = {}
        total_latency = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                start = time.perf_counter()
                
                # Forward pass - model should return output and metrics
                if hasattr(self.model, 'forward_with_metrics'):
                    output, metrics = self.model.forward_with_metrics(
                        data, use_adaptive_k=use_adaptive_k
                    )
                    batch_k = metrics.get('avg_k', 0)
                    batch_k_dist = metrics.get('k_distribution', {})
                else:
                    output = self.model(data)
                    batch_k = 0
                    batch_k_dist = {}
                
                total_latency += (time.perf_counter() - start) * 1000
                
                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Accumulate K statistics
                total_k += batch_k * target.size(0)
                for k, count in batch_k_dist.items():
                    k_counts[k] = k_counts.get(k, 0) + count
        
        # Compute final metrics
        accuracy = 100.0 * correct / total
        avg_k = total_k / total if total > 0 else 0
        
        # Normalize K distribution to percentages
        total_k_samples = sum(k_counts.values()) if k_counts else 1
        k_distribution = {k: v / total_k_samples * 100 for k, v in k_counts.items()}
        
        # Get baseline from config if not provided
        if baseline_score is None and benchmark_name in self.BENCHMARKS:
            baseline_key = [k for k in self.BENCHMARKS[benchmark_name].keys() 
                          if k.startswith('baseline_')]
            if baseline_key:
                baseline_score = self.BENCHMARKS[benchmark_name][baseline_key[0]]
        
        # Compute degradation
        degradation = None
        if baseline_score is not None:
            benchmark_info = self.BENCHMARKS.get(benchmark_name, {})
            lower_is_better = benchmark_info.get('lower_is_better', False)
            
            if lower_is_better:
                degradation = (accuracy - baseline_score) / baseline_score * 100
            else:
                degradation = (baseline_score - accuracy) / baseline_score * 100
        
        # Estimate compute saved (assuming 8 experts baseline)
        max_k = 8  # Typical MoE has 8 experts
        compute_saved = (1 - avg_k / max_k) * 100 if avg_k > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            score=accuracy,
            metric='accuracy',
            baseline_score=baseline_score,
            degradation_pct=degradation,
            compute_saved_pct=compute_saved,
            avg_k=avg_k,
            k_distribution=k_distribution,
            latency_ms=total_latency,
            samples_processed=total
        )
    
    def run_latency_benchmark(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        warmup: int = None,
        iterations: int = None
    ) -> List[LatencyProfile]:
        """
        Profile latency across different batch sizes.
        
        Args:
            input_shape: Shape of input tensor (excluding batch dim)
            batch_sizes: List of batch sizes to test
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
        
        Returns:
            List of LatencyProfile results
        """
        warmup = warmup or self.warmup_iterations
        iterations = iterations or self.benchmark_iterations
        
        self.model.eval()
        self.model.to(self.device)
        
        results = []
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = self.model(dummy_input)
            
            # Benchmark without Adaptive-K (if model supports it)
            baseline_latencies = []
            if hasattr(self.model, 'set_adaptive_k'):
                self.model.set_adaptive_k(False)
            
            with torch.no_grad():
                for _ in range(iterations):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    _ = self.model(dummy_input)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    baseline_latencies.append((time.perf_counter() - start) * 1000)
            
            # Benchmark with Adaptive-K
            adaptive_latencies = []
            if hasattr(self.model, 'set_adaptive_k'):
                self.model.set_adaptive_k(True)
            
            with torch.no_grad():
                for _ in range(iterations):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    _ = self.model(dummy_input)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    adaptive_latencies.append((time.perf_counter() - start) * 1000)
            
            # Compute statistics
            baseline_mean = np.mean(baseline_latencies)
            adaptive_mean = np.mean(adaptive_latencies)
            
            results.append(LatencyProfile(
                batch_size=batch_size,
                sequence_length=input_shape[0] if len(input_shape) > 0 else 1,
                baseline_latency_ms=baseline_mean,
                adaptive_latency_ms=adaptive_mean,
                speedup=baseline_mean / adaptive_mean if adaptive_mean > 0 else 1,
                latency_reduction_pct=(1 - adaptive_mean / baseline_mean) * 100 if baseline_mean > 0 else 0,
                p50_latency_ms=np.percentile(adaptive_latencies, 50),
                p99_latency_ms=np.percentile(adaptive_latencies, 99)
            ))
        
        return results
    
    def run_throughput_benchmark(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 32,
        sequence_lengths: List[int] = None,
        duration_seconds: float = 10.0
    ) -> List[ThroughputProfile]:
        """
        Measure throughput (samples/second) across different configs.
        
        Args:
            input_shape: Base input shape
            batch_size: Batch size to use
            sequence_lengths: List of sequence lengths (for variable-length inputs)
            duration_seconds: How long to run each test
        
        Returns:
            List of ThroughputProfile results
        """
        if sequence_lengths is None:
            sequence_lengths = [input_shape[0]] if len(input_shape) > 0 else [1]
        
        self.model.eval()
        self.model.to(self.device)
        
        results = []
        
        for seq_len in sequence_lengths:
            # Adjust input shape for sequence length
            if len(input_shape) > 0:
                adjusted_shape = (seq_len,) + input_shape[1:]
            else:
                adjusted_shape = input_shape
            
            dummy_input = torch.randn(batch_size, *adjusted_shape, device=self.device)
            
            # Measure baseline throughput
            if hasattr(self.model, 'set_adaptive_k'):
                self.model.set_adaptive_k(False)
            
            baseline_samples = self._measure_throughput(dummy_input, duration_seconds)
            
            # Measure adaptive throughput
            if hasattr(self.model, 'set_adaptive_k'):
                self.model.set_adaptive_k(True)
            
            adaptive_samples = self._measure_throughput(dummy_input, duration_seconds)
            
            baseline_tps = baseline_samples * batch_size / duration_seconds
            adaptive_tps = adaptive_samples * batch_size / duration_seconds
            
            results.append(ThroughputProfile(
                batch_size=batch_size,
                sequence_length=seq_len,
                baseline_tokens_per_sec=baseline_tps,
                adaptive_tokens_per_sec=adaptive_tps,
                throughput_improvement_pct=(adaptive_tps / baseline_tps - 1) * 100 if baseline_tps > 0 else 0
            ))
        
        return results
    
    def _measure_throughput(self, input_tensor: torch.Tensor, duration: float) -> int:
        """Measure how many forward passes in given duration"""
        count = 0
        start = time.perf_counter()
        
        with torch.no_grad():
            while time.perf_counter() - start < duration:
                _ = self.model(input_tensor)
                count += 1
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        return count
    
    def profile_entropy_distribution(
        self,
        dataloader,
        num_batches: int = 100
    ) -> Dict[str, Any]:
        """
        Profile the entropy distribution of the router.
        
        Args:
            dataloader: DataLoader with input data
            num_batches: Number of batches to profile
        
        Returns:
            Dict with entropy statistics and K distribution
        """
        self.model.eval()
        self.model.to(self.device)
        
        all_entropies = []
        k_selections = []
        
        # Get thresholds from model/router
        h_thresholds = getattr(self.model, 'h_thresholds', [0.6, 1.2])
        k_values = getattr(self.model, 'k_values', [1, 2, 4])
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                data = data.to(self.device)
                
                # Get router entropy if available
                if hasattr(self.model, 'get_router_entropy'):
                    entropy = self.model.get_router_entropy(data)
                    all_entropies.extend(entropy.cpu().tolist())
                    
                    # Compute K selection for each entropy
                    for h in entropy.cpu().tolist():
                        k = k_values[-1]  # Default to max
                        for i, threshold in enumerate(h_thresholds):
                            if h < threshold:
                                k = k_values[i]
                                break
                        k_selections.append(k)
        
        if not all_entropies:
            return {'error': 'Model does not support entropy profiling'}
        
        # Compute statistics
        entropies = np.array(all_entropies)
        
        return {
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
                'percentiles': {
                    '25': float(np.percentile(entropies, 25)),
                    '50': float(np.percentile(entropies, 50)),
                    '75': float(np.percentile(entropies, 75)),
                    '90': float(np.percentile(entropies, 90)),
                    '95': float(np.percentile(entropies, 95))
                }
            },
            'k_distribution': {
                f'k={k}': k_selections.count(k) / len(k_selections) * 100 
                for k in set(k_selections)
            } if k_selections else {},
            'effective_k': float(np.mean(k_selections)) if k_selections else 0,
            'theoretical_compute_saving': float((1 - np.mean(k_selections) / max(k_values)) * 100) if k_selections else 0,
            'thresholds_used': h_thresholds,
            'k_values': k_values,
            'samples_profiled': len(all_entropies)
        }
    
    def run_full_benchmark(
        self,
        dataloader,
        benchmark_name: str,
        input_shape: Tuple[int, ...],
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            dataloader: Test DataLoader
            benchmark_name: Name for reporting
            input_shape: Input tensor shape
            output_path: Optional path to save results
        
        Returns:
            Dict with all benchmark results
        """
        results = {
            'benchmark_name': benchmark_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device)
        }
        
        # Accuracy benchmark
        print(f"Running accuracy benchmark on {benchmark_name}...")
        accuracy_result = self.run_accuracy_benchmark(dataloader, benchmark_name)
        results['accuracy'] = asdict(accuracy_result)
        
        # Latency benchmark
        print("Running latency benchmark...")
        latency_results = self.run_latency_benchmark(input_shape)
        results['latency'] = [asdict(lr) for lr in latency_results]
        
        # Entropy profiling
        print("Running entropy profiling...")
        entropy_profile = self.profile_entropy_distribution(dataloader)
        results['entropy_profile'] = entropy_profile
        
        # Summary
        results['summary'] = {
            'accuracy': accuracy_result.score,
            'accuracy_degradation_pct': accuracy_result.degradation_pct,
            'compute_saved_pct': accuracy_result.compute_saved_pct,
            'avg_k': accuracy_result.avg_k,
            'latency_speedup': latency_results[0].speedup if latency_results else 0,
            'effective_k': entropy_profile.get('effective_k', 0)
        }
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return results


def print_benchmark_summary(results: Dict[str, Any]):
    """Pretty print benchmark summary"""
    print("\n" + "=" * 60)
    print("ADAPTIVE-K BENCHMARK SUMMARY")
    print("=" * 60)
    
    summary = results.get('summary', {})
    
    print(f"Benchmark: {results.get('benchmark_name', 'Unknown')}")
    print(f"Device: {results.get('device', 'Unknown')}")
    print("-" * 60)
    
    print(f"Accuracy: {summary.get('accuracy', 0):.2f}%")
    if summary.get('accuracy_degradation_pct') is not None:
        print(f"  vs Baseline: {summary['accuracy_degradation_pct']:+.2f}%")
    
    print(f"\nCompute Saved: {summary.get('compute_saved_pct', 0):.1f}%")
    print(f"Average K: {summary.get('avg_k', 0):.2f}")
    print(f"Effective K: {summary.get('effective_k', 0):.2f}")
    
    if summary.get('latency_speedup'):
        print(f"\nLatency Speedup: {summary['latency_speedup']:.2f}x")
    
    # K distribution
    entropy = results.get('entropy_profile', {})
    k_dist = entropy.get('k_distribution', {})
    if k_dist:
        print(f"\nK Distribution:")
        for k, pct in sorted(k_dist.items()):
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {k}: {bar} {pct:.1f}%")
    
    print("=" * 60)
