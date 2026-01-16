"""
Profiling utilities: FLOPs, latency, CUDA events, observability, benchmarks.
"""

from .observability import (
    MetricsCollector,
    AdaptiveKLogger,
    AdaptiveKTracer,
    AdaptiveKDebugger,
    InferenceTrace,
    LayerTrace,
    get_metrics,
    get_logger,
    get_tracer,
    get_debugger,
)

from .benchmark import (
    AdaptiveKBenchmark,
    BenchmarkResult,
    LatencyProfile,
    ThroughputProfile,
    print_benchmark_summary,
)

__all__ = [
    # Observability
    'MetricsCollector',
    'AdaptiveKLogger', 
    'AdaptiveKTracer',
    'AdaptiveKDebugger',
    'InferenceTrace',
    'LayerTrace',
    'get_metrics',
    'get_logger',
    'get_tracer',
    'get_debugger',
    # Benchmarks
    'AdaptiveKBenchmark',
    'BenchmarkResult',
    'LatencyProfile',
    'ThroughputProfile',
    'print_benchmark_summary',
]
