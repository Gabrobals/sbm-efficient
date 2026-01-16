"""
Adaptive-K Observability Module

Provides tracing, structured logging, and metrics for production deployments.
"""

import time
import json
import logging
from functools import wraps
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading

# Try importing optional dependencies
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InferenceTrace:
    """Single inference trace record"""
    request_id: str
    timestamp: float
    latency_ms: float
    avg_k: float
    compute_saved_pct: float
    k_distribution: Dict[int, int]
    layer_entropies: List[float] = field(default_factory=list)
    used_fallback: bool = False
    error: Optional[str] = None


@dataclass 
class LayerTrace:
    """Per-layer trace for detailed debugging"""
    layer_idx: int
    entropy_mean: float
    entropy_std: float
    k_selected: int
    experts_used: int
    latency_ms: float


# =============================================================================
# Metrics Collector (No external dependencies)
# =============================================================================

class MetricsCollector:
    """
    Lightweight metrics collector without external dependencies.
    Can export to Prometheus format if prometheus_client is available.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._lock = threading.Lock()
        
        # Internal storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        
        # Prometheus metrics (if available)
        self._prom_metrics = {}
        if HAS_PROMETHEUS:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self._prom_metrics['inferences_total'] = Counter(
            'adaptive_k_inferences_total',
            'Total number of inferences',
            ['mode']
        )
        
        self._prom_metrics['latency'] = Histogram(
            'adaptive_k_latency_seconds',
            'Inference latency in seconds',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self._prom_metrics['avg_k'] = Histogram(
            'adaptive_k_avg_k',
            'Average K per inference',
            buckets=[1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]
        )
        
        self._prom_metrics['compute_saved'] = Histogram(
            'adaptive_k_compute_saved_ratio',
            'Ratio of compute saved (0-1)',
            buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )
        
        self._prom_metrics['fallback_total'] = Counter(
            'adaptive_k_fallback_total',
            'Number of fallbacks to full-K'
        )
        
        self._prom_metrics['errors_total'] = Counter(
            'adaptive_k_errors_total',
            'Number of errors',
            ['error_type']
        )
    
    def inc_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter"""
        with self._lock:
            key = f"{name}_{labels}" if labels else name
            self._counters[key] += value
            
            if HAS_PROMETHEUS and name in self._prom_metrics:
                if labels:
                    self._prom_metrics[name].labels(**labels).inc(value)
                else:
                    self._prom_metrics[name].inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram observation"""
        with self._lock:
            key = f"{name}_{labels}" if labels else name
            self._histograms[key].append(value)
            
            if HAS_PROMETHEUS and name in self._prom_metrics:
                if labels:
                    self._prom_metrics[name].labels(**labels).observe(value)
                else:
                    self._prom_metrics[name].observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value"""
        with self._lock:
            key = f"{name}_{labels}" if labels else name
            self._gauges[key] = value
    
    def record_inference(self, trace: InferenceTrace):
        """Record a complete inference"""
        mode = 'fallback' if trace.used_fallback else 'adaptive'
        
        self.inc_counter('inferences_total', labels={'mode': mode})
        self.observe_histogram('latency', trace.latency_ms / 1000)  # Convert to seconds
        self.observe_histogram('avg_k', trace.avg_k)
        self.observe_histogram('compute_saved', trace.compute_saved_pct / 100)
        
        if trace.used_fallback:
            self.inc_counter('fallback_total')
        
        if trace.error:
            self.inc_counter('errors_total', labels={'error_type': type(trace.error).__name__})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            latencies = self._histograms.get('latency', [])
            avg_ks = self._histograms.get('avg_k', [])
            compute_saved = self._histograms.get('compute_saved', [])
            
            return {
                'total_inferences': sum(v for k, v in self._counters.items() 
                                       if 'inferences_total' in k),
                'latency': {
                    'mean': sum(latencies) / len(latencies) if latencies else 0,
                    'p50': sorted(latencies)[len(latencies)//2] if latencies else 0,
                    'p99': sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
                } if latencies else {},
                'avg_k': {
                    'mean': sum(avg_ks) / len(avg_ks) if avg_ks else 0,
                },
                'compute_saved': {
                    'mean': sum(compute_saved) / len(compute_saved) if compute_saved else 0,
                },
                'fallback_rate': self._counters.get('fallback_total', 0) / 
                                max(1, sum(v for k, v in self._counters.items() 
                                          if 'inferences_total' in k))
            }
    
    def start_http_server(self, port: int = 9090):
        """Start Prometheus HTTP server"""
        if HAS_PROMETHEUS:
            start_http_server(port)
            logging.info(f"Prometheus metrics server started on port {port}")
        else:
            logging.warning("prometheus_client not installed, HTTP server not started")
    
    def reset(self):
        """Reset all metrics (for testing)"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# =============================================================================
# Structured Logger
# =============================================================================

class AdaptiveKLogger:
    """
    Structured logger for Adaptive-K.
    Uses structlog if available, falls back to standard logging.
    """
    
    def __init__(self, component: str = "adaptive_k", json_format: bool = True):
        self.component = component
        self.json_format = json_format
        
        if HAS_STRUCTLOG:
            self._configure_structlog()
            self.logger = structlog.get_logger(component)
        else:
            self._configure_stdlib()
            self.logger = logging.getLogger(component)
    
    def _configure_structlog(self):
        """Configure structlog"""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]
        
        if self.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _configure_stdlib(self):
        """Configure standard library logging"""
        handler = logging.StreamHandler()
        
        if self.json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"component": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        
        handler.setFormatter(formatter)
        
        logger = logging.getLogger(self.component)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def _log(self, level: str, event: str, **kwargs):
        """Internal log method"""
        if HAS_STRUCTLOG:
            getattr(self.logger, level)(event, **kwargs)
        else:
            msg = f"{event} | {json.dumps(kwargs)}" if kwargs else event
            getattr(self.logger, level)(msg)
    
    def info(self, event: str, **kwargs):
        self._log('info', event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        self._log('debug', event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        self._log('warning', event, **kwargs)
    
    def error(self, event: str, **kwargs):
        self._log('error', event, **kwargs)
    
    def log_inference(self, trace: InferenceTrace):
        """Log inference completion"""
        self.info(
            "inference_completed",
            request_id=trace.request_id,
            latency_ms=trace.latency_ms,
            avg_k=trace.avg_k,
            compute_saved_pct=trace.compute_saved_pct,
            used_fallback=trace.used_fallback
        )
    
    def log_layer(self, layer_trace: LayerTrace):
        """Log layer execution"""
        self.debug(
            "layer_executed",
            layer_idx=layer_trace.layer_idx,
            entropy_mean=layer_trace.entropy_mean,
            k_selected=layer_trace.k_selected,
            experts_used=layer_trace.experts_used
        )
    
    def log_fallback(self, request_id: str, reason: str):
        """Log fallback to full-K"""
        self.warning(
            "fallback_to_full_k",
            request_id=request_id,
            reason=reason
        )
    
    def log_threshold_update(self, old: List[float], new: List[float]):
        """Log threshold update"""
        self.info(
            "thresholds_updated",
            old_thresholds=old,
            new_thresholds=new
        )


# =============================================================================
# Tracer (Decorator-based)
# =============================================================================

class AdaptiveKTracer:
    """
    Tracing for Adaptive-K inference.
    Provides decorators and context managers for instrumentation.
    """
    
    def __init__(self, enable_detailed: bool = False):
        self.enable_detailed = enable_detailed
        self.metrics = MetricsCollector()
        self.logger = AdaptiveKLogger()
        self._current_trace: Optional[InferenceTrace] = None
        self._layer_traces: List[LayerTrace] = []
    
    def trace_inference(self, func: Callable) -> Callable:
        """Decorator to trace inference function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import uuid
            request_id = str(uuid.uuid4())[:8]
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                # Extract metrics from result if available
                avg_k = result.get('avg_k', 0) if isinstance(result, dict) else 0
                compute_saved = result.get('compute_saved', 0) if isinstance(result, dict) else 0
                k_dist = result.get('k_distribution', {}) if isinstance(result, dict) else {}
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                trace = InferenceTrace(
                    request_id=request_id,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    avg_k=avg_k,
                    compute_saved_pct=compute_saved,
                    k_distribution=k_dist,
                    layer_entropies=[lt.entropy_mean for lt in self._layer_traces],
                    used_fallback=False
                )
                
                self.metrics.record_inference(trace)
                self.logger.log_inference(trace)
                self._layer_traces.clear()
                
                return result
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                trace = InferenceTrace(
                    request_id=request_id,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    avg_k=0,
                    compute_saved_pct=0,
                    k_distribution={},
                    error=str(e)
                )
                
                self.metrics.record_inference(trace)
                self.logger.error("inference_failed", request_id=request_id, error=str(e))
                raise
        
        return wrapper
    
    def trace_layer(self, layer_idx: int):
        """Context manager for tracing a layer"""
        return LayerTraceContext(self, layer_idx)
    
    def record_layer(self, layer_trace: LayerTrace):
        """Record a layer trace"""
        self._layer_traces.append(layer_trace)
        if self.enable_detailed:
            self.logger.log_layer(layer_trace)


class LayerTraceContext:
    """Context manager for layer tracing"""
    
    def __init__(self, tracer: AdaptiveKTracer, layer_idx: int):
        self.tracer = tracer
        self.layer_idx = layer_idx
        self.start_time = None
        self.entropy_mean = 0
        self.entropy_std = 0
        self.k_selected = 0
        self.experts_used = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def set_metrics(self, entropy_mean: float, entropy_std: float, 
                    k_selected: int, experts_used: int):
        """Set layer metrics before exit"""
        self.entropy_mean = entropy_mean
        self.entropy_std = entropy_std
        self.k_selected = k_selected
        self.experts_used = experts_used
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        
        trace = LayerTrace(
            layer_idx=self.layer_idx,
            entropy_mean=self.entropy_mean,
            entropy_std=self.entropy_std,
            k_selected=self.k_selected,
            experts_used=self.experts_used,
            latency_ms=latency_ms
        )
        
        self.tracer.record_layer(trace)
        return False


# =============================================================================
# Debug Tools
# =============================================================================

class AdaptiveKDebugger:
    """
    Debug tools for Adaptive-K development and troubleshooting.
    """
    
    def __init__(self):
        self.trace_history: List[Dict] = []
        self.verbose = False
    
    def enable_verbose(self):
        """Enable verbose debug output"""
        self.verbose = True
        logging.getLogger("adaptive_k").setLevel(logging.DEBUG)
    
    def trace_k_selection(self, entropies: List[float], thresholds: List[float], 
                          k_values: List[int]) -> Dict:
        """
        Trace K selection for debugging.
        
        Args:
            entropies: List of entropy values
            thresholds: Entropy thresholds [h1, h2, ...]
            k_values: K values [k1, k2, k3, ...]
        
        Returns:
            Debug trace dict
        """
        k_selections = []
        
        for h in entropies:
            k = k_values[-1]  # Default to max K
            for i, threshold in enumerate(thresholds):
                if h < threshold:
                    k = k_values[i]
                    break
            k_selections.append(k)
        
        trace = {
            'entropies': entropies,
            'thresholds': thresholds,
            'k_values': k_values,
            'k_selections': k_selections,
            'k_distribution': {k: k_selections.count(k) for k in set(k_selections)},
            'avg_k': sum(k_selections) / len(k_selections) if k_selections else 0,
            'entropy_stats': {
                'min': min(entropies) if entropies else 0,
                'max': max(entropies) if entropies else 0,
                'mean': sum(entropies) / len(entropies) if entropies else 0
            }
        }
        
        self.trace_history.append(trace)
        
        if self.verbose:
            self._print_trace(trace)
        
        return trace
    
    def _print_trace(self, trace: Dict):
        """Print trace in human-readable format"""
        print("=" * 60)
        print("ADAPTIVE-K DEBUG TRACE")
        print("=" * 60)
        print(f"Thresholds: {trace['thresholds']}")
        print(f"K values: {trace['k_values']}")
        print(f"Average K: {trace['avg_k']:.2f}")
        print(f"K distribution: {trace['k_distribution']}")
        print(f"Entropy range: [{trace['entropy_stats']['min']:.3f}, "
              f"{trace['entropy_stats']['max']:.3f}]")
        print("=" * 60)
    
    def visualize_layer_trace(self, layer_traces: List[LayerTrace]):
        """
        Visualize layer-by-layer behavior.
        """
        print("=" * 60)
        print("LAYER-BY-LAYER TRACE")
        print("=" * 60)
        
        total_experts = 0
        total_available = 0
        
        for lt in layer_traces:
            # Simple ASCII bar for entropy
            bar_len = int(lt.entropy_mean * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            print(f"Layer {lt.layer_idx:2d} | "
                  f"H={lt.entropy_mean:.3f} {bar} | "
                  f"K={lt.k_selected} | "
                  f"Exp={lt.experts_used} | "
                  f"{lt.latency_ms:.1f}ms")
            
            total_experts += lt.experts_used
            # Assume 8 experts per layer for visualization
            total_available += 8
        
        compute_saved = 1 - (total_experts / total_available) if total_available else 0
        print("=" * 60)
        print(f"Total compute saved: {compute_saved*100:.1f}%")
        print("=" * 60)
    
    def export_traces(self, filepath: str):
        """Export trace history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.trace_history, f, indent=2)
        print(f"Traces exported to {filepath}")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_metrics() -> MetricsCollector:
    """Get singleton metrics collector"""
    return MetricsCollector()


def get_logger(component: str = "adaptive_k") -> AdaptiveKLogger:
    """Get a logger instance"""
    return AdaptiveKLogger(component)


def get_tracer(enable_detailed: bool = False) -> AdaptiveKTracer:
    """Get a tracer instance"""
    return AdaptiveKTracer(enable_detailed)


def get_debugger() -> AdaptiveKDebugger:
    """Get a debugger instance"""
    return AdaptiveKDebugger()
