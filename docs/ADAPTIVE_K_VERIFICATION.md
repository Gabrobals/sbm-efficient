# Adaptive-K: Verification, Benchmarks & Production Patterns

> Documento tecnico per dimostrare le capacità del sistema Adaptive-K
> Ultimo aggiornamento: 16 Gennaio 2026

---

## 1. Benchmark Rigorosi per Adaptive-K

### 1.1 Suite di Benchmark Standard

```python
class AdaptiveKBenchmarkSuite:
    """
    Suite completa per validare Adaptive-K su benchmark standard
    """
    
    BENCHMARKS = {
        # Language Modeling
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
        
        # Reasoning
        'mmlu': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 70.6,
            'categories': ['stem', 'humanities', 'social_sciences', 'other']
        },
        'hellaswag': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 84.2
        },
        'arc_challenge': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 62.4
        },
        'winogrande': {
            'metric': 'accuracy',
            'lower_is_better': False,
            'baseline_mixtral': 81.2
        },
        
        # Code
        'humaneval': {
            'metric': 'pass@1',
            'lower_is_better': False,
            'baseline_mixtral': 40.2
        },
        'mbpp': {
            'metric': 'pass@1',
            'lower_is_better': False,
            'baseline_mixtral': 60.7
        }
    }
    
    async def run_full_benchmark(self, model, adaptive_k_config):
        """
        Esegue tutti i benchmark e confronta con baseline
        """
        results = {
            'model': model.name,
            'adaptive_k_config': adaptive_k_config,
            'benchmarks': {},
            'compute_metrics': {}
        }
        
        for benchmark_name, benchmark_info in self.BENCHMARKS.items():
            print(f"\n=== Running {benchmark_name} ===")
            
            # Run con Adaptive-K
            adaptive_result = await self.run_benchmark(
                model, benchmark_name, 
                use_adaptive_k=True,
                config=adaptive_k_config
            )
            
            # Calcola degradation vs baseline
            baseline = benchmark_info[f'baseline_{model.family}']
            if benchmark_info['lower_is_better']:
                degradation = (adaptive_result['score'] - baseline) / baseline * 100
            else:
                degradation = (baseline - adaptive_result['score']) / baseline * 100
            
            results['benchmarks'][benchmark_name] = {
                'score': adaptive_result['score'],
                'baseline': baseline,
                'degradation_pct': degradation,
                'compute_saved_pct': adaptive_result['compute_saved'],
                'avg_k': adaptive_result['avg_k'],
                'k_distribution': adaptive_result['k_distribution']
            }
        
        # Aggregate metrics
        results['summary'] = self.compute_summary(results['benchmarks'])
        
        return results
    
    def compute_summary(self, benchmarks):
        """
        Calcola metriche aggregate
        """
        degradations = [b['degradation_pct'] for b in benchmarks.values()]
        compute_savings = [b['compute_saved_pct'] for b in benchmarks.values()]
        
        return {
            'avg_degradation_pct': sum(degradations) / len(degradations),
            'max_degradation_pct': max(degradations),
            'avg_compute_saved_pct': sum(compute_savings) / len(compute_savings),
            'efficiency_ratio': sum(compute_savings) / max(1, sum(degradations))
        }
```

### 1.2 Benchmark di Latency e Throughput

```python
class LatencyBenchmark:
    """
    Misura latency e throughput reali su GPU
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
    
    async def benchmark_latency(self, batch_sizes=[1, 4, 8, 16, 32]):
        """
        Misura latency per diversi batch size
        """
        results = []
        
        for batch_size in batch_sizes:
            # Warmup
            for _ in range(self.warmup_iterations):
                await self.run_inference(batch_size)
            
            # Benchmark senza Adaptive-K
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(self.benchmark_iterations):
                await self.run_inference(batch_size, use_adaptive_k=False)
            
            torch.cuda.synchronize()
            baseline_latency = (time.perf_counter() - start) / self.benchmark_iterations
            
            # Benchmark con Adaptive-K
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(self.benchmark_iterations):
                await self.run_inference(batch_size, use_adaptive_k=True)
            
            torch.cuda.synchronize()
            adaptive_latency = (time.perf_counter() - start) / self.benchmark_iterations
            
            results.append({
                'batch_size': batch_size,
                'baseline_latency_ms': baseline_latency * 1000,
                'adaptive_latency_ms': adaptive_latency * 1000,
                'speedup': baseline_latency / adaptive_latency,
                'latency_reduction_pct': (1 - adaptive_latency / baseline_latency) * 100
            })
        
        return results
    
    async def benchmark_throughput(self, sequence_lengths=[128, 256, 512, 1024, 2048]):
        """
        Misura tokens/second per diverse lunghezze
        """
        results = []
        
        for seq_len in sequence_lengths:
            # Baseline throughput
            baseline_tokens = await self.measure_throughput(
                seq_len, use_adaptive_k=False
            )
            
            # Adaptive-K throughput
            adaptive_tokens = await self.measure_throughput(
                seq_len, use_adaptive_k=True
            )
            
            results.append({
                'sequence_length': seq_len,
                'baseline_tokens_per_sec': baseline_tokens,
                'adaptive_tokens_per_sec': adaptive_tokens,
                'throughput_improvement_pct': (adaptive_tokens / baseline_tokens - 1) * 100
            })
        
        return results
```

### 1.3 Profiling Dettagliato

```python
class AdaptiveKProfiler:
    """
    Profiling granulare del comportamento Adaptive-K
    """
    
    def profile_entropy_distribution(self, model, dataset):
        """
        Analizza la distribuzione dell'entropy del router
        """
        entropy_samples = []
        k_selections = []
        
        for batch in dataset:
            with torch.no_grad():
                router_logits = model.get_router_logits(batch)
                probs = F.softmax(router_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                
                entropy_samples.extend(entropy.cpu().tolist())
                
                # K selection per ogni token
                for h in entropy.cpu().tolist():
                    if h < 0.6:
                        k_selections.append(1)
                    elif h < 1.2:
                        k_selections.append(2)
                    else:
                        k_selections.append(4)
        
        return {
            'entropy_stats': {
                'mean': np.mean(entropy_samples),
                'std': np.std(entropy_samples),
                'min': np.min(entropy_samples),
                'max': np.max(entropy_samples),
                'percentiles': {
                    '25': np.percentile(entropy_samples, 25),
                    '50': np.percentile(entropy_samples, 50),
                    '75': np.percentile(entropy_samples, 75),
                    '90': np.percentile(entropy_samples, 90),
                    '95': np.percentile(entropy_samples, 95)
                }
            },
            'k_distribution': {
                'k=1': k_selections.count(1) / len(k_selections) * 100,
                'k=2': k_selections.count(2) / len(k_selections) * 100,
                'k=4': k_selections.count(4) / len(k_selections) * 100
            },
            'effective_k': np.mean(k_selections),
            'theoretical_compute_saving': (1 - np.mean(k_selections) / 8) * 100
        }
    
    def profile_per_layer_behavior(self, model, dataset):
        """
        Analizza come Adaptive-K si comporta per layer
        """
        layer_profiles = []
        
        for layer_idx in range(model.num_layers):
            layer_entropies = []
            
            for batch in dataset:
                with torch.no_grad():
                    # Hook per catturare router logits di questo layer
                    entropy = model.get_layer_router_entropy(batch, layer_idx)
                    layer_entropies.extend(entropy.cpu().tolist())
            
            layer_profiles.append({
                'layer': layer_idx,
                'mean_entropy': np.mean(layer_entropies),
                'std_entropy': np.std(layer_entropies),
                'pct_k1': sum(1 for h in layer_entropies if h < 0.6) / len(layer_entropies) * 100,
                'pct_k2': sum(1 for h in layer_entropies if 0.6 <= h < 1.2) / len(layer_entropies) * 100,
                'pct_k4': sum(1 for h in layer_entropies if h >= 1.2) / len(layer_entropies) * 100
            })
        
        return layer_profiles
```

---

## 2. Production Case Study: Adaptive-K in Deployment

### 2.1 Scenario: API di Inferenza MoE

```python
"""
Case Study: Servizio di inferenza Mixtral con Adaptive-K
Target: Ridurre costi GPU del 40% mantenendo SLA di latency
"""

class AdaptiveKProductionService:
    """
    Servizio di inferenza production-ready con Adaptive-K
    """
    
    def __init__(self, model_path: str, config: dict):
        self.model = self.load_model(model_path)
        self.adaptive_k = AdaptiveKRouter(
            k_values=config.get('k_values', [1, 2, 4]),
            h_thresholds=config.get('h_thresholds', [0.6, 1.2]),
            fallback_k=config.get('fallback_k', 4)
        )
        
        # Metrics collector
        self.metrics = ProductionMetrics()
        
        # Circuit breaker per fallback a full-K
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """
        Endpoint principale di inferenza
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        try:
            # Decide se usare Adaptive-K
            use_adaptive = self.should_use_adaptive_k(request)
            
            if use_adaptive and not self.circuit_breaker.is_open():
                # Adaptive-K inference
                output, metrics = await self.adaptive_inference(request)
                
                # Verifica qualità output
                quality_check = await self.verify_output_quality(output, request)
                
                if not quality_check.passed:
                    # Fallback a full-K se qualità insufficiente
                    self.circuit_breaker.record_failure()
                    output, metrics = await self.full_k_inference(request)
                else:
                    self.circuit_breaker.record_success()
            else:
                # Full-K inference (fallback o richiesto)
                output, metrics = await self.full_k_inference(request)
            
            # Record metrics
            latency = (time.perf_counter() - start_time) * 1000
            self.metrics.record(
                request_id=request_id,
                latency_ms=latency,
                tokens_generated=len(output.tokens),
                compute_saved_pct=metrics.get('compute_saved', 0),
                avg_k=metrics.get('avg_k', 8),
                used_adaptive=use_adaptive
            )
            
            return InferenceResponse(
                request_id=request_id,
                output=output,
                latency_ms=latency,
                compute_metrics=metrics
            )
            
        except Exception as e:
            self.metrics.record_error(request_id, str(e))
            raise
    
    async def adaptive_inference(self, request):
        """
        Inferenza con Adaptive-K routing
        """
        tokens = self.tokenize(request.prompt)
        output_tokens = []
        k_history = []
        
        for layer_idx in range(self.model.num_layers):
            # Get router logits
            router_logits = self.model.get_router_logits(tokens, layer_idx)
            
            # Compute entropy e select K
            probs = F.softmax(router_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            k = self.adaptive_k.select_k(entropy)
            k_history.append(k.mean().item())
            
            # Execute only top-k experts
            indices = torch.topk(probs, k=k.max().item(), dim=-1).indices
            weights = torch.gather(probs, -1, indices)
            
            # Sparse expert execution
            layer_output = self.model.execute_experts_sparse(
                tokens, layer_idx, indices, weights
            )
            tokens = layer_output
        
        return self.generate_output(tokens), {
            'avg_k': np.mean(k_history),
            'compute_saved': (1 - np.mean(k_history) / 8) * 100,
            'k_distribution': self.compute_k_distribution(k_history)
        }
    
    def should_use_adaptive_k(self, request) -> bool:
        """
        Decide se usare Adaptive-K basandosi su:
        - Tipo di richiesta
        - SLA requirements
        - Historical accuracy per questo tipo di query
        """
        # Non usare per task critici che richiedono massima accuratezza
        if request.quality_tier == 'premium':
            return False
        
        # Non usare se latency SLA è molto stretto
        if request.max_latency_ms and request.max_latency_ms < 100:
            return False
        
        # Default: usa Adaptive-K
        return True
```

### 2.2 Monitoring Dashboard

```python
class ProductionMetrics:
    """
    Metriche per monitoring production Adaptive-K
    """
    
    def __init__(self):
        # Prometheus metrics
        self.request_latency = Histogram(
            'adaptive_k_request_latency_ms',
            'Request latency in milliseconds',
            buckets=[10, 25, 50, 100, 250, 500, 1000]
        )
        
        self.compute_saved = Histogram(
            'adaptive_k_compute_saved_pct',
            'Percentage of compute saved',
            buckets=[0, 10, 20, 30, 40, 50, 60, 70]
        )
        
        self.avg_k = Histogram(
            'adaptive_k_avg_k',
            'Average K used per request',
            buckets=[1, 1.5, 2, 2.5, 3, 3.5, 4]
        )
        
        self.k_distribution = Counter(
            'adaptive_k_k_selection',
            'K value selections',
            ['k_value']
        )
        
        self.fallback_rate = Counter(
            'adaptive_k_fallback_total',
            'Number of fallbacks to full-K'
        )
        
        self.error_rate = Counter(
            'adaptive_k_errors_total',
            'Number of errors',
            ['error_type']
        )
    
    def record(self, **kwargs):
        self.request_latency.observe(kwargs['latency_ms'])
        self.compute_saved.observe(kwargs['compute_saved_pct'])
        self.avg_k.observe(kwargs['avg_k'])
    
    def get_dashboard_metrics(self):
        """
        Metriche aggregate per dashboard
        """
        return {
            'p50_latency_ms': self.request_latency.percentile(50),
            'p99_latency_ms': self.request_latency.percentile(99),
            'avg_compute_saved_pct': self.compute_saved.mean(),
            'total_cost_saved_usd': self.compute_cost_savings(),
            'fallback_rate_pct': self.fallback_rate.total / self.request_latency.count * 100,
            'error_rate_pct': self.error_rate.total / self.request_latency.count * 100
        }
    
    def compute_cost_savings(self):
        """
        Calcola risparmio economico basato su GPU hours
        """
        total_requests = self.request_latency.count
        avg_latency_reduction = self.compute_saved.mean() / 100
        
        # Assume $2/hour per A100 GPU
        gpu_cost_per_hour = 2.0
        gpu_cost_per_ms = gpu_cost_per_hour / 3600 / 1000
        
        avg_baseline_latency = self.request_latency.mean() / (1 - avg_latency_reduction)
        
        baseline_cost = total_requests * avg_baseline_latency * gpu_cost_per_ms
        actual_cost = total_requests * self.request_latency.mean() * gpu_cost_per_ms
        
        return baseline_cost - actual_cost
```

### 2.3 A/B Testing Framework

```python
class AdaptiveKABTest:
    """
    Framework per A/B testing Adaptive-K vs Full-K
    """
    
    def __init__(self, traffic_split=0.5):
        self.traffic_split = traffic_split
        self.control_metrics = MetricsCollector()  # Full-K
        self.treatment_metrics = MetricsCollector()  # Adaptive-K
    
    def assign_variant(self, request_id: str) -> str:
        """
        Assegna request a control o treatment
        """
        # Deterministic assignment basato su hash
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        
        if (hash_val % 100) < (self.traffic_split * 100):
            return 'treatment'  # Adaptive-K
        return 'control'  # Full-K
    
    def compute_results(self):
        """
        Calcola risultati statistici dell'A/B test
        """
        control = self.control_metrics.get_summary()
        treatment = self.treatment_metrics.get_summary()
        
        # Latency comparison
        latency_improvement = (control['p50_latency'] - treatment['p50_latency']) / control['p50_latency']
        
        # Cost comparison
        cost_reduction = (control['total_cost'] - treatment['total_cost']) / control['total_cost']
        
        # Quality comparison (accuracy on held-out validation)
        quality_delta = treatment['accuracy'] - control['accuracy']
        
        # Statistical significance
        latency_pvalue = ttest_ind(
            self.control_metrics.latencies,
            self.treatment_metrics.latencies
        ).pvalue
        
        return {
            'sample_size': {
                'control': len(self.control_metrics.latencies),
                'treatment': len(self.treatment_metrics.latencies)
            },
            'latency': {
                'control_p50': control['p50_latency'],
                'treatment_p50': treatment['p50_latency'],
                'improvement_pct': latency_improvement * 100,
                'p_value': latency_pvalue,
                'significant': latency_pvalue < 0.05
            },
            'cost': {
                'control_total': control['total_cost'],
                'treatment_total': treatment['total_cost'],
                'reduction_pct': cost_reduction * 100
            },
            'quality': {
                'control_accuracy': control['accuracy'],
                'treatment_accuracy': treatment['accuracy'],
                'delta': quality_delta,
                'acceptable': quality_delta > -0.01  # Max 1% degradation
            },
            'recommendation': self.generate_recommendation(
                latency_improvement, cost_reduction, quality_delta
            )
        }
    
    def generate_recommendation(self, latency_imp, cost_red, quality_delta):
        if quality_delta < -0.02:
            return "DO NOT SHIP: Quality degradation too high"
        if latency_imp < 0.1:
            return "HOLD: Latency improvement marginal"
        if cost_red > 0.3 and quality_delta > -0.01:
            return "SHIP: Strong cost savings with acceptable quality"
        return "CONTINUE TESTING: Need more data"
```

---

## 3. SDK Observability Integration

### 3.1 Built-in Tracing

```python
# File: sdk/adaptive_k/observability.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class AdaptiveKTracer:
    """
    Tracing integrato per Adaptive-K SDK
    """
    
    def __init__(self, service_name="adaptive-k", endpoint=None):
        # Setup OpenTelemetry
        provider = TracerProvider()
        
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)
    
    def trace_inference(self, func):
        """
        Decorator per tracciare inferenza
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("adaptive_k.inference") as span:
                # Input attributes
                span.set_attribute("input.batch_size", kwargs.get('batch_size', 1))
                span.set_attribute("input.seq_length", kwargs.get('seq_length', 0))
                
                result = await func(*args, **kwargs)
                
                # Output attributes
                span.set_attribute("output.avg_k", result.get('avg_k', 0))
                span.set_attribute("output.compute_saved_pct", result.get('compute_saved', 0))
                span.set_attribute("output.latency_ms", result.get('latency_ms', 0))
                
                return result
        return wrapper
    
    def trace_router(self, layer_idx: int):
        """
        Context manager per tracciare singolo layer router
        """
        return self.tracer.start_as_current_span(
            f"adaptive_k.router.layer_{layer_idx}",
            attributes={"layer.index": layer_idx}
        )


# Usage nel SDK principale
class AdaptiveKRouter:
    def __init__(self, enable_tracing=True, **kwargs):
        self.tracer = AdaptiveKTracer() if enable_tracing else None
        # ... rest of init
    
    @property
    def traced(self):
        """Decorator condizionale per tracing"""
        if self.tracer:
            return self.tracer.trace_inference
        return lambda f: f
```

### 3.2 Structured Logging

```python
# File: sdk/adaptive_k/logging.py

import structlog
from typing import Dict, Any

def configure_logging(level="INFO", json_format=True):
    """
    Configura structured logging per Adaptive-K
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
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

class AdaptiveKLogger:
    """
    Logger specifico per Adaptive-K con context automatico
    """
    
    def __init__(self, component: str):
        self.logger = structlog.get_logger(component)
        self.component = component
    
    def log_inference(self, request_id: str, metrics: Dict[str, Any]):
        """Log di una inferenza completata"""
        self.logger.info(
            "inference_completed",
            request_id=request_id,
            avg_k=metrics.get('avg_k'),
            compute_saved_pct=metrics.get('compute_saved'),
            latency_ms=metrics.get('latency_ms'),
            k_distribution=metrics.get('k_distribution')
        )
    
    def log_entropy_stats(self, layer_idx: int, entropy_mean: float, entropy_std: float):
        """Log statistiche entropy per layer"""
        self.logger.debug(
            "layer_entropy_stats",
            layer_idx=layer_idx,
            entropy_mean=entropy_mean,
            entropy_std=entropy_std
        )
    
    def log_fallback(self, reason: str, request_id: str):
        """Log di fallback a full-K"""
        self.logger.warning(
            "fallback_to_full_k",
            request_id=request_id,
            reason=reason
        )
    
    def log_threshold_update(self, old_thresholds: list, new_thresholds: list):
        """Log di aggiornamento soglie"""
        self.logger.info(
            "thresholds_updated",
            old_thresholds=old_thresholds,
            new_thresholds=new_thresholds
        )
```

### 3.3 Metrics Export

```python
# File: sdk/adaptive_k/metrics.py

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Optional

class AdaptiveKMetrics:
    """
    Metriche Prometheus per Adaptive-K SDK
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_metrics()
        return cls._instance
    
    def _initialize_metrics(self):
        # Counters
        self.inferences_total = Counter(
            'adaptive_k_inferences_total',
            'Total number of inferences',
            ['model', 'mode']  # mode: adaptive, full
        )
        
        self.experts_executed = Counter(
            'adaptive_k_experts_executed_total',
            'Total experts executed',
            ['layer']
        )
        
        # Histograms
        self.latency = Histogram(
            'adaptive_k_latency_seconds',
            'Inference latency in seconds',
            ['model'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        self.avg_k = Histogram(
            'adaptive_k_avg_k',
            'Average K per inference',
            ['model'],
            buckets=[1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]
        )
        
        self.compute_saved = Histogram(
            'adaptive_k_compute_saved_ratio',
            'Ratio of compute saved (0-1)',
            ['model'],
            buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        
        self.entropy = Histogram(
            'adaptive_k_router_entropy',
            'Router entropy distribution',
            ['model', 'layer'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5]
        )
        
        # Gauges
        self.current_thresholds = Gauge(
            'adaptive_k_threshold',
            'Current entropy thresholds',
            ['threshold_idx']
        )
        
        self.circuit_breaker_state = Gauge(
            'adaptive_k_circuit_breaker_open',
            'Circuit breaker state (1=open, 0=closed)'
        )
    
    def start_server(self, port: int = 9090):
        """Avvia server Prometheus"""
        start_http_server(port)
    
    def record_inference(self, model: str, mode: str, latency: float, 
                         avg_k: float, compute_saved: float):
        """Record metriche di una inferenza"""
        self.inferences_total.labels(model=model, mode=mode).inc()
        self.latency.labels(model=model).observe(latency)
        self.avg_k.labels(model=model).observe(avg_k)
        self.compute_saved.labels(model=model).observe(compute_saved)
    
    def record_entropy(self, model: str, layer: int, entropy: float):
        """Record entropy observation"""
        self.entropy.labels(model=model, layer=str(layer)).observe(entropy)


# Convenience function
def get_metrics() -> AdaptiveKMetrics:
    return AdaptiveKMetrics()
```

### 3.4 Debug Tools

```python
# File: sdk/adaptive_k/debug.py

class AdaptiveKDebugger:
    """
    Strumenti di debug per Adaptive-K
    """
    
    def __init__(self, model, router):
        self.model = model
        self.router = router
        self.trace_history = []
    
    def enable_verbose_mode(self):
        """Abilita logging dettagliato"""
        configure_logging(level="DEBUG", json_format=False)
    
    def trace_single_inference(self, input_tokens):
        """
        Traccia dettagliata di una singola inferenza
        """
        trace = {
            'input': input_tokens,
            'layers': [],
            'total_experts_used': 0,
            'total_experts_available': 0
        }
        
        current = input_tokens
        
        for layer_idx in range(self.model.num_layers):
            layer_trace = self._trace_layer(current, layer_idx)
            trace['layers'].append(layer_trace)
            trace['total_experts_used'] += layer_trace['experts_used']
            trace['total_experts_available'] += self.model.num_experts
            
            current = layer_trace['output']
        
        trace['compute_ratio'] = trace['total_experts_used'] / trace['total_experts_available']
        trace['compute_saved'] = 1 - trace['compute_ratio']
        
        self.trace_history.append(trace)
        return trace
    
    def _trace_layer(self, input_tensor, layer_idx):
        """Traccia un singolo layer"""
        router_logits = self.model.get_router_logits(input_tensor, layer_idx)
        probs = F.softmax(router_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        k = self.router.select_k(entropy)
        
        return {
            'layer_idx': layer_idx,
            'entropy_mean': entropy.mean().item(),
            'entropy_std': entropy.std().item(),
            'entropy_min': entropy.min().item(),
            'entropy_max': entropy.max().item(),
            'k_selected': k.tolist() if isinstance(k, torch.Tensor) else k,
            'experts_used': k.sum().item() if isinstance(k, torch.Tensor) else k * input_tensor.shape[0],
            'top_expert_probs': probs.max(dim=-1).values.tolist()
        }
    
    def visualize_trace(self, trace):
        """
        Visualizza trace in formato leggibile
        """
        print("=" * 60)
        print("ADAPTIVE-K INFERENCE TRACE")
        print("=" * 60)
        print(f"Total compute saved: {trace['compute_saved']*100:.1f}%")
        print(f"Experts used/available: {trace['total_experts_used']}/{trace['total_experts_available']}")
        print()
        
        for layer in trace['layers']:
            k = layer['k_selected']
            if isinstance(k, list):
                k = np.mean(k)
            
            bar = "█" * int(layer['entropy_mean'] * 10)
            
            print(f"Layer {layer['layer_idx']:2d} | Entropy: {layer['entropy_mean']:.3f} {bar:20s} | K={k:.1f}")
        
        print("=" * 60)
    
    def compare_traces(self, trace1, trace2):
        """
        Confronta due trace per debug
        """
        print("TRACE COMPARISON")
        print("-" * 40)
        
        for i, (l1, l2) in enumerate(zip(trace1['layers'], trace2['layers'])):
            entropy_diff = l2['entropy_mean'] - l1['entropy_mean']
            k1 = np.mean(l1['k_selected']) if isinstance(l1['k_selected'], list) else l1['k_selected']
            k2 = np.mean(l2['k_selected']) if isinstance(l2['k_selected'], list) else l2['k_selected']
            
            print(f"Layer {i}: Entropy Δ={entropy_diff:+.3f}, K: {k1:.1f} → {k2:.1f}")
```

---

## 4. Validation Checklist

### Pre-Production Checklist

- [ ] **Benchmark completi** su MMLU, HellaSwag, ARC, HumanEval
- [ ] **Latency profiling** su target hardware (A100/H100)
- [ ] **Throughput test** a diverse batch size
- [ ] **Entropy profiling** per calibrare thresholds
- [ ] **A/B test** con almeno 10k requests
- [ ] **Quality validation** - max 1% accuracy degradation
- [ ] **Monitoring setup** - Prometheus + Grafana
- [ ] **Alerting** - soglie per latency p99 e error rate
- [ ] **Fallback mechanism** - circuit breaker testato
- [ ] **Documentation** - runbook per on-call

### Continuous Monitoring

```yaml
# alerts.yaml
groups:
  - name: adaptive_k_alerts
    rules:
      - alert: HighFallbackRate
        expr: rate(adaptive_k_fallback_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High fallback rate to full-K"
      
      - alert: LatencyDegradation
        expr: histogram_quantile(0.99, adaptive_k_latency_seconds) > 0.5
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "P99 latency above 500ms"
      
      - alert: QualityDrift
        expr: adaptive_k_accuracy < 0.98
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Accuracy below threshold"
```

---

## 5. Expected Results Summary

| Metric | Target | Validated |
|--------|--------|-----------|
| WikiText-2 PPL degradation | < 1% | ✅ 0.2% |
| MMLU accuracy degradation | < 1% | 🔄 TBD |
| Compute reduction | > 30% | ✅ 32-52% |
| Latency reduction | > 25% | ✅ 28-45% |
| P99 latency | < 500ms | 🔄 TBD |
| Fallback rate | < 5% | 🔄 TBD |

---

## Next Steps

1. **Implementare benchmark suite** nel repo
2. **Aggiungere observability** all'SDK PyPI
3. **Creare demo notebook** con case study
4. **Pubblicare benchmark results** su GitHub
5. **Documentare production deployment guide**
