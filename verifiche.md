# Verifiche Adaptive-K: Quick Reference

> Questo file è un quick reference. Per documentazione completa vedi:
> **[docs/ADAPTIVE_K_VERIFICATION.md](docs/ADAPTIVE_K_VERIFICATION.md)**

---

## Quick Links

| Sezione | Descrizione |
|---------|-------------|
| [Benchmark Commands](#benchmark-commands) | Come eseguire benchmark standard |
| [Observability Setup](#observability-setup) | Setup tracing e metrics |
| [Production Checklist](#production-checklist) | Pre-deploy validation |

---

## Benchmark Commands

```bash
# 1. Run benchmark completo
python -m scripts.run_benchmark --model adaptive_k --benchmark all

# 2. Latency profiling
python -m scripts.profile_latency --batch-sizes 1,4,8,16 --warmup 10

# 3. Entropy analysis
python scripts/analyze_entropy_buckets.py results/runs/<run_id>/

# 4. Multi-seed validation (REQUIRED per risultati pubblicabili)
python -m scripts.run_multiseed --config configs/sbm_adaptive_k_mnist.yaml --seeds 42,43,44,45,46
```

---

## Observability Setup

```python
from adaptive_k.observability import AdaptiveKTracer, AdaptiveKMetrics

# 1. Enable tracing
tracer = AdaptiveKTracer(service_name="my-service", endpoint="http://jaeger:4317")

# 2. Enable metrics  
metrics = AdaptiveKMetrics()
metrics.start_server(port=9090)  # Prometheus endpoint

# 3. Structured logging
from adaptive_k.logging import configure_logging
configure_logging(level="INFO", json_format=True)
```

### Key Metrics da Monitorare

| Metric | Descrizione | Target |
|--------|-------------|--------|
| `adaptive_k_latency_seconds` | Latency inferenza | p99 < 500ms |
| `adaptive_k_avg_k` | Media K usato | < 3 per 30%+ savings |
| `adaptive_k_compute_saved_ratio` | Compute risparmiato | > 0.3 |
| `adaptive_k_fallback_total` | Numero fallback | < 5% |

---

## Production Checklist

### Pre-Deploy

- [ ] Benchmark su MMLU, HellaSwag, HumanEval completati
- [ ] Latency p99 < 500ms verificato
- [ ] Accuracy degradation < 1% validato  
- [ ] Entropy thresholds calibrati sul dataset target
- [ ] A/B test con almeno 10k requests

### Infrastructure

- [ ] Prometheus + Grafana configurati
- [ ] Alerting rules attive (vedi sotto)
- [ ] Circuit breaker testato
- [ ] Fallback mechanism funzionante
- [ ] Runbook documentato

### Alerting Rules

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
          
      - alert: LatencyDegradation
        expr: histogram_quantile(0.99, adaptive_k_latency_seconds) > 0.5
        for: 10m
        labels:
          severity: critical
          
      - alert: LowComputeSavings
        expr: avg(adaptive_k_compute_saved_ratio) < 0.2
        for: 15m
        labels:
          severity: warning
```

---

## Expected Results

| Metric | Target | Current Status |
|--------|--------|----------------|
| WikiText-2 PPL degradation | < 1% | ✅ 0.2% |
| MMLU accuracy degradation | < 1% | 🔄 TBD |
| Compute reduction | > 30% | ✅ 32-52% |
| Latency reduction | > 25% | ✅ 28-45% |
| Fallback rate | < 5% | 🔄 TBD |

---

## SDK Integration Example

```python
from adaptive_k import AdaptiveKRouter, AdaptiveKConfig

# Configuration
config = AdaptiveKConfig(
    k_values=[1, 2, 4],
    h_thresholds=[0.6, 1.2],
    enable_tracing=True,
    enable_metrics=True
)

# Initialize router
router = AdaptiveKRouter(config)

# Use in inference
for layer in model.layers:
    # Get router entropy
    entropy = layer.compute_router_entropy(hidden_states)
    
    # Adaptive K selection
    k = router.select_k(entropy)
    
    # Execute only top-k experts
    indices, weights = layer.route(hidden_states, k=k)
    output = layer.execute_experts(hidden_states, indices, weights)
```

---

## Debug Tools

```python
from adaptive_k.debug import AdaptiveKDebugger

# Enable verbose mode
debugger = AdaptiveKDebugger(model, router)
debugger.enable_verbose_mode()

# Trace single inference
trace = debugger.trace_single_inference(input_tokens)
debugger.visualize_trace(trace)

# Output:
# ============================================================
# ADAPTIVE-K INFERENCE TRACE
# ============================================================
# Total compute saved: 42.5%
# Experts used/available: 46/80
#
# Layer  0 | Entropy: 0.423 ████░░░░░░░░░░░░░░░░ | K=1.0
# Layer  1 | Entropy: 0.891 ████████░░░░░░░░░░░░ | K=2.0
# Layer  2 | Entropy: 1.234 ████████████░░░░░░░░ | K=4.0
# ...
```

---

## Documentation Links

- [ADAPTIVE_K_VERIFICATION.md](docs/ADAPTIVE_K_VERIFICATION.md) - Documentazione completa con benchmark suite, production case study, observability SDK
- [SBM_ADAPTIVE_K.md](docs/SBM_ADAPTIVE_K.md) - Architettura teorica
- [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) - Note implementative
- [copilot-instructions.md](.github/copilot-instructions.md) - Quick reference commands
