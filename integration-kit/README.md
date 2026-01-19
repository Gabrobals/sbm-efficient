# Adaptive-K Integration Starter Kit

**Reduce MoE inference costs by 40-60% in 2-5 engineering days.**

This kit provides everything needed to integrate Adaptive-K into existing MoE deployments with minimal effort.

## Quick Start (Day 1)

```bash
# 1. Install
pip install adaptive-k-routing

# 2. Get license (free for evaluation)
# Visit: https://adaptive-k.vercel.app/register

# 3. Run ROI calculator
python roi_calculator.py --tokens-per-day 1000000000 --cost-per-1k 0.001

# 4. Calibrate on your data
python calibrate.py --model mixtral-8x7b --dataset your_data.jsonl

# 5. Deploy
python deploy_example.py
```

## Kit Contents

| File | Purpose | Day |
|------|---------|-----|
| `roi_calculator.py` | Estimate savings before integration | 1 |
| `calibrate.py` | Find optimal thresholds for your workload | 1-2 |
| `integration_vllm.py` | vLLM integration example | 2-3 |
| `integration_trtllm.py` | TensorRT-LLM integration example | 2-3 |
| `integration_huggingface.py` | HuggingFace Transformers example | 2 |
| `monitoring_dashboard.py` | Grafana metrics exporter | 3-4 |
| `ab_test_framework.py` | A/B testing for production rollout | 4-5 |
| `configs/` | Ready-to-use configurations | - |

## Integration Timeline

### Day 1: Assessment & Calibration
- [ ] Run ROI calculator to estimate savings
- [ ] Set up evaluation environment
- [ ] Run calibration on representative data sample
- [ ] Review entropy distribution of your workload

### Day 2: Integration
- [ ] Choose integration path (vLLM/TRT-LLM/HF)
- [ ] Implement router wrapper
- [ ] Run unit tests
- [ ] Validate quality metrics

### Day 3: Testing
- [ ] Run benchmark suite
- [ ] Compare latency/throughput vs baseline
- [ ] Verify perplexity within tolerance
- [ ] Load testing

### Day 4: Staging Deployment
- [ ] Deploy to staging environment
- [ ] Set up monitoring dashboards
- [ ] Configure alerts
- [ ] Run shadow traffic

### Day 5: Production Rollout
- [ ] Gradual rollout (10% → 50% → 100%)
- [ ] Monitor metrics
- [ ] Document learnings

## Expected Results

Based on our experiments and production deployments:

| Metric | Baseline | With Adaptive-K | Improvement |
|--------|----------|-----------------|-------------|
| Avg K per token | 2.0 | 1.2-1.5 | 25-40% fewer experts |
| FLOPs per token | 100% | 40-60% | 40-60% reduction |
| Latency (p50) | 100% | 85-95% | 5-15% faster |
| Throughput | 100% | 140-180% | 40-80% higher |
| Quality (PPL) | baseline | +0.5-1.5% | Negligible impact |

## Support

- Documentation: https://adaptive-k.vercel.app/paper.html
- Issues: https://github.com/Gabrobals/sbm-efficient/issues
- Email: amministrazione@vertexdata.it

## License

Apache 2.0 with required registration. Free for evaluation and research.
