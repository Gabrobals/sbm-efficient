# Phase B Results (Noise Robustness)

Status: B1 planned; fill after running logit noise evaluations.

## Scope
- Task: MNIST
- Models: static_topk_mnist, sbm_adaptive_k_mnist
- Noise: Gaussian on logits in evaluation only, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$
- Sigmas: 0.05, 0.10, 0.20, 0.50, 1.00
- Seeds: 42, 43, 44, 45, 46
- Metrics per $\sigma$: accuracy, precision/recall/F1 (macro + micro), FP/FN totals, confusion matrix, k_mean (Adaptive-K only), flops_executed, latency_ms, degradation_pct vs $\sigma=0$ (per seed)

## Run Commands (after code changes)
```bash
# static_topk with noise eval
python -m scripts.run_multiseed --config configs/static_topk_mnist.yaml --seeds 42,43,44,45,46

# sbm_adaptive_k with noise eval
python -m scripts.run_multiseed --config configs/sbm_adaptive_k_mnist.yaml --seeds 42,43,44,45,46

# build aggregated noise summary
python -m scripts.summarize_noise_robustness
```

## Aggregated Outputs
- Per-run metrics: results/runs/<run_id>/metrics.json (noise.evaluations[])
- Aggregated: results/aggregated_results.json (now includes noise blocks)
- Summary: results/summaries/mnist_noise_robustness.json

## Tables (fill after runs)

### Accuracy / F1 vs $\sigma$
| $\sigma$ | static_topk acc | static_topk F1 | sbm_adaptive_k acc | sbm_adaptive_k F1 | $\Delta$acc (%) | $\Delta$F1 (%) |
|---|---|---|---|---|---|---|
| 0.00 | baseline | baseline | baseline | baseline | 0.0 | 0.0 |
| 0.05 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.10 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.20 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1.00 | TBD | TBD | TBD | TBD | TBD | TBD |

### FP / FN and FLOPs vs $\sigma$
| $\sigma$ | model | FP | FN | k_mean | flops_executed | latency_ms | degradation_acc (%) |
|---|---|---|---|---|---|---|---|
| 0.00 | static_topk | baseline | baseline | 2.0 | baseline | baseline | 0.0 |
| 0.00 | sbm_adaptive_k | baseline | baseline | baseline | baseline | baseline | 0.0 |
| 0.05 | static_topk | TBD | TBD | 2.0 | check stable | TBD | TBD |
| 0.05 | sbm_adaptive_k | TBD | TBD | TBD | check stable | TBD | TBD |
| 0.10 | static_topk | TBD | TBD | 2.0 | check stable | TBD | TBD |
| 0.10 | sbm_adaptive_k | TBD | TBD | TBD | check stable | TBD | TBD |
| 0.20 | static_topk | TBD | TBD | 2.0 | check stable | TBD | TBD |
| 0.20 | sbm_adaptive_k | TBD | TBD | TBD | check stable | TBD | TBD |
| 0.50 | static_topk | TBD | TBD | 2.0 | check stable | TBD | TBD |
| 0.50 | sbm_adaptive_k | TBD | TBD | TBD | check stable | TBD | TBD |
| 1.00 | static_topk | TBD | TBD | 2.0 | check stable | TBD | TBD |
| 1.00 | sbm_adaptive_k | TBD | TBD | TBD | check stable | TBD | TBD |

### Notes to Validate
- flops_executed must remain constant across $\sigma$ for each model.
- degradation_pct is computed per seed vs its own $\sigma=0$ baseline.
- latency_ms from evaluation loop (wall-clock average) is logged per $\sigma$; profiling p50 remains in final.latency_ms.
- For sbm_adaptive_k, k_mean in noise.evaluations must stay within routing tolerance; active_modules_mean should match k_mean.
