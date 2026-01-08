# PHASE A RESULTS — Classification Metrics Validation (MNIST, 5 seeds)

## Summary
- Dataset: MNIST
- Models compared: static_topk (K=2), sbm_adaptive_k (threshold v1)
- Seeds: 42, 43, 44, 45, 46
- Status: All runs VALID (pre-flight + post-flight)

## Aggregate Metrics (mean ± std)
- Accuracy: static_topk 0.97964 ± 0.00382; adaptive_k 0.97986 ± 0.00268
- Precision (macro): static_topk 0.97983 ± 0.00367; adaptive_k 0.98010 ± 0.00244
- Recall (macro): static_topk 0.97944 ± 0.00372; adaptive_k 0.97953 ± 0.00280
- F1 (macro): static_topk 0.97951 ± 0.00379; adaptive_k 0.97968 ± 0.00271
- Precision/Recall/F1 (micro): equal to accuracy for both models
- FLOPs executed: static_topk 2,073,924; adaptive_k 1,721,688 (≈ -17%)
- Latency (p50 ms, CPU): static_topk 1.48 ± 0.18; adaptive_k 1.30 ± 0.47 (higher variance expected on CPU)
- k_mean: static_topk 2.00 ± 0.00; adaptive_k 1.6603 ± 0.2017
- k_std: static_topk 0.0; adaptive_k 0.8940 ± 0.1413 (routing truly adaptive)

## Findings
- Adaptive-K matches accuracy/precision/recall/F1 of static_topk with slightly lower variance.
- Compute reduction is real (≈17% FLOPs) with no increase in FP/FN; confusion matrices remain balanced.
- Routing is adaptive (entropy non-collapsed, k_std > 0) with lower average active modules.
- Latency variance is higher on CPU; expected to stabilize on GPU/batched inference.

## Conclusion
Adaptive-K v1 is validated as a compute-control mechanism (not compression): same classification quality as static Top-K with measurable reduction in executed FLOPs and lower average active modules. This is the new project baseline.
