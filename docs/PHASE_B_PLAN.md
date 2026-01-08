# PHASE B PLAN — Fault Injection / Robustness (Do Not Implement Yet)

Scope: Evaluate robustness of routing and compute control under controlled faults. Execute steps sequentially (B1 → B2 → B3). Keep metrics identical to Phase A (accuracy, precision, recall, F1 macro/micro, k_mean, k_std, FLOPs, latency).

## B1 — Logit Noise Injection
- Inject Gaussian noise into router logits (zero-mean, tune sigma).
- Measure impact on: accuracy, precision/recall/F1, k_mean/k_std, FLOPs/latency.
- Compare models: static_topk vs sbm_adaptive_k.
- Expectation: adaptive_k should retain accuracy and adjust k; static_topk is unaffected structurally but serves as control.

## B2 — Entropy Corruption
- Perturb entropy estimates (e.g., jitter or clamp) before K selection in adaptive_k.
- Measure same metrics as B1.
- Expectation: adaptive_k should show graceful degradation; check if k collapses or inflates.

## B3 — Expert Dropout
- Randomly disable one expert per batch/step.
- Measure same metrics.
- Expectation: adaptive_k should reroute compute; static_topk cannot adapt (control).

## Reporting
- For each fault type: run multi-seed (≥5), report mean ± std for metrics above.
- Plot Accuracy vs FLOPs per fault level (noise sigma / corruption strength / dropout rate).
- Document in a dedicated section under docs/ (to be created after experiments).

## Guardrails
- No code changes until explicit go-ahead for B1.
- Keep logging ASCII-only; reuse Phase A schema.
- Do not introduce SBM-H/Bloch/Cayley/observables; do not add budgeted-K or differentiable gating.
