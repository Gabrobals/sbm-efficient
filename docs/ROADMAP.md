# ROADMAP

- Phase A: DONE — Classification metrics (accuracy, precision/recall/F1, FP/FN, confusion) + multi-seed validation; Adaptive-K v1 baseline with ~17% FLOPs reduction vs static_topk on MNIST.
- Phase B: NEXT — Fault Injection / Robustness (logit noise, entropy corruption, expert dropout). Do not start until approved.
- Phase C: ON HOLD — Hugging Face harness and public benchmark once robustness is validated.

Guardrails: No SBM-H/Bloch/Cayley/observables; no budgeted-K or differentiable gating; ASCII-only logging; sparse execution only.
