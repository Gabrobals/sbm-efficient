# Response to Peer Review: Adaptive-K Paper

**Date:** January 2026  
**Authors:** Adaptive-K Research Team  
**Review Source:** Qwen3-Max Peer Review Analysis  

---

## Executive Summary

This document addresses the **5 critical errors** identified in the peer review of our paper "Adaptive-K: Entropy-Guided Dynamic Expert Selection in MoE Models" published at https://adaptive-k.vertexdata.it/paper.html.

We acknowledge the validity of all identified issues and present our corrections below.

---

## 🔴 Critical Error 1: FLOPs Formula for SwiGLU (Mathematical - CRITICAL)

### Reviewer Comment
> "The formula `C_E = 4·d·d_ff` is incorrect for SwiGLU. SwiGLU requires 3 projections (up, gate, down), not 2."

### Acknowledgment
**We fully acknowledge this error.** The FLOPs calculation for SwiGLU experts was underestimated by 50%.

### Correction

**Original (Incorrect):**
```
C_E = 4·d·d_ff  (implied in simplified Savings formula)
```

**Corrected:**
```math
C_E = 6·d·d_ff
```

**Breakdown for SwiGLU:**
| Component | FLOPs |
|-----------|-------|
| Up-projection: d → d_ff | 2·d·d_ff |
| Gate-projection: d → d_ff | 2·d·d_ff |
| Down-projection: d_ff → d | 2·d_ff·d |
| Elementwise (silu + multiply) | ~4·d_ff (negligible) |
| **Total** | **6·d·d_ff** |

### Impact on Results

The simplified savings formula `Savings ≈ 1 - E[K]/K_baseline` remains **directionally correct** but should be presented with caveats:

**Exact Savings Formula:**
```math
Savings_{exact} = 1 - (C_G + C_H + E[K]·C_E) / (C_G + K_baseline·C_E)
```

For Mixtral 8×7B (d=4096, d_ff=14336):
- C_E (corrected) = 6 × 4096 × 14336 = **352.3 GFLOPs** per expert
- C_G (router) = 2 × n × d × N ≈ **0.54 GFLOPs** (negligible)
- Original reported savings: **31.0%**
- Corrected savings: **~28.1%** (still significant)

### Paper Update
- Section 2.4: Add footnote clarifying SwiGLU FLOPs formula
- Section 5.3: Add "See Appendix B for detailed FLOPs calculation"
- Appendix: Add new section B.2 with explicit FLOPs derivation

---

## 🔴 Critical Error 2: Multiplicative Savings Claim (Mathematical - CRITICAL)

### Reviewer Comment
> "The assumption of multiplicative independence is false. Optimizations interact non-linearly."

### Acknowledgment
**We acknowledge this is a theoretical projection, not an empirical result.**

### Correction

**Original (Section 8.4):**
```
Example: 1 - (0.69 × 0.67 × 0.65) = 70% savings
```

**Corrected Language:**
```
"Adaptive-K is COMPATIBLE with orthogonal optimizations. Under ideal 
conditions of independent interaction, the theoretical maximum combined 
savings would be: 1 - (0.69 × 0.67 × 0.65) = 70%.

However, actual combined savings must be validated empirically due to 
potential sub-multiplicative interactions between:
- Quantization (affects routing precision → may alter entropy distribution)
- Speculative decoding (modifies token generation order → changes calibration)

We leave empirical validation of combined optimizations to future work."
```

### Paper Update
- Section 8.4: Add disclaimer about theoretical nature
- Future Work: Add item "Empirical validation of combined optimizations"

---

## 🔴 Critical Error 3: Nemotron Validation Based on Projections (Methodological - CRITICAL)

### Reviewer Comment
> "Nemotron results are theoretical projections based on H/Hmax, not empirical measurements."

### Acknowledgment
**We acknowledge this is a limitation.** The Nemotron results demonstrate that the entropy distribution characteristics are favorable for Adaptive-K, but do not constitute full empirical validation.

### Correction

**Original Table 8 Caption:**
```
"Adaptive-K achieves 33.3% compute reduction"
```

**Corrected Caption:**
```
"Projected Adaptive-K savings based on entropy analysis. 
These are THEORETICAL projections based on H/Hmax ratios.
Full empirical validation (actual inference with variable K, 
perplexity measurement) is pending."
```

### What We DID Validate
- ✅ Router logits extraction via forward hooks
- ✅ Entropy distribution statistics (mean=5.23, std=0.48)
- ✅ H/Hmax ratios across different prompt types
- ✅ K projection formulas

### What We DID NOT Validate
- ❌ Actual forward passes with variable K
- ❌ Perplexity measurements with Adaptive-K routing
- ❌ Downstream task performance
- ❌ Latency measurements

### Paper Update
- Table 8: Add "Projected" label to savings column
- Section 5.4: Add clarifying paragraph on validation scope
- Limitations (Section 8.2): Add bullet point

---

## 🔴 Critical Error 4: Missing Formal Statistical Analysis (Methodological - GRAVE)

### Reviewer Comment
> "The claim 'no statistically significant degradation' is not supported. No p-values, confidence intervals, or power analysis provided."

### Acknowledgment
**We acknowledge the lack of formal statistical tests.** The current paper relies on informal comparison of point estimates.

### Required Corrections

**1. Two One-Sided Tests (TOST) for Non-Inferiority**

For each model, we should report:
```python
from statsmodels.stats.weightstats import ttost_paired

# Hypothesis: Adaptive-K perplexity ≤ baseline + 1%
# Equivalence margin: Δ = 0.01 × baseline_ppl

result = ttost_paired(
    baseline_ppl_samples,  # n ≥ 1250 sequences
    adaptive_ppl_samples,
    low=-np.inf,
    upp=margin
)

# Report: p-value, 95% CI
```

**2. Required Statistics to Add**

| Metric | Requirement |
|--------|-------------|
| Sample size | n ≥ 1,250 sequences (power=0.95, α=0.025, d=0.35) |
| Confidence interval | 95% CI for Δperplexity |
| P-value | From TOST test |
| Effect size | Cohen's d |

**3. Example Results Format (Future)**
```
Mixtral 8×7B Perplexity:
- Baseline: 3.84 (95% CI: 3.81-3.87)
- Adaptive-K: 3.87 (95% CI: 3.84-3.90)
- Δ: +0.03 (+0.8%)
- TOST p-value: 0.018 (non-inferiority confirmed, α=0.025)
```

### Paper Update
- Section 5: Add subsection "5.6 Statistical Analysis"
- Add Table 11 with formal statistical results
- Appendix: Add power analysis methodology

---

## 🟡 Error 5: Proposition 3.2 Without Formal Proof (Theoretical - MEDIUM)

### Reviewer Comment
> "'Mild regularity conditions' are not defined. No proof provided."

### Acknowledgment
**We acknowledge the proposition is informal.** It was intended as an intuitive guide, not a rigorous theorem.

### Correction

**Original Proposition 3.2:**
```
"Under mild regularity conditions on the expert functions..."
```

**Corrected Theorem 3.2:**

**Theorem 3.2 (Entropy-Distortion Relationship)**

*Let E = {E₁, ..., E_N} be Lipschitz-continuous expert functions with constant L. Let p(x) be the routing distribution for input x. Then for any ε > 0, there exists K* ≤ N such that:*

$$E_{x \sim D}[\|y_{K*}(x) - y_N(x)\|_2^2] \leq \varepsilon$$

*with probability at least 1-δ, where:*

$$K^* = \min\{k : \sum_{i=1}^k p_{(i)}(x) \geq 1 - \sqrt{\frac{\varepsilon}{4L^2\|x\|^2}}\}$$

*and p_(i) are probabilities sorted in descending order.*

**Proof Sketch:**
1. By Lipschitz continuity: ||E_i(x) - E_j(x)||_2 ≤ 2L||x||_2
2. Output approximation error bounded by contribution of excluded experts
3. If top-K captures (1-γ) of probability mass, distortion ≤ γ × (2L||x||_2)²
4. Setting γ = √(ε/(4L²||x||²)) gives the threshold

*Full proof in Appendix C.*

### Paper Update
- Section 3.2: Replace Proposition 3.2 with formal Theorem
- Add Appendix C with complete proof

---

## Summary of Paper Changes

### Immediate (v2 release)

| Section | Change |
|---------|--------|
| 2.4 | Add footnote on SwiGLU FLOPs: C_E = 6·d·d_ff |
| 3.2 | Replace Proposition 3.2 with formal Theorem + proof sketch |
| 5.4 | Add "Projected" labels, clarify validation scope |
| 8.4 | Add disclaimer on multiplicative composition |
| 8.2 | Add limitation about Nemotron empirical validation |
| Appendix | Add B.2 (FLOPs derivation), C (Theorem proof) |

### Future Work Required

1. **Full Statistical Analysis** (4-6 weeks)
   - Power analysis
   - TOST tests on all models
   - Bootstrap confidence intervals

2. **Nemotron Empirical Validation** (2-4 weeks)
   - Implement actual variable-K routing
   - Measure perplexity
   - Downstream tasks

3. **Combined Optimization Validation** (4-8 weeks)
   - AK + INT8 quantization
   - AK + speculative decoding
   - Measure actual (not theoretical) combined savings

---

## Response to Specific Claims

### On Savings Estimates
> "Mixtral savings overestimated: 31.0% → 28.1%"

**Response:** We accept a revised estimate of **~28%** for the exact savings when accounting for all overhead. The simplified formula remains useful for intuition but we will clarify its limitations.

### On Statistical Significance
> "No p-values or CI provided"

**Response:** This is a valid critique. We commit to adding formal statistical analysis in paper v2 or a follow-up publication.

### On Nemotron Validation
> "Theoretical projections, not empirical"

**Response:** We will relabel the Nemotron results as "Entropy Analysis" rather than "Validation" and clarify what was and was not measured.

---

## Conclusion

We thank the reviewer for the thorough and constructive critique. The identified issues are valid and will be addressed in the paper revision. The core contribution of Adaptive-K—using entropy to dynamically select K—remains sound, but the presentation and statistical rigor will be improved.

**Key Points:**
1. ✅ Core algorithm and approach are valid
2. ⚠️ Quantitative claims need refinement (28% vs 31%)
3. ⚠️ Statistical validation needs formalization
4. ⚠️ Nemotron results need relabeling
5. ⚠️ Multiplicative composition is theoretical

---

*Document prepared in response to peer review. Last updated: January 2026*
