# VERTEX-RESEARCH: Academic Research Protocol for Machine Learning

## A Rigorous Framework for Hypothesis-Driven ML Research

**Version:** 1.0.0  
**Author:** Gabriele Balsamo  
**Institution:** VertexData Research  
**Last Updated:** January 2026  
**License:** CC BY-SA 4.0

---

## Table of Contents

1. [Philosophical Foundations](#1-philosophical-foundations)
2. [Research Lifecycle Overview](#2-research-lifecycle-overview)
3. [Phase 0: Problem Identification](#3-phase-0-problem-identification)
4. [Phase 1: Literature Review](#4-phase-1-literature-review)
5. [Phase 2: Theoretical Framework](#5-phase-2-theoretical-framework)
6. [Phase 3: Hypothesis Formulation](#6-phase-3-hypothesis-formulation)
7. [Phase 4: Experimental Design](#7-phase-4-experimental-design)
8. [Phase 5: Implementation](#8-phase-5-implementation)
9. [Phase 6: Experimentation](#9-phase-6-experimentation)
10. [Phase 7: Validation & Falsification](#10-phase-7-validation--falsification)
11. [Phase 8: Analysis & Interpretation](#11-phase-8-analysis--interpretation)
12. [Phase 9: Paper Writing](#12-phase-9-paper-writing)
13. [Phase 10: Publication & Dissemination](#13-phase-10-publication--dissemination)
14. [AI-Assisted Research Guidelines](#14-ai-assisted-research-guidelines)
15. [Quality Assurance Checklists](#15-quality-assurance-checklists)

---

## 1. Philosophical Foundations

### 1.1 The Popperian Framework

This protocol is grounded in Karl Popper's **falsificationism** - the principle that scientific knowledge advances not through verification but through rigorous attempts at falsification.

> *"A theory that is not refutable by any conceivable event is non-scientific."*
> — Karl Popper, *The Logic of Scientific Discovery* (1934)

#### Core Principles

| Principle | Definition | Application in ML Research |
|-----------|------------|---------------------------|
| **Falsifiability** | A theory must make predictions that can be proven false | Hypotheses must specify conditions under which they would fail |
| **Corroboration** | Theories survive attempts at falsification, not proven true | Results are "corroborated" not "verified" |
| **Critical Rationalism** | Knowledge grows through conjecture and refutation | Actively seek to disprove your own hypotheses |
| **Asymmetry** | One counterexample can falsify; no confirmations can verify | Design experiments to find edge cases |

### 1.2 The Hypothetico-Deductive Method

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC METHOD CYCLE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                               │
│   │  PROBLEM    │ ◄──────────────────────────────────────┐      │
│   └──────┬──────┘                                        │      │
│          │                                               │      │
│          ▼                                               │      │
│   ┌─────────────┐                                        │      │
│   │ HYPOTHESIS  │ (Conjecture)                           │      │
│   └──────┬──────┘                                        │      │
│          │                                               │      │
│          ▼                                               │      │
│   ┌─────────────┐                                        │      │
│   │ PREDICTION  │ (Deduction)                            │      │
│   └──────┬──────┘                                        │      │
│          │                                               │      │
│          ▼                                               │      │
│   ┌─────────────┐                                        │      │
│   │    TEST     │ (Experimentation)                      │      │
│   └──────┬──────┘                                        │      │
│          │                                               │      │
│          ▼                                               │      │
│   ┌─────────────┐      ┌─────────────┐                  │      │
│   │  FALSIFIED  │──YES─►│  REFINE/    │──────────────────┘      │
│   │      ?      │      │  REJECT     │                          │
│   └──────┬──────┘      └─────────────┘                          │
│          │ NO                                                    │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │ CORROBORATED│ (Tentatively accepted)                        │
│   └─────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Reproducibility Framework

Following the REFORMS checklist (Kapoor et al., 2024) and NeurIPS guidelines:

| Term | Definition | Requirements |
|------|------------|--------------|
| **Repeatability** | Same team, same setup, same results | Version control, fixed seeds |
| **Reproducibility** | Different team, same data, same results | Complete code, data, environment |
| **Replicability** | Different team, new data, similar results | Clear methodology, specifications |

---

## 2. Research Lifecycle Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH LIFECYCLE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 0        PHASE 1         PHASE 2         PHASE 3         PHASE 4   │
│  ┌──────┐      ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐   │
│  │PROBLEM│ ──► │ LIT  │ ──►   │THEORY│ ──►   │HYPOTH│ ──►    │DESIGN│   │
│  │ ID   │      │REVIEW│        │FRAME │        │FORM  │        │ EXP  │   │
│  └──────┘      └──────┘        └──────┘        └──────┘        └──────┘   │
│     │              │               │               │               │       │
│     │              │               │               │               │       │
│     │              │               │               │               │       │
│  PHASE 10       PHASE 9        PHASE 8        PHASE 7        PHASE 6      │
│  ┌──────┐      ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐   │
│  │PUBLI │ ◄── │PAPER │ ◄──   │ANALY │ ◄──   │VALID │ ◄──    │ EXP  │   │
│  │CATION│      │WRITE │        │ SIS  │        │/FALS │        │ RUN  │   │
│  └──────┘      └──────┘        └──────┘        └──────┘        └──────┘   │
│                                    ▲                               │       │
│                                    │                               │       │
│                                    └────────── PHASE 5 ────────────┘       │
│                                              ┌──────┐                      │
│                                              │IMPL  │                      │
│                                              │CODE  │                      │
│                                              └──────┘                      │
│                                                                            │
│  ═══════════════════════════════════════════════════════════════════════   │
│  DURATION ESTIMATES (for a typical ML paper):                              │
│  Phase 0-1: 2-4 weeks | Phase 2-4: 2-3 weeks | Phase 5-7: 4-8 weeks       │
│  Phase 8-9: 2-4 weeks | Phase 10: 2-6 months (review cycle)               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Gate Reviews

Each phase transition requires a **gate review** - formal verification that all requirements are met before proceeding.

```yaml
Gate Review Structure:
  inputs:
    - Completed deliverables from current phase
    - Quality checklist (phase-specific)
    - Risk assessment
  
  review_criteria:
    - Completeness: All required artifacts present
    - Quality: Meets academic standards
    - Consistency: Aligned with previous phases
    - Feasibility: Next phase is achievable
  
  outputs:
    - GO: Proceed to next phase
    - CONDITIONAL: Proceed with identified risks
    - RETURN: Revise current phase
    - ABORT: Fundamental issues, reconsider approach
```

---

## 3. Phase 0: Problem Identification

### 3.1 Objectives

- Identify a **significant, unsolved problem** in the field
- Establish **relevance** and **impact potential**
- Define clear **research boundaries**

### 3.2 Problem Statement Template

```markdown
## Problem Statement

### Context
[Background that establishes why this area matters]

### Gap
[What specific problem remains unsolved?]

### Significance
[Why does solving this matter? Who benefits?]

### Scope
[What are the boundaries of this research?]

### Success Criteria
[How will we know if we've succeeded?]
```

### 3.3 Problem Validation Checklist

| Criterion | Question | Status |
|-----------|----------|--------|
| **Novelty** | Has this specific problem been addressed before? | ☐ |
| **Significance** | Would a solution have meaningful impact? | ☐ |
| **Feasibility** | Can this be solved with available resources? | ☐ |
| **Measurability** | Can success/failure be objectively measured? | ☐ |
| **Scope** | Is the problem appropriately bounded? | ☐ |

### 3.4 Example: Adaptive-K Problem Statement

```markdown
## Problem Statement: Fixed-K Inefficiency in MoE Models

### Context
Mixture-of-Experts (MoE) architectures enable massive model capacity while 
maintaining computational tractability through sparse activation. Current 
production models (Mixtral, GPT-4, etc.) use fixed top-K expert selection.

### Gap
The fixed-K constraint treats all tokens identically regardless of routing 
decision confidence. For high-confidence tokens (low routing entropy), 
activating K experts wastes compute. For uncertain tokens (high entropy), 
K experts may be insufficient.

### Significance
- 30%+ potential compute reduction → millions in inference cost savings
- Applicable to all MoE architectures without retraining
- Theoretical insight into router behavior and information geometry

### Scope
- Focus: Inference-time optimization only (no training modifications)
- Models: Production MoE architectures with learned routers
- Metrics: Compute reduction, perplexity preservation, downstream accuracy

### Success Criteria
- ≥25% compute reduction with <1% perplexity degradation
- Demonstrated on ≥3 production models
- Theoretical justification via information theory
```

---

## 4. Phase 1: Literature Review

### 4.1 Objectives

- Map the **intellectual landscape** of the problem
- Identify **gaps** in existing work
- Find **theoretical foundations** to build upon
- Discover **related methods** and their limitations

### 4.2 Systematic Review Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                 SYSTEMATIC LITERATURE REVIEW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: DEFINE SEARCH STRATEGY                                 │
│  ├── Primary keywords                                           │
│  ├── Secondary keywords                                         │
│  ├── Boolean combinations                                       │
│  └── Date range                                                 │
│                                                                  │
│  Step 2: SELECT DATABASES                                       │
│  ├── arXiv (ML/AI)                                             │
│  ├── Semantic Scholar                                          │
│  ├── Google Scholar                                            │
│  ├── ACL Anthology (NLP)                                       │
│  └── Papers With Code                                          │
│                                                                  │
│  Step 3: INITIAL SCREENING                                      │
│  ├── Title/abstract screening                                  │
│  ├── Apply inclusion/exclusion criteria                        │
│  └── Document decisions                                        │
│                                                                  │
│  Step 4: FULL-TEXT REVIEW                                       │
│  ├── Deep read of selected papers                              │
│  ├── Extract key information                                   │
│  └── Assess quality/relevance                                  │
│                                                                  │
│  Step 5: SYNTHESIS                                              │
│  ├── Identify themes and patterns                              │
│  ├── Map research landscape                                    │
│  └── Identify gaps                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Paper Analysis Template

```markdown
## Paper Analysis: [Title]

### Metadata
- Authors: 
- Venue: 
- Year: 
- Citations: 
- Code: [Available/Not Available]

### Summary (3-5 sentences)

### Key Contributions
1. 
2. 
3. 

### Methodology
- Approach:
- Datasets:
- Baselines:
- Metrics:

### Results
| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
|        |          |          |             |

### Strengths
- 

### Limitations
- 

### Relevance to Our Work
- Connection:
- Differentiation:
- Potential extension:
```

### 4.4 Deliverables

- [ ] Literature review document (organized by theme)
- [ ] Reference database (BibTeX)
- [ ] Gap analysis summary
- [ ] Positioning statement (how your work differs)

---

## 5. Phase 2: Theoretical Framework

### 5.1 Objectives

- Establish **mathematical foundations**
- Define **formal notation** and terminology
- Develop **theoretical insights** that motivate the approach
- Create **provable guarantees** where possible

### 5.2 Mathematical Rigor Standards

Following standards from top venues (NeurIPS, ICML, ICLR):

| Component | Requirement | Example |
|-----------|-------------|---------|
| **Definitions** | Precise, unambiguous | *Definition 2.1: A routing distribution p(x) is...* |
| **Theorems** | Formal statement + proof | *Theorem 3.1: Under conditions C₁, C₂, ...* |
| **Propositions** | Intermediate results | *Proposition 3.2: The entropy H(x) lower-bounds...* |
| **Lemmas** | Supporting technical results | *Lemma A.1: For any distribution p...* |
| **Corollaries** | Direct consequences | *Corollary 3.3: It follows that...* |

### 5.3 Assumption Documentation

All assumptions must be:
1. **Explicitly stated**
2. **Justified** (why reasonable)
3. **Tested** (if possible)

```markdown
## Assumptions

### A1: Router Softmax Output
**Statement:** The router produces a valid probability distribution via softmax.
**Justification:** Standard in all major MoE implementations.
**Testable:** Yes - verified empirically.

### A2: Expert Independence
**Statement:** Expert outputs are conditionally independent given the input.
**Justification:** Standard MoE architecture assumption.
**Testable:** Partially - can measure correlation.
```

---

## 6. Phase 3: Hypothesis Formulation

### 6.1 Objectives

- Transform theoretical insights into **testable predictions**
- Define **clear, falsifiable hypotheses**
- Specify **success/failure criteria** in advance

### 6.2 Hypothesis Structure

Following Popper's requirements for scientific hypotheses:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS STRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CLAIM: [What we predict will happen]                           │
│                                                                  │
│  CONDITIONS: [Under what circumstances]                          │
│                                                                  │
│  MECHANISM: [Why we expect this to happen]                       │
│                                                                  │
│  FALSIFICATION: [What would prove this wrong]                    │
│                                                                  │
│  METRICS: [How we measure success/failure]                       │
│                                                                  │
│  THRESHOLD: [Quantitative criteria]                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Hypothesis Template

```markdown
## Hypothesis H1: [Short Name]

### Claim
[Precise statement of what will happen]

### Conditions
- C1: [First condition that must hold]
- C2: [Second condition]
- ...

### Theoretical Basis
[Brief explanation of WHY this should be true, referencing Phase 2]

### Falsification Criteria
This hypothesis is FALSIFIED if:
- F1: [First falsification condition]
- F2: [Second falsification condition]
- ...

### Measurement
- Primary metric: [e.g., perplexity]
- Secondary metrics: [e.g., downstream accuracy]
- Statistical test: [e.g., paired t-test, p < 0.05]

### Quantitative Thresholds
| Metric | Baseline | Required | Falsification |
|--------|----------|----------|---------------|
| PPL    | 3.84     | ≤ 3.92   | > 4.0 (>4%)  |
| Compute| 100%     | ≤ 75%    | > 85% (<15%) |
```

### 6.4 Hypothesis Registration

**Pre-registration** of hypotheses prevents HARKing (Hypothesizing After Results are Known):

```markdown
## Hypothesis Registration

**Project:** Adaptive-K Routing
**Date:** 2026-01-15
**Principal Investigator:** Gabriele Balsamo

### Registered Hypotheses (before any experiments)

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Entropy-K Correlation | REGISTERED |
| H2 | Threshold Robustness | REGISTERED |
| H3 | Cross-Model Generalization | REGISTERED |

### Commitment
The above hypotheses were formulated before any experimental results
were obtained. Post-hoc analyses will be clearly labeled as exploratory.

**Signature:** _____________  
**Date:** _____________
```

---

## 7. Phase 4: Experimental Design

### 7.1 Objectives

- Design experiments that can **falsify** hypotheses
- Ensure **statistical validity** and **reproducibility**
- Plan for **confounders** and **controls**

### 7.2 Experimental Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                 EXPERIMENTAL DESIGN HIERARCHY                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: PROOF OF CONCEPT                                      │
│  ├── Simple dataset (MNIST, XOR)                               │
│  ├── Verify basic mechanism works                              │
│  └── Quick iteration cycle                                     │
│                                                                  │
│  Level 2: CONTROLLED EXPERIMENTS                                │
│  ├── Standard benchmarks (WikiText-2, PTB)                     │
│  ├── Multiple seeds                                            │
│  ├── Ablation studies                                          │
│  └── Statistical significance                                  │
│                                                                  │
│  Level 3: PRODUCTION VALIDATION                                 │
│  ├── Real-world models (Mixtral, Qwen)                         │
│  ├── Diverse domains                                           │
│  ├── Scale testing                                             │
│  └── Practical deployment considerations                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Experimental Matrix Template

```markdown
## Experimental Matrix

### Independent Variables
| Variable | Levels | Rationale |
|----------|--------|-----------|
| Model | Mixtral, Qwen, OLMoE, Nemotron | Architecture diversity |
| Dataset | WikiText-2, PTB, MMLU | Task diversity |
| Threshold | 0.8, 1.0, 1.275, 1.5, 1.8 | Sensitivity analysis |
| K values | {1,2}, {2,3,4}, {4,6,8} | Granularity |

### Dependent Variables
| Variable | Measurement | Precision |
|----------|-------------|-----------|
| Perplexity | Cross-entropy loss | 2 decimal places |
| Avg K | Mean experts activated | 2 decimal places |
| Compute | Avg K / Baseline K | Percentage |
| Accuracy | Task-specific | Percentage |

### Controls
| Control | Method |
|---------|--------|
| Random seed | Fixed + multiple seeds (5) |
| Data splits | Standard train/val/test |
| Hardware | Document GPU, CUDA version |
| Software | Pin all library versions |
```

### 7.4 Statistical Analysis Plan

**Specify BEFORE running experiments:**

```markdown
## Statistical Analysis Plan

### Primary Analysis
- **Test:** Paired t-test (Adaptive-K vs Baseline)
- **α level:** 0.05 (two-tailed)
- **Multiple comparisons:** Bonferroni correction
- **Effect size:** Cohen's d

### Reporting Standards
| Metric | Report |
|--------|--------|
| Mean | Yes |
| Std Dev | Yes |
| 95% CI | Yes |
| p-value | Yes (if applicable) |
| Effect size | Yes |
| Sample size | Yes |
```

---

## 8. Phase 5: Implementation

### 8.1 Objectives

- Create **reproducible, well-documented code**
- Follow **software engineering best practices**
- Enable **independent verification**

### 8.2 Repository Structure

```
project_root/
├── README.md                    # Project overview, quickstart
├── LICENSE                      # License file
├── pyproject.toml              # Project metadata, dependencies
├── requirements.txt            # Pinned dependencies
│
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   ├── data/                   # Data loading
│   ├── training/               # Training loops
│   └── utils/                  # Utilities
│
├── configs/                    # Configuration files (YAML)
├── scripts/                    # Runnable scripts
├── tests/                      # Unit tests
├── experiments/                # Experiment outputs
└── paper/                      # Paper source
```

### 8.3 Reproducibility Requirements

```python
# requirements.txt - PIN ALL VERSIONS
torch==2.1.0
transformers==4.35.0
numpy==1.24.3
# ... all versions pinned

# seed.py - Deterministic execution
def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 9. Phase 6: Experimentation

### 9.1 Experiment Execution Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXPERIMENT EXECUTION FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PRE-FLIGHT CHECK                                            │
│     ├── Verify code is committed (no uncommitted changes)       │
│     ├── Verify config matches pre-registration                  │
│     ├── Verify resources available (GPU, memory)                │
│     └── Verify data is accessible                               │
│                                                                  │
│  2. EXECUTION                                                   │
│     ├── Log start time, commit hash, config                     │
│     ├── Run experiment with all seeds                           │
│     ├── Log intermediate checkpoints                            │
│     └── Handle errors gracefully                                │
│                                                                  │
│  3. POST-FLIGHT                                                 │
│     ├── Verify outputs are complete                             │
│     ├── Calculate statistics                                    │
│     ├── Archive logs and results                                │
│     └── Update experiment tracker                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Experiment Log Template

```markdown
## Experiment Log

### Metadata
- **Experiment ID:** EXP-2026-01-15-001
- **Date:** 2026-01-15 14:30:00 UTC
- **Researcher:** Gabriele Balsamo
- **Git Commit:** a1b2c3d4e5f6
- **Config Hash:** md5:abcd1234

### Environment
- **Hardware:** 1x RTX 4090 (24GB)
- **CUDA:** 12.1
- **Python:** 3.10.12
- **PyTorch:** 2.1.0

### Execution
| Seed | Start Time | End Time | Status | Notes |
|------|------------|----------|--------|-------|
| 42 | 14:30:00 | 15:45:00 | ✓ | - |
| 43 | 15:45:00 | 17:00:00 | ✓ | - |
| ... | ... | ... | ... | ... |

### Results Summary
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Avg K | 1.38 | 0.02 | 1.35 | 1.41 |
| PPL | 3.87 | 0.01 | 3.86 | 3.88 |
```

---

## 10. Phase 7: Validation & Falsification

### 10.1 Objectives

- **Rigorously test** hypotheses against results
- **Actively seek** to falsify claims
- Document both **successes and failures**

### 10.2 Falsification Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                  FALSIFICATION PROTOCOL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each hypothesis H:                                         │
│                                                                  │
│  Step 1: CHECK PRE-REGISTERED FALSIFICATION CRITERIA            │
│          Did any F1, F2, ... occur?                             │
│                                                                  │
│  Step 2: EXAMINE EDGE CASES                                     │
│          Did any unexpected failures occur?                     │
│          Are there conditions where method fails?               │
│                                                                  │
│  Step 3: ADVERSARIAL TESTING                                    │
│          Design additional tests to break the method            │
│          Try hardest possible inputs                            │
│                                                                  │
│  Step 4: STATISTICAL VALIDATION                                 │
│          Is effect statistically significant?                   │
│          Is effect practically significant?                     │
│                                                                  │
│  Step 5: DETERMINE STATUS                                       │
│          ├── FALSIFIED: Clear evidence against                  │
│          ├── CORROBORATED: Survived falsification attempts      │
│          └── INCONCLUSIVE: Need more data                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Hypothesis Evaluation Template

```markdown
## Hypothesis Evaluation: H1 - Entropy-K Correlation

### Pre-Registered Falsification Criteria Check

| Criterion | Threshold | Observed | Status |
|-----------|-----------|----------|--------|
| F1: Random K achieves similar | Δ < 5% | Δ = 23% | NOT TRIGGERED |
| F2: Inverse correlation | r < -0.3 | r = +0.72 | NOT TRIGGERED |

### Quantitative Results

| Metric | Required | Observed | Passed? |
|--------|----------|----------|---------|
| PPL Δ | ≤ +1% | +0.8% | ✓ |
| Savings | ≥ 25% | 31.0% | ✓ |

### Verdict

**STATUS: CORROBORATED**

The hypothesis survived all pre-registered falsification criteria.
```

---

## 11. Phase 8: Analysis & Interpretation

### 11.1 Analysis Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSIS FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DESCRIPTIVE ANALYSIS                                        │
│     └── Summarize what happened                                 │
│         ├── Central tendencies (mean, median)                  │
│         ├── Variability (std, range, IQR)                      │
│         └── Distributions (histograms, box plots)              │
│                                                                  │
│  2. INFERENTIAL ANALYSIS                                        │
│     └── Test hypotheses statistically                          │
│         ├── Significance tests (t-test, ANOVA)                 │
│         ├── Effect sizes (Cohen's d, η²)                       │
│         └── Confidence intervals                               │
│                                                                  │
│  3. EXPLORATORY ANALYSIS                                        │
│     └── Discover unexpected patterns                           │
│         ├── Correlation analysis                               │
│         ├── Clustering                                         │
│         └── Visualization                                      │
│                                                                  │
│  4. INTERPRETIVE ANALYSIS                                       │
│     └── Explain WHY results occurred                           │
│         ├── Connect to theory                                  │
│         ├── Compare to related work                            │
│         └── Identify mechanisms                                │
│                                                                  │
│  5. CRITICAL ANALYSIS                                           │
│     └── Assess limitations and validity                        │
│         ├── Internal validity                                  │
│         ├── External validity                                  │
│         └── Construct validity                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Phase 9: Paper Writing

### 12.1 Paper Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    STANDARD ML PAPER STRUCTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TITLE (informative, specific)                               │
│                                                                  │
│  2. ABSTRACT (~250 words)                                       │
│     ├── Problem                                                 │
│     ├── Method                                                  │
│     ├── Results (with numbers!)                                 │
│     └── Significance                                            │
│                                                                  │
│  3. INTRODUCTION (1-1.5 pages)                                  │
│     ├── Context and motivation                                  │
│     ├── Problem statement                                       │
│     ├── Key insight/approach                                    │
│     ├── Summary of contributions                                │
│     └── Paper outline (optional)                                │
│                                                                  │
│  4. RELATED WORK (0.5-1 page)                                   │
│  5. METHOD (2-3 pages)                                          │
│  6. EXPERIMENTS (2-3 pages)                                     │
│  7. DISCUSSION (0.5-1 page)                                     │
│  8. CONCLUSION (0.25-0.5 page)                                  │
│  9. REFERENCES                                                  │
│  10. APPENDIX (supplementary)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Abstract Formula

```
[PROBLEM] The challenge of X remains unsolved because Y.
[METHOD] We propose Z, which addresses this by W.
[RESULTS] On benchmarks A, B, C, we achieve Q% improvement with P% reduction.
[SIGNIFICANCE] This demonstrates R and enables S.
```

---

## 13. Phase 10: Publication & Dissemination

### 13.1 Venue Selection Matrix

| Venue | Topics | Deadline | Format |
|-------|--------|----------|--------|
| NeurIPS | General ML | May/Jun | 9 pages |
| ICML | General ML | Jan/Feb | 8 pages |
| ICLR | Representation Learning | Sep/Oct | 8 pages |
| arXiv | Preprints | Rolling | Any |

### 13.2 Pre-Submission Checklist

- [ ] Follows venue template exactly
- [ ] Within page limit
- [ ] Anonymized for double-blind
- [ ] All experiments reproducible
- [ ] Code prepared for release
- [ ] Proofread by others

---

## 14. AI-Assisted Research Guidelines

### 14.1 Philosophy: AI as Multiplier, Not Replacement

```
┌─────────────────────────────────────────────────────────────────┐
│              AI-ASSISTED RESEARCH BOUNDARIES                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✅ AI CAN (appropriate uses):                                  │
│  ├── Search and summarize literature                           │
│  ├── Explain concepts and papers                               │
│  ├── Generate boilerplate code                                 │
│  ├── Check mathematical derivations                            │
│  ├── Suggest experimental variations                           │
│  ├── Proofread and improve writing                            │
│  ├── Debug code                                                │
│  └── Accelerate routine tasks                                  │
│                                                                  │
│  ❌ AI CANNOT (human responsibility):                          │
│  ├── Formulate hypotheses (your creative contribution)         │
│  ├── Interpret results (requires domain judgment)              │
│  ├── Decide experimental validity (scientific judgment)        │
│  ├── Make claims beyond data (integrity)                       │
│  ├── Take responsibility for errors (accountability)           │
│  └── Replace peer review (community validation)                │
│                                                                  │
│  ⚠️ AI WITH VERIFICATION (trust but verify):                   │
│  ├── Mathematical proofs → manually verify                     │
│  ├── Code generation → test thoroughly                         │
│  ├── Literature claims → check original sources                │
│  └── Statistical analysis → validate assumptions               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Disclosure Requirements

When using AI assistance, document:

```markdown
## AI Assistance Disclosure

### Tools Used
- Claude 3.5/Opus: Literature search, code review, writing assistance
- GitHub Copilot: Code completion

### Scope of Use
- AI was used for [specific tasks]
- All [hypotheses/interpretations/claims] are original human contributions
- All AI-generated content was verified for accuracy

### Verification
- Mathematical derivations: Manually verified
- Code: Tested with unit tests and integration tests
- Claims: Cross-referenced with primary sources
```

### 14.3 Prompt Engineering for Research

```yaml
Research Prompt Templates:

literature_search:
  system: |
    You are a research assistant helping with systematic literature review.
    Focus on factual information from papers. Always cite sources.
    If uncertain, say so explicitly.
  
  user: |
    Search for papers on [TOPIC].
    For each relevant paper, extract:
    1. Main contribution
    2. Methodology
    3. Key results
    4. Relevance to [MY RESEARCH QUESTION]

hypothesis_critique:
  system: |
    You are a skeptical reviewer following Popperian principles.
    Your job is to find weaknesses and potential falsification conditions.
    Be rigorous but constructive.
  
  user: |
    Critique this hypothesis:
    [HYPOTHESIS]
    
    1. Is it falsifiable? What would disprove it?
    2. What assumptions does it make?
    3. What edge cases might fail?
    4. What experiments would test it?

code_review:
  system: |
    You are a code reviewer focused on reproducibility and correctness.
    Check for: bugs, reproducibility issues, statistical errors.
  
  user: |
    Review this experiment code:
    [CODE]
    
    Check:
    1. Random seed handling
    2. Data leakage
    3. Statistical validity
    4. Edge cases
```

---

## 15. Quality Assurance Checklists

### 15.1 Pre-Experiment Checklist

```markdown
## Pre-Experiment Checklist

### Registration
- [ ] Hypotheses registered before any experiments
- [ ] Falsification criteria specified
- [ ] Statistical analysis plan defined
- [ ] Success/failure thresholds quantified

### Code
- [ ] All code committed to version control
- [ ] Tests passing
- [ ] Dependencies pinned
- [ ] Seeds set for reproducibility

### Data
- [ ] Data accessible and verified
- [ ] Train/val/test splits fixed
- [ ] Preprocessing documented

### Environment
- [ ] GPU available and verified
- [ ] Environment reproducible (requirements.txt)
```

### 15.2 Post-Experiment Checklist

```markdown
## Post-Experiment Checklist

### Results
- [ ] All seeds completed successfully
- [ ] Statistics calculated (mean, std, CI)
- [ ] Results match expected format
- [ ] No anomalies in logs

### Validation
- [ ] Falsification criteria checked
- [ ] Statistical significance tested
- [ ] Effect size calculated
- [ ] Limitations documented

### Documentation
- [ ] Experiment log completed
- [ ] Results archived
- [ ] Figures generated
- [ ] Notes on unexpected findings
```

### 15.3 Paper Submission Checklist

```markdown
## Paper Submission Checklist

### Format
- [ ] Correct template used
- [ ] Page limit respected
- [ ] Anonymized (if required)
- [ ] References complete

### Technical
- [ ] All math verified
- [ ] All tables verified
- [ ] All figures high quality
- [ ] Code ready for release

### Reproducibility
- [ ] All details documented
- [ ] Hardware/software specified
- [ ] Seeds reported
- [ ] Data accessible
```

---

## References

1. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
2. Kapoor, S., et al. (2024). REFORMS: Consensus-based Recommendations for ML-based Science. *Science Advances*.
3. Pineau, J., et al. (2021). Improving Reproducibility in Machine Learning Research. *JMLR*.
4. NeurIPS Paper Checklist Guidelines (2024). https://neurips.cc/public/guides/PaperChecklist
5. ICML Paper Guidelines (2024). https://icml.cc/Conferences/2024/PaperGuidelines

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-19 | Initial release |

---

*This protocol is licensed under CC BY-SA 4.0. You are free to share and adapt this material with attribution.*

**VERTEX-RESEARCH Protocol v1.0.0**  
**VertexData Research**  
**https://github.com/Gabrobals/sbm-efficient**
