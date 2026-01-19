# Adaptive-K Outreach Templates

> Ready-to-use email and LinkedIn templates for strategic outreach.
> Generated: January 2026

---

## 1. Red Hat / vLLM Team

### Target: Saša Zelenović
- **Role**: Principal Product Marketing Manager, Red Hat (ex Neural Magic)
- **LinkedIn**: https://www.linkedin.com/in/sasazelenovic/
- **Relevance**: Hosts vLLM Office Hours, expert in model compression

### LinkedIn Connection Request
```
Hi Saša, I read your excellent article on vLLM + model compression for inference optimization. I'm working on Adaptive-K routing - a complementary technique that dynamically selects HOW MANY experts to activate per token in MoE models (25-52% compute savings). Would love to discuss potential integration with vLLM. The two approaches multiply!
```

### Follow-up Email
```
Subject: Adaptive-K + vLLM: Completing the inference optimization stack

Hi Saša,

I recently read your Red Hat blog post on optimizing LLM inference with vLLM and was impressed by the comprehensive coverage of quantization and PagedAttention techniques.

I wanted to share a complementary approach we've developed: **Adaptive-K routing** for Mixture-of-Experts models.

**The key insight**: Current optimizations (quantization, PagedAttention, batching) reduce the cost *per expert operation*. Adaptive-K reduces the *number of expert operations* per token by dynamically selecting K based on routing entropy.

**Results on production MoE models**:
- Mixtral 8×7B: 31.0% compute reduction
- Qwen1.5-MoE: 32.4% reduction  
- OLMoE-1B-7B: 24.7% reduction

All without quality degradation (Δ perplexity < 1%).

The approaches are multiplicative: vLLM's 40-50% infrastructure savings × Adaptive-K's 25-52% compute reduction = significant combined benefit.

Would you be interested in:
1. A technical discussion for vLLM Office Hours?
2. Exploring integration into vLLM's MoE routing?
3. A joint blog post on "The Complete Inference Stack"?

**Resources**:
- Whitepaper: https://adaptive-k.vercel.app/paper.html
- Live demo: https://huggingface.co/spaces/Gabrobals/adaptive-k-demo
- PyPI: pip install adaptive-k-routing

Best regards,
Gabriele Balsamo
VertexData
```

---

## 2. AWS Machine Learning Team

### Target: AWS ML Blog / Inferentia Team
- **Contact**: aws-ml-blog@amazon.com
- **Relevance**: Inferentia case studies mention latency/cost optimization

### Guest Post Proposal
```
Subject: Guest Post Proposal: "Multiplying Inferentia Savings with Adaptive MoE Routing"

Dear AWS ML Blog Team,

I'd like to propose a technical guest post demonstrating how software-level MoE optimization can multiply the hardware efficiency gains of AWS Inferentia.

**Proposed Title**: "Beyond Hardware: How Adaptive-K Routing Multiplies Inferentia's MoE Inference Savings"

**Key angle**: Your Sprinklr case study shows 30%+ latency reduction with Inferentia. For MoE models like Mixtral, our Adaptive-K routing adds another 25-52% compute reduction on top - they multiply, not add.

**Article outline**:
1. The inference optimization stack (hardware → engine → model → dynamic compute)
2. How Adaptive-K complements Inferentia for MoE workloads
3. Benchmark: Mixtral on Inferentia with/without Adaptive-K
4. Implementation guide with AWS Neuron SDK

**Supporting materials**:
- Published whitepaper with peer-reviewed methodology
- Open-source implementation (PyPI: adaptive-k-routing)
- Reproducible benchmarks

Happy to provide draft or collaborate with AWS technical writers.

Best regards,
Gabriele Balsamo
```

---

## 3. IBM Think / AI Team

### Target: IBM Think Editorial
- **Contact**: think@ibm.com
- **Relevance**: "What is AI Inference" article covers basics but misses dynamic compute

### Pitch Email
```
Subject: Article Idea: "The Fourth Dimension of AI Inference Optimization"

Dear IBM Think Editorial,

Your article "What is AI Inference?" excellently covers the three types of inference (dynamic, batch, streaming) and the hardware landscape.

I'd like to propose a follow-up piece on an emerging fourth dimension: **dynamic compute allocation**.

**The gap**: Current optimizations assume fixed compute per inference. But with Mixture-of-Experts (MoE) models powering systems like GPT-4 and Gemini, there's opportunity to vary compute based on input complexity.

**Our contribution**: Adaptive-K routing uses information theory (Shannon entropy) to dynamically select how many experts to activate. Simple queries → fewer experts → less compute. Complex queries → more experts → maintained quality.

**Proposed article**: "From Fixed to Adaptive: The Fourth Dimension of Efficient AI Inference"

Topics covered:
- Why MoE changes the inference cost equation
- Entropy as a compute allocation signal
- 25-52% savings on production models
- Implications for enterprise AI deployment

This would complement your existing inference content while introducing readers to next-generation optimization techniques.

Best regards,
Gabriele Balsamo
Independent Researcher, VertexData
```

---

## 4. Ultralytics / Computer Vision

### Target: Ultralytics Team
- **Contact**: GitHub PR / hello@ultralytics.com
- **Relevance**: YOLO fixed architecture, opportunity for dynamic selection

### GitHub Issue / Discussion
```
Title: [Feature Discussion] Adaptive expert selection for complex scenes

Hi Ultralytics team,

I've been following your excellent work on YOLO architectures and read your glossary on inference latency optimization.

**Idea for discussion**: Current YOLO models use fixed architecture for every frame. What if we could use more compute for complex scenes (crowded, occluded) and less for simple scenes (clear, few objects)?

**Background**: In NLP, we've developed Adaptive-K routing for MoE models that achieves 25-52% compute savings by varying the number of active experts based on input entropy.

**Potential application to vision**:
- Low-entropy scenes (clear, simple) → fewer processing blocks
- High-entropy scenes (complex, ambiguous) → full processing

This could be particularly valuable for:
- Edge deployment with power constraints
- Real-time video where frame complexity varies
- Autonomous vehicles (parking lot vs highway)

Would this be interesting to explore? Happy to collaborate on a proof-of-concept.

Resources:
- Our method: https://adaptive-k.vercel.app/paper.html
- Demo: https://huggingface.co/spaces/Gabrobals/adaptive-k-demo
```

---

## 5. HuggingFace Transformers Team

### Target: Transformers Maintainers
- **Contact**: GitHub PR
- **Relevance**: MoE support in transformers, routing customization

### RFC / PR Description
```
Title: [RFC] Adaptive-K routing for MoE models

## Summary
Add entropy-based dynamic K selection for MoE routing, enabling 25-52% compute savings without quality loss.

## Motivation
Current MoE implementations use fixed top-K routing regardless of routing confidence. High-confidence tokens waste compute on unnecessary experts.

## Proposed Change
Add `AdaptiveKRouter` that:
1. Computes routing entropy
2. Selects K based on entropy thresholds
3. Falls back to standard top-K for uncertain tokens

## API Design
```python
from transformers import MixtralForCausalLM, AdaptiveKConfig

# Option 1: Config-based
config = AdaptiveKConfig(
    k_values=[1, 2],
    calibration_percentile=62
)
model = MixtralForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    adaptive_k_config=config
)

# Option 2: Post-hoc application
from transformers.models.mixtral.modeling_mixtral import apply_adaptive_k
model = apply_adaptive_k(model, k_values=[1, 2])
```

## Benchmarks
| Model | Baseline K | Adaptive-K Avg | Compute | PPL Δ |
|-------|------------|----------------|---------|-------|
| Mixtral 8×7B | 2 | 1.38 | 69.0% | +0.8% |
| Qwen1.5-MoE | 4 | 2.71 | 67.6% | +0.9% |

## Implementation
- Whitepaper: [link]
- Reference implementation: [PyPI]
- Tests: [link]

Happy to discuss design choices and implement based on feedback.
```

---

## 6. LinkedIn Post for Viral Reach

```
🎯 The inference optimization stack has a missing layer.

Most LLM efficiency work focuses on:
• Hardware (Inferentia, TPU, H100)
• Engines (vLLM, TensorRT-LLM)
• Compression (INT8, pruning)

But there's a 4th dimension nobody's optimizing: HOW MUCH compute per token.

For Mixture-of-Experts models (GPT-4, Gemini, Mixtral), we asked:
"Why use 2 experts for 'the' and 'cat' equally?"

Answer: You shouldn't.

We built Adaptive-K routing:
✅ Simple tokens → 1 expert (low entropy = high confidence)
✅ Complex tokens → 2 experts (high entropy = needs more compute)

Results on Mixtral 8×7B:
📉 31.0% less compute
📊 <1% perplexity change
⚡ Zero retraining needed

It's multiplicative with other optimizations:
vLLM savings × Adaptive-K savings = big numbers

🔗 Try it: pip install adaptive-k-routing
📄 Paper: [link]
🎮 Demo: [link]

What layer are you optimizing? 👇

#AI #LLM #Optimization #MixtureOfExperts #MachineLearning
```

---

## Follow-up Tracking

| Target | Date Sent | Response | Status | Next Action |
|--------|-----------|----------|--------|-------------|
| Saša Zelenović (Red Hat) | | | Pending | LinkedIn connect |
| AWS ML Blog | | | Pending | Email pitch |
| IBM Think | | | Pending | Email pitch |
| Ultralytics | | | Pending | GitHub discussion |
| HuggingFace | | | Pending | RFC draft |

---

*Last updated: January 17, 2026*
