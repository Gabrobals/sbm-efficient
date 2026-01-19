# DeepSeek Response - Issue #1089

**Date**: January 19, 2026
**Status**: 🟢 HIGH PRIORITY - Under Internal Assessment
**Expected Feedback**: 1-2 weeks (by Feb 2, 2026)

## Their Response Summary

1. **Quality Recognition**: "impressive proposal", "significant reference value"
2. **Internal Process**: Forwarded to model architecture + performance optimization teams
3. **Focus Areas**: Integration feasibility, accuracy impact, performance benefits
4. **Request**: Share open-source repository link
5. **Community Invitation**: DeepSeek Global Launch Community

## Action Items

- [x] Prepare reply with all resources
- [ ] Send reply (copy from below)
- [ ] Join DeepSeek Global Launch Community
- [ ] Prepare DeepSeek-V3 specific benchmarks (use `scripts/deepseek_api_profiling.py`)
- [ ] Monitor for their technical feedback (~Feb 2)

## DeepSeek-V3 Architecture Notes

For Adaptive-K integration planning:

| Parameter | Value |
|-----------|-------|
| Total params | 671B |
| Active params | 37B |
| Experts per layer | 256 |
| Top-K routing | K=8 (fixed) |
| Routing type | Aux-loss-free load balancing |

**Integration opportunity**: Replace fixed K=8 with entropy-adaptive K=[2,4,8]

---

## REPLY TO SEND

Copy and paste this into the GitHub issue:

---

Thank you for the thoughtful and encouraging response! I'm genuinely excited that Adaptive-K has caught your team's attention.

### Resources

Here are all the materials for your evaluation:

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/Gabrobals/sbm-efficient |
| **Citable DOI (Zenodo)** | https://doi.org/10.5281/zenodo.18282008 |
| **PyPI Package** | https://pypi.org/project/adaptive-k-routing/ |
| **Live Demo** | https://huggingface.co/spaces/Gabrobals/adaptive-k-demo |
| **Technical Paper** | https://adaptive-k.vercel.app/paper.html |
| **TensorRT-LLM PR** | https://github.com/NVIDIA/TensorRT-LLM/pull/10672 |

### DeepSeek-V3 Specific Integration Points

For your architecture specifically, I'd highlight:

1. **Current state**: K=8 fixed routing with aux-loss-free load balancing
2. **Proposed**: Entropy-adaptive K=[2,4,8] based on router logits
3. **Integration point**: After router MLP, before expert selection
4. **Expected savings**: 30-50% FLOPs with <1% quality degradation

The key insight is that many tokens (especially in "easy" contexts) don't need all 8 experts. Router entropy naturally indicates confidence - low entropy means fewer experts suffice.

### I Can Prepare

If it would help the evaluation, I'm happy to:
- Run benchmarks using DeepSeek-V3 API to profile entropy distributions across task types
- Provide a proof-of-concept patch for your router module
- Share detailed analysis on quality vs efficiency tradeoffs

### Community Invitation

I'd love to join the DeepSeek Global Launch Community! It would be great to connect with others working on model efficiency and share insights from the Adaptive-K development process.

Looking forward to the technical discussion and deeper collaboration.

Best regards,
**Gabriele Balsamo**
amministrazione@vertexdata.it

---

## Benchmark Plan (Before Their Response)

Run `scripts/deepseek_api_profiling.py` to gather data:

```bash
# Set API key
export DEEPSEEK_API_KEY="your-key"

# Run profiling
python scripts/deepseek_api_profiling.py --output results/deepseek_profile.json

# Or via OpenRouter/Together
export TOGETHER_API_KEY="your-key"
python scripts/deepseek_api_profiling.py --provider together
```

**Goal**: Show entropy distribution across prompt types → justify Adaptive-K thresholds

## Community Notes

**DeepSeek Global Launch Community**:
- Purpose: Researchers, engineers, enthusiasts discussing model architecture
- Value: Relationship building, early access to developments, credibility
- Action: Accept invitation, introduce yourself, share Adaptive-K journey

## Timeline

| Date | Action | Status |
|------|--------|--------|
| Jan 19 | Received response | ✅ |
| Jan 20 | Send reply with resources | ⏳ |
| Jan 21-28 | Run DeepSeek API profiling | ⏳ |
| Jan 25 | Join community, introduce | ⏳ |
| ~Feb 2 | Expect technical feedback | ⏳ |
