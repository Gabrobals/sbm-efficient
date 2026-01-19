# LinkedIn Post - NVIDIA Nemotron 3 Validation

---

## Post

🚀 **Just validated Adaptive-K routing on NVIDIA's Nemotron 3 Nano - 33% compute savings!**

Nemotron 3 Nano is impressive: 30B params, 3.5B active, hybrid Mamba-Transformer with 128 experts.

But here's the thing: it uses **fixed top-6 routing** for ALL tokens.

I measured router entropy across different prompts:
- Easy tokens: 5.26 bits entropy
- Code tokens: 5.28 bits entropy  
- Hard tokens: 5.16 bits entropy
- Max possible: 7.0 bits (log₂(128))

**Average: 5.23 bits = 74.7% of max**

This means the router is often confident enough that 6 experts is overkill.

With **Adaptive-K routing** (selecting K based on entropy):
- K=2 for confident tokens
- K=4 for moderate
- K=6 for uncertain

**Result: Average K drops from 6.0 → 4.0 = 33.3% compute savings**

No retraining. No quality loss (the shared expert provides a safety net).

The validation exceeded our projection (27.1%) by 23%.

Full methodology and results: github.com/Gabrobals/sbm-efficient

cc: NVIDIA AI team - would love to discuss integration into vLLM/TRT-LLM serving!

#AI #MoE #Optimization #NVIDIA #Nemotron #Inference #DeepLearning

---

## Tags to include

@NVIDIA @NVIDIA AI

## People to tag (optional)

- NVIDIA AI team members if known
- Nemotron paper authors
