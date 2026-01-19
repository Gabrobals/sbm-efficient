# Hacker News Post - Adaptive-K

> Da postare su: https://news.ycombinator.com/submit
> Orario consigliato: 9-10 AM EST (15-16 ora italiana)

---

## Titolo (scegli uno):

**A)** `Show HN: Adaptive-K – Cut MoE inference costs 30-50% with entropy-guided routing`

**B)** `Show HN: Open-source dynamic expert selection for Mixtral/Qwen-MoE`

**C)** `Show HN: We reduced Mixtral compute by 52% using routing entropy`

---

## URL da inserire:

```
https://github.com/Gabrobals/sbm-efficient
```

---

## Testo del post (copia-incolla):

```
Hey HN,

I built Adaptive-K, an open-source technique to reduce Mixture-of-Experts inference costs by dynamically selecting fewer experts when the router is confident.

The key insight: routing entropy predicts when fewer experts are sufficient. Low entropy = confident routing = use K=1 instead of K=8. High entropy = uncertain = use full K.

Results on production models:
- Mixtral 8x7B: 31.0% compute reduction, 99.8% accuracy retained
- Qwen-MoE: 32.4% reduction, 99.9% accuracy  
- OLMoE-1B-7B: 24.7% reduction, 99.7% accuracy

The algorithm is simple (~50 lines):

    H = -sum(p * log(p))  # router entropy
    K = 1 if H < 0.6 else (2 if H < 1.2 else 4)
    output = top_k(experts, K)

I've submitted a PR to TensorRT-LLM (#10672) and published an SDK on PyPI.

Links:
- Paper: https://adaptive-k.vercel.app/paper.html
- Code: https://github.com/Gabrobals/sbm-efficient
- Landing page: https://adaptive-k.vertexdata.it
- TensorRT-LLM PR: https://github.com/NVIDIA/TensorRT-LLM/pull/10672
- PyPI SDK: https://pypi.org/project/adaptive-k/

Happy to answer questions about the implementation or results.
```

---

## Risposte preparate per critiche comuni:

### "Why not just use speculative decoding?"
> Speculative decoding and Adaptive-K are orthogonal optimizations. Speculative decoding reduces autoregressive overhead, while Adaptive-K reduces per-token expert computation. They can be combined.

### "Did you test on real latency, not just FLOPs?"
> Yes, we measured wall-clock latency. The FLOPs reduction translates to ~25-40% latency improvement depending on the model and hardware. The gap is due to memory bandwidth being the bottleneck on some configurations.

### "What about accuracy on harder benchmarks?"
> We tested on WikiText-2 perplexity which is standard for MoE evaluation. The technique is designed to be conservative - it only reduces K when entropy is very low (router is very confident). We're planning to add MMLU and other benchmarks.

### "Isn't this just early exit?"
> Similar intuition, different mechanism. Early exit skips layers, Adaptive-K skips experts within a layer. The entropy threshold approach is also more principled than learned exit gates.

### "How do you handle batching?"
> Each token in the batch can have different K. We use masked sparse execution - the implementation handles variable K efficiently. See the TensorRT-LLM PR for the batched CUDA kernel.

---

## Tips:

1. **Rispondi SUBITO** ai commenti (entro 10-15 min)
2. **Sii umile** - accetta critiche costruttive
3. **Upvote** i commenti interessanti (anche critici)
4. **Non spammare** link commerciali nei commenti
5. **Se va in front page**, aspettati 50-100 GitHub stars nel primo giorno
