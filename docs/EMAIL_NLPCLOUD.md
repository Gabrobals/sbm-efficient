# Email per NLP Cloud

## A: experts@nlpcloud.com
## Oggetto: MoE Routing Optimization - Complementing Your Inference Guide

---

Hi Julien,

I read your excellent article on LLM inference optimization techniques (https://nlpcloud.com/llm-inference-optimization-techniques.html) and found it very comprehensive.

I noticed the article covers batching, attention optimization, KV caching, quantization, and speculative decoding—but doesn't mention **Mixture-of-Experts routing optimization**, which is increasingly relevant given that Mixtral, DeepSeek-V2, Grok, and most 2024-2025 frontier models use MoE architectures.

We've developed **Adaptive-K**, an entropy-based dynamic expert selection method that reduces MoE inference costs by 30-50% with minimal accuracy loss. The key insight: not all inputs need the same number of experts.

I've written a technical deep-dive that complements your article:
**"MoE Inference Optimization: The Missing Piece"**
https://adaptive-k.vertexdata.it/blog/moe-inference-optimization.html

Would you be interested in:
1. Linking to this as a complementary resource in your article, or
2. A guest post on NLP Cloud covering MoE routing optimization?

The approach is open-source (MIT license) and works alongside vLLM, TensorRT-LLM, and other engines you mention.

Best regards,

[Il tuo nome]
Vertex Data
https://adaptive-k.vertexdata.it

---

## Note:
- Personalizza con il tuo nome
- Julien è il CTO di NLP Cloud (citato nell'articolo)
- L'email è breve e diretta al punto
- Offre valore (contenuto complementare) invece di chiedere solo un link
