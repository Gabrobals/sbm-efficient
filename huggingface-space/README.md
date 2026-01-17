---
title: Adaptive-K Demo
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: true
license: mit
short_description: Dynamic Expert Selection for MoE Models
tags:
  - moe
  - mixture-of-experts
  - inference-optimization
  - llm
  - adaptive-routing
---

# Adaptive-K: Dynamic Expert Selection for MoE

This demo shows how **Adaptive-K** dynamically selects the number of experts based on input complexity, reducing inference costs by 30-50% while maintaining accuracy.

## How it works

1. **Input text** → Router computes routing probabilities
2. **Entropy calculation** → Measures router uncertainty
3. **K selection** → Low entropy = fewer experts, High entropy = more experts
4. **Cost savings** → Visualize the compute reduction

## Links

- 📄 [Paper](https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf)
- 📖 [Whitepaper](https://adaptive-k.vertexdata.it/whitepaper.html)
- 💻 [GitHub](https://github.com/Gabrobals/sbm-efficient)
- 📦 [PyPI](https://pypi.org/project/adaptive-k-routing/)
