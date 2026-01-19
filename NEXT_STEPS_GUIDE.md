# Step-by-Step: TensorRT-LLM PR, arXiv, LinkedIn

## 1. TensorRT-LLM Pull Request

### Step 1.1: Fork the Repository
1. Go to: https://github.com/NVIDIA/TensorRT-LLM
2. Click **"Fork"** button (top right)
3. Select your account as destination
4. Wait for fork to complete

### Step 1.2: Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/TensorRT-LLM.git
cd TensorRT-LLM
```

### Step 1.3: Create Feature Branch
```bash
git checkout -b feature/adaptive-k-routing
```

### Step 1.4: Copy the Implementation
Copy these files from `C:\Users\ottic\Desktop\SBM Efficent\tensorrt_llm_contribution\`:

- `routing.py` content → Add to `tensorrt_llm/_torch/modules/fused_moe/routing.py`
- `test_adaptive_k_routing.py` → Add to `tests/`

### Step 1.5: Commit and Push
```bash
git add .
git commit -m "feat: Add AdaptiveKMoeRoutingMethod for dynamic expert selection

Adds entropy-based adaptive K selection for MoE routing.
- 31.0% compute reduction on Mixtral 8x7B
- 32.4% on Qwen-MoE
- 24.7% on OLMoE-1B-7B

Closes #XXXX"

git push origin feature/adaptive-k-routing
```

### Step 1.6: Create Pull Request
1. Go to your fork on GitHub
2. Click **"Compare & pull request"**
3. Copy content from `tensorrt_llm_contribution/PULL_REQUEST.md`
4. Submit!

---

## 2. arXiv Submission

### Step 2.1: Go to arXiv
1. Open: https://arxiv.org/submit
2. Login (or create account)

### Step 2.2: Start New Submission
- Category: **cs.LG** (Machine Learning)
- Cross-list: **cs.CL** (Computation and Language)

### Step 2.3: Upload Files
Upload from `C:\Users\ottic\Desktop\SBM Efficent\arxiv_paper\`:
- `main.tex` (primary)
- `references.bib`

### Step 2.4: Fill Metadata

**Title:**
```
Entropy-Guided Dynamic Expert Selection in Mixture-of-Experts Models
```

**Authors:**
```
Gabriele Balsamo
```

**Abstract:**
```
We present Adaptive-K routing, a method that dynamically selects the number of experts in Mixture-of-Experts (MoE) models based on routing entropy. Instead of using a fixed top-k experts per token, our approach uses fewer experts when the router is confident (low entropy) and more experts when uncertain (high entropy). We validate this approach on four production MoE architectures: Nemotron 3 Nano (33.3% compute reduction), Mixtral 8x7B (31.0%), Qwen-MoE (32.4%), and OLMoE-1B-7B (24.7%), demonstrating significant efficiency gains without quality degradation. Our method is a drop-in replacement for existing MoE routing and requires no model retraining.
```

**Comments:**
```
10 pages, 8 tables
```

### Step 2.5: Submit
- License: arXiv perpetual non-exclusive
- Click Submit
- Wait 1-2 days for processing

---

## 3. LinkedIn Post

### Step 3.1: Go to LinkedIn
1. Open: https://www.linkedin.com
2. Click "Start a post"

### Step 3.2: Copy Post Text

```
🚀 33.3% Compute Reduction in Nemotron 3 Nano - Without Retraining!

I'm excited to share results from my research on Adaptive-K routing for Mixture-of-Experts (MoE) models.

The Problem: Current MoE models use a fixed number of experts (top-k) for ALL tokens, regardless of how confident the router is.

The Insight: Routing entropy varies significantly across tokens. Easy tokens have low entropy (router is confident) → they don't need many experts. Hard tokens have high entropy (router is uncertain) → they benefit from more experts.

The Solution: Adaptive-K dynamically selects K based on entropy:
• Low entropy → fewer experts → save compute
• High entropy → more experts → maintain quality

Results on Production Models:
• 🔥 Mixtral 8x7B: 31.0% reduction
• Qwen-MoE: 32.4% reduction
• OLMoE-1B-7B: 24.7% reduction

All with <0.5% quality impact!

Key Stats (Mixtral):
• 62% of tokens use K=1 (instead of K=2)
• Average K dropped from 2.0 to 1.38
• Perplexity increased by only 0.03

Why This Matters:
💰 Direct cost savings for inference
🌱 Lower energy consumption
⚡ Faster inference without model changes
🔧 Drop-in replacement - no retraining needed

The method is inspired by quantum mechanics concepts (entropy as "measurement uncertainty") from my SBM-Efficient research.

Would love to hear your thoughts! Have you experimented with dynamic expert selection in MoE models?

#MachineLearning #AI #MixtureOfExperts #Efficiency #LLM #DeepLearning #Research
```

### Step 3.3: Post
- Best time: Tuesday-Thursday, 8-10 AM
- Engage with comments within 24h

---

## Quick Links

| Action | URL |
|--------|-----|
| Fork TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM/fork |
| arXiv Submit | https://arxiv.org/submit |
| LinkedIn | https://www.linkedin.com |

## Files Location

```
C:\Users\ottic\Desktop\SBM Efficent\
├── tensorrt_llm_contribution\
│   ├── routing.py              # Implementation
│   ├── test_adaptive_k_routing.py  # Tests
│   └── PULL_REQUEST.md         # PR description
├── arxiv_paper\
│   ├── main.tex                # Paper
│   └── references.bib          # Bibliography
└── docs\
    └── LINKEDIN_POST.md        # Full post + alternatives
```
