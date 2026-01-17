# arXiv Submission Instructions

## Pre-submission Checklist

### Files to Upload
```
arxiv_paper/
├── main.tex          # Main paper
├── references.bib    # Bibliography
```

### arXiv Category
**Primary**: `cs.LG` (Machine Learning)
**Cross-list**: `cs.CL` (Computation and Language)

## Submission Steps

### 1. Create arXiv Account (if needed)
- Go to https://arxiv.org/user/register
- Use institutional email if available (faster endorsement)

### 2. Start New Submission
- Go to https://arxiv.org/submit
- Select "Computer Science > Machine Learning (cs.LG)"
- Cross-list to cs.CL

### 3. Upload Files
- Upload `main.tex` as primary file
- Upload `references.bib`
- arXiv will auto-compile LaTeX

### 4. Fill Metadata

**Title**: 
```
Entropy-Guided Dynamic Expert Selection in Mixture-of-Experts Models
```

**Authors**:
```
Gabriele Balsamo
```

**Abstract**:
```
We present Adaptive-K routing, a method that dynamically selects the number of experts in Mixture-of-Experts (MoE) models based on routing entropy. Instead of using a fixed top-k experts per token, our approach uses fewer experts when the router is confident (low entropy) and more experts when uncertain (high entropy). We validate this approach on three production MoE architectures: Mixtral 8x7B (52.5% compute reduction), Qwen-MoE (32.4%), and OLMoE-1B-7B (24.7%). Furthermore, we demonstrate that Adaptive-K composes multiplicatively with other optimizations, achieving up to 96% total compute reduction. Our method is a drop-in replacement for existing MoE routing and requires no model retraining.
```

**Comments**:
```
11 pages, 9 tables, code: https://github.com/Gabrobals/sbm-efficient, PyPI: adaptive-k-routing
```

**ACM Classification** (optional):
- I.2.6 Learning
- I.2.7 Natural Language Processing

### 5. License
Select: **arXiv.org perpetual, non-exclusive license**
(Recommended for maximum visibility)

### 6. Submit and Wait
- Processing takes 1-2 business days
- Check status at https://arxiv.org/user

## Post-Submission

### Expected arXiv ID Format
```
arXiv:2601.XXXXX
```

### Update Links
After receiving arXiv ID, update:
1. GitHub README with arXiv link
2. LinkedIn post with arXiv link
3. TensorRT-LLM PR with paper reference

## Alternative: Submit to OpenReview

If targeting a conference (NeurIPS, ICML, ICLR):

1. Go to https://openreview.net
2. Find relevant venue/workshop
3. Submit PDF version

### Potential Venues
- **NeurIPS 2026**: Deadline ~May 2026
- **ICML 2026**: Deadline ~Jan 2026
- **ICLR 2027**: Deadline ~Oct 2026
- **EMNLP 2026**: Deadline ~June 2026

### Workshops to Consider
- Efficient Natural Language and Speech Processing (ENLSP)
- Machine Learning and Systems (MLSys)
- Sparsity in Neural Networks

---

## Compile Locally (Optional)

To verify LaTeX compiles correctly:

```bash
cd arxiv_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use Overleaf:
1. Go to https://www.overleaf.com
2. New Project > Upload Project
3. Upload main.tex and references.bib
4. Compile and verify PDF
