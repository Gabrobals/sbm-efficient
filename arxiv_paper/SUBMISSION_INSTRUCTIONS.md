# arXiv Submission Instructions

## ✅ Paper Ready for Submission

**Main file**: `main_neurips.tex` (NeurIPS format with figures)
**DOI**: [10.5281/zenodo.18282008](https://doi.org/10.5281/zenodo.18282008)

---

## 📁 Files to Upload

```
arxiv_paper/
├── main_neurips.tex     # Main paper (rename to main.tex for arXiv)
├── neurips_2024.sty     # Style file
├── references.bib       # Bibliography
└── figures/
    ├── entropy_distribution.pdf
    ├── architecture.pdf
    ├── results_comparison.pdf
    ├── multiplicative_savings.pdf
    └── entropy_vs_perplexity.pdf
```

## 📊 Regenerate Figures

```bash
cd arxiv_paper/scripts
pip install matplotlib numpy
python generate_figures.py
```

---

## Submission Options

### Option 1: Overleaf (Recommended)

1. Go to [Overleaf](https://www.overleaf.com) → New Project
2. Upload all files from `arxiv_paper/`
3. Set `main_neurips.tex` as main document
4. Compile with pdfLaTeX
5. Download PDF for arXiv or submit directly via Overleaf

### Option 2: arXiv Direct

1. Rename `main_neurips.tex` → `main.tex`
2. Zip all files:
   ```bash
   cd arxiv_paper
   zip -r submission.zip main.tex neurips_2024.sty references.bib figures/
   ```
3. Upload to https://arxiv.org/submit

### Option 3: Local LaTeX

```bash
cd arxiv_paper
pdflatex main_neurips.tex
bibtex main_neurips
pdflatex main_neurips.tex
pdflatex main_neurips.tex
```

---

## arXiv Metadata

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
We present Adaptive-K routing, a method that dynamically selects the number of experts in Mixture-of-Experts (MoE) models based on routing entropy. Instead of using a fixed top-k experts per token, our approach uses fewer experts when the router is confident (low entropy) and more experts when uncertain (high entropy). We validate this approach on four production MoE architectures: Nemotron 3 Nano (33.3% compute reduction), Mixtral 8x7B (31.0%), Qwen-MoE (32.4%), and OLMoE-1B-7B (24.7%). Furthermore, we demonstrate that Adaptive-K composes multiplicatively with other optimizations, achieving up to 90.7% total compute reduction. Our method is a drop-in replacement for existing MoE routing and requires no model retraining.
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
