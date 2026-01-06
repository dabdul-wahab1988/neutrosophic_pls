# Neutrosophic PLS - Mathematical Documentation

## Overview

This directory contains comprehensive mathematical documentation for all modules in the Neutrosophic PLS (N-PLS) package that involve mathematical foundations.

---

## Document Index

### User Guides

| Document | Description |
|----------|-------------|
| [Interactive_Mode_Guide.md](Interactive_Mode_Guide.md) | **Start here!** Step-by-step guide for non-programmers |

### Mathematical Foundations

| Document | Module | Description |
|----------|--------|-------------|
| [NVIP_Mathematics.md](NVIP_Mathematics.md) | `vip.py` | Neutrosophic VIP decomposition with L2-norm proof |
| [Encoders_Mathematics.md](Encoders_Mathematics.md) | `encoders.py` | T-I-F encoding methods (RPCA, Wavelet, Quantile, etc.) |
| [Models_Mathematics.md](Models_Mathematics.md) | `model.py` | NPLS, NPLSW, and PNPLS algorithms |
| [Algebra_Mathematics.md](Algebra_Mathematics.md) | `algebra.py` | Neutrosophic algebra operations |
| [Metrics_Mathematics.md](Metrics_Mathematics.md) | `metrics.py` | Performance metrics (RMSE, R², F1, etc.) |

---

## Quick Reference

### Core Mathematical Relationships

**NVIP L2 Decomposition:**
$$\text{VIP}_{\text{aggregate}} = \sqrt{\text{VIP}_T^2 + \text{VIP}_I^2 + \text{VIP}_F^2}$$

**Sample Reliability Weight (NPLS):**
$$w_i = \exp(-\lambda_I \bar{I}_i - \lambda_F \bar{F}_i)$$

**RPCA Decomposition:**
$$X = L + S + E \quad \text{(Low-rank + Sparse + Error)}$$

**Neutrosophic Norm:**
$$\|(T, I, F)\|_w = \sqrt{w_T T^2 + w_I I^2 + w_F F^2}$$

**RPD (Ratio of Performance to Deviation):**
$$\text{RPD} = \frac{\sigma_y}{\text{RMSEP}} = \frac{1}{\sqrt{1 - R^2}}$$

---

## Module-to-Document Mapping

```
neutrosophic_pls/
├── algebra.py      → Algebra_Mathematics.md
├── encoders.py     → Encoders_Mathematics.md
├── metrics.py      → Metrics_Mathematics.md
├── model.py        → Models_Mathematics.md
└── vip.py          → NVIP_Mathematics.md
```

---

## Document Format

All documents are written in Markdown with LaTeX equations, making them suitable for:

- ✅ **Academic publications** - LaTeX equations render properly
- ✅ **Documentation websites** - GitHub-flavored markdown
- ✅ **PDF generation** - Via pandoc: `pandoc file.md -o file.pdf`
- ✅ **GitHub rendering** - Equations render on GitHub

### Converting to PDF

Using pandoc with LaTeX:

```bash
# Single document
pandoc NVIP_Mathematics.md -o NVIP_Mathematics.pdf --pdf-engine=xelatex

# All documents
for f in *.md; do
    pandoc "$f" -o "${f%.md}.pdf" --pdf-engine=xelatex
done
```

### Converting to HTML

```bash
pandoc NVIP_Mathematics.md -o NVIP_Mathematics.html --mathjax
```

---

## Key Theorems and Proofs

### 1. NVIP L2 Decomposition Theorem

**Location:** [NVIP_Mathematics.md](NVIP_Mathematics.md), Section 5

**Statement:** For any feature j, the aggregate VIP satisfies:
$$\text{VIP}_{\text{aggregate}}(j) = \sqrt{\text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}$$

**Proof:** Complete formal proof provided in the document.

### 2. EM-NIPALS Convergence

**Location:** [Models_Mathematics.md](Models_Mathematics.md), Section 5

**Statement:** The EM-NIPALS algorithm converges to a local maximum of the weighted likelihood.

### 3. RPCA Exact Recovery

**Location:** [Encoders_Mathematics.md](Encoders_Mathematics.md), Section 3

**Statement:** Under incoherence conditions, RPCA exactly recovers L and S.

---

## Symbol Reference

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| $T$ | Truth channel | $\mathbb{R}$ or $[0,1]$ |
| $I$ | Indeterminacy channel | $[0, 1]$ |
| $F$ | Falsity channel | $[0, 1]$ |
| $\omega_T, \omega_I, \omega_F$ | Channel weights | $[0, 1]$ |
| $\lambda_F, \lambda_I$ | Penalty parameters | $\mathbb{R}^+$ |
| VIP | Variable Importance in Projection | $[0, \infty)$ |
| $R^2$ | Coefficient of determination | $(-\infty, 1]$ |
| RPD | Ratio of Performance to Deviation | $[0, \infty)$ |

---

## Citation

If you use these mathematical foundations in academic work, please cite:

```bibtex
@software{asare_npls_2026,
  author = {Asare, Ebenezer Aquisman and Abdul-Wahab, Dickson},
  title = {Neutrosophic Partial Least Squares (N-PLS)},
  year = {2026},
  version = {1.0.2},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18131413},
  url = {https://doi.org/10.5281/zenodo.18131413}
}
```

---

## Contributing

To add mathematical documentation for a new module:

1. Create a new `ModuleName_Mathematics.md` file
2. Follow the existing structure:
   - Introduction
   - Mathematical formulation
   - Theorems and proofs
   - Algorithm pseudocode
   - Properties and edge cases
   - References
3. Update this index file

---

*Last updated: December 2025*
