# Neutrosophic PLS Models

## Mathematical Foundation for NPLS, NPLSW, and PNPLS

**Author:** Ebenezer Aquisman Asare and Dickson Abdul-Wahab  
**Date:** December 2025  
**Version:** 1.0
---

## 1. Introduction

This document provides the complete mathematical foundation for the three Neutrosophic Partial Least Squares (N-PLS) model variants:

1. **NPLS** — Standard Neutrosophic PLS with sample weighting
2. **NPLSW** — Reliability-Weighted Neutrosophic PLS
3. **PNPLS** — Probabilistic Neutrosophic PLS (Element-wise EM-NIPALS)

---

## 2. Classical PLS Background

### 2.1 The PLS Regression Problem

Given:

- $X \in \mathbb{R}^{n \times p}$ — predictor matrix
- $Y \in \mathbb{R}^{n \times m}$ — response matrix

Find latent structures that maximize covariance between X and Y.

### 2.2 NIPALS Algorithm

The Nonlinear Iterative Partial Least Squares (NIPALS) algorithm:

**For each component a = 1, ..., A:**

1. Initialize: $u_a = Y_{:,1}$ (first column of $Y_a$)

2. Repeat until convergence:

   a. **X-side weight:**
   $$w_a = \frac{X_a^T u_a}{\|X_a^T u_a\|}$$

   b. **X-side score:**
   $$t_a = X_a w_a$$

   c. **Y-side weight:**
   $$c_a = \frac{Y_a^T t_a}{\|Y_a^T t_a\|}$$

   d. **Y-side score:**
   $$u_a = Y_a c_a$$

3. **X-loading:**
   $$p_a = \frac{X_a^T t_a}{t_a^T t_a}$$

4. **Y-loading:**
   $$q_a = \frac{Y_a^T t_a}{t_a^T t_a}$$

5. **Deflation:**
   $$X_{a+1} = X_a - t_a p_a^T$$
   $$Y_{a+1} = Y_a - t_a q_a^T$$

### 2.3 Regression Coefficients

The final regression coefficients:

$$B = W(P^T W)^{-1} Q^T$$

Prediction:

$$\hat{Y} = X \cdot B$$

---

## 3. NPLS: Standard Neutrosophic PLS

### 3.1 Overview

NPLS extends classical PLS by incorporating sample-level reliability weights derived from the Indeterminacy and Falsity channels.

### 3.2 Sample Weight Computation

Given neutrosophic tensor $\mathcal{X} \in \mathbb{R}^{n \times p \times 3}$:

**Mean reliability per sample:**

$$\bar{I}_i = \frac{1}{p} \sum_{j=1}^{p} I_{ij}$$

$$\bar{F}_i = \frac{1}{p} \sum_{j=1}^{p} F_{ij}$$

**Sample weight:**

$$r_i = \exp\left(-\lambda_I \bar{I}_i - \lambda_F \bar{F}_i\right)$$

Where:

- $\lambda_I$ = indeterminacy penalty (default: 0.3)
- $\lambda_F$ = falsity penalty (default: 0.5)

**Normalization:**

$$w_i = \frac{r_i}{\sum_{k=1}^{n} r_k} \cdot n$$

This ensures $\sum_i w_i = n$.

### 3.3 Weighted NIPALS

Replace dot products with weighted versions:

**Weighted X-side weight:**

$$w_a = \frac{X_a^T (D_w u_a)}{\|X_a^T (D_w u_a)\|}$$

Where $D_w = \text{diag}(w_1, ..., w_n)$ is the sample weight matrix.

**Weighted score:**

$$t_a = X_a w_a$$

**Weighted centering:**

$$\bar{X}_w = \frac{\sum_i w_i \cdot X_i}{\sum_i w_i}$$

### 3.4 Properties

- Samples with high I or F contribute less to model fitting
- Preserves PLS structure (latent variable interpretation)
- Reduces impact of noisy samples without removing them

---

## 4. NPLSW: Reliability-Weighted NPLS

### 4.1 Overview

NPLSW uses a more sophisticated reliability weighting scheme based on the reliability matrix.

### 4.2 Reliability Matrix

$$R_{ij} = 1 - \max(I_{ij}, F_{ij})$$

This gives element-wise reliability, but NPLSW aggregates to sample level.

### 4.3 Sample Reliability Score

**Geometric mean reliability:**

$$r_i = \left(\prod_{j=1}^{p} R_{ij}\right)^{1/p}$$

Or equivalently in log-space:

$$\log r_i = \frac{1}{p} \sum_{j=1}^{p} \log R_{ij}$$

**Soft weighting:**

$$w_i = r_i^\alpha$$

Where $\alpha$ controls the sharpness of weighting (default: 1.0).

### 4.4 Weighted NIPALS with Reliability

The NIPALS iteration uses reliability-weighted operations:

**Score computation:**

$$t_a^{(w)} = (D_{r^\alpha})^{1/2} X_a w_a$$

**Weight update:**

$$w_a = \frac{X_a^T D_{r^\alpha} u_a}{\|X_a^T D_{r^\alpha} u_a\|}$$

### 4.5 Mathematical Justification

The weighted least squares objective:

$$\min_{w, p} \sum_{i=1}^{n} w_i \|X_{i,:} - t_i p^T\|^2$$

Becomes the standard objective when $w_i = 1$ for all i.

---

## 5. PNPLS: Probabilistic Neutrosophic PLS

### 5.1 Overview

PNPLS is the most sophisticated variant, handling noise at the **element level** using an EM-NIPALS algorithm.

### 5.2 Generative Model

Assume a heteroscedastic noise model:

$$X_{ij} = \sum_{a=1}^{A} t_{ia} p_{ja} + \epsilon_{ij}$$

Where:

$$\epsilon_{ij} \sim \mathcal{N}(0, \sigma_{ij}^2)$$

The variance is element-specific:

$$\sigma_{ij}^2 \propto \exp(\lambda_F \cdot F_{ij})$$

### 5.3 Precision Weights

Define the **precision matrix** (inverse variance):

$$W_{ij} = \exp(-\lambda_F \cdot F_{ij} \cdot \gamma)$$

Where:

- $\lambda_F$ = falsity sensitivity (default: 0.5)
- $\gamma$ = scaling factor (default: 3.0)

**Properties:**

- $W_{ij} \in (0, 1]$
- High $F_{ij}$ → low weight (less trusted)
- Low $F_{ij}$ → weight ≈ 1 (fully trusted)

### 5.4 EM-NIPALS Algorithm

**E-Step (Imputation):**

Replace unreliable values with expected values:

$$X_{ij}^{\text{imp}} = W_{ij} \cdot X_{ij}^{\text{obs}} + (1 - W_{ij}) \cdot X_{ij}^{\text{pred}}$$

Where:

$$X_{ij}^{\text{pred}} = \sum_{a=1}^{A} t_{ia} p_{ja}$$

**M-Step (Maximization):**

Run standard NIPALS on the imputed matrix $X^{\text{imp}}$.

**Iterative refinement:**

Repeat E and M steps until convergence:

$$\|X^{\text{imp},(k)} - X^{\text{imp},(k-1)}\|_F < \tau$$

### 5.5 Complete EM-NIPALS Algorithm

```
Algorithm: EM-NIPALS for PNPLS
Input: X_obs, W (precision weights), n_components, max_iter, tol
Output: T (scores), P (loadings), W_x (weights)

1. Initialize: X_imp = X_obs
2. Compute weighted mean: μ = (W ⊙ X_obs).sum(0) / W.sum(0)
3. Center: X_c = X_imp - μ

For iter = 1 to max_iter:
    X_prev = X_imp.copy()
    
    For component a = 1 to n_components:
        # Standard NIPALS
        u = Y[:, 0]
        For nipals_iter = 1 to 100:
            w = normalize(X_c.T @ u)
            t = X_c @ w
            c = normalize(Y.T @ t)
            u_new = Y @ c
            if converged(u, u_new): break
            u = u_new
        
        p = (X_c.T @ t) / (t.T @ t)
        q = (Y.T @ t) / (t.T @ t)
        
        # Reconstruct
        X_rec = t @ p.T
        
        # E-step: Impute
        X_imp = W ⊙ X_obs + (1 - W) ⊙ X_rec
        
        # Deflate
        X_c = X_c - t @ p.T
        Y = Y - t @ q.T
    
    # Check overall convergence
    if ||X_imp - X_prev||_F < tol:
        break

Return T, P, W_x
```

### 5.6 Mathematical Properties

**Theorem (EM Convergence):**

The EM-NIPALS algorithm converges to a local maximum of the weighted likelihood:

$$\mathcal{L}(T, P | X, W) = -\frac{1}{2} \sum_{i,j} W_{ij} \left(X_{ij} - \sum_a t_{ia} p_{ja}\right)^2$$

**Proof sketch:**

- E-step minimizes reconstruction error for fixed T, P
- M-step finds optimal T, P for current imputation
- Both steps are non-decreasing in likelihood
- Bounded above → convergence guaranteed

### 5.7 Numerical Stability

To prevent numerical instability:

1. **Clamp F values:** $F' = \text{clip}(F, 0, 1)$
2. **Floor weights:** $W' = \max(W, 0.05)$
3. **Normalize scores:** If $\|t\| > 10^6$, renormalize
4. **Regularize inversion:** $(P^T W)^{-1}$ → $(P^T W + \epsilon I)^{-1}$

---

## 6. Regression Coefficient Computation

### 6.1 Standard Formula

For all NPLS variants:

$$B = W_x (P^T W_x)^{-1} Q^T$$

Where:

- $W_x \in \mathbb{R}^{p \times A}$ = X-weights
- $P \in \mathbb{R}^{p \times A}$ = X-loadings
- $Q \in \mathbb{R}^{A \times m}$ = Y-loadings

### 6.2 Regularized Version (PNPLS)

To handle ill-conditioning:

$$B = W_x (P^T W_x + \epsilon I)^{-1} Q^T$$

Or using pseudo-inverse:

$$B = W_x (P^T W_x)^{\dagger} Q^T$$

### 6.3 Prediction

$$\hat{Y} = (X - \bar{X}) \cdot B + \bar{Y}$$

---

## 7. Comparison of Variants

| Aspect | NPLS | NPLSW | PNPLS |
|--------|------|-------|-------|
| Weighting level | Sample | Sample | Element |
| Weight source | Mean I/F | Reliability matrix | F only |
| Algorithm | Weighted NIPALS | Reliability NIPALS | EM-NIPALS |
| Complexity | O(npA) | O(npA) | O(npA × iter) |
| Best for | General noise | Sample-level issues | Localized corruption |

---

## 8. Theoretical Guarantees

### 8.1 Consistency

**Theorem:** As $n \to \infty$, NPLS estimates converge to population PLS parameters when:

- Sample weights are bounded away from 0
- True model has finite latent components

### 8.2 Efficiency

**Theorem:** NPLSW achieves lower asymptotic variance than unweighted PLS when:

- Reliability weights correctly identify low-quality samples
- Weights are proportional to inverse noise variance

### 8.3 Robustness

**Theorem:** PNPLS breakdown point is at least $(1 - \max_i w_i)$, meaning it can tolerate up to this fraction of corrupted elements per feature.

---

## 9. Symbol Table

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $X$ | Predictor matrix (Truth channel) | $n \times p$ |
| $Y$ | Response matrix | $n \times m$ |
| $T$ | X-scores | $n \times A$ |
| $U$ | Y-scores | $n \times A$ |
| $P$ | X-loadings | $p \times A$ |
| $Q$ | Y-loadings | $A \times m$ |
| $W_x$ | X-weights | $p \times A$ |
| $B$ | Regression coefficients | $p \times m$ |
| $w_i$ | Sample weight | scalar |
| $W_{ij}$ | Element precision weight | scalar |
| $\lambda_F$ | Falsity penalty | scalar |
| $\lambda_I$ | Indeterminacy penalty | scalar |

---

## 10. References

1. Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130.

2. Geladi, P., & Kowalski, B. R. (1986). Partial least-squares regression: a tutorial. *Analytica Chimica Acta*, 185, 1-17.

3. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society*, 39(1), 1-38.

4. Smarandache, F. (1998). Neutrosophy: Neutrosophic Probability, Set, and Logic. American Research Press.

5. Hubert, M., & Engelen, S. (2004). Robust PCA and classification in biosciences. *Bioinformatics*, 20(11), 1728-1736.

---

*Document generated for the Neutrosophic PLS package.*
