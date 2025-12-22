# Neutrosophic Encoding Methods

## Mathematical Foundation for Truth-Indeterminacy-Falsity Encoding

**Author:** Dickson Abdul-Wahab and Ebenezer Aquisman Asare  
**Date:** December 2025  
**Version:** 1.0

---

## 1. Introduction

This document provides the mathematical foundation for the various encoding methods used to transform raw data into neutrosophic Truth-Indeterminacy-Falsity (T-I-F) representation.

### 1.1 Objective

Given a data matrix $X \in \mathbb{R}^{n \times p}$, produce a neutrosophic tensor:

$$\mathcal{X} \in \mathbb{R}^{n \times p \times 3}$$

Where:

- $\mathcal{X}_{:,:,0} = T$ (Truth) — the clean signal estimate
- $\mathcal{X}_{:,:,1} = I$ (Indeterminacy) — measurement uncertainty
- $\mathcal{X}_{:,:,2} = F$ (Falsity) — outlier/noise magnitude

---

## 2. Probabilistic Encoder

### 2.1 Overview

The probabilistic encoder uses statistical residuals from a fitted model to estimate T, I, and F.

### 2.2 Mathematical Formulation

**Step 1: Fit a base model**

Using Principal Component Analysis (PCA) or similar:

$$X \approx T \cdot P^T + E$$

Where:

- $T \in \mathbb{R}^{n \times k}$ = score matrix
- $P \in \mathbb{R}^{p \times k}$ = loading matrix
- $E$ = residual matrix

**Step 2: Compute residuals**

$$E = X - T \cdot P^T$$

**Step 3: Robust location and scale estimation**

For each feature j, compute:

$$\mu_j = \text{median}(E_{:,j})$$

$$\sigma_j = 1.4826 \cdot \text{MAD}(E_{:,j})$$

Where MAD is the Median Absolute Deviation:

$$\text{MAD}(x) = \text{median}(|x - \text{median}(x)|)$$

The factor 1.4826 makes MAD consistent with standard deviation for normal distributions.

**Step 4: Compute z-scores**

$$z_{ij} = \frac{|E_{ij} - \mu_j|}{\sigma_j + \epsilon}$$

**Step 5: Apply squashing function**

Using the hyperbolic tangent squasher:

$$\phi(z; s, \beta) = \tanh^\beta\left(\frac{z}{s}\right)$$

Where:

- $s$ = saturation threshold (typically 3-4)
- $\beta$ = sharpness parameter

**Step 6: Assign channels**

$$T = X \quad \text{(original values)}$$

$$F = \phi(z; s_F, \beta_F) \quad \text{(outlier degree)}$$

$$I = \min(I_{\text{base}}, 1 - F) \quad \text{(uncertainty)}$$

### 2.3 Properties

- $T, I, F \in [0, 1]$ for normalized inputs
- Higher residuals → Higher F
- High F implies low I (inverse relationship)

---

## 3. RPCA Encoder (Robust PCA)

### 3.1 Overview

The RPCA encoder uses Robust Principal Component Analysis to decompose data into low-rank (signal) and sparse (outlier) components.

### 3.2 Mathematical Formulation

**Problem:** Solve the convex optimization:

$$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad X = L + S$$

Where:

- $\|L\|_* = \sum_i \sigma_i(L)$ is the nuclear norm (sum of singular values)
- $\|S\|_1 = \sum_{ij} |S_{ij}|$ is the entry-wise L1 norm
- $\lambda$ = regularization parameter

**Default λ:**

$$\lambda = \frac{1}{\sqrt{\max(n, p)}}$$

### 3.3 Principal Component Pursuit (PCP) Algorithm

**Initialization:**

$$L_0 = 0, \quad S_0 = 0, \quad Y_0 = 0, \quad \mu_0 > 0$$

**Iteration k:**

1. Singular Value Thresholding for L:
   $$L_{k+1} = \mathcal{D}_{\mu_k^{-1}}(X - S_k + \mu_k^{-1} Y_k)$$

   Where $\mathcal{D}_\tau(M) = U \cdot \text{diag}(\max(\sigma - \tau, 0)) \cdot V^T$ for SVD $M = U\Sigma V^T$

2. Soft Thresholding for S:
   $$S_{k+1} = \mathcal{S}_{\lambda\mu_k^{-1}}(X - L_{k+1} + \mu_k^{-1} Y_k)$$

   Where $\mathcal{S}_\tau(x) = \text{sign}(x) \cdot \max(|x| - \tau, 0)$

3. Update dual variable:
   $$Y_{k+1} = Y_k + \mu_k(X - L_{k+1} - S_{k+1})$$

### 3.4 Channel Assignment

$$T = L \quad \text{(low-rank component = truth)}$$

$$F = \phi\left(\frac{|S|}{\sigma_S}\right) \quad \text{(sparse magnitude = falsity)}$$

$$I = \text{ambiguity}(R) \quad \text{(residual = indeterminacy)}$$

Where the ambiguity function measures uncertainty between sparse and residual:

$$\text{ambiguity} = \frac{|R|}{|R| + |S| + \epsilon}$$

---

## 4. Wavelet Encoder

### 4.1 Overview

The wavelet encoder uses multi-resolution wavelet decomposition to separate signal from noise at different frequency scales.

### 4.2 Mathematical Formulation

**Discrete Wavelet Transform (DWT):**

For each sample (row) x:

$$x = \sum_{k} c_{J,k} \phi_{J,k} + \sum_{j=1}^{J} \sum_{k} d_{j,k} \psi_{j,k}$$

Where:

- $\phi_{J,k}$ = scaling function at coarsest level J
- $\psi_{j,k}$ = wavelet functions at level j
- $c_{J,k}$ = approximation coefficients
- $d_{j,k}$ = detail coefficients

**Multi-level decomposition:**

$$[c_J, d_J, d_{J-1}, ..., d_1] = \text{DWT}(x, J)$$

### 4.3 Denoising via Thresholding

**Universal threshold (VisuShrink):**

$$\tau = \sigma \sqrt{2 \ln n}$$

Where σ is estimated from the finest detail coefficients:

$$\sigma = \frac{\text{MAD}(d_1)}{0.6745}$$

**Soft thresholding:**

$$\tilde{d}_{j,k} = \text{sign}(d_{j,k}) \cdot \max(|d_{j,k}| - \tau, 0)$$

### 4.4 Channel Assignment

$$T = \text{IDWT}([c_J, \tilde{d}_J, ..., \tilde{d}_1]) \quad \text{(denoised signal)}$$

$$F_j = \frac{|d_j - \tilde{d}_j|}{\max|d_j|} \quad \text{(noise at scale j)}$$

$$F = \frac{1}{J} \sum_{j=1}^{J} F_j \quad \text{(averaged across scales)}$$

$$I = 1 - \max(T_{\text{norm}}, F) \quad \text{(residual uncertainty)}$$

---

## 5. Quantile Envelope Encoder

### 5.1 Overview

The quantile encoder uses non-parametric quantile estimation to determine T, I, F boundaries.

### 5.2 Mathematical Formulation

**Quantile estimation:**

For each feature j, compute sample quantiles:

$$Q_\alpha^{(j)} = F_j^{-1}(\alpha)$$

Where $F_j^{-1}$ is the empirical quantile function.

**Envelope boundaries:**

$$L_j = Q_{\alpha_L}^{(j)} \quad \text{(lower boundary)}$$

$$U_j = Q_{1-\alpha_U}^{(j)} \quad \text{(upper boundary)}$$

Typical values: $\alpha_L = \alpha_U = 0.05$ (5th and 95th percentiles)

### 5.3 Channel Assignment

For each observation $x_{ij}$:

**Truth:** (scaled to [0,1] within envelope)

$$T_{ij} = \frac{x_{ij} - L_j}{U_j - L_j}$$

**Falsity:** (distance outside envelope)

$$F_{ij} = \begin{cases}
\frac{L_j - x_{ij}}{U_j - L_j} & \text{if } x_{ij} < L_j \\
\frac{x_{ij} - U_j}{U_j - L_j} & \text{if } x_{ij} > U_j \\
0 & \text{otherwise}
\end{cases}$$

**Indeterminacy:** (proximity to boundaries)

$$I_{ij} = \exp\left(-\gamma \cdot \min(|x_{ij} - L_j|, |x_{ij} - U_j|)\right)$$

---

## 6. Augmentation Stability Encoder

### 6.1 Overview

The augmentation encoder measures feature stability under data augmentation (noise injection, bootstrap, etc.).

### 6.2 Mathematical Formulation

**Data augmentation:**

Generate K augmented versions of the data:

$$\tilde{X}^{(k)} = X + \epsilon^{(k)}, \quad k = 1, ..., K$$

Where $\epsilon^{(k)} \sim \mathcal{N}(0, \sigma^2 I)$

**Fit models on each augmentation:**

$$\hat{y}^{(k)} = f(\tilde{X}^{(k)}; \theta^{(k)})$$

**Variance across augmentations:**

$$V_{ij} = \text{Var}_{k}\left(\tilde{X}_{ij}^{(k)}\right)$$

### 6.3 Channel Assignment

$$T = X \quad \text{(original data)}$$

$$I = \text{normalize}\left(\sqrt{V}\right) \quad \text{(augmentation variance)}$$

$$F = \mathbb{1}\left[\text{prediction differs significantly}\right]$$

---

## 7. Robust MAD Encoder

### 7.1 Overview

The robust encoder uses Median Absolute Deviation for outlier detection.

### 7.2 Mathematical Formulation

**For each feature j:**

$$\mu_j = \text{median}(X_{:,j})$$

$$\text{MAD}_j = \text{median}(|X_{:,j} - \mu_j|)$$

$$\sigma_j = 1.4826 \cdot \text{MAD}_j$$

**Modified z-score:**

$$z_{ij} = \frac{|X_{ij} - \mu_j|}{\sigma_j + \epsilon}$$

### 7.3 Channel Assignment

$$T = X$$

$$F = \phi(z; z_{\text{threshold}})$$

$$I = \text{baseline} \cdot (1 - F)$$

Where the threshold is typically $z_{\text{threshold}} = 3.5$ (outlier criterion).

---

## 8. NDG Manifold Encoder (Neutrosophic Differential Geometry)

### 8.1 Overview

The NDG encoder implements a physics-based encoding derived from Neutrosophic Differential Geometry, mapping spectroscopic data onto a manifold $\mathcal{M}_\mathcal{N}$ with principled T/I/F channels.

### 8.2 Theoretical Foundation

**The Neutrosophic Vector Space:**

Let the spectral domain be $\Lambda = \{\lambda_1, \lambda_2, ..., \lambda_p\}$. We define the Neutrosophic Vector Space $V_N$ as:

$$V_N = \{S_N \mid S_N(\lambda_k) = \langle T_k, I_k, F_k \rangle, \forall k \in \{1, ..., p\}\}$$

**The Manifold Mapping $\phi$:**

$$\phi: \mathbb{R}^p \rightarrow \mathcal{M}_\mathcal{N}$$

Component-wise:

$$\phi(x)_k = \begin{pmatrix} T_k \\ I_k \\ F_k \end{pmatrix} = \begin{pmatrix} \mathcal{N}(x_k) \\ \mathcal{H}(\sigma_k^2) \\ 1 - \mathcal{N}(x_k) \cdot (1 - \epsilon_k) \end{pmatrix}$$

### 8.3 Channel Derivations

**Truth Channel: $T_k = \mathcal{N}(x_k)$**

The normalization function can be:

- **None** (default): $T = X$ (preserves concentration)
- **SNV**: $T_k = \frac{x_k - \bar{x}}{\sigma_x}$ (removes baseline and scatter)
- **Min-Max**: $T_k = \frac{x_k - x_{min}}{x_{max} - x_{min}}$

**Indeterminacy Channel: $I_k = \mathcal{H}(\sigma_k^2)$**

Based on Shannon entropy of local variance. For Gaussian noise with variance $\sigma^2$:

$$H = \frac{1}{2} \ln(2\pi e \sigma^2)$$

The local variance is estimated from deviation from a locally-smoothed signal:

$$\sigma_k^2 = \text{smooth}\left((X_k - X_k^{\text{smooth}})^2\right)$$

Normalized to $[0, 1]$:

$$I_k = \left(\frac{\ln(\sigma_k^2) - \ln(\sigma_{min}^2)}{\ln(\sigma_{max}^2) - \ln(\sigma_{min}^2)}\right)^\beta$$

**Falsity Channel: $F_k = 1 - \mathcal{N}(x_k) \cdot (1 - \epsilon_k)$**

The systematic error coefficient $\epsilon_k$ is derived from low-rank model deviation:

1. Fit low-rank approximation: $X \approx L$ via truncated SVD
2. Compute residuals: $R = X - L$
3. Robust z-scores: $z_R = |R| / (\text{MAD}(R) \cdot 1.4826)$
4. Sigmoid transform: $\epsilon_k = \sigma(z_R - 3)$

### 8.4 Metric Tensor Interpretation

The encoding implicitly defines a **Neutrosophic Metric Tensor**:

$$g_{ij}^N = \alpha (g_{ij})_T - \beta (g_{ij})_I - \gamma (g_{ij})_F$$

Where:

- $(g_{ij})_T$: Fisher Information Matrix (signal curvature)
- $(g_{ij})_I$: Shannon Entropy diagonal matrix (noise uncertainty)
- $(g_{ij})_F$: Inverse bias covariance $(\Sigma_{bias}^{-1})$

**Interpretation:**
- High $I$ or $F$ "stretch" distances, making samples harder to distinguish
- The Ricci Scalar $R^I$ quantifies "noise density" of a spectral region

### 8.5 Support for Replicate Scans

When replicate scans are available:

$$\sigma_k^2 = \text{Var}_{\text{replicates}}(X_k)$$

This provides true measurement variance instead of smoothing approximation.

### 8.6 Usage

```python
from neutrosophic_pls.encoders import encode_ndg_manifold

# Default: preserves concentration signal
result = encode_ndg_manifold(X, normalization='none')

# With replicate scans for true variance
result = encode_ndg_manifold(X, replicate_scans=replicate_data)

# Access geometric metadata
print(f"Complexity score: {result.metadata['complexity_score']}")
```

---

## 9. Spectroscopy-Specific Encoder

### 8.1 Overview

Optimized for NIR/IR spectroscopy with noise-floor awareness.

### 8.2 Mathematical Formulation

**Noise floor estimation:**

For spectroscopy data, the noise floor varies with wavelength. Estimate local noise:

$$\sigma_j^{\text{local}} = \text{MAD}(\nabla^2 X_{:,j})$$

Where $\nabla^2$ is the second-difference operator:

$$(\nabla^2 X)_{i,j} = X_{i,j+1} - 2X_{i,j} + X_{i,j-1}$$

**Signal-to-noise ratio:**

$$\text{SNR}_j = \frac{|X_{:,j}|}{\sigma_j^{\text{local}} + \epsilon}$$

### 8.3 Channel Assignment

$$T = X$$

$$I = 1 - \tanh(\alpha \cdot \text{SNR})$$

$$F = \text{robust outlier detection based on 2nd derivative}$$

---

## 9. Common Squashing Functions

### 9.1 Hyperbolic Tangent

$$\phi(z; s, \beta) = \tanh^\beta\left(\frac{z}{s}\right)$$

**Properties:**
- Output range: [0, 1]
- Saturates at z >> s
- β controls sharpness

### 9.2 Sigmoid

$$\phi(z; k, z_0) = \frac{1}{1 + e^{-k(z - z_0)}}$$

### 9.3 Power Transform

$$\phi(z; \beta) = e^{-z^\beta}$$

---

## 11. Encoder Selection Guidelines

| Data Type | Recommended Encoder | Reason |
|-----------|---------------------|--------|
| Clean, normal | Probabilistic | Statistical foundation |
| Sparse outliers | RPCA | Separates sparse corruption |
| Multi-scale signals | Wavelet | Frequency decomposition |
| Unknown distribution | Quantile | Non-parametric |
| High-dimensional | Augmentation | Stability-based |
| Spectroscopy | Spectroscopy | Domain-specific noise model |
| General robust | Robust MAD | Simple and effective |
| Physics-based | **NDG** | Principled geometric derivation |
| Replicate scans | **NDG** | Uses true measurement variance |

---

## 11. Summary of Channel Properties

| Property | Truth (T) | Indeterminacy (I) | Falsity (F) |
|----------|-----------|-------------------|-------------|
| What it represents | Clean signal | Uncertainty | Outlier/noise |
| Typical range | [min, max] of X | [0, 1] | [0, 1] |
| Computation | Original or reconstructed | Ambiguity measure | Deviation score |
| High values mean | Normal data | Uncertain measurement | Likely outlier |

---

## 12. References

1. Candès, E. J., et al. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), 1-37.

2. Donoho, D. L. (1995). De-noising by soft-thresholding. *IEEE transactions on information theory*, 41(3), 613-627.

3. Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute deviation. *Journal of the American Statistical Association*, 88(424), 1273-1283.

4. Smarandache, F. (2005). Neutrosophic set - a generalization of the intuitionistic fuzzy set. *International Journal of Pure and Applied Mathematics*, 24(3), 287-297.

---

*Document generated for the Neutrosophic PLS package.*
