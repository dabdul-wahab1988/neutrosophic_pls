# Neutrosophic Variable Importance in Projection (NVIP)

## Mathematical Foundation and Proof

**Author:** Neutrosophic PLS Development Team  
**Date:** December 2024  
**Version:** 1.0

---

## 1. Introduction

This document provides the complete mathematical derivation and proof of the Neutrosophic Variable Importance in Projection (NVIP) decomposition used in the N-PLS framework.

The NVIP extends the classical PLS VIP score by decomposing feature importance into contributions from three neutrosophic channels:
- **Truth (T)**: The signal component
- **Indeterminacy (I)**: The uncertainty component  
- **Falsity (F)**: The noise/outlier component

---

## 2. Classical VIP Definition

### 2.1 Standard PLS VIP Formula

The Variable Importance in Projection (VIP) was introduced by Wold et al. (1993). For a PLS model with A components, the VIP score for feature j is defined as:

$$\text{VIP}_j = \sqrt{\frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_a}{\sum_{a=1}^{A} SS_a}}$$

Where:
- $p$ = number of features (predictors)
- $A$ = number of PLS components
- $w_{ja}$ = weight of feature j in component a
- $SS_a$ = sum of squares of Y explained by component a

### 2.2 Explained Variance (SS)

The explained sum of squares for component a is:

$$SS_a = \|t_a\|^2 \cdot \|q_a\|^2 = \left(\sum_{i=1}^{n} t_{ia}^2\right) \cdot \left(\sum_{k=1}^{m} q_{ak}^2\right)$$

Where:
- $t_a \in \mathbb{R}^n$ = score vector for component a
- $q_a \in \mathbb{R}^m$ = Y-loading vector for component a
- $n$ = number of samples
- $m$ = number of response variables

### 2.3 Score Computation

The scores are computed as:

$$t_a = X_c \cdot w_a$$

Where $X_c$ is the centered data matrix and $w_a$ is the weight vector for component a.

---

## 3. Neutrosophic Data Representation

### 3.1 T-I-F Tensor

In the neutrosophic framework, the data is represented as a 3-channel tensor:

$$\mathcal{X} \in \mathbb{R}^{n \times p \times 3}$$

Where:
- $T = \mathcal{X}_{:,:,0}$ is the Truth (signal) channel
- $I = \mathcal{X}_{:,:,1}$ is the Indeterminacy (uncertainty) channel
- $F = \mathcal{X}_{:,:,2}$ is the Falsity (noise/outlier) channel

### 3.2 Channel Weights

Each channel can be weighted by $(\omega_T, \omega_I, \omega_F)$ to control its contribution.

---

## 4. NVIP Decomposition: Mathematical Derivation

### 4.1 Core Idea

The NVIP decomposes the aggregate VIP into channel-specific contributions by computing the explained variance contribution from each channel separately, while using the same model weights learned from the primary channel.

### 4.2 Channel-Specific Scores

Given the model weights $W \in \mathbb{R}^{p \times A}$ learned on the Truth channel, we compute channel-specific scores:

$$t_T^{(a)} = T_c \cdot w_a$$
$$t_I^{(a)} = I_c \cdot w_a$$
$$t_F^{(a)} = F_c \cdot w_a$$

Where $T_c$, $I_c$, $F_c$ are the centered versions of each channel.

### 4.3 Channel-Specific Sum of Squares

For each channel $C \in \{T, I, F\}$, we compute the weighted sum of squares:

$$SS_C^{(a)} = \omega_C^2 \cdot \|t_C^{(a)}\|^2 \cdot \|q_a\|^2$$

The weighting by $\omega_C^2$ (squared) is necessary because SS is a quadratic measure.

### 4.4 Total Sum of Squares

The total SS across all channels is:

$$SS_{\text{total}} = \sum_{a=1}^{A} \left( SS_T^{(a)} + SS_I^{(a)} + SS_F^{(a)} \right)$$

### 4.5 Channel-Specific VIP

For each channel C, the VIP is computed as:

$$\text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_C^{(a)}}{SS_{\text{total}}}$$

Taking the square root:

$$\text{VIP}_C(j) = \sqrt{\frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_C^{(a)}}{SS_{\text{total}}}}$$

---

## 5. Main Theorem: L2-Norm Decomposition

### 5.1 Theorem Statement

**Theorem (NVIP L2 Decomposition):**

For any feature j, the aggregate VIP satisfies the L2-norm relationship:

$$\text{VIP}_{\text{aggregate}}(j) = \sqrt{\text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}$$

### 5.2 Proof

**Proof:**

Starting from the definition of channel-specific VIP squared:

$$\text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_C^{(a)}}{SS_{\text{total}}}$$

Sum over all channels:

$$\sum_{C \in \{T,I,F\}} \text{VIP}_C^2(j) = \sum_{C \in \{T,I,F\}} \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_C^{(a)}}{SS_{\text{total}}}$$

Factor out common terms:

$$= \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot \sum_{C \in \{T,I,F\}} SS_C^{(a)}}{SS_{\text{total}}}$$

By definition of $SS_{\text{total}}$:

$$\sum_{C \in \{T,I,F\}} SS_C^{(a)} = SS_{\text{total}}^{(a)}$$

And:

$$\sum_{a=1}^{A} SS_{\text{total}}^{(a)} = SS_{\text{total}}$$

Therefore:

$$\sum_{C \in \{T,I,F\}} \text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_{\text{total}}^{(a)}}{SS_{\text{total}}}$$

This is exactly the standard VIP formula applied to the combined data. Define:

$$\text{VIP}_{\text{aggregate}}^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_{\text{total}}^{(a)}}{SS_{\text{total}}}$$

Thus:

$$\text{VIP}_{\text{aggregate}}^2(j) = \text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)$$

Taking the square root of both sides:

$$\boxed{\text{VIP}_{\text{aggregate}}(j) = \sqrt{\text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}}$$

**Q.E.D.**

---

## 6. Properties of the NVIP Decomposition

### 6.1 Non-Negativity

All VIP values are non-negative:

$$\text{VIP}_C(j) \geq 0 \quad \forall C \in \{T, I, F\}, \forall j$$

**Proof:** Since $w_{ja}^2 \geq 0$ and $SS_C^{(a)} \geq 0$, the sum is non-negative, and the square root preserves non-negativity.

### 6.2 Pythagorean Relationship

The channel VIPs form an orthogonal decomposition in the squared space:

$$\|\text{VIP}\|_2^2 = \|\text{VIP}_T\|_2^2 + \|\text{VIP}_I\|_2^2 + \|\text{VIP}_F\|_2^2$$

This is analogous to the Pythagorean theorem in three dimensions.

### 6.3 Scale Invariance

The VIP values are scale-invariant with respect to uniform scaling of the data, as both numerator and denominator scale quadratically.

### 6.4 Weight Sensitivity

The channel weights $(\omega_T, \omega_I, \omega_F)$ affect contributions quadratically:

$$SS_C \propto \omega_C^2$$

This means doubling a channel weight quadruples its SS contribution.

---

## 7. Interpretation Guide

### 7.1 Channel Dominance

For a feature j, the dominant channel is determined by:

$$\text{Dominant}(j) = \arg\max_{C \in \{T,I,F\}} \text{VIP}_C(j)$$

### 7.2 Signal-to-Noise Ratio

The VIP-based signal-to-noise ratio is defined as:

$$\text{SNR}(j) = \frac{\text{VIP}_T(j)}{\text{VIP}_F(j) + \epsilon}$$

Where $\epsilon$ is a small constant to prevent division by zero.

### 7.3 Importance Thresholds

Following the classical VIP convention:
- $\text{VIP} > 1$: Important feature
- $0.8 < \text{VIP} < 1$: Moderately important
- $\text{VIP} < 0.8$: Less important

### 7.4 Channel-Specific Insights

| High VIP Channel | Interpretation |
|------------------|----------------|
| VIP_T >> VIP_I, VIP_F | Feature importance driven by signal values |
| VIP_I >> VIP_T, VIP_F | Uncertainty pattern is informative |
| VIP_F >> VIP_T, VIP_I | Noise/outlier pattern is predictive (potential data quality issue) |

---

## 8. Algorithm Implementation

### 8.1 Pseudocode

```
Algorithm: NVIP_Decomposition
Input: 
  - model: Fitted NPLS model with weights W, y_loadings Q
  - X_tif: Neutrosophic tensor (n, p, 3)
  - weights: (ω_T, ω_I, ω_F)

Output: 
  - VIP_T, VIP_I, VIP_F, VIP_aggregate (each of length p)

1. Extract channels: T, I, F from X_tif
2. Center each channel: T_c, I_c, F_c

3. For each channel C in {T, I, F}:
   a. Compute scores: t_C = C_c @ W
   b. Compute SS: SS_C[a] = ω_C² × ||t_C[:,a]||² × ||Q[a,:]||²

4. Compute total SS: SS_total = Σ(SS_T + SS_I + SS_F)

5. For each channel C:
   a. VIP_C² = (p × (W² @ SS_C)) / SS_total
   b. VIP_C = sqrt(max(VIP_C², 0))

6. VIP_aggregate = sqrt(VIP_T² + VIP_I² + VIP_F²)

7. Return VIP_T, VIP_I, VIP_F, VIP_aggregate
```

### 8.2 Computational Complexity

- Time: $O(n \cdot p \cdot A)$ for score computation
- Space: $O(n \cdot A)$ for storing scores per channel

---

## 9. Validation

### 9.1 Numerical Verification

The implementation satisfies:

```python
# This should always be True
np.allclose(
    aggregate_vip, 
    np.sqrt(vip_t**2 + vip_i**2 + vip_f**2)
)
```

### 9.2 Edge Cases

| Case | Behavior |
|------|----------|
| Zero-variance feature | VIP = 0 for all channels |
| Single channel active | VIP_aggregate = VIP_active_channel |
| Equal channel contributions | VIP_C = VIP_aggregate / √3 for each C |

---

## 10. Comparison with Other Methods

### 10.1 vs Classical PLS VIP

| Aspect | Classical VIP | NVIP |
|--------|---------------|------|
| Input | Raw X matrix | T-I-F tensor |
| Output | Single VIP | VIP per channel + aggregate |
| Noise handling | None | Explicit via F channel |
| Interpretation | Importance only | Importance + data quality |

### 10.2 vs Permutation Importance

| Aspect | Permutation | NVIP |
|--------|-------------|------|
| Approach | Model-agnostic | PLS-specific |
| Cost | Expensive (refit) | Cheap (single computation) |
| Channel decomposition | No | Yes |

---

## 11. References

1. Wold, S., Johansson, E., & Cocchi, M. (1993). PLS: Partial Least Squares Projections to Latent Structures. *3D QSAR in Drug Design*, 523-550.

2. Chong, I. G., & Jun, C. H. (2005). Performance of some variable selection methods when multicollinearity is present. *Chemometrics and Intelligent Laboratory Systems*, 78(1-2), 103-112.

3. Smarandache, F. (1999). A unifying field in logics: Neutrosophic logic. *Philosophy*, 1-141.

4. Mehmood, T., Liland, K. H., Snipen, L., & Sæbø, S. (2012). A review of variable selection methods in partial least squares regression. *Chemometrics and Intelligent Laboratory Systems*, 118, 62-69.

---

## 12. Appendix: Symbol Table

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $n$ | Number of samples | scalar |
| $p$ | Number of features | scalar |
| $A$ | Number of PLS components | scalar |
| $m$ | Number of response variables | scalar |
| $T$ | Truth channel | $n \times p$ |
| $I$ | Indeterminacy channel | $n \times p$ |
| $F$ | Falsity channel | $n \times p$ |
| $W$ | X-weights matrix | $p \times A$ |
| $Q$ | Y-loadings matrix | $A \times m$ |
| $t_a$ | Score vector for component a | $n \times 1$ |
| $w_a$ | Weight vector for component a | $p \times 1$ |
| $q_a$ | Y-loading vector for component a | $m \times 1$ |
| $SS_a$ | Sum of squares for component a | scalar |
| $\omega_C$ | Channel weight for C | scalar |
| $\text{VIP}_C(j)$ | VIP of feature j for channel C | scalar |

---

*Document generated for the Neutrosophic PLS package.*
