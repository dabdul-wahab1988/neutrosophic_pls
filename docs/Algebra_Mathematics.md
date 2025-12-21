# Neutrosophic Algebra

## Mathematical Foundation for T-I-F Operations

**Author:** Neutrosophic PLS Development Team  
**Date:** December 2024  
**Version:** 1.0

---

## 1. Introduction

This document provides the mathematical foundation for the neutrosophic algebra operations used in the N-PLS framework.

Neutrosophic logic, introduced by Florentin Smarandache, extends fuzzy logic by explicitly representing three components of knowledge:

- **Truth (T)**: Degree of membership/validity
- **Indeterminacy (I)**: Degree of uncertainty/unknownness
- **Falsity (F)**: Degree of non-membership/rejection

---

## 2. Neutrosophic Set Theory

### 2.1 Definition

A **neutrosophic set** A on a universe X is characterized by three membership functions:

$$A = \{(x, T_A(x), I_A(x), F_A(x)) : x \in X\}$$

Where:

- $T_A : X \to [0, 1]$ — truth membership function
- $I_A : X \to [0, 1]$ — indeterminacy membership function
- $F_A : X \to [0, 1]$ — falsity membership function

### 2.2 Constraint Variations

**Standard neutrosophic set:**

$$0 \leq T_A(x) + I_A(x) + F_A(x) \leq 3$$

**Single-valued neutrosophic set (SVNS):**

$$0 \leq T_A(x) + I_A(x) + F_A(x) \leq 3$$

with $T, I, F \in [0, 1]$

**Normalized neutrosophic set:**

$$T_A(x) + I_A(x) + F_A(x) = 1$$

---

## 3. Neutrosophic Triplet

### 3.1 Definition

A **neutrosophic triplet** is an ordered triple:

$$\tau = (T, I, F)$$

Where $T, I, F \in [0, 1]$ (or more generally, from a suitable lattice).

### 3.2 Triplet Space

The space of all neutrosophic triplets:

$$\mathcal{N} = \{(T, I, F) : T, I, F \in [0, 1]\}$$

This is the unit cube $[0, 1]^3$.

---

## 4. Neutrosophic Inner Product

### 4.1 Definition

For two neutrosophic triplets $x = (T_x, I_x, F_x)$ and $y = (T_y, I_y, F_y)$, with weight triplet $w = (w_T, w_I, w_F)$:

$$\langle x, y \rangle_w = w_T \cdot T_x \cdot T_y + w_I \cdot I_x \cdot I_y + w_F \cdot F_x \cdot F_y$$

### 4.2 Properties

**Symmetry:**
$$\langle x, y \rangle_w = \langle y, x \rangle_w$$

**Linearity (in first argument):**
$$\langle \alpha x, y \rangle_w = \alpha \langle x, y \rangle_w$$

**Positive semi-definiteness:**
$$\langle x, x \rangle_w \geq 0$$

with equality iff $x = (0, 0, 0)$.

### 4.3 Matrix Form

For neutrosophic data tensors, the weighted inner product becomes:

$$\langle X, Y \rangle_w = w_T \cdot X_T^T Y_T + w_I \cdot X_I^T Y_I + w_F \cdot X_F^T Y_F$$

---

## 5. Neutrosophic Norm

### 5.1 Definition

The **neutrosophic norm** induced by the weighted inner product:

$$\|x\|_w = \sqrt{\langle x, x \rangle_w} = \sqrt{w_T \cdot T^2 + w_I \cdot I^2 + w_F \cdot F^2}$$

### 5.2 Special Cases

**Unweighted norm** ($w = (1, 1, 1)$):
$$\|x\| = \sqrt{T^2 + I^2 + F^2}$$

**Truth-only norm** ($w = (1, 0, 0)$):
$$\|x\|_T = |T|$$

**Reliability norm** ($w = (1, 0, 1)$):
$$\|x\|_R = \sqrt{T^2 + F^2}$$

### 5.3 Properties

1. **Non-negativity:** $\|x\|_w \geq 0$
2. **Identity of indiscernibles:** $\|x\|_w = 0 \Leftrightarrow x = (0, 0, 0)$
3. **Homogeneity:** $\|\alpha x\|_w = |\alpha| \|x\|_w$
4. **Triangle inequality:** $\|x + y\|_w \leq \|x\|_w + \|y\|_w$

---

## 6. Channel Combination Functions

### 6.1 Truth-Only Mode

The simplest combination extracts only the truth channel:

$$\text{combine}_{\text{truth}}(\mathcal{X}, w) = w_T \cdot T$$

**Use case:** When I/F are used for sample weighting rather than feature modification.

### 6.2 Weighted Sum Mode

Linear combination of all channels:

$$\text{combine}_{\text{sum}}(\mathcal{X}, w) = w_T \cdot T + w_I \cdot I + w_F \cdot F$$

**Warning:** May not preserve semantic meaning if channels have different scales.

### 6.3 Attenuation Mode

High uncertainty/falsity attenuates the truth signal:

$$\text{combine}_{\text{atten}}(\mathcal{X}, w, \theta) = w_T \cdot T \cdot a(I, F)$$

Where the attenuation function is:

$$a(I, F) = (1 - w_I \cdot I')(1 - w_F \cdot F')$$

With soft-thresholded values:

$$I' = \max(0, I - \theta) / (1 - \theta)$$
$$F' = \max(0, F - \theta) / (1 - \theta)$$

**Properties:**

- $a \in [0, 1]$
- $a = 1$ when $I, F < \theta$ (no attenuation)
- $a \to 0$ as $I, F \to 1$ (complete attenuation)

---

## 7. Neutrosophic Distance Metrics

### 7.1 Euclidean Distance

$$d_E(x, y) = \sqrt{(T_x - T_y)^2 + (I_x - I_y)^2 + (F_x - F_y)^2}$$

### 7.2 Hamming Distance

$$d_H(x, y) = |T_x - T_y| + |I_x - I_y| + |F_x - F_y|$$

### 7.3 Weighted Distance

$$d_w(x, y) = \sqrt{w_T(T_x - T_y)^2 + w_I(I_x - I_y)^2 + w_F(F_x - F_y)^2}$$

### 7.4 Normalized Distance

$$d_N(x, y) = \frac{1}{3}(|T_x - T_y| + |I_x - I_y| + |F_x - F_y|)$$

Range: $[0, 1]$

---

## 8. Neutrosophic Similarity Measures

### 8.1 Cosine Similarity

$$\text{sim}_{\cos}(x, y) = \frac{\langle x, y \rangle}{\|x\| \cdot \|y\|}$$

$$= \frac{T_x T_y + I_x I_y + F_x F_y}{\sqrt{T_x^2 + I_x^2 + F_x^2} \cdot \sqrt{T_y^2 + I_y^2 + F_y^2}}$$

### 8.2 Dice Similarity

$$\text{sim}_{\text{Dice}}(x, y) = \frac{2(T_x T_y + I_x I_y + F_x F_y)}{(T_x^2 + I_x^2 + F_x^2) + (T_y^2 + I_y^2 + F_y^2)}$$

### 8.3 Jaccard Similarity

$$\text{sim}_J(x, y) = \frac{\sum_c \min(x_c, y_c)}{\sum_c \max(x_c, y_c)}$$

Where $c \in \{T, I, F\}$.

---

## 9. Neutrosophic Aggregation Operators

### 9.1 Neutrosophic Maximum

$$\max_N(x, y) = (\max(T_x, T_y), \min(I_x, I_y), \min(F_x, F_y))$$

### 9.2 Neutrosophic Minimum

$$\min_N(x, y) = (\min(T_x, T_y), \max(I_x, I_y), \max(F_x, F_y))$$

### 9.3 Neutrosophic Mean

$$\text{mean}_N(x_1, ..., x_n) = \left(\frac{1}{n}\sum T_i, \frac{1}{n}\sum I_i, \frac{1}{n}\sum F_i\right)$$

### 9.4 Weighted Neutrosophic Mean

$$\text{mean}_{N,w}(x_1, ..., x_n; \lambda_1, ..., \lambda_n) = \left(\sum \lambda_i T_i, \sum \lambda_i I_i, \sum \lambda_i F_i\right)$$

Where $\sum \lambda_i = 1$.

---

## 10. Score and Accuracy Functions

### 10.1 Score Function

Used to rank neutrosophic triplets:

$$S(x) = T - I - F$$

Range: $[-2, 1]$

**Interpretation:**

- $S > 0$: More true than false/uncertain
- $S < 0$: More false/uncertain than true
- $S = 0$: Balanced

### 10.2 Accuracy Function

$$A(x) = T - F$$

Range: $[-1, 1]$

### 10.3 Certainty Function

$$C(x) = 1 - I$$

Range: $[0, 1]$

### 10.4 Combined Decision Function

$$D(x) = \alpha \cdot S(x) + \beta \cdot A(x) + \gamma \cdot C(x)$$

Where $\alpha + \beta + \gamma = 1$.

---

## 11. Neutrosophic Matrix Operations

### 11.1 Neutrosophic Matrix

A neutrosophic matrix $M \in \mathcal{N}^{m \times n}$ has entries:

$$M_{ij} = (T_{ij}, I_{ij}, F_{ij})$$

Can be represented as a 3D tensor $\mathcal{M} \in \mathbb{R}^{m \times n \times 3}$.

### 11.2 Channel Extraction

$$T(M) = \mathcal{M}_{:,:,0}$$
$$I(M) = \mathcal{M}_{:,:,1}$$
$$F(M) = \mathcal{M}_{:,:,2}$$

### 11.3 Neutrosophic Matrix Addition

$$(A + B)_{ij} = (T_A + T_B, I_A + I_B, F_A + F_B)_{ij}$$

With optional normalization to keep values in $[0, 1]$.

### 11.4 Neutrosophic Scalar Multiplication

$$(\alpha \cdot M)_{ij} = (\alpha T_{ij}, \alpha I_{ij}, \alpha F_{ij})$$

### 11.5 Hadamard (Element-wise) Product

$$(A \odot B)_{ij} = (T_A T_B, I_A I_B, F_A F_B)_{ij}$$

---

## 12. Reliability Computation

### 12.1 Element-wise Reliability

$$R_{ij} = T_{ij} \cdot (1 - I_{ij}) \cdot (1 - F_{ij})$$

Or simplified:

$$R_{ij} = 1 - \max(I_{ij}, F_{ij})$$

### 12.2 Sample Reliability (geometric mean)

$$R_i = \left(\prod_{j=1}^{p} R_{ij}\right)^{1/p}$$

### 12.3 Feature Reliability (across samples)

$$R_j = \frac{1}{n} \sum_{i=1}^{n} R_{ij}$$

---

## 13. Implementation in Code

### 13.1 Neutrosophic Triplet Type

```python
NeutroTriplet = Tuple[float, float, float]  # (T, I, F)
```

### 13.2 Inner Product

```python
def neutro_inner(x: NeutroTriplet, y: NeutroTriplet, 
                 weights: NeutroTriplet = (1, 1, 1)) -> float:
    tx, ix, fx = x
    ty, iy, fy = y
    wt, wi, wf = weights
    return wt * tx * ty + wi * ix * iy + wf * fx * fy
```

### 13.3 Norm

```python
def neutro_norm(x: NeutroTriplet, 
                weights: NeutroTriplet = (1, 1, 1)) -> float:
    return sqrt(neutro_inner(x, x, weights))
```

### 13.4 Channel Combination

```python
def combine_channels(X: ndarray, weights: NeutroTriplet, 
                     mode: str = "truth_only") -> ndarray:
    T, I, F = X[..., 0], X[..., 1], X[..., 2]
    
    if mode == "truth_only":
        return weights[0] * T
    elif mode == "attenuation":
        atten = (1 - weights[1] * I) * (1 - weights[2] * F)
        return weights[0] * T * atten
```

---

## 14. Connections to Other Frameworks

### 14.1 Fuzzy Sets

Neutrosophic sets generalize fuzzy sets:

- Fuzzy: $(T, 1 - T)$ (T and F = 1 - T, no I)
- Neutrosophic: $(T, I, F)$ independent

### 14.2 Intuitionistic Fuzzy Sets

Neutrosophic includes intuitionistic fuzzy as special case:

- IFS: $(T, I, F)$ where $I = 1 - T - F$
- Neutrosophic: $(T, I, F)$ independent

### 14.3 Interval-Valued Sets

Neutrosophic can represent intervals:

- Lower bound: $T - I$
- Upper bound: $T + I$
- Falsity: $F$

---

## 15. Summary Table

| Operation | Formula | Range |
|-----------|---------|-------|
| Inner product | $w_T T_x T_y + w_I I_x I_y + w_F F_x F_y$ | $\mathbb{R}$ |
| Norm | $\sqrt{w_T T^2 + w_I I^2 + w_F F^2}$ | $[0, \sqrt{3}]$ |
| Score | $T - I - F$ | $[-2, 1]$ |
| Accuracy | $T - F$ | $[-1, 1]$ |
| Reliability | $1 - \max(I, F)$ | $[0, 1]$ |
| Distance | $\sqrt{\sum_c (x_c - y_c)^2}$ | $[0, \sqrt{3}]$ |

---

## 16. References

1. Smarandache, F. (1999). A Unifying Field in Logics: Neutrosophic Logic. *Philosophy*, American Research Press.

2. Smarandache, F. (2005). Neutrosophic set - a generalization of the intuitionistic fuzzy set. *International Journal of Pure and Applied Mathematics*, 24(3), 287-297.

3. Wang, H., Smarandache, F., Zhang, Y., & Sunderraman, R. (2010). Single valued neutrosophic sets. *Multispace and Multistructure*, 4, 410-413.

4. Ye, J. (2014). A multicriteria decision-making method using aggregation operators for simplified neutrosophic sets. *Journal of Intelligent & Fuzzy Systems*, 26(5), 2459-2466.

5. Peng, J. J., et al. (2015). Simplified neutrosophic sets and their applications in multi-criteria group decision-making problems. *International Journal of Systems Science*, 47(10), 2342-2358.

---

*Document generated for the Neutrosophic PLS package.*
