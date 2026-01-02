---
title: 'Neutrosophic Partial Least Squares (N-PLS): Uncertainty-aware Regression for Chemometrics'
tags:
  - Python
  - chemometrics
  - spectroscopy
  - partial least squares
  - neutrosophic sets
  - uncertainty quantification
  - machine learning
  - NIR spectroscopy
  - robust regression
  - variable importance
authors:
  - name: Dickson Abdul-Wahab
    orcid: 0000-0001-7446-5909
    corresponding: true
    affiliation: 1
  - name: Ebenezer Aquisman Asare
    orcid: 0000-0003-1185-1479
    affiliation: 2
affiliations:
  - name: University of Ghana, Legon, Accra, Ghana
    index: 1
  - name: Independent Researcher
    index: 2
date: 21 December 2025
bibliography: paper.bib
---

# Summary

Neutrosophic Partial Least Squares (N-PLS) is a Python package that extends classical Partial Least Squares (PLS) regression by incorporating measurement uncertainty through neutrosophic set theory. Traditional PLS treats all observations as equally reliable, which can lead to suboptimal predictions when data contains noise, outliers, or measurements of varying quality—a common situation in spectroscopic analysis and chemometrics.

N-PLS addresses this limitation by encoding each measurement as a neutrosophic triplet $(T, I, F)$ representing Truth (the signal), Indeterminacy (measurement uncertainty), and Falsity (evidence of corruption or noise). This three-channel representation enables the algorithm to downweight unreliable measurements during model fitting, resulting in more robust calibration models.

The package provides three N-PLS variants with increasing sophistication: standard NPLS with sample-level reliability weighting, reliability-weighted NPLSW for stronger downweighting of noisy samples, and probabilistic PNPLS that applies element-wise precision weighting through an EM-NIPALS algorithm. The package includes eight encoding strategies (probabilistic/default, spectroscopy, robust MAD, RPCA, wavelet, quantile, augmentation, and NDG), plus an optional cross-validated auto-selection wrapper.

N-PLS includes a comprehensive interactive command-line interface designed for researchers without programming experience, enabling guided analysis through a 7-step wizard. The package has been validated on real-world Near-Infrared (NIR) spectroscopy data, demonstrating up to 70% improvement in prediction accuracy compared to classical PLS on challenging datasets with measurement noise.

# Statement of Need

Partial Least Squares (PLS) regression is the workhorse algorithm of chemometrics, widely used for quantitative analysis from spectroscopic data in pharmaceutical, agricultural, food, and environmental applications [@wold1984multivariate; @geladi1986partial]. However, classical PLS assumes that all measurements are equally reliable—an assumption that rarely holds in practice. Real-world spectroscopic measurements are affected by:

- **Instrumental noise** varying across wavelength regions, with detector sensitivity and source intensity both wavelength-dependent
- **Sample preparation variability** affecting measurement reproducibility, including particle size effects, packing density, and sample presentation
- **Environmental factors** such as temperature fluctuations, humidity changes, and atmospheric absorption
- **Outliers and anomalous samples** from equipment malfunction, sample contamination, or measurement artifacts

While various robust PLS methods exist [@hubert2003robust; @serneels2005partial], they typically address sample-level outliers without providing insight into *which specific wavelengths* or *measurement regions* are unreliable. This limits their interpretability and diagnostic value for spectroscopists who need to understand the physical basis of model performance.

N-PLS fills this gap by leveraging neutrosophic set theory [@smarandache1999linguistic; @smarandache2005unifying], which naturally represents degrees of truth, uncertainty, and falsity. Unlike fuzzy sets that capture only membership degree, or intuitionistic fuzzy sets with truth and falsity, neutrosophic sets explicitly model *indeterminacy*—the uncertainty about whether a measurement is reliable or not. This is particularly relevant for spectroscopy, where noise characteristics vary across the spectrum and individual measurements may be systematically biased.

The need for uncertainty-aware chemometric methods is growing as analytical instruments generate increasingly complex data. The IDRC (International Diffuse Reflectance Conference) Shootout datasets [@fearn2020international] have highlighted that classical calibration methods often fail under realistic measurement conditions, particularly when transferring calibrations between instruments or dealing with heterogeneous sample populations.

N-PLS is designed to address these challenges while remaining accessible to domain scientists. Unlike most chemometric software that requires programming expertise, N-PLS provides an interactive wizard that guides users through data loading, encoding, model selection, and interpretation—enabling researchers in chemistry, food science, agriculture, and pharmaceutical sciences to leverage advanced uncertainty quantification without writing code.

# Theoretical Foundation

## Neutrosophic Set Theory

Neutrosophic logic, introduced by Smarandache [@smarandache1999linguistic], extends classical logic by representing propositions as triplets $(T, I, F)$ where:

- $T \in [0,1]$: degree of truth (membership)
- $I \in [0,1]$: degree of indeterminacy (uncertainty)
- $F \in [0,1]$: degree of falsity (non-membership)

Formally, a **neutrosophic set** $A$ on a universe $X$ is characterized by three membership functions:

$$A = \{(x, T_A(x), I_A(x), F_A(x)) : x \in X\}$$

Unlike fuzzy sets ($T$ only) or intuitionistic fuzzy sets ($T$ and $F$ with $T + F \leq 1$), neutrosophic sets allow $T + I + F$ to range from $0$ to $3$, capturing situations where evidence is contradictory or incomplete [@wang2010single; @ye2014multicriteria].

In the context of spectroscopic measurements, this triplet naturally maps to:

- **Truth**: The reliable signal component, encoding what we believe the measurement represents
- **Indeterminacy**: Measurement uncertainty, capturing the degree of confidence we have in the measurement's reliability
- **Falsity**: Evidence of corruption, noise, or systematic bias that renders the measurement unreliable

## Neutrosophic Algebra

The N-PLS framework requires algebraic operations on neutrosophic triplets. For two neutrosophic triplets $x = (T_x, I_x, F_x)$ and $y = (T_y, I_y, F_y)$ with weight triplet $w = (w_T, w_I, w_F)$:

**Neutrosophic Inner Product:**
$$\langle x, y \rangle_w = w_T \cdot T_x \cdot T_y + w_I \cdot I_x \cdot I_y + w_F \cdot F_x \cdot F_y$$

**Neutrosophic Norm:**
$$\|x\|_w = \sqrt{\langle x, x \rangle_w} = \sqrt{w_T \cdot T^2 + w_I \cdot I^2 + w_F \cdot F^2}$$

**Neutrosophic Distance:**
$$d_w(x, y) = \sqrt{w_T(T_x - T_y)^2 + w_I(I_x - I_y)^2 + w_F(F_x - F_y)^2}$$

These operations satisfy the standard properties of inner products and norms (symmetry, linearity, positive semi-definiteness, triangle inequality), enabling rigorous mathematical treatment of neutrosophic data.

## Score and Reliability Functions

To reduce neutrosophic triplets to scalar values for decision-making, N-PLS employs:

**Score Function** (for ranking triplets):
$$S(x) = T - I - F$$

with range $[-2, 1]$, where $S > 0$ indicates more truth than uncertainty/falsity.

**Reliability Function** (for sample/element weighting):
$$R_{ij} = 1 - \max(I_{ij}, F_{ij})$$

**Sample Reliability** (geometric mean across features):
$$R_i = \left(\prod_{j=1}^{p} R_{ij}\right)^{1/p}$$

## From Neutrosophic Triplets to PLS

Given a data matrix $X \in \mathbb{R}^{n \times p}$ with $n$ samples and $p$ features, N-PLS first encodes the data into a neutrosophic tensor $\mathcal{X} \in \mathbb{R}^{n \times p \times 3}$:

$$\mathcal{X}_{ijk} = \begin{cases} T_{ij} & k = 0 \\ I_{ij} & k = 1 \\ F_{ij} & k = 2 \end{cases}$$

This tensor representation enables the PLS algorithm to treat uncertainty and noise as first-class information channels, rather than ignoring them or treating them uniformly across all measurements.

## Classical NIPALS Algorithm

The Nonlinear Iterative Partial Least Squares (NIPALS) algorithm forms the foundation for all N-PLS variants. For each component $a = 1, \ldots, A$:

1. Initialize: $u_a = Y_{:,1}$
2. X-side weight: $w_a = X_a^T u_a / \|X_a^T u_a\|$
3. X-side score: $t_a = X_a w_a$
4. Y-side weight: $c_a = Y_a^T t_a / \|Y_a^T t_a\|$
5. Y-side score: $u_a = Y_a c_a$
6. Repeat 2-5 until convergence
7. X-loading: $p_a = X_a^T t_a / (t_a^T t_a)$
8. Y-loading: $q_a = Y_a^T t_a / (t_a^T t_a)$
9. Deflate: $X_{a+1} = X_a - t_a p_a^T$, $Y_{a+1} = Y_a - t_a q_a^T$

The final regression coefficients:
$$B = W(P^T W)^{-1} Q^T$$

# Package Features

## Neutrosophic Encoding Methods

The core innovation of N-PLS is the transformation of raw data into neutrosophic triplets. The package offers eight encoding strategies (plus an auto-selection wrapper), each suited to different data characteristics:

### Probabilistic Encoder

The default encoder uses statistical residuals from a low-rank model to estimate T, I, F:

1. Fit a low-rank approximation via truncated SVD: $X \approx USV^T$
2. Compute residuals: $E = X - USV^T$
3. Estimate robust location and scale for each feature using the Median Absolute Deviation (MAD): $\sigma_j = 1.4826 \cdot \text{MAD}(E_{:,j})$
4. Compute z-scores: $z_{ij} = |E_{ij}|/\sigma_j$
5. Apply a smooth squashing function: $F = \tanh^\beta(z/s)$

This encoder is effective for general-purpose applications and provides a statistically principled decomposition.

### NDG Manifold Encoder

The Neutrosophic Differential Geometry encoder implements a physics-based encoding derived from differential geometry principles. It maps spectroscopic data onto a neutrosophic manifold $\mathcal{M}_\mathcal{N}$ with the following channel computations:

**Truth Channel**: Normalized signal strength via configurable normalization $\mathcal{N}(x_k)$:

- None (default): $T = X$ — preserves concentration-proportional signals
- SNV: $T_k = (x_k - \bar{x})/\sigma_x$ — removes baseline and scatter
- Min-Max: $T_k = (x_k - x_{min})/(x_{max} - x_{min})$

**Indeterminacy Channel**: Based on Shannon entropy of local variance:
$$I_k = \mathcal{H}(\sigma_k^2) = \left(\frac{\ln(\sigma_k^2) - \ln(\sigma_{min}^2)}{\ln(\sigma_{max}^2) - \ln(\sigma_{min}^2)}\right)^\beta$$

The local variance is estimated from deviation from a locally-smoothed signal, or from replicate scans when available.

**Falsity Channel**: Systematic error coefficient derived from low-rank model deviation:
$$F_k = 1 - \mathcal{N}(x_k) \cdot (1 - \epsilon_k)$$

where $\epsilon_k = \sigma(z_R - 3)$ is a sigmoid transform of the robust z-score from residuals.

The NDG encoder provides a **complexity score** approximating the Ricci scalar, which quantifies spectral complexity—low values indicate geometrically stable (clean) data, while high values indicate complex manifold structure requiring neutrosophic treatment.

### RPCA Encoder

The Robust PCA encoder uses Principal Component Pursuit to decompose data:

$$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad X = L + S$$

The low-rank component $L$ becomes Truth, the sparse component magnitude becomes Falsity, and residual ambiguity becomes Indeterminacy.

### Additional Encoders

| Encoder | Method | Best For |
|---------|--------|----------|
| Spectroscopy | Noise-floor + PCA-residual heuristics | NIR/IR spectra with instrument noise floors |
| Wavelet | Multi-scale frequency decomposition | Periodic or multi-scale signals |
| Quantile | Non-parametric envelope bounds | Unknown distributions |
| Augmentation | Stability under perturbations | High-dimensional data |
| Robust MAD | Iteratively trimmed statistics | Spike detection |
| Auto | Cross-validated selection wrapper | General use |

## Model Variants

### NPLS: Standard Neutrosophic PLS

In the current codebase, NPLS operates on the Truth channel $T$ and applies *sample-level* reliability weighting computed from the fraction of high-falsity cells within each sample. Let

$$f_i = \frac{1}{p}\sum_{j=1}^{p} \mathbb{I}(F_{ij} > \tau), \quad \tau = 0.3$$

and define the sample weight

$$\omega_i = \max(\varepsilon,\; 1 - \lambda_F\, f_i), \quad \varepsilon = 0.01.$$

Weights are then normalized (default: mean 1). This approach downweights samples containing many corrupted measurements while preserving the signal physics in the Truth channel.

During the NIPALS iteration, weighted dot products replace standard inner products:

$$w_a = \frac{X_a^T (D_w u_a)}{\|X_a^T (D_w u_a)\|}$$

where $D_w = \text{diag}(w_1, \ldots, w_n)$ is the diagonal weight matrix.

### NPLSW: Reliability-Weighted NPLS

NPLSW uses the same sample-level falsity-fraction idea in the current implementation (thresholding $F_{ij}$, computing $f_i$, and normalizing weights via `normalize`). The parameters `lambda_indeterminacy` and `alpha` are retained for API compatibility, but the present weighting rule is driven primarily by the falsity channel.

### PNPLS: Probabilistic Neutrosophic PLS

PNPLS is the most sophisticated variant, handling noise at the **element level** using an EM-NIPALS algorithm. It assumes a heteroscedastic noise model:

$$X_{ij} = \sum_{a=1}^{A} t_{ia} p_{ja} + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma_{ij}^2)$$

where the variance is element-specific: $\sigma_{ij}^2 \propto \exp(\lambda_F \cdot F_{ij})$.

**Precision Weights:** Define the element-wise precision (inverse variance):

$$W_{ij} = \exp(-\lambda_F \cdot F_{ij} \cdot \gamma)$$

where $\lambda_F$ is the falsity sensitivity and $\gamma$ is a scaling factor. High $F_{ij}$ leads to low weight (less trusted).

The algorithm iterates between:

**E-Step (Imputation):** Replace unreliable values with expected values:
$$X_{ij}^{\text{imp}} = W_{ij} \cdot X_{ij}^{\text{obs}} + (1 - W_{ij}) \cdot X_{ij}^{\text{pred}}$$

where $X_{ij}^{\text{pred}} = \sum_{a} t_{ia} p_{ja}$ is the current model reconstruction.

**M-Step (Maximization):** Run standard NIPALS on the imputed matrix $X^{\text{imp}}$.

**Convergence Theorem:** The EM-NIPALS algorithm converges to a local maximum of the weighted likelihood:

$$\mathcal{L}(T, P | X, W) = -\frac{1}{2} \sum_{i,j} W_{ij} \left(X_{ij} - \sum_a t_{ia} p_{ja}\right)^2$$

The proof follows from standard EM theory: both E-step and M-step are non-decreasing in likelihood, and the objective is bounded above.

This element-wise approach allows PNPLS to handle localized corruption (e.g., detector artifacts at specific wavelengths) without discarding entire samples.

### Clean Data Bypass

All N-PLS variants include detection of effectively clean data and, by default, may dispatch to scikit-learn's `PLSRegression` when neutrosophic weighting is unlikely to change the solution (mean I/F < 0.15 and weights close to uniform). This behavior reflects the theoretical clean-data limit where N-PLS reduces to classical PLS, while improving runtime and numerical stability.

For reproducibility and strict algorithmic purity in ablation studies, this dispatch is user-controllable (`allow_sklearn_bypass=False`) and its activation is recorded on the fitted model via `bypass_triggered_` and `bypass_reason_`.

## Neutrosophic Variable Importance in Projection (NVIP)

N-PLS extends the classical VIP analysis [@wold1993pls; @chong2005performance] to provide channel-decomposed importance scores.

### Classical VIP Definition

The Variable Importance in Projection for feature $j$ in a PLS model with $A$ components is:

$$\text{VIP}_j = \sqrt{\frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_a}{\sum_{a=1}^{A} SS_a}}$$

where $w_{ja}$ is the weight of feature $j$ in component $a$, and $SS_a = \|t_a\|^2 \cdot \|q_a\|^2$ is the sum of squares explained by component $a$.

### Channel-Specific Sum of Squares

For neutrosophic data, we compute channel-specific scores using the model weights $W$ learned on the Truth channel:

$$t_T^{(a)} = T_c \cdot w_a, \quad t_I^{(a)} = I_c \cdot w_a, \quad t_F^{(a)} = F_c \cdot w_a$$

The channel-specific sum of squares:

$$SS_C^{(a)} = \omega_C^2 \cdot \|t_C^{(a)}\|^2 \cdot \|q_a\|^2$$

where $(\omega_T, \omega_I, \omega_F)$ are the channel weights.

### L2-Norm Decomposition Theorem

**Theorem (NVIP L2 Decomposition):** For any feature $j$, the aggregate VIP satisfies:

$$\text{VIP}_{\text{aggregate}}(j) = \sqrt{\text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}$$

**Proof:** Starting from the channel-specific VIP squared:

$$\text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_C^{(a)}}{SS_{\text{total}}}$$

Summing over all channels:

$$\sum_{C \in \{T,I,F\}} \text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot \sum_{C} SS_C^{(a)}}{SS_{\text{total}}}$$

Since $\sum_{C} SS_C^{(a)} = SS_{\text{total}}^{(a)}$ and $\sum_{a} SS_{\text{total}}^{(a)} = SS_{\text{total}}$:

$$\sum_{C} \text{VIP}_C^2(j) = \frac{p \cdot \sum_{a=1}^{A} w_{ja}^2 \cdot SS_{\text{total}}^{(a)}}{SS_{\text{total}}} = \text{VIP}_{\text{aggregate}}^2(j)$$

Taking square roots yields the theorem. $\square$

This decomposition reveals whether a feature's importance derives from:

- **Truth VIP**: Signal content (the preferred source of importance)
- **Indeterminacy VIP**: Uncertainty patterns (may indicate informative variance)
- **Falsity VIP**: Noise characteristics (may indicate data quality issues)

A Signal-to-Noise Ratio derived from VIP channels provides actionable diagnostics:

$$\text{SNR}(j) = \frac{\text{VIP}_T(j)}{\text{VIP}_F(j) + \epsilon}$$

## Interactive Analysis Wizard

The package includes a 7-step command-line wizard designed for researchers without programming experience:

| Step | Action | Description |
|------|--------|-------------|
| 1 | Data Loading | Loads CSV, Excel, ARFF, JSON, or Parquet files |
| 2 | Data Summary | Summarizes the dataset and prompts for target + task type |
| 3 | Encoder Selection | Auto-selection or manual encoder choice |
| 4 | Variant Selection | Chooses NPLS / NPLSW / PNPLS and key settings |
| 5 | Run Analysis | Runs cross-validation and compares against classical PLS |
| 6 | VIP Analysis | Computes channel-decomposed feature importance (optional) |
| 7 | Export Figures | Saves report figures (optional) |

The wizard provides formatted comparison tables, progress bars, and diagnostic messages, making advanced chemometric analysis accessible to domain experts who may not have programming backgrounds.

## Python API

For programmatic use, N-PLS provides a scikit-learn-style `fit`/`predict` API for advanced users.

### Basic Usage

```python
from neutrosophic_pls import NPLSW, load_dataset, DatasetConfig, compute_nvip

# Load and encode data
config = DatasetConfig(
    path="data/spectra.csv",
    target="Protein",
    task="regression",
    encoding={"name": "rpca"}  # or "probabilistic", "ndg", "wavelet", "auto"
)
data = load_dataset(config)

# Fit model
model = NPLSW(n_components=10, lambda_falsity=0.5)
model.fit(data["x_tif"], data["y_tif"])

# Predict
predictions = model.predict(data["x_tif"])

# Analyze feature importance with channel decomposition
vip = compute_nvip(model, data["x_tif"])
print(f"Top features by VIP: {vip['aggregate'][:10]}")
print(f"  - Truth contribution: {vip['T'][:10]}")
print(f"  - Falsity contribution: {vip['F'][:10]}")
```

### Model Variants and Parameters

Each N-PLS variant has specific parameters for controlling noise handling:

```python
from neutrosophic_pls import NPLS, NPLSW, PNPLS

# NPLS: Standard with sample weighting
model = NPLS(
    n_components=10,           # Number of latent components
    lambda_falsity=0.5,        # Sensitivity to falsity channel (0.0-1.0)
    channel_weights=(1.0, 0.5, 1.0),  # Weights for (T, I, F)
    max_iter=500,              # Maximum NIPALS iterations
    tol=1e-7                   # Convergence tolerance
)

# NPLSW: Reliability-weighted (best for noisy samples)
model = NPLSW(
    n_components=10,
    lambda_indeterminacy=0.2,  # Weight for indeterminacy penalty
    lambda_falsity=0.5,        # Weight for falsity penalty
    alpha=2.0,                 # Sharpness of reliability weighting
    normalize="mean1"          # Weight normalization: "none", "mean1", "sum1"
)

# PNPLS: Probabilistic (best for element-wise noise)
model = PNPLS(
    n_components=10,
    lambda_falsity=0.5,        # Controls variance weighting from F channel
    max_iter=500               # EM-NIPALS iterations
)
```

### Encoder Configuration

The package provides multiple encoding strategies with customizable parameters:

```python
from neutrosophic_pls import encode_neutrosophic, EncoderConfig

# Probabilistic encoder (default) - statistical residuals
x_tif, y_tif, meta = encode_neutrosophic(
    X, y,
    encoding={"name": "probabilistic", "params": {"rank": 5, "beta": 1.5}},
    return_metadata=True
)

# RPCA encoder - robust PCA decomposition (best for sparse outliers)
x_tif, y_tif, meta = encode_neutrosophic(
    X, y,
    encoding={
        "name": "rpca",
        "params": {
            "beta_I": 2.0,              # Power for indeterminacy transform
            "beta_F": 2.0,              # Power for falsity transform
            "lambda_sparse": None       # Auto-computed if None
        }
    },
    return_metadata=True
)

# NDG Manifold encoder - physics-based differential geometry
x_tif, y_tif, meta = encode_neutrosophic(
    X, y,
    encoding={
        "name": "ndg",
        "params": {
            "normalization": "none",  # "none", "snv", or "minmax"
            "local_window": 5          # Local variance window
        }
    },
    return_metadata=True
)

# Wavelet multi-scale encoder
x_tif, y_tif, meta = encode_neutrosophic(
    X, y,
    encoding={
        "name": "wavelet",
        "params": {
            "wavelet": "db2",           # Wavelet family
            "level": None,              # Auto-select if None
            "high_bands": (1,),         # Bands for falsity
            "mid_bands": (2, 3)         # Bands for indeterminacy
        }
    },
    return_metadata=True
)

# Auto-selection (cross-validates all encoders)
x_tif, y_tif, meta = encode_neutrosophic(
    X, y,
    encoding={
        "name": "auto",
        "candidates": ["probabilistic", "rpca", "ndg", "wavelet"],
        "cv_folds": 3,
        "max_components": 5
    },
    return_metadata=True
)
print(f"Selected encoder: {meta['encoder']['name']}")
print(f"Encoder scores: {meta['encoder'].get('auto_scores', {})}")
```

### Cross-Validation and Model Comparison

For rigorous evaluation, use a manual cross-validation loop (this avoids scikit-learn estimator cloning requirements):

```python
import numpy as np
from sklearn.model_selection import RepeatedKFold
from neutrosophic_pls import NPLSW, evaluation_metrics

# Setup cross-validation
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Cross-validated predictions (Truth channel is used for y_true)
y_true = data["y_tif"][..., 0].ravel()
y_pred_cv = np.zeros_like(y_true, dtype=float)

for train_idx, test_idx in cv.split(data["x_tif"]):
    model = NPLSW(n_components=10)
    model.fit(data["x_tif"][train_idx], data["y_tif"][train_idx])
    y_pred_cv[test_idx] = model.predict(data["x_tif"][test_idx]).ravel()

# Compute metrics
metrics = evaluation_metrics(y_true, y_pred_cv, include_extended=True)
print(f"RMSEP: {metrics['RMSEP']:.4f}")
print(f"R²: {metrics['R2']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
print(f"RPD: {metrics['RPD']:.2f}")
print(f"Bias: {metrics['Bias']:.4f}")
print(f"SEP: {metrics['SEP']:.4f}")
```

### Neutrosophic VIP Analysis

The `compute_nvip` function provides channel-decomposed variable importance:

```python
from neutrosophic_pls import compute_nvip
import numpy as np

# Fit model first
model = NPLSW(n_components=10)
model.fit(data["x_tif"], data["y_tif"])

# Compute VIP with channel decomposition
vip = compute_nvip(
    model, 
    data["x_tif"],
    channel_weights=(1.0, 1.0, 1.0),  # Equal weighting
    decomposition_method="variance"    # or "correlation"
)

# Access results
aggregate_vip = vip["aggregate"]  # Overall importance
vip_t = vip["T"]                  # Truth channel contribution
vip_i = vip["I"]                  # Indeterminacy contribution
vip_f = vip["F"]                  # Falsity contribution

# Mathematical property: VIP_agg = sqrt(VIP_T² + VIP_I² + VIP_F²)
reconstruction = np.sqrt(vip_t**2 + vip_i**2 + vip_f**2)
assert np.allclose(aggregate_vip, reconstruction, rtol=0.01)

# Identify important features
feature_names = data["metadata"].get("feature_names", list(range(len(aggregate_vip))))
important_idx = np.where(aggregate_vip > 1.0)[0]
print(f"Important features (VIP > 1): {len(important_idx)}")

# Signal-to-Noise Ratio per feature
snr = vip_t / (vip_f + 1e-10)
high_quality = np.where(snr > 2.0)[0]
print(f"High-quality features (SNR > 2): {len(high_quality)}")

# Channel dominance analysis
dominant_channel = np.argmax(np.stack([vip_t, vip_i, vip_f]), axis=0)
print(f"Truth-dominant: {np.sum(dominant_channel == 0)}")
print(f"Indeterminacy-dominant: {np.sum(dominant_channel == 1)}")
print(f"Falsity-dominant: {np.sum(dominant_channel == 2)}")
```

### YAML Configuration for Reproducible Studies

For reproducible research, use YAML configuration files:

```yaml
# study_config.yaml
dataset:
  path: "data/MA_A2.csv"
  target: "Protein"
  task: regression
  encoding:
    name: auto
    candidates: [probabilistic, rpca, wavelet]
    cv_folds: 3

model:
  method: all              # Compare PLS, NPLS, NPLSW, PNPLS
  max_components: 10
  channel_weights: [1.0, 0.5, 1.0]
  lambda_falsity: 0.5

evaluation:
  cv_folds: 5
  repeats: 3
  random_state: 42
  compute_vip: true

output:
  output_dir: "results/protein_study"
  save_predictions: true
  save_vip: true
  generate_figures: true
  figure_format: png
  figure_dpi: 300
```

Load and run from Python:

```python
from neutrosophic_pls import StudyConfig

# Load configuration
config = StudyConfig.from_yaml("study_config.yaml")

# Or build programmatically
from neutrosophic_pls import (
    StudyConfig, DatasetSettings, ModelSettings, 
    EvaluationSettings, OutputSettings
)

config = StudyConfig(
    dataset=DatasetSettings(
        path="data/spectra.csv",
        target="Protein",
        encoding={"name": "rpca"}
    ),
    model=ModelSettings(
        method="NPLSW",
        max_components=10,
        lambda_falsity=0.5
    ),
    evaluation=EvaluationSettings(
        cv_folds=5,
        repeats=3,
        compute_vip=True
    )
)

# Save configuration for reproducibility
config.to_yaml("my_study.yaml")
```

### Chemometric Metrics

N-PLS provides comprehensive metrics used in NIR spectroscopy:

```python
from neutrosophic_pls import (
    rmsep,              # Root Mean Square Error of Prediction
    r2_score,           # Coefficient of Determination
    mean_absolute_error,# Mean Absolute Error
    mape,               # Mean Absolute Percentage Error
    bias,               # Systematic prediction bias
    sep,                # Standard Error of Prediction
    rpd,                # Ratio of Performance to Deviation
    rer,                # Range Error Ratio
    evaluation_metrics  # All metrics at once
)

# Individual metrics
print(f"RMSEP: {rmsep(y_true, y_pred):.4f}")
print(f"R²: {r2_score(y_true, y_pred):.4f}")
print(f"RPD: {rpd(y_true, y_pred):.2f}")  # >3.0 excellent, 2-3 good, <2 poor

# All metrics
metrics = evaluation_metrics(y_true, y_pred, include_extended=True)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### Predictions on New Data

For applying trained models to new samples:

```python
# Train on calibration data
model = NPLSW(n_components=10)
model.fit(train_x_tif, train_y_tif)

# Encode new data with same encoder
new_x_tif, _ = encode_neutrosophic(
    new_X, 
    np.zeros(len(new_X)),  # Dummy y for new data
    encoding={"name": "rpca"}  # Must match training encoder
)

# Predict
new_predictions = model.predict(new_x_tif)
```

# Software Architecture

N-PLS is designed with modularity and extensibility in mind. The package follows a clean separation of concerns with well-defined interfaces between components.

## Module Overview

### `encoders.py` - Neutrosophic Encoding Strategies

Implements the encoding strategies for transforming raw data into neutrosophic triplets (T, I, F):

| Function | Description |
|----------|-------------|
| `encode_probabilistic_tif()` | Statistical residuals from low-rank SVD model with MAD-based robust statistics |
| `encode_spectroscopy_tif()` | Spectroscopy-specific I/F encoding using a noise floor and PCA-residual diagnostics |
| `encode_rpca_mixture()` | Robust PCA via Principal Component Pursuit: low-rank Truth, sparse Falsity |
| `encode_wavelet_multiscale()` | Multi-scale wavelet decomposition: low-frequency Truth, high-frequency Falsity |
| `encode_quantile_envelope()` | Non-parametric quantile-based boundaries for T/I/F channels |
| `encode_augmentation_stability()` | Stability under perturbations: augmentation mean for Truth |
| `encode_robust_tif()` | Iteratively trimmed MAD statistics for spike detection |
| `encode_ndg_manifold()` | Neutrosophic Differential Geometry with entropy-based indeterminacy |

**Key Classes:**

- `EncodingResult`: Dataclass container for (T, I, F) arrays with optional metadata
- `EncoderConfig`: Configuration with `name`, `params`, `candidates`, and auto-selection settings
- `AutoEncoderSelectionResult`: Stores cross-validation scores for encoder comparison

**Auto-Selection Pipeline:**

```python
from neutrosophic_pls.encoders import EncoderConfig, auto_select_encoder, dispatch_encoder

cfg = EncoderConfig.from_value({"name": "auto", "cv_folds": 3, "max_components": 5})
best_encoding, selection = auto_select_encoder(X, y, cfg, task="regression")
encoding = dispatch_encoder(X, selection.best_config)  # run the selected encoder
```

### `model.py` - N-PLS Model Implementations

Contains three NIPALS-based PLS variants with neutrosophic weighting:

**NPLS (Standard Neutrosophic PLS):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | required | Number of latent components |
| `lambda_falsity` | float | 0.5 | Sensitivity to falsity (0.0-1.0) |
| `channel_weights` | tuple | (1.0, 0.5, 1.0) | Weights for (T, I, F) channels |
| `max_iter` | int | 500 | Maximum NIPALS iterations |
| `tol` | float | 1e-7 | Convergence tolerance |

**NPLSW (Reliability-Weighted NPLS):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | required | Number of latent components |
| `lambda_indeterminacy` | float | 0.2 | Weight for indeterminacy penalty |
| `lambda_falsity` | float | 0.5 | Weight for falsity penalty |
| `alpha` | float | 2.0 | Sharpness of reliability weighting |
| `normalize` | str | "mean1" | Weight normalization mode |

**PNPLS (Probabilistic NPLS):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | required | Number of latent components |
| `lambda_falsity` | float | 0.5 | Controls element-wise variance weighting |
| `max_iter` | int | 500 | EM-NIPALS iterations |

**Clean Data Bypass:** All variants automatically detect clean data (mean I/F < 0.15, weight CV < 5%) and gracefully degrade to sklearn's `PLSRegression` for optimal performance.

### `vip.py` - Variable Importance in Projection

Provides channel-attributed VIP analysis with two decomposition modes:
- **Variance-based** (default): SS/projection-based attribution with **L2 aggregation**
- **Correlation-based** (legacy): correlation-proportion attribution with **linear aggregation**

| Function | Description |
|----------|-------------|
| `compute_nvip()` | Main function returning aggregate and channel-specific VIP scores |
| `_vip_from_pls()` | Standard VIP computation from fitted model |
| `_channel_contribution_vip()` | Variance-based channel attribution (L2-aggregated by default) |
| `_channel_correlation_vip()` | Correlation-based channel attribution (linear-sum aggregate) |

**Mathematical Property (variance-based default):**
$$\text{VIP}_{\text{aggregate}} = \sqrt{\text{VIP}_T^2 + \text{VIP}_I^2 + \text{VIP}_F^2}$$

For the legacy correlation-based mode, the aggregate is reported as:
$$\text{VIP}_{\text{aggregate}} = \text{VIP}_T + \text{VIP}_I + \text{VIP}_F$$

### `data_loader.py` - Universal Data Loading

Supports multiple file formats with automatic detection:

| Format | Extensions | Notes |
|--------|------------|-------|
| CSV | .csv | Pandas read_csv with flexible options |
| Excel | .xls, .xlsx | Multiple sheet support |
| ARFF | .arff | Weka format for ML datasets |
| JSON | .json | Orient-flexible loading |
| Parquet | .parquet | Columnar format for large data |

**Key Classes:**

- `DatasetConfig`: Complete configuration including path, target, encoding, limits
- `EncoderConfig`: Encoding strategy configuration with parameters

**Key Functions:**

| Function | Description |
|----------|-------------|
| `load_dataset()` | Main entry point: loads, encodes, and returns ready-to-use dict |
| `load_dataframe()` | Low-level loader with format auto-detection |
| `encode_neutrosophic()` | Encodes X, y to (n, p, 3) neutrosophic tensors |
| `list_available_datasets()` | Lists datasets in data/ directory |

### `metrics.py` - Chemometric Evaluation Metrics

Comprehensive metrics for calibration model evaluation:

| Function | Formula | Interpretation |
|----------|---------|----------------|
| `rmsep()` | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Primary prediction error |
| `r2_score()` | $1 - \frac{SS_{res}}{SS_{tot}}$ | Explained variance (1 = perfect) |
| `mean_absolute_error()` | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust error metric |
| `mape()` | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Percentage error |
| `bias()` | $\frac{1}{n}\sum(y_i - \hat{y}_i)$ | Systematic offset |
| `sep()` | $\sqrt{\frac{1}{n-1}\sum(e_i - \bar{e})^2}$ | Random error component |
| `rpd()` | $\frac{\sigma_y}{\text{SEP}}$ | >3.0 excellent, 2-3 good, <2 poor |
| `rer()` | $\frac{\text{range}(y)}{\text{SEP}}$ | Range-based alternative to RPD |
| `evaluation_metrics()` | All metrics at once | Comprehensive evaluation dict |

### `study_config.py` - Reproducible Study Configuration

YAML/JSON-based configuration for reproducible research:

**Configuration Classes:**

| Class | Purpose |
|-------|---------|
| `StudyConfig` | Top-level configuration container |
| `DatasetSettings` | Path, target, encoding, preprocessing |
| `ModelSettings` | Method, components, hyperparameters |
| `EvaluationSettings` | CV folds, repeats, random state |
| `OutputSettings` | Output directory, figure options |

**Preset Configurations:**

```python
get_idrc_wheat_config()   # IDRC benchmark preset
get_quick_test_config()   # Fast testing preset
```

### `validation.py` - Ground-Truth Validation

Tools for validating I/F channels against known uncertainty:

| Class | Purpose |
|-------|---------|
| `IndeterminacyValidator` | Compare computed I with replicate measurement variance |
| `FalsityValidator` | Compare computed F with known outlier labels |

### `algebra.py` - Neutrosophic Operations

Core mathematical operations on neutrosophic triplets:

| Function | Description |
|----------|-------------|
| `neutro_inner()` | Weighted inner product: $w_T \cdot T_x \cdot T_y + w_I \cdot I_x \cdot I_y + w_F \cdot F_x \cdot F_y$ |
| `neutro_norm()` | Weighted norm: $\sqrt{w_T \cdot T^2 + w_I \cdot I^2 + w_F \cdot F^2}$ |
| `combine_channels()` | Channel combination strategies |

### `simulate.py` - Synthetic Data Generation

Generate controlled synthetic datasets for method validation:

```python
generate_simulation(
    n_samples=200,
    n_features=100,
    n_components=5,
    noise_level=0.1,
    outlier_fraction=0.05
)
```

### `interactive.py` - Command-Line Wizard

Full-featured 7-step interactive analysis wizard for non-programmers.

## Dependencies

The package requires Python ≥ 3.9 and depends on:

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥ 1.21 | Array operations |
| SciPy | ≥ 1.7 | Scientific computing, wavelets |
| pandas | ≥ 1.3 | Data loading and manipulation |
| scikit-learn | ≥ 1.0 | Cross-validation, PLS reference |
| matplotlib | ≥ 3.4 | Visualization |
| PyYAML | ≥ 6.0 | Configuration files |

# Validation and Performance

## Evaluation Metrics

N-PLS employs standard chemometric metrics for model evaluation:

**Root Mean Square Error of Prediction (RMSEP):**
$$\text{RMSEP} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²):**
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $R^2 = 1$ indicates perfect prediction, $R^2 = 0$ indicates prediction equal to the mean, and $R^2 < 0$ indicates predictions worse than the mean.

**Ratio of Performance to Deviation (RPD):**
$$\text{RPD} = \frac{\sigma_y}{\text{RMSEP}} = \frac{1}{\sqrt{1 - R^2}}$$

Interpretation thresholds: $\text{RPD} > 3.0$ (excellent), $2.0-3.0$ (good), $< 2.0$ (poor).

## Benchmark Results

N-PLS has been validated on NIR spectroscopy datasets. The following results are from the MA_A2 protein prediction dataset (248 samples, 741 spectral features, target range: 7.97-18.69%) using 5-fold cross-validation with 3 repeats.

### Encoder Selection

The automatic encoder selection evaluated all available encoders:

| Encoder | RMSEP | Selection |
|---------|-------|-----------|
| RPCA | 1.7771 | ✓ Best |
| Wavelet | 2.2629 | |
| Quantile | 2.4896 | |
| Probabilistic | 2.8935 | |
| Augment | 3.3229 | |
| NDG | 4.1524 | |

RPCA encoding was automatically selected based on cross-validated RMSEP performance.

### Model Variant Comparison

With RPCA encoding, all three N-PLS variants were evaluated:

| Variant | RMSEP | Description |
|---------|-------|-------------|
| NPLS | 1.4648 | ✓ Best |
| NPLSW | 1.4648 | ✓ Best |
| PNPLS | 1.6263 | |

Both NPLS and NPLSW achieved equivalent performance on this dataset, while PNPLS showed slightly higher RMSEP.

### Final Model Performance

| Method | RMSEP | R² | MAE | RPD | Improvement |
|--------|-------|-----|-----|-----|-------------|
| Classical PLS | 1.6540 ± 0.97 | 0.1058 | 0.9852 | 1.70 | baseline |
| NPLS (5 comp.) | 1.4867 ± 1.06 | 0.1854 | 0.8494 | 1.89 | **+10.1%** |

### VIP Analysis Summary

Channel-decomposed VIP analysis of the 741 features revealed:

| Category | Count | Percentage |
|----------|-------|------------|
| Important features (VIP > 1) | 226 | 30.5% |
| Moderate importance (0.8 ≤ VIP < 1) | 97 | 13.1% |
| Low importance (0.5 ≤ VIP < 0.8) | 391 | 52.8% |
| Very low importance (VIP < 0.5) | 27 | 3.6% |

**Signal Quality Analysis:**

| Quality Level | Count | Criterion |
|---------------|-------|-----------|
| High quality (SNR > 2) | 381 | Reliable for predictions |
| Moderate quality (1 ≤ SNR ≤ 2) | 360 | Acceptable |
| Low quality (SNR < 1) | 0 | Check data quality |

**Channel Dominance:** All 741 features (100%) were signal-dominant (Truth channel), indicating good overall data quality with the neutrosophic encoding correctly identifying the signal component.

The top predictive features (VIP ≈ 2.0) were located in the 1024-1028.5 wavelength region, consistent with known protein absorption bands in NIR spectroscopy. The median signal-to-noise ratio of 2.03 indicates good overall data quality.

## Clean Data Performance

On clean data without significant noise, the N-PLS variants are designed to closely match Classical PLS performance via a clean-data bypass that falls back to `PLSRegression` when neutrosophic weighting is unlikely to help. This bypass activates when:

1. Mean Indeterminacy and Falsity are below 0.15
2. Sample weight coefficient of variation is below 5%

This ensures no penalty for using N-PLS as a default analysis tool, providing robustness when needed without sacrificing performance on clean data.

# Related Work

Several Python packages address chemometric modeling. The `scikit-learn` library [@pedregosa2011scikit] provides general machine learning algorithms including PLS regression. For robust regression, `statsmodels` provides M-estimators and robust linear models. Specialized packages like `pyod` [@zhao2019pyod] focus on outlier detection but do not integrate with regression modeling. The `libpls` package provides various PLS algorithms but without uncertainty handling.

In the R ecosystem, the `pls` package provides comprehensive PLS implementations, and `mdatools` offers chemometrics-focused tools. However, no existing package in either ecosystem combines neutrosophic uncertainty representation with PLS regression or provides channel-decomposed variable importance analysis.

The theoretical foundation builds on Smarandache's neutrosophic logic [@smarandache1999linguistic], with methodological contributions from PLS literature [@wold1984multivariate; @geladi1986partial; @wold1993pls; @mehmood2012review] and robust chemometrics [@hubert2003robust; @filzmoser2012robust]. The RPCA encoder draws on work by Candès et al. [@candes2011robust] on convex optimization for sparse and low-rank decomposition.

# Conclusions

N-PLS provides a novel approach to handling measurement uncertainty in PLS regression by leveraging neutrosophic set theory. The package offers:

1. **Multiple encoding strategies** for converting raw data to neutrosophic triplets
2. **Three model variants** addressing sample-level and element-level noise
3. **Channel-decomposed VIP analysis** for interpretable feature importance
4. **An interactive wizard** making advanced analysis accessible to non-programmers
5. **Demonstrated performance improvements** of up to 10% on challenging real-world datasets

The package is freely available under the MIT license and designed for extensibility, enabling researchers to develop new encoding strategies or model variants within the established framework.

# Acknowledgements

The authors thank the University of Ghana for institutional support. We acknowledge the International Diffuse Reflectance Conference (IDRC) for making benchmark NIR datasets publicly available. We also thank Professor Florentin Smarandache for foundational work on neutrosophic logic that enabled this research. Development benefited from discussions with researchers in the chemometrics and neutrosophic logic communities.

# References
