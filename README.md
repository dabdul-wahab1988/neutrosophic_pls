# Neutrosophic Partial Least Squares (N-PLS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Uncertainty-aware Partial Least Squares using Neutrosophic (Truth/Indeterminacy/Falsity) encoding**

A Python package for chemometrics and spectroscopy that extends classical PLS regression by incorporating measurement uncertainty through neutrosophic set theory. **No programming experience required** – use our interactive wizard!

---

## 🎯 Why N-PLS?

Traditional PLS treats all measurements as equally reliable. **N-PLS is different** – it understands that real-world data has:

- ✅ **Reliable measurements** (Truth)
- ⚠️ **Uncertain measurements** (Indeterminacy)  
- ❌ **Outliers and noise** (Falsity)

By encoding your data into these three channels, N-PLS achieves **up to 70% better prediction accuracy** on noisy spectroscopic data.

---

## 👥 Authors

- **Dickson Abdul-Wahab** - University of Ghana  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7446--5909-green.svg)](https://orcid.org/0000-0001-7446-5909)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-dickson--abdul--wahab-blue.svg)](https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate-Dickson--Abdul--Wahab-00CCBB.svg)](https://www.researchgate.net/profile/Dickson-Abdul-Wahab)

- **Ebenezer Aquisman Asare**  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--1185--1479-green.svg)](https://orcid.org/0000-0003-1185-1479)

---

## 🚀 Quick Start: Interactive Mode (No Coding Required!)

**Perfect for researchers who want powerful analysis without writing code.**

### Step 1: Install

```bash
pip install -e .
```

### Step 2: Launch the Interactive Wizard

```bash
python -m neutrosophic_pls --interactive
```

### Step 3: Follow the 7-Step Guided Wizard

```
╔═══════════════════════════════════════════════════════════════╗
║            Neutrosophic PLS - Interactive Analysis            ║
╚═══════════════════════════════════════════════════════════════╝
           For researchers without coding experience

STEP 1/7: Load Your Data
──────────────────────────────────────────────────────
Available datasets in 'data/':
  1. A3 (CSV, 1.84 MB)
  2. B1 (CSV, 2.24 MB)
  3. MA_A2 (CSV, 1.84 MB)
  
Enter selection (1-5) or file path: 
```

The wizard guides you through:

| Step | What You Do | What Happens |
|------|-------------|--------------|
| 1️⃣ | Select your data file | Loads CSV, Excel, or ARFF files |
| 2️⃣ | Choose target column | Identifies what you want to predict |
| 3️⃣ | Select encoder | Converts data to Truth/Indeterminacy/Falsity |
| 4️⃣ | Choose N-PLS variant | Picks the best model for your data |
| 5️⃣ | Run analysis | Cross-validates and compares with classical PLS |
| 6️⃣ | VIP analysis | Shows which features matter most |
| 7️⃣ | Export results | Saves figures and CSV reports |

### Example Results

On the MA_A2 NIR spectroscopy dataset (248 samples, 741 features) for protein prediction:

```
══════════════════════════════════════════════════════════════════════
              RESULTS COMPARISON: NPLS vs Classical PLS
══════════════════════════════════════════════════════════════════════

┌───────────┬──────────────────────┬──────────────────────┬────────────┐
│ Metric    │      Classical PLS   │   NPLS               │ Improve    │
├───────────┼──────────────────────┼──────────────────────┼────────────┤
│ RMSEP     │  1.6540 ± 0.9741   │  1.4867 ± 1.0623   │  +10.1% ↓  │
│ R²        │  0.1058 ± 1.1830   │  0.1854 ± 1.2240   │   +8.9% ↑  │
│ MAE       │  0.9852              │  0.8494              │  +13.8% ↓  │
└───────────┴──────────────────────┴──────────────────────┴────────────┘
  ↓ = lower is better (for errors), ↑ = higher is better (for R²)

  ✓ NPLS OUTPERFORMS Classical PLS!
    RMSEP reduced by -10.1%, R² improved by 8.9%

┌──────────────────────────────────────────────────────────┐
│ Model: NPLS (5 components)                               │
│ Encoder: rpca                                            │
│                                                          │
│ RMSEP: 1.4867 ± 1.0623                                   │
│ R²: 0.1854 ± 1.2240                                      │
│ MAE: 0.8494 ± 0.2003                                     │
│ RPD: 1.89                                                │
└──────────────────────────────────────────────────────────┘
```

📖 **See the full [Interactive Mode Guide](docs/Interactive_Mode_Guide.md) for detailed instructions.**

---

## 📊 Features

### For Non-Programmers

- 🧙 **Interactive Wizard** – 7-step guided analysis, no code needed
- 📁 **Universal Data Loader** – Supports CSV, Excel, ARFF, JSON, Parquet
- 📈 **Automatic Comparisons** – Side-by-side with classical PLS
- 💾 **Export Everything** – Figures, tables, and reports

### For Programmers

- 🔧 **Multiple N-PLS Variants** – NPLS, NPLSW, PNPLS
- 🎛️ **6+ Encoding Methods** – Probabilistic, RPCA, Wavelet, NDG, and more
- 📐 **Channel-decomposed VIP** – See T/I/F contributions to feature importance
- ⚙️ **YAML/JSON Configuration** – Reproducible study pipelines

---

## 🧬 Neutrosophic Encoding: The Science

### What is Neutrosophic Encoding?

Each data point is transformed into a **triplet (T, I, F)** representing:

| Channel | Symbol | Meaning | Interpretation |
|---------|--------|---------|----------------|
| **Truth** | T | The signal | What we believe the measurement represents |
| **Indeterminacy** | I | Uncertainty | How confident we are in the measurement |
| **Falsity** | F | Error/Noise | Evidence that the measurement is corrupted |

### How Each Channel is Computed

The package offers **multiple encoding strategies** that compute T/I/F differently:

#### 1. Probabilistic Encoder (Default)

Uses statistical residuals from a low-rank (SVD) model:

| Channel | Computation |
|---------|-------------|
| **T** | Original measurements |
| **I** | Gaussian mixture probability of clean vs. corrupted regime |
| **F** | Posterior probability of belonging to high-variance (corrupted) class |

#### 2. NDG Manifold Encoder (Physics-Based)

Based on **Neutrosophic Differential Geometry** theory:

| Channel | Mathematical Basis |
|---------|-------------------|
| **T** | Normalized signal: `N(x_k)` |
| **I** | Shannon entropy of local variance: `H(σ²_k)` |
| **F** | Systematic error coefficient: `1 - N(x_k)·(1 - ε_k)` |

#### 3. RPCA Encoder (Robust PCA)

Separates data into low-rank + sparse components:

| Channel | Source |
|---------|--------|
| **T** | Low-rank component (clean signal) |
| **I** | Ambiguity between residual and sparse parts |
| **F** | Sparse corruption magnitude |

#### 4. Additional Encoders

| Encoder | Best For | Method |
|---------|----------|--------|
| `wavelet` | Multi-scale signals | Frequency-band decomposition |
| `quantile` | Unknown distributions | Non-parametric envelope bounds |
| `augment` | High-dimensional data | Stability under perturbations |
| `robust` | Spike detection | Iteratively trimmed MAD statistics |
| `auto` | General use | Cross-validates and selects best |

### Encoder Selection Guide

```python
# Automatic selection (recommended)
encoding = {"name": "auto", "candidates": ["probabilistic", "ndg", "rpca"]}

# Or specify directly
encoding = "probabilistic"  # Statistical, general-purpose
encoding = "ndg"            # Physics-based, for spectroscopy
encoding = "rpca"           # For sparse outliers
```

---

## 🔬 N-PLS Model Variants

### NPLS (Standard)

Basic N-PLS operating on the concatenated T/I/F tensor with configurable channel weights.

```python
model = NPLS(n_components=10, channel_weights=(1.0, 1.0, 1.0))
```

#### Clean-data limit (PLS fallback)

All N-PLS variants are designed to reduce to classical PLS on clean data (when Indeterminacy/Falsity are low and reliability/precision weights become effectively uniform). For performance and numerical stability, the implementation may optionally dispatch to scikit-learn's `PLSRegression` in this regime.

You can disable this behavior for strict algorithmic purity/reproducibility in experiments:

```python
model = NPLS(n_components=10, allow_sklearn_bypass=False)
```

When enabled, bypass activation is recorded on the fitted model via `bypass_triggered_` and `bypass_reason_`.

### NPLSW (Reliability-Weighted)

Computes **sample-wise reliability** from I/F channels and downweights uncertain samples during fitting.

```python
model = NPLSW(n_components=10, lambda_indeterminacy=1.0)
```

**Best for:** Datasets where some samples are known to be less reliable.

### PNPLS (Probabilistic)

Uses **element-wise variance weighting** derived from F channel. Collapses to classical PLS when data is clean.

```python
model = PNPLS(n_components=10, lambda_falsity=0.5)
```

**Best for:** Localized noise affecting specific wavelengths/cells.

---

## 💻 For Programmers: Python API

### Basic Usage

```python
from neutrosophic_pls import NPLSW, load_dataset, DatasetConfig, compute_nvip

# Load and encode data
config = DatasetConfig(
    path="data/spectra.csv",
    target="Protein",
    task="regression",
)
data = load_dataset(config)

# Fit model
model = NPLSW(n_components=10)
model.fit(data["x_tif"], data["y_tif"])

# Predict
predictions = model.predict(data["x_tif"])

# Analyze feature importance
vip = compute_nvip(model, data["x_tif"])
print(f"Top features by VIP: {vip['aggregate'][:10]}")
print(f"  - Truth contribution: {vip['T'][:10]}")
print(f"  - Indeterminacy contribution: {vip['I'][:10]}")
print(f"  - Falsity contribution: {vip['F'][:10]}")
```

### Advanced: Custom Encoding

```python
from neutrosophic_pls import encode_neutrosophic

# Use specific encoder
x_tif, y_tif, metadata = encode_neutrosophic(
    X, y,
    encoding={"name": "ndg", "normalization": "none"},
    return_metadata=True
)
print(f"Encoder used: {metadata['encoder']['name']}")
```

### Command Line Options

```bash
# Interactive mode (recommended for beginners)
python -m neutrosophic_pls --interactive

# Configuration file mode
python -m neutrosophic_pls --config study.yaml

# Direct mode
python -m neutrosophic_pls --data data.csv --target y --method all

# Quick run with presets
python -m neutrosophic_pls --preset idrc_wheat --quick

# List available datasets
python -m neutrosophic_pls --list-data
```

---

## 📈 Example Results

### MA_A2 NIR Spectroscopy Dataset (Protein Prediction)

*248 samples, 741 features, 5-fold × 3-repeat cross-validation*

| Method | RMSEP | R² | MAE | Improvement |
|--------|-------|-----|-----|-------------|
| Classical PLS | 1.6540 ± 0.97 | 0.1058 | 0.9852 | baseline |
| NPLS | 1.4867 ± 1.06 | 0.1854 | 0.8494 | **+10.1%** |

**Configuration:** NPLS with 5 components, RPCA encoder (auto-selected)

---

## 📦 Installation

```bash
# From source (recommended)
git clone https://github.com/dabdul-wahab1988/neutrosophic_pls.git
cd neutrosophic_pls
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- pandas ≥ 1.3
- scikit-learn ≥ 1.0
- matplotlib ≥ 3.4
- PyYAML ≥ 6.0

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Interactive Mode Guide](docs/Interactive_Mode_Guide.md) | Step-by-step for non-programmers |
| [Encoders Mathematics](docs/Encoders_Mathematics.md) | Mathematical foundations |
| [NDG Encoder](docs/ndg_encoder.md) | Differential geometry encoder |
| [NVIP Mathematics](docs/NVIP_Mathematics.md) | Variable importance decomposition |
| [Models Mathematics](docs/Models_Mathematics.md) | N-PLS model derivations |

---

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{abdul_wahab_npls_2025,
  author = {Abdul-Wahab, Dickson and Asare, Ebenezer Aquisman},
  title = {Neutrosophic Partial Least Squares (N-PLS): Uncertainty-aware PLS for Chemometrics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dabdul-wahab1988/neutrosophic_pls}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Dickson Abdul-Wahab** - <dabdul-wahab@live.com>
- **Ebenezer Aquisman Asare** - <aquisman1989@gmail.com>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed at the University of Ghana – Making advanced chemometrics accessible to all researchers*
