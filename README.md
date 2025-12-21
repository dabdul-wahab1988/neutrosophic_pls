# Neutrosophic Partial Least Squares (N-PLS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Uncertainty-aware Partial Least Squares using Neutrosophic (Truth/Indeterminacy/Falsity) encoding**

A Python package for chemometrics and spectroscopy that extends classical PLS regression by incorporating measurement uncertainty through neutrosophic set theory.

## Authors

- **Dickson Abdul-Wahab** - University of Ghana  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7446--5909-green.svg)](https://orcid.org/0000-0001-7446-5909)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-dickson--abdul--wahab-blue.svg)](https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate-Dickson--Abdul--Wahab-00CCBB.svg)](https://www.researchgate.net/profile/Dickson-Abdul-Wahab)

- **Ebenezer Aquisman Asare**  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--1185--1479-green.svg)](https://orcid.org/0000-0003-1185-1479)

## Features

- **Neutrosophic Encoding**: Represents data as (Truth, Indeterminacy, Falsity) tensors
- **Multiple N-PLS Variants**:
  - `NPLS`: Basic Neutrosophic PLS
  - `NPLSW`: Reliability-Weighted N-PLS (downweights uncertain samples)
  - `PNPLS`: Probabilistic Neutrosophic PLS (element-wise variance model; collapses to classical PLS when clean)
- **Variable Importance in Projection (VIP)**: Channel-decomposed VIP showing T/I/F contributions
- **Universal Data Loader**: Supports CSV, Excel, ARFF, JSON, Parquet with auto-detection
- **Interactive CLI**: Guided setup for non-programmers
- **YAML/JSON Configuration**: Reproducible study configurations

## Installation

```bash
# From PyPI (when published)
pip install neutrosophic-pls

# From source
git clone https://github.com/abdulwahab.dickson/neutrosophic-pls.git
cd neutrosophic-pls
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

### As a Python Library

```python
from neutrosophic_pls import NPLS, NPLSW, load_dataset, DatasetConfig

# Load and encode your data
config = DatasetConfig(
    path="data/spectra.csv",
    target="concentration",
    task="regression",
    snv=True,  # Standard Normal Variate for spectral data
    exclude_columns=["ID"],
)
data = load_dataset(config)

# Fit N-PLS model
model = NPLSW(n_components=10, channel_weights=(1.0, 1.0, 1.0))
model.fit(data["x_tif"], data["y_tif"])

# Predict
predictions = model.predict(data["x_tif"])

# Compute Variable Importance
from neutrosophic_pls import compute_nvip
vip = compute_nvip(model, data["x_tif"])
print(f"VIP (Aggregate): {vip['aggregate']}")
print(f"VIP (Truth): {vip['T']}")
print(f"VIP (Indeterminacy): {vip['I']}")
print(f"VIP (Falsity): {vip['F']}")
```

### Command Line Interface

```bash
# Interactive mode (guided setup)
python -m neutrosophic_pls --interactive

# Run from configuration file
python -m neutrosophic_pls --config study.yaml

# Direct mode
python -m neutrosophic_pls --data data.csv --target y --snv --method all

# Calibration utilities
python scripts/calibrate_entropy.py --dataset data.csv --target y --output results_calibration
python scripts/validation_study.py --dataset data.csv --target y --output results_validation

# Quick run with preset (IDRC wheat protein data)
python -m neutrosophic_pls --preset idrc_wheat --quick

# List available datasets
python -m neutrosophic_pls --list-data
```

### Configuration File (YAML)

```yaml
name: My N-PLS Study
description: Protein prediction from NIR spectra

dataset:
  path: data/spectra.csv
  target: Protein
  task: regression
  exclude_columns: [ID, SampleName]
  snv: true

model:
  method: all  # PLS, NPLS, NPLSW, or all
  max_components: 15
  channel_weights: [1.0, 1.0, 1.0]

evaluation:
  cv_folds: 5
  repeats: 3
  compute_vip: true

output:
  output_dir: results/my_study
  generate_figures: true
```

## Neutrosophic Encoding

The package encodes each measurement as a neutrosophic triplet (T, I, F):

| Channel | Meaning | Computation |
|---------|---------|-------------|
| **T (Truth)** | The observed measurement | Raw or SNV-normalized values |
| **I (Indeterminacy)** | Measurement uncertainty | Robust z-score magnitude (MAD-scaled) |
| **F (Falsity)** | Unreliability flag | Binary outlier indicator (\|z\| > 3.5) |

This allows N-PLS to:

- Downweight uncertain measurements
- Identify and handle outliers
- Provide channel-decomposed variable importance

### Encoder options & auto-selection

- `probabilistic` (default): low-rank SVD + residual mixture.
- `rpca`: low-rank Truth + sparse falsity (Robust PCA).
- `wavelet`: multi-scale smooth vs spike separation (requires `pywavelets`).
- `quantile`: non-parametric quantile envelope with inner/outer bands.
- `augment`: stability under light augmentations drives I/F.
- `auto`: cross-validates candidate encoders (via lightweight NPLSW) and selects the best performer; configure with `EncoderConfig` or `encoding={"name": "auto", ...}`.

## Methods

### NPLS (Neutrosophic PLS)

Basic N-PLS that operates on the combined T/I/F tensor with configurable channel weights.

### NPLSW (Reliability-Weighted N-PLS)

Extends NPLS by computing sample-wise reliability weights from the Indeterminacy/Falsity channels, downweighting uncertain observations during model fitting.

### PNPLS (Probabilistic Neutrosophic PLS)

Uses a probabilistic element-wise variance model derived from the Falsity channel; when I/F≈0 it collapses to classical PLS, and when I/F grow it softly downweights corrupted cells for robustness.

## Simulation Study

The package includes a complete simulation study framework:

```bash
# Run full study (Stages 1-3)
python scripts/run_study.py --full

# Quick demo
python scripts/run_study.py --quick

# Individual stages
python scripts/run_study.py --stage 1  # Screening
python scripts/run_study.py --stage 2  # Response Surface
python scripts/run_study.py --stage 3  # IDRC Wheat Confirmatory
```

### Study Stages

1. **Stage 1: Screening** - Fractional factorial design to identify dominant factors
2. **Stage 2: Response Surface** - 3-level factorial for detailed method comparison
3. **Stage 3: Confirmatory** - Real-world validation on IDRC 2016 wheat protein NIR data

## Example Results (IDRC Wheat Protein)

| Method | RMSEP (%) | R² | Improvement |
|--------|-----------|-----|-------------|
| PLS | 7.04 | -72.6 | baseline |
| NPLS | 2.06 | -0.16 | **70.8%** |
| NPLSW | 1.92 | 0.08 | **72.7%** |

## API Reference

### Core Classes

```python
# Models
NPLS(n_components, channel_weights=(1,1,1))
NPLSW(n_components, channel_weights=(1,1,1), lambda_indeterminacy=1.0)
PNPLS(n_components, lambda_falsity=0.5)

# Data Loading
load_dataset(config: DatasetConfig) -> dict
load_idrc_wheat(snv=True) -> dict
load_micromass() -> dict

# Analysis
compute_nvip(model, X, channel_weights) -> dict
```

### Configuration Classes

```python
DatasetConfig(path, target, task, snv, exclude_columns, ...)
StudyConfig(dataset, model, evaluation, output)
```

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- pandas ≥ 1.3
- scikit-learn ≥ 1.0
- matplotlib ≥ 3.4
- PyYAML ≥ 6.0

## Citation

If you use this package in your research, please cite:

```bibtex
@software{abdul_wahab_npls_2025,
  author = {Abdul-Wahab, Dickson and Asare, Ebenezer Aquisman},
  title = {Neutrosophic Partial Least Squares (N-PLS): Uncertainty-aware PLS for Chemometrics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/abdulwahab.dickson/neutrosophic-pls}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

- **Dickson Abdul-Wahab** - <dabdul-wahab@live.com>
- **Ebenezer Aquisman Asare** - <aquisman1989@gmail.com>

---

*Developed at the University of Ghana*
