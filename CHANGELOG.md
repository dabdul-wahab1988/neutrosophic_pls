# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-21

### Added

- **NDG Manifold Encoder** (`encode_ndg_manifold`)
  - Physics-based encoder derived from Neutrosophic Differential Geometry framework
  - Maps spectra to manifold M_N with principled T/I/F channels
  - Truth from normalized signal, Indeterminacy from Shannon entropy of local variance
  - Falsity from systematic error coefficient (low-rank model deviation)
  - Supports replicate scan variance when available
  - Configurable normalization ('none', 'snv', 'minmax')

### Changed

- **Clean Data Bypass for NPLS Models**
  - NPLS, NPLSW, PNPLS now gracefully degrade to sklearn PLSRegression on clean data
  - Raised I/F threshold from 0.01 to 0.15 for bypass activation
  - Added weight uniformity check (CV < 5% triggers bypass)
  - New `_use_sklearn` and `_bypass_reason` attributes for diagnostics
  - Ensures identical performance to Classical PLS when data is clean

- **Removed SNV from Interactive Workflow**
  - Pre-encoding SNV normalization removed from Step 2
  - Encoders now handle normalization internally
  - Avoids double-normalization issues with NDG and other encoders

- **Performance Summary Figure**
  - Removed misleading "Mild (5%)" and "Severe (30%)" simulated corruption bars
  - Now shows only actual data comparison between PLS and NPLS variants
  - Clearer interpretation without confusing simulated scenarios

### Fixed

- NPLS variants now match Classical PLS exactly on clean data (was ~5% worse)
- NDG encoder default normalization changed to 'none' to preserve concentration signal

## [1.0.0] - 2025-12-01

### Added

- **Core N-PLS Models**
  - `NPLS`: Basic Neutrosophic Partial Least Squares
  - `NPLSW`: Reliability-Weighted N-PLS with indeterminacy-based sample weighting
  - `PNPLS`: Probabilistic N-PLS with element-wise variance model

- **Neutrosophic Encoding**
  - Truth (T): observed measurements
  - Indeterminacy (I): uncertainty proxy from robust z-scores
  - Falsity (F): binary outlier flags

- **Variable Importance in Projection (VIP)**
  - Channel-decomposed VIP (T, I, F contributions)
  - Aggregate VIP with proper normalization
  - Visualization functions for VIP profiles

- **Universal Data Loader**
  - Auto-detection of file formats (CSV, Excel, ARFF, JSON, Parquet)
  - Configurable neutrosophic encoding
  - SNV normalization for spectral data
  - Interactive dataset selection

- **Command Line Interface**
  - Interactive mode (`--interactive`)
  - Configuration file mode (`--config study.yaml`)
  - Direct mode (`--data file.csv --target y`)
  - Preset configurations (`--preset idrc_wheat`)

- **Study Configuration System**
  - YAML/JSON configuration files
  - Dataclass-based settings
  - Reproducible study definitions

- **Simulation Study Framework**
  - Stage 1: Screening (fractional factorial design)
  - Stage 2: Response Surface (3-level factorial)
  - Stage 3: Confirmatory (IDRC wheat protein data)

- **Datasets**
  - IDRC 2016 Wheat Protein NIR data loader
  - MicroMass (OpenML 1514) loader
  - Universal loader for custom datasets

- **Visualization**
  - Performance heatmaps
  - VIP bar charts with channel decomposition
  - Response surface plots
  - Method comparison charts

### Documentation

- Comprehensive README with examples
- Example configuration files
- API documentation in docstrings

## [0.1.0] - 2025-11-01 (Development)

### Added

- Initial package structure
- Basic NPLS implementation
- Prototype VIP computation
