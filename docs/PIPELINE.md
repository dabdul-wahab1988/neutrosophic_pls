# Pipeline Usage

## Overview

The pipeline provides a simplified interface for running neutrosophic PLS experiments via CLI or notebook. It supports:
- Simulated data generation
- MicroMass dataset loading
- Multiple N-PLS model variants
- Automated metrics, VIP analysis, and semantic projections

## CLI Usage

### Step 1: Generate Configuration with Wizard

The wizard command creates a YAML configuration file with all pipeline parameters:

```bash
python -m neutrosophic_pls.cli wizard --output pipeline_config.yaml \
  --mode simulate \
  --n-samples 50 \
  --n-features 20 \
  --n-components 2 \
  --model-type npls \
  --seed 0
```

**Available wizard options:**
- `--output`: Path for generated config file (default: `pipeline_config.yaml`)
- `--mode`: Data mode - `simulate` or `micromass` (default: `simulate`)
- `--n-samples`: Number of samples for simulation (default: 50)
- `--n-features`: Number of features for simulation (default: 20)
- `--n-components`: Number of latent components (default: 2)
- `--model-type`: Model variant - `npls`, `nplsw`, or `pnpls` (default: `npls`)
- `--lambda-indeterminacy`: Indeterminacy weighting (default: 1.0)
- `--lambda-falsity`: Falsity weighting (default: 0.0)
- `--alpha`: PNPLS falsity prior softness parameter (default: 1.0)
- `--weight-normalize`: Sample weight normalization - `none`, `mean1`, or `sum1` (default: `mean1`)
- `--seed`: Random seed for reproducibility (default: 0)

### Step 2: Run Pipeline

Execute the pipeline using the generated configuration:

```bash
python -m neutrosophic_pls.cli run-pipeline --config pipeline_config.yaml --output-dir artifacts
```

**Options:**
- `--config`: Path to YAML configuration file (optional, defaults to PipelineConfig defaults if not provided)
- `--output-dir`: Directory for output files (default: `artifacts`)

### Example: MicroMass Dataset

To run on the MicroMass fixture:

1. Create a configuration file manually or via wizard with `--mode micromass`
2. Optionally set `micromass_path` in the config (defaults to packaged fixture)
3. Run the pipeline:

```bash
python -m neutrosophic_pls.cli run-pipeline --config micromass_config.yaml --output-dir artifacts_micromass
```

## Model Types

### NPLS (Standard Neutrosophic PLS)
Basic N-PLS with channel weighting and sample-level reliability weighting.

**Key Parameters:**
- `n_components`: Number of latent components
- `channel_weights`: Tuple of (T, I, F) weights (default: `(1.0, 1.0, 1.0)`)
- `lambda_falsity`: Strength of sample-level falsity downweighting (default: 0.5)

**Usage in config:**
```yaml
model_type: npls
n_components: 2
channel_weights: [1.0, 1.0, 1.0]
lambda_falsity: 0.5
```

### NPLSW (Reliability-Weighted Variant)
Sample-weighted N-PLS with configurable weight normalization.

**Key Parameters:**
- All NPLS parameters, plus:
- `lambda_indeterminacy`: Indeterminacy weighting parameter (default: 1.0)
- `weight_normalize`: Weight normalization - `none`, `mean1`, or `sum1` (default: `mean1`)
- `alpha`: Exponent for soft falsity weighting (default: 1.0)

**Usage in config:**
```yaml
model_type: nplsw
n_components: 2
lambda_indeterminacy: 1.0
lambda_falsity: 0.5
alpha: 1.0
weight_normalize: mean1
channel_weights: [1.0, 1.0, 1.0]
```

### PNPLS (Probabilistic Variant)
Element-wise variance weighting derived from falsity channel.

**Key Parameters:**
- `n_components`: Number of latent components
- `lambda_indeterminacy`: Indeterminacy parameter (default: 1.0)
- `lambda_falsity`: Falsity prior strength (default: 0.5)
- `alpha`: Softness parameter for falsity prior (default: 1.0)

**Usage in config:**
```yaml
model_type: pnpls
n_components: 2
lambda_indeterminacy: 1.0
lambda_falsity: 0.5
alpha: 1.0
```

## Notebook Helper

For interactive notebooks, use the simplified API:

```python
from neutrosophic_pls.notebook import run_pipeline

# Using dictionary configuration
result = run_pipeline({
    "mode": "simulate",
    "n_samples": 50,
    "n_features": 20,
    "n_components": 2,
    "model_type": "npls",
    "output_dir": "artifacts_nb",
    "seed": 42
})

# Or using YAML config file
result = run_pipeline("path/to/config.yaml")
```

**Returns a dictionary containing:**
- `model`: Fitted model object
- `metrics`: Dictionary of evaluation metrics (RMSE, R², MAE, etc.)
- `vips`: Dictionary of VIP scores for each channel (`T`, `I`, `F`, `aggregate`)
- `semantics`: List of top features with interpretations
- `metadata`: Dataset metadata including feature names and generation parameters
- `output_dir`: Path to output directory

## Configuration File Structure

Complete PipelineConfig fields (YAML format):

```yaml
# Data source
mode: simulate              # 'simulate' or 'micromass'
micromass_path: null        # Optional path to MicroMass data file

# Simulation parameters (for mode='simulate')
n_samples: 50
n_features: 20
seed: 0

# Model configuration
model_type: npls            # 'npls', 'nplsw', or 'pnpls'
n_components: 2

# Model hyperparameters
lambda_indeterminacy: 1.0   # For nplsw/pnpls
lambda_falsity: 0.0         # For all models
alpha: 1.0                  # For pnpls and nplsw
weight_normalize: mean1     # For nplsw: 'none', 'mean1', or 'sum1'
channel_weights:            # (T, I, F) weights for all models
  - 1.0
  - 1.0
  - 1.0

# Output
output_dir: artifacts
```

## Outputs

### File-based Outputs
The pipeline writes the following files to `output_dir`:

1. **`metrics.txt`**: Evaluation metrics including RMSE, R², MAE
   ```
   rmse: 0.234
   r2: 0.891
   mae: 0.187
   ```

2. **`vip.txt`**: Variable Importance in Projection scores
   ```
   T: [1.2, 0.9, 1.5, ...]
   I: [0.3, 0.2, 0.4, ...]
   F: [0.1, 0.0, 0.2, ...]
   aggregate: [1.6, 1.1, 2.1, ...]
   ```

### In-Memory Returns
The `run()` function and notebook helper return a dictionary with:

- **`model`**: Fitted model object (NPLS, NPLSW, or PNPLS instance)
- **`metrics`**: Dict of evaluation metrics
- **`vips`**: Dict with channel-specific and aggregate VIP arrays
- **`semantics`**: List of dicts with top-k feature interpretations
  ```python
  [
    {
      "feature": "f0",
      "vip": 1.234,
      "vip_T": 1.1,
      "vip_I": 0.3,
      "vip_F": 0.1,
      "interpretation": "f0 is influential across T/I/F with VIP=1.234."
    },
    ...
  ]
  ```
- **`metadata`**: Dataset information (feature names, generation seeds, checksums)
- **`output_dir`**: Path to output directory

## Notes

- **MicroMass data**: The loader uses a packaged fixture by default. To use full MicroMass data, set `micromass_path` in the config to point to your dataset.
- **CI testing**: Set `SKIP_MICROMASS=1` environment variable to avoid network downloads in CI. The packaged fixture ensures deterministic tests.
- **Channel weights**: The `channel_weights` tuple weights the T/I/F channels when computing combined metrics and VIPs. Default `(1.0, 1.0, 1.0)` treats all channels equally.
- **Reproducibility**: Set the `seed` parameter for deterministic simulation data generation.
