# Neutrosophic Differential Geometry (NDG) Encoder

## Theoretical Foundation

The NDG encoder implements a principled mapping from spectroscopic data to a neutrosophic manifold $\mathcal{M}_\mathcal{N}$, based on differential geometry concepts.

### Phase 1: The Topological Foundation

The manifold mapping $\phi: \mathbb{R}^p \rightarrow \mathcal{M}_\mathcal{N}$ is defined component-wise:

$$\phi(x)_k = (T_k, I_k, F_k)$$

where:

- **$T_k$ (Truth)**: Normalized signal strength via $\mathcal{N}(x_k)$
- **$I_k$ (Indeterminacy)**: Shannon entropy transform of local variance $\mathcal{H}(\sigma_k^2)$
- **$F_k$ (Falsity)**: Systematic error coefficient $1 - \mathcal{N}(x_k) \cdot (1 - \epsilon_k)$

### Phase 2: The Metric Architecture

The encoding implicitly defines a **Neutrosophic Metric Tensor**:

$$g_{ij}^N = \alpha (g_{ij})_T - \beta (g_{ij})_I - \gamma (g_{ij})_F$$

where:

- $(g_{ij})_T$ comes from **Fisher Information** (signal curvature)
- $(g_{ij})_I$ comes from **Shannon Entropy** (noise uncertainty)
- $(g_{ij})_F$ comes from **Systematic Error Covariance** $(\Sigma_{bias}^{-1})$

### Phase 3: Connection & Transport

The framework supports **Calibration Transfer** via parallel transport:

$$\nabla_{\dot{\gamma}}^N b = 0$$

The corrected calibration vector:

$$b_{new}^k = b_{old}^k - \int_P^Q \Gamma_{ij}^k b^j dx^i$$

### Phase 4: Curvature Analysis

The **Ricci Scalar** $R^I$ quantifies spectral complexity:

- Low $R^I$: Geometrically stable (clean data)
- High $R^I$: Complex manifold (noisy/structured data)

## Implementation

```python
from neutrosophic_pls.encoders import encode_ndg_manifold

# Basic usage
result = encode_ndg_manifold(X)
x_tif = np.stack([result.truth, result.indeterminacy, result.falsity], axis=-1)

# With replicate scans for true variance
result = encode_ndg_manifold(
    X,
    replicate_scans=replicate_data,  # shape: (n_samples, n_replicates, n_features)
    normalization="none",  # Preserve concentration signal
)

# Access geometric metadata
print(f"Complexity score (≈Ricci scalar): {result.metadata['complexity_score']}")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalization` | `"none"` | `"none"`, `"snv"`, or `"minmax"` |
| `local_window` | 5 | Window for local variance estimation |
| `low_rank_components` | 5 | Components for systematic error model |
| `entropy_scale` | 1.5 | Sensitivity of entropy-based I |
| `beta_I`, `beta_F` | 2.0 | Power calibration for I/F |
| `replicate_scans` | None | Actual replicate variance data |

## Geometric Interpretation

The NDG framework provides:

1. **Distance penalization**: High I/F "stretch" distances, making samples harder to distinguish
2. **Geodesics**: Paths of maximum information retention as conditions change
3. **Curvature**: Quantifies intrinsic non-linearity requiring neutrosophic approach
