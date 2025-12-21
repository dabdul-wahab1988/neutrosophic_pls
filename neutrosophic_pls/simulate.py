"""Synthetic data generation for neutrosophic PLS experiments."""

from typing import Dict, Tuple
import numpy as np


def generate_simulation(
    n_samples: int,
    n_features: int,
    n_components: int,
    noise_config: Dict | None = None,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic T/I/F data with latent factors and configurable noise.
    """
    rng = np.random.default_rng(seed)
    noise_config = noise_config or {}
    noise_t = float(noise_config.get("truth_noise", 0.05))
    noise_i = float(noise_config.get("indeterminacy_noise", 0.1))
    noise_f = float(noise_config.get("falsity_noise", 0.05))

    latent = rng.normal(size=(n_samples, n_components))
    loadings_x = rng.normal(size=(n_components, n_features))
    loadings_y = rng.normal(size=(n_components, 1))

    x_truth = latent @ loadings_x + rng.normal(scale=noise_t, size=(n_samples, n_features))
    y_truth = latent @ loadings_y + rng.normal(scale=noise_t, size=(n_samples, 1))

    # Indeterminacy modeled as variability noise
    x_ind = rng.normal(scale=noise_i, size=(n_samples, n_features))
    y_ind = rng.normal(scale=noise_i, size=(n_samples, 1))

    # Falsity modeled as sparse outlier perturbations
    falsity_mask = rng.random(size=(n_samples, n_features)) < noise_f
    x_falsity = np.zeros_like(x_truth)
    x_falsity[falsity_mask] = rng.normal(scale=noise_f * 5, size=falsity_mask.sum())
    y_falsity = rng.normal(scale=noise_f, size=(n_samples, 1))

    x_tif = np.stack([x_truth, x_ind, x_falsity], axis=-1)
    y_tif = np.stack([y_truth, y_ind, y_falsity], axis=-1)

    metadata = {
        "latent": latent,
        "loadings_x": loadings_x,
        "loadings_y": loadings_y,
        "noise_config": {"truth_noise": noise_t, "indeterminacy_noise": noise_i, "falsity_noise": noise_f},
    }
    return x_tif, y_tif, metadata


def generate_synthetic_spectrum(
    n_samples: int = 100,
    n_features: int = 700,
    n_peaks: int = 5,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic spectroscopic data with Gaussian peaks.
    
    Creates realistic-looking NIR spectra for testing and demonstration
    when real spectroscopic datasets are not available.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of wavelength points (features)
        n_peaks: Number of Gaussian absorption peaks
        noise_level: Standard deviation of additive noise
        seed: Random seed for reproducibility
        
    Returns:
        X: Spectral data array (n_samples, n_features)
        y: Response variable array (n_samples,)
        wavelengths: Wavelength axis array (n_features,)
    """
    rng = np.random.default_rng(seed)
    
    # Wavelengths (typical NIR range)
    wavelengths = np.linspace(1000, 2500, n_features)
    
    # Generate peak parameters
    peak_centers = rng.uniform(1100, 2400, size=n_peaks)
    peak_widths = rng.uniform(20, 100, size=n_peaks)
    
    # Generate concentration matrix (simulating chemical components)
    concentrations = rng.uniform(0.5, 2.0, size=(n_samples, n_peaks))
    
    # Build spectra from Gaussian peaks
    X = np.zeros((n_samples, n_features))
    for k in range(n_peaks):
        peak = np.exp(-((wavelengths - peak_centers[k]) ** 2) / (2 * peak_widths[k] ** 2))
        X += np.outer(concentrations[:, k], peak)
    
    # Add measurement noise
    X += rng.normal(0, noise_level, size=X.shape)
    
    # Response: linear combination of concentrations
    coef = rng.uniform(0.5, 1.5, size=n_peaks)
    y = concentrations @ coef + rng.normal(0, noise_level * 0.5, size=n_samples)
    
    return X, y, wavelengths


def add_spike_corruption(
    X: np.ndarray,
    proportion: float,
    seed: int = 42,
    *,
    magnitude_range: Tuple[float, float] = (15.0, 40.0),
    n_spikes_range: Tuple[int, int] = (5, 11),
    return_indices: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Add spike artifacts to a proportion of samples.
    
    Calibrated to be severe enough to be detected by the encoder
    (z-score > 6 in 2nd derivative space).
    
    Args:
        X: Input spectral data (n_samples, n_features)
        proportion: Fraction of samples to corrupt (0-1)
        seed: Random seed for reproducibility
        magnitude_range: (min, max) multiplier for spike magnitude relative to std
        n_spikes_range: (min, max) number of spikes per corrupted sample
        return_indices: If True, return tuple (corrupted_X, corrupt_indices)
        
    Returns:
        If return_indices=False: Corrupted spectral data
        If return_indices=True: Tuple of (corrupted_X, corrupt_indices)
    """
    rng = np.random.default_rng(seed)
    X_corrupted = X.copy()
    n_samples, n_features = X.shape
    n_corrupt = int(n_samples * proportion)
    
    corrupt_idx = np.array([], dtype=int)
    if n_corrupt > 0:
        corrupt_idx = rng.choice(n_samples, size=n_corrupt, replace=False)
        for idx in corrupt_idx:
            # Add spikes per sample
            n_spikes = rng.integers(n_spikes_range[0], n_spikes_range[1])
            spike_positions = rng.choice(n_features, size=min(n_spikes, n_features), replace=False)
            # Spike magnitudes relative to local std
            spike_magnitudes = rng.uniform(magnitude_range[0], magnitude_range[1], size=len(spike_positions)) * X[idx].std()
            X_corrupted[idx, spike_positions] += spike_magnitudes
    
    if return_indices:
        return X_corrupted, corrupt_idx
    return X_corrupted


def corrupt_training_samples(
    X_train: np.ndarray,
    *,
    pattern: str,
    outlier_fraction: float,
    rng: np.random.Generator,
    informative_features: np.ndarray | None = None,
    outlier_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply DOE corruption to training data (test remains clean).
    
    Patterns:
      - "sample": sample-level corruption (scatter + slope drift + broadband noise)
      - "spikes": localized narrow spikes (high curvature)
      
    Args:
        X_train: Training spectral data
        pattern: Corruption pattern ("sample" or "spikes")
        outlier_fraction: Fraction of samples to corrupt
        rng: NumPy random generator
        informative_features: Optional array of feature indices to target
        outlier_indices: Optional specific sample indices to corrupt
        
    Returns:
        Corrupted training data
    """
    X_train = np.asarray(X_train, dtype=float).copy()
    n_train, n_features = X_train.shape
    if outlier_fraction <= 0.0:
        return X_train

    if outlier_indices is not None:
        outlier_idx = np.asarray(outlier_indices, dtype=int)
        outlier_idx = outlier_idx[(outlier_idx >= 0) & (outlier_idx < n_train)]
        outlier_idx = np.unique(outlier_idx)
        if len(outlier_idx) == 0:
            return X_train
    else:
        n_outliers = int(round(n_train * outlier_fraction))
        if n_outliers <= 0:
            return X_train
        outlier_idx = rng.choice(n_train, size=n_outliers, replace=False)

    overall_sd = float(X_train.std()) + 1e-12
    feat_sd = X_train.std(axis=0) + 1e-12
    x_axis = np.linspace(-1.0, 1.0, n_features)

    if pattern == "baseline":
        pattern = "sample"

    if pattern == "sample":
        for i in outlier_idx:
            scale = 1.0 + float(rng.normal(loc=0.0, scale=0.25))
            slope = float(rng.normal(loc=0.0, scale=1.5 * overall_sd))
            drift = slope * x_axis
            noise = rng.normal(scale=0.8 * overall_sd, size=n_features)
            X_train[i, :] = scale * X_train[i, :] + drift + noise
        return X_train

    if pattern == "spikes":
        if informative_features is None or len(informative_features) == 0:
            candidate_features = np.arange(n_features)
        else:
            candidate_features = np.asarray(informative_features, dtype=int)
            candidate_features = candidate_features[(candidate_features >= 0) & (candidate_features < n_features)]
            if len(candidate_features) == 0:
                candidate_features = np.arange(n_features)

        for i in outlier_idx:
            n_spikes = int(rng.integers(3, 8))
            centers = rng.choice(
                candidate_features,
                size=min(n_spikes, len(candidate_features)),
                replace=False,
            )
            for c in centers:
                half_width = 0
                left = max(0, c - half_width)
                right = min(n_features, c + half_width + 1)
                pos = np.arange(left, right)
                profile = np.ones_like(pos, dtype=float)
                sign = 1.0
                amp = float(rng.uniform(120.0, 260.0))
                X_train[i, pos] += sign * amp * feat_sd[pos] * profile
        return X_train

    raise ValueError(f"Unknown DOE corruption pattern: {pattern}")
