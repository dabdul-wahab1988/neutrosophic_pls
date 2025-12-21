"""Loader and encoding utilities for IDRC 2016 Wheat Protein NIR dataset."""

import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def _default_data_dir() -> Path:
    return Path("data")


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + 1e-8
    return (x - med) / (1.4826 * mad)


def _encode_neutrosophic_regression(
    X: np.ndarray,
    y: np.ndarray,
    snv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode spectral data and continuous target as neutrosophic (T, I, F) tensors.
    
    For spectral (X) data:
    - Truth (T): The measured absorbance values (optionally SNV normalized)
    - Indeterminacy (I): Measurement uncertainty proxy from robust z-score deviation
    - Falsity (F): Outlier flags based on extreme z-scores
    
    For continuous target (y):
    - Truth (T): The measured protein value
    - Indeterminacy (I): Relative uncertainty based on measurement variability
    - Falsity (F): Outlier flag for extreme values
    
    Parameters
    ----------
    X : np.ndarray
        Spectral features (n_samples, n_wavelengths)
    y : np.ndarray
        Continuous target (n_samples,) - protein content in %
    snv : bool
        Apply Standard Normal Variate (SNV) normalization to spectra
        
    Returns
    -------
    x_tif : np.ndarray
        Neutrosophic encoded features (n_samples, n_features, 3)
    y_tif : np.ndarray
        Neutrosophic encoded target (n_samples, 1, 3)
    """
    # --- X encoding ---
    x_truth = X.copy()
    
    # Optional SNV normalization (standard for NIR spectroscopy)
    if snv:
        x_mean = x_truth.mean(axis=1, keepdims=True)
        x_std = x_truth.std(axis=1, keepdims=True) + 1e-8
        x_truth = (x_truth - x_mean) / x_std
    
    # Indeterminacy: robust z-score magnitude (measurement uncertainty)
    # Scale to [0, 1] range for interpretability (same as Y encoding)
    z_x = _robust_z(x_truth)
    x_ind = np.clip(np.abs(z_x) / 3.0, 0, 1)
    
    # Falsity: outlier flags for extreme values
    x_falsity = (np.abs(z_x) > 3.5).astype(float)
    
    # --- Y encoding (continuous regression target) ---
    y_col = y.reshape(-1, 1)
    y_truth = y_col.copy()
    
    # Indeterminacy: based on deviation from robust center
    z_y = _robust_z(y_col)
    # Scale to [0, 1] range for interpretability
    y_ind = np.clip(np.abs(z_y) / 3.0, 0, 1)
    
    # Falsity: extreme outliers in target
    y_falsity = (np.abs(z_y) > 3.5).astype(float)
    
    # Stack into TIF tensors
    x_tif = np.stack([x_truth, x_ind, x_falsity], axis=-1)
    y_tif = np.stack([y_truth, y_ind, y_falsity], axis=-1)
    
    return x_tif, y_tif


def load_idrc_wheat(
    path: str | Path | None = None,
    snv: bool = True,
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> Dict:
    """
    Load IDRC 2016 Wheat Protein NIR dataset and return neutrosophic-encoded tensors.
    
    Dataset: NIR spectroscopy data from IDRC 2016 "Software Shootout"
    - 248 wheat samples
    - 741 wavelengths (730-1100 nm at 0.5 nm resolution)
    - Target: crude protein content (%)
    
    Parameters
    ----------
    path : str or Path, optional
        Path to MA_A2.csv file. Defaults to data/MA_A2.csv
    snv : bool
        Apply Standard Normal Variate normalization (default True)
    wavelength_range : tuple of float, optional
        Select wavelength range, e.g., (800, 1000) for 800-1000nm
        
    Returns
    -------
    dict with keys:
        - x_tif: Neutrosophic encoded features (n_samples, n_features, 3)
        - y_tif: Neutrosophic encoded target (n_samples, 1, 3)
        - metadata: Dataset information
    """
    if path is None:
        path = _default_data_dir() / "MA_A2.csv"
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"IDRC wheat dataset not found at {path}. "
            f"Please download MA_A2.csv from the IDRC 2016 shootout data."
        )
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Extract components
    # Handle both formats: with and without ID column
    if "ID" in df.columns:
        sample_ids = df["ID"].values
        protein = df["Protein"].values.astype(float)
        # Wavelength columns (all columns except ID and Protein)
        wavelength_cols = [c for c in df.columns if c not in ("ID", "Protein")]
    else:
        # No ID column - create sequential IDs
        sample_ids = np.arange(len(df))
        protein = df["Protein"].values.astype(float)
        # Wavelength columns (all columns except Protein)
        wavelength_cols = [c for c in df.columns if c != "Protein"]
    wavelengths = np.array([float(c) for c in wavelength_cols])
    X = df[wavelength_cols].values.astype(float)
    
    # Optional wavelength selection
    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        wavelengths = wavelengths[mask]
        X = X[:, mask]
        wavelength_cols = [c for c in wavelength_cols if wl_min <= float(c) <= wl_max]
    
    # Encode as neutrosophic tensors
    x_tif, y_tif = _encode_neutrosophic_regression(X, protein, snv=snv)
    
    # Compute file hash for reproducibility tracking
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    
    metadata = {
        "path": str(path),
        "md5": md5,
        "dataset_name": "IDRC 2016 Wheat Protein (MA_A2)",
        "task": "regression",
        "feature_names": wavelength_cols,
        "wavelengths": wavelengths.tolist(),
        "target_name": "Protein",
        "target_unit": "%",
        "n_samples": len(df),
        "n_features": len(wavelength_cols),
        "sample_ids": sample_ids.tolist(),
        "protein_range": (float(protein.min()), float(protein.max())),
        "protein_mean": float(protein.mean()),
        "protein_std": float(protein.std()),
        "snv_applied": snv,
        "wavelength_range_nm": (float(wavelengths.min()), float(wavelengths.max())),
        "wavelength_resolution_nm": 0.5,
    }
    
    return {"x_tif": x_tif, "y_tif": y_tif, "metadata": metadata}


def load_idrc_train_test(
    path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    snv: bool = True,
) -> Dict:
    """
    Load IDRC wheat dataset with train/test split.
    
    Parameters
    ----------
    path : str or Path, optional
        Path to MA_A2.csv
    test_size : float
        Fraction of data for testing (default 0.2)
    random_state : int
        Random seed for reproducibility
    snv : bool
        Apply SNV normalization
        
    Returns
    -------
    dict with train/test splits and metadata
    """
    data = load_idrc_wheat(path, snv=snv)
    
    n = data["x_tif"].shape[0]
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    
    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return {
        "x_train": data["x_tif"][train_idx],
        "y_train": data["y_tif"][train_idx],
        "x_test": data["x_tif"][test_idx],
        "y_test": data["y_tif"][test_idx],
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "metadata": data["metadata"],
    }
