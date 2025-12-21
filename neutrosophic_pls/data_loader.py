"""
Universal Data Loader for Neutrosophic PLS
==========================================

Provides flexible data loading with:
- Auto-detection of file formats (CSV, ARFF, Excel, etc.)
- Configurable neutrosophic encoding for different data types
- Support for both classification and regression tasks
- Interactive dataset selection

Author: NeutroProject
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .encoders import (
    EncoderConfig,
    auto_select_encoder,
    dispatch_encoder,
)

# =============================================================================
# Encoding Functions
# =============================================================================

def _robust_z(x: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + 1e-8
    return (x - med) / (1.4826 * mad)


def _snv_normalize(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate normalization for spectral data."""
    x_mean = X.mean(axis=1, keepdims=True)
    x_std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - x_mean) / x_std


def _limits_uncertainty(
    X: np.ndarray,
    lower: Optional[Union[float, np.ndarray]],
    upper: Optional[Union[float, np.ndarray]],
    band_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build indeterminacy/falsity from detector limits.

    - Falsity: 1 if value is outside [lower, upper], else 0.
    - Indeterminacy: ramps from 0 to 1 within a band near each limit.

    Parameters
    ----------
    X : np.ndarray
        Data array (n_samples, n_features)
    lower, upper : float or array or None
        Detector limits per feature (broadcastable). Use None for no bound.
    band_fraction : float
        Fraction of the valid range used for the ramp (default 5%).
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    if lower is None:
        lower_arr = np.full(n_features, -np.inf)
    else:
        lower_arr = np.asarray(lower, dtype=float).reshape(-1)
        if lower_arr.size == 1:
            lower_arr = np.full(n_features, lower_arr.item())

    if upper is None:
        upper_arr = np.full(n_features, np.inf)
    else:
        upper_arr = np.asarray(upper, dtype=float).reshape(-1)
        if upper_arr.size == 1:
            upper_arr = np.full(n_features, upper_arr.item())

    # If no finite bounds are provided, return zeros to avoid invalid operations
    finite_mask = np.isfinite(lower_arr) & np.isfinite(upper_arr) & (upper_arr > lower_arr)
    ind = np.zeros_like(X, dtype=float)
    falsity = np.zeros_like(X, dtype=float)
    if not np.any(finite_mask):
        return ind, falsity

    rng = upper_arr - lower_arr
    rng = np.where(rng > 0, rng, 1.0)
    band = band_fraction * rng

    # Compute only for finite positions
    lower_f = lower_arr.copy()
    upper_f = upper_arr.copy()
    band_f = band.copy()

    below = np.zeros_like(X, dtype=float)
    above = np.zeros_like(X, dtype=float)
    below[:, finite_mask] = np.clip(
        (lower_f[finite_mask] + band_f[finite_mask] - X[:, finite_mask]) / (band_f[finite_mask] + 1e-12),
        0,
        1,
    )
    above[:, finite_mask] = np.clip(
        (X[:, finite_mask] - (upper_f[finite_mask] - band_f[finite_mask])) / (band_f[finite_mask] + 1e-12),
        0,
        1,
    )
    ind = np.clip(below + above, 0, 1)

    falsity[:, finite_mask] = ((X[:, finite_mask] < lower_f[finite_mask]) | (X[:, finite_mask] > upper_f[finite_mask])).astype(float)
    return ind, falsity


def _entropy_surprisal(x: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample surprisal (-log p) for each feature via histogram density.
    
    Returns
    -------
    surprisal : np.ndarray
        Shape (n_samples, n_features), larger = rarer observation.
    probs : np.ndarray
        Bin probabilities per feature, shape (bins, n_features) for debugging/analysis.
    """
    n_samples, n_features = x.shape
    surprisal = np.zeros_like(x, dtype=float)
    probs_store = np.zeros((bins, n_features), dtype=float)
    for j in range(n_features):
        vals = x[:, j]
        hist, edges = np.histogram(vals, bins=bins)
        p = hist / (hist.sum() + 1e-12)
        probs_store[:, j] = p
        bin_idx = np.digitize(vals, edges[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, bins - 1)
        p_vals = p[bin_idx]
        surprisal[:, j] = -np.log(p_vals + 1e-12)
    return surprisal, probs_store


def _normalize_entropy_scores(surprisal: np.ndarray, high_tail: float = 95.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize surprisal to [0, 1] for Indeterminacy and derive Falsity flags from upper tail.
    """
    p5 = np.percentile(surprisal, 5, axis=0)
    p95 = np.percentile(surprisal, high_tail, axis=0)
    denom = p95 - p5 + 1e-12
    ind = np.clip((surprisal - p5) / denom, 0.0, 1.0)
    falsity = (surprisal > p95).astype(float)
    return ind, falsity


def _compute_soft_falsity(x: np.ndarray, threshold_hard: float = 3.5, threshold_soft: float = 2.0) -> np.ndarray:
    """
    Compute gradual falsity instead of binary outlier flags using robust z-scores.

    Args:
        x: Data (n_samples, n_features)
        threshold_hard: z-score where F=1.0 (e.g., 3.5)
        threshold_soft: z-score where F=0.5 (e.g., 2.0)

    Returns:
        F: Soft falsity [0,1] with smooth gradient
    """
    z = np.abs(_robust_z(x))
    F = np.zeros_like(z)

    # Values between soft and hard thresholds ramp up
    mask_soft = (z >= threshold_soft) & (z < threshold_hard)
    F[mask_soft] = 0.5 * (z[mask_soft] - threshold_soft) / (threshold_hard - threshold_soft + 1e-12)

    # Values at or above hard threshold are fully false
    F[z >= threshold_hard] = 1.0

    return np.clip(F, 0, 1)


def _indeterminacy_from_robust_z(x: np.ndarray, z_max: float = 3.0) -> np.ndarray:
    """
    Map robust |z|-scores to [0, 1] indeterminacy.

    z = 0 -> 0 indeterminacy, |z| >= z_max -> ~1 (saturates).
    """
    z = np.abs(_robust_z(x))
    ind = z / (z_max + 1e-12)
    return np.clip(ind, 0.0, 1.0)


def encode_neutrosophic(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["regression", "classification"] = "regression",
    snv: bool = False,
    lower_limits: Optional[Union[float, np.ndarray]] = None,
    upper_limits: Optional[Union[float, np.ndarray]] = None,
    y_lower_limits: Optional[Union[float, np.ndarray]] = None,
    y_upper_limits: Optional[Union[float, np.ndarray]] = None,
    limit_band_fraction: float = 0.05,
    encoding: Union[str, EncoderConfig, Dict[str, Any]] = "default",
    spectral_noise_db: float = -20.0,
    return_metadata: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Encode features and target as neutrosophic (T, I, F) tensors.
    
    The encoding follows the neutrosophic interpretation:
    - Truth (T): The measured/observed values
    - Indeterminacy (I): Uncertainty indicator (higher = less certain)
    - Falsity (F): Outlier/unreliability flags
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples, n_features)
    y : np.ndarray
        Target (n_samples,) or (n_samples, 1)
    task : str
        "regression" or "classification"
    snv : bool
        Apply SNV normalization to features (recommended for spectral data).
    lower_limits, upper_limits : float or array, optional
        Detector limits for features. If provided, indeterminacy/falsity are
        derived from proximity/violation of these bounds.
    y_lower_limits, y_upper_limits : float or array, optional
        Detector limits for targets (optional).
    limit_band_fraction : float
        Fraction of the valid range used to ramp indeterminacy near limits.
    encoding : str or dict or EncoderConfig
        Encoder specification. Use names ("probabilistic", "rpca", "wavelet",
        "quantile", "augment", "spectroscopy") or a dict/EncoderConfig with
        parameters. Set {"name": "auto", "candidates": [...]} to enable
        auto-selection.
    spectral_noise_db : float
        Noise floor in dB for spectroscopy encoders (default -20.0).
    return_metadata : bool
        When True, also return encoder metadata (e.g., auto-selection scores).
        
    Returns
    -------
    x_tif : np.ndarray
        Neutrosophic encoded features (n_samples, n_features, 3)
    y_tif : np.ndarray
        Neutrosophic encoded target (n_samples, 1, 3)
    metadata : dict (optional)
        Returned when return_metadata=True, contains encoder/class map info.
    """
    # --- Feature preprocessing ---
    x_truth = X.copy().astype(float)

    # Optional SNV normalization (standard for spectroscopy)
    if snv:
        x_truth = _snv_normalize(x_truth)

    encoder_cfg = EncoderConfig.from_value(encoding)

    # --- Target preprocessing (needed for auto-selection) ---
    y_col = np.asarray(y).reshape(-1, 1)
    class_map: Optional[Dict[Any, int]] = None
    if task == "classification":
        if y_col.dtype == object or not np.issubdtype(y_col.dtype, np.number):
            unique_classes = np.unique(y_col)
            class_map = {c: i for i, c in enumerate(unique_classes)}
            y_col = np.array([[class_map[v[0]]] for v in y_col], dtype=float)
        else:
            y_col = y_col.astype(float)
    else:
        y_col = y_col.astype(float)

    # --- Feature encoding ---
    encoder_meta: Dict[str, Any] = {}
    if lower_limits is not None or upper_limits is not None:
        x_ind, x_falsity = _limits_uncertainty(
            x_truth, lower_limits, upper_limits, band_fraction=limit_band_fraction
        )
        x_truth_channel = x_truth
        encoder_meta["name"] = "limits"
    elif encoder_cfg.name == "auto":
        auto_result, auto_info = auto_select_encoder(
            x_truth,
            y_col.squeeze(),
            encoder_cfg,
            task=task,
        )
        x_truth_channel = auto_result.truth
        x_ind = auto_result.indeterminacy
        x_falsity = auto_result.falsity
        encoder_meta.update({
            "name": auto_info.best_config.name,
            "auto_scores": auto_info.scores,
            "auto_metric": auto_info.metric,
            "auto_selected": True,
        })
    else:
        params = dict(encoder_cfg.params or {})
        if encoder_cfg.name == "spectroscopy":
            params.setdefault("spectral_noise_db", spectral_noise_db)
        selected_cfg = EncoderConfig(name=encoder_cfg.name, params=params)
        result = dispatch_encoder(x_truth, selected_cfg)
        x_truth_channel = result.truth
        x_ind = result.indeterminacy
        x_falsity = result.falsity
        encoder_meta["name"] = selected_cfg.name
        if result.metadata:
            encoder_meta.update(result.metadata)

    # --- Target encoding ---
    y_truth = y_col.copy()

    if y_lower_limits is not None or y_upper_limits is not None:
        y_ind, y_falsity = _limits_uncertainty(
            y_col, y_lower_limits, y_upper_limits, band_fraction=limit_band_fraction
        )
    else:
        y_ind = np.zeros_like(y_truth)
        y_falsity = np.zeros_like(y_truth)

    # Stack into TIF tensors
    x_tif = np.stack([x_truth_channel, x_ind, x_falsity], axis=-1)
    y_tif = np.stack([y_truth, y_ind, y_falsity], axis=-1)

    metadata: Dict[str, Any] = {"encoder": encoder_meta}
    if class_map is not None:
        metadata["class_map"] = class_map

    if return_metadata:
        return x_tif, y_tif, metadata
    return x_tif, y_tif


# =============================================================================
# File Format Detection and Loading
# =============================================================================

def detect_format(path: Path) -> str:
    """Detect file format from extension."""
    suffix = path.suffix.lower()
    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".arff": "arff",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".parquet": "parquet",
        ".pkl": "pickle",
        ".pickle": "pickle",
    }
    return format_map.get(suffix, "csv")


def load_dataframe(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load data from various file formats into a DataFrame.
    
    Parameters
    ----------
    path : str or Path
        Path to data file
    format : str, optional
        File format. Auto-detected if not specified.
    **kwargs
        Additional arguments passed to the reader function
        
    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if format is None:
        format = detect_format(path)
    
    if format == "csv":
        return pd.read_csv(path, **kwargs)
    elif format == "tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    elif format == "excel":
        return pd.read_excel(path, **kwargs)
    elif format == "json":
        return pd.read_json(path, **kwargs)
    elif format == "parquet":
        return pd.read_parquet(path, **kwargs)
    elif format == "pickle":
        return pd.read_pickle(path, **kwargs)
    elif format == "arff":
        try:
            from scipy.io import arff
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            # Decode byte strings (common in ARFF)
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode("utf-8")
                    except AttributeError:
                        pass
            return df
        except ImportError:
            raise ImportError("scipy is required to read ARFF files")
    else:
        raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Universal Dataset Loader
# =============================================================================

class DatasetConfig:
    """Configuration for loading and encoding a dataset."""
    
    def __init__(
        self,
        path: Union[str, Path],
        target: str,
        task: Literal["regression", "classification"] = "regression",
        features: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        snv: bool = False,
        lower_limits: Optional[Union[float, np.ndarray]] = None,
        upper_limits: Optional[Union[float, np.ndarray]] = None,
        y_lower_limits: Optional[Union[float, np.ndarray]] = None,
        y_upper_limits: Optional[Union[float, np.ndarray]] = None,
        limit_band_fraction: float = 0.05,
        encoding: Union[str, EncoderConfig, Dict[str, Any]] = "default",
        spectral_noise_db: float = -20.0,
        format: Optional[str] = None,
        name: Optional[str] = None,
        **load_kwargs,
    ):
        """
        Parameters
        ----------
        path : str or Path
            Path to data file
        target : str
            Name of target column
        task : str
            "regression" or "classification"
        features : list of str, optional
            Feature column names. If None, uses all non-target columns.
        exclude_columns : list of str, optional
            Columns to exclude from features (e.g., ID columns)
        snv : bool
            Apply SNV normalization (for spectral data).
        lower_limits, upper_limits : float or array, optional
            Detector limits.
        y_lower_limits, y_upper_limits : float or array, optional
            Detector limits for targets.
        limit_band_fraction : float
            Fraction of the valid range used to ramp indeterminacy near limits.
        encoding : str or dict or EncoderConfig
            Encoding method: "default", "spectroscopy", "rpca", "wavelet",
            "quantile", "augment", or {"name": "auto", ...} for auto-selection.
        spectral_noise_db : float
            Noise floor in dB for spectroscopy encoders (default -20.0).
        format : str, optional
            File format (auto-detected if None)
        name : str, optional
            Dataset name (used in reporting)
        **load_kwargs
            Additional arguments for file loading
        """
        self.path = Path(path)
        self.target = target
        self.task = task
        self.features = features
        self.exclude_columns = exclude_columns or []
        self.snv = snv
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.y_lower_limits = y_lower_limits
        self.y_upper_limits = y_upper_limits
        self.limit_band_fraction = limit_band_fraction
        self.encoding = encoding
        self.spectral_noise_db = spectral_noise_db
        self.format = format
        self.name = name or self.path.stem
        self.load_kwargs = load_kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "target": self.target,
            "task": self.task,
            "features": self.features,
            "exclude_columns": self.exclude_columns,
            "snv": self.snv,
            "lower_limits": self.lower_limits,
            "upper_limits": self.upper_limits,
            "y_lower_limits": self.y_lower_limits,
            "y_upper_limits": self.y_upper_limits,
            "limit_band_fraction": self.limit_band_fraction,
            "encoding": asdict(self.encoding) if is_dataclass(self.encoding) else self.encoding,
            "spectral_noise_db": self.spectral_noise_db,
            "format": self.format,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(**d)


def load_dataset(config: Union[DatasetConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and encode a dataset using the universal loader.
    
    Parameters
    ----------
    config : DatasetConfig or dict
        Dataset configuration
        
    Returns
    -------
    dict with keys:
        - x_tif: Neutrosophic encoded features (n_samples, n_features, 3)
        - y_tif: Neutrosophic encoded target (n_samples, 1, 3)
        - metadata: Dataset information
        - dataframe: Original pandas DataFrame
    """
    if isinstance(config, dict):
        config = DatasetConfig.from_dict(config)
    
    # Load raw data
    df = load_dataframe(config.path, format=config.format, **config.load_kwargs)
    
    # Validate target column
    if config.target not in df.columns:
        raise ValueError(
            f"Target column '{config.target}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Determine feature columns
    if config.features is not None:
        feature_cols = config.features
    else:
        exclude = [config.target] + config.exclude_columns
        feature_cols = [c for c in df.columns if c not in exclude]
    
    # Extract arrays
    X = df[feature_cols].values.astype(float)
    y = df[config.target].values
    
    # Encode as neutrosophic tensors
    enc_result = encode_neutrosophic(
        X,
        y,
        task=config.task,
        snv=config.snv,
        lower_limits=config.lower_limits,
        upper_limits=config.upper_limits,
        y_lower_limits=config.y_lower_limits,
        y_upper_limits=config.y_upper_limits,
        limit_band_fraction=config.limit_band_fraction,
        encoding=config.encoding,
        spectral_noise_db=config.spectral_noise_db,
        return_metadata=True,
    )
    x_tif, y_tif, enc_meta = enc_result
    
    # Compute file hash for reproducibility
    md5 = hashlib.md5(config.path.read_bytes()).hexdigest()
    
    # Build metadata
    metadata = {
        "path": str(config.path),
        "md5": md5,
        "name": config.name,
        "task": config.task,
        "target_name": config.target,
        "feature_names": feature_cols,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "snv_applied": config.snv,
    }
    if isinstance(enc_meta, dict):
        if "encoder" in enc_meta:
            metadata["encoder"] = enc_meta["encoder"]
        if "class_map" in enc_meta:
            metadata["class_map"] = enc_meta["class_map"]
    
    # Add task-specific metadata
    if config.task == "regression":
        y_values = df[config.target].astype(float)
        metadata.update({
            "target_range": (float(y_values.min()), float(y_values.max())),
            "target_mean": float(y_values.mean()),
            "target_std": float(y_values.std()),
        })
    else:  # classification
        classes = df[config.target].unique().tolist()
        class_counts = df[config.target].value_counts().to_dict()
        metadata.update({
            "classes": classes,
            "n_classes": len(classes),
            "class_counts": class_counts,
        })
    
    return {
        "x_tif": x_tif,
        "y_tif": y_tif,
        "metadata": metadata,
        "dataframe": df,
        "config": config.to_dict(),
    }


# =============================================================================
# Interactive Dataset Selection
# =============================================================================

def list_available_datasets(data_dir: Union[str, Path] = "data") -> List[Dict[str, str]]:
    """
    List available datasets in a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory to search for data files
        
    Returns
    -------
    list of dict with keys: name, path, format, size
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    
    supported_formats = {".csv", ".tsv", ".arff", ".xlsx", ".xls", ".json", ".parquet"}
    datasets = []
    
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() in supported_formats:
            size_mb = path.stat().st_size / (1024 * 1024)
            datasets.append({
                "name": path.stem,
                "path": str(path),
                "format": detect_format(path),
                "size_mb": round(size_mb, 2),
            })
    
    return sorted(datasets, key=lambda x: x["name"])


def interactive_load_dataset(data_dir: Union[str, Path] = "data") -> Dict[str, Any]:
    """
    Interactively select and load a dataset.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory to search for data files
        
    Returns
    -------
    dict with loaded dataset (same as load_dataset)
    """
    datasets = list_available_datasets(data_dir)
    
    if not datasets:
        print(f"No datasets found in '{data_dir}'")
        path = input("Enter path to data file: ").strip()
        if not path:
            raise ValueError("No path provided")
    else:
        print("\nAvailable datasets:")
        print("-" * 60)
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']} ({ds['format']}, {ds['size_mb']} MB)")
        print("-" * 60)
        
        choice = input(f"Select dataset (1-{len(datasets)}) or enter path: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(datasets):
            path = datasets[int(choice) - 1]["path"]
        else:
            path = choice
    
    # Load to inspect columns
    df = load_dataframe(path)
    print(f"\nLoaded: {path}")
    print(f"Shape: {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    # Get target column
    target = input("\nEnter target column name: ").strip()
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found")
    
    # Determine task type
    task_input = input("Task type (r=regression, c=classification) [r]: ").strip().lower()
    task = "classification" if task_input == "c" else "regression"
    
    # Ask about excluded columns
    exclude_input = input("Columns to exclude (comma-separated, or leave empty): ").strip()
    exclude_columns = [c.strip() for c in exclude_input.split(",") if c.strip()]
    
    # Ask about SNV (for spectral data)
    snv_input = input("Apply SNV normalization? (y/n) [n]: ").strip().lower()
    snv = snv_input == "y"
    
    # Create config and load
    config = DatasetConfig(
        path=path,
        target=target,
        task=task,
        exclude_columns=exclude_columns,
        snv=snv,
    )
    
    data = load_dataset(config)
    
    print(f"\nDataset loaded successfully!")
    print(f"  X shape: {data['x_tif'].shape}")
    print(f"  y shape: {data['y_tif'].shape}")
    if task == "regression":
        print(f"  Target range: {data['metadata']['target_range']}")
    else:
        print(f"  Classes: {data['metadata']['classes']}")
    
    return data


# =============================================================================
# Preset Dataset Loaders
# =============================================================================

def get_preset_config(name: str) -> DatasetConfig:
    """
    Get preset configuration for known datasets.
    
    Parameters
    ----------
    name : str
        Dataset name: "idrc_wheat", "micromass", etc.
        
    Returns
    -------
    DatasetConfig
    """
    presets = {
        "idrc_wheat": DatasetConfig(
            path="data/MA_A2.csv",
            target="Protein",
            task="regression",
            exclude_columns=["ID"],
            snv=True,
            name="IDRC 2016 Wheat Protein",
        ),
        "micromass": DatasetConfig(
            path="data/micro-mass.arff",
            target="Class",
            task="classification",
            snv=False,
            name="MicroMass (OpenML 1514)",
        ),
    }
    
    if name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    
    return presets[name]


def load_preset(name: str) -> Dict[str, Any]:
    """Load a preset dataset by name."""
    config = get_preset_config(name)
    return load_dataset(config)
