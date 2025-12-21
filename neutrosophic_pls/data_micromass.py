"""Loader and encoding utilities for MicroMass (OpenML 1514)."""

import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.io import arff

from .algebra import combine_channels

MICROMASS_URLS = [
    "https://openml.org/data/v1/download/1593706",
    "https://api.openml.org/data/v1/download/1593706",
]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "micromass_fixture.arff"
FIXTURE_MD5: Optional[str] = hashlib.md5(FIXTURE_PATH.read_bytes()).hexdigest() if FIXTURE_PATH.exists() else None


def _default_cache_dir() -> Path:
    return Path("data")


def _download_first_available(dest: Path, expected_md5: Optional[str] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in MICROMASS_URLS:
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310
            if expected_md5:
                md5 = hashlib.md5(dest.read_bytes()).hexdigest()
                if md5 != expected_md5:
                    continue
            return dest
        except Exception:
            continue
    raise RuntimeError("Unable to download MicroMass dataset from provided URLs.")


def download_micromass(dest: Path | None = None, expected_md5: Optional[str] = None) -> Path:
    dest = dest or _default_cache_dir() / "micro-mass.arff"
    if dest.exists():
        return dest
    return _download_first_available(dest, expected_md5=expected_md5)


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + 1e-8
    return (x - med) / (1.4826 * mad)


def _encode_neutrosophic(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]
    x = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].astype("category").cat.codes.to_numpy(dtype=float).reshape(-1, 1)

    x_truth = x
    z = _robust_z(x)
    # Indeterminacy: scale to [0, 1] range (same as IDRC encoder)
    x_ind = np.clip(np.abs(z) / 3.0, 0, 1)
    x_falsity = (np.abs(z) > 3.5).astype(float)  # robust outlier flags

    y_truth = y
    y_ind = np.zeros_like(y_truth)
    y_falsity = np.zeros_like(y_truth)

    x_tif = np.stack([x_truth, x_ind, x_falsity], axis=-1)
    y_tif = np.stack([y_truth, y_ind, y_falsity], axis=-1)
    return x_tif, y_tif


def load_micromass(path: str | Path | None = None, prefer_fixture: bool = True) -> Dict:
    """
    Load MicroMass dataset and return neutrosophic-encoded tensors and metadata.
    Preference order: user-supplied path > packaged fixture > download.
    """
    if path:
        local_path = Path(path)
    elif prefer_fixture and FIXTURE_PATH.exists():
        local_path = FIXTURE_PATH
    else:
        local_path = download_micromass()

    data, meta = arff.loadarff(local_path)
    df = pd.DataFrame(data)
    x_tif, y_tif = _encode_neutrosophic(df)
    md5 = hashlib.md5(local_path.read_bytes()).hexdigest()
    metadata = {
        "path": str(local_path),
        "md5": md5,
        "feature_names": [c for c in df.columns if c != "Class"],
        "target_name": "Class",
        "n_samples": len(df),
        "n_features": len([c for c in df.columns if c != "Class"]),
        "fixture": local_path == FIXTURE_PATH,
    }
    return {"x_tif": x_tif, "y_tif": y_tif, "metadata": metadata}
