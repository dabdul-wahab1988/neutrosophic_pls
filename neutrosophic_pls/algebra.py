"""Basic neutrosophic algebra utilities."""

from typing import Iterable, Tuple
import numpy as np

NeutroTriplet = Tuple[float, float, float]


def _as_triplet(val: Iterable[float]) -> NeutroTriplet:
    arr = tuple(val)
    if len(arr) != 3:
        raise ValueError("Neutrosophic triplets must have three elements (T, I, F).")
    return arr  # type: ignore


def neutro_inner(x: NeutroTriplet, y: NeutroTriplet, weights: NeutroTriplet = (1.0, 1.0, 1.0)) -> float:
    """
    Neutrosophic inner product between two triplets using channel weights.
    """
    tx, ix, fx = _as_triplet(x)
    ty, iy, fy = _as_triplet(y)
    wt, wi, wf = _as_triplet(weights)
    return wt * tx * ty + wi * ix * iy + wf * fx * fy


def neutro_norm(x: NeutroTriplet, weights: NeutroTriplet = (1.0, 1.0, 1.0)) -> float:
    """
    Neutrosophic norm induced by the weighted inner product.
    """
    return float(np.sqrt(neutro_inner(x, x, weights=weights)))


def combine_channels(
    data: np.ndarray,
    weights: NeutroTriplet = (1.0, 0.5, 1.0),
    mode: str = "truth_only",
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Collapse a T/I/F tensor (n_samples, n_features, 3) into a weighted matrix.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_features, 3) with channels [T, I, F].
    weights : tuple of 3 floats
        (w_T, w_I, w_F) controlling channel contributions.
    mode : str
        - "truth_only": X_eff = w_T * T (default, recommended for clean data)
          Simply extracts the Truth channel. I/F can be consumed by downstream
          reliability models (e.g., NPLSW/PNPLS) for sample- or element-level
          weighting instead of acting as additional features.
        - "attenuate": X_eff = w_T * T * attenuation(I, F)
          High I/F attenuates the Truth signal using soft thresholding.
    threshold : float
        For "attenuate" mode: minimum I/F value before attenuation starts.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_features).
    """
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError("Expected data with shape (n_samples, n_features, 3).")

    w_T, w_I, w_F = weights
    T = data[..., 0]
    I = data[..., 1]
    F = data[..., 2]

    if mode == "truth_only":
        # Simply return the Truth channel (scaled by w_T).
        # This ensures clean data behaves identically to standard PLS.
        # I/F information is used for sample-level weighting separately.
        return w_T * T
    elif mode == "attenuate":
        # Soft-thresholded attenuation: I/F below threshold have no effect.
        I_clipped = np.clip(I, 0, 1)
        F_clipped = np.clip(F, 0, 1)
        I_eff = np.maximum(0, I_clipped - threshold) / (1.0 - threshold + 1e-8)
        F_eff = np.maximum(0, F_clipped - threshold) / (1.0 - threshold + 1e-8)
        attenuation = (1.0 - w_I * I_eff) * (1.0 - w_F * F_eff)
        return w_T * T * attenuation
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'truth_only' or 'attenuate'.")
