import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class EncodingResult:
    """Container for a (T, I, F) encoding result."""

    truth: np.ndarray
    indeterminacy: np.ndarray
    falsity: np.ndarray
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EncoderConfig:
    """
    Configuration for choosing an encoder.

    Attributes
    ----------
    name : str
        Encoder name (e.g., "probabilistic", "rpca", "wavelet", "quantile", "augment", "auto").
    params : dict
        Hyperparameters forwarded to the encoder implementation.
    candidates : list[EncoderConfig], optional
        Candidate encoders for auto-selection.
    cv_folds : int
        Number of folds for auto-selection CV.
    random_state : int, optional
        Seed for reproducibility in stochastic encoders/selection.
    scorer : str
        Metric name for auto-selection ("rmse" or "mae").
    max_components : int
        Max components for quick NPLS/NPLSW evaluation in auto mode.
    """

    name: str = "probabilistic"
    params: Dict[str, Any] = field(default_factory=dict)
    candidates: Optional[List["EncoderConfig"]] = None
    cv_folds: int = 3
    random_state: Optional[int] = None
    scorer: str = "rmse"
    max_components: int = 5

    @classmethod
    def from_value(cls, value: Any) -> "EncoderConfig":
        """Normalize strings/dicts to EncoderConfig."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            cand_values = value.get("candidates")
            candidates = None
            if cand_values is not None:
                candidates = [
                    cls.from_value(v) for v in cand_values
                ]
            return cls(
                name=value.get("name", "probabilistic"),
                params=value.get("params", {}) or {},
                candidates=candidates,
                cv_folds=int(value.get("cv_folds", 3)),
                random_state=value.get("random_state"),
                scorer=value.get("scorer", value.get("metric", "rmse")),
                max_components=int(value.get("max_components", 5)),
            )
        raise TypeError(f"Unsupported encoder config type: {type(value)}")

    def ensure_candidates(self) -> List["EncoderConfig"]:
        """Return a non-empty candidate list (used for auto mode)."""
        if self.candidates:
            return self.candidates
        # Default candidate portfolio
        defaults = [
            EncoderConfig(name="probabilistic"),
            EncoderConfig(name="rpca"),
            EncoderConfig(name="wavelet"),
            EncoderConfig(name="quantile"),
            EncoderConfig(name="augment"),
            EncoderConfig(name="ndg"),  # Neutrosophic Differential Geometry
        ]
        return defaults


@dataclass
class AutoEncoderSelectionResult:
    """Metadata describing auto-selection outcomes."""

    best_config: EncoderConfig
    scores: Dict[str, float]
    metric: str


def _robust_location_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return median and MAD-based scale with safety floor."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    mad = mad + 1e-8
    return med, mad


def _tanh_squash(z: np.ndarray, scale: float = 3.0, beta: float = 1.0) -> np.ndarray:
    """Map non-negative scores to [0,1] using a smooth tanh ramp."""
    z = np.maximum(z, 0.0) / (scale + 1e-12)
    return np.tanh(z) ** beta


def _normalize_power(x: np.ndarray, beta: float) -> np.ndarray:
    """Clamp to [0,1] then apply power transform."""
    return np.clip(x, 0.0, 1.0) ** beta


# =============================================================================
# Alternative Encoders
# =============================================================================

def _robust_pca_pcp(
    X: np.ndarray,
    lambda_: Optional[float] = None,
    mu: Optional[float] = None,
    tol: float = 1e-7,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Robust PCA via principal component pursuit (inexact ALM)."""
    n_samples, n_features = X.shape
    norm_X = np.linalg.norm(X, ord="fro")
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(n_samples, n_features))
    spectral_norm = np.linalg.norm(X, ord=2)
    if mu is None:
        mu = 1.25 / (spectral_norm + 1e-8)

    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    for it in range(max_iter):
        # Low-rank update via singular value thresholding
        U, s, Vt = np.linalg.svd(X - S + (1.0 / mu) * Y, full_matrices=False)
        s_thresh = np.maximum(s - 1.0 / mu, 0.0)
        L = (U * s_thresh) @ Vt

        # Sparse update via soft thresholding
        residual = X - L + (1.0 / mu) * Y
        S = np.sign(residual) * np.maximum(np.abs(residual) - lambda_ / mu, 0.0)

        # Dual update and convergence check
        err_mat = X - L - S
        err = np.linalg.norm(err_mat, ord="fro") / (norm_X + 1e-12)
        Y = Y + mu * err_mat

        if err < tol:
            break

    return L, S, it + 1, float(err)


def encode_rpca_mixture(
    X: np.ndarray,
    beta_I: float = 2.0,
    beta_F: float = 2.0,
    falsity_z_sat: float = 4.0,
    residual_z_max: float = 3.5,
    lambda_sparse: Optional[float] = None,
    **kwargs: Any,
) -> EncodingResult:
    """
    RPCA-style encoder: low-rank Truth, sparse falsity, residual ambiguity.

    Truth  := low-rank component (L)
    Falsity: magnitude of sparse component (S)
    Indeterminacy: ambiguity between residual and sparse parts
    """
    L, S, iters, err = _robust_pca_pcp(X, lambda_=lambda_sparse, **kwargs)
    R = X - L

    # Falsity from sparse magnitude (robust z)
    med_s, mad_s = _robust_location_scale(S)
    scale_s = 1.4826 * mad_s
    scale_s = np.where(scale_s > 0, scale_s, 1.0)
    z_s = np.abs(S - med_s) / scale_s
    F = _tanh_squash(z_s, scale=falsity_z_sat, beta=beta_F)

    # Indeterminacy from residual ambiguity
    med_r, mad_r = _robust_location_scale(R)
    scale_r = 1.4826 * mad_r
    scale_r = np.where(scale_r > 0, scale_r, 1.0)
    z_r = np.abs(R - med_r) / scale_r
    ambiguity = np.abs(R) / (np.abs(R) + np.abs(S) + 1e-8)
    I = np.maximum(ambiguity, _tanh_squash(z_r, scale=residual_z_max))
    I = _normalize_power(I, beta_I)

    metadata = {"encoder": "rpca", "rpca_iters": iters, "rpca_error": err}
    return EncodingResult(truth=L, indeterminacy=I, falsity=F, metadata=metadata)


def encode_wavelet_multiscale(
    X: np.ndarray,
    wavelet: str = "db2",
    level: Optional[int] = None,
    high_bands: Sequence[int] = (1,),
    mid_bands: Sequence[int] = (2, 3),
    beta_I: float = 2.0,
    beta_F: float = 2.0,
) -> EncodingResult:
    """
    Multi-scale wavelet encoder: low-frequency Truth, high-frequency falsity,
    mid-frequency indeterminacy.
    """
    try:
        import pywt  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in error handling
        raise ImportError(
            "Wavelet encoder requires PyWavelets (`pip install pywavelets`). "
            "Alternatively choose encoding='probabilistic' or 'rpca'."
        ) from exc

    n_samples, n_features = X.shape
    T = np.zeros_like(X)
    I = np.zeros_like(X)
    F = np.zeros_like(X)

    for i in range(n_samples):
        coeffs = pywt.wavedec(X[i], wavelet=wavelet, level=level)
        approx = coeffs[0]
        zeros_details = [np.zeros_like(c) for c in coeffs[1:]]
        T_rec = pywt.waverec([approx] + zeros_details, wavelet=wavelet)
        T[i, : min(n_features, T_rec.shape[0])] = T_rec[:n_features]

        total_detail = np.zeros(n_features)
        mid_energy = np.zeros(n_features)
        high_energy = np.zeros(n_features)

        for lvl, detail in enumerate(coeffs[1:], start=1):
            comp = [np.zeros_like(approx)] + [np.zeros_like(c) for c in coeffs[1:]]
            comp[lvl] = detail
            rec = pywt.waverec(comp, wavelet=wavelet)
            rec = rec[:n_features]
            energy = np.abs(rec)
            total_detail += energy
            if lvl in high_bands:
                high_energy += energy
            if lvl in mid_bands:
                mid_energy += energy

        # Falsity: spikes from highest-frequency details
        high_scale = np.median(high_energy) + 1e-8
        F[i] = _tanh_squash(high_energy / (high_scale + 1e-8), scale=3.0, beta=beta_F)

        # Indeterminacy: wobbliness from mid-bands relative to total energy
        denom = total_detail + np.abs(T[i]) + 1e-8
        I[i] = _normalize_power(mid_energy / denom, beta_I)

    metadata = {"encoder": "wavelet", "wavelet": wavelet, "level": level}
    return EncodingResult(truth=T, indeterminacy=I, falsity=F, metadata=metadata)


def encode_quantile_envelope(
    X: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    rho: float = 0.3,
    beta_I: float = 1.5,
    beta_F: float = 2.0,
) -> EncodingResult:
    """
    Quantile-envelope encoder: non-parametric boundaries for T/I/F.
    """
    q_low = np.quantile(X, lower_q, axis=0)
    q_high = np.quantile(X, upper_q, axis=0)
    q_med = np.quantile(X, 0.5, axis=0)
    spread = q_high - q_low
    spread = np.where(spread > 1e-8, spread, 1.0)

    T = X.copy()
    T = np.where(
        T < q_low,
        q_low + rho * (T - q_low),
        T,
    )
    T = np.where(
        T > q_high,
        q_high + rho * (T - q_high),
        T,
    )

    # Falsity from outside-envelope distance
    dist_low = (q_low - X) / spread
    dist_high = (X - q_high) / spread
    q_score = np.where(X < q_low, dist_low, 0.0)
    q_score = np.where(X > q_high, dist_high, q_score)
    F = _tanh_squash(q_score, scale=1.0, beta=beta_F)

    # Indeterminacy high near boundaries, low in the center
    center_ratio = np.abs(X - q_med) / spread
    I = 1.0 - np.clip(center_ratio, 0.0, 1.0)
    I = _normalize_power(I, beta_I)

    metadata = {"encoder": "quantile", "lower_q": lower_q, "upper_q": upper_q}
    return EncodingResult(truth=T, indeterminacy=I, falsity=F, metadata=metadata)


def _apply_augmentations(
    X: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
    offset_scale: float,
    scale_std: float,
) -> np.ndarray:
    """Generate a single augmented view of X."""
    n_samples, _ = X.shape
    offsets = rng.normal(scale=offset_scale, size=(n_samples, 1))
    scales = 1.0 + rng.normal(scale=scale_std, size=(n_samples, 1))
    noise = rng.normal(scale=noise_scale, size=X.shape)
    return (X + offsets) * scales + noise


def encode_augmentation_stability(
    X: np.ndarray,
    n_augmentations: int = 5,
    noise_scale: float = 0.01,
    offset_scale: float = 0.01,
    scale_std: float = 0.01,
    beta_I: float = 2.0,
    beta_F: float = 2.0,
    random_state: Optional[int] = None,
) -> EncodingResult:
    """
    Augmentation-based encoder: Truth from augmentation mean, I/F from instability.
    """
    rng = np.random.default_rng(random_state)
    views = np.stack(
        [
            _apply_augmentations(X, rng, noise_scale, offset_scale, scale_std)
            for _ in range(n_augmentations)
        ],
        axis=0,
    )  # (n_aug, n_samples, n_features)

    mean_view = views.mean(axis=0)
    std_view = views.std(axis=0, ddof=1)

    # Indeterminacy from variability across augmentations
    med_s, mad_s = _robust_location_scale(std_view)
    scale_s = 1.4826 * mad_s
    scale_s = np.where(scale_s > 0, scale_s, 1.0)
    z_std = np.abs(std_view - med_s) / scale_s
    I = _normalize_power(_tanh_squash(z_std, scale=3.0), beta_I)

    # Falsity from disagreement between original and stable mean + instability
    delta = np.abs(X - mean_view)
    med_d, mad_d = _robust_location_scale(delta)
    scale_d = 1.4826 * mad_d
    scale_d = np.where(scale_d > 0, scale_d, 1.0)
    z_delta = delta / scale_d
    F = _tanh_squash(z_delta, scale=3.0)
    F *= _tanh_squash(z_std, scale=3.0)
    F = _normalize_power(F, beta_F)

    metadata = {"encoder": "augment", "n_augmentations": n_augmentations}
    return EncodingResult(truth=mean_view, indeterminacy=I, falsity=F, metadata=metadata)


def encode_probabilistic_tif(X: np.ndarray, **kwargs: Any) -> EncodingResult:
    """Wrapper returning an EncodingResult for the probabilistic residual encoder."""
    I, F = encode_residual_probabilistic(X, **kwargs)
    return EncodingResult(truth=X, indeterminacy=I, falsity=F, metadata={"encoder": "probabilistic"})


def encode_spectroscopy_tif(X: np.ndarray, **kwargs: Any) -> EncodingResult:
    """Wrapper returning an EncodingResult for the spectroscopy encoder."""
    I, F = encode_spectroscopy(X, **kwargs)
    return EncodingResult(truth=X, indeterminacy=I, falsity=F, metadata={"encoder": "spectroscopy"})


def encode_robust(X: np.ndarray, trim_iterations: int = 3, z_threshold: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust encoding with iteratively trimmed MAD-based statistics.
    
    This encoder is designed for spike detection in contaminated spectral data.
    It uses iterative trimming to compute robust center/scale estimates that
    are not influenced by outliers, then applies sigmoid-based falsity scoring.
    
    Args:
        X: Input matrix (n_samples, n_features).
        trim_iterations: Number of iterative trimming passes.
        z_threshold: MAD-based z-score threshold for trimming outliers.
    
    Returns:
        I: Indeterminacy in [0,1] per cell (scaled z-score).
        F: Falsity probabilities [0,1] per cell (sigmoid of z-score).
    """
    n_samples, n_features = X.shape
    I = np.zeros_like(X)
    F = np.zeros_like(X)
    
    for j in range(n_features):
        col = X[:, j]
        
        # Iterative trimming: remove extremes, recompute, repeat
        mask = np.ones(n_samples, dtype=bool)
        for _ in range(trim_iterations):
            valid = col[mask]
            if len(valid) < 10:
                break
            center = np.median(valid)
            mad = np.median(np.abs(valid - center)) * 1.4826 + 1e-8
            z = np.abs(col - center) / mad
            mask = z < z_threshold  # Keep points within threshold
        
        # Final robust center/scale from trimmed data
        valid = col[mask]
        if len(valid) < 2:
            # Fallback to global statistics if too few points remain
            center = np.median(col)
            mad = np.median(np.abs(col - center)) * 1.4826 + 1e-8
        else:
            center = np.median(valid)
            mad = np.median(np.abs(valid - center)) * 1.4826 + 1e-8
        
        # Compute z-scores using robust estimates
        z = np.abs(col - center) / mad
        
        # Indeterminacy: scaled z-score (linear ramp to 1 at z=3)
        I[:, j] = np.clip(z / 3.0, 0, 1)
        
        # Falsity: sigmoid of z-score for smooth transition
        F[:, j] = 1 / (1 + np.exp(-(z - 3.0)))
    
    return I, F


def encode_robust_tif(X: np.ndarray, **kwargs: Any) -> EncodingResult:
    """Wrapper returning an EncodingResult for the robust MAD-based encoder."""
    I, F = encode_robust(X, **kwargs)
    return EncodingResult(truth=X, indeterminacy=I, falsity=F, metadata={"encoder": "robust"})


def dispatch_encoder(X: np.ndarray, encoder: EncoderConfig) -> EncodingResult:
    """Dispatch to the requested encoder implementation."""
    name = encoder.name.lower()
    params = encoder.params or {}

    if name in ("probabilistic", "default"):
        return encode_probabilistic_tif(X, **params)
    if name == "spectroscopy":
        return encode_spectroscopy_tif(X, **params)
    if name == "rpca":
        return encode_rpca_mixture(X, **params)
    if name == "wavelet":
        return encode_wavelet_multiscale(X, **params)
    if name == "quantile":
        return encode_quantile_envelope(X, **params)
    if name in ("augment", "augmentation"):
        return encode_augmentation_stability(X, **params)
    if name == "robust":
        return encode_robust_tif(X, **params)
    if name in ("ndg", "ndg_manifold", "manifold"):
        return encode_ndg_manifold_tif(X, **params)

    raise ValueError(f"Unknown encoder '{encoder.name}'")


def _quick_cv_score(
    x_tif: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "rmse",
    cv_folds: int = 3,
    max_components: int = 5,
    random_state: Optional[int] = None,
    task: str = "regression",
) -> float:
    """Lightweight CV loop for auto-selection."""
    from sklearn.model_selection import KFold
    from .model_factory import create_model_from_params

    y = np.asarray(y).reshape(-1, 1)
    y_tif = np.stack([y, np.zeros_like(y), np.zeros_like(y)], axis=-1)

    splits = min(cv_folds, len(y))
    if splits < 2:
        return float("inf")

    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores: List[float] = []
    for train_idx, test_idx in kf.split(y):
        n_comp = max(1, min(max_components, x_tif.shape[1], len(train_idx)))
        model = create_model_from_params(method="NPLSW", n_components=n_comp)
        model.fit(x_tif[train_idx], y_tif[train_idx])
        preds = model.predict(x_tif[test_idx]).reshape(-1, 1)
        true = y[test_idx]

        if task == "classification":
            pred_labels = np.rint(preds).astype(int).flatten()
            true_labels = true.flatten().astype(int)
            score = float(np.mean(pred_labels != true_labels))  # error rate (lower is better)
        elif metric == "mae":
            score = float(np.mean(np.abs(preds - true)))
        else:  # default RMSE
            score = float(np.sqrt(np.mean((preds - true) ** 2)))
        scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")


def auto_select_encoder(
    X: np.ndarray,
    y: np.ndarray,
    config: EncoderConfig,
    task: str = "regression",
) -> Tuple[EncodingResult, AutoEncoderSelectionResult]:
    """Evaluate candidate encoders and return the best-performing result."""
    candidates = config.ensure_candidates()
    metric = config.scorer if task == "regression" else "error_rate"
    best_score = float("inf")
    best_result: Optional[EncodingResult] = None
    best_config: Optional[EncoderConfig] = None
    scores: Dict[str, float] = {}

    for cand in candidates:
        result = dispatch_encoder(X, cand)
        x_tif = np.stack([result.truth, result.indeterminacy, result.falsity], axis=-1)
        score = _quick_cv_score(
            x_tif,
            y,
            metric=metric,
            cv_folds=config.cv_folds,
            max_components=config.max_components,
            random_state=config.random_state,
            task=task,
        )
        scores[cand.name] = score
        if score < best_score:
            best_score = score
            best_result = result
            best_config = cand

    if best_result is None or best_config is None:
        # Fallback: return the first candidate encoding even if scoring failed
        best_config = candidates[0]
        best_result = dispatch_encoder(X, best_config)
        scores.setdefault(best_config.name, float("inf"))

    metadata = AutoEncoderSelectionResult(best_config=best_config, scores=scores, metric=metric)
    if best_result.metadata is None:
        best_result.metadata = {}
    best_result.metadata.update({
        "auto_selected": True,
        "auto_metric": metric,
        "auto_scores": scores,
        "auto_best": best_config.name,
    })
    return best_result, metadata

def encode_spectroscopy(
    absorbance: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    spectral_noise_db: float = -20.0,
    derivative_order: int = 2,
    window_length: int = 5,
    pca_components: int = 5,
    residual_z_max: float = 5.0,
    falsity_z_thresh: float = 4.5,
    use_spike_detection: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode I/F based on spectroscopy domain knowledge.

    This encoder combines:
    - Global SNR-based indeterminacy (low SNR -> high I)
    - PCA residual-based indeterminacy (poorly reconstructed cells -> high I)
    - Falsity from residual outliers, physical bounds, and smoothness violations.

    Args:
        absorbance: Measured spectra (n_samples, n_wavelengths).
        wavelengths: Wavenumber/wavelength values (1D) - currently unused.
        spectral_noise_db: Instrument noise floor (dB) relative to max signal.
        derivative_order: Order of derivative for smoothness anomalies.
        window_length: Placeholder for more advanced smoothing (unused here).
        pca_components: Max number of components for low-rank reconstruction.
        residual_z_max: Robust z where I from residuals saturates at 1.
        falsity_z_thresh: Robust z threshold for residual-based falsity.

    Returns:
        I: Indeterminacy in [0,1] per cell.
        F: Falsity probabilities/flags (0..1) per cell.
    """
    n_samples, n_features = absorbance.shape

    # ------------------------------------------------------------------
    # 1. Initialize base indeterminacy (start at 0, let residuals drive it)
    # ------------------------------------------------------------------
    # Note: We removed the SNR-based global indeterminacy as it was too
    # aggressive for clean data. Instead, we rely on the residual-based
    # mixture model which is sample-specific and more discriminating.
    I = np.zeros((n_samples, n_features))

    # ------------------------------------------------------------------
    # 2. PCA residual-based generative model for residuals
    # ------------------------------------------------------------------
    # Center spectra
    X = absorbance
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # Choose rank for reconstruction
    max_rank = min(n_samples, n_features)
    k = min(pca_components, max_rank - 1) if max_rank > 1 else 0

    if k > 0:
        # SVD-based low-rank reconstruction
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        X_recon = (U_k * S_k) @ Vt_k
        R = Xc - X_recon

        # Robust per-wavelength location/scale for residuals
        med = np.median(R, axis=0)
        mad = np.median(np.abs(R - med), axis=0)
        mad = mad + 1e-8

        # Approximate "clean" residual scale (Normal model from MAD)
        sigma_clean = mad / 0.6745
        # Avoid degenerate scales
        positive = sigma_clean > 0
        if not np.all(positive):
            if positive.any():
                fallback = float(np.median(sigma_clean[positive]))
            else:
                fallback = 1.0
            sigma_clean = np.where(positive, sigma_clean, fallback)

        # "Bad" regime: much heavier-tailed residuals.
        # Use residual_z_max as a proxy for how much broader the bad regime is.
        bad_scale = max(residual_z_max, 3.0)
        sigma_bad = bad_scale * sigma_clean

        # Residuals relative to each regime
        r_centered = R - med
        r_clean = r_centered / sigma_clean
        r_bad = r_centered / sigma_bad

        # Mixture priors for clean vs bad
        # CALIBRATED: Use pi_bad=0.02 for more conservative flagging on real data
        # Investigation showed pi_bad=0.05 was too aggressive, flagging normal spectral variation
        pi_clean = 0.98
        pi_bad = 1.0 - pi_clean

        # Log-likelihoods up to an additive constant
        log_sigma_clean = np.log(sigma_clean)
        log_sigma_bad = np.log(sigma_bad)
        log_pi_clean = np.log(pi_clean)
        log_pi_bad = np.log(pi_bad)

        log_p_clean = log_pi_clean - 0.5 * (r_clean ** 2) - log_sigma_clean
        log_p_bad = log_pi_bad - 0.5 * (r_bad ** 2) - log_sigma_bad

        # Posterior probability of "bad" regime per cell:
        #   F_prob = p_bad / (p_bad + p_clean)
        # Compute via stable log-odds
        logit_F = log_p_bad - log_p_clean
        logit_F = np.clip(logit_F, -50.0, 50.0)
        F_prob = 1.0 / (1.0 + np.exp(-logit_F))

        # Posterior probability of "clean" regime
        T_clean = 1.0 - F_prob

        # Indeterminacy: high near the decision boundary (T_clean ~ F_prob)
        # I_mix = 1 - |T_clean - F_prob|, so:
        I_mix = 1.0 - np.abs(T_clean - F_prob)

        # Use residual mixture-based Indeterminacy directly (no SNR combination)
        I = I_mix

        # Falsity from mixture posterior
        F_res = F_prob

        # Strong falsity for very extreme residual z-scores (safety net)
        z_res = np.abs(r_centered) / sigma_clean
        F_extreme = (z_res >= falsity_z_thresh).astype(float)
    else:
        # Degenerate case: fall back to SNR-only, no residual falsity
        F_res = np.zeros_like(I)
        F_extreme = np.zeros_like(I)

    # ------------------------------------------------------------------
    # 3. Spectral anomaly-based Falsity (smoothness-based only)
    # ------------------------------------------------------------------
    # NOTE: We removed physical bounds checking (absorbance < -0.1) because:
    # - After SNV normalization, spectra are zero-mean with unit variance,
    #   so negative values are completely normal and expected.
    # - The residual-based mixture model (F_res) already captures anomalies
    #   in a statistically principled way.
    # We keep only the spike detection for truly anomalous sharp features.

    # Smoothness / spikes via second derivative
    if n_features > derivative_order:
        d2 = np.diff(absorbance, n=derivative_order, axis=1)
        padding = derivative_order // 2
        pad_width = ((0, 0), (padding, derivative_order - padding))
        d2_padded = np.pad(d2, pad_width, mode="edge")
        spike_score = np.abs(d2_padded)
        
        # ------------------------------------------------------------------
        # IMPROVED: Conservative spike detection with edge protection
        # ------------------------------------------------------------------
        # CALIBRATED: Exclude edge wavelengths from spike detection
        # Real spectra often have boundary artifacts that are NOT true spikes
        edge_margin = min(15, n_features // 15)  # Exclude first/last ~7% of wavelengths
        
        # Use only z-score and absolute magnitude detection (percentile was too aggressive)
        # Level 1: MAD-based z-score detection per sample (robust to global outliers)
        # CALIBRATED: Raised threshold to 6.0 for very few false positives
        spike_median_per_sample = np.median(spike_score, axis=1, keepdims=True)
        spike_mad_per_sample = np.median(np.abs(spike_score - spike_median_per_sample), axis=1, keepdims=True)
        spike_mad_per_sample = np.maximum(spike_mad_per_sample, 1e-8)  # Avoid division by zero
        spike_z_per_sample = (spike_score - spike_median_per_sample) / (spike_mad_per_sample * 1.4826)
        # ------------------------------------------------------------------
        # SOFT SPIKE DETECTION: Graded F based on z-score severity
        # ------------------------------------------------------------------
        # Use sigmoid for graded response instead of binary:
        # - z < 4: F_spike ≈ 0 (not a spike)
        # - z = 6: F_spike ≈ 0.5 (uncertain)  
        # - z > 8: F_spike ≈ 1 (definite spike)
        z_center = 8.0  # Conservative: only extreme outliers
        z_slope = 0.5
        F_spike_z = 1.0 / (1.0 + np.exp(-z_slope * (spike_z_per_sample - z_center)))
        
        # Also check absolute magnitude (sigmoid centered at 10x median)
        global_median_d2 = np.median(spike_score)
        abs_ratio = spike_score / (global_median_d2 + 1e-8)
        F_spike_abs = 1.0 / (1.0 + np.exp(-0.3 * (abs_ratio - 10.0)))
        
        # Take maximum of z-score and absolute F values
        F_spike = np.maximum(F_spike_z, F_spike_abs)
        
        # Protect edge wavelengths
        if edge_margin > 0:
            F_spike[:, :edge_margin] = 0.0
            F_spike[:, -edge_margin:] = 0.0
        
    else:
        F_spike = np.zeros_like(absorbance)

    F_anom = F_spike  # Soft spike values instead of binary

    # Union of residual-based and anomaly-based falsity via probabilistic OR.
    # Default stays conservative (residual-only) for backward compatibility.
    if use_spike_detection:
        F = 1.0 - (1.0 - F_res) * (1.0 - F_extreme) * (1.0 - F_anom)
    else:
        F = 1.0 - (1.0 - F_res) * (1.0 - F_extreme)
    F = np.clip(F, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. Calibration of I/F for downstream reliability models
    # ------------------------------------------------------------------
    # Goal:
    # - Clean spectra: I,F close to 0 so NPLSW/PNPLS behave
    #   similarly to classical PLS.
    # - Clearly corrupted regions: I,F remain near 1.
    # 
    # CALIBRATED: Use beta=3.0 to compress moderate F values toward 0
    # This ensures clean real spectra have near-zero F while still detecting true spikes
    # Investigation showed beta=2.0 left too much background F on real spectral data
    beta_I = 2.5  # Moderate compression for uncertainty
    beta_F = 4.0  # Strong compression - F=0.5 becomes 0.0625
    I = np.clip(I, 0.0, 1.0) ** beta_I
    F = np.clip(F, 0.0, 1.0) ** beta_F

    return I, F


def encode_residual_probabilistic(
    X: np.ndarray,
    pca_components: int = 5,
    residual_z_max: float = 5.0,
    falsity_z_thresh: float = 4.5,
    beta_I: float = 2.0,  # Reduced from 4.0 - preserve more uncertainty signal
    beta_F: float = 2.0,  # Reduced from 4.0 - preserve more falsity signal
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic probabilistic residual encoder for I/F based on a low-rank model.

    This encoder is domain-agnostic (no spectroscopy-specific SNR or bounds).
    It fits a low-rank reconstruction via SVD, then uses a two-regime Gaussian
    mixture on residuals (clean vs bad) to derive:

      - F: posterior probability of belonging to the bad regime, reinforced
           by extreme residual z-scores.
      - I: ambiguity about the regime (high when clean/bad are similar).

    Args:
        X: Input matrix (n_samples, n_features).
        pca_components: Rank for low-rank reconstruction.
        residual_z_max: Controls the width of the bad regime.
        falsity_z_thresh: Robust z threshold for extreme falsity.
        beta_I: Power transform exponent for I calibration.
        beta_F: Power transform exponent for F calibration.
    """
    n_samples, n_features = X.shape

    # Center features
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    max_rank = min(n_samples, n_features)
    k = min(pca_components, max_rank - 1) if max_rank > 1 else 0

    if k > 0:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        X_recon = (U_k * S_k) @ Vt_k
        R = Xc - X_recon

        med = np.median(R, axis=0)
        mad = np.median(np.abs(R - med), axis=0)
        mad = mad + 1e-8

        sigma_clean = mad / 0.6745
        positive = sigma_clean > 0
        if not np.all(positive):
            if positive.any():
                fallback = float(np.median(sigma_clean[positive]))
            else:
                fallback = 1.0
            sigma_clean = np.where(positive, sigma_clean, fallback)

        bad_scale = max(residual_z_max, 3.0)
        sigma_bad = bad_scale * sigma_clean

        r_centered = R - med
        r_clean = r_centered / sigma_clean
        r_bad = r_centered / sigma_bad

        pi_clean = 0.95
        pi_bad = 1.0 - pi_clean

        log_sigma_clean = np.log(sigma_clean)
        log_sigma_bad = np.log(sigma_bad)
        log_pi_clean = np.log(pi_clean)
        log_pi_bad = np.log(pi_bad)

        log_p_clean = log_pi_clean - 0.5 * (r_clean ** 2) - log_sigma_clean
        log_p_bad = log_pi_bad - 0.5 * (r_bad ** 2) - log_sigma_bad

        logit_F = log_p_bad - log_p_clean
        logit_F = np.clip(logit_F, -50.0, 50.0)
        F_prob = 1.0 / (1.0 + np.exp(-logit_F))

        T_clean = 1.0 - F_prob
        I_mix = 1.0 - np.abs(T_clean - F_prob)

        # Base I from mixture ambiguity
        I = I_mix

        # Falsity from mixture posterior with extreme z safety net
        z_res = np.abs(r_centered) / sigma_clean
        F_extreme = (z_res >= falsity_z_thresh).astype(float)

        F = 1.0 - (1.0 - F_prob) * (1.0 - F_extreme)
    else:
        I = np.zeros_like(X)
        F = np.zeros_like(X)

    # Calibration: compress mid-range values, keep extremes near 0/1.
    I = np.clip(I, 0.0, 1.0) ** beta_I
    F = np.clip(F, 0.0, 1.0) ** beta_F

    return I, F


# =============================================================================
# Neutrosophic Differential Geometry (NDG) Manifold Encoder
# =============================================================================

def _compute_local_variance(X: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local variance σ²_k at each wavelength using a sliding window.
    
    This approximates replicate variance when replicate scans are unavailable.
    Uses deviation from locally-smoothed signal as a proxy for measurement noise.
    
    Args:
        X: Input spectra (n_samples, n_features)
        window_size: Size of the sliding window for local smoothing
    
    Returns:
        Local variance estimate (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    
    # Compute locally smoothed signal (simple moving average)
    kernel = np.ones(window_size) / window_size
    X_smooth = np.zeros_like(X)
    for i in range(n_samples):
        X_smooth[i] = np.convolve(X[i], kernel, mode='same')
    
    # Local variance = squared deviation from smooth signal
    local_var = (X - X_smooth) ** 2
    
    # Smooth the variance estimate to reduce noise
    for i in range(n_samples):
        local_var[i] = np.convolve(local_var[i], kernel, mode='same')
    
    return local_var


def _shannon_entropy_transform(sigma_sq: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Transform local variance to indeterminacy using Shannon entropy analog.
    
    Based on Phase 2 of NDG framework:
        (g_ij)_I = δ_ij · H(p_i)
    
    For Gaussian noise with variance σ², the differential entropy is:
        H = 0.5 * log(2πeσ²)
    
    We normalize this to [0, 1] range.
    
    Args:
        sigma_sq: Local variance estimates
        scale: Scaling factor for entropy normalization
    
    Returns:
        Indeterminacy values in [0, 1]
    """
    # Avoid log(0) by adding small floor
    sigma_sq_safe = np.maximum(sigma_sq, 1e-12)
    
    # Differential entropy of Gaussian: H = 0.5 * log(2πeσ²)
    # We use a simplified form: H ∝ log(σ²)
    log_var = np.log(sigma_sq_safe)
    
    # Normalize to [0, 1] using robust min-max across the dataset
    # Higher entropy (larger variance) -> higher indeterminacy
    min_log = np.percentile(log_var, 1)
    max_log = np.percentile(log_var, 99)
    
    if max_log - min_log < 1e-8:
        return np.zeros_like(sigma_sq)
    
    H_normalized = (log_var - min_log) / (max_log - min_log + 1e-8)
    H_normalized = np.clip(H_normalized, 0.0, 1.0)
    
    # Apply scaling (adjusts sensitivity)
    return H_normalized ** scale


def _compute_systematic_error(
    X: np.ndarray,
    low_rank_components: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute systematic error coefficient ε_k from low-rank model deviation.
    
    Based on Phase 1 of NDG framework:
        F_k = 1 - N(x_k) · (1 - ε_k)
    
    The systematic error is derived from the covariance of residuals from
    a low-rank model, corresponding to:
        (g_ij)_F = (Σ_bias^{-1})_ij
    
    Args:
        X: Input spectra (n_samples, n_features)
        low_rank_components: Number of components for low-rank reconstruction
    
    Returns:
        epsilon: Systematic error coefficient ε_k per cell
        L: Low-rank reconstruction (for Truth channel)
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    
    # Compute low-rank approximation via SVD
    max_rank = min(n_samples, n_features)
    k = min(low_rank_components, max_rank - 1) if max_rank > 1 else 0
    
    if k > 0:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Low-rank reconstruction
        L = (U_k * S_k) @ Vt_k + X_mean
        
        # Residual = systematic deviation from ideal model
        R = X - L
        
        # Compute per-wavelength residual statistics (robust)
        med_R = np.median(R, axis=0, keepdims=True)
        mad_R = np.median(np.abs(R - med_R), axis=0, keepdims=True)
        sigma_R = mad_R * 1.4826 + 1e-8
        
        # Systematic error coefficient: normalized absolute residual
        # ε → 1 when residual is large, ε → 0 when residual is small
        z_R = np.abs(R) / sigma_R
        
        # Transform to [0, 1] using sigmoid (smooth transition)
        # Centered at z=3 (where 99.7% of Gaussian data should lie)
        epsilon = 1.0 / (1.0 + np.exp(-(z_R - 3.0)))
    else:
        L = X.copy()
        epsilon = np.zeros_like(X)
    
    return epsilon, L


def encode_ndg_manifold(
    X: np.ndarray,
    normalization: str = "none",  # Changed from "snv" - preserves concentration signal
    local_window: int = 5,
    low_rank_components: int = 5,
    entropy_scale: float = 1.5,
    beta_I: float = 2.0,
    beta_F: float = 2.0,
    use_fisher_metric: bool = False,
    replicate_scans: Optional[np.ndarray] = None,
) -> EncodingResult:
    r"""
    Neutrosophic Differential Geometry (NDG) Manifold Encoder.
    
    This encoder implements the theoretical framework from Neutrosophic 
    Differential Geometry, mapping spectroscopic data onto a manifold 
    $\mathcal{M}_\mathcal{N}$ with principled T/I/F channels.
    
    Mathematical Foundation (Phase 1):
    ----------------------------------
    The manifold mapping φ: R^p → M_N is defined component-wise:
    
        φ(x)_k = (T_k, I_k, F_k)
    
    where:
        - T_k = N(x_k): Normalized signal strength (or raw values if norm='none')
        - I_k = H(σ²_k): Shannon entropy transform of local variance
        - F_k = 1 - N(x_k)·(1 - ε_k): Systematic error coefficient
    
    Metric Interpretation (Phase 2):
    ---------------------------------
    The encoding implicitly defines a Neutrosophic Metric Tensor:
    
        g^N_ij = α(g_ij)_T - β(g_ij)_I - γ(g_ij)_F
    
    where high I/F "stretch" distances, making samples harder to distinguish.
    
    Parameters
    ----------
    X : np.ndarray
        Input spectra (n_samples, n_features).
    normalization : str, default="none"
        Truth normalization method: "snv", "minmax", or "none".
        NOTE: "snv" normalizes each sample to zero mean/unit variance, which 
        destroys concentration information. Use "none" for quantitative analysis.
    local_window : int, default=5
        Window size for local variance estimation.
    low_rank_components : int, default=5
        Number of components for systematic error model.
    entropy_scale : float, default=1.5
        Sensitivity of entropy-based indeterminacy.
    beta_I : float, default=2.0
        Power calibration for indeterminacy compression.
    beta_F : float, default=2.0
        Power calibration for falsity compression.
    use_fisher_metric : bool, default=False
        If True, incorporate Fisher Information in Truth metric (experimental).
    replicate_scans : np.ndarray, optional
        If provided (n_samples, n_replicates, n_features), use actual replicate
        variance instead of local smoothing approximation.
    
    Returns
    -------
    EncodingResult
        Encoded (T, I, F) triplets with NDG metadata.
    
    Notes
    -----
    The NDG framework provides a geometric interpretation where:
    - The Ricci Scalar R^I quantifies "noise density" of a spectral region
    - Geodesics on M_N represent "paths of maximum information retention"
    - Parallel transport enables principled calibration transfer
    
    References
    ----------
    Based on the Neutrosophic Differential Geometry framework combining:
    - Phase 1: Topological Foundation (Neutrosophic Vector Space V_N)
    - Phase 2: Metric Architecture (Fisher/Entropy/Error tensors)
    - Phase 3: Connection & Transport (Christoffel symbols, geodesics)
    - Phase 4: Curvature Analysis (Riemann tensor, Ricci scalar)
    """
    n_samples, n_features = X.shape
    
    # =========================================================================
    # TRUTH CHANNEL: T_k = N(x_k)
    # =========================================================================
    # The "Truth" represents the normalized signal strength, corresponding to
    # the (g_ij)_T component of the metric tensor.
    
    if normalization == "snv":
        # Standard Normal Variate: zero mean, unit variance per sample
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True)
        X_std = np.where(X_std > 1e-8, X_std, 1.0)
        T = (X - X_mean) / X_std
    elif normalization == "minmax":
        # Min-Max normalization to [0, 1] per sample
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        X_range = X_max - X_min
        X_range = np.where(X_range > 1e-8, X_range, 1.0)
        T = (X - X_min) / X_range
    else:
        # No normalization - use raw values
        T = X.copy()
    
    # =========================================================================
    # INDETERMINACY CHANNEL: I_k = H(σ²_k)
    # =========================================================================
    # From Phase 2: (g_ij)_I = δ_ij · H(p_i)
    # The indeterminacy metric is derived from Shannon entropy of local noise.
    
    if replicate_scans is not None:
        # Use actual replicate variance if available
        # replicate_scans shape: (n_samples, n_replicates, n_features)
        local_var = replicate_scans.var(axis=1)
    else:
        # Approximate local variance from smoothness deviation
        local_var = _compute_local_variance(X, window_size=local_window)
    
    # Transform variance to entropy-based indeterminacy
    I = _shannon_entropy_transform(local_var, scale=entropy_scale)
    
    # =========================================================================
    # FALSITY CHANNEL: F_k = 1 - N(x_k)·(1 - ε_k)
    # =========================================================================
    # From Phase 1: The falsity represents distance from ideal model.
    # From Phase 2: (g_ij)_F = (Σ_bias^{-1})_ij
    
    epsilon, L = _compute_systematic_error(X, low_rank_components)
    
    # Compute normalized signal strength for falsity formula
    # Using per-feature normalization for consistency
    X_feat_min = X.min(axis=0, keepdims=True)
    X_feat_max = X.max(axis=0, keepdims=True)
    X_feat_range = X_feat_max - X_feat_min
    X_feat_range = np.where(X_feat_range > 1e-8, X_feat_range, 1.0)
    N_x = (X - X_feat_min) / X_feat_range
    
    # F_k = 1 - N(x_k) · (1 - ε_k)
    # When ε_k ≈ 0 (no systematic error): F_k ≈ 1 - N(x_k) (inversely related to signal)
    # When ε_k ≈ 1 (high systematic error): F_k ≈ 1 (high falsity regardless of signal)
    F = 1.0 - N_x * (1.0 - epsilon)
    
    # Alternative: Use epsilon directly as falsity (simpler, more intuitive)
    # This makes F directly represent systematic error magnitude
    F_direct = epsilon
    
    # Blend the theoretical formula with direct epsilon
    # The theoretical formula can produce counter-intuitive results for low signals
    alpha_blend = 0.7  # 70% direct epsilon, 30% theoretical formula
    F = alpha_blend * F_direct + (1.0 - alpha_blend) * F
    
    # =========================================================================
    # CALIBRATION
    # =========================================================================
    # Power transform to compress middle values, keeping extremes distinct
    I = np.clip(I, 0.0, 1.0) ** beta_I
    F = np.clip(F, 0.0, 1.0) ** beta_F
    
    # =========================================================================
    # OPTIONAL: Fisher Information enhancement for Truth metric
    # =========================================================================
    if use_fisher_metric:
        # From Phase 2: (g_ij)_T = E[(∂lnL/∂θ_i)(∂lnL/∂θ_j)]
        # Approximate Fisher Information from score variance
        # This is experimental and requires concentration gradient data
        pass  # Placeholder for future implementation
    
    # Metadata including geometric quantities
    metadata = {
        "encoder": "ndg_manifold",
        "normalization": normalization,
        "local_window": local_window,
        "low_rank_components": low_rank_components,
        "entropy_scale": entropy_scale,
        # Geometric diagnostics
        "mean_indeterminacy": float(I.mean()),
        "mean_falsity": float(F.mean()),
        "mean_epsilon": float(epsilon.mean()),
        # Rough estimate of "spectral complexity" (analog to Ricci scalar)
        "complexity_score": float(I.mean() + F.mean()),
    }
    
    return EncodingResult(truth=T, indeterminacy=I, falsity=F, metadata=metadata)


def encode_ndg_manifold_tif(X: np.ndarray, **kwargs: Any) -> EncodingResult:
    """Wrapper for NDG Manifold encoder matching the standard interface."""
    return encode_ndg_manifold(X, **kwargs)
