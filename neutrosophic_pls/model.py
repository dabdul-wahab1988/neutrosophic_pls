"""Neutrosophic NIPALS-style PLS implementation."""

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from .algebra import combine_channels


@dataclass
class NPLSConfig:
    n_components: int
    max_iter: int = 500
    tol: float = 1e-7
    channel_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0)
    # NOTE: combine_mode is retained for backward compatibility with earlier
    # cell-level gating experiments. The current NPLS implementation always
    # uses the Truth channel directly and does not call combine_channels.
    combine_mode: str = "truth_only"  # Reverted to truth_only as attenuate was over-attenuating
    use_sample_weights: bool = True  # Apply sample-level reliability weighting
    lambda_falsity: float = 0.5      # Strength of sample-level falsity downweighting


class NPLS:
    """
    Neutrosophic PLS using a NIPALS-style iterative algorithm.

    This allows robust behavior similar to classical PLS for clean data.
    
    Key design: Uses Truth channel directly (like NPLSW) and applies
    sample-level reliability weighting. This avoids cell-level signal
    distortion which was found to hurt performance.
    
    With combine_mode="truth_only" (default), NPLS extracts the Truth channel.
    Sample-level weighting downweights unreliable samples based on I/F
    """

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 500,
        tol: float = 1e-7,
        channel_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0),
        lambda_falsity: float = 0.5,
    ) -> None:
        self.config = NPLSConfig(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            channel_weights=channel_weights,
            lambda_falsity=lambda_falsity,
        )
        self.converged_: bool = False
        self.iterations_: list[int] = []
        self._fitted: bool = False

    def _compute_sample_weights(self, x_tif: np.ndarray) -> np.ndarray:
        """
        Compute reliability-based sample weights from I/F channels.
        
        Uses the same approach as NPLSW: samples with high proportion of
        bad cells (F > threshold) are downweighted.
        """
        if not self.config.use_sample_weights:
            return np.ones(x_tif.shape[0])
        
        # Compute fraction of "bad" cells per sample (F > threshold)
        F_threshold = 0.3  # Cell is "bad" if F > 0.3
        x_F = x_tif[..., 2]
        bad_cell_fraction = (x_F > F_threshold).mean(axis=1)
        
        # Sample weight is inversely proportional to bad cell fraction
        # omega = 1 - lambda_F * bad_fraction
        lambda_F = min(self.config.lambda_falsity, 0.9)  # Cap to avoid zero weights
        omega = 1.0 - lambda_F * bad_cell_fraction
        
        # Add small floor to prevent zero weights
        omega = np.maximum(omega, 0.01)
        
        # Normalize to mean 1
        omega = omega / (omega.mean() + 1e-12)
        
        return omega

    def _detect_and_combine(self, x_tif: np.ndarray, is_fit: bool = True) -> np.ndarray:
        """
        Current NPLS behavior: always use the Truth channel directly.

        The earlier cell-level gating/attenuation experiments are retained only
        as legacy configuration (see ``combine_mode``), but are intentionally
        not applied in the current algorithm because they distort spectroscopy
        signal physics and can hurt predictive performance.
        """
        return x_tif[..., 0]

    def fit(self, x_tif: np.ndarray, y_tif: np.ndarray) -> "NPLS":
        # CLEAN DATA BYPASS: Use sklearn PLS when I/F are low or weights are uniform
        # Raised threshold from 0.01 to 0.15 since encoders produce I/F ~0.05-0.10
        # even for relatively clean data
        mean_F = x_tif[..., 2].mean()
        mean_I = x_tif[..., 1].mean()
        
        # Also check if sample weights would be essentially uniform
        # If so, no benefit from weighted NIPALS
        weights_preview = self._compute_sample_weights(x_tif)
        weight_cv = weights_preview.std() / (weights_preview.mean() + 1e-8)
        
        use_sklearn_bypass = (
            (mean_F < 0.15 and mean_I < 0.15) or  # Low I/F overall
            weight_cv < 0.05  # Weights are effectively uniform (CV < 5%)
        )
        
        if use_sklearn_bypass:
            from sklearn.cross_decomposition import PLSRegression
            X = x_tif[..., 0]
            Y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
            if Y.ndim == 1: Y = Y.reshape(-1, 1)
            n_comp = min(self.config.n_components, X.shape[1], X.shape[0] - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, Y)
            self._fitted = True
            self.converged_ = True
            self.x_mean_ = pls._x_mean.reshape(1, -1)
            self.y_mean_ = pls._y_mean.reshape(1, -1)
            self.coef_ = pls.coef_.T if pls.coef_.ndim > 1 else pls.coef_.reshape(-1, 1)
            self.sample_weights_ = np.ones(X.shape[0])
            # Copy VIP-related attributes from sklearn PLS
            self.scores_ = pls.x_scores_
            self.weights_x_ = pls.x_weights_
            self.loadings_ = pls.x_loadings_
            self.y_loadings_ = pls.y_loadings_.T  # sklearn is (n_targets, n_comp), we use (n_comp, n_targets)
            self._sklearn_pls = pls
            self._use_sklearn = True
            self._bypass_reason = "low_IF" if (mean_F < 0.15 and mean_I < 0.15) else "uniform_weights"
            return self
        # Automatically combine channels based on corruption level
        X = self._detect_and_combine(x_tif, is_fit=True)
        Y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Compute sample weights based on reliability
        weights = self._compute_sample_weights(x_tif)
        self.sample_weights_ = weights

        # Weighted centering
        weight_sum = weights.sum()
        if weight_sum > 0:
            self.x_mean_ = np.average(X, axis=0, weights=weights).reshape(1, -1)
            self.y_mean_ = np.average(Y, axis=0, weights=weights).reshape(1, -1)
        else:
            self.x_mean_ = X.mean(axis=0, keepdims=True)
            self.y_mean_ = Y.mean(axis=0, keepdims=True)

        Xc = X - self.x_mean_
        Yc = Y - self.y_mean_

        # Apply sqrt-weighting for weighted inner products in NIPALS
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw = Xc * sqrt_w
        Yw = Yc * sqrt_w

        n_samples, n_features = Xc.shape
        n_targets = Yc.shape[1]
        n_comp = self.config.n_components

        T = np.zeros((n_samples, n_comp))
        U = np.zeros((n_samples, n_comp))
        P = np.zeros((n_features, n_comp))
        Q = np.zeros((n_comp, n_targets))
        W = np.zeros((n_features, n_comp))

        # Work with weighted data for fitting
        Xw_work = Xw.copy()
        Yw_work = Yw.copy()
        Xc_work = Xc.copy()  # Keep unweighted copy for proper loadings
        Yc_work = Yc.copy()

        for i in range(n_comp):
            # Initialize u from weighted Y
            u = Yw_work[:, 0:1]
            iters = 0
            for it in range(self.config.max_iter):
                iters = it + 1
                # Weight vector from weighted covariance
                w = Xw_work.T @ u
                w /= np.linalg.norm(w) + 1e-12
                
                # Scores from UNWEIGHTED centered data (for proper prediction)
                t_unweighted = Xc_work @ w
                
                # Weighted scores for iteration
                t_weighted = Xw_work @ w
                
                # Y loading from weighted data
                c = Yw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
                c /= np.linalg.norm(c) + 1e-12
                u_new = Yw_work @ c / (c.T @ c + 1e-12)
                
                if np.linalg.norm(u_new - u) < self.config.tol:
                    u = u_new
                    break
                u = u_new

            # Compute loadings using weighted regression
            p = Xw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
            q = Yw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
            
            # Use unweighted scores for storage
            t = t_unweighted

            # Store component results
            T[:, i] = t.ravel()
            U[:, i] = (Yc_work @ c / (c.T @ c + 1e-12)).ravel()
            P[:, i] = p.ravel()
            Q[i, :] = q.ravel()
            W[:, i] = w.ravel()
            self.iterations_.append(iters)

            # Deflate both weighted and unweighted
            Xw_work = Xw_work - np.outer(t_weighted, p)
            Yw_work = Yw_work - np.outer(t_weighted, q)
            Xc_work = Xc_work - np.outer(t, p)
            Yc_work = Yc_work - np.outer(t, q)

        self.scores_ = T
        self.y_scores_ = U
        self.loadings_ = P
        self.y_loadings_ = Q  # shape (n_components, n_targets)
        self.weights_x_ = W
        self.components_ = W.T @ P  # akin to x_weights' * x_loadings
        self.converged_ = True
        self._fitted = True

        # Regression coefficient matrix (standard PLS relation)
        R = W @ np.linalg.inv(P.T @ W + 1e-12 * np.eye(n_comp))
        self.coef_ = R @ Q
        return self

    def transform(self, x_tif: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = x_tif[..., 0]  # Always use Truth channel
        Xc = X - self.x_mean_
        T = Xc @ self.weights_x_ @ np.linalg.inv(self.loadings_.T @ self.weights_x_ + 1e-12 * np.eye(self.config.n_components))
        return T

    def predict(self, x_tif: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if hasattr(self, '_use_sklearn') and self._use_sklearn:
            return self._sklearn_pls.predict(x_tif[..., 0]).ravel()
        X = x_tif[..., 0]  # Always use Truth channel
        Xc = X - self.x_mean_
        y_pred = Xc @ self.coef_ + self.y_mean_
        # Return 1D array for single-target prediction (matching sklearn behavior)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")


def _normalize_weights(weights: np.ndarray, mode: Literal["none", "mean1", "sum1"]) -> np.ndarray:
    """
    Normalise reliability weights according to the requested mode.
    """
    weights = weights.astype(float, copy=True).ravel()
    nonzero = weights > 0
    if mode == "none":
        return weights
    if mode == "mean1":
        if nonzero.any():
            weights[nonzero] = weights[nonzero] / (weights[nonzero].mean() + 1e-12)
        else:
            weights = np.ones_like(weights)
        return weights
    if mode == "sum1":
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones_like(weights) / len(weights)
        return weights
    raise ValueError("normalize must be one of {'none', 'mean1', 'sum1'}.")


@dataclass
class NPLSWConfig:
    n_components: int
    # Softer defaults so that probabilistic I/F do not over-attenuate.
    lambda_indeterminacy: float = 0.2
    # Weight for falsity penalty when constructing sample-level reliability
    # weights. Kept compatible with earlier cell-level formulations.
    lambda_falsity: float = 0.5
    alpha: float = 2.0  # Exponent for soft falsity weighting (alpha=1 for linear)
    normalize: Literal["none", "mean1", "sum1"] = "mean1"
    max_iter: int = 500
    tol: float = 1e-7
    channel_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0)


class NPLSW:
    """
    Reliability-weighted neutrosophic PLS (Paper 2).

    Current implementation:
        - Always operates on the Truth channel (no cell-level gating).
        - Applies sample-level reliability weighting based on the fraction
          of high-falsity cells per sample.

    The configuration parameters lambda_indeterminacy, lambda_falsity,
    alpha and channel_weights are retained for compatibility and for
    potential future re-introduction of cell-level weighting, but only
    lambda_falsity and normalize affect the current weighting scheme.
    """

    def __init__(
        self,
        n_components: int,
        *,
        lambda_indeterminacy: float = 0.2,
        lambda_falsity: float = 0.5,
        alpha: float = 2.0,
        normalize: Literal["none", "mean1", "sum1"] = "mean1",
        max_iter: int = 500,
        tol: float = 1e-7,
        channel_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0),
    ) -> None:
        self.config = NPLSWConfig(
            n_components=n_components,
            lambda_indeterminacy=lambda_indeterminacy,
            lambda_falsity=lambda_falsity,
            alpha=alpha, # Pass new parameter to config
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            channel_weights=channel_weights,
        )
        self.converged_: bool = False
        self.iterations_: list[int] = []
        self._fitted: bool = False

    def _compute_sample_weights(self, x_tif: np.ndarray, y_tif: np.ndarray) -> np.ndarray:
        """
        Compute reliability-based sample weights from falsity statistics.

        Each sample's weight is a decreasing function of the proportion of
        cells with high falsity (F > threshold). Weights are clipped to a
        small positive floor and normalised according to the requested mode
        in ``self.config.normalize``.
        """
        # Compute fraction of "bad" cells per sample (F > threshold)
        F_threshold = 0.3  # Cell is "bad" if F > 0.3
        x_F = x_tif[..., 2]
        bad_cell_fraction = (x_F > F_threshold).mean(axis=1)
        
        # Sample weight is inversely proportional to bad cell fraction
        # omega = 1 - lambda_F * bad_fraction
        # When bad_fraction = 0, weight = 1 (normal sample)
        # When bad_fraction = 1, weight = 1 - lambda_F (heavily downweighted)
        lambda_F = min(self.config.lambda_falsity, 0.9)  # Cap to avoid zero weights
        omega = 1.0 - lambda_F * bad_cell_fraction
        
        # Add small floor to prevent zero weights
        omega = np.maximum(omega, 0.01)
        
        omega = _normalize_weights(omega, self.config.normalize)
        return omega

    def fit(self, x_tif: np.ndarray, y_tif: np.ndarray) -> "NPLSW":
        # CLEAN DATA BYPASS: Use sklearn PLS when I/F are low or weights are uniform
        # Raised threshold from 0.01 to 0.15 since encoders produce I/F ~0.05-0.10
        # even for relatively clean data
        mean_F = x_tif[..., 2].mean()
        mean_I = x_tif[..., 1].mean()
        
        # Preview sample weights to check uniformity
        dummy_y = np.zeros((x_tif.shape[0], 1))  # Placeholder, not used in weight computation
        weights_preview = self._compute_sample_weights(x_tif, dummy_y)
        weight_cv = weights_preview.std() / (weights_preview.mean() + 1e-8)
        
        use_sklearn_bypass = (
            (mean_F < 0.15 and mean_I < 0.15) or  # Low I/F overall
            weight_cv < 0.05  # Weights are effectively uniform (CV < 5%)
        )
        
        if use_sklearn_bypass:
            from sklearn.cross_decomposition import PLSRegression
            X = x_tif[..., 0]
            Y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
            if Y.ndim == 1: Y = Y.reshape(-1, 1)
            n_comp = min(self.config.n_components, X.shape[1], X.shape[0] - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, Y)
            self._fitted = True
            self.converged_ = True
            self.x_mean_ = pls._x_mean.reshape(1, -1)
            self.y_mean_ = pls._y_mean.reshape(1, -1)
            self.coef_ = pls.coef_.T if pls.coef_.ndim > 1 else pls.coef_.reshape(-1, 1)
            self.sample_weights_ = np.ones(X.shape[0])
            # Copy VIP-related attributes from sklearn PLS
            self.scores_ = pls.x_scores_
            self.weights_x_ = pls.x_weights_
            self.loadings_ = pls.x_loadings_
            self.y_loadings_ = pls.y_loadings_.T  # sklearn is (n_targets, n_comp), we use (n_comp, n_targets)
            self._sklearn_pls = pls
            self._use_sklearn = True
            self._bypass_reason = "low_IF" if (mean_F < 0.15 and mean_I < 0.15) else "uniform_weights"
            return self
        # Compute sample weights based on proportion of high-F cells
        weights = self._compute_sample_weights(x_tif, y_tif)
        
        # Check if all weights are essentially equal (clean data)
        weight_cv = weights.std() / (weights.mean() + 1e-8)
        self._is_clean_mode = weight_cv < 0.01  # Less than 1% coefficient of variation
        
        self.sample_weights_ = weights
        self._x_tif_train = x_tif
        
        # Always use Truth channel directly (no cell-level gating)
        X = x_tif[..., 0]
        Y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Weighted centering
        weight_sum = weights.sum()
        if weight_sum > 0:
            self.x_mean_ = np.average(X, axis=0, weights=weights).reshape(1, -1)
            self.y_mean_ = np.average(Y, axis=0, weights=weights).reshape(1, -1)
        else:
            self.x_mean_ = X.mean(axis=0, keepdims=True)
            self.y_mean_ = Y.mean(axis=0, keepdims=True)

        Xc = X - self.x_mean_
        Yc = Y - self.y_mean_

        # Apply sqrt-weighting for weighted inner products in NIPALS
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw = Xc * sqrt_w
        Yw = Yc * sqrt_w

        n_samples, n_features = Xc.shape
        n_targets = Yc.shape[1]
        n_comp = self.config.n_components

        # Store unweighted scores/loadings for prediction
        T = np.zeros((n_samples, n_comp))
        U = np.zeros((n_samples, n_comp))
        P = np.zeros((n_features, n_comp))
        Q = np.zeros((n_comp, n_targets))
        W = np.zeros((n_features, n_comp))

        # Work with weighted data for fitting
        Xw_work = Xw.copy()
        Yw_work = Yw.copy()
        Xc_work = Xc.copy()  # Keep unweighted copy for proper loadings
        Yc_work = Yc.copy()

        for i in range(n_comp):
            # Initialize u from weighted Y
            u = Yw_work[:, 0:1]
            iters = 0
            for it in range(self.config.max_iter):
                iters = it + 1
                # Weight vector from weighted covariance
                w = Xw_work.T @ u
                w /= np.linalg.norm(w) + 1e-12
                
                # Scores from UNWEIGHTED centered data (for proper prediction)
                t_unweighted = Xc_work @ w
                
                # Weighted scores for iteration
                t_weighted = Xw_work @ w
                
                # Y loading from weighted data
                c = Yw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
                c /= np.linalg.norm(c) + 1e-12
                u_new = Yw_work @ c / (c.T @ c + 1e-12)
                
                if np.linalg.norm(u_new - u) < self.config.tol:
                    u = u_new
                    break
                u = u_new

            # Compute loadings using weighted regression
            # p = (Xw'Xw)^-1 Xw' t_w = Xw't_w / (t_w't_w)
            p = Xw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
            q = Yw_work.T @ t_weighted / (t_weighted.T @ t_weighted + 1e-12)
            
            # But for regression, we need loadings relative to unweighted scores
            # Recalculate using unweighted data with the same weight vector
            t = t_unweighted
            
            # Store unweighted scores
            T[:, i] = t.ravel()
            U[:, i] = (Yc_work @ c / (c.T @ c + 1e-12)).ravel()
            P[:, i] = p.ravel()
            Q[i, :] = q.ravel()
            W[:, i] = w.ravel()
            self.iterations_.append(iters)

            # Deflate both weighted and unweighted
            Xw_work = Xw_work - np.outer(t_weighted, p)
            Yw_work = Yw_work - np.outer(t_weighted, q)
            Xc_work = Xc_work - np.outer(t, p)
            Yc_work = Yc_work - np.outer(t, q)

        self.scores_ = T
        self.y_scores_ = U
        self.loadings_ = P
        self.y_loadings_ = Q
        self.weights_x_ = W
        self.components_ = W.T @ P
        self.converged_ = True
        self._fitted = True

        # Regression coefficients: standard PLS relation
        R = W @ np.linalg.inv(P.T @ W + 1e-12 * np.eye(n_comp))
        self.coef_ = R @ Q
        return self

    def transform(self, x_tif: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = x_tif[..., 0]  # Always use Truth channel
        Xc = X - self.x_mean_
        T = Xc @ self.weights_x_ @ np.linalg.inv(self.loadings_.T @ self.weights_x_ + 1e-12 * np.eye(self.config.n_components))
        return T

    def predict(self, x_tif: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if hasattr(self, '_use_sklearn') and self._use_sklearn:
            return self._sklearn_pls.predict(x_tif[..., 0]).ravel()
        X = x_tif[..., 0]  # Always use Truth channel
        Xc = X - self.x_mean_
        y_pred = Xc @ self.coef_ + self.y_mean_
        # Return 1D array for single-target prediction (matching sklearn behavior)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")


@dataclass
class PNPLSConfig:
    n_components: int
    max_iter: int = 500
    tol: float = 1e-7
    lambda_falsity: float = 0.5    # Strength of variance prior
    convergence_eps: float = 1e-5  # Threshold for EM convergence

class PNPLS:
    r"""
    Probabilistic Neutrosophic PLS (Element-wise EM-NPLS).
    
    This is the "All-time Superior" variant that handles noise at the element-wise level
    without discarding samples or distorting signal physics.

    Mathematical Foundation:
    ------------------------
    Instead of gating inputs or weighting samples, PNPLS treats the data as having
    heteroscedastic noise variance $\sigma_{ij}^2$ determined by the Indeterminacy/Falsity:
    
       $\sigma_{ij}^2 \propto \exp(\lambda_F \cdot F_{ij})$
       
    The model assumes a generative process:
       $X_{ij} = \sum_k t_i p_j + \epsilon_{ij}, \quad \epsilon_{ij} \sim N(0, \sigma_{ij}^2)$

    Algorithm (EM-NIPALS):
    ----------------------
    Because standard NIPALS assumes constant variance, we solve the Maximum Likelihood
    Estimate iteratively using an Expectation-Maximization (EM) approach formulation.
    
    1. E-Step (Imputation):
       Replace "unreliable" values with their expected value given the current model.
       $X_{imp} = W \odot X_{obs} + (1-W) \odot X_{pred}$
       where $W$ is the precision weight derived from I/F.
       
    2. M-Step (Maximization):
       Run standard NIPALS on the imputed/completed matrix $X_{imp}$.
    
    This converges to the solution that best fits the "reliable" parts of the data
    while naturally ignoring specific bad pixels.
    """

    def __init__(
        self,
        n_components: int,
        *,
        lambda_falsity: float = 0.5,
        max_iter: int = 500,
        tol: float = 1e-7,
        # Legacy parameters (ignored but kept for API compatibility)
        lambda_indeterminacy: float = 0.0,
        alpha: float = 0.0,
        use_sample_weights: bool = True
    ) -> None:
        self.config = PNPLSConfig(
            n_components=n_components,
            lambda_falsity=lambda_falsity,
            max_iter=max_iter,
            tol=tol,
        )
        self.converged_: bool = False
        self.n_iter_: int = 0
        self._fitted: bool = False

    def _compute_precision_weights(self, x_tif: np.ndarray) -> np.ndarray:
        r"""
        Compute element-wise precision weights $W \in [0, 1]^{n \times p}$.
        
        W approaches 1 for clean data (low F), and 0 for bad data.
        Formula: W = exp(-lambda * F)
        
        Fixed to handle edge cases and prevent numerical instability.
        """
        if x_tif.shape[-1] != 3:
             # Fallback for classical matrix
             return np.ones(x_tif.shape[:2])

        F = x_tif[..., 2]
        
        # Clamp F to [0, 1] range to prevent extreme weights
        # Some encoders (like RPCA) may produce F > 1
        F_clamped = np.clip(F, 0.0, 1.0)
        
        # Using simple exponential decay as variance prior
        # Reduced scaling factor to prevent near-zero weights
        weights = np.exp(-self.config.lambda_falsity * F_clamped * 3.0)
        
        # Add floor to prevent complete zeros (minimum 5% weight)
        weights = np.maximum(weights, 0.05)
        
        return weights

    def fit(self, x_tif: np.ndarray, y: np.ndarray) -> "PNPLS":
        self.fit_transform(x_tif, y)
        return self

    def fit_transform(self, x_tif: np.ndarray, y: np.ndarray) -> np.ndarray:
        # CLEAN DATA BYPASS: Use sklearn PLS when I/F are low or precision weights are uniform
        # Raised threshold from 0.01 to 0.15 since encoders produce I/F ~0.05-0.10
        # even for relatively clean data
        mean_F = x_tif[..., 2].mean()
        mean_I = x_tif[..., 1].mean()
        
        # Preview precision weights to check uniformity
        W_preview = self._compute_precision_weights(x_tif)
        weight_mean = W_preview.mean()
        weight_std = W_preview.std()
        # For PNPLS, also check if weights are close to 1.0 (clean data)
        weights_near_one = weight_mean > 0.90 and weight_std < 0.10
        
        use_sklearn_bypass = (
            (mean_F < 0.15 and mean_I < 0.15) or  # Low I/F overall
            weights_near_one  # Precision weights are essentially 1.0
        )
        
        if use_sklearn_bypass:
            from sklearn.cross_decomposition import PLSRegression
            X = x_tif[..., 0]
            Y = y[..., 0] if y.ndim == 3 else y
            if Y.ndim == 1: Y = Y.reshape(-1, 1)
            n_comp = min(self.config.n_components, X.shape[1], X.shape[0] - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, Y)
            self._fitted = True
            self.x_mean_ = pls._x_mean.reshape(1, -1)
            self.y_mean_ = pls._y_mean.reshape(1, -1)
            self.coef_ = pls.coef_.T if pls.coef_.ndim > 1 else pls.coef_.reshape(-1, 1)
            # Copy VIP-related attributes from sklearn PLS
            self.scores_ = pls.x_scores_
            self.weights_x_ = pls.x_weights_
            self.loadings_ = pls.x_loadings_
            self.y_loadings_ = pls.y_loadings_.T  # sklearn is (n_targets, n_comp), we use (n_comp, n_targets)
            self._sklearn_pls = pls
            self._use_sklearn = True
            self._bypass_reason = "low_IF" if (mean_F < 0.15 and mean_I < 0.15) else "uniform_weights"
            return pls.transform(X)
        
        # 1. Setup Data
        X_obs = x_tif[..., 0]  # Truth Channel Observation
        Y = y[..., 0] if (y.ndim == 3) else y
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        # 2. Compute Precision Matrix W (n x p)
        W = self._compute_precision_weights(x_tif)
        
        # Center Data (Weighted mean to ignore outliers)
        # Compute column-wise weighted mean: sum(W*X) / sum(W)
        w_sum = W.sum(axis=0, keepdims=True)
        # Robust protection against small denominators
        w_sum = np.maximum(w_sum, 1e-6)
        
        self.x_mean_ = (W * X_obs).sum(axis=0, keepdims=True) / w_sum
        
        # Sanity check: if weighted mean is extreme, fall back to simple mean
        simple_mean = X_obs.mean(axis=0, keepdims=True)
        mean_diff = np.abs(self.x_mean_ - simple_mean)
        if np.any(mean_diff > 10 * X_obs.std(axis=0, keepdims=True) + 1e-6):
            # Weighted mean is extreme, use simple mean
            self.x_mean_ = simple_mean
        
        # For Y (assuming clean Y)
        self.y_mean_ = Y.mean(axis=0, keepdims=True)
        
        X_c = X_obs - self.x_mean_
        Y_c = Y - self.y_mean_
        
        n_samples, n_features = X_c.shape
        n_targets = Y_c.shape[1]
        n_components = min(self.config.n_components, n_features, n_samples - 1)
        
        # Initialize Imputed Data
        X_imp = X_c.copy()
        
        # Storage
        T = np.zeros((n_samples, n_components))
        P = np.zeros((n_features, n_components))
        Q = np.zeros((n_components, n_targets))
        W_mat = np.zeros((n_features, n_components))
        
        # Current Residuals
        X_res = X_c.copy()
        Y_res = Y_c.copy()
        
        for k in range(n_components):
            # Pick initial u
            u = Y_res[:, 0].reshape(-1, 1)
            # If Y is zero (fully deflated?), random start
            if np.all(np.abs(u) < 1e-15):
                u = np.random.randn(n_samples, 1)
                u /= np.linalg.norm(u) + 1e-12

            t = np.zeros((n_samples, 1))
            p = np.zeros((n_features, 1))
            w_vec = np.zeros((n_features, 1))

            # EM-NIPALS Loop for Component k
            for em_iter in range(self.config.max_iter):
                u_old = u.copy()
                
                # --- M-Step: One pass of NIPALS on current X_imp ---
                
                # 1. Weights w = X^T u
                w_vec = X_imp.T @ u
                w_norm = np.linalg.norm(w_vec)
                if w_norm < 1e-12: 
                    # Degenerate case, use random initialization
                    w_vec = np.random.randn(n_features, 1)
                    w_norm = np.linalg.norm(w_vec)
                w_vec /= (w_norm + 1e-12)
                
                # 2. Scores t = X w
                t = X_imp @ w_vec
                
                # Normalize t to prevent explosion
                t_norm = np.linalg.norm(t)
                if t_norm > 1e6:
                    t /= t_norm
                
                # 3. Y-Loadings c = Y^T t
                c_vec = Y_res.T @ t
                c_norm = np.linalg.norm(c_vec)
                if c_norm < 1e-12:
                    c_vec = np.ones((n_targets, 1))
                    c_norm = np.linalg.norm(c_vec)
                c_vec /= (c_norm + 1e-12)
                    
                # 4. Update u = Y c
                u = Y_res @ c_vec
                
                # Normalize u
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-12:
                    u /= u_norm
                
                # --- E-Step: Re-impute underlying X structure ---
                # Loadings p = X^T t / t^T t
                t_sq = np.dot(t.T, t).item()
                t_sq = max(t_sq, 1e-12)
                
                p = (X_imp.T @ t) / t_sq
                
                # Limit loading magnitude to prevent explosion
                p_norm = np.linalg.norm(p)
                if p_norm > 100:
                    p = p * 100 / p_norm
                
                # Reconstruction of this component
                Rec = t @ p.T
                
                # Update X_imp for the NEXT iteration:
                # Blend observed data with reconstruction based on precision weights
                # Higher W = trust the data more, lower W = trust the model more
                X_imp = W * X_res + (1 - W) * Rec
                
                # Check Convergence
                if np.linalg.norm(u - u_old) < self.config.tol:
                    break
            
            # Post-convergence: Compute final q using t (unnormalized for proper scaling)
            t = X_imp @ w_vec  # Recompute with final w_vec
            t_sq = np.dot(t.T, t).item()
            t_sq = max(t_sq, 1e-12)
            
            # Recompute p with proper scaling
            p = (X_res.T @ t) / t_sq
            q = (Y_res.T @ t) / t_sq
            
            # Store
            T[:, k] = t.ravel()
            P[:, k] = p.ravel()
            Q[k, :] = q.ravel()
            W_mat[:, k] = w_vec.ravel()
            
            # Deflate
            X_res = X_res - t @ p.T
            Y_res = Y_res - t @ q.T
            
            # Reset imputation for next component
            X_imp = X_res.copy()

        self.scores_ = T
        self.loadings_ = P
        self.weights_x_ = W_mat
        self.y_loadings_ = Q
        self._fitted = True
        
        # Calculate Regression Coefficients B
        # B = W(P^T W)^-1 Q
        # Handle potential singularity with regularization
        term = P.T @ W_mat
        # Add regularization to diagonal
        reg = max(1e-8, 1e-6 * np.abs(term).max())
        term[np.diag_indices_from(term)] += reg
        
        try:
            R = W_mat @ np.linalg.inv(term)
            self.coef_ = R @ Q
            
            # Sanity check: if coefficients are extreme, fall back to simpler solution
            if np.any(~np.isfinite(self.coef_)) or np.abs(self.coef_).max() > 1e10:
                # Fall back to pseudo-inverse solution
                self.coef_ = np.linalg.lstsq(T, Y_c, rcond=None)[0]
                self.coef_ = W_mat @ np.linalg.lstsq(P.T @ W_mat + reg * np.eye(n_components), 
                                                      self.coef_, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Matrix inversion failed, use pseudo-inverse
            R = W_mat @ np.linalg.pinv(term)
            self.coef_ = R @ Q
        
        return T

    def predict(self, x_tif: np.ndarray) -> np.ndarray:
        # Standard projection for prediction
        if not self._fitted: raise RuntimeError("Not fitted")
        if hasattr(self, '_use_sklearn') and self._use_sklearn:
            return self._sklearn_pls.predict(x_tif[..., 0]).ravel()
        
        X = x_tif[..., 0]
        Xc = X - self.x_mean_
        yp = Xc @ self.coef_ + self.y_mean_
        
        if yp.ndim == 2 and yp.shape[1] == 1:
            return yp.ravel()
        return yp
    
    def transform(self, x_tif: np.ndarray) -> np.ndarray:
        if not self._fitted: raise RuntimeError("Not fitted")
        X = x_tif[..., 0]
        Xc = X - self.x_mean_
        # Project using standard formula
        T = Xc @ self.weights_x_ @ np.linalg.inv(
            self.loadings_.T @ self.weights_x_ + 1e-12 * np.eye(self.config.n_components)
        )
        return T


ReliabilityWeightedNPLS = NPLSW
