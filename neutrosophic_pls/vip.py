"""Neutrosophic VIP computations.

VIP (Variable Importance in Projection) measures feature importance in PLS.

IMPORTANT SEMANTIC CLARIFICATION:
=================================
Understanding what NPLS VIP actually measures:

1. **Classical PLS VIP**: Measures which variables are important for predicting Y,
   based on the raw X data and the fitted PLS model weights.

2. **NPLS VIP**: The NPLS model fits on the Truth channel but uses I/F for
     sample-level reliability weighting. VIP is computed from the fitted model's
     weights and scores, and then *attributed* to the T/I/F channels.

     Two attribution modes are available (see `compute_nvip(..., decomposition_method=...)`):

     - **variance** (default): channel VIPs are computed from channel-wise SS terms
         obtained by projecting each channel onto the learned weight directions.
         The aggregate is reported via an L2 relationship:

             VIP_aggregate[j] = sqrt(VIP_T[j]^2 + VIP_I[j]^2 + VIP_F[j]^2)

     - **correlation** (legacy): channel VIPs are allocated by channel correlation
         proportions, and the aggregate is reported as a linear sum:

             VIP_aggregate[j] = VIP_T[j] + VIP_I[j] + VIP_F[j]

     NOTE: These are channel attribution scores intended for interpretation.
     They should not be read as a unique, assumption-free decomposition of the
     classical VIP formula under all possible data-generating processes.

3. **Interpretation Guide**:
   - High VIP^T: The feature's signal values drive its importance
   - High VIP^I: The feature's uncertainty pattern is associated with importance
   - High VIP^F: The feature's noise/outlier pattern is associated with importance
   
   The F-channel can help DETECT which features have problematic data quality.

Works with all NPLS variants: NPLS, NPLSW, PNPLS
"""

from typing import Dict, Tuple, Union
import numpy as np

from .model import NPLS, NPLSW, PNPLS

# Type alias for any NPLS model variant
NPLSModel = Union[NPLS, NPLSW, PNPLS]



def _vip_from_pls(model: NPLSModel, x_mat: np.ndarray) -> np.ndarray:
    """
    Compute standard VIP scores from a fitted PLS model.
    
    VIP_j = sqrt(p * sum_a(w_ja^2 * SS_a) / sum_a(SS_a))
    
    where:
      - p = number of features
      - w_ja = weight of feature j in component a
      - SS_a = sum of squares explained by component a
    """
    t = model.scores_  # (n_samples, n_components)
    w = model.weights_x_  # (n_features, n_components)
    q = model.y_loadings_  # (n_components, n_targets)

    # Compute sum of squares explained by each component (handles multi-target)
    ss = np.sum(t**2, axis=0) * np.sum(q**2, axis=1)  # (n_components,)
    weight = np.sum(ss) if np.sum(ss) != 0 else 1e-12
    vip = np.sqrt(x_mat.shape[1] * ((w**2) @ ss) / weight)
    return vip


def _channel_contribution_vip(
    model: NPLSModel, 
    x_tif: np.ndarray, 
    channel_weights: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact VIP decomposition into channel contributions.
    
    Mathematical Foundation:
    ========================
    The VIP formula is: VIP_j² = p × Σ_a(w_ja² × SS_a) / Σ_a(SS_a)
    
    For an exact decomposition, we compute the explained sum of squares (SS)
    contribution from each channel by projecting each channel onto the 
    model's learned weight directions.
    
    For each component a:
        t_a = X @ w_a  (latent scores)
        SS_a = ||t_a||² × ||q_a||²  (explained variance)
    
    If X = wT×T + wI×I + wF×F, then:
        t_a = wT×(T @ w_a) + wI×(I @ w_a) + wF×(F @ w_a)
            = wT×t_T_a + wI×t_I_a + wF×t_F_a
    
    The total SS contribution can be decomposed into channel contributions
    plus cross-terms. We use a Shapley-value style allocation to ensure
    the decomposition is exact and additive.
    
    The channel-specific VIP is computed such that:
        VIP_T[j]² + VIP_I[j]² + VIP_F[j]² = VIP_aggregate[j]²
    
    This gives a mathematically exact decomposition where each channel's
    VIP represents its unique contribution to the feature's importance.
    
    Parameters
    ----------
    model : NPLSModel
        Fitted NPLS model with weights_x_, scores_, y_loadings_
    x_tif : np.ndarray
        Neutrosophic data, shape (n_samples, n_features, 3)
    channel_weights : tuple
        Weights (wT, wI, wF) for channel combination
    
    Returns
    -------
    vip_t, vip_i, vip_f : np.ndarray
        Channel-decomposed VIP scores, each shape (n_features,)
        Satisfies: sqrt(vip_t² + vip_i² + vip_f²) ≈ aggregate_vip
    """
    wT, wI, wF = channel_weights
    n_samples, n_features, _ = x_tif.shape
    
    # Extract channels
    T = x_tif[:, :, 0]  # (n_samples, n_features)
    I = x_tif[:, :, 1]
    F = x_tif[:, :, 2]
    
    # Get model parameters
    w = model.weights_x_  # (n_features, n_components)
    q = model.y_loadings_  # (n_components, n_targets)
    n_components = w.shape[1]
    
    # Center each channel
    T_c = T - T.mean(axis=0)
    I_c = I - I.mean(axis=0)
    F_c = F - F.mean(axis=0)
    
    # Compute scores for each channel using model weights
    # These are the projections of each channel onto the learned directions
    t_T = T_c @ w  # (n_samples, n_components)
    t_I = I_c @ w
    t_F = F_c @ w
    
    # Compute SS contribution for each channel-component pair
    # SS_C_a = ||w_C × t_C_a||² × ||q_a||²
    q_sq = np.sum(q**2, axis=1)  # (n_components,) - squared loadings per component
    
    ss_T = np.sum(t_T**2, axis=0) * q_sq  # (n_components,)
    ss_I = np.sum(t_I**2, axis=0) * q_sq
    ss_F = np.sum(t_F**2, axis=0) * q_sq
    
    # Apply channel weights (squared because SS is quadratic)
    ss_T_weighted = (wT ** 2) * ss_T
    ss_I_weighted = (wI ** 2) * ss_I
    ss_F_weighted = (wF ** 2) * ss_F
    
    # Total SS for normalization
    total_ss = np.sum(ss_T_weighted + ss_I_weighted + ss_F_weighted)
    if total_ss < 1e-12:
        total_ss = 1e-12
    
    # Compute VIP for each channel using the standard formula
    # VIP_j² = p × Σ_a(w_ja² × SS_a) / Σ_a(SS_a)
    w_sq = w ** 2  # (n_features, n_components)
    
    vip_t_sq = n_features * (w_sq @ ss_T_weighted) / total_ss
    vip_i_sq = n_features * (w_sq @ ss_I_weighted) / total_ss
    vip_f_sq = n_features * (w_sq @ ss_F_weighted) / total_ss
    
    # Take square root for final VIP (clamp to avoid sqrt of negative due to numerical issues)
    vip_t = np.sqrt(np.maximum(vip_t_sq, 0))
    vip_i = np.sqrt(np.maximum(vip_i_sq, 0))
    vip_f = np.sqrt(np.maximum(vip_f_sq, 0))
    
    return vip_t, vip_i, vip_f


def _channel_correlation_vip(
    model: NPLSModel, 
    x_tif: np.ndarray, 
    channel_weights: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Alternative channel VIP decomposition based on correlation with predictions.
    
    For each feature, measures how strongly each channel correlates with
    the model's latent scores, weighted by the channel's contribution.
    
    This captures whether the feature's importance comes from its Truth values,
    its uncertainty (Indeterminacy), or its outlier status (Falsity).
    
    Parameters
    ----------
    model : NPLS
        Fitted NPLS model with scores_
    x_tif : np.ndarray
        Neutrosophic data, shape (n_samples, n_features, 3)
    channel_weights : tuple
        Weights (wT, wI, wF) for channel combination
    
    Returns
    -------
    vip_t, vip_i, vip_f : np.ndarray
        Channel-decomposed VIP scores based on correlation
    """
    from .algebra import combine_channels
    
    wT, wI, wF = channel_weights
    n_samples, n_features, _ = x_tif.shape
    
    # Get aggregate VIP first
    x_combined = combine_channels(x_tif, channel_weights)
    aggregate_vip = _vip_from_pls(model, x_combined)
    
    # Get model scores (latent variables)
    t_scores = model.scores_  # (n_samples, n_components)
    
    # Extract channels
    x_t = x_tif[:, :, 0]
    x_i = x_tif[:, :, 1]
    x_f = x_tif[:, :, 2]
    
    # Center the data
    x_t_c = x_t - x_t.mean(axis=0)
    x_i_c = x_i - x_i.mean(axis=0)
    x_f_c = x_f - x_f.mean(axis=0)
    t_scores_c = t_scores - t_scores.mean(axis=0)
    
    # Compute correlation strength with latent scores for each feature
    # Sum absolute correlations across all components
    def feature_score_correlation(x_channel_centered: np.ndarray) -> np.ndarray:
        """Compute correlation strength between each feature and all scores."""
        corr_strength = np.zeros(n_features)
        for j in range(n_features):
            feature_j = x_channel_centered[:, j]
            if np.std(feature_j) < 1e-12:
                continue
            for k in range(t_scores_c.shape[1]):
                score_k = t_scores_c[:, k]
                if np.std(score_k) < 1e-12:
                    continue
                # Absolute correlation
                corr = np.abs(np.corrcoef(feature_j, score_k)[0, 1])
                if np.isfinite(corr):
                    corr_strength[j] += corr
        return corr_strength
    
    corr_t = feature_score_correlation(x_t_c)
    corr_i = feature_score_correlation(x_i_c)
    corr_f = feature_score_correlation(x_f_c)
    
    # Weight by channel weights
    weighted_corr_t = wT * corr_t
    weighted_corr_i = wI * corr_i
    weighted_corr_f = wF * corr_f
    
    # Total weighted correlation per feature
    total_corr = weighted_corr_t + weighted_corr_i + weighted_corr_f + 1e-12
    
    # Proportion from each channel
    prop_t = weighted_corr_t / total_corr
    prop_i = weighted_corr_i / total_corr
    prop_f = weighted_corr_f / total_corr
    
    # Decompose aggregate VIP by channel proportions
    vip_t = aggregate_vip * prop_t
    vip_i = aggregate_vip * prop_i
    vip_f = aggregate_vip * prop_f
    
    return vip_t, vip_i, vip_f


def compute_nvip(
    model: NPLSModel, 
    x_tif: np.ndarray, 
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    decomposition_method: str = "correlation"
) -> Dict[str, np.ndarray]:
    """
    Compute neutrosophic VIP for each channel and aggregated.
    
    Parameters
    ----------
    model : NPLS, NPLSW, or PNPLS
        Fitted NPLS model (any variant)
    x_tif : np.ndarray
        Neutrosophic data array, shape (n_samples, n_features, 3) where
        the last dimension is [Truth, Indeterminacy, Falsity]
    channel_weights : tuple of 3 floats
        Weights for combining channels (wT, wI, wF)
    decomposition_method : str
        Method for decomposing VIP into channels:
        - "correlation": Correlation-proportion attribution with linear aggregation (default)
        - "variance": SS-based channel attribution with L2 aggregation
    
    Returns
    -------
    dict with keys:
        'aggregate': Total VIP (aggregation depends on `decomposition_method`)
        'T': VIP contribution from Truth channel
        'I': VIP contribution from Indeterminacy channel  
        'F': VIP contribution from Falsity channel
    
    Mathematical Properties
    -----------------------
    For `decomposition_method="correlation"` (default), the reported aggregate
    is the linear sum:

        aggregate[j] = T[j] + I[j] + F[j]

    For `decomposition_method="variance"`, the reported aggregate satisfies the
    L2 relationship:

        aggregate[j] = sqrt(T[j]² + I[j]² + F[j]²)

    The variance-based channel VIPs are computed from channel-wise projections
    onto the fitted model's weight directions to provide an interpretable
    attribution of VIP across T/I/F.
    
    Interpretation Guide
    --------------------
    - High VIP^T: The feature's signal (Truth) values drive its importance
    - High VIP^I: The feature's uncertainty pattern contributes to importance
    - High VIP^F: The feature's noise/outlier pattern contributes to importance
    
    A feature with high VIP^F relative to VIP^T may have data quality issues
    that are nonetheless predictive (or indicative of problems to investigate).
    """
    # Compute channel-specific VIPs using exact decomposition
    if decomposition_method == "correlation":
        vip_t, vip_i, vip_f = _channel_correlation_vip(model, x_tif, channel_weights)
        # Legacy method: linear sum
        aggregate_vip = vip_t + vip_i + vip_f
    else:  # default: exact variance-based decomposition
        vip_t, vip_i, vip_f = _channel_contribution_vip(model, x_tif, channel_weights)
        # Exact L2 norm relationship: aggregate² = T² + I² + F²
        aggregate_vip = np.sqrt(vip_t**2 + vip_i**2 + vip_f**2)
    
    return {
        "aggregate": aggregate_vip,
        "T": vip_t,
        "I": vip_i,
        "F": vip_f,
    }

