"""Evaluation utilities for chemometric calibration models.

This module provides local implementations of standard regression metrics
without relying on sklearn, plus additional metrics commonly used in
NIR spectroscopy and chemometrics literature.

Metrics included:
    - RMSEP: Root Mean Square Error of Prediction
    - R2: Coefficient of Determination
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - SEP: Standard Error of Prediction
    - Bias: Systematic prediction offset
    - RPD: Ratio of Performance to Deviation
    - RER: Ratio to Error Range
"""

from typing import Dict, Optional
import numpy as np


# =============================================================================
# Core Metric Functions (Local Implementations)
# =============================================================================

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error.
    
    $$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Mean squared error.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Coefficient of Determination (R²).
    
    $$R^2 = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}$$
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        R² score. Returns 0.0 if total variance is zero.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.
    
    $$\\text{MAE} = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Mean absolute error.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmsep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error of Prediction.
    
    $$\\text{RMSEP} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Root mean square error of prediction.
    """
    return float(np.sqrt(_mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute Mean Absolute Percentage Error.
    
    $$\\text{MAPE} = \\frac{100}{n}\\sum_{i=1}^{n}\\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$
    
    Standard metric for concentration prediction; interpretable as average
    percentage deviation from true values.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, optional
        Small constant to prevent division by zero. Default 1e-10.
    
    Returns
    -------
    float
        Mean absolute percentage error (in %).
    
    Notes
    -----
    Values near zero in y_true can inflate MAPE; epsilon provides stability.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute prediction bias (systematic offset).
    
    $$\\text{Bias} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)$$
    
    Critical for calibration transfer and detecting systematic model errors.
    Positive bias indicates underprediction; negative indicates overprediction.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Mean prediction bias.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true - y_pred))


def _sep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Standard Error of Prediction.
    
    $$\\text{SEP} = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(e_i - \\bar{e})^2}$$
    
    where $e_i = y_i - \\hat{y}_i$ is the prediction error.
    
    SEP measures random error component separately from bias, providing
    a cleaner assessment of model precision.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Standard error of prediction.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    errors = y_true - y_pred
    n = len(errors)
    if n <= 1:
        return 0.0
    return float(np.sqrt(np.sum((errors - np.mean(errors)) ** 2) / (n - 1)))


def _rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Ratio of Performance to Deviation.
    
    $$\\text{RPD} = \\frac{\\text{SD}(y)}{\\text{SEP}}$$
    
    Standard chemometrics metric for model utility classification:
        - RPD < 1.5: Poor, not usable
        - 1.5 ≤ RPD < 2.0: Marginal, rough screening
        - 2.0 ≤ RPD < 2.5: Moderate, approximate quantification
        - 2.5 ≤ RPD < 3.0: Good, screening
        - RPD ≥ 3.0: Excellent, quality control
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Ratio of performance to deviation.
    """
    y_true = np.asarray(y_true).ravel()
    sep = _sep(y_true, y_pred)
    if sep < 1e-12:
        return float('inf')
    sd_y = float(np.std(y_true, ddof=1))
    return sd_y / sep


def _rer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Ratio to Error Range.
    
    $$\\text{RER} = \\frac{\\text{range}(y)}{\\text{SEP}}$$
    
    Alternative to RPD using range instead of standard deviation.
    Less sensitive to reference value distribution shape.
    
    Interpretation guidelines:
        - RER < 4: Poor
        - 4 ≤ RER < 8: Marginal
        - 8 ≤ RER < 12: Acceptable
        - RER ≥ 12: Excellent
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    float
        Ratio to error range.
    """
    y_true = np.asarray(y_true).ravel()
    sep = _sep(y_true, y_pred)
    if sep < 1e-12:
        return float('inf')
    range_y = float(np.max(y_true) - np.min(y_true))
    return range_y / sep


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_extended: bool = True
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for calibration models.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    include_extended : bool, optional
        If True, include extended chemometric metrics (MAPE, SEP, Bias, RPD, RER).
        Default True.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - RMSEP: Root Mean Square Error of Prediction
        - R2: Coefficient of Determination
        - MAE: Mean Absolute Error
        
        If include_extended=True, also includes:
        - MAPE: Mean Absolute Percentage Error (%)
        - SEP: Standard Error of Prediction
        - Bias: Systematic prediction offset
        - RPD: Ratio of Performance to Deviation
        - RER: Ratio to Error Range
    
    Examples
    --------
    >>> y_true = np.array([10.0, 12.0, 14.0, 11.0, 13.0])
    >>> y_pred = np.array([10.2, 11.8, 14.1, 10.9, 13.2])
    >>> metrics = evaluation_metrics(y_true, y_pred)
    >>> print(f"RMSEP: {metrics['RMSEP']:.3f}, RPD: {metrics['RPD']:.2f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Core metrics (always computed)
    results = {
        "RMSEP": _rmsep(y_true, y_pred),
        "R2": _r2_score(y_true, y_pred),
        "MAE": _mean_absolute_error(y_true, y_pred),
    }
    
    # Extended chemometric metrics
    if include_extended:
        results.update({
            "MAPE": _mape(y_true, y_pred),
            "SEP": _sep(y_true, y_pred),
            "Bias": _bias(y_true, y_pred),
            "RPD": _rpd(y_true, y_pred),
            "RER": _rer(y_true, y_pred),
        })
    
    return results


# =============================================================================
# Individual Metric Accessors (for convenience)
# =============================================================================

def rmsep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSEP. See _rmsep for details."""
    return _rmsep(y_true, y_pred)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score. See _r2_score for details."""
    return _r2_score(y_true, y_pred)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE. See _mean_absolute_error for details."""
    return _mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute MAPE. See _mape for details."""
    return _mape(y_true, y_pred, epsilon)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Bias. See _bias for details."""
    return _bias(y_true, y_pred)


def sep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute SEP. See _sep for details."""
    return _sep(y_true, y_pred)


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RPD. See _rpd for details."""
    return _rpd(y_true, y_pred)


def rer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RER. See _rer for details."""
    return _rer(y_true, y_pred)


def component_recovery(true_components: np.ndarray, estimated_components: np.ndarray) -> Dict[str, float]:
    """
    Compare true vs estimated component spaces using correlation.
    """
    true_norm = true_components / (np.linalg.norm(true_components, axis=0, keepdims=True) + 1e-12)
    est_norm = estimated_components / (np.linalg.norm(estimated_components, axis=0, keepdims=True) + 1e-12)
    corr = np.abs((true_norm * est_norm).sum(axis=0))
    return {"mean_corr": float(np.mean(corr)), "min_corr": float(np.min(corr))}
