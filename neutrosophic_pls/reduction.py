"""Reduction checks comparing N-PLS outputs to classical PLS."""

from typing import Dict, Tuple
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from .algebra import combine_channels
from .model import NPLS


def check_reduction(
    x_tif: np.ndarray,
    y_tif: np.ndarray,
    n_components: int,
    *,
    tol: float = 1e-6,
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compare N-PLS to classical PLS when I=F=0; returns max deltas for scores and predictions.
    """
    x_mat = combine_channels(x_tif, channel_weights)
    y_mat = combine_channels(y_tif, channel_weights)

    npls = NPLS(n_components=n_components, channel_weights=channel_weights).fit(x_tif, y_tif)
    pls = PLSRegression(n_components=n_components, scale=False).fit(x_mat, y_mat)

    scores_delta = float(np.max(np.abs(npls.scores_ - pls.x_scores_)))
    pred_delta = float(np.max(np.abs(npls.predict(x_tif) - pls.predict(x_mat))))

    result = {"scores_delta": scores_delta, "pred_delta": pred_delta, "passed": scores_delta <= tol and pred_delta <= tol}
    if not result["passed"]:
        raise AssertionError(f"Reduction check failed: scores_delta={scores_delta}, pred_delta={pred_delta}, tol={tol}")
    return result
