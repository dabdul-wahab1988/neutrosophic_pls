"""Semantic projection utilities for neutrosophic VIP/loads."""

from typing import List, Dict, Sequence
import numpy as np

from .vip import compute_nvip


def semantic_projection(
    model,
    x_tif: np.ndarray,
    feature_names: Sequence[str] | None = None,
    top_k: int = 10,
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> List[Dict]:
    """
    Produce ranked semantic summaries using VIP scores.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(x_tif.shape[1])]
    vips = compute_nvip(model, x_tif, channel_weights=channel_weights)
    aggregate = vips["aggregate"]
    order = np.argsort(aggregate)[::-1][:top_k]
    summaries = []
    for idx in order:
        summaries.append(
            {
                "feature": feature_names[idx],
                "vip": float(aggregate[idx]),
                "vip_T": float(vips["T"][idx]),
                "vip_I": float(vips["I"][idx]),
                "vip_F": float(vips["F"][idx]),
                "interpretation": f"{feature_names[idx]} is influential across T/I/F with VIP={aggregate[idx]:.3f}.",
            }
        )
    return summaries
