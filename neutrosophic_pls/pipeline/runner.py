"""Pipeline runner for CLI and notebook."""

from pathlib import Path
from typing import Dict, Any
import numpy as np

from ..simulate import generate_simulation
from ..data_micromass import load_micromass
from ..model_factory import create_model_from_params
from ..metrics import evaluation_metrics
from ..vip import compute_nvip
from ..semantic import semantic_projection
from ..algebra import combine_channels
from .config import PipelineConfig


def run(cfg: PipelineConfig) -> Dict[str, Any]:
    if cfg.mode == "simulate":
        x_tif, y_tif, meta = generate_simulation(
            cfg.n_samples, cfg.n_features, cfg.n_components, seed=cfg.seed
        )
        feature_names = [f"f{i}" for i in range(cfg.n_features)]
    elif cfg.mode == "micromass":
        loaded = load_micromass(cfg.micromass_path)
        x_tif, y_tif = loaded["x_tif"], loaded["y_tif"]
        meta = loaded["metadata"]
        feature_names = meta["feature_names"]
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    model = _build_model(cfg)
    model.fit(x_tif, y_tif)
    preds = model.predict(x_tif)
    metrics = evaluation_metrics(combine_channels(y_tif, cfg.channel_weights), preds)
    vips = compute_nvip(model, x_tif, channel_weights=cfg.channel_weights)
    semantics = semantic_projection(model, x_tif, feature_names=feature_names, channel_weights=cfg.channel_weights)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_report(out_dir / "metrics.txt", metrics)
    _write_report(out_dir / "vip.txt", {k: v.tolist() for k, v in vips.items()})
    return {
        "model": model,
        "metrics": metrics,
        "vips": vips,
        "semantics": semantics,
        "metadata": meta,
        "output_dir": str(out_dir),
    }


def _build_model(cfg: PipelineConfig):
    return create_model_from_params(
        method=cfg.model_type,
        n_components=cfg.n_components,
        lambda_indeterminacy=cfg.lambda_indeterminacy,
        lambda_falsity=cfg.lambda_falsity,
        alpha=cfg.alpha,
        channel_weights=cfg.channel_weights,
        normalize=cfg.weight_normalize
    )


def _write_report(path: Path, content: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key, val in content.items():
            f.write(f"{key}: {val}\n")
