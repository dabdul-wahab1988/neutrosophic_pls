"""Pipeline configuration utilities."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class PipelineConfig:
    mode: str = "simulate"  # simulate | micromass
    n_samples: int = 50
    n_features: int = 20
    n_components: int = 2
    model_type: str = "npls"  # npls | nplsw | pnpls
    lambda_indeterminacy: float = 1.0  # for nplsw/pnpls
    lambda_falsity: float = 0.0  # for pnpls
    alpha: float = 1.0  # for PNPLS falsity prior softness
    weight_normalize: str = "mean1"  # for nplsw
    seed: int = 0
    output_dir: str = "artifacts"
    micromass_path: Optional[str] = None
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = PipelineConfig(**data)
    return cfg


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f)
