"""Notebook helper wrapping the pipeline runner."""

from typing import Dict, Any
from .pipeline.config import PipelineConfig, load_config
from .pipeline.runner import run


def run_pipeline(config: Dict[str, Any] | None = None):
    if config is None:
        cfg = PipelineConfig()
    elif isinstance(config, dict):
        cfg = PipelineConfig(**config)
    else:
        cfg = load_config(config)
    return run(cfg)
