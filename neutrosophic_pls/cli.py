"""CLI entrypoint for npls-pipeline."""

import typer
from pathlib import Path
from typing import Optional

from .pipeline.config import load_config, save_config, PipelineConfig
from .pipeline.runner import run

app = typer.Typer(add_completion=False, help="Neutrosophic PLS pipeline")


@app.command()
def wizard(
    output: str = typer.Option("pipeline_config.yaml", help="Path to write generated config."),
    mode: str = typer.Option("simulate", help="simulate or micromass"),
    n_samples: int = typer.Option(50, help="Samples for simulation mode."),
    n_features: int = typer.Option(20, help="Features for simulation mode."),
    n_components: int = typer.Option(2, help="Number of latent components."),
    model_type: str = typer.Option("npls", help="Model: npls | nplsw | pnpls"),
    lambda_indeterminacy: float = typer.Option(1.0, help="For nplsw/pnpls"),
    lambda_falsity: float = typer.Option(0.0, help="For pnpls"),
    alpha: float = typer.Option(1.0, help="For PNPLS falsity prior softness (>0)"),
    weight_normalize: str = typer.Option("mean1", help="For nplsw: none|mean1|sum1"),
    seed: int = typer.Option(0, help="Random seed."),
):
    cfg = PipelineConfig(
        mode=mode,
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        model_type=model_type,
        lambda_indeterminacy=lambda_indeterminacy,
        lambda_falsity=lambda_falsity,
        alpha=alpha,
        weight_normalize=weight_normalize,
        seed=seed,
    )
    save_config(cfg, output)
    typer.echo(f"Config written to {output}")


@app.command()
def run_pipeline(
    config: Optional[str] = typer.Option(None, help="YAML config path."),
    output_dir: str = typer.Option("artifacts", help="Output directory for reports."),
):
    cfg = load_config(config)
    cfg.output_dir = output_dir
    result = run(cfg)
    typer.echo(f"Pipeline completed. Metrics: {result['metrics']}")
    typer.echo(f"Artifacts in {result['output_dir']}")


def main():
    app()


if __name__ == "__main__":
    main()
