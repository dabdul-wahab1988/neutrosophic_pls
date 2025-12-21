#!/usr/bin/env python
"""
Run all manuscript experiments, figures, and tables with one command.
"""

from pathlib import Path
from typing import Tuple

from neutrosophic_pls.manuscript.experiments import (
    run_realdata_cv,
    run_runtime_scaling,
    run_simulation_robustness,
    run_sensitivity_sweep,
    run_corruption_robustness,
)
from neutrosophic_pls.manuscript.figures import make_all_figures
from neutrosophic_pls.manuscript.tables import make_all_tables
from neutrosophic_pls.manuscript.utils import ensure_dir


DATA_PATHS = [
    r"C:\Users\DicksonAbdul-Wahab\Desktop\NeutroProject\Finalised_Projects\formanuscript\data\A3.csv",
    r"C:\Users\DicksonAbdul-Wahab\Desktop\NeutroProject\Finalised_Projects\formanuscript\data\B1.csv",
    r"C:\Users\DicksonAbdul-Wahab\Desktop\NeutroProject\Finalised_Projects\formanuscript\data\MA_A2.csv",
    r"C:\Users\DicksonAbdul-Wahab\Desktop\NeutroProject\Finalised_Projects\formanuscript\data\MB_B2.csv",
]

# Default hyperparameters for reproducibility
MAX_COMPONENTS = 15
CV_FOLDS = 5
REPEATS = 3
CHANNEL_WEIGHTS: Tuple[float, float, float] = (1.0, 1.0, 1.0)
LAMBDA_I = 1.0
LAMBDA_F = 1.0
ALPHA = 1.0


def _resolve_dataset_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    # Handle Windows-style absolute paths when running under WSL/Posix
    if raw_path[1:3] == ":\\" or raw_path[1:3] == ":/":
        drive_letter = raw_path[0].lower()
        unix_path = Path(f"/mnt/{drive_letter}") / raw_path[3:].replace("\\", "/")
        if unix_path.exists():
            return unix_path

    # Fallback to local data directory by basename
    alt = project_root / "data" / Path(raw_path.replace("\\", "/")).name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Dataset missing: {raw_path} (also tried {alt})")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiments_root = ensure_dir(project_root / "manuscript_artifacts" / "experiments")
    figures_root = ensure_dir(project_root / "manuscript_artifacts" / "figures")
    tables_root = ensure_dir(project_root / "manuscript_artifacts" / "tables")

    dataset_paths = [_resolve_dataset_path(p, project_root) for p in DATA_PATHS]

    run_realdata_cv(
        [str(p) for p in dataset_paths],
        experiments_root,
        cv_folds=CV_FOLDS,
        repeats=REPEATS,
        max_components=MAX_COMPONENTS,
        channel_weights=CHANNEL_WEIGHTS,
        lambda_I=LAMBDA_I,
        lambda_F=LAMBDA_F,
        alpha=ALPHA,
    )

    # Sensitivity on MA_A2 (protein) by default
    run_sensitivity_sweep(
        dataset_paths[2],
        experiments_root / "sensitivity",
        cv_folds=3,
        max_components=min(10, MAX_COMPONENTS),
    )

    run_simulation_robustness(experiments_root / "simulation")
    run_runtime_scaling(experiments_root / "runtime")

    # NEW: Spike corruption robustness experiment with significance testing
    run_corruption_robustness(
        [str(p) for p in dataset_paths],
        experiments_root / "corruption",
        corruption_levels=(0.0, 0.15, 0.30),
        cv_folds=CV_FOLDS,
        repeats=REPEATS,
        max_components=MAX_COMPONENTS,
    )

    make_all_tables(
        experiments_root,
        tables_root,
        max_components=MAX_COMPONENTS,
        lambda_I=LAMBDA_I,
        lambda_F=LAMBDA_F,
        alpha=ALPHA,
        channel_weights=CHANNEL_WEIGHTS,
    )
    make_all_figures(experiments_root, figures_root)

    print("Manuscript artifacts generated.")
    print(f"Experiments: {experiments_root}")
    print(f"Figures: {figures_root}")
    print(f"Tables: {tables_root}")


if __name__ == "__main__":
    main()
