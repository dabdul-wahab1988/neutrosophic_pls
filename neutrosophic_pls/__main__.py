"""
Neutrosophic PLS - Command Line Interface
==========================================

Run N-PLS studies from the command line with:
- Interactive mode for guided setup
- Config file mode for reproducible studies
- Direct mode for quick runs

Usage:
    python -m neutrosophic_pls --interactive
    python -m neutrosophic_pls --config study.yaml
    python -m neutrosophic_pls --data data.csv --target y --task regression

Author: NeutroProject
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

from .metrics import evaluation_metrics as compute_metrics
from .data_loader import (
    DatasetConfig,
    load_dataset,
    interactive_load_dataset,
    list_available_datasets,
)
from .study_config import (
    StudyConfig,
    DatasetSettings,
    ModelSettings,
    EvaluationSettings,
    OutputSettings,
    interactive_build_config,
    get_idrc_wheat_config,
)

from .model_factory import create_model
from .vip import compute_nvip
from .algebra import combine_channels
from .manuscript.utils import select_components_cv
from .interactive import run_interactive_session


def print_banner():
    """Print application banner."""
    print("""
===============================================================
         Neutrosophic Partial Least Squares (N-PLS)            
                                                               
   Uncertainty-aware PLS with Truth/Indeterminacy/Falsity      
===============================================================
    """)


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    """Compute evaluation metrics based on task type."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if task == "regression":
        # Use local metrics module (no sklearn dependency)
        return compute_metrics(y_true, y_pred, include_extended=True)
    else:  # classification
        y_pred_class = np.round(y_pred).astype(int)
        accuracy = float(np.mean(y_true.astype(int) == y_pred_class))
        return {
            "Accuracy": accuracy,
            "Error_Rate": 1.0 - accuracy,
        }


def run_study(config: StudyConfig) -> Dict[str, Any]:
    """
    Run a complete N-PLS study based on configuration.
    
    Parameters
    ----------
    config : StudyConfig
        Study configuration
        
    Returns
    -------
    dict with results, summary, and metadata
    """
    print(f"\n{'='*60}")
    print(f"Running Study: {config.name}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_config = DatasetConfig(
        path=config.dataset.path,
        target=config.dataset.target,
        task=config.dataset.task,
        features=config.dataset.features,
        exclude_columns=config.dataset.exclude_columns,
        snv=config.dataset.snv,
        format=config.dataset.format,
        name=config.dataset.name,
    )
    data = load_dataset(dataset_config)
    
    x_tif = data["x_tif"]
    y_tif = data["y_tif"]
    metadata = data["metadata"]
    
    print(f"  Dataset: {metadata['name']}")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Task: {metadata['task']}")
    
    # Determine which methods to run
    if config.model.method == "all":
        methods = ["PLS", "NPLS", "NPLSW", "PNPLS"]
    else:
        methods = [config.model.method]
    
    print(f"\nMethods: {', '.join(methods)}")
    print(f"CV: {config.evaluation.repeats} repeats × {config.evaluation.cv_folds} folds")
    
    # Run cross-validation
    results = []
    vip_results = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for repeat in range(config.evaluation.repeats):
            print(f"\n  Repeat {repeat + 1}/{config.evaluation.repeats}")
            
            outer_cv = KFold(
                n_splits=config.evaluation.cv_folds,
                shuffle=True,
                random_state=config.evaluation.random_state + repeat,
            )
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_tif)):
                x_train, x_test = x_tif[train_idx], x_tif[test_idx]
                y_train, y_test = y_tif[train_idx], y_tif[test_idx]
                
                # Component selection
                best_n_comp = select_components_cv(
                    x_train[..., 0],
                    y_train[..., 0],
                    config.model.max_components,
                    config.evaluation.inner_cv_folds,
                    random_state=repeat * 100 + fold_idx,
                )
                print(f"    Fold {fold_idx + 1}/{config.evaluation.cv_folds} (n_comp={best_n_comp})")
                
                y_true_test = y_test[..., 0].ravel()
                
                for method in methods:
                    # Create model using factory
                    # We pass 'best_n_comp' to override the default config if needed
                    model = create_model(config, method, n_components=best_n_comp)
                    
                    if hasattr(model, "fit"):
                         if method == "PLS":
                             # PLS takes (X, Y)
                             model.fit(x_train[..., 0], y_train[..., 0])
                             y_pred = model.predict(x_test[..., 0]).ravel()
                         else:
                             # Neutrosophic methods take (X_tif, Y_tif)
                             model.fit(x_train, y_train)
                             y_pred = model.predict(x_test).ravel()
                    
                    metrics = evaluation_metrics(y_true_test, y_pred, config.dataset.task)
                    results.append({
                        "repeat": repeat,
                        "fold": fold_idx,
                        "method": method,
                        "n_components": best_n_comp,
                        **metrics,
                    })
                    
                    # Compute VIP for neutrosophic methods
                    if config.evaluation.compute_vip and method in ("NPLS", "NPLSW", "PNPLS"):
                        vip = compute_nvip(model, x_train, channel_weights=config.model.channel_weights)
                        vip_results.append({
                            "repeat": repeat,
                            "fold": fold_idx,
                            "method": method,
                            "aggregate_vip": vip["aggregate"].tolist(),
                            "T_vip": vip["T"].tolist(),
                            "I_vip": vip["I"].tolist(),
                            "F_vip": vip["F"].tolist(),
                        })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Compute summary
    if config.dataset.task == "regression":
        summary = df.groupby("method").agg({
            "RMSEP": ["mean", "std"],
            "R2": ["mean", "std"],
            "MAE": ["mean", "std"],
        }).reset_index()
    else:
        summary = df.groupby("method").agg({
            "Accuracy": ["mean", "std"],
            "Error_Rate": ["mean", "std"],
        }).reset_index()
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    
    # Save results
    df.to_csv(output_dir / "cv_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    
    if vip_results:
        with open(output_dir / "vip_results.json", "w") as f:
            json.dump(vip_results, f)
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/")
    
    return {
        "results": df,
        "summary": summary,
        "vip_results": vip_results,
        "metadata": metadata,
        "config": config,
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neutrosophic Partial Least Squares (N-PLS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (guided setup)
  python -m neutrosophic_pls --interactive
  
  # Run from config file
  python -m neutrosophic_pls --config study.yaml
  
  # Direct mode (specify data and target)
  python -m neutrosophic_pls --data data.csv --target y
  
  # Quick run with preset
  python -m neutrosophic_pls --preset idrc_wheat
  
  # List available datasets
  python -m neutrosophic_pls --list-data
        """,
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode with guided setup",
    )
    mode_group.add_argument(
        "--config", "-c",
        type=str,
        metavar="FILE",
        help="Path to YAML/JSON config file",
    )
    mode_group.add_argument(
        "--data", "-d",
        type=str,
        metavar="FILE",
        help="Path to data file (use with --target)",
    )
    mode_group.add_argument(
        "--preset", "-p",
        type=str,
        choices=["idrc_wheat"],
        help="Use preset configuration",
    )
    mode_group.add_argument(
        "--list-data",
        action="store_true",
        help="List available datasets in data/ directory",
    )
    
    # Direct mode options
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Target column name (required with --data)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Task type (default: regression)",
    )
    parser.add_argument(
        "--snv",
        action="store_true",
        help="Apply SNV normalization (for spectral data)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        help="Columns to exclude from features",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["PLS", "NPLS", "NPLSW", "PNPLS", "all"],
        default="all",
        help="Method to use (default: all)",
    )
    parser.add_argument(
        "--lambda-indeterminacy",
        type=float,
        default=1.0,
        help="Lambda for indeterminacy weighting (NPLSW/PNPLS)",
    )
    parser.add_argument(
        "--lambda-falsity",
        type=float,
        default=1.0,
        help="Lambda controlling falsity variance prior (PNPLS)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Falsity prior softness (>0, PNPLS)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=10,
        help="Maximum number of components (default: 10)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of CV repeats (default: 3)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (1 repeat, 3 folds)",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle different modes
    if args.list_data:
        datasets = list_available_datasets()
        if datasets:
            print("Available datasets in data/:")
            print("-" * 50)
            for ds in datasets:
                print(f"  {ds['name']} ({ds['format']}, {ds['size_mb']} MB)")
        else:
            print("No datasets found in data/ directory")
        return
    
    if args.interactive:
        # Use the enhanced interactive workflow
        result = run_interactive_session()
        if result:
            print("\nInteractive session completed successfully!")
        return
    
    if args.config:
        config = StudyConfig.from_file(args.config)
        run_study(config)
        return
    
    if args.preset:
        if args.preset == "idrc_wheat":
            config = get_idrc_wheat_config(
                output_dir=args.output,
                repeats=1 if args.quick else args.repeats,
                cv_folds=3 if args.quick else args.cv_folds,
            )
        run_study(config)
        return
    
    if args.data:
        if not args.target:
            parser.error("--target is required when using --data")
        
        # Build config from command line arguments
        config = StudyConfig(
            name="CLI Study",
            dataset=DatasetSettings(
                path=args.data,
                target=args.target,
                task=args.task,
                exclude_columns=args.exclude,
                snv=args.snv,
            ),
            model=ModelSettings(
                method=args.method,
                max_components=args.max_components,
                lambda_indeterminacy=args.lambda_indeterminacy,
                lambda_falsity=args.lambda_falsity,
                alpha=args.alpha,
            ),
            evaluation=EvaluationSettings(
                cv_folds=3 if args.quick else args.cv_folds,
                repeats=1 if args.quick else args.repeats,
            ),
            output=OutputSettings(
                output_dir=args.output,
            ),
        )
        run_study(config)
        return
    
    # No mode specified - show help
    parser.print_help()
    print("\nTip: Use --interactive for guided setup or --preset idrc_wheat for a quick demo")


if __name__ == "__main__":
    main()
