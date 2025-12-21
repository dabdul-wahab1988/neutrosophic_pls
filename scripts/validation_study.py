"""
Comprehensive I/F validation study:
- Calibrate entropy bins using repeat measurements (synthetic repeats if none provided)
- Compare encoding variants: default encoder vs detector-limit-based encoding
- Report RMSEP/R2 across CV folds
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from neutrosophic_pls.metrics import rmsep as compute_rmsep, r2_score
from neutrosophic_pls.data_loader import DatasetConfig, load_dataset
from neutrosophic_pls.model import NPLSW
from neutrosophic_pls.validation import IndeterminacyValidator


def run_validation_study(
    dataset_path: str,
    target_column: str,
    output_dir: str = "results_validation",
    n_splits: int = 5,
    n_components: int = 5,
    entropy_bins_grid: list[int] = [10, 20, 30, 40, 50],
    repeats_path: str | None = None,
    lower_limits: str | None = None,
    upper_limits: str | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting validation study for {dataset_path} (target: {target_column})")

    # Phase 1: entropy calibration (synthetic repeats if none provided)
    main_cfg = DatasetConfig(path=dataset_path, target=target_column)
    main_data = load_dataset(main_cfg)
    X_truth = main_data["x_tif"][:, :, 0]

    if repeats_path:
        rep_cfg = DatasetConfig(path=repeats_path, target=target_column)
        rep_data = load_dataset(rep_cfg)
        X_rep_truth = rep_data["x_tif"][:, :, 0]
        # Simple heuristic: assume 3 synthetic repeats around each measurement
        rng = np.random.default_rng(0)
        X_repeated = np.stack(
            [X_rep_truth + rng.normal(scale=0.01, size=X_rep_truth.shape) for _ in range(3)],
            axis=1,
        )
    else:
        rng = np.random.default_rng(0)
        X_repeated = np.stack(
            [X_truth + rng.normal(scale=0.01, size=X_truth.shape) for _ in range(3)],
            axis=1,
        )

    validator_i = IndeterminacyValidator()
    calib = validator_i.calibrate_entropy_parameters(
        X_truth, X_repeated, bins_grid=entropy_bins_grid
    )
    optimal_bins = calib["optimal_bins"]
    print(f"Optimal entropy_bins: {optimal_bins} (R²={calib['optimal_r2']:.3f})")

    # Phase 2: encoding comparison
    # Optional limits from CLI (comma-separated) override dummy bounds
    lower_arr = None
    upper_arr = None
    if lower_limits:
        lower_arr = np.fromstring(lower_limits, sep=",")
        if lower_arr.size == 1:
            lower_arr = np.full(X_truth.shape[1], lower_arr.item())
    if upper_limits:
        upper_arr = np.fromstring(upper_limits, sep=",")
        if upper_arr.size == 1:
            upper_arr = np.full(X_truth.shape[1], upper_arr.item())

    encodings = {
        "default_encoder": DatasetConfig(
            path=dataset_path,
            target=target_column,
        ),
        "limits_dummy": DatasetConfig(
            path=dataset_path,
            target=target_column,
            lower_limits=lower_arr if lower_arr is not None else X_truth.min(axis=0),
            upper_limits=upper_arr if upper_arr is not None else X_truth.max(axis=0),
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, cfg in encodings.items():
        data = load_dataset(cfg)
        X_tif = data["x_tif"]
        y_tif = data["y_tif"]

        rmseps, r2s = [], []
        for train_idx, test_idx in kf.split(X_tif):
            X_train, X_test = X_tif[train_idx], X_tif[test_idx]
            y_train, y_test = y_tif[train_idx], y_tif[test_idx]

            model = NPLSW(n_components=n_components)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true = y_test[:, 0, 0]
            y_hat = y_pred[:, 0] if y_pred.ndim == 2 else y_pred[:, 0, 0]
            rmseps.append(compute_rmsep(y_true, y_hat))
            r2s.append(r2_score(y_true, y_hat))

        metrics[name] = {
            "mean_rmsep": float(np.mean(rmseps)),
            "std_rmsep": float(np.std(rmseps)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
        }
        print(
            f"  {name:20s} RMSEP={metrics[name]['mean_rmsep']:.3f} "
            f"R²={metrics[name]['mean_r2']:.3f}"
        )

    pd.DataFrame.from_dict(metrics, orient="index").to_csv(output_path / "validation_report.csv")
    return {"entropy_calibration": calib, "model_performance": metrics}


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="results_validation", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--components", type=int, default=5, help="NPLSW components")
    parser.add_argument("--bins", nargs="+", type=int, default=[10, 20, 30, 40, 50], help="Entropy bin grid")
    parser.add_argument("--repeats", help="Optional dataset with repeats for calibration")
    parser.add_argument("--lower-limits", help="Comma-separated detector lower limits per feature or single value")
    parser.add_argument("--upper-limits", help="Comma-separated detector upper limits per feature or single value")
    args = parser.parse_args()

    results = run_validation_study(
        dataset_path=args.dataset,
        target_column=args.target,
        output_dir=args.output,
        n_splits=args.folds,
        n_components=args.components,
        entropy_bins_grid=args.bins,
        repeats_path=args.repeats,
        lower_limits=args.lower_limits,
        upper_limits=args.upper_limits,
    )
    print("Completed validation study.")
    print(results)


if __name__ == "__main__":
    cli()
