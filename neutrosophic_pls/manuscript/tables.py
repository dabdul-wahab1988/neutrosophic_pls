#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table Generation Module for Neutrosophic PLS Publication.

This module provides functions to generate all manuscript tables including:
- Dataset protocol and experimental setup
- Model comparison and hyperparameters
- Main results with statistical significance
- DOE robustness analysis
- Supplementary tables

The module is organized into three main sections:
1. UTILITIES: Private helper functions
2. MANUSCRIPT TABLES: Formal paper tables (Table 1-5, Supplementary)
3. INTERACTIVE TABLES: Flexible table generation for custom experiments

Usage:
    # Generate all manuscript tables
    from neutrosophic_pls.manuscript.tables import make_all_tables
    make_all_tables(experiments_root, tables_root, ...)
    
    # Interactive use - generate specific tables
    from neutrosophic_pls.manuscript.tables import TableGenerator
    gen = TableGenerator(output_dir="./results")
    gen.comparison_table(dataset="A3")
    gen.doe_table()

Authors:
    Dickson Abdul-Wahab, Ebenezer Aquisman Asare
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    ensure_dir,
    save_csv,
    save_tex,
    get_project_root,
    compute_significance,
    run_pairwise_significance_tests,
    format_mean_std,
)
from ..simulate import generate_synthetic_spectrum, add_spike_corruption


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Main entry points
    "make_all_tables",
    "TableGenerator",
    # Manuscript tables
    "make_table1_dataset_protocol",
    "make_table2_methods_variants", 
    "make_table3_hyperparameters",
    "make_table4_main_results",
    "make_table5_robustness_ablation",
    "make_supplementary_tables",
    # Interactive tables
    "generate_table_doe",
    "generate_table_real_results",
    "generate_comparison_table",
]


# =============================================================================
# SECTION 1: PRIVATE UTILITIES
# =============================================================================

def _dataset_dirs(experiments_root: Path) -> List[Path]:
    """Return dataset experiment directories (those with cv_results.csv)."""
    root = Path(experiments_root)
    dirs: List[Path] = []
    if not root.exists():
        return dirs
    for item in root.iterdir():
        if item.is_dir() and (item / "cv_results.csv").exists():
            dirs.append(item)
    return sorted(dirs)


def _load_json(path: Path) -> Dict:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_dual(df: pd.DataFrame, base_path: Path) -> None:
    """Save DataFrame as both CSV and LaTeX."""
    save_csv(df, base_path.with_suffix(".csv"))
    save_tex(df, base_path.with_suffix(".tex"), escape=False)


def _load_real_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real spectroscopic dataset.
    
    Args:
        dataset_name: Name of dataset file (without extension)
        
    Returns:
        Tuple of (X, y) arrays
    """
    data_dir = get_project_root() / "data"
    data_path = data_dir / f"{dataset_name}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y


def _run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_components: int = 6,
    models: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run cross-validation experiment comparing models.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_folds: Number of CV folds
        n_components: PLS components
        models: List of models to evaluate (default: all)
        
    Returns:
        Dict mapping model name -> {'rmsep': [...], 'r2': [...]}
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from ..model import NPLS, NPLSW, PNPLS
    from ..data_loader import encode_neutrosophic
    
    if models is None:
        models = ['PLS', 'NPLS', 'NPLSW', 'PNPLS']
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {model: {'rmsep': [], 'r2': []} for model in models}
    
    model_classes = {
        'PLS': lambda: PLSRegression(n_components=n_components),
        'NPLS': lambda: NPLS(n_components=n_components),
        'NPLSW': lambda: NPLSW(n_components=n_components),
        'PNPLS': lambda: PNPLS(n_components=n_components),
    }
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Encode for neutrosophic models
        x_train_tif, y_train_tif = encode_neutrosophic(
            X_train, y_train.reshape(-1, 1), encoding='spectroscopy'
        )
        x_test_tif, y_test_tif = encode_neutrosophic(
            X_test, y_test.reshape(-1, 1), encoding='spectroscopy'
        )
        
        for name in models:
            if name not in model_classes:
                continue
            model = model_classes[name]()
            
            if name == 'PLS':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test).ravel()
            else:
                model.fit(x_train_tif, y_train_tif)
                y_pred = model.predict(x_test_tif).ravel()
            
            rmsep = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-12))
            
            results[name]['rmsep'].append(rmsep)
            results[name]['r2'].append(r2)
    
    return results


# =============================================================================
# SECTION 2: MANUSCRIPT TABLES (Formal Paper Tables)
# =============================================================================

def make_table1_dataset_protocol(
    dataset_dirs: Iterable[Path],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Table 1: Dataset Protocol Summary.
    
    Summarizes dataset characteristics including sample size, features,
    target variable, preprocessing, and CV protocol.
    """
    rows: List[Dict] = []
    for ds in dataset_dirs:
        meta_path = ds / "metadata.json"
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        rows.append({
            "Dataset": meta.get("name", ds.name),
            "n": meta.get("n_samples"),
            "p": meta.get("n_features"),
            "Target": meta.get("target"),
            "SNV": bool(meta.get("snv_applied") or meta.get("spectral", False)),
            "Encoder": meta.get("encoder", meta.get("encoding", "default")),
            "CV": f"{meta.get('cv_folds', '')}-fold",
            "Repeats": meta.get("repeats", ""),
        })
    df = pd.DataFrame(rows)
    _save_dual(df, Path(out_dir) / "Table1_dataset_protocol")
    return df


def make_table2_methods_variants(out_dir: Path) -> pd.DataFrame:
    """
    Table 2: Model Variants Comparison.
    
    Compares PLS, NPLS, NPLSW, and PNPLS across key characteristics.
    """
    rows = [
        {
            "Method": "PLS",
            "TIF Channels": "T only",
            "Weights (ω)": "–",
            "Imputation (π)": "–",
            "Solver": "NIPALS",
            "Key Hyperparameters": "n_components",
        },
        {
            "Method": "NPLS",
            "TIF Channels": "T, I, F",
            "Weights (ω)": "sample-level",
            "Imputation (π)": "–",
            "Solver": "weighted NIPALS",
            "Key Hyperparameters": "channel_weights",
        },
        {
            "Method": "NPLSW",
            "TIF Channels": "T, I, F",
            "Weights (ω)": "sample-level (I)",
            "Imputation (π)": "–",
            "Solver": "weighted NIPALS",
            "Key Hyperparameters": "lambda_I",
        },
        {
            "Method": "PNPLS",
            "TIF Channels": "T, I, F",
            "Weights (ω)": "element-level (F)",
            "Imputation (π)": "EM",
            "Solver": "EM-NIPALS",
            "Key Hyperparameters": "lambda_F, max_iter, tol",
        },
    ]
    df = pd.DataFrame(rows)
    _save_dual(df, Path(out_dir) / "Table2_methods_variants")
    return df


def make_table3_hyperparameters(
    out_dir: Path,
    *,
    max_components: int = 10,
    lambda_I: float = 1.0,
    lambda_F: float = 1.0,
    alpha: float = 0.5,
    channel_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> pd.DataFrame:
    """
    Table 3: Hyperparameter Settings.
    
    Documents all hyperparameters used in experiments.
    """
    rows = [
        {"Parameter": "H (components)", "Value": max_components, "Notes": "selected via inner CV"},
        {"Parameter": "λ_I", "Value": lambda_I, "Notes": "NPLSW indeterminacy weight"},
        {"Parameter": "λ_F", "Value": lambda_F, "Notes": "PNPLS falsity weight"},
        {"Parameter": "α", "Value": alpha, "Notes": "PNPLS mixing coefficient"},
        {"Parameter": "Channel weights", "Value": str(channel_weights), "Notes": "(w_T, w_I, w_F)"},
        {"Parameter": "Tolerance", "Value": 1e-7, "Notes": "solver convergence"},
        {"Parameter": "Max iterations", "Value": 500, "Notes": "solver limit"},
    ]
    df = pd.DataFrame(rows)
    _save_dual(df, Path(out_dir) / "Table3_hyperparameters")
    return df


def make_table4_main_results(
    experiments_root: Path,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Table 4: Main Results on Real Datasets.
    
    Aggregates CV results from all experiment directories.
    """
    frames: List[pd.DataFrame] = []
    for ds in _dataset_dirs(experiments_root):
        summary_path = ds / "summary_main_table.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            frames.append(df)
    
    if not frames:
        return None
    
    combined = pd.concat(frames, ignore_index=True)
    _save_dual(combined, Path(out_dir) / "Table4_main_results_realdata")
    return combined


def make_table5_robustness_ablation(
    experiments_root: Path,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Table 5: Robustness Ablation Study.
    
    Shows model performance under different corruption conditions.
    """
    sim_path = Path(experiments_root) / "simulation" / "sim_results.csv"
    if not sim_path.exists():
        return None
    
    df = pd.read_csv(sim_path)
    summary = (
        df.groupby("method")
        .agg({"RMSEP": ["mean", "std"], "R2": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = ["Method", "RMSEP_mean", "RMSEP_std", "R2_mean", "R2_std"]
    
    pretty = []
    for _, r in summary.iterrows():
        pretty.append({
            "Method": r["Method"],
            "RMSEP": format_mean_std(r["RMSEP_mean"], r["RMSEP_std"]),
            "R²": format_mean_std(r["R2_mean"], r["R2_std"]),
        })
    
    table = pd.DataFrame(pretty)
    _save_dual(table, Path(out_dir) / "Table5_robustness_ablation")
    return table


def make_supplementary_tables(
    experiments_root: Path,
    supp_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all supplementary tables.
    
    Returns dict mapping table name to file path.
    """
    root = Path(experiments_root)
    supp_dir = Path(supp_dir)
    ensure_dir(supp_dir)
    saved = {}
    
    # STable2: Full CV results
    cv_path = root / "combined_cv_results.csv"
    if cv_path.exists():
        out_path = supp_dir / "STable2_full_cv_results.csv"
        save_csv(pd.read_csv(cv_path), out_path)
        saved["cv_results"] = out_path
    
    # STable3: Runtime comparison
    runtime_path = root / "runtime" / "runtime.csv"
    if runtime_path.exists():
        out_path = supp_dir / "STable3_runtime.csv"
        save_csv(pd.read_csv(runtime_path), out_path)
        saved["runtime"] = out_path
    
    # STable4: VIP analysis
    vip_frames: List[pd.DataFrame] = []
    for ds in _dataset_dirs(experiments_root):
        vip_path = ds / "vip_summary.csv"
        if vip_path.exists():
            vip_frames.append(pd.read_csv(vip_path))
    if vip_frames:
        out_path = supp_dir / "STable4_vip_top100.csv"
        save_csv(pd.concat(vip_frames, ignore_index=True), out_path)
        saved["vip"] = out_path
    
    # STable5: Sensitivity analysis
    sensitivity_path = root / "sensitivity" / "sensitivity.csv"
    if sensitivity_path.exists():
        out_path = supp_dir / "STable5_sensitivity.csv"
        save_csv(pd.read_csv(sensitivity_path), out_path)
        saved["sensitivity"] = out_path
    
    return saved


def make_all_tables(
    experiments_root: Path,
    tables_root: Path,
    *,
    max_components: int = 10,
    lambda_I: float = 1.0,
    lambda_F: float = 1.0,
    alpha: float = 0.5,
    channel_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> Dict[str, Path]:
    """
    Generate all manuscript tables.
    
    Args:
        experiments_root: Path to experiment results directory
        tables_root: Output directory for tables
        max_components: Number of PLS components
        lambda_I: NPLSW indeterminacy weight
        lambda_F: PNPLS falsity weight
        alpha: PNPLS mixing coefficient
        channel_weights: (w_T, w_I, w_F) channel weights
        
    Returns:
        Dict with 'main' and 'supp' paths
    """
    tables_root = Path(tables_root)
    main_dir = ensure_dir(tables_root / "main")
    supp_dir = ensure_dir(tables_root / "supp")
    
    print("Generating manuscript tables...")
    
    dataset_dirs = _dataset_dirs(experiments_root)
    make_table1_dataset_protocol(dataset_dirs, main_dir)
    make_table2_methods_variants(main_dir)
    make_table3_hyperparameters(
        main_dir,
        max_components=max_components,
        lambda_I=lambda_I,
        lambda_F=lambda_F,
        alpha=alpha,
        channel_weights=channel_weights,
    )
    make_table4_main_results(experiments_root, main_dir)
    make_table5_robustness_ablation(experiments_root, main_dir)
    make_supplementary_tables(experiments_root, supp_dir)
    
    print(f"Tables saved to: {tables_root}")
    return {"main": main_dir, "supp": supp_dir}


# =============================================================================
# SECTION 3: INTERACTIVE TABLE GENERATION
# =============================================================================

def generate_comparison_table(
    X: np.ndarray,
    y: np.ndarray,
    *,
    output_dir: Optional[Path] = None,
    dataset_name: str = "custom",
    n_folds: int = 5,
    n_components: int = 6,
    models: Optional[List[str]] = None,
    with_significance: bool = True,
) -> pd.DataFrame:
    """
    Generate a comparison table for custom data (interactive use).
    
    This is the main entry point for interactive table generation,
    allowing users to compare models on their own data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        output_dir: Optional directory to save results
        dataset_name: Name for the dataset (used in filenames)
        n_folds: Number of CV folds
        n_components: Number of PLS components
        models: List of models to compare (default: all)
        with_significance: Include significance tests vs PLS
        
    Returns:
        DataFrame with comparison results
        
    Example:
        >>> from neutrosophic_pls.manuscript.tables import generate_comparison_table
        >>> df = generate_comparison_table(X, y, dataset_name="my_data")
        >>> print(df)
    """
    if models is None:
        models = ['PLS', 'NPLS', 'NPLSW', 'PNPLS']
    
    # Run CV experiment
    cv_results = _run_cv_experiment(X, y, n_folds, n_components, models)
    
    # Build results table
    rows = []
    for model in models:
        rmsep_mean = np.mean(cv_results[model]['rmsep'])
        rmsep_std = np.std(cv_results[model]['rmsep'])
        r2_mean = np.mean(cv_results[model]['r2'])
        r2_std = np.std(cv_results[model]['r2'])
        
        sig_marker = ""
        if with_significance and model != 'PLS' and 'PLS' in models:
            sig_result = compute_significance(
                np.array(cv_results['PLS']['rmsep']),
                np.array(cv_results[model]['rmsep']),
            )
            sig_marker = sig_result.get('sig_marker', '')
        
        rows.append({
            'Model': model,
            'RMSEP': f"{rmsep_mean:.4f} ± {rmsep_std:.4f}{sig_marker}",
            'R²': f"{r2_mean:.4f} ± {r2_std:.4f}",
            'RMSEP_mean': rmsep_mean,
            'R2_mean': r2_mean,
        })
    
    df = pd.DataFrame(rows)
    
    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        csv_path = output_dir / f"comparison_{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    return df


def generate_table_doe(
    output_dir: Path,
    dataset_name: str = "MA_A2",
    conditions: Optional[List[Dict]] = None,
    n_replicates: int = 5,
) -> pd.DataFrame:
    """
    Generate DOE (Design of Experiments) results table.
    
    Tests model robustness under different corruption conditions.
    
    Args:
        output_dir: Directory to save results
        dataset_name: Dataset to use (default: MA_A2)
        conditions: List of condition dicts with 'name', 'sigma', 'p_out', 'pattern'
        n_replicates: Number of replicates per condition
        
    Returns:
        DataFrame with DOE results
    """
    from .figures import run_doe_experiment_real
    
    print(f"Generating DOE Table ({dataset_name})...")
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    X_all, y_all = _load_real_data(dataset_name)
    
    if conditions is None:
        conditions = [
            {'name': 'Clean (σ=0.00, p=0.00)', 'sigma': 0.00, 'p_out': 0.00, 'pattern': 'spikes'},
            {'name': 'Moderate (σ=0.02, p=0.30)', 'sigma': 0.02, 'p_out': 0.30, 'pattern': 'spikes'},
            {'name': 'Severe (σ=0.05, p=0.30)', 'sigma': 0.05, 'p_out': 0.30, 'pattern': 'spikes'},
            {'name': 'Extreme (σ=0.10, p=0.30)', 'sigma': 0.10, 'p_out': 0.30, 'pattern': 'spikes'},
        ]
    
    table_data = []
    for cond in conditions:
        results = run_doe_experiment_real(
            X_all, y_all,
            sigma_x=cond['sigma'],
            p_out=cond['p_out'],
            pattern=cond['pattern'],
            n_replicates=n_replicates,
        )
        row = {'Condition': cond['name']}
        for model in ['PLS', 'NPLS', 'NPLSW', 'PNPLS']:
            row[model] = float(results[model]['mean'])
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save
    csv_path = output_dir / f"table_doe_{dataset_name.lower()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    return df


def generate_table_real_results(
    output_dir: Path,
    dataset_name: str = 'A3',
    scenarios: Optional[List[Tuple[str, float]]] = None,
    n_folds: int = 5,
    n_components: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate real data results table with significance tests.
    
    Args:
        output_dir: Directory to save results
        dataset_name: Dataset name (e.g., 'A3', 'B1')
        scenarios: List of (name, corruption_level) tuples
        n_folds: Number of CV folds
        n_components: Number of PLS components
        
    Returns:
        Tuple of (results_df, significance_df)
    """
    print(f"Generating Results Table ({dataset_name})...")
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    try:
        X, y = _load_real_data(dataset_name)
    except FileNotFoundError:
        print(f"  Warning: {dataset_name} not found. Using synthetic data.")
        X, y, _ = generate_synthetic_spectrum(n_samples=88, n_features=100, seed=42)
    
    if scenarios is None:
        scenarios = [('Clean', 0.0), ('Corrupted (30%)', 0.30)]
    
    table_data = []
    sig_report_data = []
    
    for scenario_name, corruption_level in scenarios:
        if corruption_level > 0:
            X_scenario = add_spike_corruption(X.copy(), corruption_level, seed=42)
        else:
            X_scenario = X.copy()
        
        cv_results = _run_cv_experiment(X_scenario, y, n_folds, n_components)
        sig_rmsep = run_pairwise_significance_tests(cv_results, baseline='PLS', metric='rmsep')
        sig_r2 = run_pairwise_significance_tests(cv_results, baseline='PLS', metric='r2')
        
        for model in ['PLS', 'NPLS', 'NPLSW', 'PNPLS']:
            rmsep_mean = np.mean(cv_results[model]['rmsep'])
            rmsep_std = np.std(cv_results[model]['rmsep'])
            r2_mean = np.mean(cv_results[model]['r2'])
            r2_std = np.std(cv_results[model]['r2'])
            
            rmsep_sig = sig_rmsep[model].get('sig_marker', '')
            r2_sig = sig_r2[model].get('sig_marker', '')
            
            table_data.append({
                'Scenario': scenario_name,
                'Model': model,
                'RMSEP': f"{rmsep_mean:.4f} ± {rmsep_std:.4f}{rmsep_sig}",
                'R²': f"{r2_mean:.4f} ± {r2_std:.4f}{r2_sig}",
            })
            
            if model != 'PLS':
                sig_info = sig_rmsep[model]
                sig_report_data.append({
                    'Scenario': scenario_name,
                    'Model': model,
                    'Improvement_%': f"{sig_info['improvement_pct']:.1f}%",
                    'p-value': f"{sig_info.get('wilcoxon_pvalue', np.nan):.4f}",
                    'Significant': 'Yes' if sig_info.get('wilcoxon_pvalue', 1) < 0.05 else 'No',
                })
    
    df = pd.DataFrame(table_data)
    df_sig = pd.DataFrame(sig_report_data)
    
    # Save
    csv_path = output_dir / f"table_results_{dataset_name.lower()}.csv"
    df.to_csv(csv_path, index=False)
    
    sig_csv_path = output_dir / f"table_significance_{dataset_name.lower()}.csv"
    df_sig.to_csv(sig_csv_path, index=False)
    
    print(f"Saved: {csv_path}")
    return df, df_sig


# =============================================================================
# SECTION 4: INTERACTIVE TABLE GENERATOR CLASS
# =============================================================================

class TableGenerator:
    """
    Interactive table generator for custom experiments.
    
    Provides a convenient interface for generating tables from custom data
    during interactive analysis or notebook sessions.
    
    Example:
        >>> gen = TableGenerator(output_dir="./results")
        >>> gen.comparison(X, y, name="my_experiment")
        >>> gen.doe(dataset="MA_A2")
        >>> gen.summary()  # Show all generated tables
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "./tables",
        n_folds: int = 5,
        n_components: int = 6,
    ):
        """
        Initialize the table generator.
        
        Args:
            output_dir: Directory to save generated tables
            n_folds: Default number of CV folds
            n_components: Default number of PLS components
        """
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.n_components = n_components
        self._generated: List[Dict[str, Any]] = []
        
        ensure_dir(self.output_dir)
    
    def comparison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str = "comparison",
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate comparison table for custom data."""
        df = generate_comparison_table(
            X, y,
            output_dir=self.output_dir,
            dataset_name=name,
            n_folds=self.n_folds,
            n_components=self.n_components,
            models=models,
        )
        self._generated.append({
            "type": "comparison",
            "name": name,
            "path": self.output_dir / f"comparison_{name}.csv",
        })
        return df
    
    def doe(
        self,
        dataset: str = "MA_A2",
        conditions: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """Generate DOE robustness table."""
        df = generate_table_doe(
            self.output_dir,
            dataset_name=dataset,
            conditions=conditions,
        )
        self._generated.append({
            "type": "doe",
            "name": dataset,
            "path": self.output_dir / f"table_doe_{dataset.lower()}.csv",
        })
        return df
    
    def results(
        self,
        dataset: str = "A3",
        scenarios: Optional[List[Tuple[str, float]]] = None,
    ) -> pd.DataFrame:
        """Generate results table with significance tests."""
        df, df_sig = generate_table_real_results(
            self.output_dir,
            dataset_name=dataset,
            scenarios=scenarios,
            n_folds=self.n_folds,
            n_components=self.n_components,
        )
        self._generated.append({
            "type": "results",
            "name": dataset,
            "path": self.output_dir / f"table_results_{dataset.lower()}.csv",
        })
        return df
    
    def summary(self) -> pd.DataFrame:
        """Show summary of all generated tables."""
        if not self._generated:
            print("No tables generated yet.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self._generated)
        print(f"\nGenerated {len(df)} tables:")
        print(df.to_string(index=False))
        return df
    
    def __repr__(self) -> str:
        return f"TableGenerator(output_dir='{self.output_dir}', tables={len(self._generated)})"
