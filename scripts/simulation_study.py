#!/usr/bin/env python
"""
Simulation Study for N-PLS vs PLS Paper
========================================

A designed simulation study with three stages:
  Stage 1: Screening (fractional factorial) - identify dominant factors
  Stage 2: Response surface (3-level factorial) - detailed comparison
  Stage 3: MicroMass confirmatory analysis with nested CV

Author: NeutroProject
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

# Neutrosophic PLS imports
from neutrosophic_pls import (
    NPLS,
    NPLSW,
    generate_simulation,
    load_micromass,
    evaluation_metrics,
    component_recovery,
    compute_nvip,
)
from neutrosophic_pls.algebra import combine_channels

# ==============================================================================
# Configuration Dataclasses
# ==============================================================================


@dataclass
class ScreeningFactor:
    """Definition of a factor for the screening design."""

    name: str
    levels: Tuple[Any, Any]
    description: str = ""


@dataclass
class ScreeningConfig:
    """Configuration for Stage 1 screening experiment."""

    factors: List[ScreeningFactor] = field(default_factory=list)
    n_replicates: int = 3
    n_components_true: int = 2
    output_dir: Path = Path("results/stage1_screening")

    def __post_init__(self):
        if not self.factors:
            self.factors = [
                ScreeningFactor("sigma_T", (0.05, 0.20), "Truth channel noise"),
                ScreeningFactor("sigma_I", (0.00, 0.20), "Indeterminacy noise"),
                ScreeningFactor("falsity_prop", (0.00, 0.10), "Falsity/outlier proportion"),
                ScreeningFactor("n_samples", (100, 500), "Sample size"),
                ScreeningFactor("n_features", (20, 100), "Number of predictors"),
                ScreeningFactor(
                    "n_components_fit",
                    (2, 3),
                    "Components fitted (true=2, overfit=3)",
                ),
                ScreeningFactor(
                    "weight_pattern",
                    ("equal", "emphasize_T"),
                    "Channel weights: equal=(1,1,1), emphasize_T=(2,1,0.5)",
                ),
            ]


@dataclass
class ResponseSurfaceConfig:
    """Configuration for Stage 2 response surface experiment."""

    sigma_T_levels: Tuple[float, ...] = (0.05, 0.15, 0.25)
    sigma_I_levels: Tuple[float, ...] = (0.00, 0.10, 0.20)
    falsity_prop_levels: Tuple[float, ...] = (0.00, 0.05, 0.10)
    n_samples: int = 200
    n_features: int = 50
    n_components_true: int = 2
    n_components_fit: int = 2
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_replicates: int = 5
    output_dir: Path = Path("results/stage2_response_surface")


@dataclass
class MicroMassConfig:
    """Configuration for Stage 3 MicroMass confirmatory study."""

    n_outer_folds: int = 5
    n_inner_folds: int = 3
    max_components: int = 5
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_repeats: int = 3
    output_dir: Path = Path("results/stage3_micromass")


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_channel_weights(pattern: str) -> Tuple[float, float, float]:
    """Convert weight pattern string to actual weights."""
    patterns = {
        "equal": (1.0, 1.0, 1.0),
        "emphasize_T": (2.0, 1.0, 0.5),
        "downweight_F": (1.0, 1.0, 0.2),
        "T_only": (1.0, 0.0, 0.0),
    }
    return patterns.get(pattern, (1.0, 1.0, 1.0))


def enhanced_simulation(
    n_samples: int,
    n_features: int,
    n_components: int,
    sigma_T: float = 0.05,
    sigma_I: float = 0.10,
    falsity_prop: float = 0.05,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Enhanced data generation with separate control over:
      - Truth channel noise (sigma_T)
      - Indeterminacy noise (sigma_I) 
      - Falsity/outlier proportion (falsity_prop)
    
    The simulation now properly encodes:
      - T channel: true signal (with noise), CORRUPTED in outlier cells
      - I channel: uncertainty indicator (higher = less certain)
      - F channel: binary outlier flags (1 = this cell is unreliable)
    
    This design allows NPLSW to properly downweight unreliable samples.
    """
    rng = np.random.default_rng(seed)

    # Generate latent structure
    latent = rng.normal(size=(n_samples, n_components))
    loadings_x = rng.normal(size=(n_components, n_features))
    loadings_y = rng.normal(size=(n_components, 1))

    # Truth channel: signal + noise
    x_truth_clean = latent @ loadings_x + rng.normal(scale=sigma_T, size=(n_samples, n_features))
    y_truth_clean = latent @ loadings_y + rng.normal(scale=sigma_T, size=(n_samples, 1))

    # Initialize working copies
    x_truth = x_truth_clean.copy()
    y_truth = y_truth_clean.copy()

    # Indeterminacy channel: uncertainty/variability (scaled 0-1)
    # Higher values indicate more uncertainty
    x_ind = np.abs(rng.normal(scale=sigma_I, size=(n_samples, n_features)))
    y_ind = np.abs(rng.normal(scale=sigma_I, size=(n_samples, 1)))
    # Normalize to [0, 1] range
    x_ind = x_ind / (x_ind.max() + 1e-10)
    y_ind = y_ind / (y_ind.max() + 1e-10)

    # Falsity channel: binary outlier flags
    x_falsity = np.zeros_like(x_truth)
    y_falsity = np.zeros_like(y_truth)

    if falsity_prop > 0:
        # Determine outlier cells based on proportion
        n_outliers_x = int(n_samples * n_features * falsity_prop)
        n_outliers_y = int(n_samples * falsity_prop)

        if n_outliers_x > 0:
            outlier_idx_x = rng.choice(
                n_samples * n_features, size=n_outliers_x, replace=False
            )
            row_idx = outlier_idx_x // n_features
            col_idx = outlier_idx_x % n_features
            
            # Mark these cells as falsity (unreliable)
            x_falsity[row_idx, col_idx] = 1.0
            
            # CORRUPT the truth channel at these locations
            # This simulates measurement errors/outliers
            outlier_scale = np.std(x_truth_clean) * 3
            x_truth[row_idx, col_idx] = x_truth_clean[row_idx, col_idx] + rng.normal(scale=outlier_scale, size=n_outliers_x)
            
            # Also increase indeterminacy at outlier locations
            x_ind[row_idx, col_idx] = np.clip(x_ind[row_idx, col_idx] + 0.5, 0, 1)

        if n_outliers_y > 0:
            outlier_idx_y = rng.choice(n_samples, size=n_outliers_y, replace=False)
            y_falsity[outlier_idx_y, 0] = 1.0
            
            outlier_scale_y = np.std(y_truth_clean) * 3
            y_truth[outlier_idx_y, 0] = y_truth_clean[outlier_idx_y, 0] + rng.normal(scale=outlier_scale_y, size=n_outliers_y)
            y_ind[outlier_idx_y, 0] = np.clip(y_ind[outlier_idx_y, 0] + 0.5, 0, 1)

    # Stack into T/I/F tensors
    x_tif = np.stack([x_truth, x_ind, x_falsity], axis=-1)
    y_tif = np.stack([y_truth, y_ind, y_falsity], axis=-1)

    metadata = {
        "latent": latent,
        "loadings_x": loadings_x,
        "loadings_y": loadings_y,
        "noise_config": {
            "sigma_T": sigma_T,
            "sigma_I": sigma_I,
            "falsity_prop": falsity_prop,
        },
    }

    return x_tif, y_tif, metadata


def fit_classical_pls(
    x_tif: np.ndarray,
    y_tif: np.ndarray,
    n_components: int,
    method: Literal["T_only", "collapsed"] = "T_only",
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PLSRegression:
    """
    Fit classical sklearn PLS on either:
      - T_only: just the truth channel
      - collapsed: weighted combination of T/I/F
    """
    if method == "T_only":
        X = x_tif[..., 0]
        Y = y_tif[..., 0]
    else:
        X = combine_channels(x_tif, channel_weights)
        Y = combine_channels(y_tif, channel_weights)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    return pls


def evaluate_models(
    x_tif: np.ndarray,
    y_tif: np.ndarray,
    metadata: Dict,
    n_components_fit: int,
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """
    Fit and evaluate both PLS and N-PLS on the given data.
    Returns metrics for both methods.
    """
    results = {}

    # Ground truth for component recovery
    true_latent = metadata["latent"]

    # Classical PLS (T-only)
    pls_t = fit_classical_pls(x_tif, y_tif, n_components_fit, method="T_only")
    y_true_collapsed = combine_channels(y_tif, channel_weights)
    y_pred_pls_t = pls_t.predict(x_tif[..., 0])
    metrics_pls_t = evaluation_metrics(y_true_collapsed, y_pred_pls_t)

    # Component recovery for PLS
    pls_scores = pls_t.x_scores_
    n_comp = min(pls_scores.shape[1], true_latent.shape[1])
    recovery_pls = component_recovery(true_latent[:, :n_comp], pls_scores[:, :n_comp])
    metrics_pls_t.update({f"pls_{k}": v for k, v in recovery_pls.items()})
    results["PLS_T"] = metrics_pls_t

    # Classical PLS (collapsed)
    pls_c = fit_classical_pls(
        x_tif, y_tif, n_components_fit, method="collapsed", channel_weights=channel_weights
    )
    X_collapsed = combine_channels(x_tif, channel_weights)
    y_pred_pls_c = pls_c.predict(X_collapsed)
    metrics_pls_c = evaluation_metrics(y_true_collapsed, y_pred_pls_c)
    pls_c_scores = pls_c.x_scores_
    recovery_pls_c = component_recovery(true_latent[:, :n_comp], pls_c_scores[:, :n_comp])
    metrics_pls_c.update({f"pls_{k}": v for k, v in recovery_pls_c.items()})
    results["PLS_collapsed"] = metrics_pls_c

    # N-PLS
    npls = NPLS(n_components=n_components_fit, channel_weights=channel_weights)
    npls.fit(x_tif, y_tif)
    y_pred_npls = npls.predict(x_tif)
    metrics_npls = evaluation_metrics(y_true_collapsed, y_pred_npls)
    npls_scores = npls.scores_
    recovery_npls = component_recovery(true_latent[:, :n_comp], npls_scores[:, :n_comp])
    metrics_npls.update({f"npls_{k}": v for k, v in recovery_npls.items()})
    results["NPLS"] = metrics_npls

    # N-PLS with reliability weighting (NPLSW)
    nplsw = NPLSW(
        n_components=n_components_fit,
        channel_weights=channel_weights,
        lambda_indeterminacy=1.0,
    )
    nplsw.fit(x_tif, y_tif)
    y_pred_nplsw = nplsw.predict(x_tif)
    metrics_nplsw = evaluation_metrics(y_true_collapsed, y_pred_nplsw)
    nplsw_scores = nplsw.scores_
    recovery_nplsw = component_recovery(true_latent[:, :n_comp], nplsw_scores[:, :n_comp])
    metrics_nplsw.update({f"nplsw_{k}": v for k, v in recovery_nplsw.items()})
    results["NPLSW"] = metrics_nplsw

    return results


# ==============================================================================
# Stage 1: Screening Experiment (Fractional Factorial)
# ==============================================================================


def generate_fractional_factorial(n_factors: int, resolution: int = 4) -> np.ndarray:
    """
    Generate a 2^(n-k) fractional factorial design.
    For 7 factors at resolution IV, we use 2^(7-3) = 16 runs,
    or 2^(7-2) = 32 runs for better coverage.
    """
    if n_factors <= 4:
        # Full factorial for small number of factors
        return np.array(list(itertools.product([0, 1], repeat=n_factors)))

    # Use generators for fractional factorial
    # For 7 factors, resolution IV: D=AB, E=AC, F=BC, G=ABC
    # This gives 2^4 = 16 base runs
    base_factors = 4
    base_design = np.array(list(itertools.product([0, 1], repeat=base_factors)))

    if n_factors == 7:
        # Generate columns D, E, F, G from generators
        A, B, C = base_design[:, 0], base_design[:, 1], base_design[:, 2]
        D_col = (A + B) % 2  # D = AB
        E_col = (A + C) % 2  # E = AC
        F_col = (B + C) % 2  # F = BC

        design = np.column_stack([base_design, D_col, E_col, F_col])

        # For 32 runs, replicate with center point variation
        # Actually, let's just use 2^(7-3) = 16 or expand to 32
        # For simplicity, we'll generate 2^5 = 32 with 2 generators
        base_5 = np.array(list(itertools.product([0, 1], repeat=5)))
        A, B, C, D, E = (
            base_5[:, 0],
            base_5[:, 1],
            base_5[:, 2],
            base_5[:, 3],
            base_5[:, 4],
        )
        F_col = (A + B + C) % 2  # F = ABC
        G_col = (A + B + D) % 2  # G = ABD

        design = np.column_stack([base_5, F_col, G_col])
        return design

    # Default: smaller factorial
    return np.array(list(itertools.product([0, 1], repeat=min(n_factors, 5))))


def run_screening_study(config: ScreeningConfig) -> pd.DataFrame:
    """
    Execute Stage 1 screening study.
    Returns a DataFrame with all results.
    """
    print("=" * 70)
    print("STAGE 1: SCREENING STUDY (Fractional Factorial Design)")
    print("=" * 70)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate design matrix
    n_factors = len(config.factors)
    design_matrix = generate_fractional_factorial(n_factors)
    n_runs = len(design_matrix)

    print(f"\nDesign: 2^({n_factors}-k) fractional factorial")
    print(f"Number of design points: {n_runs}")
    print(f"Replicates per design point: {config.n_replicates}")
    print(f"Total simulation runs: {n_runs * config.n_replicates}")
    print("\nFactors:")
    for i, f in enumerate(config.factors):
        print(f"  {f.name}: {f.levels} - {f.description}")

    results = []

    for run_idx, design_point in enumerate(design_matrix):
        # Map design point to actual factor values
        params = {}
        for i, factor in enumerate(config.factors):
            level_idx = int(design_point[i])
            params[factor.name] = factor.levels[level_idx]

        # Get channel weights
        weight_pattern = params.get("weight_pattern", "equal")
        channel_weights = get_channel_weights(weight_pattern)

        print(f"\nRun {run_idx + 1}/{n_runs}: {params}")

        for rep in range(config.n_replicates):
            seed = run_idx * 1000 + rep

            # Generate data
            x_tif, y_tif, meta = enhanced_simulation(
                n_samples=int(params["n_samples"]),
                n_features=int(params["n_features"]),
                n_components=config.n_components_true,
                sigma_T=float(params["sigma_T"]),
                sigma_I=float(params["sigma_I"]),
                falsity_prop=float(params["falsity_prop"]),
                seed=seed,
            )

            # Evaluate models
            model_results = evaluate_models(
                x_tif,
                y_tif,
                meta,
                n_components_fit=int(params["n_components_fit"]),
                channel_weights=channel_weights,
            )

            # Store results
            for method, metrics in model_results.items():
                result_row = {
                    "run_idx": run_idx,
                    "replicate": rep,
                    "seed": seed,
                    "method": method,
                    **params,
                    **metrics,
                }
                results.append(result_row)

    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv(config.output_dir / "screening_raw_results.csv", index=False)

    # Compute summary statistics
    summary = compute_screening_summary(df, config)
    summary.to_csv(config.output_dir / "screening_summary.csv", index=False)

    # Compute factor effects
    effects = compute_factor_effects(df, config)
    effects.to_csv(config.output_dir / "factor_effects.csv", index=False)

    print(f"\nResults saved to {config.output_dir}/")

    return df


def compute_screening_summary(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    """Compute summary statistics by method and design point."""
    factor_names = [f.name for f in config.factors]
    group_cols = factor_names + ["method"]

    summary = (
        df.groupby(group_cols)
        .agg(
            {
                "RMSEP": ["mean", "std"],
                "R2": ["mean", "std"],
                "MAE": ["mean", "std"],
            }
        )
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def compute_factor_effects(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    """
    Compute main effects of each factor on key metrics.
    Effect = mean(high level) - mean(low level)
    """
    effects_list = []

    for method in df["method"].unique():
        method_df = df[df["method"] == method]

        for factor in config.factors:
            low_level, high_level = factor.levels

            for metric in ["RMSEP", "R2", "MAE"]:
                low_mean = method_df[method_df[factor.name] == low_level][metric].mean()
                high_mean = method_df[method_df[factor.name] == high_level][metric].mean()
                effect = high_mean - low_mean

                effects_list.append(
                    {
                        "method": method,
                        "factor": factor.name,
                        "metric": metric,
                        "low_level": low_level,
                        "high_level": high_level,
                        "low_mean": low_mean,
                        "high_mean": high_mean,
                        "effect": effect,
                        "abs_effect": abs(effect),
                    }
                )

    return pd.DataFrame(effects_list)


# ==============================================================================
# Stage 2: Response Surface Study (3-Level Factorial)
# ==============================================================================


def run_response_surface_study(config: ResponseSurfaceConfig) -> pd.DataFrame:
    """
    Execute Stage 2 response surface study with 3-level factorial design.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: RESPONSE SURFACE STUDY (3-Level Factorial Design)")
    print("=" * 70)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3x3x3 factorial design
    scenarios = list(
        itertools.product(
            config.sigma_T_levels,
            config.sigma_I_levels,
            config.falsity_prop_levels,
        )
    )

    n_scenarios = len(scenarios)
    print(f"\nDesign: 3³ = {n_scenarios} scenarios")
    print(f"Replicates per scenario: {config.n_replicates}")
    print(f"Total simulation runs: {n_scenarios * config.n_replicates}")
    print(f"\nFixed parameters:")
    print(f"  n_samples: {config.n_samples}")
    print(f"  n_features: {config.n_features}")
    print(f"  n_components (true): {config.n_components_true}")
    print(f"  n_components (fit): {config.n_components_fit}")
    print(f"  channel_weights: {config.channel_weights}")
    print(f"\nFactor levels:")
    print(f"  sigma_T: {config.sigma_T_levels}")
    print(f"  sigma_I: {config.sigma_I_levels}")
    print(f"  falsity_prop: {config.falsity_prop_levels}")

    results = []

    for scenario_idx, (sigma_T, sigma_I, falsity_prop) in enumerate(scenarios):
        print(
            f"\nScenario {scenario_idx + 1}/{n_scenarios}: "
            f"σ_T={sigma_T:.2f}, σ_I={sigma_I:.2f}, F_prop={falsity_prop:.2f}"
        )

        for rep in range(config.n_replicates):
            seed = scenario_idx * 1000 + rep

            # Generate data
            x_tif, y_tif, meta = enhanced_simulation(
                n_samples=config.n_samples,
                n_features=config.n_features,
                n_components=config.n_components_true,
                sigma_T=sigma_T,
                sigma_I=sigma_I,
                falsity_prop=falsity_prop,
                seed=seed,
            )

            # Evaluate models
            model_results = evaluate_models(
                x_tif,
                y_tif,
                meta,
                n_components_fit=config.n_components_fit,
                channel_weights=config.channel_weights,
            )

            # Store results
            for method, metrics in model_results.items():
                result_row = {
                    "scenario_idx": scenario_idx,
                    "replicate": rep,
                    "seed": seed,
                    "method": method,
                    "sigma_T": sigma_T,
                    "sigma_I": sigma_I,
                    "falsity_prop": falsity_prop,
                    **metrics,
                }
                results.append(result_row)

    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv(config.output_dir / "response_surface_raw_results.csv", index=False)

    # Generate scenario table (Table 3 style)
    scenario_table = generate_scenario_table(scenarios, config)
    scenario_table.to_csv(config.output_dir / "scenario_definitions.csv", index=False)

    # Generate performance summary (Table 5/6 style)
    performance_summary = generate_performance_summary(df)
    performance_summary.to_csv(config.output_dir / "performance_summary.csv", index=False)

    # Generate method comparison
    method_comparison = generate_method_comparison(df)
    method_comparison.to_csv(config.output_dir / "method_comparison.csv", index=False)

    print(f"\nResults saved to {config.output_dir}/")

    return df


def generate_scenario_table(
    scenarios: List[Tuple], config: ResponseSurfaceConfig
) -> pd.DataFrame:
    """Generate Table 3: Scenario definitions."""
    rows = []
    for idx, (sigma_T, sigma_I, falsity_prop) in enumerate(scenarios):
        rows.append(
            {
                "Scenario": idx + 1,
                "σ_T (truth noise)": sigma_T,
                "σ_I (indeterminacy)": sigma_I,
                "Falsity proportion": falsity_prop,
                "n": config.n_samples,
                "p": config.n_features,
                "True components": config.n_components_true,
                "Fitted components": config.n_components_fit,
            }
        )
    return pd.DataFrame(rows)


def generate_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 5/6: Performance metrics by scenario and method."""
    summary = (
        df.groupby(["scenario_idx", "sigma_T", "sigma_I", "falsity_prop", "method"])
        .agg(
            {
                "RMSEP": ["mean", "std"],
                "R2": ["mean", "std"],
                "MAE": ["mean", "std"],
            }
        )
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def generate_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Generate method comparison: NPLS vs PLS improvement."""
    comparison_rows = []

    for scenario_idx in df["scenario_idx"].unique():
        scenario_df = df[df["scenario_idx"] == scenario_idx]

        # Get mean metrics for each method
        method_means = scenario_df.groupby("method")[["RMSEP", "R2", "MAE"]].mean()

        if "NPLS" in method_means.index and "PLS_T" in method_means.index:
            npls_rmsep = method_means.loc["NPLS", "RMSEP"]
            pls_rmsep = method_means.loc["PLS_T", "RMSEP"]
            npls_r2 = method_means.loc["NPLS", "R2"]
            pls_r2 = method_means.loc["PLS_T", "R2"]

            rmsep_improvement = (pls_rmsep - npls_rmsep) / pls_rmsep * 100
            r2_improvement = (npls_r2 - pls_r2) / max(abs(pls_r2), 0.001) * 100

            row_info = scenario_df.iloc[0][["sigma_T", "sigma_I", "falsity_prop"]]

            comparison_rows.append(
                {
                    "scenario": scenario_idx + 1,
                    "sigma_T": row_info["sigma_T"],
                    "sigma_I": row_info["sigma_I"],
                    "falsity_prop": row_info["falsity_prop"],
                    "PLS_RMSEP": pls_rmsep,
                    "NPLS_RMSEP": npls_rmsep,
                    "RMSEP_improvement_%": rmsep_improvement,
                    "PLS_R2": pls_r2,
                    "NPLS_R2": npls_r2,
                    "R2_improvement_%": r2_improvement,
                }
            )

    return pd.DataFrame(comparison_rows)


# ==============================================================================
# Stage 3: MicroMass Confirmatory Study
# ==============================================================================


def run_micromass_study(config: MicroMassConfig) -> Dict[str, Any]:
    """
    Execute Stage 3 MicroMass confirmatory study with nested cross-validation.
    """
    import warnings
    
    print("\n" + "=" * 70)
    print("STAGE 3: MICROMASS CONFIRMATORY STUDY (Nested Cross-Validation)")
    print("=" * 70)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MicroMass data - IMPORTANT: prefer_fixture=False to get real data
    print("\nLoading MicroMass dataset (downloading if needed)...")
    data = load_micromass(prefer_fixture=False)
    x_tif = data["x_tif"]
    y_tif = data["y_tif"]
    metadata = data["metadata"]

    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Target: {metadata['target_name']}")
    
    # Validate dataset size
    if metadata['n_samples'] < 20:
        print(f"\n  WARNING: Dataset too small ({metadata['n_samples']} samples).")
        print("  The fixture file contains only test data.")
        print("  Attempting to download full MicroMass dataset...")
        from neutrosophic_pls.data_micromass import download_micromass
        try:
            download_path = download_micromass()
            data = load_micromass(path=download_path)
            x_tif = data["x_tif"]
            y_tif = data["y_tif"]
            metadata = data["metadata"]
            print(f"  Downloaded: {metadata['n_samples']} samples, {metadata['n_features']} features")
        except Exception as e:
            print(f"  Could not download: {e}")
            print("  Proceeding with fixture data (results may not be meaningful)...")

    # Save dataset summary (Table 4)
    dataset_summary = {
        "Dataset": "MicroMass (OpenML 1514)",
        "Samples": metadata["n_samples"],
        "Features": metadata["n_features"],
        "Target": metadata["target_name"],
        "Encoding": "Neutrosophic (T/I/F)",
        "T_channel": "Raw spectral values",
        "I_channel": "Robust MAD-scaled |z-scores|",
        "F_channel": "Outlier flags (|z| > 3.5)",
    }
    with open(config.output_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    results = []
    vip_results = []
    
    # Adjust CV folds if dataset is small
    n_samples = metadata['n_samples']
    actual_outer_folds = min(config.n_outer_folds, n_samples // 2)
    actual_inner_folds = min(config.n_inner_folds, (n_samples - n_samples // actual_outer_folds) // 2)
    
    if actual_outer_folds < config.n_outer_folds:
        print(f"\n  Adjusted outer CV folds from {config.n_outer_folds} to {actual_outer_folds} due to small sample size")
    if actual_inner_folds < config.n_inner_folds:
        print(f"  Adjusted inner CV folds from {config.n_inner_folds} to {actual_inner_folds} due to small sample size")

    print(f"\nRunning {config.n_repeats} repeats of {actual_outer_folds}-fold CV...")
    
    # Suppress sklearn warnings for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*R\\^2 score.*")

        for repeat in range(config.n_repeats):
            print(f"\n  Repeat {repeat + 1}/{config.n_repeats}")

            outer_cv = KFold(n_splits=actual_outer_folds, shuffle=True, random_state=repeat)

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_tif)):
                print(f"    Fold {fold_idx + 1}/{actual_outer_folds}", end=" ")

                x_train, x_test = x_tif[train_idx], x_tif[test_idx]
                y_train, y_test = y_tif[train_idx], y_tif[test_idx]
                
                # Skip if test set too small for meaningful evaluation
                if len(test_idx) < 2:
                    print("(skipped - test set too small)")
                    continue

                # Inner CV for component selection
                best_n_comp = select_components_cv(
                    x_train, y_train, config.max_components, actual_inner_folds
                )
                print(f"(n_comp={best_n_comp})")

                # Fit and evaluate PLS (T-only)
                pls = PLSRegression(n_components=best_n_comp, scale=False)
                pls.fit(x_train[..., 0], y_train[..., 0])
                y_pred_pls = pls.predict(x_test[..., 0])
                y_true = combine_channels(y_test, config.channel_weights)
                metrics_pls = evaluation_metrics(y_true, y_pred_pls)
                results.append(
                    {
                        "repeat": repeat,
                        "fold": fold_idx,
                        "method": "PLS",
                        "n_components": best_n_comp,
                        **metrics_pls,
                    }
                )

                # Fit and evaluate NPLS
                npls = NPLS(n_components=best_n_comp, channel_weights=config.channel_weights)
                npls.fit(x_train, y_train)
                y_pred_npls = npls.predict(x_test)
                metrics_npls = evaluation_metrics(y_true, y_pred_npls)
                results.append(
                    {
                        "repeat": repeat,
                        "fold": fold_idx,
                        "method": "NPLS",
                        "n_components": best_n_comp,
                        **metrics_npls,
                    }
                )

                # Compute VIP for this fold
                vip = compute_nvip(npls, x_train, channel_weights=config.channel_weights)
                vip_results.append(
                    {
                        "repeat": repeat,
                        "fold": fold_idx,
                        "aggregate_vip": vip["aggregate"].tolist(),
                        "T_vip": vip["T"].tolist(),
                        "I_vip": vip["I"].tolist(),
                        "F_vip": vip["F"].tolist(),
                    }
                )

                # Fit and evaluate NPLSW
                nplsw = NPLSW(
                    n_components=best_n_comp,
                    channel_weights=config.channel_weights,
                    lambda_indeterminacy=1.0,
                )
                nplsw.fit(x_train, y_train)
                y_pred_nplsw = nplsw.predict(x_test)
                metrics_nplsw = evaluation_metrics(y_true, y_pred_nplsw)
                results.append(
                    {
                        "repeat": repeat,
                        "fold": fold_idx,
                        "method": "NPLSW",
                        "n_components": best_n_comp,
                        **metrics_nplsw,
                    }
                )

    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv(config.output_dir / "micromass_cv_results.csv", index=False)

    # Generate Table 7: PLS vs N-PLS metrics
    table7 = generate_micromass_summary(df)
    table7.to_csv(config.output_dir / "micromass_summary.csv", index=False)

    # Save VIP results
    with open(config.output_dir / "vip_results.json", "w") as f:
        json.dump(vip_results, f)

    # Compute aggregate VIP profile
    aggregate_vip = compute_aggregate_vip(vip_results, metadata["feature_names"])
    aggregate_vip.to_csv(config.output_dir / "aggregate_vip.csv", index=False)

    print(f"\nResults saved to {config.output_dir}/")

    return {
        "results": df,
        "vip_results": vip_results,
        "metadata": metadata,
    }


def select_components_cv(
    x_tif: np.ndarray,
    y_tif: np.ndarray,
    max_components: int,
    n_folds: int,
) -> int:
    """Select optimal number of components using inner CV."""
    import warnings
    
    X = x_tif[..., 0]  # Use T channel for simplicity
    Y = y_tif[..., 0]
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Adjust folds if dataset is small
    actual_folds = min(n_folds, n_samples // 2)
    if actual_folds < 2:
        return 1  # Not enough data for CV
    
    # Limit max components based on data dimensions
    max_possible = min(max_components, n_samples - n_samples // actual_folds - 1, n_features)
    if max_possible < 1:
        return 1

    best_score = -np.inf
    best_n_comp = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for n_comp in range(1, max_possible + 1):
            try:
                pls = PLSRegression(n_components=n_comp, scale=False)
                cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
                y_pred = cross_val_predict(pls, X, Y, cv=cv)
                
                var_y = np.var(Y)
                if var_y > 0:
                    score = 1 - np.mean((Y - y_pred) ** 2) / var_y  # Q² approximation
                else:
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_n_comp = n_comp
            except Exception:
                continue

    return best_n_comp


def generate_micromass_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 7: Summary statistics by method."""
    summary = (
        df.groupby("method")
        .agg(
            {
                "RMSEP": ["mean", "std"],
                "R2": ["mean", "std"],
                "MAE": ["mean", "std"],
                "n_components": "mean",
            }
        )
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def compute_aggregate_vip(
    vip_results: List[Dict], feature_names: List[str]
) -> pd.DataFrame:
    """Compute mean and std of VIP across all folds."""
    all_vips = np.array([r["aggregate_vip"] for r in vip_results])
    mean_vip = np.mean(all_vips, axis=0)
    std_vip = np.std(all_vips, axis=0)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_vip": mean_vip,
            "std_vip": std_vip,
            "important": mean_vip > 1.0,  # VIP > 1 threshold
        }
    ).sort_values("mean_vip", ascending=False)


# ==============================================================================
# Main Entry Point
# ==============================================================================


def run_full_study(
    run_screening: bool = True,
    run_response_surface: bool = True,
    run_micromass: bool = True,
    output_base: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Execute the complete simulation study.

    Parameters
    ----------
    run_screening : bool
        Whether to run Stage 1 screening study
    run_response_surface : bool
        Whether to run Stage 2 response surface study
    run_micromass : bool
        Whether to run Stage 3 MicroMass confirmatory study
    output_base : Path
        Base directory for all outputs

    Returns
    -------
    Dict with results from each stage
    """
    print("=" * 70)
    print("N-PLS vs PLS SIMULATION STUDY")
    print("A Designed Experiment for the Neutrosophic PLS Paper")
    print("=" * 70)

    start_time = time.time()
    results = {}

    # Stage 1: Screening
    if run_screening:
        screening_config = ScreeningConfig(
            output_dir=output_base / "stage1_screening",
            n_replicates=3,
        )
        results["screening"] = run_screening_study(screening_config)

    # Stage 2: Response Surface
    if run_response_surface:
        rs_config = ResponseSurfaceConfig(
            output_dir=output_base / "stage2_response_surface",
            n_replicates=5,
        )
        results["response_surface"] = run_response_surface_study(rs_config)

    # Stage 3: MicroMass
    if run_micromass:
        mm_config = MicroMassConfig(
            output_dir=output_base / "stage3_micromass",
            n_repeats=3,
        )
        results["micromass"] = run_micromass_study(mm_config)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"STUDY COMPLETE")
    print(f"Total elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {output_base}/")
    print("=" * 70)

    # Save study configuration
    study_config = {
        "stages_run": {
            "screening": run_screening,
            "response_surface": run_response_surface,
            "micromass": run_micromass,
        },
        "elapsed_seconds": elapsed,
    }
    with open(output_base / "study_config.json", "w") as f:
        json.dump(study_config, f, indent=2)

    return results


# ==============================================================================
# CLI Interface
# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="N-PLS vs PLS Simulation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full study
  python simulation_study.py --all

  # Run only screening stage
  python simulation_study.py --screening

  # Run response surface with custom output
  python simulation_study.py --response-surface --output results/custom

  # Run only MicroMass confirmatory
  python simulation_study.py --micromass
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three stages of the study",
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="Run Stage 1: Screening (fractional factorial)",
    )
    parser.add_argument(
        "--response-surface",
        action="store_true",
        help="Run Stage 2: Response surface (3-level factorial)",
    )
    parser.add_argument(
        "--micromass",
        action="store_true",
        help="Run Stage 3: MicroMass confirmatory (nested CV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Base output directory (default: results)",
    )

    args = parser.parse_args()

    # Determine which stages to run
    if args.all:
        run_s, run_r, run_m = True, True, True
    else:
        run_s = args.screening
        run_r = args.response_surface
        run_m = args.micromass

        # If nothing specified, default to all
        if not (run_s or run_r or run_m):
            print("No stages specified. Running full study (--all).\n")
            run_s, run_r, run_m = True, True, True

    # Run the study
    run_full_study(
        run_screening=run_s,
        run_response_surface=run_r,
        run_micromass=run_m,
        output_base=Path(args.output),
    )
