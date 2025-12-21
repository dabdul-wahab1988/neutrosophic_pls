#!/usr/bin/env python
"""
Visualization Module for N-PLS Simulation Study
================================================

Generates publication-quality figures:
  - Figure 4: RMSEP / R² vs scenario (method comparison)
  - Figure 5: Component recovery (mean_corr, min_corr) vs scenario
  - Figure 6: N-VIP profiles from MicroMass
  - Figure 7: Score plots with feature annotations

Author: NeutroProject
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Style settings for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette for methods
METHOD_COLORS = {
    "PLS_T": "#1f77b4",        # Blue
    "PLS_collapsed": "#aec7e8", # Light blue
    "NPLS": "#ff7f0e",          # Orange
    "NPLSW": "#d62728",         # Red
    "PLS": "#1f77b4",           # Blue (for MicroMass)
}

METHOD_MARKERS = {
    "PLS_T": "o",
    "PLS_collapsed": "s",
    "NPLS": "^",
    "NPLSW": "D",
    "PLS": "o",
}

# Display labels for methods (no underscores)
METHOD_LABELS = {
    "PLS_T": "PLS (Truth)",
    "PLS_collapsed": "PLS (Collapsed)",
    "NPLS": "N-PLS",
    "NPLSW": "N-PLS-W",
    "PLS": "PLS",
}

# Display labels for factors/parameters (no underscores)
FACTOR_LABELS = {
    "sigma_T": r"$\sigma_T$ (Truth noise)",
    "sigma_I": r"$\sigma_I$ (Indeterminacy)",
    "falsity_prop": r"$\pi_F$ (Falsity proportion)",
    "n_samples": r"$n$ (Sample size)",
    "n_features": r"$p$ (Features)",
    "n_components_fit": r"$K$ (Components)",
    "weight_pattern": "Weight Pattern",
}


# ==============================================================================
# Figure 4: Predictive Performance vs Scenario
# ==============================================================================


def plot_predictive_performance(
    df: pd.DataFrame,
    output_dir: Path,
    figsize: Tuple[float, float] = (10, 10),
) -> None:
    """
    Generate Figure 4: RMSEP and R² vs scenario index for each method.

    Creates a two-panel figure (2 rows x 1 column):
      - Top panel: RMSEP by scenario
      - Bottom panel: R² by scenario
    """
    print("Generating Figure 4: Predictive Performance...")

    # Aggregate by scenario and method
    summary = df.groupby(["scenario_idx", "method"]).agg({
        "RMSEP": ["mean", "std"],
        "R2": ["mean", "std"],
        "sigma_T": "first",
        "sigma_I": "first",
        "falsity_prop": "first",
    }).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    scenarios = sorted(summary["scenario_idx"].unique())
    methods = ["PLS_T", "NPLS", "NPLSW"]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=False)

    # Panel A: RMSEP
    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(scenarios))

    for i, method in enumerate(methods):
        method_data = summary[summary["method"] == method].set_index("scenario_idx")
        means = [method_data.loc[s, "RMSEP_mean"] if s in method_data.index else np.nan for s in scenarios]
        stds = [method_data.loc[s, "RMSEP_std"] if s in method_data.index else np.nan for s in scenarios]

        ax1.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            capsize=2,
            alpha=0.85,
        )

    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("RMSEP")
    ax1.set_title("(A) Root Mean Square Error of Prediction")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([str(s + 1) for s in scenarios], rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: R²
    ax2 = axes[1]

    for i, method in enumerate(methods):
        method_data = summary[summary["method"] == method].set_index("scenario_idx")
        means = [method_data.loc[s, "R2_mean"] if s in method_data.index else np.nan for s in scenarios]
        stds = [method_data.loc[s, "R2_std"] if s in method_data.index else np.nan for s in scenarios]

        ax2.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            capsize=2,
            alpha=0.85,
        )

    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("R²")
    ax2.set_title("(B) Coefficient of Determination")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(s + 1) for s in scenarios], rotation=45, ha="right")
    ax2.legend(loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(str(output_dir / "figure4_predictive_performance.pdf"))
    fig.savefig(str(output_dir / "figure4_predictive_performance.png"))
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure4_predictive_performance.pdf")


def plot_performance_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "RMSEP",
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Generate a heatmap showing metric values across noise conditions.
    Useful for visualizing the response surface.
    """
    print(f"Generating heatmap for {metric}...")

    # Pivot for each method
    methods = ["PLS_T", "NPLS"]

    fig, axes = plt.subplots(1, len(methods), figsize=figsize, sharey=True)

    for ax, method in zip(axes, methods):
        method_df = df[df["method"] == method]
        pivot = method_df.pivot_table(
            values=metric,
            index="sigma_T",
            columns="sigma_I",
            aggfunc="mean",
        )

        im = ax.imshow(pivot.values, cmap="RdYlGn_r" if metric == "RMSEP" else "RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
        ax.set_xlabel(FACTOR_LABELS.get("sigma_I", "σ_I (Indeterminacy)"))
        ax.set_ylabel(FACTOR_LABELS.get("sigma_T", "σ_T (Truth noise)"))
        ax.set_title(f"{METHOD_LABELS.get(method, method)}: {metric}")

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(output_dir / f"heatmap_{metric.lower()}.pdf"))
    fig.savefig(str(output_dir / f"heatmap_{metric.lower()}.png"))
    plt.close(fig)


def plot_improvement_surface(
    df: pd.DataFrame,
    output_dir: Path,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """
    Plot the improvement of NPLS over PLS as a function of noise parameters.
    """
    print("Generating improvement surface plot...")

    # Compute improvement
    pivot_pls = df[df["method"] == "PLS_T"].pivot_table(
        values="RMSEP", index=["sigma_T", "sigma_I"], columns="falsity_prop", aggfunc="mean"
    )
    pivot_npls = df[df["method"] == "NPLS"].pivot_table(
        values="RMSEP", index=["sigma_T", "sigma_I"], columns="falsity_prop", aggfunc="mean"
    )

    improvement = (pivot_pls - pivot_npls) / pivot_pls * 100

    # Plot for middle falsity level
    if improvement.shape[1] >= 2:
        mid_col = improvement.columns[len(improvement.columns) // 2]
        plot_data = improvement[mid_col].unstack()

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_data.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=30)

        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in plot_data.columns])
        ax.set_yticks(range(len(plot_data.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in plot_data.index])
        ax.set_xlabel(FACTOR_LABELS.get("sigma_I", "σ_I (Indeterminacy)"))
        ax.set_ylabel(FACTOR_LABELS.get("sigma_T", "σ_T (Truth noise)"))
        ax.set_title(f"RMSEP Improvement of N-PLS over PLS (%)\n(Falsity prop = {mid_col})")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Improvement (%)")

        # Add value annotations
        for i in range(len(plot_data.index)):
            for j in range(len(plot_data.columns)):
                val = plot_data.values[i, j]
                color = "white" if abs(val) > 15 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color=color)

        plt.tight_layout()
        fig.savefig(str(output_dir / "improvement_surface.pdf"))
        fig.savefig(str(output_dir / "improvement_surface.png"))
        plt.close(fig)


# ==============================================================================
# Figure 5: Component Recovery
# ==============================================================================


def plot_component_recovery(
    df: pd.DataFrame,
    output_dir: Path,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Generate Figure 5: Latent component recovery (mean_corr, min_corr) vs scenario.
    """
    print("Generating Figure 5: Component Recovery...")

    # Check which recovery columns exist
    recovery_cols = [c for c in df.columns if "corr" in c.lower()]
    if not recovery_cols:
        print("  Warning: No component recovery columns found. Skipping Figure 5.")
        return

    # Try to find NPLS-specific recovery columns
    npls_mean_col = next((c for c in recovery_cols if "npls_mean" in c.lower()), None)
    npls_min_col = next((c for c in recovery_cols if "npls_min" in c.lower()), None)
    pls_mean_col = next((c for c in recovery_cols if "pls_mean" in c.lower()), None)
    pls_min_col = next((c for c in recovery_cols if "pls_min" in c.lower()), None)

    # Aggregate by scenario and method
    agg_dict = {}
    for col in recovery_cols:
        agg_dict[col] = ["mean", "std"]

    summary = df.groupby(["scenario_idx", "method"]).agg(agg_dict).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    scenarios = sorted(summary["scenario_idx"].unique())
    methods = ["PLS_T", "NPLS", "NPLSW"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Panel A: Mean correlation
    ax1 = axes[0]
    for method in methods:
        method_data = summary[summary["method"] == method].sort_values(by="scenario_idx")

        # Find the mean correlation column for this method
        if method == "NPLS" and npls_mean_col:
            col = f"{npls_mean_col}_mean"
        elif "PLS" in method and pls_mean_col:
            col = f"{pls_mean_col}_mean"
        else:
            col = next((c for c in summary.columns if "mean_corr_mean" in c.lower()), None)

        if col and col in method_data.columns:
            ax1.plot(
                method_data["scenario_idx"] + 1,
                method_data[col],
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "gray"),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
                markersize=6,
            )

    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Correlation with True Latent Factors")
    ax1.set_title("(A) Mean Component Recovery")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Panel B: Min correlation
    ax2 = axes[1]
    for method in methods:
        method_data = summary[summary["method"] == method].sort_values(by="scenario_idx")

        # Find the min correlation column for this method
        if method == "NPLS" and npls_min_col:
            col = f"{npls_min_col}_mean"
        elif "PLS" in method and pls_min_col:
            col = f"{pls_min_col}_mean"
        else:
            col = next((c for c in summary.columns if "min_corr_mean" in c.lower()), None)

        if col and col in method_data.columns:
            ax2.plot(
                method_data["scenario_idx"] + 1,
                method_data[col],
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "gray"),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
                markersize=6,
            )

    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Correlation with True Latent Factors")
    ax2.set_title("(B) Minimum Component Recovery")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(str(output_dir / "figure5_component_recovery.pdf"))
    fig.savefig(str(output_dir / "figure5_component_recovery.png"))
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure5_component_recovery.pdf")


# ==============================================================================
# Figure 6: N-VIP Profiles (MicroMass)
# ==============================================================================


def plot_nvip_profile(
    vip_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 30,
    figsize: Tuple[float, float] = (10, 10),
) -> None:
    """
    Generate Figure 6: N-VIP profile from MicroMass data.
    Shows top features by VIP, colored by importance (VIP > 1 or not).
    """
    print("Generating Figure 6: N-VIP Profile...")

    # Simply take top N features by VIP - color will show importance
    vip_sorted = vip_df.nlargest(top_n, "mean_vip")

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(vip_sorted))
    bars = ax.barh(
        y_pos,
        vip_sorted["mean_vip"],
        xerr=vip_sorted["std_vip"],
        capsize=3,
        color=np.where(vip_sorted["important"], "#ff7f0e", "#1f77b4"),
        alpha=0.85,
    )

    # Add VIP > 1 threshold line
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="VIP = 1 threshold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vip_sorted["feature"])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Variable Importance in Projection (N-VIP)")
    ax.set_title(f"Top {top_n} Features by N-VIP (MicroMass Dataset)")
    ax.grid(axis="x", alpha=0.3)

    # Color legend
    important_patch = mpatches.Patch(color="#ff7f0e", label="VIP > 1 (Important)")
    other_patch = mpatches.Patch(color="#1f77b4", label="VIP ≤ 1")
    ax.legend(handles=[important_patch, other_patch, Line2D([0], [0], color="red", linestyle="--", label="Threshold")],
              loc="lower right")

    plt.tight_layout()
    fig.savefig(str(output_dir / "figure6_nvip_profile.pdf"))
    fig.savefig(str(output_dir / "figure6_nvip_profile.png"))
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure6_nvip_profile.pdf")


def plot_vip_channels(
    vip_results: List[Dict],
    feature_names: List[str],
    output_dir: Path,
    top_n: int = 15,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Plot VIP decomposed by T/I/F channels.
    """
    print("Generating channel-decomposed VIP plot...")

    # Aggregate VIP across folds
    all_t = np.mean([r["T_vip"] for r in vip_results], axis=0)
    all_i = np.mean([r["I_vip"] for r in vip_results], axis=0)
    all_f = np.mean([r["F_vip"] for r in vip_results], axis=0)
    all_agg = np.mean([r["aggregate_vip"] for r in vip_results], axis=0)

    # Create DataFrame
    vip_df = pd.DataFrame({
        "feature": feature_names,
        "T": all_t,
        "I": all_i,
        "F": all_f,
        "aggregate": all_agg,
    })

    # Sort by aggregate and take top N
    vip_sorted = vip_df.nlargest(top_n, "aggregate")

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(vip_sorted))
    width = 0.2

    ax.bar(x - width, vip_sorted["T"], width, label="Truth (T)", color="#2ecc71", alpha=0.85)
    ax.bar(x, vip_sorted["I"], width, label="Indeterminacy (I)", color="#f39c12", alpha=0.85)
    ax.bar(x + width, vip_sorted["F"], width, label="Falsity (F)", color="#e74c3c", alpha=0.85)

    ax.axhline(y=1.0 / 3, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(vip_sorted["feature"], rotation=45, ha="right")
    ax.set_ylabel("Channel VIP Contribution")
    ax.set_title(f"Top {top_n} Features: VIP Decomposition by T/I/F Channels")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_dir / "vip_channels.pdf"))
    fig.savefig(str(output_dir / "vip_channels.png"))
    plt.close(fig)


# ==============================================================================
# Figure 7: Score Plots
# ==============================================================================


def plot_score_plot(
    scores: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    loadings: Optional[np.ndarray] = None,
    output_dir: Path = Path("."),
    title: str = "Score Plot",
    figsize: Tuple[float, float] = (8, 7),
) -> None:
    """
    Generate Figure 7: Score plot colored by class with optional loading annotations.
    """
    print(f"Generating {title}...")

    fig, ax = plt.subplots(figsize=figsize)

    # Color by class
    unique_classes = np.unique(y)
    colors = cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        mask = y.ravel() == cls
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=[colors[i]],
            label=f"Class {int(cls)}",
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(loc="best", title="Class")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    # Add loading arrows if provided (biplot style)
    if loadings is not None and feature_names is not None:
        # Scale loadings for visibility
        scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.8

        # Only plot top features by loading magnitude
        loading_mag = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        top_idx = np.argsort(loading_mag)[-10:]  # Top 10

        for idx in top_idx:
            ax.annotate(
                "",
                xy=(loadings[idx, 0] * scale, loadings[idx, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
            )
            ax.text(
                loadings[idx, 0] * scale * 1.1,
                loadings[idx, 1] * scale * 1.1,
                feature_names[idx],
                fontsize=7,
                color="red",
                alpha=0.8,
            )

    plt.tight_layout()

    filename = title.lower().replace(" ", "_").replace(":", "")
    fig.savefig(str(output_dir / f"{filename}.pdf"))
    fig.savefig(str(output_dir / f"{filename}.png"))
    plt.close(fig)

    print(f"  Saved to {output_dir}/{filename}.pdf")


# ==============================================================================
# Screening Factor Effects Plot
# ==============================================================================


def plot_factor_effects(
    effects_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "RMSEP",
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Generate Pareto chart of factor effects from screening study.
    """
    print(f"Generating factor effects plot for {metric}...")

    # Filter for the metric
    metric_effects = effects_df[effects_df["metric"] == metric].copy()

    # Sort by absolute effect
    metric_effects = metric_effects.sort_values(by="abs_effect", ascending=False)

    # Pivot for grouped bar chart
    pivot = metric_effects.pivot(index="factor", columns="method", values="effect")

    # Rename columns (methods) to display labels
    pivot.columns = [METHOD_LABELS.get(m, m) for m in pivot.columns]

    # Rename index (factors) to display labels
    pivot.index = [FACTOR_LABELS.get(f, f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for renamed columns
    color_map = {METHOD_LABELS.get(m, m): METHOD_COLORS.get(m, "gray") for m in METHOD_COLORS}
    pivot.plot(kind="barh", ax=ax, color=[color_map.get(m, "gray") for m in pivot.columns])

    ax.set_xlabel(f"Effect on {metric} (High - Low)")
    ax.set_ylabel("Factor")
    ax.set_title(f"Main Effects on {metric} (Screening Study)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.legend(title="Method", loc="best")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_dir / f"factor_effects_{metric.lower()}.pdf"))
    fig.savefig(str(output_dir / f"factor_effects_{metric.lower()}.png"))
    plt.close(fig)


# ==============================================================================
# Main Visualization Runner
# ==============================================================================


def generate_all_figures(results_dir: Path = Path("results")) -> None:
    """
    Generate all figures from saved study results.
    """
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    # Stage 1: Screening figures
    screening_dir = results_dir / "stage1_screening"
    if screening_dir.exists():
        print("\n--- Stage 1: Screening Figures ---")
        effects_path = screening_dir / "factor_effects.csv"
        if effects_path.exists():
            effects_df = pd.read_csv(effects_path)
            plot_factor_effects(effects_df, screening_dir, metric="RMSEP")
            plot_factor_effects(effects_df, screening_dir, metric="R2")

    # Stage 2: Response surface figures
    rs_dir = results_dir / "stage2_response_surface"
    if rs_dir.exists():
        print("\n--- Stage 2: Response Surface Figures ---")
        raw_path = rs_dir / "response_surface_raw_results.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            plot_predictive_performance(df, rs_dir)
            plot_component_recovery(df, rs_dir)
            plot_performance_heatmap(df, rs_dir, metric="RMSEP")
            plot_performance_heatmap(df, rs_dir, metric="R2")
            plot_improvement_surface(df, rs_dir)

    # Stage 3: MicroMass figures
    mm_dir = results_dir / "stage3_micromass"
    if mm_dir.exists():
        print("\n--- Stage 3: MicroMass Figures ---")

        # VIP profile
        vip_path = mm_dir / "aggregate_vip.csv"
        if vip_path.exists():
            vip_df = pd.read_csv(vip_path)
            plot_nvip_profile(vip_df, mm_dir, top_n=20)

        # Channel VIP decomposition
        vip_json_path = mm_dir / "vip_results.json"
        dataset_path = mm_dir / "dataset_summary.json"
        if vip_json_path.exists():
            with open(vip_json_path, "r") as f:
                vip_results = json.load(f)

            # Get feature names
            if dataset_path.exists():
                with open(dataset_path, "r") as f:
                    dataset_info = json.load(f)
                n_features = dataset_info.get("Features", len(vip_results[0]["aggregate_vip"]))
            else:
                n_features = len(vip_results[0]["aggregate_vip"])

            feature_names = [f"f{i}" for i in range(n_features)]
            plot_vip_channels(vip_results, feature_names, mm_dir)

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)


# ==============================================================================
# CLI Interface
# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualization figures for N-PLS simulation study"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing study results (default: results)",
    )

    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir))
