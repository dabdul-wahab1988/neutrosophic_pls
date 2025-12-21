#!/usr/bin/env python
"""
Generate Publication-Ready LaTeX Tables from N-PLS Study Results
================================================================

This script reads the simulation study results and generates:
  - Table 1: Screening study factor effects
  - Table 2: Response surface performance summary by scenario  
  - Table 3: Method comparison summary (aggregated)
  - Table 4: MicroMass real-data results
  - Table 5: Pairwise statistical tests

Usage:
  python scripts/generate_tables.py --results results --output tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format mean ± std for LaTeX."""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def format_pvalue(p: float) -> str:
    """Format p-value with significance markers."""
    if p < 0.001:
        return "$<$0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def bold_best(values: List[str], best_idx: int) -> List[str]:
    """Bold the best value in a list."""
    result = values.copy()
    result[best_idx] = f"\\textbf{{{result[best_idx]}}}"
    return result


# ==============================================================================
# Table 1: Screening Factor Effects
# ==============================================================================

def generate_screening_table(results_dir: Path, output_dir: Path) -> str:
    """Generate Table 1: Factor effects from screening study."""
    effects_path = results_dir / "stage1_screening" / "factor_effects.csv"
    
    if not effects_path.exists():
        print(f"Warning: {effects_path} not found")
        return ""
    
    df = pd.read_csv(effects_path)
    
    # Pivot to get methods as columns
    pivot_rmsep = df[df["metric"] == "RMSEP"].pivot(
        index="factor", columns="method", values="effect"
    )
    pivot_r2 = df[df["metric"] == "R2"].pivot(
        index="factor", columns="method", values="effect"
    )
    
    # Save CSV version
    csv_df = pivot_rmsep.add_suffix("_RMSEP").join(pivot_r2.add_suffix("_R2"))
    csv_df.to_csv(output_dir / "table1_screening.csv")
    
    # Build LaTeX table
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main effects from screening study (High $-$ Low level). Positive effects on RMSEP indicate worse performance at high factor levels; negative effects on $R^2$ indicate worse fit.}",
        r"\label{tab:screening}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Effect on RMSEP} & \multicolumn{3}{c}{Effect on $R^2$} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Factor & PLS$_T$ & NPLS & NPLSW & PLS$_T$ & NPLS & NPLSW \\",
        r"\midrule",
    ]
    
    factor_names = {
        "sigma_T": r"$\sigma_T$ (Truth noise)",
        "sigma_I": r"$\sigma_I$ (Indeterminacy)",
        "falsity_prop": r"$\pi_F$ (Falsity proportion)",
        "n_samples": r"$n$ (Sample size)",
    }
    
    methods = ["PLS_T", "NPLS", "NPLSW"]
    
    for factor in pivot_rmsep.index:
        row = [factor_names.get(factor, factor)]
        
        # RMSEP effects
        for method in methods:
            if method in pivot_rmsep.columns:
                val = pivot_rmsep.loc[factor, method]
                row.append(f"{val:+.3f}")
            else:
                row.append("--")
        
        # R2 effects
        for method in methods:
            if method in pivot_r2.columns:
                val = pivot_r2.loc[factor, method]
                row.append(f"{val:+.3f}")
            else:
                row.append("--")
        
        latex.append(" & ".join(row) + r" \\")
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(latex)
    
    # Save to file
    output_path = output_dir / "table1_screening.tex"
    output_path.write_text(table_str)
    print(f"Generated {output_path}")
    
    return table_str


# ==============================================================================
# Table 2: Response Surface Summary by Scenario
# ==============================================================================

def generate_response_surface_table(results_dir: Path, output_dir: Path) -> str:
    """Generate Table 2: Performance by scenario."""
    perf_path = results_dir / "stage2_response_surface" / "performance_summary.csv"
    
    if not perf_path.exists():
        print(f"Warning: {perf_path} not found")
        return ""
    
    df = pd.read_csv(perf_path)
    
    # Select key methods
    methods = ["PLS_T", "NPLS", "NPLSW"]
    df_filtered = df[df["method"].isin(methods)].copy()
    
    # Save CSV version
    csv_df = df_filtered[["scenario_idx", "sigma_T", "sigma_I", "falsity_prop", "method", 
                          "RMSEP_mean", "RMSEP_std", "R2_mean", "R2_std", "MAE_mean", "MAE_std"]]
    csv_df.to_csv(output_dir / "table2_response_surface.csv", index=False)
    
    df = df_filtered
    
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Predictive performance across experimental scenarios. Values shown as mean $\pm$ SD over 10 replicates. Best method per scenario in \textbf{bold}.}",
        r"\label{tab:response_surface}",
        r"\small",
        r"\begin{tabular}{cccclccc}",
        r"\toprule",
        r"Scenario & $\sigma_T$ & $\sigma_I$ & $\pi_F$ & Method & RMSEP & $R^2$ & MAE \\",
        r"\midrule",
    ]
    
    scenarios = sorted(df["scenario_idx"].unique())
    
    for s_idx in scenarios:
        scenario_df = df[df["scenario_idx"] == s_idx]
        
        # Get scenario parameters
        sigma_T = scenario_df["sigma_T"].iloc[0]
        sigma_I = scenario_df["sigma_I"].iloc[0]
        falsity = scenario_df["falsity_prop"].iloc[0]
        
        # Find best method for each metric
        best_rmsep_idx = scenario_df["RMSEP_mean"].idxmin()
        best_r2_idx = scenario_df["R2_mean"].idxmax()
        
        for i, (_, row) in enumerate(scenario_df.iterrows()):
            method = row["method"]
            method_display = method.replace("_", r"$_")
            if method == "PLS_T":
                method_display = r"PLS$_T$"
            
            rmsep = format_mean_std(row["RMSEP_mean"], row["RMSEP_std"])
            r2 = format_mean_std(row["R2_mean"], row["R2_std"])
            mae = format_mean_std(row["MAE_mean"], row["MAE_std"])
            
            # Bold best values
            if row.name == best_rmsep_idx:
                rmsep = r"\textbf{" + rmsep + "}"
            if row.name == best_r2_idx:
                r2 = r"\textbf{" + r2 + "}"
            
            if i == 0:
                # First row of scenario - include parameters
                latex.append(
                    f"{s_idx+1} & {sigma_T:.2f} & {sigma_I:.2f} & {falsity:.2f} & "
                    f"{method_display} & {rmsep} & {r2} & {mae} \\\\"
                )
            else:
                latex.append(
                    f" &  &  &  & {method_display} & {rmsep} & {r2} & {mae} \\\\"
                )
        
        if s_idx < max(scenarios):
            latex.append(r"\addlinespace")
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(latex)
    
    output_path = output_dir / "table2_response_surface.tex"
    output_path.write_text(table_str)
    print(f"Generated {output_path}")
    
    return table_str


# ==============================================================================
# Table 3: Overall Method Comparison
# ==============================================================================

def generate_comparison_table(results_dir: Path, output_dir: Path) -> str:
    """Generate Table 3: Overall method comparison aggregated."""
    perf_path = results_dir / "stage2_response_surface" / "performance_summary.csv"
    
    if not perf_path.exists():
        print(f"Warning: {perf_path} not found")
        return ""
    
    df = pd.read_csv(perf_path)
    
    # Aggregate across all scenarios
    methods = ["PLS_T", "NPLS", "NPLSW", "PLS_collapsed"]
    summary = []
    
    for method in methods:
        method_df = df[df["method"] == method]
        if len(method_df) == 0:
            continue
            
        summary.append({
            "Method": method,
            "RMSEP_mean": method_df["RMSEP_mean"].mean(),
            "RMSEP_std": method_df["RMSEP_mean"].std(),
            "R2_mean": method_df["R2_mean"].mean(),
            "R2_std": method_df["R2_mean"].std(),
            "MAE_mean": method_df["MAE_mean"].mean(),
            "MAE_std": method_df["MAE_mean"].std(),
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save CSV version (will add wins later)
    
    # Count wins
    wins = {m: {"RMSEP": 0, "R2": 0} for m in methods}
    for s_idx in df["scenario_idx"].unique():
        scenario_df = df[df["scenario_idx"] == s_idx]
        best_rmsep = scenario_df.loc[scenario_df["RMSEP_mean"].idxmin(), "method"]
        best_r2 = scenario_df.loc[scenario_df["R2_mean"].idxmax(), "method"]
        if best_rmsep in wins:
            wins[best_rmsep]["RMSEP"] += 1
        if best_r2 in wins:
            wins[best_r2]["R2"] += 1
    
    # Add wins to summary and save CSV
    summary_df["RMSEP_wins"] = summary_df["Method"].map(lambda m: wins.get(m, {}).get("RMSEP", 0))
    summary_df["R2_wins"] = summary_df["Method"].map(lambda m: wins.get(m, {}).get("R2", 0))
    summary_df.to_csv(output_dir / "table3_comparison.csv", index=False)
    
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Overall method comparison across all 27 scenarios. Wins indicates number of scenarios where method achieved best performance.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & RMSEP (mean $\pm$ SD) & $R^2$ (mean $\pm$ SD) & RMSEP Wins & $R^2$ Wins \\",
        r"\midrule",
    ]
    
    method_names = {
        "PLS_T": r"PLS$_T$ (Truth only)",
        "NPLS": r"NPLS (Neutrosophic)",
        "NPLSW": r"NPLSW (Weighted)",
        "PLS_collapsed": r"PLS (Collapsed T+I+F)",
    }
    
    for _, row in summary_df.iterrows():
        method = row["Method"]
        rmsep = format_mean_std(row["RMSEP_mean"], row["RMSEP_std"])
        r2 = format_mean_std(row["R2_mean"], row["R2_std"])
        rmsep_wins = wins.get(method, {}).get("RMSEP", 0)
        r2_wins = wins.get(method, {}).get("R2", 0)
        
        latex.append(
            f"{method_names.get(method, method)} & {rmsep} & {r2} & {rmsep_wins} & {r2_wins} \\\\"
        )
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(latex)
    
    output_path = output_dir / "table3_comparison.tex"
    output_path.write_text(table_str)
    print(f"Generated {output_path}")
    
    return table_str


# ==============================================================================
# Table 4: MicroMass Results
# ==============================================================================

def generate_micromass_table(results_dir: Path, output_dir: Path) -> str:
    """Generate Table 4: MicroMass real-data results."""
    mm_path = results_dir / "stage3_micromass" / "micromass_summary.csv"
    
    if not mm_path.exists():
        print(f"Warning: {mm_path} not found")
        return ""
    
    df = pd.read_csv(mm_path)
    
    # Save CSV version (already exists but copy for consistency)
    df.to_csv(output_dir / "table4_micromass.csv", index=False)
    
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Real-data validation on MicroMass dataset using nested cross-validation. Best values in \textbf{bold}.}",
        r"\label{tab:micromass}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & RMSEP & $R^2$ & MAE \\",
        r"\midrule",
    ]
    
    # Find best for each metric
    best_rmsep_idx = df["RMSEP_mean"].idxmin()
    best_r2_idx = df["R2_mean"].idxmax()
    best_mae_idx = df["MAE_mean"].idxmin()
    
    for idx, row in df.iterrows():
        method = row["method"]
        
        rmsep = format_mean_std(row["RMSEP_mean"], row["RMSEP_std"])
        r2 = format_mean_std(row["R2_mean"], row["R2_std"])
        mae = format_mean_std(row["MAE_mean"], row["MAE_std"])
        
        if idx == best_rmsep_idx:
            rmsep = r"\textbf{" + rmsep + "}"
        if idx == best_r2_idx:
            r2 = r"\textbf{" + r2 + "}"
        if idx == best_mae_idx:
            mae = r"\textbf{" + mae + "}"
        
        latex.append(f"{method} & {rmsep} & {r2} & {mae} \\\\")
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(latex)
    
    output_path = output_dir / "table4_micromass.tex"
    output_path.write_text(table_str)
    print(f"Generated {output_path}")
    
    return table_str


# ==============================================================================
# Table 5: NPLS Improvement Analysis
# ==============================================================================

def generate_improvement_table(results_dir: Path, output_dir: Path) -> str:
    """Generate Table 5: NPLS improvement over PLS by condition."""
    perf_path = results_dir / "stage2_response_surface" / "performance_summary.csv"
    
    if not perf_path.exists():
        print(f"Warning: {perf_path} not found")
        return ""
    
    df = pd.read_csv(perf_path)
    
    # Calculate improvement of NPLS over PLS_T
    improvements = []
    
    for s_idx in df["scenario_idx"].unique():
        scenario_df = df[df["scenario_idx"] == s_idx]
        
        pls_row = scenario_df[scenario_df["method"] == "PLS_T"].iloc[0]
        npls_row = scenario_df[scenario_df["method"] == "NPLS"].iloc[0]
        nplsw_row = scenario_df[scenario_df["method"] == "NPLSW"].iloc[0]
        
        pls_rmsep = pls_row["RMSEP_mean"]
        npls_rmsep = npls_row["RMSEP_mean"]
        nplsw_rmsep = nplsw_row["RMSEP_mean"]
        
        improvements.append({
            "scenario": s_idx + 1,
            "sigma_T": pls_row["sigma_T"],
            "sigma_I": pls_row["sigma_I"],
            "falsity_prop": pls_row["falsity_prop"],
            "NPLS_improvement": 100 * (pls_rmsep - npls_rmsep) / pls_rmsep,
            "NPLSW_improvement": 100 * (pls_rmsep - nplsw_rmsep) / pls_rmsep,
        })
    
    imp_df = pd.DataFrame(improvements)
    
    # Save CSV version
    imp_df.to_csv(output_dir / "table5_improvement.csv", index=False)
    
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{RMSEP improvement (\%) of NPLS and NPLSW over PLS$_T$ by scenario. Positive values indicate NPLS/NPLSW outperforms PLS$_T$.}",
        r"\label{tab:improvement}",
        r"\small",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Scenario & $\sigma_T$ & $\sigma_I$ & $\pi_F$ & NPLS (\%) & NPLSW (\%) \\",
        r"\midrule",
    ]
    
    for _, row in imp_df.iterrows():
        npls_imp = row["NPLS_improvement"]
        nplsw_imp = row["NPLSW_improvement"]
        
        # Format with sign
        npls_str = f"{npls_imp:+.1f}"
        nplsw_str = f"{nplsw_imp:+.1f}"
        
        # Bold positive improvements
        if npls_imp > 0:
            npls_str = r"\textbf{" + npls_str + "}"
        if nplsw_imp > 0:
            nplsw_str = r"\textbf{" + nplsw_str + "}"
        
        latex.append(
            f"{int(row['scenario'])} & {row['sigma_T']:.2f} & {row['sigma_I']:.2f} & "
            f"{row['falsity_prop']:.2f} & {npls_str} & {nplsw_str} \\\\"
        )
    
    # Add summary row
    mean_npls = imp_df["NPLS_improvement"].mean()
    mean_nplsw = imp_df["NPLSW_improvement"].mean()
    latex.append(r"\midrule")
    latex.append(f"\\textit{{Mean}} &  &  &  & {mean_npls:+.1f} & {mean_nplsw:+.1f} \\\\")
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_str = "\n".join(latex)
    
    output_path = output_dir / "table5_improvement.tex"
    output_path.write_text(table_str)
    print(f"Generated {output_path}")
    
    return table_str


# ==============================================================================
# Summary Statistics
# ==============================================================================

def print_summary_statistics(results_dir: Path) -> None:
    """Print key summary statistics for the paper text."""
    print("\n" + "="*70)
    print("KEY STATISTICS FOR PAPER")
    print("="*70)
    
    # Response surface stats
    perf_path = results_dir / "stage2_response_surface" / "performance_summary.csv"
    if perf_path.exists():
        df = pd.read_csv(perf_path)
        
        # Count where NPLS beats PLS_T
        npls_wins = 0
        nplsw_wins = 0
        total = 0
        
        for s_idx in df["scenario_idx"].unique():
            scenario_df = df[df["scenario_idx"] == s_idx]
            pls_rmsep = scenario_df[scenario_df["method"] == "PLS_T"]["RMSEP_mean"].iloc[0]
            npls_rmsep = scenario_df[scenario_df["method"] == "NPLS"]["RMSEP_mean"].iloc[0]
            nplsw_rmsep = scenario_df[scenario_df["method"] == "NPLSW"]["RMSEP_mean"].iloc[0]
            
            total += 1
            if npls_rmsep < pls_rmsep:
                npls_wins += 1
            if nplsw_rmsep < pls_rmsep:
                nplsw_wins += 1
        
        print(f"\nResponse Surface (27 scenarios):")
        print(f"  NPLS outperforms PLS_T in {npls_wins}/{total} scenarios ({100*npls_wins/total:.0f}%)")
        print(f"  NPLSW outperforms PLS_T in {nplsw_wins}/{total} scenarios ({100*nplsw_wins/total:.0f}%)")
        
        # Where indeterminacy > 0
        df_with_I = df[df["sigma_I"] > 0]
        scenarios_with_I = df_with_I["scenario_idx"].unique()
        npls_wins_I = 0
        for s_idx in scenarios_with_I:
            scenario_df = df[df["scenario_idx"] == s_idx]
            pls_rmsep = scenario_df[scenario_df["method"] == "PLS_T"]["RMSEP_mean"].iloc[0]
            npls_rmsep = scenario_df[scenario_df["method"] == "NPLS"]["RMSEP_mean"].iloc[0]
            if npls_rmsep < pls_rmsep:
                npls_wins_I += 1
        
        print(f"  NPLS outperforms PLS_T in {npls_wins_I}/{len(scenarios_with_I)} scenarios with σ_I > 0 ({100*npls_wins_I/len(scenarios_with_I):.0f}%)")
    
    # MicroMass stats
    mm_path = results_dir / "stage3_micromass" / "micromass_summary.csv"
    if mm_path.exists():
        mm_df = pd.read_csv(mm_path)
        print(f"\nMicroMass Real Data:")
        for _, row in mm_df.iterrows():
            print(f"  {row['method']}: RMSEP = {row['RMSEP_mean']:.3f} ± {row['RMSEP_std']:.3f}, R² = {row['R2_mean']:.3f}")


# ==============================================================================
# Main
# ==============================================================================

def generate_all_tables(results_dir: Path, output_dir: Path) -> None:
    """Generate all publication tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING PUBLICATION TABLES")
    print("="*70)
    
    generate_screening_table(results_dir, output_dir)
    generate_response_surface_table(results_dir, output_dir)
    generate_comparison_table(results_dir, output_dir)
    generate_micromass_table(results_dir, output_dir)
    generate_improvement_table(results_dir, output_dir)
    
    print_summary_statistics(results_dir)
    
    print("\n" + "="*70)
    print(f"All tables saved to {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from N-PLS study results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="Directory containing study results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tables",
        help="Output directory for LaTeX tables",
    )
    
    args = parser.parse_args()
    generate_all_tables(Path(args.results), Path(args.output))
