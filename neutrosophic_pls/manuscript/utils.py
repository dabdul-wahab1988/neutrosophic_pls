from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

from ..data_loader import DatasetConfig, load_dataset, load_dataframe


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame to CSV, ensuring parent directories exist."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def save_tex(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame to LaTeX (without index) and ensure directories."""
    ensure_dir(path.parent)
    df.to_latex(path, index=False, **kwargs)


def save_json(data: Dict, path: Path) -> None:
    """Persist a JSON file with indentation."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column using a priority list or fallback to last column."""
    priority = ["Protein", "y", "Y", "target", "Target", "Class", "label"]
    for col in priority:
        if col in df.columns:
            return col
    warnings.warn(
        "Target column not explicitly provided; defaulting to last column.",
        RuntimeWarning,
    )
    return str(df.columns[-1])


def detect_exclude_columns(df: pd.DataFrame) -> List[str]:
    """Columns to exclude from features if present."""
    candidates = ["ID", "Id", "SampleID", "sample_id"]
    return [c for c in candidates if c in df.columns]


def _is_numeric_string(value: str) -> bool:
    """Return True if a string can be parsed as a float."""
    try:
        float(str(value))
        return True
    except Exception:
        return False


def is_spectral(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Heuristic: treat dataset as spectral if >=80% of feature names are numeric strings.
    """
    if not feature_cols:
        return False
    numeric_named = sum(1 for c in feature_cols if _is_numeric_string(c))
    return numeric_named / len(feature_cols) >= 0.8


def detect_task(df: pd.DataFrame, target: str) -> str:
    """Infer task type based on target dtype."""
    series = df[target]
    if pd.api.types.is_numeric_dtype(series):
        return "regression"
    try:
        pd.to_numeric(series)
        return "regression"
    except Exception:
        return "classification"


def build_dataset_config(path: Path) -> Tuple[DatasetConfig, pd.DataFrame, Dict]:
    """
    Auto-build a DatasetConfig with sensible defaults for manuscript experiments.
    """
    df = load_dataframe(path)
    target = detect_target_column(df)
    exclude_columns = detect_exclude_columns(df)
    feature_cols = [c for c in df.columns if c not in exclude_columns + [target]]
    spectral = is_spectral(df, feature_cols)
    task = detect_task(df, target)

    # Use pre-determined best encoders for each dataset (from prior auto-select results)
    # This avoids the slow cross-validation encoder selection process
    DATASET_ENCODER_MAP = {
        "A3": "quantile",
        "B1": "spectroscopy", 
        "MA_A2": "rpca",
        "MB_B2": "robust",
    }
    
    dataset_name = path.stem
    encoder_name = DATASET_ENCODER_MAP.get(dataset_name, "probabilistic")  # fallback to probabilistic
    
    cfg = DatasetConfig(
        path=path,
        target=target,
        task=task,
        exclude_columns=exclude_columns,
        snv=spectral,
        name=dataset_name,
        encoding={"name": encoder_name},
    )
    meta = {
        "target": target,
        "exclude_columns": exclude_columns,
        "feature_cols": feature_cols,
        "spectral": spectral,
        "task": task,
    }
    return cfg, df, meta


def load_real_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict, pd.DataFrame, Dict]:
    """
    Load a dataset with automatic target/feature detection and encoding.
    """
    cfg, df, detected_meta = build_dataset_config(path)
    data = load_dataset(cfg)
    metadata = data["metadata"]
    metadata.update(detected_meta)
    return data["x_tif"], data["y_tif"], metadata, df, data["config"]


def select_components_cv(
    X: np.ndarray,
    Y: np.ndarray,
    max_components: int,
    n_folds: int,
    random_state: int = 42,
) -> int:
    """
    Select optimal components using inner CV (moved from ``__main__``).
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    actual_folds = min(n_folds, n_samples // 2)
    if actual_folds < 2:
        return 1

    max_possible = min(max_components, n_samples - n_samples // actual_folds - 1, n_features)
    if max_possible < 1:
        return 1

    best_score = np.inf
    best_n_comp = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for n_comp in range(1, max_possible + 1):
            try:
                pls = PLSRegression(n_components=n_comp, scale=False)
                cv = KFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
                y_pred = cross_val_predict(pls, X, Y, cv=cv)
                rmsecv = np.sqrt(np.mean((Y - y_pred) ** 2))
                if rmsecv < best_score:
                    best_score = rmsecv
                    best_n_comp = n_comp
            except Exception:
                continue
    return best_n_comp


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    """Format a value as mean ± std."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def format_with_significance(mean: float, std: float, sig_marker: str, digits: int = 3) -> str:
    """Format a metric with its significance marker (e.g., '0.234 ± 0.012**')."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}{sig_marker}"


def get_project_root() -> Path:
    """Get project root directory (three levels up from manuscript/utils.py)."""
    return Path(__file__).resolve().parents[2]


def compute_significance(
    baseline_scores: np.ndarray,
    model_scores: np.ndarray,
    alpha: float = 0.05,
    test_type: str = 'both'
) -> Dict:
    """
    Compute statistical significance between baseline (PLS) and a model.
    
    Uses paired t-test for parametric comparison and Wilcoxon signed-rank 
    test for non-parametric comparison (more robust for small samples).
    
    Args:
        baseline_scores: RMSEP scores from baseline model (e.g., PLS) per fold
        model_scores: RMSEP scores from comparison model per fold
        alpha: Significance level (default 0.05)
        test_type: 'ttest', 'wilcoxon', or 'both'
    
    Returns:
        Dictionary with test results and significance markers
    """
    from scipy.stats import ttest_rel, wilcoxon
    
    results = {
        'n_folds': len(baseline_scores),
        'baseline_mean': np.mean(baseline_scores),
        'baseline_std': np.std(baseline_scores),
        'model_mean': np.mean(model_scores),
        'model_std': np.std(model_scores),
        'difference': np.mean(baseline_scores) - np.mean(model_scores),
        'improvement_pct': 100 * (1 - np.mean(model_scores) / np.mean(baseline_scores)),
    }
    
    # Paired t-test (parametric)
    if test_type in ['ttest', 'both']:
        try:
            t_stat, p_ttest = ttest_rel(baseline_scores, model_scores)
            results['ttest_statistic'] = t_stat
            results['ttest_pvalue'] = p_ttest
            results['ttest_significant'] = p_ttest < alpha
        except Exception:
            results['ttest_statistic'] = np.nan
            results['ttest_pvalue'] = np.nan
            results['ttest_significant'] = False
    
    # Wilcoxon signed-rank test (non-parametric, better for small n)
    if test_type in ['wilcoxon', 'both']:
        try:
            diff = baseline_scores - model_scores
            if np.all(diff == 0):
                results['wilcoxon_statistic'] = np.nan
                results['wilcoxon_pvalue'] = 1.0
                results['wilcoxon_significant'] = False
            else:
                w_stat, p_wilcox = wilcoxon(baseline_scores, model_scores, alternative='two-sided')
                results['wilcoxon_statistic'] = w_stat
                results['wilcoxon_pvalue'] = p_wilcox
                results['wilcoxon_significant'] = p_wilcox < alpha
        except Exception:
            results['wilcoxon_statistic'] = np.nan
            results['wilcoxon_pvalue'] = np.nan
            results['wilcoxon_significant'] = False
    
    # Generate significance marker
    p_val = results.get('wilcoxon_pvalue', results.get('ttest_pvalue', 1.0))
    if p_val < 0.001:
        results['sig_marker'] = '***'
    elif p_val < 0.01:
        results['sig_marker'] = '**'
    elif p_val < 0.05:
        results['sig_marker'] = '*'
    else:
        results['sig_marker'] = ''
    
    results['model_better'] = results['difference'] > 0
    return results


def run_pairwise_significance_tests(
    cv_results: Dict[str, Dict[str, List]],
    baseline: str = 'PLS',
    metric: str = 'rmsep'
) -> Dict[str, Dict]:
    """
    Run pairwise significance tests comparing all models to baseline.
    
    Args:
        cv_results: Dictionary with model -> {'rmsep': [...], 'r2': [...]} 
        baseline: Name of baseline model (default 'PLS')
        metric: Which metric to test ('rmsep' or 'r2')
    
    Returns:
        Dictionary with model -> significance test results
    """
    baseline_scores = np.array(cv_results[baseline][metric])
    sig_results = {}
    
    for model, scores in cv_results.items():
        if model == baseline:
            sig_results[model] = {
                'model_mean': np.mean(baseline_scores),
                'model_std': np.std(baseline_scores),
                'sig_marker': '',
                'is_baseline': True
            }
        else:
            model_scores = np.array(scores[metric])
            sig_results[model] = compute_significance(baseline_scores, model_scores)
            sig_results[model]['is_baseline'] = False
    
    return sig_results

