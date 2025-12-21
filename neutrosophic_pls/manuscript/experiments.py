from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from ..metrics import evaluation_metrics
from ..model_factory import create_model_from_params
from ..simulate import generate_simulation, add_spike_corruption
from ..vip import compute_nvip
from ..data_loader import _snv_normalize
from .utils import (
    ensure_dir,
    format_mean_std,
    load_real_dataset,
    save_csv,
    save_json,
    select_components_cv,
    compute_significance,
    run_pairwise_significance_tests,
)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    """Wrapper to compute metrics for regression and classification."""
    metrics = evaluation_metrics(y_true, y_pred, include_extended=True)
    if task == "classification":
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        y_pred_class = np.rint(y_pred_arr)
        metrics["Accuracy"] = float(np.mean(y_true_arr == y_pred_class))
    return metrics


def _aggregate_vip(
    vip_store: Dict[str, Dict[str, List[np.ndarray]]],
    feature_names: Sequence[str],
    dataset: str,
    out_dir: Path,
) -> None:
    """Aggregate VIP results and persist summary artifacts."""
    vip_records: List[Dict] = []
    vip_npz: Dict[str, np.ndarray] = {"feature_names": np.array(feature_names)}

    for method, channel_dict in vip_store.items():
        if not channel_dict["aggregate"]:
            continue
        agg = np.stack(channel_dict["aggregate"])
        vip_npz[f"{method}_aggregate"] = agg
        vip_npz[f"{method}_T"] = np.stack(channel_dict["T"])
        vip_npz[f"{method}_I"] = np.stack(channel_dict["I"])
        vip_npz[f"{method}_F"] = np.stack(channel_dict["F"])

        agg_mean = agg.mean(axis=0)
        agg_std = agg.std(axis=0)
        t_mean = np.stack(channel_dict["T"]).mean(axis=0)
        t_std = np.stack(channel_dict["T"]).std(axis=0)
        i_mean = np.stack(channel_dict["I"]).mean(axis=0)
        i_std = np.stack(channel_dict["I"]).std(axis=0)
        f_mean = np.stack(channel_dict["F"]).mean(axis=0)
        f_std = np.stack(channel_dict["F"]).std(axis=0)

        for idx, feat in enumerate(feature_names):
            vip_records.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "feature": feat,
                    "aggregate_mean": float(agg_mean[idx]),
                    "aggregate_std": float(agg_std[idx]),
                    "T_mean": float(t_mean[idx]),
                    "T_std": float(t_std[idx]),
                    "I_mean": float(i_mean[idx]),
                    "I_std": float(i_std[idx]),
                    "F_mean": float(f_mean[idx]),
                    "F_std": float(f_std[idx]),
                }
            )

    if vip_records:
        vip_df = pd.DataFrame(vip_records)
        vip_df_sorted = (
            vip_df.sort_values("aggregate_mean", ascending=False)
            .groupby("method")
            .head(100)
            .reset_index(drop=True)
        )
        save_csv(vip_df_sorted, out_dir / "vip_summary.csv")
        ensure_dir(out_dir)
        np.savez(out_dir / "vip_full.npz", **vip_npz)


def run_realdata_cv(
    paths: List[str],
    out_root: Path,
    *,
    cv_folds: int = 5,
    repeats: int = 3,
    max_components: int = 15,
    random_state: int = 42,
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_I: float = 1.0,
    lambda_F: float = 1.0,
    alpha: float = 1.0,
) -> Dict[str, Path]:
    """
    Run cross-validated experiments on real datasets and save artifacts.
    """
    out_root = Path(out_root)
    ensure_dir(out_root)

    all_cv_rows: List[Dict] = []
    all_pred_rows: List[Dict] = []
    dataset_dirs: Dict[str, Path] = {}
    methods = ["PLS", "NPLS", "NPLSW", "PNPLS"]

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        dataset_name = path.stem
        ds_dir = ensure_dir(out_root / dataset_name)

        x_tif, y_tif, metadata, _df, dataset_cfg = load_real_dataset(path)
        metadata.update(
            {
                "cv_folds": cv_folds,
                "repeats": repeats,
                "max_components": max_components,
                "channel_weights": channel_weights,
                "lambda_I": lambda_I,
                "lambda_F": lambda_F,
                "alpha": alpha,
            }
        )
        save_json(metadata, ds_dir / "metadata.json")
        save_json(dataset_cfg, ds_dir / "dataset_config.json")

        vip_store = {
            "NPLS": {"aggregate": [], "T": [], "I": [], "F": []},
            "NPLSW": {"aggregate": [], "T": [], "I": [], "F": []},
            "PNPLS": {"aggregate": [], "T": [], "I": [], "F": []},
        }

        ds_cv_rows: List[Dict] = []
        ds_pred_rows: List[Dict] = []
        task = metadata.get("task", "regression")

        for repeat in range(repeats):
            outer_cv = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state + repeat,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_tif)):
                x_train, x_test = x_tif[train_idx], x_tif[test_idx]
                y_train, y_test = y_tif[train_idx], y_tif[test_idx]

                n_comp = select_components_cv(
                    x_train[..., 0],
                    y_train[..., 0],
                    max_components=max_components,
                    n_folds=cv_folds,
                    random_state=random_state + repeat + fold_idx,
                )
                y_true = y_test[..., 0].ravel()

                for method in methods:
                    model = create_model_from_params(
                        method=method,
                        n_components=n_comp,
                        channel_weights=channel_weights,
                        lambda_indeterminacy=lambda_I,
                        lambda_falsity=lambda_F,
                        alpha=alpha
                    )
                    
                    if method == "PLS":
                        model.fit(x_train[..., 0], y_train[..., 0])
                        y_pred = model.predict(x_test[..., 0])
                    else:
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)

                    y_pred = np.asarray(y_pred).ravel()
                    metrics = _compute_metrics(y_true, y_pred, task)
                    row = {
                        "dataset": dataset_name,
                        "repeat": repeat,
                        "fold": fold_idx,
                        "method": method,
                        "n_components": n_comp,
                        **metrics,
                    }
                    ds_cv_rows.append(row)
                    all_cv_rows.append(row)

                    for idx, y_t, y_p in zip(test_idx, y_true, y_pred):
                        pred_row = {
                            "dataset": dataset_name,
                            "repeat": repeat,
                            "fold": fold_idx,
                            "method": method,
                            "sample_index": int(idx),
                            "y_true": float(y_t),
                            "y_pred": float(y_p),
                        }
                        ds_pred_rows.append(pred_row)
                        all_pred_rows.append(pred_row)

                    if method in vip_store:
                        vip = compute_nvip(model, x_train, channel_weights=channel_weights)
                        vip_store[method]["aggregate"].append(vip["aggregate"])
                        vip_store[method]["T"].append(vip["T"])
                        vip_store[method]["I"].append(vip["I"])
                        vip_store[method]["F"].append(vip["F"])

        feature_names = metadata.get("feature_names") or metadata.get("feature_cols") or [
            f"feat_{i}" for i in range(x_tif.shape[1])
        ]
        save_csv(pd.DataFrame(ds_cv_rows), ds_dir / "cv_results.csv")
        save_csv(pd.DataFrame(ds_pred_rows), ds_dir / "predictions.csv")
        _aggregate_vip(vip_store, feature_names, dataset_name, ds_dir)

        summary = (
            pd.DataFrame(ds_cv_rows)
            .groupby("method")
            .agg({"RMSEP": ["mean", "std"], "R2": ["mean", "std"], "MAE": ["mean", "std"]})
            .reset_index()
        )
        summary.columns = ["method", "RMSEP_mean", "RMSEP_std", "R2_mean", "R2_std", "MAE_mean", "MAE_std"]
        summary["dataset"] = dataset_name
        save_csv(summary, ds_dir / "summary.csv")

        pretty_rows = []
        for _, r in summary.iterrows():
            pretty_rows.append(
                {
                    "dataset": dataset_name,
                    "method": r["method"],
                    "RMSEP": format_mean_std(r["RMSEP_mean"], r["RMSEP_std"]),
                    "R2": format_mean_std(r["R2_mean"], r["R2_std"]),
                    "MAE": format_mean_std(r["MAE_mean"], r["MAE_std"]),
                }
            )
        save_csv(pd.DataFrame(pretty_rows), ds_dir / "summary_main_table.csv")

        dataset_dirs[dataset_name] = ds_dir

    if all_cv_rows:
        save_csv(pd.DataFrame(all_cv_rows), out_root / "combined_cv_results.csv")
    if all_pred_rows:
        save_csv(pd.DataFrame(all_pred_rows), out_root / "combined_predictions.csv")
    return dataset_dirs


def run_simulation_robustness(
    out_dir: Path,
    *,
    n: int = 200,
    p: int = 200,
    h: int = 10,
    seed: int = 0,
    falsity_levels: Iterable[float] = (0.0, 0.02, 0.05, 0.1, 0.2),
    truth_noise: float = 0.05,
    ind_noise: float = 0.1,
) -> Path:
    """
    Sweep falsity corruption levels and record performance.
    """
    out_dir = ensure_dir(Path(out_dir))
    rows: List[Dict] = []
    methods = ["PLS", "NPLS", "NPLSW", "PNPLS"]

    for falsity in falsity_levels:
        x_tif, y_tif, _ = generate_simulation(
            n_samples=n,
            n_features=p,
            n_components=h,
            seed=seed,
            noise_config={
                "truth_noise": truth_noise,
                "indeterminacy_noise": ind_noise,
                "falsity_noise": falsity,
            },
        )
        split = int(0.8 * n)
        x_train, x_test = x_tif[:split], x_tif[split:]
        y_train, y_test = y_tif[:split], y_tif[split:]
        y_true = y_test[..., 0].ravel()
        n_comp = min(h, x_train.shape[1], x_train.shape[0] - 1)

        for method in methods:
            model = create_model_from_params(
                method=method,
                n_components=n_comp,
                lambda_falsity=0.5 if method == "PNPLS" else 0.5 # Maintain existing PNPLS override if intentional
            )
            
            if method == "PLS":
                model.fit(x_train[..., 0], y_train[..., 0])
                y_pred = model.predict(x_test[..., 0])
            else:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

            metrics = evaluation_metrics(y_true, np.asarray(y_pred).ravel(), include_extended=False)
            rows.append(
                {
                    "falsity_level": falsity,
                    "method": method,
                    "RMSEP": metrics["RMSEP"],
                    "R2": metrics["R2"],
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "sim_results.csv")
    return out_dir


def run_runtime_scaling(
    out_dir: Path,
    *,
    ns: Iterable[int] = (50, 100, 200, 400),
    ps: Iterable[int] = (50, 100, 200, 400),
    h: int = 10,
) -> Path:
    """
    Benchmark fit time scaling across sample and feature sizes.
    """
    out_dir = ensure_dir(Path(out_dir))
    rows: List[Dict] = []
    methods = ["PLS", "NPLS", "NPLSW", "PNPLS"]

    for n in ns:
        for p in ps:
            x_tif, y_tif, _ = generate_simulation(
                n_samples=n,
                n_features=p,
                n_components=h,
                seed=n + p,
                noise_config={"truth_noise": 0.05, "indeterminacy_noise": 0.1, "falsity_noise": 0.05},
            )
            n_comp = min(h, p - 1, n - 1)

            for method in methods:
                times: List[float] = []
                start = time.perf_counter()
                model = create_model_from_params(
                    method=method,
                    n_components=n_comp,
                    lambda_falsity=0.5 if method == "PNPLS" else 0.5
                )
                
                if method == "PLS":
                    model.fit(x_tif[..., 0], y_tif[..., 0])
                else:
                    model.fit(x_tif, y_tif)
                times.append(time.perf_counter() - start)
                rows.append(
                    {
                        "n": n,
                        "p": p,
                        "method": method,
                        "fit_time_sec": float(np.median(times)),
                    }
                )

    save_csv(pd.DataFrame(rows), out_dir / "runtime.csv")
    return out_dir


def run_sensitivity_sweep(
    dataset_path: Path,
    out_dir: Path,
    *,
    lambda_grid: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
    channel_weight_grid: Iterable[Tuple[float, float, float]] = (
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 0.2, 0.8),
        (1.0, 1.0, 0.5),
    ),
    cv_folds: int = 3,
    max_components: int = 10,
    random_state: int = 0,
) -> Path:
    """
    Small sensitivity sweep for FigS2 (lambda_I and channel weights).
    """
    out_dir = ensure_dir(Path(out_dir))
    x_tif, y_tif, metadata, _, _ = load_real_dataset(Path(dataset_path))
    dataset_name = metadata.get("name", Path(dataset_path).stem)
    n_comp = select_components_cv(
        x_tif[..., 0],
        y_tif[..., 0],
        max_components=max_components,
        n_folds=cv_folds,
        random_state=random_state,
    )
    results: List[Dict] = []

    def _cv_eval(build_fn, label: str, sweep: str, value) -> None:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        for train_idx, test_idx in cv.split(x_tif):
            model = build_fn()
            model.fit(x_tif[train_idx], y_tif[train_idx])
            preds = np.asarray(model.predict(x_tif[test_idx])).ravel()
            y_pred_all.append(preds)
            y_true_all.append(y_tif[test_idx][..., 0].ravel())
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        metrics = evaluation_metrics(y_true, y_pred, include_extended=False)
        results.append(
            {
                "dataset": dataset_name,
                "sweep": sweep,
                "value": value if sweep == "lambda_I" else str(value),
                "method": label,
                "RMSEP": metrics["RMSEP"],
                "R2": metrics["R2"],
            }
        )

    for lam in lambda_grid:
        _cv_eval(
            lambda: create_model_from_params(method="NPLSW", n_components=n_comp, lambda_indeterminacy=lam),
            "NPLSW",
            "lambda_I",
            lam,
        )

    for weights in channel_weight_grid:
        _cv_eval(
            lambda: create_model_from_params(method="NPLS", n_components=n_comp, channel_weights=weights),
            "NPLS",
            "channel_weights",
            weights,
        )
        _cv_eval(
            lambda: create_model_from_params(method="PNPLS", n_components=n_comp, lambda_falsity=0.5, lambda_indeterminacy=0.5, channel_weights=weights),
            "PNPLS",
            "channel_weights",
            weights,
        )

    save_csv(pd.DataFrame(results), out_dir / "sensitivity.csv")
    return out_dir





def _compute_falsity_from_clean_subspace(
    X_clean_ref: np.ndarray,
    X_obs: np.ndarray,
    *,
    pca_components: int = 8,
    z_threshold: float = 4.0,
) -> np.ndarray:
    """
    Compute per-cell falsity from reconstruction residuals vs clean reference.
    
    Fit low-rank basis on X_clean_ref, compute residuals for X_obs,
    and convert extreme residuals to soft falsity in [0,1].
    """
    X_clean_ref = np.asarray(X_clean_ref, dtype=float)
    X_obs = np.asarray(X_obs, dtype=float)
    n_ref, n_feat = X_clean_ref.shape
    k = min(int(pca_components), max(0, min(n_ref, n_feat) - 1))
    if k <= 0:
        return np.zeros_like(X_obs)

    # Fit PCA on clean reference
    mean = X_clean_ref.mean(axis=0, keepdims=True)
    Xc = X_clean_ref - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt_k = Vt[:k, :]

    def _recon(X_in: np.ndarray) -> np.ndarray:
        Xin_c = X_in - mean
        scores = Xin_c @ Vt_k.T
        return scores @ Vt_k + mean

    # Compute residuals on clean ref to get baseline stats
    R_ref = X_clean_ref - _recon(X_clean_ref)
    med = np.median(R_ref, axis=0, keepdims=True)
    mad = np.median(np.abs(R_ref - med), axis=0, keepdims=True) + 1e-8
    
    # Compute z-scores for observed data
    R_obs = X_obs - _recon(X_obs)
    z = np.abs(R_obs - med) / (mad * 1.4826)

    # Soft falsity: sigmoid around z_threshold
    F = 1.0 / (1.0 + np.exp(-(z - z_threshold)))
    return np.clip(F, 0.0, 1.0)


def run_corruption_robustness(
    paths: List[str],
    out_root: Path,
    *,
    corruption_levels: Tuple[float, ...] = (0.0, 0.15, 0.30),
    cv_folds: int = 5,
    repeats: int = 3,
    max_components: int = 15,
    random_state: int = 42,
    use_ground_truth_falsity: bool = True,
) -> Path:
    """
    Run corruption robustness experiments with TRAINING-ONLY corruption.
    
    This implements Option A DOE methodology:
    - TRAINING data is corrupted with spike artifacts
    - TEST data remains CLEAN (evaluation target)
    - Falsity is computed from clean reference subspace OR set to ground-truth
    - Statistical significance tests compare NPLS variants to PLS baseline
    
    This is the key experiment to demonstrate NPLS robustness advantage:
    when training contains corrupted samples, NPLS should downweight them
    and maintain good generalization to clean test data.
    
    Args:
        paths: List of dataset paths
        out_root: Output directory for results
        corruption_levels: Tuple of corruption proportions to test
        cv_folds: Number of CV folds
        repeats: Number of repeat runs
        max_components: Maximum PLS components to search
        random_state: Random seed for reproducibility
        use_ground_truth_falsity: If True, set F=1 for known corrupted samples
                                   (isolates model robustness from encoder accuracy)
    
    Returns:
        Path to output directory
    """
    out_root = Path(out_root)
    ensure_dir(out_root)
    
    methods = ["PLS", "NPLS", "NPLSW", "PNPLS"]
    all_results: List[Dict] = []
    significance_results: List[Dict] = []
    
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
            
        dataset_name = path.stem
        ds_dir = ensure_dir(out_root / dataset_name)
        
        # Load dataset (get X from Truth channel)
        x_tif, y_tif, metadata, _df, _ = load_real_dataset(path)
        X_clean = x_tif[..., 0]  # Truth channel (original clean data)
        y = y_tif[..., 0].ravel() if y_tif.ndim == 3 else y_tif.ravel()
        n_samples, n_features = X_clean.shape
        
        for corruption in corruption_levels:
            scenario = "clean" if corruption == 0.0 else f"spike_{int(corruption*100)}pct"
            
            # Collect fold-level results for significance testing
            cv_fold_results: Dict[str, Dict[str, List[float]]] = {
                m: {"RMSEP": [], "R2": []} for m in methods
            }
            
            for repeat in range(repeats):
                rng = np.random.default_rng(random_state + repeat)
                
                # Cross-validation with TRAINING-ONLY corruption
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state + repeat)
                
                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_clean)):
                    # Split data - TEST remains CLEAN!
                    X_train_clean = X_clean[train_idx].copy()
                    X_test_clean = X_clean[test_idx].copy()
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    
                    # Apply SNV normalization (common for NIR)
                    X_train_snv = _snv_normalize(X_train_clean)
                    X_test_snv = _snv_normalize(X_test_clean)
                    
                    # Corrupt TRAINING data only
                    if corruption > 0:
                        n_corrupt = int(len(train_idx) * corruption)
                        corrupt_mask = np.zeros(len(train_idx), dtype=bool)
                        if n_corrupt > 0:
                            corrupt_sample_idx = rng.choice(len(train_idx), size=n_corrupt, replace=False)
                            corrupt_mask[corrupt_sample_idx] = True
                            
                            # Add spikes to corrupted training samples (in SNV space)
                            for i in corrupt_sample_idx:
                                n_spikes = rng.integers(5, 11)
                                spike_pos = rng.choice(n_features, size=min(n_spikes, n_features), replace=False)
                                spike_mag = rng.uniform(10.0, 25.0, size=len(spike_pos))
                                X_train_snv[i, spike_pos] += spike_mag
                        
                        # Build TIF tensor for training with Falsity
                        if use_ground_truth_falsity:
                            # Ground-truth: F=1 for all cells of corrupted samples
                            F_train = np.zeros_like(X_train_snv)
                            F_train[corrupt_mask, :] = 1.0
                        else:
                            # Encoder-based: compute F from clean reference subspace
                            F_train = _compute_falsity_from_clean_subspace(
                                X_train_snv[~corrupt_mask],  # Clean samples as reference
                                X_train_snv,  # All training samples
                                pca_components=8,
                                z_threshold=4.0,
                            )
                        
                        x_train_tif = np.stack([X_train_snv, np.zeros_like(X_train_snv), F_train], axis=-1)
                    else:
                        # Clean scenario: no corruption, F=0
                        x_train_tif = np.stack([
                            X_train_snv, 
                            np.zeros_like(X_train_snv), 
                            np.zeros_like(X_train_snv)
                        ], axis=-1)
                    
                    # Test TIF: always clean (F=0)
                    x_test_tif = np.stack([
                        X_test_snv, 
                        np.zeros_like(X_test_snv), 
                        np.zeros_like(X_test_snv)
                    ], axis=-1)
                    
                    # Y as TIF format
                    y_train_tif = np.stack([
                        y_train.reshape(-1, 1),
                        np.zeros((len(y_train), 1)),
                        np.zeros((len(y_train), 1))
                    ], axis=-1)
                    y_test_tif = np.stack([
                        y_test.reshape(-1, 1),
                        np.zeros((len(y_test), 1)),
                        np.zeros((len(y_test), 1))
                    ], axis=-1)
                    
                    # Select optimal components
                    n_comp = select_components_cv(
                        x_train_tif[..., 0], y_train,
                        max_components=max_components,
                        n_folds=3,
                        random_state=random_state + repeat + fold_idx,
                    )
                    
                    for method in methods:
                        model = create_model_from_params(
                            method=method,
                            n_components=n_comp,
                            channel_weights=(1.0, 0.0, 0.0),
                            lambda_falsity=1.5 if method == "PNPLS" else 0.8 # Maintain specific robustness priors
                        )
                        
                        if method == "PLS":
                             model.fit(x_train_tif[..., 0], y_train)
                             y_pred = model.predict(x_test_tif[..., 0])
                        else:
                             model.fit(x_train_tif, y_train_tif)
                             y_pred = model.predict(x_test_tif)
                        
                        y_pred = np.asarray(y_pred).ravel()
                        metrics = evaluation_metrics(y_test, y_pred, include_extended=False)
                        
                        cv_fold_results[method]["RMSEP"].append(metrics["RMSEP"])
                        cv_fold_results[method]["R2"].append(metrics["R2"])
                        
                        all_results.append({
                            "dataset": dataset_name,
                            "scenario": scenario,
                            "corruption": corruption,
                            "repeat": repeat,
                            "fold": fold_idx,
                            "method": method,
                            "RMSEP": metrics["RMSEP"],
                            "R2": metrics["R2"],
                        })
            
            # Compute significance tests vs PLS baseline
            sig_rmsep = run_pairwise_significance_tests(cv_fold_results, baseline="PLS", metric="RMSEP")
            sig_r2 = run_pairwise_significance_tests(cv_fold_results, baseline="PLS", metric="R2")
            
            for method in methods:
                rmsep_mean = np.mean(cv_fold_results[method]["RMSEP"])
                rmsep_std = np.std(cv_fold_results[method]["RMSEP"])
                r2_mean = np.mean(cv_fold_results[method]["R2"])
                r2_std = np.std(cv_fold_results[method]["R2"])
                
                significance_results.append({
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "corruption": corruption,
                    "method": method,
                    "RMSEP_mean": rmsep_mean,
                    "RMSEP_std": rmsep_std,
                    "RMSEP_formatted": f"{rmsep_mean:.3f} ± {rmsep_std:.3f}{sig_rmsep[method].get('sig_marker', '')}",
                    "R2_mean": r2_mean,
                    "R2_std": r2_std,
                    "R2_formatted": f"{r2_mean:.3f} ± {r2_std:.3f}{sig_r2[method].get('sig_marker', '')}",
                    "improvement_pct": sig_rmsep[method].get("improvement_pct", 0.0),
                    "wilcoxon_pvalue": sig_rmsep[method].get("wilcoxon_pvalue", np.nan),
                    "significant": sig_rmsep[method].get("wilcoxon_significant", False),
                })
        
        # Save per-dataset summary
        ds_summary = pd.DataFrame([r for r in significance_results if r["dataset"] == dataset_name])
        save_csv(ds_summary, ds_dir / "corruption_robustness.csv")
    
    # Save combined results
    save_csv(pd.DataFrame(all_results), out_root / "corruption_cv_results.csv")
    save_csv(pd.DataFrame(significance_results), out_root / "corruption_significance.csv")
    
    return out_root


