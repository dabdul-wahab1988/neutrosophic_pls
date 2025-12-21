from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .metrics import r2_score
from .data_loader import _entropy_surprisal, _normalize_entropy_scores


class IndeterminacyValidator:
    """Validates computed indeterminacy against ground-truth repeats."""

    def __init__(self) -> None:
        self.validation_results_: Dict[str, np.ndarray | float | str] = {}
        self.calibration_params_: Dict[str, float | Dict[int, Dict[str, float]]] = {}

    def validate_against_repeats(
        self,
        x_computed: np.ndarray,
        x_repeated: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, np.ndarray | float | str]:
        """
        Compare computed I with actual measurement variance from repeat measurements.

        Args:
            x_computed: Computed indeterminacy (n_samples, n_features)
            x_repeated: Repeated measurements (n_samples, n_replicates, n_features)
            feature_names: Optional names for per-feature diagnostics
        """
        if x_repeated.ndim != 3:
            raise ValueError("x_repeated must be (n_samples, n_replicates, n_features)")

        actual_std = np.std(x_repeated, axis=1)
        x_comp_norm = self._normalize_to_range(x_computed)
        actual_norm = self._normalize_to_range(actual_std)

        overall_corr = np.corrcoef(x_comp_norm.flatten(), actual_norm.flatten())[0, 1]

        feature_correlations: Dict[str, float] = {}
        names = feature_names or [f"feature_{i}" for i in range(x_computed.shape[1])]
        for j, name in enumerate(names):
            feature_correlations[name] = np.corrcoef(
                x_comp_norm[:, j], actual_norm[:, j]
            )[0, 1]

        diagnosis: str
        if overall_corr < 0.3:
            diagnosis = "POOR: I encoding not aligned with measurement variance"
        elif overall_corr < 0.5:
            diagnosis = "FAIR: weak relationship; recalibration recommended"
        elif overall_corr < 0.7:
            diagnosis = "GOOD: captures variance structure reasonably"
        else:
            diagnosis = "EXCELLENT: highly correlated with actual variance"

        self.validation_results_ = {
            "overall_correlation": overall_corr,
            "feature_correlations": feature_correlations,
            "diagnosis": diagnosis,
            "actual_std": actual_std,
            "computed_i": x_computed,
        }
        return self.validation_results_

    def calibrate_entropy_parameters(
        self,
        X: np.ndarray,
        x_repeated: np.ndarray,
        bins_grid: list[int] = [5, 10, 15, 20, 30, 50],
    ) -> Dict[str, float | Dict[int, Dict[str, float]]]:
        """
        Grid search over entropy bin counts to maximize alignment with measurement variance.
        """
        actual_std = np.std(x_repeated, axis=1)
        actual_norm = self._normalize_to_range(actual_std)

        results: Dict[int, Dict[str, float]] = {}
        for bins in bins_grid:
            x_surprisal, _ = _entropy_surprisal(X, bins=bins)
            x_ind, _ = _normalize_entropy_scores(x_surprisal, high_tail=95.0)
            x_ind_norm = self._normalize_to_range(x_ind)

            r2 = self._compute_r2(x_ind_norm, actual_norm)
            corr = np.corrcoef(x_ind_norm.flatten(), actual_norm.flatten())[0, 1]
            results[bins] = {"r2": r2, "correlation": corr}

        optimal_bins = max(results.keys(), key=lambda b: results[b]["r2"])
        self.calibration_params_ = {
            "optimal_bins": optimal_bins,
            "optimal_r2": results[optimal_bins]["r2"],
            "all_results": results,
        }
        return self.calibration_params_

    @staticmethod
    def _normalize_to_range(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        return low + (high - low) * (x - x_min) / (x_max - x_min + 1e-12)

    @staticmethod
    def _compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return r2_score(y_true.flatten(), y_pred.flatten())


class FalsityValidator:
    """Validates falsity scores against known unreliable measurements."""

    def __init__(self) -> None:
        self.validation_results_: Dict[str, float | np.ndarray] = {}
        self.soft_falsity_params_: Optional[Dict[str, float]] = None

    def validate_against_ground_truth(
        self, x_computed_f: np.ndarray, x_ground_truth_f: np.ndarray
    ) -> Dict[str, float | np.ndarray]:
        """
        Compare computed falsity against ground-truth outlier labels (binary or soft).
        """
        f_bin = (x_computed_f > 0.5).astype(float)
        y_bin = (x_ground_truth_f > 0.5).astype(float)

        cm = confusion_matrix(y_bin.flatten(), f_bin.flatten())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_bin.flatten(), f_bin.flatten(), average="binary"
        )
        auc = roc_auc_score(y_bin.flatten(), x_computed_f.flatten())

        self.validation_results_ = {
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        }
        return self.validation_results_

    def optimize_soft_falsity_thresholds(
        self,
        z_scores: np.ndarray,
        ground_truth_f: np.ndarray,
        threshold_hard_grid: list[float] = [2.5, 3.0, 3.5, 4.0],
        threshold_soft_grid: list[float] = [1.5, 2.0, 2.5],
    ) -> Dict[str, float]:
        """
        Find soft/hard thresholds that maximize F1 against ground-truth outliers.
        """
        best_f1 = -np.inf
        best_params: Dict[str, float] = {}

        z_abs = np.abs(z_scores)
        y_bin = (ground_truth_f > 0.5).astype(float)

        for t_hard in threshold_hard_grid:
            for t_soft in threshold_soft_grid:
                if t_soft >= t_hard:
                    continue
                F = np.zeros_like(z_abs)
                mask_soft = (z_abs >= t_soft) & (z_abs < t_hard)
                F[mask_soft] = 0.5 * (z_abs[mask_soft] - t_soft) / (t_hard - t_soft + 1e-12)
                F[z_abs >= t_hard] = 1.0

                f_bin = (F > 0.5).astype(float)
                f1 = precision_recall_fscore_support(
                    y_bin.flatten(), f_bin.flatten(), average="binary"
                )[2]

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "threshold_soft": float(t_soft),
                        "threshold_hard": float(t_hard),
                        "f1_score": float(f1),
                    }

        self.soft_falsity_params_ = best_params
        return best_params


def create_validation_report(
    validator_i: IndeterminacyValidator,
    validator_f: FalsityValidator,
    output_dir: Path = Path("results_validation"),
) -> Path:
    """
    Dump a simple text report combining I/F validation outcomes.
    """
    output_dir.mkdir(exist_ok=True)
    report = f"""
===============================================================================
T/I/F VALIDATION REPORT
===============================================================================

INDETERMINACY (I)
-----------------
Overall Correlation: {validator_i.validation_results_.get('overall_correlation', np.nan):.3f}
Diagnosis: {validator_i.validation_results_.get('diagnosis', 'n/a')}
Optimal Entropy Bins: {validator_i.calibration_params_.get('optimal_bins', 'n/a')}
R2 (Optimal Bins): {validator_i.calibration_params_.get('optimal_r2', np.nan)}

FALSITY (F)
-----------
Precision: {validator_f.validation_results_.get('precision', np.nan):.3f}
Recall:    {validator_f.validation_results_.get('recall', np.nan):.3f}
F1-score:  {validator_f.validation_results_.get('f1_score', np.nan):.3f}
AUC:       {validator_f.validation_results_.get('auc', np.nan):.3f}
Optimal Soft Thresholds: {validator_f.soft_falsity_params_}
===============================================================================
""".strip()

    report_path = output_dir / "validation_report.txt"
    report_path.write_text(report)
    return report_path
