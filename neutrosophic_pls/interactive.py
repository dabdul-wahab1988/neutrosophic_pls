"""
Interactive CLI Workflow for Neutrosophic PLS
==============================================

A user-friendly, step-by-step interactive interface for non-coders
to use the neutrosophic_pls package effectively.

Usage:
    python -m neutrosophic_pls --interactive
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .data_loader import (
    DatasetConfig,
    load_dataset,
    load_dataframe,
    list_available_datasets,
    encode_neutrosophic,
    _snv_normalize as snv_normalize,
)
from .encoders import (
    EncoderConfig,
    auto_select_encoder,
    dispatch_encoder,
    EncodingResult,
)

from .study_config import StudyConfig, DatasetSettings, ModelSettings
from .model_factory import create_model
from .metrics import evaluation_metrics as compute_metrics
from .vip import compute_nvip


# =============================================================================
# Constants
# =============================================================================

AVAILABLE_ENCODERS = [
    ("probabilistic", "General purpose - Statistical residual-based T/I/F"),
    ("spectroscopy", "NIR/IR spectra - Noise-floor aware encoding"),
    ("rpca", "Corrupted data - Low-rank truth, sparse falsity"),
    ("wavelet", "Multi-scale signals - Frequency-based separation"),
    ("quantile", "Non-parametric - Envelope-based boundaries"),
    ("augment", "Stability testing - Data augmentation variance"),
    ("robust", "Outlier detection - MAD-based spike detection"),
    ("ndg", "Differential Geometry - Physics-based manifold encoding"),
]

AVAILABLE_MODELS = [
    ("NPLS", "Standard neutrosophic PLS with sample weighting"),
    ("NPLSW", "Reliability-weighted (best for noisy samples)"),
    ("PNPLS", "Probabilistic (best for element-wise/localized noise)"),
]


# =============================================================================
# Helper Functions
# =============================================================================

def clear_line():
    """Clear current line in terminal."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def print_header(title: str, width: int = 65):
    """Print a styled header box."""
    print()
    print("╔" + "═" * (width - 2) + "╗")
    title_padded = title.center(width - 2)
    print(f"║{title_padded}║")
    print("╚" + "═" * (width - 2) + "╝")


def print_step(step_num: int, total_steps: int, title: str):
    """Print step header."""
    print(f"\nSTEP {step_num}/{total_steps}: {title}")
    print("─" * 50)


def print_box(lines: List[str], width: int = 60):
    """Print text in a box."""
    print("┌" + "─" * (width - 2) + "┐")
    for line in lines:
        if len(line) > width - 4:
            line = line[:width - 7] + "..."
        padded = line.ljust(width - 4)
        print(f"│ {padded} │")
    print("└" + "─" * (width - 2) + "┘")


def print_success(message: str):
    """Print success message with checkmark."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"✗ Error: {message}")


def prompt_choice(
    prompt: str,
    valid_choices: List[str],
    default: Optional[str] = None,
) -> str:
    """Prompt user for choice with validation."""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if user_input.upper() in [c.upper() for c in valid_choices]:
            # Return with original case from valid_choices
            for c in valid_choices:
                if c.upper() == user_input.upper():
                    return c
            return user_input
        
        print(f"  Please enter one of: {', '.join(valid_choices)}")


def prompt_int(prompt: str, default: int, min_val: int = 1, max_val: int = 100) -> int:
    """Prompt user for integer with validation."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("  Please enter a valid number")


def prompt_float(
    prompt: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Prompt user for float with validation."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"  Please enter a number >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Please enter a number <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")


def prompt_float_tuple(
    prompt: str,
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Prompt for a 3-tuple of floats."""
    default_text = ", ".join(str(v) for v in default)
    while True:
        user_input = input(f"{prompt} [{default_text}]: ").strip()
        if not user_input:
            return default
        parts = [p.strip() for p in user_input.split(",") if p.strip()]
        if len(parts) != 3:
            print("  Please enter three comma-separated values (e.g., 1.0,0.5,1.0)")
            continue
        try:
            values = tuple(float(p) for p in parts)
            return values  # type: ignore[return-value]
        except ValueError:
            print("  Please enter valid numbers (e.g., 1.0,0.5,1.0)")


def format_number(value: float, precision: int = 2) -> str:
    """Format number for display."""
    if abs(value) >= 1000:
        return f"{value:.0f}"
    elif abs(value) >= 1:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{min(precision + 2, 6)}f}"


def _nearest_class_labels(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map continuous predictions to nearest class labels."""
    values = np.asarray(values).ravel()
    classes = np.asarray(classes).ravel()
    if classes.size == 0:
        return values
    distances = np.abs(values.reshape(-1, 1) - classes.reshape(1, -1))
    idx = np.argmin(distances, axis=1)
    return classes[idx]


# =============================================================================
# Auto-Selection Functions
# =============================================================================

def auto_select_npls_variant(
    x_tif: np.ndarray,
    y_tif: np.ndarray,
    task: str = "regression",
    cv_folds: int = 3,
    max_components: int = 5,
    random_state: int = 42,
    model_settings: Optional[ModelSettings] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Evaluate NPLS variants and return the best performing one.
    
    Parameters
    ----------
    x_tif : np.ndarray
        Neutrosophic encoded features (n_samples, n_features, 3)
    y_tif : np.ndarray
        Neutrosophic encoded target
    task : str
        "regression" or "classification"
    cv_folds : int
        Number of CV folds for evaluation
    max_components : int
        Maximum components to try
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    best_variant : str
        Name of the best variant ("NPLS", "NPLSW", or "PNPLS")
    scores : dict
        Scores for all variants evaluated
    """
    n_samples = x_tif.shape[0]
    n_features = x_tif.shape[1]
    
    # Determine reasonable n_components
    n_comp = min(
        max_components,
        n_features,
        n_samples // cv_folds - 1,  # Need at least n_comp+1 samples per fold
    )
    n_comp = max(1, n_comp)
    
    y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    
    base_settings = model_settings or ModelSettings()
    settings = ModelSettings(
        method=base_settings.method,
        max_components=n_comp,
        channel_weights=base_settings.channel_weights,
        lambda_indeterminacy=base_settings.lambda_indeterminacy,
        lambda_falsity=base_settings.lambda_falsity,
        alpha=base_settings.alpha,
    )
    config = StudyConfig(
        dataset=DatasetSettings(path="auto_select", target="dummy"),
        model=settings,
    )
    
    variant_names = ["NPLS", "NPLSW", "PNPLS"]
    
    scores = {}
    
    print("  Evaluating variants...")
    
    for name in variant_names:
        fold_scores = []
        
        for train_idx, test_idx in cv.split(x_tif):
            x_train, x_test = x_tif[train_idx], x_tif[test_idx]
            y_train, y_test = y_tif[train_idx], y_tif[test_idx]
            
            try:
                model = create_model(config, name, n_components=n_comp)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                y_true = y_test[..., 0] if y_test.ndim == 3 else y_test
                y_true = y_true.ravel()
                y_pred = y_pred.ravel()
                
                if task == "regression":
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    fold_scores.append(rmse)
                else:
                    accuracy = np.mean(np.round(y_pred) == y_true)
                    fold_scores.append(1.0 - accuracy)  # Error rate
                    
            except Exception:
                fold_scores.append(float("inf"))
        
        mean_score = np.mean(fold_scores)
        scores[name] = mean_score
        
        # Display with indicator
        best_so_far = min(scores.values())
        indicator = " ✓ Best" if mean_score == best_so_far else ""
        metric_name = "RMSE" if task == "regression" else "Error"
        print(f"    {name:8s}: {metric_name}={mean_score:.4f}{indicator}")
    
    best_variant = min(scores, key=scores.get)
    return best_variant, scores


# =============================================================================
# Interactive Session
# =============================================================================

@dataclass
class SessionState:
    """State container for interactive session."""
    # Data
    data_path: Optional[Path] = None
    dataframe: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    task: str = "regression"
    snv: bool = False
    exclude_columns: List[str] = field(default_factory=list)
    
    # Encoding
    encoder_name: Optional[str] = None
    encoder_auto_selected: bool = False
    x_tif: Optional[np.ndarray] = None
    y_tif: Optional[np.ndarray] = None
    encoding_metadata: Optional[Dict[str, Any]] = None
    encoder_auto_scores: Optional[Dict[str, float]] = None
    
    # Model
    model_name: Optional[str] = None
    model_auto_selected: bool = False
    n_components: int = 5
    model_auto_scores: Optional[Dict[str, float]] = None
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    lambda_indeterminacy: float = 1.0
    lambda_falsity: float = 1.0
    alpha: float = 1.0
    
    # Evaluation
    cv_folds: int = 5
    cv_repeats: int = 3
    
    # Results
    results: Optional[pd.DataFrame] = None
    summary: Optional[pd.DataFrame] = None
    fitted_model: Optional[Any] = None
    confusion_matrix: Optional[np.ndarray] = None


class InteractiveSession:
    """
    Interactive CLI workflow manager for neutrosophic PLS analysis.
    
    Guides non-coders through:
    1. Data loading with summary
    2. Encoder selection (auto/manual)
    3. NPLS variant selection (auto/manual)
    4. Training configuration
    5. Analysis and results
    6. Feature importance (VIP) analysis
    7. Figure export (analysis report)
    """
    
    TOTAL_STEPS = 7
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.state = SessionState()
    
    def run(self) -> Optional[Dict[str, Any]]:
        """Run the full interactive workflow."""
        try:
            self._print_welcome()
            
            # Step 1: Load data
            if not self._step_load_data():
                return None
            
            # Step 2: Summarize data
            if not self._step_summarize_data():
                return None
            
            # Step 3: Select encoder
            if not self._step_select_encoder():
                return None
            
            # Step 4: Select NPLS variant
            if not self._step_select_model():
                return None
            
            # Step 5: Run analysis
            analysis_result = self._step_run_analysis()
            if analysis_result is None:
                return None
            
            # Step 6: VIP Analysis (optional)
            self._step_vip_analysis()

            # Step 7: Figure export (optional)
            self._step_export_figures()
            
            return analysis_result
            
        except KeyboardInterrupt:
            print("\n\nSession cancelled by user.")
            return None
        except Exception as e:
            print_error(str(e))
            return None
    
    def _print_welcome(self):
        """Print welcome banner."""
        print_header("Neutrosophic PLS - Interactive Analysis")
        print("           For researchers without coding experience")
        print()
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    
    def _step_load_data(self) -> bool:
        """Step 1: Load data from file."""
        print_step(1, self.TOTAL_STEPS, "Load Your Data")
        
        # List available datasets
        datasets = list_available_datasets(self.data_dir)
        
        if datasets:
            print(f"Available datasets in '{self.data_dir}/':")
            for i, ds in enumerate(datasets, 1):
                print(f"  {i}. {ds['name']} ({ds['format'].upper()}, {ds['size_mb']} MB)")
            print()
            
            choice = input(f"Enter selection (1-{len(datasets)}) or file path: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(datasets):
                self.state.data_path = Path(datasets[int(choice) - 1]["path"])
            else:
                self.state.data_path = Path(choice)
        else:
            print(f"No datasets found in '{self.data_dir}/'")
            path = input("Enter path to data file: ").strip()
            if not path:
                print_error("No path provided")
                return False
            self.state.data_path = Path(path)
        
        # Validate and load
        if not self.state.data_path.exists():
            print_error(f"File not found: {self.state.data_path}")
            return False
        
        try:
            self.state.dataframe = load_dataframe(self.state.data_path)
            print_success(f"Loaded: {self.state.data_path.name}")
            return True
        except Exception as e:
            print_error(f"Could not load file: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Step 2: Summarize Data
    # -------------------------------------------------------------------------
    
    def _step_summarize_data(self) -> bool:
        """Step 2: Display data summary and get configuration."""
        print_step(2, self.TOTAL_STEPS, "Data Summary")
        
        df = self.state.dataframe
        n_samples, n_cols = df.shape
        
        # Show columns
        print(f"Columns ({n_cols}): {', '.join(df.columns[:10])}", end="")
        if n_cols > 10:
            print(f" ... and {n_cols - 10} more")
        else:
            print()
        
        # Suggest a target column
        default_target = None
        for candidate in ("Protein", "protein", "y", "Y", "target", "Target", "Class", "label"):
            if candidate in df.columns:
                default_target = candidate
                break
        if default_target is None:
            default_target = str(df.columns[-1])

        # Get target column
        print()
        target_input = input(f"Enter target column name [{default_target}]: ").strip()
        self.state.target_column = target_input or default_target
        
        if self.state.target_column not in df.columns:
            print_error(f"Column '{self.state.target_column}' not found")
            return False
        
        # Determine task type
        target_series = df[self.state.target_column]
        default_task = "r"
        if not pd.api.types.is_numeric_dtype(target_series):
            default_task = "c"
        else:
            try:
                target_numeric = pd.to_numeric(target_series, errors="coerce")
                target_numeric = target_numeric[np.isfinite(target_numeric)]
                unique_vals = np.unique(target_numeric)
                if unique_vals.size > 0:
                    integer_like = np.allclose(unique_vals, np.round(unique_vals))
                    small_cardinality = unique_vals.size <= min(20, max(2, n_samples // 5))
                    if integer_like and small_cardinality:
                        default_task = "c"
            except Exception:
                default_task = "r"
        task_input = input(
            f"Task type (r=regression, c=classification) [{default_task}]: "
        ).strip().lower()
        task_input = task_input or default_task
        self.state.task = "classification" if task_input == "c" else "regression"
        
        # Exclude columns
        exclude_defaults = [c for c in ("ID", "Id", "SampleID", "sample_id") if c in df.columns]
        exclude_default_text = ", ".join(exclude_defaults) if exclude_defaults else ""
        exclude_prompt = "Columns to exclude (comma-separated, or empty)"
        exclude_input = input(f"{exclude_prompt} [{exclude_default_text}]: ").strip()
        if not exclude_input and exclude_default_text:
            exclude_input = exclude_default_text
        self.state.exclude_columns = [c.strip() for c in exclude_input.split(",") if c.strip()]
        
        # SNV normalization: Removed from workflow
        # Encoders handle normalization internally (e.g., NDG has 'normalization' param)
        # This avoids double-normalization and gives encoders more control
        self.state.snv = False
        
        # Set feature columns
        exclude = [self.state.target_column] + self.state.exclude_columns
        self.state.feature_columns = [c for c in df.columns if c not in exclude]
        
        # Print summary box
        print()
        target_values = df[self.state.target_column]
        
        summary_lines = [
            f"Dataset: {self.state.data_path.name}",
            f"Samples: {n_samples}   Features: {len(self.state.feature_columns)}",
            f"Target: {self.state.target_column}   Task: {self.state.task}",
        ]
        
        if self.state.task == "regression":
            try:
                target_num = pd.to_numeric(target_values, errors="coerce")
                summary_lines.append(
                    f"Target range: {target_num.min():.2f} - {target_num.max():.2f}"
                )
            except Exception:
                pass
        else:
            n_classes = target_values.nunique()
            summary_lines.append(f"Classes: {n_classes}")
        
        # Feature range
        try:
            feature_data = df[self.state.feature_columns].select_dtypes(include=[np.number])
            if not feature_data.empty:
                feat_min = feature_data.min().min()
                feat_max = feature_data.max().max()
                summary_lines.append(f"Feature range: [{format_number(feat_min)}, {format_number(feat_max)}]")
        except Exception:
            pass
        
        print_box(summary_lines)
        
        # Confirm
        print()
        confirm = input("Continue with this configuration? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("Configuration cancelled.")
            return False
        
        return True
    
    # -------------------------------------------------------------------------
    # Step 3: Select Encoder
    # -------------------------------------------------------------------------
    
    def _step_select_encoder(self) -> bool:
        """Step 3: Select encoder for T/I/F conversion."""
        print_step(3, self.TOTAL_STEPS, "Encoder Selection")
        
        print("How would you like to encode your data as Truth/Indeterminacy/Falsity?")
        print()
        print("  [A] Automatic - Let the system find the best encoder")
        print("  [M] Manual    - Choose a specific encoder")
        print()
        
        choice = prompt_choice("Selection", ["A", "M"], default="A")
        
        # Extract data arrays
        df = self.state.dataframe
        X = df[self.state.feature_columns].values.astype(float)
        y = df[self.state.target_column].values
        X_auto = snv_normalize(X) if self.state.snv else X
        
        if choice.upper() == "A":
            # Automatic selection
            print()
            print("Evaluating encoders... (this may take a moment)")
            
            config = EncoderConfig(name="auto", params={})
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    y_auto = y
                    if self.state.task == "classification":
                        try:
                            y_auto = y.astype(float)
                        except Exception:
                            classes = np.unique(y)
                            class_map = {c: i for i, c in enumerate(classes)}
                            y_auto = np.array([class_map[val] for val in y], dtype=float)
                    result, metadata = auto_select_encoder(X_auto, y_auto, config, task=self.state.task)
                
                self.state.encoder_name = metadata.best_config.name
                self.state.encoder_auto_selected = True
                self.state.encoder_auto_scores = metadata.scores
                
                # Display results
                for enc_name, score in metadata.scores.items():
                    indicator = " ✓ Best" if enc_name == self.state.encoder_name else ""
                    metric = "RMSE" if self.state.task == "regression" else "Error"
                    print(f"  {enc_name:15s}: {metric}={score:.4f}{indicator}")
                
                print()
                print_success(f"Selected: {self.state.encoder_name} encoder")
                
            except Exception as e:
                print(f"  Auto-selection failed: {e}")
                print("  Falling back to 'probabilistic' encoder")
                self.state.encoder_name = "probabilistic"
                self.state.encoder_auto_selected = False
                self.state.encoder_auto_scores = None
        
        else:
            # Manual selection
            print()
            print("Available encoders:")
            for i, (name, desc) in enumerate(AVAILABLE_ENCODERS, 1):
                print(f"  {i}. {name:15s} - {desc}")
            print()
            
            while True:
                enc_choice = input(f"Enter selection (1-{len(AVAILABLE_ENCODERS)}) [1]: ").strip()
                enc_choice = enc_choice or "1"
                if enc_choice.isdigit() and 1 <= int(enc_choice) <= len(AVAILABLE_ENCODERS):
                    self.state.encoder_name = AVAILABLE_ENCODERS[int(enc_choice) - 1][0]
                    self.state.encoder_auto_selected = False
                    self.state.encoder_auto_scores = None
                    break
                print(f"  Please enter a number between 1 and {len(AVAILABLE_ENCODERS)}")
            
            print_success(f"Selected: {self.state.encoder_name} encoder")
        
        # Perform encoding
        print()
        print("Encoding data...")
        
        try:
            enc_result = encode_neutrosophic(
                X, y,
                task=self.state.task,
                snv=self.state.snv,
                encoding=self.state.encoder_name,
                return_metadata=True,
            )
            self.state.x_tif, self.state.y_tif, self.state.encoding_metadata = enc_result
            print_success(f"Encoded to shape: {self.state.x_tif.shape}")
            return True
            
        except Exception as e:
            print_error(f"Encoding failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Step 4: Select NPLS Variant
    # -------------------------------------------------------------------------
    
    def _step_select_model(self) -> bool:
        """Step 4: Select NPLS variant."""
        print_step(4, self.TOTAL_STEPS, "NPLS Variant Selection")
        
        print("Which NPLS variant would you like to use?")
        print()
        print("  [A] Automatic - Let the system select based on your data")
        print("  [M] Manual    - Choose from available variants:")
        for i, (name, desc) in enumerate(AVAILABLE_MODELS, 1):
            print(f"        {i}. {name:6s} - {desc}")
        print()
        
        choice = prompt_choice("Selection", ["A", "M"], default="A")
        
        def _prompt_training_config() -> None:
            print()
            print("Training configuration:")
            self.state.n_components = prompt_int(
                "  Number of components", default=5, min_val=1, max_val=20
            )
            self.state.cv_folds = prompt_int(
                "  CV folds", default=5, min_val=2, max_val=10
            )
            self.state.cv_repeats = prompt_int(
                "  CV repeats", default=3, min_val=1, max_val=10
            )

            advanced = prompt_choice("Configure advanced model settings?", ["Y", "N"], default="N")
            if advanced.upper() == "Y":
                self.state.channel_weights = prompt_float_tuple(
                    "  Channel weights (T,I,F)", self.state.channel_weights
                )
                self.state.lambda_indeterminacy = prompt_float(
                    "  Lambda indeterminacy", self.state.lambda_indeterminacy, min_val=0.0
                )
                self.state.lambda_falsity = prompt_float(
                    "  Lambda falsity", self.state.lambda_falsity, min_val=0.0
                )
                self.state.alpha = prompt_float(
                    "  Alpha (PNPLS softness)", self.state.alpha, min_val=0.0
                )

        if choice.upper() == "A":
            _prompt_training_config()
            print()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    settings = ModelSettings(
                        max_components=self.state.n_components,
                        channel_weights=self.state.channel_weights,
                        lambda_indeterminacy=self.state.lambda_indeterminacy,
                        lambda_falsity=self.state.lambda_falsity,
                        alpha=self.state.alpha,
                    )
                    best_variant, scores = auto_select_npls_variant(
                        self.state.x_tif,
                        self.state.y_tif,
                        task=self.state.task,
                        cv_folds=self.state.cv_folds,
                        max_components=self.state.n_components,
                        model_settings=settings,
                    )

                self.state.model_name = best_variant
                self.state.model_auto_selected = True
                self.state.model_auto_scores = scores
                print()
                print_success(f"Selected: {self.state.model_name}")

            except Exception as e:
                print(f"  Auto-selection failed: {e}")
                print("  Falling back to NPLS")
                self.state.model_name = "NPLS"
                self.state.model_auto_selected = False
                self.state.model_auto_scores = None

        else:
            # Manual selection
            print()
            while True:
                model_choice = input(f"Enter selection (1-{len(AVAILABLE_MODELS)}) [1]: ").strip()
                model_choice = model_choice or "1"
                if model_choice.isdigit() and 1 <= int(model_choice) <= len(AVAILABLE_MODELS):
                    self.state.model_name = AVAILABLE_MODELS[int(model_choice) - 1][0]
                    self.state.model_auto_selected = False
                    self.state.model_auto_scores = None
                    break
                print(f"  Please enter a number between 1 and {len(AVAILABLE_MODELS)}")

            print_success(f"Selected: {self.state.model_name}")
            _prompt_training_config()
        
        return True
    
    # -------------------------------------------------------------------------
    # Step 5: Run Analysis
    # -------------------------------------------------------------------------
    
    def _step_run_analysis(self) -> Dict[str, Any]:
        """Step 5: Run analysis and display results."""
        print_step(5, self.TOTAL_STEPS, "Run Analysis")
        
        # Ask if user wants to compare with Classical PLS
        compare_pls = prompt_choice(
            "Compare with Classical PLS?",
            ["Y", "N"],
            default="Y",
        )
        self.state.compare_pls = compare_pls.upper() == "Y"
        
        methods_to_run = [self.state.model_name]
        if self.state.compare_pls:
            methods_to_run.append("PLS")
        
        print()
        print(f"Running {self.state.cv_folds}-fold × {self.state.cv_repeats}-repeat cross-validation...")
        if self.state.compare_pls:
            print(f"  Evaluating: {self.state.model_name} vs Classical PLS")
        print()
        
        # Create config for factory
        config = StudyConfig(
            dataset=DatasetSettings(
                path=str(self.state.data_path) if self.state.data_path else "interactive",
                target=self.state.target_column or "target",
                task=self.state.task
            ),
            model=ModelSettings(
                max_components=self.state.n_components,
                channel_weights=self.state.channel_weights,
                lambda_indeterminacy=self.state.lambda_indeterminacy,
                lambda_falsity=self.state.lambda_falsity,
                alpha=self.state.alpha,
            )
        )
        
        x_tif = self.state.x_tif
        y_tif = self.state.y_tif
        y = y_tif[..., 0] if y_tif.ndim == 3 else y_tif
        self.state.confusion_matrix = None
        
        # Results for each method
        results = {method: [] for method in methods_to_run}
        fold_count = 0
        total_folds = self.state.cv_folds * self.state.cv_repeats * len(methods_to_run)
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        classes = np.unique(y)
        
        # Import PLS for comparison
        from sklearn.cross_decomposition import PLSRegression
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            for repeat in range(self.state.cv_repeats):
                cv = KFold(
                    n_splits=self.state.cv_folds,
                    shuffle=True,
                    random_state=42 + repeat,
                )
                
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x_tif)):
                    x_train, x_test = x_tif[train_idx], x_tif[test_idx]
                    y_train, y_test = y_tif[train_idx], y_tif[test_idx]
                    
                    y_true = y_test[..., 0] if y_test.ndim == 3 else y_test
                    y_true = y_true.ravel()
                    
                    for method in methods_to_run:
                        fold_count += 1
                        
                        # Progress display
                        progress = fold_count / total_folds
                        bar_width = 40
                        filled = int(bar_width * progress)
                        bar = "█" * filled + "░" * (bar_width - filled)
                        sys.stdout.write(f"\r  [{bar}] {int(progress * 100):3d}%")
                        sys.stdout.flush()
                        
                        if method == "PLS":
                            # Classical PLS - uses only Truth channel
                            X_train = x_train[..., 0]
                            X_test = x_test[..., 0]
                            Y_train = y_train[..., 0] if y_train.ndim == 3 else y_train
                            if Y_train.ndim == 1:
                                Y_train = Y_train.reshape(-1, 1)
                            
                            n_comp = min(self.state.n_components, X_train.shape[1], X_train.shape[0] - 1)
                            pls = PLSRegression(n_components=n_comp)
                            pls.fit(X_train, Y_train)
                            y_pred = pls.predict(X_test).ravel()
                        else:
                            # NPLS variant
                            model = create_model(config, method, n_components=self.state.n_components)
                            model.fit(x_train, y_train)
                            y_pred = model.predict(x_test).ravel()
                        
                        # Compute metrics
                        if self.state.task == "classification":
                            y_true_labels = _nearest_class_labels(y_true, classes)
                            y_pred_labels = _nearest_class_labels(y_pred, classes)
                            metrics = {
                                "Accuracy": accuracy_score(y_true_labels, y_pred_labels),
                                "Precision": precision_score(
                                    y_true_labels, y_pred_labels, average="weighted", zero_division=0
                                ),
                                "Recall": recall_score(
                                    y_true_labels, y_pred_labels, average="weighted", zero_division=0
                                ),
                                "F1": f1_score(
                                    y_true_labels, y_pred_labels, average="weighted", zero_division=0
                                ),
                                "Error_Rate": float(np.mean(y_true_labels != y_pred_labels)),
                            }
                            if method == self.state.model_name:
                                all_true.append(y_true_labels)
                                all_pred.append(y_pred_labels)
                        else:
                            metrics = compute_metrics(y_true, y_pred, include_extended=True)
                        
                        results[method].append({
                            "repeat": repeat + 1,
                            "fold": fold_idx + 1,
                            "method": method,
                            **metrics,
                        })
        
        print()  # New line after progress bar
        print()
        
        # Create results DataFrames for each method
        df_results_dict = {m: pd.DataFrame(r) for m, r in results.items()}
        self.state.results = df_results_dict[self.state.model_name]
        
        # Print results comparison
        print("═" * 70)
        if self.state.compare_pls:
            print("              RESULTS COMPARISON: NPLS vs Classical PLS")
        else:
            print("                      RESULTS SUMMARY")
        print("═" * 70)
        print()
        
        if self.state.compare_pls:
            # Side-by-side comparison table
            npls_df = df_results_dict[self.state.model_name]
            pls_df = df_results_dict["PLS"]
            
            if self.state.task == "regression":
                # Compute means and stds for both
                npls_rmsep = npls_df['RMSEP'].mean()
                npls_rmsep_std = npls_df['RMSEP'].std()
                npls_r2 = npls_df['R2'].mean()
                npls_r2_std = npls_df['R2'].std()
                npls_mae = npls_df['MAE'].mean()
                
                pls_rmsep = pls_df['RMSEP'].mean()
                pls_rmsep_std = pls_df['RMSEP'].std()
                pls_r2 = pls_df['R2'].mean()
                pls_r2_std = pls_df['R2'].std()
                pls_mae = pls_df['MAE'].mean()
                
                # Compute improvement
                rmsep_improve = (pls_rmsep - npls_rmsep) / pls_rmsep * 100 if pls_rmsep > 0 else 0
                r2_improve = (npls_r2 - pls_r2) / (1 - pls_r2) * 100 if pls_r2 < 1 else 0
                
                print("┌───────────┬──────────────────────┬──────────────────────┬────────────┐")
                print("│ Metric    │      Classical PLS   │   {:8s}           │ Improve    │".format(self.state.model_name))
                print("├───────────┼──────────────────────┼──────────────────────┼────────────┤")
                
                # RMSEP row
                rmsep_symbol = "↓" if npls_rmsep < pls_rmsep else ("↑" if npls_rmsep > pls_rmsep else "=")
                print(f"│ RMSEP     │ {pls_rmsep:7.4f} ± {pls_rmsep_std:.4f}   │ {npls_rmsep:7.4f} ± {npls_rmsep_std:.4f}   │ {rmsep_improve:+6.1f}% {rmsep_symbol}  │")
                
                # R² row
                r2_symbol = "↑" if npls_r2 > pls_r2 else ("↓" if npls_r2 < pls_r2 else "=")
                print(f"│ R²        │ {pls_r2:7.4f} ± {pls_r2_std:.4f}   │ {npls_r2:7.4f} ± {npls_r2_std:.4f}   │ {r2_improve:+6.1f}% {r2_symbol}  │")
                
                # MAE row
                mae_improve = (pls_mae - npls_mae) / pls_mae * 100 if pls_mae > 0 else 0
                mae_symbol = "↓" if npls_mae < pls_mae else ("↑" if npls_mae > pls_mae else "=")
                print(f"│ MAE       │ {pls_mae:7.4f}              │ {npls_mae:7.4f}              │ {mae_improve:+6.1f}% {mae_symbol}  │")
                
                print("└───────────┴──────────────────────┴──────────────────────┴────────────┘")
                print("  ↓ = lower is better (for errors), ↑ = higher is better (for R²)")
                
                # Overall verdict
                print()
                if npls_rmsep < pls_rmsep and npls_r2 > pls_r2:
                    print(f"  ✓ {self.state.model_name} OUTPERFORMS Classical PLS!")
                    print(f"    RMSEP reduced by {-rmsep_improve:.1f}%, R² improved by {r2_improve:.1f}%")
                elif npls_rmsep > pls_rmsep and npls_r2 < pls_r2:
                    print(f"  ⚠ Classical PLS performs better on this dataset")
                    print(f"    The neutrosophic encoding may not benefit this data.")
                else:
                    print(f"  ~ Mixed results - {self.state.model_name} and PLS perform similarly")
                    
            else:  # Classification
                npls_acc = npls_df['Accuracy'].mean()
                npls_f1 = npls_df['F1'].mean()
                pls_acc = pls_df['Accuracy'].mean()
                pls_f1 = pls_df['F1'].mean()
                
                acc_improve = (npls_acc - pls_acc) * 100
                f1_improve = (npls_f1 - pls_f1) * 100
                
                print("┌───────────┬──────────────────────┬──────────────────────┬────────────┐")
                print("│ Metric    │      Classical PLS   │   {:8s}           │ Improve    │".format(self.state.model_name))
                print("├───────────┼──────────────────────┼──────────────────────┼────────────┤")
                print(f"│ Accuracy  │ {pls_acc:7.4f}              │ {npls_acc:7.4f}              │ {acc_improve:+6.1f}% pts │")
                print(f"│ F1 Score  │ {pls_f1:7.4f}              │ {npls_f1:7.4f}              │ {f1_improve:+6.1f}% pts │")
                print("└───────────┴──────────────────────┴──────────────────────┴────────────┘")
        
        # Also show the detailed summary for the selected model
        print()
        df_results = df_results_dict[self.state.model_name]
        if self.state.task == "regression":
            summary = {
                "RMSEP": f"{df_results['RMSEP'].mean():.4f} ± {df_results['RMSEP'].std():.4f}",
                "R²": f"{df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}",
                "MAE": f"{df_results['MAE'].mean():.4f} ± {df_results['MAE'].std():.4f}",
            }
            if "RPD" in df_results.columns:
                summary["RPD"] = f"{df_results['RPD'].mean():.2f}"
        else:
            summary = {
                "Accuracy": f"{df_results['Accuracy'].mean():.4f} ± {df_results['Accuracy'].std():.4f}",
                "F1": f"{df_results['F1'].mean():.4f} ± {df_results['F1'].std():.4f}",
                "Precision": f"{df_results['Precision'].mean():.4f} ± {df_results['Precision'].std():.4f}",
                "Recall": f"{df_results['Recall'].mean():.4f} ± {df_results['Recall'].std():.4f}",
            }
        
        result_lines = [
            f"Model: {self.state.model_name} ({self.state.n_components} components)",
            f"Encoder: {self.state.encoder_name}",
            "",
        ]
        for metric, value in summary.items():
            result_lines.append(f"{metric}: {value}")

        if self.state.encoder_auto_scores:
            result_lines.append("")
            result_lines.append("Encoder auto-selection:")
            sorted_scores = sorted(self.state.encoder_auto_scores.items(), key=lambda x: x[1])
            for name, score in sorted_scores[:5]:
                result_lines.append(f"  {name}: {score:.4f}")
            if len(sorted_scores) > 5:
                result_lines.append("  ...")

        if self.state.model_auto_scores:
            result_lines.append("")
            result_lines.append("Model auto-selection:")
            sorted_scores = sorted(self.state.model_auto_scores.items(), key=lambda x: x[1])
            for name, score in sorted_scores:
                result_lines.append(f"  {name}: {score:.4f}")
        
        print_box(result_lines)

        if self.state.task == "classification" and all_true and all_pred:
            cm = confusion_matrix(
                np.concatenate(all_true),
                np.concatenate(all_pred),
                labels=classes,
            )
            self.state.confusion_matrix = cm
            display_labels = classes
            if self.state.encoding_metadata and "class_map" in self.state.encoding_metadata:
                reverse_map = {v: k for k, v in self.state.encoding_metadata["class_map"].items()}
                display_labels = np.array([reverse_map.get(c, c) for c in classes], dtype=object)
            print()
            print("Confusion matrix (rows=true, cols=predicted):")
            print(f"Labels: {list(display_labels)}")
            print(cm)
        
        # Fit final model on all data for VIP analysis
        print()
        print("Fitting final model on all data for VIP analysis...")
        final_model = create_model(config, self.state.model_name, n_components=self.state.n_components)
        final_model.fit(x_tif, y_tif)
        self.state.fitted_model = final_model
        print_success("Final model fitted")
        
        # Offer to save
        print()
        save_path = input("Save results to directory? (path or ENTER to skip): ").strip()
        
        if save_path:
            output_dir = Path(save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results CSV
            df_results.to_csv(output_dir / "cv_results.csv", index=False)
            
            # Save summary
            with open(output_dir / "summary.txt", "w") as f:
                f.write(f"Neutrosophic PLS Analysis Results\n")
                f.write(f"=" * 40 + "\n\n")
                f.write(f"Dataset: {self.state.data_path.name}\n")
                f.write(f"Target: {self.state.target_column}\n")
                f.write(f"Model: {self.state.model_name}\n")
                f.write(f"Encoder: {self.state.encoder_name}\n")
                f.write(f"Components: {self.state.n_components}\n\n")
                f.write(f"Channel weights: {self.state.channel_weights}\n")
                f.write(f"Lambda indeterminacy: {self.state.lambda_indeterminacy}\n")
                f.write(f"Lambda falsity: {self.state.lambda_falsity}\n")
                f.write(f"Alpha: {self.state.alpha}\n\n")
                f.write("Metrics:\n")
                for metric, value in summary.items():
                    f.write(f"  {metric}: {value}\n")

                auto_lines = []
                if self.state.encoder_auto_scores:
                    auto_lines.append("Encoder auto-selection scores saved to auto_selection_scores.csv")
                if self.state.model_auto_scores:
                    auto_lines.append("Model auto-selection scores saved to auto_selection_scores.csv")
                if auto_lines:
                    f.write("\nAuto-selection:\n")
                    for line in auto_lines:
                        f.write(f"  {line}\n")

            auto_selection: Dict[str, Any] = {}
            if self.state.encoder_auto_scores:
                auto_selection["encoder"] = {
                    "selected": self.state.encoder_name,
                    "scores": self.state.encoder_auto_scores,
                }
            if self.state.model_auto_scores:
                auto_selection["model"] = {
                    "selected": self.state.model_name,
                    "scores": self.state.model_auto_scores,
                }

            if auto_selection:
                with open(output_dir / "auto_selection.json", "w") as f:
                    json.dump(auto_selection, f, indent=2)

                rows = []
                if self.state.encoder_auto_scores:
                    rows.extend(
                        {"stage": "encoder", "candidate": name, "score": score}
                        for name, score in self.state.encoder_auto_scores.items()
                    )
                if self.state.model_auto_scores:
                    rows.extend(
                        {"stage": "model", "candidate": name, "score": score}
                        for name, score in self.state.model_auto_scores.items()
                    )
                if rows:
                    pd.DataFrame(rows).to_csv(output_dir / "auto_selection_scores.csv", index=False)
            
            print_success(f"Results saved to: {output_dir}/")
            print(f"  - cv_results.csv")
            print(f"  - summary.txt")
            if self.state.task == "classification" and getattr(self.state, "confusion_matrix", None) is not None:
                cm_labels = classes
                if self.state.encoding_metadata and "class_map" in self.state.encoding_metadata:
                    reverse_map = {v: k for k, v in self.state.encoding_metadata["class_map"].items()}
                    cm_labels = [reverse_map.get(c, c) for c in classes]
                pd.DataFrame(self.state.confusion_matrix, index=cm_labels, columns=cm_labels).to_csv(
                    output_dir / "confusion_matrix.csv"
                )
                print(f"  - confusion_matrix.csv")

            if auto_selection:
                print(f"  - auto_selection.json")
                print(f"  - auto_selection_scores.csv")
        
        return {
            "results": df_results,
            "summary": summary,
            "state": self.state,
        }
    
    # -------------------------------------------------------------------------
    # Step 6: VIP Analysis
    # -------------------------------------------------------------------------
    
    def _step_vip_analysis(self) -> None:
        """Step 6: Compute and display VIP feature importance analysis."""
        print_step(6, self.TOTAL_STEPS, "Feature Importance (VIP) Analysis")
        
        print("Would you like to analyze feature importance?")
        print()
        print("  VIP (Variable Importance in Projection) shows which features")
        print("  are most important for predictions, with channel breakdown:")
        print("    - VIP^T: Importance from signal (Truth)")
        print("    - VIP^I: Importance from uncertainty (Indeterminacy)")
        print("    - VIP^F: Importance from noise/outliers (Falsity)")
        print()
        
        choice = prompt_choice("Run VIP analysis?", ["Y", "N"], default="Y")
        
        if choice.upper() == "N":
            print("Skipping VIP analysis.")
            return
        
        print()
        print("Computing VIP scores...")
        
        try:
            # Compute VIP
            vip_result = compute_nvip(
                self.state.fitted_model,
                self.state.x_tif,
                channel_weights=self.state.channel_weights,
            )
            
            aggregate_vip = vip_result["aggregate"]
            vip_t = vip_result["T"]
            vip_i = vip_result["I"]
            vip_f = vip_result["F"]
            
            n_features = len(aggregate_vip)
            feature_names = self.state.feature_columns
            
            # Create VIP DataFrame
            vip_df = pd.DataFrame({
                "Feature": feature_names,
                "VIP": aggregate_vip,
                "VIP_T": vip_t,
                "VIP_I": vip_i,
                "VIP_F": vip_f,
            })
            vip_df = vip_df.sort_values("VIP", ascending=False).reset_index(drop=True)
            vip_df["Rank"] = range(1, len(vip_df) + 1)
            
            # Summary statistics
            n_important = (aggregate_vip > 1.0).sum()
            high_falsity_count = (vip_f > 0.3 * aggregate_vip).sum()  # Features with >30% falsity contribution
            
            print_success(f"VIP computed for {n_features} features")
            print()
            
            # Display top N features
            top_n = prompt_int("How many top features to display?", default=10, min_val=1, max_val=min(50, n_features))
            
            print()
            print("═" * 70)
            print(f"              TOP {top_n} MOST IMPORTANT FEATURES")
            print("═" * 70)
            
            # Table header
            print("┌──────┬─────────────────┬─────────┬─────────┬─────────┬─────────┐")
            print("│ Rank │ Feature         │   VIP   │  VIP^T  │  VIP^I  │  VIP^F  │")
            print("├──────┼─────────────────┼─────────┼─────────┼─────────┼─────────┤")
            
            for i in range(top_n):
                row = vip_df.iloc[i]
                feat_name = str(row["Feature"])[:15].ljust(15)
                importance = "★" if row["VIP"] > 1.0 else " "
                print(f"│ {i+1:4d} │ {feat_name} │ {row['VIP']:7.3f}{importance}│ {row['VIP_T']:7.3f} │ {row['VIP_I']:7.3f} │ {row['VIP_F']:7.3f} │")
            
            print("└──────┴─────────────────┴─────────┴─────────┴─────────┴─────────┘")
            print("  ★ = VIP > 1 (important feature)")
            
            # === BOTTOM N LEAST IMPORTANT FEATURES ===
            print()
            bottom_choice = prompt_choice(
                "Display least important (potentially removable) features?",
                ["Y", "N"],
                default="Y",
            )
            
            if bottom_choice.upper() == "Y":
                bottom_n = prompt_int(
                    "How many bottom features to display?",
                    default=10,
                    min_val=1,
                    max_val=min(50, n_features),
                )
                
                print()
                print("═" * 70)
                print(f"           BOTTOM {bottom_n} LEAST IMPORTANT FEATURES")
                print("═" * 70)
                print("  These features contribute little to predictions and may be candidates")
                print("  for removal to simplify the model.")
                print()
                
                # Table header
                print("┌──────┬─────────────────┬─────────┬─────────┬─────────┬─────────┐")
                print("│ Rank │ Feature         │   VIP   │  VIP^T  │  VIP^I  │  VIP^F  │")
                print("├──────┼─────────────────┼─────────┼─────────┼─────────┼─────────┤")
                
                # Get bottom N features (sorted ascending by VIP)
                vip_df_asc = vip_df.sort_values("VIP", ascending=True).reset_index(drop=True)
                for i in range(bottom_n):
                    row = vip_df_asc.iloc[i]
                    feat_name = str(row["Feature"])[:15].ljust(15)
                    low_marker = "○" if row["VIP"] < 0.5 else " "  # Very low importance
                    print(f"│ {n_features - i:4d} │ {feat_name} │ {row['VIP']:7.3f}{low_marker}│ {row['VIP_T']:7.3f} │ {row['VIP_I']:7.3f} │ {row['VIP_F']:7.3f} │")
                
                print("└──────┴─────────────────┴─────────┴─────────┴─────────┴─────────┘")
                print("  ○ = VIP < 0.5 (very low importance - consider removing)")
                
                # Count features below threshold
                n_very_low = (aggregate_vip < 0.5).sum()
                n_low = ((aggregate_vip >= 0.5) & (aggregate_vip < 0.8)).sum()
                
                print()
                low_summary_lines = [
                    f"Very low importance (VIP < 0.5): {n_very_low} features",
                    f"Low importance (0.5 ≤ VIP < 0.8): {n_low} features",
                    f"Moderate+ importance (VIP ≥ 0.8): {n_features - n_very_low - n_low} features",
                    "",
                    f"Potential removal candidates: {n_very_low} features ({100*n_very_low/n_features:.1f}%)",
                ]
                print_box(low_summary_lines)
            
            # Summary box
            print()
            summary_lines = [
                f"Total features: {n_features}",
                f"Important features (VIP > 1): {n_important} ({100*n_important/n_features:.1f}%)",
                f"Features with high noise contribution: {high_falsity_count}",
                "",
                f"Top feature: {vip_df.iloc[0]['Feature']} (VIP={vip_df.iloc[0]['VIP']:.3f})",
            ]
            print_box(summary_lines)
            
            # === ENHANCED NVIP CHANNEL ANALYSIS ===
            print()
            channel_choice = prompt_choice(
                "Run detailed channel analysis (signal vs noise insights)?",
                ["Y", "N"],
                default="Y",
            )
            
            if channel_choice.upper() == "Y":
                self._display_channel_analysis(vip_df, aggregate_vip, vip_t, vip_i, vip_f, n_features)
            
            # Offer to save VIP results
            print()
            vip_save_path = input("Save VIP analysis to CSV? (path or ENTER to skip): ").strip()
            
            if vip_save_path:
                if not vip_save_path.endswith(".csv"):
                    vip_save_path += ".csv"
                
                # Add channel analysis columns
                vip_df["Dominant_Channel"] = vip_df.apply(
                    lambda r: "T" if r["VIP_T"] >= r["VIP_I"] and r["VIP_T"] >= r["VIP_F"]
                    else ("I" if r["VIP_I"] >= r["VIP_F"] else "F"), axis=1
                )
                vip_df["SNR"] = vip_df["VIP_T"] / (vip_df["VIP_F"] + 1e-12)
                vip_df["Noise_Dominant"] = vip_df["VIP_F"] > vip_df["VIP_T"]
                
                # Reorder columns for output
                vip_df_out = vip_df[[
                    "Rank", "Feature", "VIP", "VIP_T", "VIP_I", "VIP_F",
                    "Dominant_Channel", "SNR", "Noise_Dominant"
                ]]
                vip_df_out.to_csv(vip_save_path, index=False)
                print_success(f"VIP analysis saved to: {vip_save_path}")
                
        except Exception as e:
            print_error(f"VIP analysis failed: {e}")
            return

    def _display_channel_analysis(
        self,
        vip_df: pd.DataFrame,
        aggregate_vip: np.ndarray,
        vip_t: np.ndarray,
        vip_i: np.ndarray,
        vip_f: np.ndarray,
        n_features: int,
    ) -> None:
        """Display detailed NVIP channel analysis with actionable insights."""
        
        print()
        print("═" * 70)
        print("           NVIP CHANNEL ANALYSIS - Signal vs Noise Insights")
        print("═" * 70)
        print()
        print("  The neutrosophic VIP decomposes feature importance into:")
        print("    VIP^T (Truth)  : Importance from actual signal values")
        print("    VIP^I (Indet.) : Importance from measurement uncertainty")
        print("    VIP^F (Falsity): Importance from noise/outlier patterns")
        print()
        
        # === 1. CHANNEL DOMINANCE ANALYSIS ===
        # Determine which channel dominates for each feature
        # Only consider features with meaningful VIP (> 0.01)
        meaningful = aggregate_vip > 0.01
        n_meaningful = meaningful.sum()
        n_sparse = n_features - n_meaningful
        
        dominant_t = (vip_t >= vip_i) & (vip_t >= vip_f) & meaningful
        dominant_i = (vip_i > vip_t) & (vip_i >= vip_f) & meaningful
        dominant_f = (vip_f > vip_t) & (vip_f > vip_i) & meaningful
        
        n_t_dominant = dominant_t.sum()
        n_i_dominant = dominant_i.sum()
        n_f_dominant = dominant_f.sum()
        
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│               CHANNEL DOMINANCE BREAKDOWN                       │")
        print("├─────────────────────────────────────────────────────────────────┤")
        
        if n_meaningful > 0:
            # Visual bar for channel dominance (of meaningful features only)
            bar_width = 40
            t_bar = int(bar_width * n_t_dominant / n_meaningful)
            i_bar = int(bar_width * n_i_dominant / n_meaningful)
            f_bar = bar_width - t_bar - i_bar
            
            print(f"│ Signal-dominant (T)     : {n_t_dominant:4d} ({100*n_t_dominant/n_meaningful:5.1f}%) │")
            print(f"│   [{'█' * t_bar}{'░' * (bar_width - t_bar)}] │")
            print(f"│ Uncertainty-dominant (I): {n_i_dominant:4d} ({100*n_i_dominant/n_meaningful:5.1f}%) │")
            print(f"│   [{'█' * i_bar}{'░' * (bar_width - i_bar)}] │")
            print(f"│ Noise-dominant (F)      : {n_f_dominant:4d} ({100*n_f_dominant/n_meaningful:5.1f}%) │")
            print(f"│   [{'█' * f_bar}{'░' * (bar_width - f_bar)}] │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│ Meaningful features (VIP > 0.01): {n_meaningful:4d} / {n_features}           │")
            print(f"│ Sparse/zero features:            {n_sparse:4d} (excluded from above) │")
        else:
            print("│ Warning: No features with meaningful VIP (> 0.01)            │")
            print("│ This suggests model fitting issues or all-zero features.     │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        # === 2. SIGNAL QUALITY ANALYSIS ===
        print()
        
        # Signal-to-Noise Ratio based on VIP channels
        # Use a meaningful floor (1% of VIP^T) to prevent astronomical values when VIP^F ≈ 0
        vip_f_floor = np.maximum(vip_f, 0.01 * vip_t + 0.001)
        snr = vip_t / vip_f_floor
        # Cap SNR at 100 for display purposes (beyond this, it's essentially "clean")
        snr_display = np.minimum(snr, 100.0)
        
        # Categorize features by signal quality
        high_snr = (snr > 2.0)  # Good signal quality
        medium_snr = (snr >= 1.0) & (snr <= 2.0)  # Moderate
        low_snr = (snr < 1.0)  # Noise exceeds signal
        noise_dominant = (vip_f > vip_t)  # F > T specifically
        
        n_high_snr = high_snr.sum()
        n_medium_snr = medium_snr.sum()
        n_low_snr = low_snr.sum()
        n_noise_dom = noise_dominant.sum()
        
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│                   SIGNAL QUALITY ANALYSIS                       │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ High quality (SNR > 2)    : {n_high_snr:4d} features  ✓ Good       │")
        print(f"│ Moderate quality (1≤SNR≤2): {n_medium_snr:4d} features  ~ Acceptable │")
        print(f"│ Low quality (SNR < 1)     : {n_low_snr:4d} features  ⚠ Check data │")
        print(f"│ Noise-dominant (F > T)    : {n_noise_dom:4d} features  ⚠ Investigate│")
        print("├─────────────────────────────────────────────────────────────────┤")
        # Use median SNR for robustness (mean is skewed by clean features)
        median_snr = np.median(snr_display)
        print(f"│ Median SNR (VIP^T / VIP^F): {median_snr:.2f}                          │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        # === 3. NOISE-DOMINANT FEATURES (Data Quality Issues) ===
        if n_noise_dom > 0:
            print()
            show_noisy = prompt_choice(
                f"Display {min(10, n_noise_dom)} noise-dominant features (potential data issues)?",
                ["Y", "N"],
                default="Y",
            )
            
            if show_noisy.upper() == "Y":
                print()
                print("═" * 70)
                print("           NOISE-DOMINANT FEATURES (VIP^F > VIP^T)")
                print("═" * 70)
                print("  These features have MORE noise contribution than signal.")
                print("  This may indicate: sensor issues, calibration problems, or outliers.")
                print()
                
                # Get noise-dominant features sorted by F/T ratio
                # IMPORTANT: Only consider features with meaningful VIP (> 0.01)
                # to avoid showing features with VIP ≈ 0 that have misleading F/T ratios
                vip_df_copy = vip_df.copy()
                vip_df_copy["F_T_Ratio"] = vip_df_copy["VIP_F"] / (vip_df_copy["VIP_T"] + 0.001)
                
                # Filter: noise-dominant AND meaningful VIP
                meaningful_mask = (vip_df_copy["VIP"] > 0.01)
                noise_dom_mask = (vip_df_copy["VIP_F"] > vip_df_copy["VIP_T"])
                noise_dom_df = vip_df_copy[meaningful_mask & noise_dom_mask].sort_values(
                    "F_T_Ratio", ascending=False
                ).head(10)
                
                if len(noise_dom_df) == 0:
                    print("  No meaningful noise-dominant features found (all have VIP ≈ 0).")
                    print("  This is common in sparse data like mass spectrometry.")
                else:
                    print("┌─────────────────┬─────────┬─────────┬─────────┬──────────┬─────────┐")
                    print("│ Feature         │   VIP   │  VIP^T  │  VIP^F  │  F/T     │ Status  │")
                    print("├─────────────────┼─────────┼─────────┼─────────┼──────────┼─────────┤")
                    
                    for _, row in noise_dom_df.iterrows():
                        feat_name = str(row["Feature"])[:15].ljust(15)
                        ratio = row["F_T_Ratio"]
                        # Cap ratio display at 999 for readability
                        ratio_str = ">999   " if ratio > 999 else f"{ratio:7.2f} "
                        status = "⚠ HIGH " if ratio > 3 else "⚠ Check"
                        print(f"│ {feat_name} │ {row['VIP']:7.3f} │ {row['VIP_T']:7.3f} │ {row['VIP_F']:7.3f} │ {ratio_str} │ {status} │")
                    
                    print("└─────────────────┴─────────┴─────────┴─────────┴──────────┴─────────┘")
                    print("  F/T Ratio > 1 means noise exceeds signal for that feature")
                    print("  Note: Only features with VIP > 0.01 are shown")
        
        # === 4. BEST SIGNAL FEATURES (Reliable for predictions) ===
        print()
        show_best = prompt_choice(
            "Display best signal-quality features (most reliable)?",
            ["Y", "N"],
            default="Y",
        )
        
        if show_best.upper() == "Y":
            print()
            print("═" * 70)
            print("           BEST SIGNAL QUALITY FEATURES (Highest VIP^T / VIP^F)")
            print("═" * 70)
            print("  These features have the best signal-to-noise ratio.")
            print("  They are the MOST RELIABLE for predictions.")
            print()
            
            # Sort by SNR (VIP_T / VIP_F) with meaningful floor
            vip_df_copy = vip_df.copy()
            # Use same floor as above
            vip_f_floored = np.maximum(vip_df_copy["VIP_F"], 0.01 * vip_df_copy["VIP_T"] + 0.001)
            vip_df_copy["SNR"] = np.minimum(vip_df_copy["VIP_T"] / vip_f_floored, 100.0)
            best_snr_df = vip_df_copy[vip_df_copy["VIP"] > 0.8].sort_values("SNR", ascending=False).head(10)
            
            if len(best_snr_df) > 0:
                print("┌─────────────────┬─────────┬─────────┬─────────┬──────────┬─────────┐")
                print("│ Feature         │   VIP   │  VIP^T  │  VIP^F  │   SNR    │ Quality │")
                print("├─────────────────┼─────────┼─────────┼─────────┼──────────┼─────────┤")
                
                for _, row in best_snr_df.iterrows():
                    feat_name = str(row["Feature"])[:15].ljust(15)
                    snr_val = row["SNR"]
                    # Show as ">100" if capped, otherwise show value
                    snr_str = ">100   " if snr_val >= 99.9 else f"{snr_val:7.2f} "
                    quality = "★ Clean" if snr_val >= 99.9 else ("★ Excel" if snr_val > 10 else ("✓ Good " if snr_val > 2 else "~ Mod  "))
                    print(f"│ {feat_name} │ {row['VIP']:7.3f} │ {row['VIP_T']:7.3f} │ {row['VIP_F']:7.3f} │ {snr_str} │ {quality} │")
                
                print("└─────────────────┴─────────┴─────────┴─────────┴──────────┴─────────┘")
                print("  SNR = Signal-to-Noise Ratio (capped at 100, >100 means essentially clean)")
            else:
                print("  No features with VIP > 0.8 found.")
        
        # === 5. ACTIONABLE RECOMMENDATIONS ===
        print()
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│                    📋 RECOMMENDATIONS                            │")
        print("├─────────────────────────────────────────────────────────────────┤")
        
        recommendations = []
        
        # Recommendation based on noise-dominant features
        if n_noise_dom > n_features * 0.3:
            recommendations.append(
                "⚠ HIGH NOISE: >30% of features are noise-dominant. Consider:"
            )
            recommendations.append("    - Reviewing data collection procedures")
            recommendations.append("    - Checking for sensor calibration issues")
            recommendations.append("    - Applying additional preprocessing")
        elif n_noise_dom > n_features * 0.1:
            recommendations.append(
                "~ MODERATE NOISE: 10-30% features are noise-dominant."
            )
            recommendations.append("    - Investigate noise-dominant wavelengths")
        else:
            recommendations.append(
                "✓ GOOD DATA QUALITY: <10% features are noise-dominant."
            )
        
        # Recommendation based on signal-dominant features
        if n_t_dominant > n_features * 0.5:
            recommendations.append(
                "✓ STRONG SIGNAL: >50% features are signal-dominant."
            )
        
        # Feature selection recommendation
        n_important = (aggregate_vip > 1.0).sum()
        n_removable = (aggregate_vip < 0.5).sum()
        if n_removable > n_features * 0.2:
            recommendations.append(
                f"💡 FEATURE SELECTION: {n_removable} features (VIP<0.5) can be removed"
            )
            recommendations.append(f"    to reduce model complexity by {100*n_removable/n_features:.0f}%")
        
        for rec in recommendations:
            # Pad to fit box
            rec_padded = rec[:63].ljust(63)
            print(f"│ {rec_padded} │")
        
        print("└─────────────────────────────────────────────────────────────────┘")

    # -------------------------------------------------------------------------
    # Step 7: Figure Export
    # -------------------------------------------------------------------------

    def _step_export_figures(self) -> None:
        """Step 7: Export analysis report figures based on the session context."""
        print_step(7, self.TOTAL_STEPS, "Export Figures")

        print("Would you like to export analysis report figures?")
        print("  This creates a small, dataset-specific report (encoding, performance, VIP).")
        print()

        choice = prompt_choice("Export figures?", ["Y", "N"], default="N")
        if choice.upper() == "N":
            print("Skipping figure export.")
            return

        output_path = input("Output directory [analysis_figures]: ").strip()
        output_dir = Path(output_path or "analysis_figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from .manuscript.figure_context import FigureContext
            from .manuscript import figures as manuscript_figures

            available = manuscript_figures.list_available_report_figures()
            print()
            print("Available report figures:")
            for key, title in available.items():
                print(f"  {key}. {title}")
            print(f"Default set: {', '.join(manuscript_figures.ANALYSIS_REPORT_FIGURES)}")
            print()

            selection_input = input(
                "Select figures by number (comma-separated) or ENTER for default set: "
            ).strip()

            if selection_input:
                selection = [s.strip() for s in selection_input.split(",") if s.strip()]
            else:
                selection = list(manuscript_figures.ANALYSIS_REPORT_FIGURES)

            ctx = FigureContext.from_session(self.state)
            combine = prompt_choice("Combine into a multi-panel figure?", ["Y", "N"], default="N")
            if combine.upper() == "Y":
                layout_input = input("Layout (rows x cols) [auto]: ").strip()
                layout = manuscript_figures._parse_layout(layout_input) if layout_input else None
                filename = input("Combined filename [analysis_report_combined.png]: ").strip()
                filename = filename or "analysis_report_combined.png"
                generated = manuscript_figures.generate_report_figures(
                    ctx,
                    output_dir,
                    selection=selection,
                    combine=True,
                    layout=layout,
                    filename=filename,
                )
            else:
                generated = manuscript_figures.generate_report_figures(
                    ctx,
                    output_dir,
                    selection=selection,
                    combine=False,
                )

            print_success(f"Figures saved to: {output_dir}")
            for path in generated:
                print(f"  - {path.name}")
        except Exception as e:
            print_error(f"Figure export failed: {e}")
            return


def run_interactive_session(data_dir: str = "data") -> Optional[Dict[str, Any]]:
    """
    Run an interactive neutrosophic PLS session.
    
    This is the main entry point for the interactive CLI.
    
    Parameters
    ----------
    data_dir : str
        Directory to search for data files
        
    Returns
    -------
    dict or None
        Results dictionary if successful, None if cancelled
    """
    session = InteractiveSession(data_dir=data_dir)
    return session.run()


if __name__ == "__main__":
    run_interactive_session()
