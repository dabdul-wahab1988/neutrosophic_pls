"""
Study Configuration Schema for Neutrosophic PLS
================================================

Provides structured configuration for reproducible N-PLS studies using
YAML/JSON configuration files.

Author: NeutroProject
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml


@dataclass
class DatasetSettings:
    """Dataset configuration."""
    path: str
    target: str
    task: Literal["regression", "classification"] = "regression"
    features: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    snv: bool = False
    encoding: Union[str, Dict[str, Any]] = "default"
    spectral_noise_db: float = -20.0
    format: Optional[str] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.exclude_columns is None:
            self.exclude_columns = []


@dataclass
class ModelSettings:
    """Model configuration."""
    method: Literal["PLS", "NPLS", "NPLSW", "PNPLS", "all"] = "all"
    max_components: int = 10
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    lambda_indeterminacy: float = 1.0  # For NPLSW/PNPLS
    lambda_falsity: float = 1.0  # For PNPLS
    alpha: float = 1.0  # For PNPLS falsity softness
    
    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.channel_weights, list):
            self.channel_weights = tuple(self.channel_weights)


@dataclass
class EvaluationSettings:
    """Evaluation/CV configuration."""
    cv_folds: int = 5
    inner_cv_folds: int = 3
    repeats: int = 3
    random_state: int = 42
    compute_vip: bool = True


@dataclass
class OutputSettings:
    """Output configuration."""
    output_dir: str = "results"
    save_predictions: bool = True
    save_vip: bool = True
    generate_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300


@dataclass
class StudyConfig:
    """
    Complete study configuration.
    
    Can be loaded from YAML or JSON files for reproducible studies.
    """
    dataset: DatasetSettings
    model: ModelSettings = field(default_factory=ModelSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    name: str = "npls_study"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyConfig":
        """Create from dictionary."""
        dataset = DatasetSettings(**d.get("dataset", {}))
        model = ModelSettings(**d.get("model", {}))
        evaluation = EvaluationSettings(**d.get("evaluation", {}))
        output = OutputSettings(**d.get("output", {}))
        
        return cls(
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            output=output,
            name=d.get("name", "npls_study"),
            description=d.get("description", ""),
        )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StudyConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StudyConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "StudyConfig":
        """Load configuration from file (auto-detect format)."""
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".json":
            return cls.from_json(path)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(path)
            except Exception:
                return cls.from_json(path)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_idrc_wheat_config(
    output_dir: str = "results/idrc_wheat",
    repeats: int = 3,
    cv_folds: int = 5,
) -> StudyConfig:
    """Get preset configuration for IDRC wheat protein study."""
    return StudyConfig(
        name="IDRC Wheat Protein Study",
        description="NIR spectroscopy protein prediction using N-PLS methods",
        dataset=DatasetSettings(
            path="data/MA_A2.csv",
            target="Protein",
            task="regression",
            exclude_columns=["ID"],
            snv=True,
            name="IDRC 2016 Wheat Protein",
        ),
        model=ModelSettings(
            method="all",
            max_components=15,
            channel_weights=(1.0, 1.0, 1.0),
        ),
        evaluation=EvaluationSettings(
            cv_folds=cv_folds,
            inner_cv_folds=3,
            repeats=repeats,
            compute_vip=True,
        ),
        output=OutputSettings(
            output_dir=output_dir,
            save_predictions=True,
            save_vip=True,
            generate_figures=True,
        ),
    )


def get_quick_test_config(
    data_path: str,
    target: str,
    task: Literal["regression", "classification"] = "regression",
) -> StudyConfig:
    """Get quick test configuration."""
    return StudyConfig(
        name="Quick Test",
        description="Quick test run with minimal iterations",
        dataset=DatasetSettings(
            path=data_path,
            target=target,
            task=task,
        ),
        model=ModelSettings(
            method="all",
            max_components=5,
        ),
        evaluation=EvaluationSettings(
            cv_folds=3,
            inner_cv_folds=2,
            repeats=1,
        ),
        output=OutputSettings(
            output_dir="results_quick",
            generate_figures=False,
        ),
    )


# =============================================================================
# Interactive Config Builder
# =============================================================================

def interactive_build_config() -> StudyConfig:
    """Interactively build a study configuration."""
    print("\n" + "=" * 60)
    print("N-PLS Study Configuration Builder")
    print("=" * 60)
    
    # Dataset settings
    print("\n--- Dataset Settings ---")
    data_path = input("Data file path: ").strip()
    
    # Try to load and show columns
    try:
        from .data_loader import load_dataframe
        df = load_dataframe(data_path)
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"(Could not preview file: {e})")
    
    target = input("Target column name: ").strip()
    
    task_input = input("Task (r=regression, c=classification) [r]: ").strip().lower()
    task = "classification" if task_input == "c" else "regression"
    
    exclude_input = input("Columns to exclude (comma-separated, or empty): ").strip()
    exclude_columns = [c.strip() for c in exclude_input.split(",") if c.strip()] or None
    
    snv_input = input("Apply SNV normalization? (y/n) [n]: ").strip().lower()
    snv = snv_input == "y"
    
    # Model settings
    print("\n--- Model Settings ---")
    method_input = input("Method (PLS/NPLS/NPLSW/PNPLS/all) [all]: ").strip().upper() or "all"
    method = method_input if method_input in ("PLS", "NPLS", "NPLSW", "PNPLS", "ALL") else "all"
    method = method if method != "ALL" else "all"

    max_comp_input = input("Max components [10]: ").strip()
    max_components = int(max_comp_input) if max_comp_input else 10
    lambda_ind_input = input("Lambda indeterminacy (for NPLSW/PNPLS) [1.0]: ").strip()
    lambda_ind = float(lambda_ind_input) if lambda_ind_input else 1.0
    lambda_f_input = input("Lambda falsity (for PNPLS) [1.0]: ").strip()
    lambda_f = float(lambda_f_input) if lambda_f_input else 1.0
    alpha_input = input("Alpha (falsity softness, PNPLS >0) [1.0]: ").strip()
    alpha = float(alpha_input) if alpha_input else 1.0
    
    # Evaluation settings
    print("\n--- Evaluation Settings ---")
    cv_input = input("CV folds [5]: ").strip()
    cv_folds = int(cv_input) if cv_input else 5
    
    repeats_input = input("CV repeats [3]: ").strip()
    repeats = int(repeats_input) if repeats_input else 3
    
    # Output settings
    print("\n--- Output Settings ---")
    output_dir = input("Output directory [results]: ").strip() or "results"
    
    study_name = input("Study name [npls_study]: ").strip() or "npls_study"
    
    # Build config
    config = StudyConfig(
        name=study_name,
        dataset=DatasetSettings(
            path=data_path,
            target=target,
            task=task,
            exclude_columns=exclude_columns,
            snv=snv,
        ),
        model=ModelSettings(
            method=method,
            max_components=max_components,
            lambda_indeterminacy=lambda_ind,
            lambda_falsity=lambda_f,
            alpha=alpha,
        ),
        evaluation=EvaluationSettings(
            cv_folds=cv_folds,
            repeats=repeats,
        ),
        output=OutputSettings(
            output_dir=output_dir,
        ),
    )
    
    # Offer to save
    save_input = input("\nSave config to file? (path or empty to skip): ").strip()
    if save_input:
        if save_input.endswith(".json"):
            config.to_json(save_input)
        else:
            if not save_input.endswith((".yaml", ".yml")):
                save_input += ".yaml"
            config.to_yaml(save_input)
        print(f"Config saved to: {save_input}")
    
    return config
