"""
Neutrosophic Partial Least Squares (N-PLS) package.

A Python package for uncertainty-aware Partial Least Squares using
neutrosophic (Truth, Indeterminacy, Falsity) encoding.

Authors:
    Dickson Abdul-Wahab <dabdul-wahab@live.com>
    Ebenezer Aquisman Asare <aquisman1989@gmail.com>

University of Ghana

Usage:
    # As a library
    from neutrosophic_pls import NPLS, NPLSW, load_dataset
    
    # From command line
    python -m neutrosophic_pls --interactive
    python -m neutrosophic_pls --config study.yaml
    python -m neutrosophic_pls --data data.csv --target y
"""

__version__ = "1.0.0"
__author__ = "Dickson Abdul-Wahab, Ebenezer Aquisman Asare"
__email__ = "dabdul-wahab@live.com"
__license__ = "MIT"

from .algebra import neutro_inner, neutro_norm
from .model import NPLS, NPLSW, ReliabilityWeightedNPLS, PNPLS
from .interactive import run_interactive_session, InteractiveSession
from .reduction import check_reduction
from .vip import compute_nvip
from .semantic import semantic_projection
from .simulate import generate_simulation
from .data_micromass import load_micromass
from .data_idrc import load_idrc_wheat, load_idrc_train_test
from .data_loader import (
    load_dataset,
    load_dataframe,
    encode_neutrosophic,
    DatasetConfig,
    EncoderConfig,
    list_available_datasets,
    interactive_load_dataset,
    load_preset,
)
from .study_config import (
    StudyConfig,
    DatasetSettings,
    ModelSettings,
    EvaluationSettings,
    OutputSettings,
)
from .metrics import (
    evaluation_metrics,
    component_recovery,
    rmsep,
    r2_score,
    mean_absolute_error,
    mape,
    bias,
    sep,
    rpd,
    rer,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core models
    "NPLS",
    "NPLSW",
    "ReliabilityWeightedNPLS",
    "PNPLS",
    # Interactive
    "run_interactive_session",
    "InteractiveSession",
    # Algebra
    "neutro_inner",
    "neutro_norm",
    # Analysis
    "compute_nvip",
    "check_reduction",
    "semantic_projection",
    # Data loading
    "load_dataset",
    "load_dataframe",
    "encode_neutrosophic",
    "DatasetConfig",
    "EncoderConfig",
    "list_available_datasets",
    "interactive_load_dataset",
    "load_preset",
    "load_micromass",
    "load_idrc_wheat",
    "load_idrc_train_test",
    # Configuration
    "StudyConfig",
    "DatasetSettings",
    "ModelSettings",
    "EvaluationSettings",
    "OutputSettings",
    # Simulation & metrics
    "generate_simulation",
    "evaluation_metrics",
    "component_recovery",
    "rmsep",
    "r2_score",
    "mean_absolute_error",
    "mape",
    "bias",
    "sep",
    "rpd",
    "rer",
]
