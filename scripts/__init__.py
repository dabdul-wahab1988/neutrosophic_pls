"""
Scripts for N-PLS Simulation Study
==================================

This package contains the designed simulation study for the
Neutrosophic PLS vs Classical PLS paper.

Modules:
- simulation_study: Main three-stage experimental design
- visualization: Publication-quality figure generation
"""

from .simulation_study import (
    run_full_study,
    run_screening_study,
    run_response_surface_study,
    run_micromass_study,
    ScreeningConfig,
    ResponseSurfaceConfig,
    MicroMassConfig,
)

from .visualization import generate_all_figures

__all__ = [
    "run_full_study",
    "run_screening_study",
    "run_response_surface_study",
    "run_micromass_study",
    "ScreeningConfig",
    "ResponseSurfaceConfig",
    "MicroMassConfig",
    "generate_all_figures",
]
