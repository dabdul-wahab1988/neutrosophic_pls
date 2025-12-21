"""
Model Factory for Neutrosophic PLS
==================================

Centralizes the creation of model instances (PLS, NPLS, NPLSW, PNPLS)
from a StudyConfig configuration object.

This decouples the CLI/Interactive runners from specific model parameter
handling, making the codebase easier to maintain and extend.
"""

from typing import Any, Optional, Union, Tuple

from sklearn.cross_decomposition import PLSRegression

from .model import NPLS, NPLSW, PNPLS
from .study_config import StudyConfig


def create_model(
    config: StudyConfig, 
    method: str, 
    n_components: Optional[int] = None
) -> Union[PLSRegression, NPLS, NPLSW, PNPLS]:
    """
    Create a model instance based on configuration and method name.

    Parameters
    ----------
    config : StudyConfig
        The complete study configuration object.
    method : str
        The method name ("PLS", "NPLS", "NPLSW", "PNPLS").
    n_components : int, optional
        Override the number of components from config (e.g., specific to a CV fold).
        If None, uses `config.model.max_components`.

    Returns
    -------
    Model instance (sklearn PLS or neutrosophic variant).
    """
    # Use the more granular factory function
    return create_model_from_params(
        method=method,
        n_components=n_components if n_components is not None else config.model.max_components,
        channel_weights=config.model.channel_weights,
        lambda_indeterminacy=config.model.lambda_indeterminacy,
        lambda_falsity=config.model.lambda_falsity,
        alpha=config.model.alpha
    )


def create_model_from_params(
    method: str,
    n_components: int = 5,
    channel_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0),
    lambda_indeterminacy: float = 0.5,
    lambda_falsity: float = 0.5,
    alpha: float = 2.0,
    **kwargs: Any
) -> Union[PLSRegression, NPLS, NPLSW, PNPLS]:
    """
    Create a model instance from direct parameters.

    This is useful when a full StudyConfig is not available (e.g., in sub-modules
    or research scripts).

    Parameters
    ----------
    method : str
        The method name ("PLS", "NPLS", "NPLSW", "PNPLS").
    n_components : int
        Number of components.
    channel_weights : tuple
        Weights for T/I/F channels.
    lambda_indeterminacy : float
        Indeterminacy weighting parameter.
    lambda_falsity : float
        Falsity weighting/prior parameter.
    alpha : float
        Softness parameter for PNPLS.
    **kwargs : dict
        Additional parameters passed to model constructors.

    Returns
    -------
    Model instance.
    """
    method = method.upper()

    if method == "PLS":
        return PLSRegression(n_components=n_components, scale=False, **kwargs)

    elif method == "NPLS":
        return NPLS(
            n_components=n_components,
            channel_weights=channel_weights,
            lambda_falsity=lambda_falsity,
            **kwargs
        )

    elif method == "NPLSW":
        return NPLSW(
            n_components=n_components,
            channel_weights=channel_weights,
            lambda_indeterminacy=lambda_indeterminacy,
            lambda_falsity=lambda_falsity,
            alpha=alpha,
            **kwargs
        )

    elif method == "PNPLS":
        return PNPLS(
            n_components=n_components,
            lambda_indeterminacy=lambda_indeterminacy,
            lambda_falsity=lambda_falsity,
            alpha=alpha,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown method: {method}")
