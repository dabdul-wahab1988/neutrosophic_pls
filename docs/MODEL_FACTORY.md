# Model Factory Walkthrough

This document explains how model creation is centralized in `neutrosophic_pls/model_factory.py`,
why it exists, and how to use it when adding new code paths.

## Goals

- Keep model creation consistent across CLI, interactive flows, pipeline runs, and manuscript scripts.
- Avoid duplicated constructor arguments scattered across modules.
- Provide a single place to add new model variants or change defaults safely.

## Entry Points

### `create_model(config, method, n_components=None)`

Use this when you already have a `StudyConfig` instance.

- Reads parameters from `config.model`.
- Uses `config.model.max_components` unless `n_components` overrides it.
- Delegates to `create_model_from_params` to keep logic in one place.

Example:

```python
from neutrosophic_pls.model_factory import create_model

model = create_model(config, method="NPLSW", n_components=8)
```

### `create_model_from_params(method, n_components=5, ..., **kwargs)`

Use this when you do not have a full `StudyConfig` (e.g., manuscript scripts,
encoder scoring loops, pipeline config objects).

- Accepts direct parameters and forwards optional kwargs to the constructor.
- Supports PLS, NPLS, NPLSW, PNPLS by method name.
- Allows extra flags like `scale=True` for PLS via `**kwargs`.

Example:

```python
from neutrosophic_pls.model_factory import create_model_from_params

model = create_model_from_params(
    method="PLS",
    n_components=10,
    scale=True,
)
```

## Parameter Flow

The factory expects the same parameter names used throughout the codebase:

- `n_components`: number of components for all models.
- `channel_weights`: only used by NPLS and NPLSW.
- `lambda_indeterminacy`: used by NPLSW and PNPLS.
- `lambda_falsity`: used by NPLS (for falsity scaling), NPLSW, and PNPLS.
- `alpha`: used by NPLSW and PNPLS.

`create_model` pulls all of these from `config.model`:

```text
config.model.max_components -> n_components
config.model.channel_weights -> channel_weights
config.model.lambda_indeterminacy -> lambda_indeterminacy
config.model.lambda_falsity -> lambda_falsity
config.model.alpha -> alpha
```

## Where It Is Used

These modules now use the factory to construct models:

- CLI / main: `neutrosophic_pls/__main__.py`
- Interactive workflow: `neutrosophic_pls/interactive.py`
- Pipeline runner: `neutrosophic_pls/pipeline/runner.py`
- Encoder auto-selection: `neutrosophic_pls/encoders.py`
- Manuscript scripts: `neutrosophic_pls/manuscript/experiments.py`, `neutrosophic_pls/manuscript/figures.py`

If you add new scripts or flows, prefer `create_model` when a `StudyConfig`
exists, otherwise use `create_model_from_params`.

## Extending the Factory

To add a new model variant:

1. Add the new class import in `neutrosophic_pls/model_factory.py`.
2. Extend the `if/elif` dispatch in `create_model_from_params`.
3. Update documentation or user-facing method lists if needed.

Keep all constructor arguments in the factory so other modules stay decoupled
from model details.
