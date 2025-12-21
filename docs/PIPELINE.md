# Pipeline Usage

## CLI wizard + run

```bash
python -m neutrosophic_pls.cli wizard --output examples/sim_config.yaml --n-components 2 --n-samples 50 --n-features 20
python -m neutrosophic_pls.cli run-pipeline --config examples/sim_config.yaml --output-dir artifacts
```

## MicroMass fixture

```bash
python -m neutrosophic_pls.cli run-pipeline --config examples/micromass_fixture.yaml --output-dir artifacts_micromass
```

## Model types

- `npls`: standard neutrosophic PLS (channel weighting)
- `nplsw`: sample-weighted variant (lambda_indeterminacy, weight_normalize)
- `pnpls`: probabilistic variant (lambda_indeterminacy, lambda_falsity, alpha)

## Notebook helper

```python
from neutrosophic_pls.notebook import run_pipeline
result = run_pipeline({"mode": "simulate", "n_samples": 30, "n_features": 8, "n_components": 2, "output_dir": "artifacts_nb"})
```

## Outputs

- Metrics (`metrics.txt`), VIPs (`vip.txt`), semantics (via returned dict), model object in-memory.
- Metadata includes config, md5 for dataset, and seeds.

## Notes

- MicroMass loader prefers the packaged fixture; replace `micromass_path` with a real dataset path to run on full data (update checksum/licensing in DECISIONS.md).
- Set `SKIP_MICROMASS=1` in CI to avoid network downloads. The fixture remains deterministic for tests.
