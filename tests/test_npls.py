"""
Tests for Neutrosophic PLS package.
"""

import numpy as np
import pytest
import importlib.util


def test_npls_fit_predict():
    """Test basic NPLS fitting and prediction."""
    from neutrosophic_pls import NPLS
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 50, 10
    
    # Create TIF tensors
    X = np.random.randn(n_samples, n_features, 3)
    X[..., 0] = np.random.randn(n_samples, n_features)  # Truth
    X[..., 1] = np.abs(np.random.randn(n_samples, n_features)) * 0.1  # Indeterminacy
    X[..., 2] = (np.random.rand(n_samples, n_features) > 0.95).astype(float)  # Falsity
    
    y = np.random.randn(n_samples, 1, 3)
    y[..., 1] = 0
    y[..., 2] = 0
    
    # Fit model
    model = NPLS(n_components=2)
    model.fit(X, y)
    
    # Predict
    predictions = model.predict(X)
    
    assert predictions.shape[0] == n_samples


def test_nplsw_fit_predict():
    """Test NPLSW fitting and prediction."""
    from neutrosophic_pls import NPLSW
    
    np.random.seed(42)
    n_samples, n_features = 50, 10
    
    X = np.random.randn(n_samples, n_features, 3)
    X[..., 1] = np.abs(X[..., 1]) * 0.1
    X[..., 2] = 0
    
    y = np.random.randn(n_samples, 1, 3)
    y[..., 1] = 0
    y[..., 2] = 0
    
    model = NPLSW(n_components=2, lambda_indeterminacy=1.0)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert predictions.shape[0] == n_samples


def test_vip_computation():
    """Test VIP computation."""
    from neutrosophic_pls import NPLS, compute_nvip
    
    np.random.seed(42)
    n_samples, n_features = 50, 10
    
    X = np.random.randn(n_samples, n_features, 3)
    X[..., 1] = np.abs(X[..., 1]) * 0.1
    X[..., 2] = 0
    
    y = np.random.randn(n_samples, 1, 3)
    y[..., 1] = 0
    y[..., 2] = 0
    
    model = NPLS(n_components=2)
    model.fit(X, y)
    
    vip = compute_nvip(model, X)
    
    assert "aggregate" in vip
    assert "T" in vip
    assert "I" in vip
    assert "F" in vip
    assert len(vip["aggregate"]) == n_features
    
    # VIP channels should sum to aggregate
    channel_sum = vip["T"] + vip["I"] + vip["F"]
    np.testing.assert_array_almost_equal(vip["aggregate"], channel_sum, decimal=5)


def test_data_loader():
    """Test universal data loader."""
    from neutrosophic_pls import encode_neutrosophic
    
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.random.randn(20)
    
    x_tif, y_tif = encode_neutrosophic(X, y, task="regression")
    
    assert x_tif.shape == (20, 5, 3)
    assert y_tif.shape == (20, 1, 3)


def test_study_config():
    """Test study configuration."""
    from neutrosophic_pls import StudyConfig, DatasetSettings, ModelSettings
    
    config = StudyConfig(
        name="Test Study",
        dataset=DatasetSettings(
            path="test.csv",
            target="y",
            task="regression",
        ),
        model=ModelSettings(
            method="all",
            max_components=5,
        ),
    )
    
    assert config.name == "Test Study"
    assert config.dataset.target == "y"
    assert config.model.max_components == 5
    
    # Test serialization
    d = config.to_dict()
    assert d["name"] == "Test Study"


def test_indeterminacy_validator():
    """Validate I against synthetic repeat measurements."""
    from neutrosophic_pls.validation import IndeterminacyValidator
    from neutrosophic_pls.data_loader import _entropy_surprisal, _normalize_entropy_scores

    rng = np.random.default_rng(0)
    n_samples, n_features = 30, 8
    X = rng.normal(size=(n_samples, n_features))

    # Create three repeats with small noise
    repeats = np.stack(
        [X + rng.normal(scale=0.05, size=X.shape) for _ in range(3)], axis=1
    )  # (n_samples, n_repeats, n_features)

    # Compute entropy-based indeterminacy on one replicate
    surprisal, _ = _entropy_surprisal(repeats[:, 0, :], bins=15)
    i_entropy, _ = _normalize_entropy_scores(surprisal, high_tail=95.0)

    validator = IndeterminacyValidator()
    results = validator.validate_against_repeats(i_entropy, repeats)

    assert "overall_correlation" in results
    assert results["overall_correlation"] > -1  # sanity check real number


def test_soft_falsity_helper():
    """Soft falsity should produce values in [0,1] with gradient."""
    from neutrosophic_pls.data_loader import _compute_soft_falsity

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    F = _compute_soft_falsity(X, threshold_soft=2.0, threshold_hard=3.5)

    assert F.min() >= 0.0
    assert F.max() <= 1.0
    assert np.any((F > 0) & (F < 1))  # gradient region exists


def test_indeterminacy_from_robust_z():
    """Robust-z indeterminacy stays in [0,1] and increases with |z|."""
    from neutrosophic_pls.data_loader import _indeterminacy_from_robust_z

    X = np.array([[0.0, 1.0, 3.5], [-3.5, 0.0, 0.5]])
    ind = _indeterminacy_from_robust_z(X, z_max=3.5)

    assert ind.shape == X.shape
    assert ind.min() >= 0.0 and ind.max() <= 1.0
    # Larger |z| should map to higher indeterminacy
    assert ind[0, 2] >= ind[0, 1] - 1e-6
    assert ind[0, 1] >= ind[0, 0] - 1e-6


def test_encode_neutrosophic_indeterminacy_basic():
    """Encoding should produce bounded I in [0, 1] for default encoder."""
    from neutrosophic_pls.data_loader import encode_neutrosophic

    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    x_tif, _ = encode_neutrosophic(X, y, snv=False)
    I = x_tif[..., 1]

    assert I.shape == X.shape
    assert I.min() >= 0.0
    assert I.max() <= 1.0


def test_encode_neutrosophic_falsity_robust_z_default():
    """Falsity encoding should produce bounded values with proper calibration."""
    from neutrosophic_pls.data_loader import encode_neutrosophic

    # Create data with proper variance - the probabilistic encoder will detect anomalies
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0, scale=1, size=(50, 10))
    # Add some clear outliers
    X[0, 0] = 20.0  # extreme outlier
    X[1, 5] = -15.0  # another outlier
    y = rng.normal(size=50)

    x_tif, _ = encode_neutrosophic(X, y, snv=False)
    T = x_tif[..., 0]
    I = x_tif[..., 1]
    F = x_tif[..., 2]

    # All channels should be bounded
    assert T.min() >= X.min() - 1e-6  # T is raw data
    assert I.min() >= 0 and I.max() <= 1.0  # I in [0, 1]
    assert F.min() >= 0 and F.max() <= 1.0  # F in [0, 1]
    
    # With calibrated power transforms, most I/F values should be compressed
    # towards zero (clean data), so medians should be small
    assert np.median(I) < 0.5
    assert np.median(F) < 0.5


# def test_cell_reliability_gate():
#    """Reliability gate should attenuate T as I/F increase."""
#    # _cell_reliability_gate function was removed in favor of precision weighting
#    pass


def test_pnpls_fit_predict():
    """PNPLS should fit on T-only data and produce accurate predictions."""
    from neutrosophic_pls import PNPLS

    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 6
    T = rng.normal(size=(n_samples, n_features))
    coef_true = rng.normal(size=(n_features, 1))
    y_truth = T @ coef_true + 0.01 * rng.normal(size=(n_samples, 1))

    x_tif = np.stack([T, np.zeros_like(T), np.zeros_like(T)], axis=-1)
    y_tif = np.stack([y_truth, np.zeros_like(y_truth), np.zeros_like(y_truth)], axis=-1)

    model = PNPLS(n_components=3, lambda_falsity=1.0)
    model.fit(x_tif, y_tif)
    preds = model.predict(x_tif)

    # predict returns 1D array for single-target (matching sklearn behavior)
    y_truth_flat = y_truth.ravel()
    r2 = 1.0 - float(np.sum((y_truth_flat - preds) ** 2) / np.sum((y_truth_flat - y_truth_flat.mean()) ** 2))

    assert preds.shape == y_truth_flat.shape
    assert r2 > 0.8


def test_rpca_encoder_outputs():
    """RPCA encoder should return bounded I/F and low-rank truth."""
    from neutrosophic_pls.encoders import encode_rpca_mixture

    rng = np.random.default_rng(5)
    n_samples, n_features = 20, 8
    low_rank = rng.normal(size=(n_samples, 2)) @ rng.normal(size=(2, n_features))
    sparse = np.zeros((n_samples, n_features))
    sparse[0, 0] = 8.0
    sparse[3, 5] = -6.0
    X = low_rank + sparse + 0.01 * rng.normal(size=(n_samples, n_features))

    result = encode_rpca_mixture(X)

    assert result.truth.shape == X.shape
    assert np.all(result.indeterminacy >= 0) and np.all(result.indeterminacy <= 1)
    assert np.all(result.falsity >= 0) and np.all(result.falsity <= 1)
    # Sparse corruption should raise falsity near injected spikes
    assert result.falsity[0, 0] > 0.5


def test_quantile_and_augmentation_encoders():
    """Quantile and augmentation encoders should stay bounded and preserve shape."""
    from neutrosophic_pls.encoders import encode_quantile_envelope, encode_augmentation_stability

    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 6))

    quantile_result = encode_quantile_envelope(X, lower_q=0.1, upper_q=0.9)
    augment_result = encode_augmentation_stability(X, n_augmentations=3, random_state=3)

    for res in (quantile_result, augment_result):
        assert res.truth.shape == X.shape
        assert np.all(res.indeterminacy >= 0) and np.all(res.indeterminacy <= 1)
        assert np.all(res.falsity >= 0) and np.all(res.falsity <= 1)


def test_wavelet_encoder_optional_dependency():
    """Wavelet encoder should guide users when PyWavelets is missing."""
    from neutrosophic_pls.encoders import encode_wavelet_multiscale

    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 32))

    if importlib.util.find_spec("pywt") is None:
        with pytest.raises(ImportError):
            encode_wavelet_multiscale(X)
    else:
        result = encode_wavelet_multiscale(X, high_bands=(1,), mid_bands=(2,))
        assert result.truth.shape == X.shape
        assert np.all(result.indeterminacy >= 0) and np.all(result.indeterminacy <= 1)
        assert np.all(result.falsity >= 0) and np.all(result.falsity <= 1)


def test_auto_encoder_selection_metadata():
    """Auto encoder selection should return metadata about the winning encoder."""
    from neutrosophic_pls.data_loader import encode_neutrosophic

    rng = np.random.default_rng(7)
    X = rng.normal(size=(25, 5))
    y = X[:, 0] * 0.8 + 0.1 * rng.normal(size=25)

    x_tif, y_tif, meta = encode_neutrosophic(
        X,
        y,
        encoding={
            "name": "auto",
            "candidates": ["probabilistic", "quantile"],
            "cv_folds": 2,
            "max_components": 2,
        },
        return_metadata=True,
    )

    assert x_tif.shape[0] == X.shape[0]
    assert "encoder" in meta
    assert meta["encoder"].get("auto_selected") is True
    assert isinstance(meta["encoder"].get("auto_scores"), dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
