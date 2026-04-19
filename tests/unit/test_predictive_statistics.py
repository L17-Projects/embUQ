from __future__ import annotations

import numpy as np
import pytest

from meso_uq.predictive_statistics import (
    _validate_shapes,
    compute_interval_statistics,
    gaussian_mixture_quantiles,
)


def test_gaussian_mixture_quantiles_matches_empirical_quantiles_when_std_zero():
    means = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    stds = np.zeros_like(means)
    got = gaussian_mixture_quantiles(means, stds, [0.05, 0.5, 0.95])
    expected = np.quantile(means, [0.05, 0.5, 0.95], axis=0)
    assert np.allclose(got, expected)


def test_compute_interval_statistics_uses_observation_noise_in_std():
    means = np.array([[1.0, 2.0], [3.0, 4.0]])
    stds = np.ones_like(means) * 2.0
    latent = compute_interval_statistics(means, None, percentiles=(90,), include_observation_noise=False)
    predictive = compute_interval_statistics(means, stds, percentiles=(90,), include_observation_noise=True)
    expected_predictive_std = np.sqrt(np.var(means, axis=0) + np.mean(stds**2, axis=0))
    assert np.allclose(predictive["std"], expected_predictive_std)
    assert np.all(predictive["std"] > latent["std"])


# ---------------------------------------------------------------------------
# _validate_shapes error paths (lines 17, 22)
# ---------------------------------------------------------------------------

def test_validate_shapes_raises_for_1d_input():
    with pytest.raises(ValueError, match="shape"):
        _validate_shapes(np.array([1.0, 2.0, 3.0]), None)


def test_validate_shapes_raises_for_shape_mismatch():
    mean = np.ones((3, 2))
    std = np.ones((3, 3))
    with pytest.raises(ValueError, match="same shape"):
        _validate_shapes(mean, std)


# ---------------------------------------------------------------------------
# gaussian_mixture_quantiles error paths (lines 31, 33)
# ---------------------------------------------------------------------------

def test_gaussian_mixture_quantiles_raises_for_empty_quantiles():
    means = np.ones((3, 2))
    stds = np.zeros_like(means)
    with pytest.raises(ValueError, match="non-empty"):
        gaussian_mixture_quantiles(means, stds, [])


def test_gaussian_mixture_quantiles_raises_for_out_of_range_quantiles():
    means = np.ones((3, 2))
    stds = np.zeros_like(means)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gaussian_mixture_quantiles(means, stds, [0.5, 1.5])


# ---------------------------------------------------------------------------
# All-non-finite column (lines 41-42)
# ---------------------------------------------------------------------------

def test_gaussian_mixture_quantiles_nan_column_returns_nan():
    means = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    stds = np.array([[np.nan, 0.1], [np.nan, 0.1]])
    result = gaussian_mixture_quantiles(means, stds, [0.5])
    assert np.isnan(result[0, 0])
    assert np.isfinite(result[0, 1])


# ---------------------------------------------------------------------------
# Mixture CDF branches: positive-std and zero-std components (lines 58-62)
# ---------------------------------------------------------------------------

def test_gaussian_mixture_quantiles_mixed_zero_positive_std():
    """Column with some zero-std and some positive-std samples exercises both CDF branches."""
    rng = np.random.default_rng(42)
    # 4 samples with positive std, 4 samples with zero std (point masses)
    means = np.concatenate([rng.normal(0.0, 0.1, (4, 1)), np.full((4, 1), 0.0)], axis=0)
    stds = np.concatenate([np.full((4, 1), 0.2), np.zeros((4, 1))], axis=0)
    result = gaussian_mixture_quantiles(means, stds, [0.05, 0.5, 0.95])
    assert result.shape == (3, 1)
    assert np.all(np.isfinite(result))
    # Quantiles must be monotonically non-decreasing
    assert result[0, 0] <= result[1, 0] <= result[2, 0]
