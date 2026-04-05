from __future__ import annotations

import numpy as np

from meso_uq.predictive_statistics import compute_interval_statistics, gaussian_mixture_quantiles


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
