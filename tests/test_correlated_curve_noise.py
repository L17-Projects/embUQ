from math import log, pi

import numpy as np
import pytest

from meso_uq.noise import (
    CorrelatedCurveNoiseConfig,
    CurveGrid,
    build_block_correlated_curve_covariance,
    build_correlated_curve_covariance,
    compose_total_covariance,
    gaussian_log_likelihood_from_covariance,
)
from meso_uq.noise.covariance import _cholesky_with_jitter


def test_correlated_curve_covariance_is_symmetric_and_summarized():
    grid = CurveGrid((0.0, 1.0, 2.0), curve_id="stretch")
    config = CorrelatedCurveNoiseConfig(amplitude=0.4, length_scale=1.25)

    result = build_correlated_curve_covariance(grid, config)

    assert result.covariance.shape == (3, 3)
    assert np.allclose(result.covariance, result.covariance.T)
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)
    assert np.diag(result.covariance) == pytest.approx((0.16, 0.16, 0.16))
    assert result.summary["kernel"] == "squared_exponential"
    assert result.summary["curve_ids"] == ["stretch"]
    assert result.summary["cholesky_success"] is True


def test_disabled_correlated_noise_preserves_diagonal_gaussian_likelihood():
    grid = CurveGrid((0.0, 1.0), curve_id="diag")
    correlated = build_correlated_curve_covariance(
        grid,
        CorrelatedCurveNoiseConfig(enabled=False, amplitude=0.5),
    )

    total = compose_total_covariance([0.25, 0.36], correlated)
    value = gaussian_log_likelihood_from_covariance([0.2, -0.3], total)

    expected = sum(
        -0.5 * ((residual * residual) / variance + log(2.0 * pi * variance))
        for residual, variance in zip((0.2, -0.3), (0.25, 0.36))
    )
    assert np.allclose(correlated.covariance, np.zeros((2, 2)))
    assert np.allclose(total.covariance, np.diag([0.25, 0.36]))
    assert value == pytest.approx(expected)


def test_block_correlated_curve_covariance_has_no_cross_curve_terms():
    result = build_block_correlated_curve_covariance(
        (CurveGrid((0.0, 1.0), "a"), CurveGrid((0.0,), "b")),
        CorrelatedCurveNoiseConfig(amplitude=0.2, length_scale=1.0),
    )

    assert result.covariance[0, 2] == pytest.approx(0.0)
    assert result.covariance[1, 2] == pytest.approx(0.0)
    assert result.summary["curve_ids"] == ["a", "b"]


def test_repeated_grid_points_are_stabilized_with_recorded_jitter():
    result = build_correlated_curve_covariance(
        CurveGrid((0.0, 0.0), "repeated"),
        CorrelatedCurveNoiseConfig(amplitude=1.0, length_scale=1.0, jitter=1e-8, max_jitter=1e-6),
    )

    assert result.jitter_added > 0.0
    assert result.cholesky is not None
    assert result.summary["jitter_added"] == pytest.approx(result.jitter_added)


def test_cholesky_jitter_attempts_maximum_before_failure():
    covariance = np.diag([-9e-7, 1.0])

    stabilized, cholesky, jitter_added = _cholesky_with_jitter(covariance, jitter=6e-7, max_jitter=1e-6)

    assert jitter_added == pytest.approx(1e-6)
    assert np.diag(stabilized) == pytest.approx((1e-7, 1.000001))
    assert cholesky.shape == (2, 2)


def test_covariance_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="amplitude"):
        CorrelatedCurveNoiseConfig(amplitude=-0.1)
    with pytest.raises(ValueError, match="Unsupported"):
        CorrelatedCurveNoiseConfig(kernel="periodic")
    with pytest.raises(ValueError, match="at least one"):
        CurveGrid(())

    disabled = build_correlated_curve_covariance(
        CurveGrid((0.0, 1.0), "singular"),
        CorrelatedCurveNoiseConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="not positive definite"):
        compose_total_covariance([0.0, 0.0], disabled)
    with pytest.raises(ValueError, match="shape"):
        compose_total_covariance([1.0], disabled)
    with pytest.raises(ValueError, match="Cholesky"):
        gaussian_log_likelihood_from_covariance([0.0, 0.0], disabled)
