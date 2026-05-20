from math import log, pi

import pytest

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    build_additive_relative_observation_noise,
    compose_additive_relative_gaussian_likelihood,
    legacy_multiplicative_likelihood,
)


def test_additive_relative_observation_noise_exposes_named_components():
    config = AdditiveRelativeObservationNoiseConfig(additive_sigma=0.2, relative_sigma=0.1)

    result = build_additive_relative_observation_noise([-2.0, 0.0, 4.0], config)

    assert result.additive_variance == pytest.approx((0.04, 0.04, 0.04))
    assert result.relative_variance == pytest.approx((0.04, 0.0, 0.16))
    assert result.floor_variance == pytest.approx((0.0, 0.0, 0.0))
    assert result.total_variance == pytest.approx((0.08, 0.04, 0.20))
    assert result.standard_deviation == pytest.approx((0.08**0.5, 0.2, 0.20**0.5))
    assert result.variance_components["additive"] == result.additive_variance
    assert result.variance_components["relative"] == result.relative_variance


def test_relative_only_config_recovers_abs_scaled_legacy_sigma():
    config = AdditiveRelativeObservationNoiseConfig.legacy_equivalent(0.25)

    result = build_additive_relative_observation_noise([-2.0, 4.0], config)
    legacy = legacy_multiplicative_likelihood([-2.0, 4.0], 0.25, absolute_reference=True)

    assert result.standard_deviation == pytest.approx(legacy.standard_deviation)
    assert result.total_variance == pytest.approx((0.5**2, 1.0**2))


def test_gaussian_likelihood_is_finite_for_zero_prediction_with_additive_noise():
    config = AdditiveRelativeObservationNoiseConfig(additive_sigma=0.5, relative_sigma=0.25)

    likelihood = compose_additive_relative_gaussian_likelihood([0.1], [0.0], config)

    expected = -0.5 * ((0.1 * 0.1) / 0.25 + log(2.0 * pi * 0.25))
    assert likelihood.noise.total_variance == pytest.approx((0.25,))
    assert likelihood.log_likelihood == pytest.approx(expected)


def test_relative_only_zero_prediction_fails_only_at_likelihood_stage():
    config = AdditiveRelativeObservationNoiseConfig(relative_sigma=0.5)

    result = build_additive_relative_observation_noise([0.0], config)

    assert result.total_variance == pytest.approx((0.0,))
    with pytest.raises(ValueError, match="positive"):
        compose_additive_relative_gaussian_likelihood([0.0], [0.0], config)


def test_floors_stabilize_near_zero_prediction_scales():
    config = AdditiveRelativeObservationNoiseConfig(
        relative_sigma=0.5,
        prediction_scale_floor=2.0,
        minimum_total_variance=2.0,
    )

    result = build_additive_relative_observation_noise([0.0], config)

    assert result.relative_variance == pytest.approx((1.0,))
    assert result.floor_variance == pytest.approx((1.0,))
    assert result.total_variance == pytest.approx((2.0,))


def test_observation_noise_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="At least one"):
        AdditiveRelativeObservationNoiseConfig()
    with pytest.raises(ValueError, match=">= 0.0"):
        AdditiveRelativeObservationNoiseConfig(additive_sigma=-1.0)
    with pytest.raises(ValueError, match="count"):
        compose_additive_relative_gaussian_likelihood(
            [1.0, 2.0],
            [1.0],
            AdditiveRelativeObservationNoiseConfig(additive_sigma=0.1),
        )
    with pytest.raises(ValueError, match="finite"):
        build_additive_relative_observation_noise(
            [1e308],
            AdditiveRelativeObservationNoiseConfig(relative_sigma=1e308),
        )
