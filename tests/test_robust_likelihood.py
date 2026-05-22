from math import lgamma, log, pi

import numpy as np
import pytest

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    LikelihoodInputs,
    RobustLikelihoodConfig,
    RobustLikelihoodKind,
    build_additive_relative_observation_noise,
    evaluate_observation_likelihood,
)


def test_gaussian_robust_fallback_matches_normal_pointwise_values():
    inputs = LikelihoodInputs(
        observed=(1.2, -0.5),
        predicted=(1.0, -0.2),
        standard_deviation=(0.5, 0.25),
    )

    result = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind="gaussian"))

    expected = []
    for residual, sigma in zip((0.2, -0.3), (0.5, 0.25)):
        z = residual / sigma
        expected.append(-0.5 * (z * z + log(2.0 * pi * sigma * sigma)))
    assert result.covariance_mode == "diagonal"
    assert result.pointwise_log_likelihood == pytest.approx(expected)
    assert result.log_likelihood == pytest.approx(sum(expected))
    assert result.influence_weights == pytest.approx((1.0, 1.0))


def test_student_t_reduces_outlier_leverage_without_changing_inputs():
    inputs = LikelihoodInputs(observed=(8.0,), predicted=(0.0,), standard_deviation=(1.0,))

    gaussian = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig())
    student = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=4.0))

    assert student.log_likelihood > gaussian.log_likelihood
    assert student.influence_weights[0] < 1.0
    assert student.standardized_residuals == pytest.approx((8.0,))


def test_student_t_numeric_value_uses_scaled_density():
    inputs = LikelihoodInputs(observed=(1.5,), predicted=(1.0,), standard_deviation=(0.25,))
    nu = 5.0

    result = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind=RobustLikelihoodKind.STUDENT_T, degrees_of_freedom=nu))

    z = 0.5 / 0.25
    expected = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * pi) - log(0.25)
    expected -= ((nu + 1.0) / 2.0) * log(1.0 + (z * z) / nu)
    assert result.log_likelihood == pytest.approx(expected)


def test_covariance_gaussian_and_student_t_modes_are_finite():
    covariance = np.asarray([[0.25, 0.05], [0.05, 0.36]], dtype=float)
    inputs = LikelihoodInputs(observed=(1.0, 1.5), predicted=(0.8, 1.4), covariance=covariance)

    gaussian = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind="gaussian"))
    student = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=6.0))

    assert gaussian.covariance_mode == "full"
    assert student.covariance_mode == "full"
    assert len(student.standardized_residuals) == 2
    assert np.isfinite(student.log_likelihood)
    assert all(weight > 0.0 for weight in student.influence_weights)


def test_robust_likelihood_accepts_observation_noise_components():
    noise = build_additive_relative_observation_noise(
        [2.0, 4.0],
        AdditiveRelativeObservationNoiseConfig(additive_sigma=0.1, relative_sigma=0.25),
    )
    inputs = LikelihoodInputs(
        observed=(2.1, 3.7),
        predicted=noise.predictions,
        standard_deviation=noise.standard_deviation,
        variance_components=noise.variance_components,
    )

    result = evaluate_observation_likelihood(inputs, RobustLikelihoodConfig(kind="student_t"))

    assert result.covariance_mode == "diagonal"
    assert len(result.pointwise_log_likelihood) == 2
    assert np.isfinite(result.log_likelihood)


def test_robust_likelihood_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=2.0)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=101.0)
    with pytest.raises(ValueError, match="positive"):
        LikelihoodInputs(observed=(0.0,), predicted=(0.0,), standard_deviation=(0.0,))
    with pytest.raises(ValueError, match="either"):
        LikelihoodInputs(observed=(0.0,), predicted=(0.0,), standard_deviation=(1.0,), covariance=np.eye(1))
    with pytest.raises(ValueError, match="requires"):
        evaluate_observation_likelihood(LikelihoodInputs(observed=(0.0,), predicted=(0.0,)))
    with pytest.raises(ValueError, match="positive definite"):
        evaluate_observation_likelihood(
            LikelihoodInputs(observed=(0.0, 0.0), predicted=(0.0, 0.0), covariance=np.zeros((2, 2)))
        )
