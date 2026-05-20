import math

import numpy as np
import pytest

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    LikelihoodInputs,
    RobustLikelihoodConfig,
    SurrogateCovarianceConfig,
    SurrogateCovarianceInputs,
    build_additive_relative_observation_noise,
    build_surrogate_covariance,
    compose_total_covariance,
    evaluate_observation_likelihood,
    legacy_compression_surrogate_likelihood,
)


def test_surrogate_covariance_diagonal_std_is_named_and_summarized():
    result = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=(2.0, 4.0),
            predictive_standard_deviation=(0.3, 0.4),
            curve_grid=(0.0, 1.0),
            curve_id="emb/compression",
        ),
        SurrogateCovarianceConfig(kind="diagonal"),
    )

    assert result.summary["enabled"] is True
    assert result.summary["active"] is True
    assert result.summary["kind"] == "diagonal"
    assert result.summary["grid_count"] == 2
    assert result.variance_components["surrogate_predictive_diagonal"] == pytest.approx((0.09, 0.16))
    assert result.variance_components["surrogate_covariance_total"] == pytest.approx((0.09, 0.16))
    assert np.allclose(result.covariance.covariance, np.diag([0.09, 0.16]))


def test_surrogate_covariance_disabled_is_zero_even_with_payload():
    result = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=(2.0, 4.0),
            predictive_standard_deviation=(0.3, 0.4),
            predictive_covariance=((1.0, 0.2), (0.2, 1.0)),
        ),
        SurrogateCovarianceConfig(enabled=False, kind="full"),
    )

    assert result.summary["enabled"] is False
    assert result.summary["active"] is False
    assert np.allclose(result.covariance.covariance, np.zeros((2, 2)))
    assert result.standard_deviation == pytest.approx((0.0, 0.0))


def test_surrogate_covariance_full_covariance_is_accepted_when_psd():
    result = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=(1.0, 2.0, 3.0),
            predictive_covariance=((0.09, 0.02, 0.01), (0.02, 0.16, 0.03), (0.01, 0.03, 0.25)),
        ),
        SurrogateCovarianceConfig(kind="full"),
    )

    expected = np.asarray(((0.09, 0.02, 0.01), (0.02, 0.16, 0.03), (0.01, 0.03, 0.25)))
    assert np.allclose(result.covariance_components["surrogate_predictive_full"], expected)
    assert result.standard_deviation == pytest.approx((0.3, 0.4, 0.5))
    assert result.summary["min_eigenvalue"] > 0.0


def test_surrogate_covariance_low_rank_can_include_diagonal_residual():
    result = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=(1.0, 2.0, 3.0),
            low_rank_factors=((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)),
            predictive_standard_deviation=(0.1, 0.2, 0.3),
        ),
        SurrogateCovarianceConfig(kind="low_rank"),
    )

    factors = np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)))
    expected = factors @ factors.T + np.diag([0.01, 0.04, 0.09])
    assert np.allclose(result.covariance.covariance, expected)
    assert result.summary["kind"] == "low_rank"
    assert result.summary["matrix_rank"] == 3
    assert result.summary["low_rank_factor_rank"] == 2
    assert set(result.covariance_components) == {
        "surrogate_predictive_low_rank",
        "surrogate_predictive_diagonal",
        "surrogate_covariance_total",
    }


def test_surrogate_covariance_matches_legacy_bnn_predictive_std_quadrature():
    predictions = (2.0, 4.0)
    sigma = 0.5
    surrogate_std = (0.3, 0.4)
    legacy = legacy_compression_surrogate_likelihood(
        predictions,
        sigma,
        surrogate_standard_deviation=surrogate_std,
    )
    observation = build_additive_relative_observation_noise(
        predictions,
        AdditiveRelativeObservationNoiseConfig(relative_sigma=sigma),
    )
    surrogate = build_surrogate_covariance(
        SurrogateCovarianceInputs(predictions=predictions, predictive_standard_deviation=surrogate_std),
        SurrogateCovarianceConfig(kind="diagonal"),
    )
    total = compose_total_covariance(observation.total_variance, surrogate.covariance)

    assert tuple(math.sqrt(value) for value in np.diag(total.covariance)) == pytest.approx(legacy.standard_deviation)


def test_surrogate_covariance_composes_to_finite_likelihood():
    predictions = (2.0, 4.0)
    observation = build_additive_relative_observation_noise(
        predictions,
        AdditiveRelativeObservationNoiseConfig(additive_sigma=0.1, relative_sigma=0.05),
    )
    surrogate = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=predictions,
            predictive_covariance=((0.09, 0.015), (0.015, 0.16)),
        ),
        SurrogateCovarianceConfig(kind="full"),
    )
    total = compose_total_covariance(observation.total_variance, surrogate.covariance)
    likelihood = evaluate_observation_likelihood(
        LikelihoodInputs(observed=(2.1, 3.9), predicted=predictions, covariance=total.covariance),
        RobustLikelihoodConfig(),
    )

    assert likelihood.covariance_mode == "full"
    assert math.isfinite(likelihood.log_likelihood)



def test_surrogate_covariance_rejects_derived_nonfinite_covariance():
    with pytest.raises(ValueError, match="surrogate_predictive_diagonal values must be finite"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0,), predictive_standard_deviation=(1.0e200,)),
            SurrogateCovarianceConfig(kind="diagonal"),
        )
    with pytest.raises(ValueError, match="surrogate_predictive_low_rank values must be finite"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0,), low_rank_factors=((1.0e200,),)),
            SurrogateCovarianceConfig(kind="low_rank"),
        )


def test_surrogate_covariance_rejects_batched_predictions_explicitly():
    with pytest.raises(ValueError, match="predictions values must be finite numeric scalars"):
        SurrogateCovarianceInputs(predictions=((1.0, 2.0), (3.0, 4.0)))  # type: ignore[arg-type]

def test_surrogate_covariance_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="requires predictive_standard_deviation"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0, 2.0)),
            SurrogateCovarianceConfig(kind="diagonal"),
        )
    with pytest.raises(ValueError, match="does not match prediction count"):
        SurrogateCovarianceInputs(predictions=(1.0, 2.0), predictive_standard_deviation=(0.1,))
    with pytest.raises(ValueError, match="curve_grid count"):
        SurrogateCovarianceInputs(predictions=(1.0, 2.0), curve_grid=(0.0,))
    with pytest.raises(ValueError, match="symmetric"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0, 2.0), predictive_covariance=((1.0, 0.2), (0.0, 1.0))),
            SurrogateCovarianceConfig(kind="full"),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0, 2.0), predictive_covariance=((1.0, 2.0), (2.0, 1.0))),
            SurrogateCovarianceConfig(kind="full"),
        )
    with pytest.raises(ValueError, match="row count"):
        build_surrogate_covariance(
            SurrogateCovarianceInputs(predictions=(1.0, 2.0), low_rank_factors=((1.0,), (2.0,), (3.0,))),
            SurrogateCovarianceConfig(kind="low_rank"),
        )
    with pytest.raises(ValueError, match="Unsupported surrogate covariance kind"):
        SurrogateCovarianceConfig(kind="unknown")
