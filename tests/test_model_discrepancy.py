import warnings

import numpy as np
import pytest

from meso_uq.noise import (
    LowRankDiscrepancyConfig,
    LowRankDiscrepancyInputs,
    build_low_rank_model_discrepancy_covariance,
    build_polynomial_discrepancy_basis,
    fit_low_rank_discrepancy_coefficients,
)


def test_low_rank_model_discrepancy_disabled_is_exact_zero():
    inputs = LowRankDiscrepancyInputs(
        predictions=(1.0, 2.0, 3.0),
        basis=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        basis_names=("linear", "curvature"),
    )

    result = build_low_rank_model_discrepancy_covariance(
        inputs,
        LowRankDiscrepancyConfig(enabled=False, coefficient_scale=10.0, minimum_variance=1.0),
    )

    assert result.summary["enabled"] is False
    assert result.summary["active"] is False
    assert np.allclose(result.covariance.covariance, np.zeros((3, 3)))
    assert result.standard_deviation == pytest.approx((0.0, 0.0, 0.0))


def test_low_rank_model_discrepancy_builds_basis_covariance_and_cross_terms():
    basis = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    coefficient_covariance = np.asarray(((0.25, 0.05), (0.05, 0.09)))
    inputs = LowRankDiscrepancyInputs(
        predictions=(1.0, 2.0, 3.0),
        basis=tuple(tuple(float(value) for value in row) for row in basis),
        basis_names=("linear", "curvature"),
        curve_id="fixture",
    )

    result = build_low_rank_model_discrepancy_covariance(
        inputs,
        LowRankDiscrepancyConfig(enabled=True, coefficient_covariance=tuple(tuple(row) for row in coefficient_covariance), minimum_variance=0.01),
    )

    expected = basis @ coefficient_covariance @ basis.T + 0.01 * np.eye(3)
    reconstructed = sum(
        result.covariance_components[name]
        for name in result.summary["component_names"]
        if name != "model_discrepancy_total"
    )
    assert np.allclose(result.covariance.covariance, expected)
    assert np.allclose(reconstructed, expected)
    assert "model_discrepancy_cross:linear:curvature" in result.covariance_components
    assert result.coefficient_prior_variance == pytest.approx((0.25, 0.09))
    assert result.summary["coefficient_covariance_rank"] == 2


def test_polynomial_basis_and_synthetic_recovery_do_not_move_protected_offset():
    grid = np.linspace(-1.0, 1.0, 21)
    basis = np.asarray(build_polynomial_discrepancy_basis(grid, degree=2, include_intercept=False))
    true_coefficients = np.asarray((0.4, -0.25))
    residuals = basis @ true_coefficients

    fit = fit_low_rank_discrepancy_coefficients(
        tuple(float(value) for value in residuals),
        tuple(tuple(float(value) for value in row) for row in basis),
        regularization=1e-12,
        protected_sensitivities={"offset": tuple(1.0 for _ in grid)},
    )

    assert fit.coefficients == pytest.approx(tuple(true_coefficients), abs=1e-10)
    assert fit.residual_rmse < 1e-11
    assert abs(fit.protected_parameter_drift["offset"]) < 1e-12


def test_low_rank_model_discrepancy_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="positive coefficient_scale"):
        LowRankDiscrepancyConfig(enabled=True, coefficient_scale=0.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        LowRankDiscrepancyConfig(enabled=True, coefficient_covariance=((1.0, 2.0), (2.0, 1.0)))
    with pytest.raises(ValueError, match="predictions values must be finite numeric scalars"):
        LowRankDiscrepancyInputs(predictions=((1.0, 2.0),), basis=((1.0,),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="basis row count"):
        LowRankDiscrepancyInputs(predictions=(1.0, 2.0), basis=((1.0,), (2.0,), (3.0,)))
    with pytest.raises(ValueError, match="coefficient_covariance rank"):
        build_low_rank_model_discrepancy_covariance(
            LowRankDiscrepancyInputs(predictions=(1.0, 2.0), basis=((1.0,), (2.0,))),
            LowRankDiscrepancyConfig(enabled=True, coefficient_covariance=((1.0, 0.0), (0.0, 1.0))),
        )
    with pytest.raises(ValueError, match="model_discrepancy:basis_0 values must be finite"):
        build_low_rank_model_discrepancy_covariance(
            LowRankDiscrepancyInputs(predictions=(1.0,), basis=((1.0e200,),)),
            LowRankDiscrepancyConfig(enabled=True, coefficient_scale=1.0),
        )


def test_low_rank_model_discrepancy_rejects_sum_overflow_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="model_discrepancy_total values must be finite"):
            build_low_rank_model_discrepancy_covariance(
                LowRankDiscrepancyInputs(predictions=(1.0,), basis=((1.0e154, 1.0e154),)),
                LowRankDiscrepancyConfig(enabled=True, coefficient_covariance=((1.0, 0.0), (0.0, 1.0))),
            )

    assert caught == []
