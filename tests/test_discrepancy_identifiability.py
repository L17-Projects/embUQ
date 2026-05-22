import numpy as np
import pytest

from meso_uq.noise import (
    DiscrepancyIdentifiabilityInputs,
    DiscrepancyIdentifiabilityThresholds,
    evaluate_discrepancy_identifiability,
)


def _tuple_vector(values):
    return tuple(float(value) for value in values)


def _tuple_matrix(values):
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _components(point_count, *, discrepancy_variance=1e-5, surrogate_variance=9e-4, observation_variance=1.6e-3):
    observation = observation_variance * np.eye(point_count)
    surrogate = surrogate_variance * np.eye(point_count)
    measurement = 2e-4 * np.eye(point_count)
    discrepancy = discrepancy_variance * np.eye(point_count)
    total = observation + surrogate + measurement + discrepancy
    return {
        "observation:additive_relative": _tuple_matrix(observation),
        "surrogate:predictive": _tuple_matrix(surrogate),
        "measurement:geometry": _tuple_matrix(measurement),
        "discrepancy:low_rank": _tuple_matrix(discrepancy),
    }, _tuple_matrix(total)


def test_identifiable_discrepancy_reports_metrics_and_passes():
    grid = np.linspace(0.0, 1.0, 80)
    predictions = 2.0 + 0.35 * grid
    discrepancy = 0.01 * np.sin(2.0 * np.pi * grid)
    noise = 0.04 * np.cos(4.0 * np.pi * grid)
    observations = predictions + discrepancy + noise
    basis = np.column_stack((np.sin(2.0 * np.pi * grid), np.cos(2.0 * np.pi * grid)))
    components, total = _components(grid.size)

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(observations),
            predictions_without_discrepancy=_tuple_vector(predictions),
            discrepancy_mean=_tuple_vector(discrepancy),
            basis=_tuple_matrix(basis),
            basis_names=("wave_sin", "wave_cos"),
            parameter_names=("offset",),
            parameters_without_discrepancy=(2.0,),
            parameters_with_discrepancy=(2.01,),
            parameter_posterior_sd_without_discrepancy=(0.08,),
            parameter_sensitivities={"offset": _tuple_vector(np.ones_like(grid))},
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.1, 0.1),
            coefficient_posterior_mean=(0.01, 0.0),
            coefficient_posterior_sd=(0.02, 0.02),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            fixture_id="identifiable_residual",
        )
    )

    assert result.gate_status == "pass"
    assert result.failures == ()
    assert result.metrics["discrepancy_rms_over_response"] < 0.01
    assert result.metrics["discrepancy_explained_residual_share"] < 0.30
    assert result.metrics["max_abs_theta_beta_correlation"] == pytest.approx(0.0)
    assert result.covariance_group_trace_shares["observation"] > result.covariance_group_trace_shares["model_discrepancy"]


def test_negative_control_parameter_absorption_fails_identifiability_gate():
    grid = np.linspace(-1.0, 1.0, 64)
    predictions = 1.5 + 0.4 * grid
    discrepancy = 0.12 * grid
    observations = predictions + discrepancy
    components, total = _components(grid.size, discrepancy_variance=2.5e-3)

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(observations),
            predictions_without_discrepancy=_tuple_vector(predictions),
            discrepancy_mean=_tuple_vector(discrepancy),
            basis=_tuple_matrix(grid.reshape(-1, 1)),
            basis_names=("slope_like",),
            parameter_names=("slope",),
            parameters_without_discrepancy=(0.4,),
            parameters_with_discrepancy=(0.4,),
            parameter_posterior_sd_without_discrepancy=(0.05,),
            parameter_sensitivities={"slope": _tuple_vector(grid)},
            coefficient_names=("slope_like",),
            coefficient_prior_sd=(0.2,),
            coefficient_posterior_mean=(0.12,),
            coefficient_posterior_sd=(0.01,),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            negative_control=True,
            fixture_id="parameter_absorption",
        )
    )

    assert result.gate_status == "fail"
    assert result.metrics["max_abs_theta_beta_correlation"] > 0.99
    assert any("correlation" in failure for failure in result.failures)
    assert result.metrics["discrepancy_explained_residual_share"] == pytest.approx(1.0)


def test_null_fixture_passes_only_when_discrepancy_shrinks_to_zero():
    grid = np.linspace(0.0, 1.0, 50)
    predictions = 1.0 + 0.2 * grid
    observations = predictions + 0.02 * np.sin(6.0 * np.pi * grid)
    basis = np.column_stack((np.sin(2.0 * np.pi * grid), np.cos(2.0 * np.pi * grid)))
    components, total = _components(grid.size, discrepancy_variance=0.0)

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(observations),
            predictions_without_discrepancy=_tuple_vector(predictions),
            discrepancy_mean=_tuple_vector(np.zeros_like(grid)),
            basis=_tuple_matrix(basis),
            basis_names=("wave_sin", "wave_cos"),
            parameter_names=("offset",),
            parameters_without_discrepancy=(1.0,),
            parameters_with_discrepancy=(1.0,),
            parameter_posterior_sd_without_discrepancy=(0.05,),
            parameter_sensitivities={"offset": _tuple_vector(np.ones_like(grid))},
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.1, 0.1),
            coefficient_posterior_mean=(0.0, 0.0),
            coefficient_posterior_sd=(0.005, 0.005),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            expect_discrepancy=False,
            fixture_id="null_control",
        )
    )

    assert result.gate_status == "pass"
    assert result.metrics["active_discrepancy_coefficients"] == 0
    assert result.coefficient_shrinkage["wave_sin"]["posterior_sd_over_prior_sd"] == pytest.approx(0.05)

    active = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(observations),
            predictions_without_discrepancy=_tuple_vector(predictions),
            discrepancy_mean=_tuple_vector(0.08 * np.sin(2.0 * np.pi * grid)),
            basis=_tuple_matrix(basis),
            basis_names=("wave_sin", "wave_cos"),
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.1, 0.1),
            coefficient_posterior_mean=(0.08, 0.0),
            coefficient_posterior_sd=(0.01, 0.01),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            expect_discrepancy=False,
            fixture_id="null_control_active_discrepancy",
        )
    )

    assert active.gate_status == "fail"
    assert any("no-discrepancy fixture" in failure for failure in active.failures)


def test_opt_in_and_disabled_state_are_enforced():
    components, total = _components(3)
    missing_opt_in = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0, 3.0),
            predictions_without_discrepancy=(1.0, 2.0, 3.0),
            discrepancy_mean=(0.0, 0.0, 0.0),
            covariance_components=components,
            total_covariance=total,
        )
    )
    assert missing_opt_in.gate_status == "fail"
    assert any("explicit opt-in" in failure for failure in missing_opt_in.failures)

    disabled_nonzero = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0, 3.0),
            predictions_without_discrepancy=(1.0, 2.0, 3.0),
            discrepancy_mean=(0.1, 0.0, 0.0),
            covariance_components=components,
            total_covariance=total,
            discrepancy_enabled=False,
        )
    )
    assert disabled_nonzero.gate_status == "fail"
    assert any("marked disabled" in failure for failure in disabled_nonzero.failures)


def test_zero_discrepancy_over_zero_total_sigma_is_safe():
    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0, 2.0),
            discrepancy_mean=(0.0, 0.0),
            covariance_components={"discrepancy:low_rank": ((0.0, 0.0), (0.0, 0.0))},
            total_covariance=((0.0, 0.0), (0.0, 0.0)),
            discrepancy_opt_in=True,
        )
    )

    assert result.metrics["discrepancy_abs_over_total_sigma_mean"] == pytest.approx(0.0)
    assert result.metrics["discrepancy_abs_over_total_sigma_max"] == pytest.approx(0.0)
    assert not any("dominates total predictive" in failure for failure in result.failures)


def test_predictions_with_discrepancy_are_not_double_subtracted():
    predictions_without = np.asarray((1.0, 2.0, 3.0), dtype=float)
    discrepancy = np.asarray((0.1, -0.2, 0.05), dtype=float)
    observations = predictions_without + discrepancy
    components, total = _components(3)

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(observations),
            predictions_without_discrepancy=_tuple_vector(predictions_without),
            predictions_with_discrepancy=_tuple_vector(observations),
            discrepancy_mean=_tuple_vector(discrepancy),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
        )
    )

    assert result.metrics["residual_rms_after_discrepancy"] == pytest.approx(0.0)
    assert result.metrics["discrepancy_explained_residual_share"] == pytest.approx(1.0)


def test_missing_covariance_decomposition_is_a_gate_failure():
    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0, 3.0),
            predictions_without_discrepancy=(1.0, 2.0, 3.0),
            discrepancy_mean=(0.0, 0.0, 0.0),
            discrepancy_opt_in=True,
        )
    )

    assert result.gate_status == "fail"
    assert any("covariance component report is missing" in failure for failure in result.failures)


def test_measurement_covariance_group_is_required_by_default():
    components, total = _components(3)
    components = {name: matrix for name, matrix in components.items() if not name.startswith("measurement:")}

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0, 3.0),
            predictions_without_discrepancy=(1.0, 2.0, 3.0),
            discrepancy_mean=(0.0, 0.0, 0.0),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
        )
    )

    assert result.gate_status == "fail"
    assert any("measurement" in failure for failure in result.failures)


def test_rank_fraction_uses_effective_basis_rank_not_column_count():
    basis = ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (4.0, 4.0, 4.0))
    components, total = _components(4)

    result = evaluate_discrepancy_identifiability(
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0, 3.0, 4.0),
            predictions_without_discrepancy=(1.0, 2.0, 3.0, 4.0),
            discrepancy_mean=(0.0, 0.0, 0.0, 0.0),
            basis=basis,
            coefficient_names=("duplicate_0", "duplicate_1", "duplicate_2"),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
        ),
        DiscrepancyIdentifiabilityThresholds(warning_basis_rank_fraction=0.5),
    )

    assert result.metrics["basis_rank"] == 1
    assert result.metrics["basis_columns"] == 3
    assert result.metrics["basis_rank_fraction"] == pytest.approx(0.25)
    assert "discrepancy basis is rank deficient." in result.warnings
    assert not any("rank is large relative" in warning for warning in result.warnings)


def test_identifiability_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="length must match observations"):
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0,),
            discrepancy_mean=(0.0, 0.0),
        )
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0, 2.0),
            discrepancy_mean=(0.0, 0.0),
            basis=((1.0,), (2.0,)),
            theta_beta_correlation=((1.5,),),
        )
    with pytest.raises(ValueError, match="must be symmetric"):
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0, 2.0),
            discrepancy_mean=(0.0, 0.0),
            total_covariance=((1.0, 0.1), (0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="coefficient_names count must match basis column count"):
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0, 2.0),
            discrepancy_mean=(0.0, 0.0),
            basis=((1.0, 0.0), (0.0, 1.0)),
            coefficient_names=("only_one",),
        )
    with pytest.raises(ValueError, match="missing declared parameters"):
        DiscrepancyIdentifiabilityInputs(
            observations=(1.0, 2.0),
            predictions_without_discrepancy=(1.0, 2.0),
            discrepancy_mean=(0.0, 0.0),
            parameter_names=("offset", "slope"),
            parameters_without_discrepancy=(1.0, 0.5),
            parameters_with_discrepancy=(1.0, 0.5),
            parameter_sensitivities={"offset": (1.0, 1.0)},
        )


def test_thresholds_reject_negative_values():
    with pytest.raises(ValueError, match="warning_discrepancy_rms_over_response"):
        DiscrepancyIdentifiabilityThresholds(warning_discrepancy_rms_over_response=-1.0)
