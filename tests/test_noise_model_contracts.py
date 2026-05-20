from __future__ import annotations

import math
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    CompositeLikelihoodSpec,
    build_composite_likelihood,
    ContactAlignmentInputs,
    ContactAlignmentUncertaintyConfig,
    CorrelatedCurveNoiseConfig,
    CovarianceTerm,
    CurveGrid,
    DiscrepancyConfig,
    GeometryParameterUncertainty,
    GeometrySensitivityInputs,
    GeometryUncertaintyConfig,
    LikelihoodInputs,
    LikelihoodComponent,
    LikelihoodStage,
    LowRankDiscrepancyConfig,
    LowRankDiscrepancyInputs,
    MeasurementErrorConfig,
    PosteriorUncertaintyConfig,
    PosteriorUncertaintyKind,
    RobustLikelihoodConfig,
    SurrogateCovarianceConfig,
    SurrogateCovarianceInputs,
    SurrogateErrorConfig,
    TotalCovarianceConfig,
    assemble_total_covariance,
    build_additive_relative_observation_noise,
    build_contact_alignment_covariance,
    build_correlated_curve_covariance,
    build_geometry_uncertainty_covariance,
    build_low_rank_model_discrepancy_covariance,
    build_model_config,
    build_surrogate_covariance,
    compose_toy_likelihood,
    compose_total_covariance,
    covariance_term_from_diagonal,
    evaluate_observation_likelihood,
    get_model_support,
    legacy_compression_direct_likelihood,
    legacy_compression_surrogate_batch_likelihood,
    legacy_compression_surrogate_likelihood,
    legacy_indentation_adjusted_batch_likelihood,
    legacy_indentation_direct_standard_deviation,
    legacy_indentation_surrogate_likelihood,
    legacy_multiplicative_likelihood,
    list_model_ids,
    list_model_support_metadata,
)


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_noise_model_registry_resolution():
    assert set(list_model_ids()) == {"emb/compression", "emb/indentation", "gv/stretching"}
    assert len(list_model_support_metadata()) == 3

    emb = get_model_support("emb/compression")
    indentation = get_model_support("emb/indentation")
    gv = get_model_support("gv/stretching")

    assert emb.model_id == "emb/compression"
    assert indentation.unit_for_observable("displacement") == "micrometer"
    assert gv.unit_for_observable("extension") == "micrometer"
    assert gv.supports_observable("force")


def test_noise_model_invalid_model_and_observable():
    with pytest.raises(ValueError, match="Unknown noise model 'unknown'"):
        get_model_support("unknown")

    with pytest.raises(ValueError, match="does not support observable 'temperature'"):
        build_model_config(
            "emb/compression",
            observable="temperature",
            unit="micro_newton",
            measurement_error=MeasurementErrorConfig(kind="absolute_gaussian", sigma=0.4),
        )

    with pytest.raises(ValueError, match="Unit mismatch"):
        build_model_config(
            "emb/compression",
            observable="force",
            unit="pascal",
            measurement_error=MeasurementErrorConfig(kind="absolute_gaussian", sigma=0.4),
        )

    with pytest.raises(ValueError, match="Measurement error 'relative_gaussian' is not supported"):
        build_model_config(
            "gv/stretching",
            observable="force",
            unit="nano_newton",
            measurement_error=MeasurementErrorConfig(kind="relative_gaussian", sigma=0.4),
        )


def test_required_measurement_error_is_validated():
    with pytest.raises(
        ValueError,
        match="requires a measurement error choice",
    ):
        build_model_config(
            "gv/stretching",
            observable="extension",
            unit="micrometer",
        )


def test_model_discrepancy_support_is_distinct_and_composable():
    config = build_model_config(
        "gv/stretching",
        observable="force",
        unit="nano_newton",
        measurement_error=MeasurementErrorConfig(kind="absolute_gaussian", sigma=0.2),
        discrepancy=DiscrepancyConfig(kind="additive_gaussian", sigma=0.4),
    )

    composed = compose_toy_likelihood(
        residuals=(1.0, -2.0),
        config=config,
    )

    assert composed.measurement_variance == pytest.approx((0.04, 0.04))
    assert composed.discrepancy_variance == pytest.approx((0.16, 0.16))
    assert composed.surrogate_variance == pytest.approx((0.0, 0.0))
    assert composed.posterior_variance == pytest.approx((0.0, 0.0))
    assert composed.total_variance == pytest.approx((0.2, 0.2))
    expected = -0.5 * (
        1.0 / 0.2
        + math.log(2.0 * math.pi * 0.2)
        + 4.0 / 0.2
        + math.log(2.0 * math.pi * 0.2)
    )
    assert math.isclose(composed.log_likelihood, expected, rel_tol=1e-12)


def test_deterministic_surrogate_outputs_do_not_contribute_uncertainty():
    config = build_model_config(
        "emb/compression",
        observable="force",
        unit="micro_newton",
        measurement_error=MeasurementErrorConfig(kind="absolute_gaussian", sigma=0.1),
        surrogate_error=SurrogateErrorConfig(kind="deterministic"),
        posterior_uncertainty=PosteriorUncertaintyConfig(kind=PosteriorUncertaintyKind.NONE),
    )

    composed = compose_toy_likelihood(
        residuals=(0.2, -0.2),
        config=config,
    )

    assert composed.surrogate_variance == (0.0, 0.0)
    assert composed.posterior_variance == (0.0, 0.0)
    assert composed.total_variance == composed.measurement_variance


def test_posterior_none_never_contributes_uncertainty_even_with_sigma_value():
    config = build_model_config(
        "emb/compression",
        observable="force",
        unit="micro_newton",
        measurement_error=MeasurementErrorConfig(kind="absolute_gaussian", sigma=0.1),
        posterior_uncertainty=PosteriorUncertaintyConfig(kind=PosteriorUncertaintyKind.NONE, sigma=2.0),
    )

    composed = compose_toy_likelihood(
        residuals=(0.2, -0.2),
        config=config,
    )

    assert composed.posterior_variance == (0.0, 0.0)
    assert composed.total_variance == composed.measurement_variance


def test_legacy_batch_likelihood_allows_empty_batch_edges():
    empty = legacy_compression_surrogate_batch_likelihood([], 0.1)
    assert empty.reference_evaluations == ()
    assert empty.standard_deviation == ()
    sample: dict[str, object] = {}
    empty.assign_to_sample(sample)
    assert sample["Batch Reference Evaluations"] == []
    assert sample["Batch Standard Deviation"] == []

    zero_width = legacy_indentation_adjusted_batch_likelihood([[], []], [0.1, 0.2])
    assert zero_width.reference_evaluations == ((), ())
    assert zero_width.standard_deviation == ((), ())


def test_legacy_likelihood_wrapper_preserves_emb_and_gv_golden_values():
    compression = legacy_compression_surrogate_likelihood([2.0, 4.0], 0.5)
    assert compression.reference_evaluations == (2.0, 4.0)
    assert compression.standard_deviation == pytest.approx((1.0, 2.0))

    compression_bnn = legacy_compression_surrogate_likelihood(
        [2.0, 4.0],
        0.5,
        surrogate_standard_deviation=[0.3, 0.4],
    )
    assert compression_bnn.standard_deviation == pytest.approx(
        (math.sqrt(0.3 * 0.3 + 1.0 * 1.0), math.sqrt(0.4 * 0.4 + 2.0 * 2.0))
    )

    indentation = legacy_indentation_surrogate_likelihood([1.0, 3.0], 0.1, d0=-2.0)
    assert indentation.reference_evaluations == (0.0, 1.0)
    assert indentation.standard_deviation == pytest.approx((0.0, 0.1))

    compression_direct = legacy_compression_direct_likelihood([3.0, 5.0], 0.7)
    assert compression_direct.standard_deviation == pytest.approx((0.7, 0.7))
    assert legacy_indentation_direct_standard_deviation(5.0, 0.2) == pytest.approx(1.0)

    gv = legacy_multiplicative_likelihood([-2.0, 4.0], 0.25, absolute_reference=True)
    assert gv.standard_deviation == pytest.approx((0.5, 1.0))

    with pytest.raises(ValueError, match="legacy multiplicative sigma must be positive"):
        legacy_multiplicative_likelihood([1.0], 0.0)
    with pytest.raises(ValueError, match="standard deviations must be positive"):
        legacy_multiplicative_likelihood([0.0], 0.25)


def test_composite_likelihood_spec_switches_components_and_recovers_legacy():
    legacy = CompositeLikelihoodSpec.from_mapping(
        {"stage": "M0", "legacy_mode": "emb/compression", "components": ["legacy"]}
    )
    assert legacy.stage is LikelihoodStage.M0
    assert legacy.components == (LikelihoodComponent.LEGACY,)
    assert legacy.legacy_mode == "emb/compression"
    assert legacy.is_legacy

    primitives = CompositeLikelihoodSpec.from_mapping(
        {"stage": "M2", "components": ["additive_noise", "relative_noise"]}
    )
    assert primitives.components == (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
    )
    assert not primitives.is_legacy

    with pytest.raises(ValueError, match="Legacy component requires legacy_mode"):
        CompositeLikelihoodSpec.from_mapping({"stage": "M0"})

    with pytest.raises(ValueError, match="is not available for stage M1"):
        CompositeLikelihoodSpec.from_mapping(
            {"stage": "M1", "legacy_mode": "emb/compression", "components": ["model_discrepancy"]}
        )


def test_composite_likelihood_dispatches_configured_components():
    likelihood = build_composite_likelihood(
        {"stage": "M0", "legacy_mode": "emb/compression", "components": ["legacy"]},
        {"legacy": lambda payload: legacy_compression_surrogate_likelihood(payload["predictions"], payload["sigma"])},
    )

    result = likelihood.evaluate({"predictions": [2.0, 4.0], "sigma": 0.5})

    assert result["legacy"].standard_deviation == pytest.approx((1.0, 2.0))

    with pytest.raises(ValueError, match="no evaluator"):
        build_composite_likelihood(
            {"stage": "M2", "components": ["additive_noise", "relative_noise"]},
            {"additive_noise": lambda payload: payload},
        )


def test_m2_composite_likelihood_dispatches_executable_noise_primitives():
    def _observation(payload, *, additive_sigma: float, relative_sigma: float):
        return build_additive_relative_observation_noise(
            payload["predictions"],
            AdditiveRelativeObservationNoiseConfig(
                additive_sigma=additive_sigma,
                relative_sigma=relative_sigma,
            ),
        )

    def _correlated(payload):
        return build_correlated_curve_covariance(
            CurveGrid(tuple(payload["grid"]), "fixture"),
            CorrelatedCurveNoiseConfig(
                amplitude=payload["correlated_amplitude"],
                length_scale=1.0,
            ),
        )

    def _heavy_tail(payload):
        observation = _observation(
            payload,
            additive_sigma=payload["additive_sigma"],
            relative_sigma=payload["relative_sigma"],
        )
        total = compose_total_covariance(observation.total_variance, _correlated(payload))
        return evaluate_observation_likelihood(
            LikelihoodInputs(
                observed=tuple(payload["observations"]),
                predicted=observation.predictions,
                covariance=total.covariance,
                variance_components=observation.variance_components,
            ),
            RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=4.0),
        )

    likelihood = build_composite_likelihood(
        {
            "stage": "M2",
            "components": [
                "additive_noise",
                "relative_noise",
                "correlated_curve_noise",
                "heavy_tail",
            ],
        },
        {
            "additive_noise": lambda payload: _observation(
                payload,
                additive_sigma=payload["additive_sigma"],
                relative_sigma=0.0,
            ).variance_components["additive"],
            "relative_noise": lambda payload: _observation(
                payload,
                additive_sigma=0.0,
                relative_sigma=payload["relative_sigma"],
            ).variance_components["relative"],
            "correlated_curve_noise": _correlated,
            "heavy_tail": _heavy_tail,
        },
    )

    result = likelihood.evaluate(
        {
            "observations": [2.1, 3.9],
            "predictions": [2.0, 4.0],
            "grid": [0.0, 1.0],
            "additive_sigma": 0.1,
            "relative_sigma": 0.25,
            "correlated_amplitude": 0.05,
        }
    )

    assert result["additive_noise"] == pytest.approx((0.01, 0.01))
    assert result["relative_noise"] == pytest.approx((0.25, 1.0))
    assert result["correlated_curve_noise"].covariance.shape == (2, 2)
    assert result["heavy_tail"].covariance_mode == "full"
    assert math.isfinite(result["heavy_tail"].log_likelihood)


def test_m3_composite_likelihood_dispatches_measurement_uncertainty_components():
    def _contact_alignment(payload):
        return build_contact_alignment_covariance(
            ContactAlignmentInputs(
                controls=tuple(payload["controls"]),
                predictions=tuple(payload["predictions"]),
                curve_id="emb/indentation",
            ),
            ContactAlignmentUncertaintyConfig(
                contact_offset_sigma=0.1,
                alignment_slope_sigma=0.2,
                minimum_variance=0.005,
            ),
        )

    def _geometry(payload):
        return build_geometry_uncertainty_covariance(
            GeometrySensitivityInputs(
                predictions=tuple(payload["predictions"]),
                sensitivities={
                    "radius": tuple(payload["geometry_sensitivities"]["radius"]),
                    "height": tuple(payload["geometry_sensitivities"]["height"]),
                },
                curve_id="gv/stretching",
            ),
            GeometryUncertaintyConfig(
                parameters=(
                    GeometryParameterUncertainty("radius", sigma=0.1, units="micrometer", nominal=2.0),
                    GeometryParameterUncertainty("height", sigma=0.2, units="micrometer", nominal=14.28),
                ),
            ),
        )

    likelihood = build_composite_likelihood(
        {"stage": "M3", "components": ["contact_alignment", "geometry"]},
        {
            "contact_alignment": _contact_alignment,
            "geometry": _geometry,
        },
    )

    result = likelihood.evaluate(
        {
            "controls": [0.0, 1.0],
            "predictions": [2.0, 4.0],
            "geometry_sensitivities": {
                "radius": [1.0, 2.0],
                "height": [0.5, 1.0],
            },
        }
    )

    contact = result["contact_alignment"]
    geometry = result["geometry"]
    assert contact.summary["active"] is True
    assert geometry.summary["active"] is True
    assert contact.variance_components["contact_alignment_total"] == pytest.approx((0.025, 0.025))
    assert geometry.variance_components["geometry_jacobian"] == pytest.approx((0.02, 0.08))
    assert contact.covariance.covariance.shape == (2, 2)
    assert geometry.covariance.covariance.shape == (2, 2)


def test_m4_default_composite_spec_includes_measurement_and_surrogate_components():
    spec = CompositeLikelihoodSpec.from_mapping({"stage": "M4"})

    assert spec.components == (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CONTACT_ALIGNMENT,
        LikelihoodComponent.GEOMETRY,
        LikelihoodComponent.SURROGATE_COVARIANCE,
    )


def test_m4_composite_likelihood_dispatches_surrogate_covariance_component():
    likelihood = build_composite_likelihood(
        {"stage": "M4", "components": ["surrogate_covariance"]},
        {
            "surrogate_covariance": lambda payload: build_surrogate_covariance(
                SurrogateCovarianceInputs(
                    predictions=tuple(payload["predictions"]),
                    predictive_standard_deviation=tuple(payload["surrogate_std"]),
                    curve_grid=tuple(payload["grid"]),
                    curve_id="emb/compression",
                ),
                SurrogateCovarianceConfig(kind="diagonal"),
            ),
        },
    )

    result = likelihood.evaluate(
        {
            "predictions": [2.0, 4.0],
            "surrogate_std": [0.3, 0.4],
            "grid": [0.0, 1.0],
        }
    )

    surrogate = result["surrogate_covariance"]
    assert surrogate.summary["kind"] == "diagonal"
    assert surrogate.variance_components["surrogate_covariance_total"] == pytest.approx((0.09, 0.16))



def test_m5_default_composite_spec_includes_full_hierarchy_components():
    spec = CompositeLikelihoodSpec.from_mapping({"stage": "M5"})

    assert spec.components == (
        LikelihoodComponent.ADDITIVE_NOISE,
        LikelihoodComponent.RELATIVE_NOISE,
        LikelihoodComponent.CONTACT_ALIGNMENT,
        LikelihoodComponent.GEOMETRY,
        LikelihoodComponent.SURROGATE_COVARIANCE,
        LikelihoodComponent.MODEL_DISCREPANCY,
        LikelihoodComponent.TOTAL_COVARIANCE,
    )


def test_m5_composite_likelihood_dispatches_discrepancy_and_shared_total_assembler():
    def _discrepancy(payload):
        return build_low_rank_model_discrepancy_covariance(
            LowRankDiscrepancyInputs(
                predictions=tuple(payload["predictions"]),
                basis=tuple(tuple(row) for row in payload["basis"]),
                basis_names=("linear",),
                curve_id="m5/contract",
            ),
            LowRankDiscrepancyConfig(enabled=True, coefficient_scale=0.2, shrinkage_strength=1.0),
        )

    def _total(payload):
        discrepancy = _discrepancy(payload)
        return assemble_total_covariance(
            (
                covariance_term_from_diagonal(
                    "observation:additive_relative",
                    tuple(payload["observation_variance"]),
                ),
                CovarianceTerm(
                    "discrepancy:low_rank",
                    discrepancy.covariance.covariance,
                    children={
                        name: matrix
                        for name, matrix in discrepancy.covariance_components.items()
                        if name != "model_discrepancy_total"
                    },
                ),
            ),
            TotalCovarianceConfig(jitter=1e-12, max_jitter=1e-8),
        )

    likelihood = build_composite_likelihood(
        {"stage": "M5", "components": ["model_discrepancy", "total_covariance"]},
        {
            "model_discrepancy": _discrepancy,
            "total_covariance": _total,
        },
    )

    result = likelihood.evaluate(
        {
            "predictions": [1.0, 2.0],
            "basis": [[1.0], [0.5]],
            "observation_variance": [0.1, 0.2],
        }
    )

    discrepancy = result["model_discrepancy"]
    total = result["total_covariance"]
    expected = np.diag([0.1, 0.2]) + discrepancy.covariance.covariance
    assert discrepancy.summary["active"] is True
    assert total.included_term_names == ("observation:additive_relative", "discrepancy:low_rank")
    assert "model_discrepancy:linear" in total.child_covariance_components
    assert np.allclose(total.covariance.covariance, expected)
    assert total.summary["cholesky_success"] is True


def test_supported_observables_and_units_contract():
    support = get_model_support("gv/stretching")
    assert "extension" in support.supported_observables
    assert "force" in support.supported_observables
    assert support.observable_units["extension"] == "micrometer"
    assert support.observable_units["force"] == "nano_newton"


def test_noise_module_imports_without_heavy_optionals():
    code = """
import meso_uq.noise
import sys
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'mirheo', 'korali', 'slurm') if name in sys.modules]
assert loaded == [], loaded
"""
    result = _run_import_probe(code)
    assert result.returncode == 0, result.stderr
