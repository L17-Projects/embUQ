from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.noise import (
    CompositeLikelihoodSpec,
    build_composite_likelihood,
    DiscrepancyConfig,
    LikelihoodComponent,
    LikelihoodStage,
    MeasurementErrorConfig,
    PosteriorUncertaintyConfig,
    PosteriorUncertaintyKind,
    SurrogateErrorConfig,
    build_model_config,
    compose_toy_likelihood,
    get_model_support,
    legacy_compression_direct_likelihood,
    legacy_compression_surrogate_likelihood,
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
