from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.noise import (
    DiscrepancyConfig,
    MeasurementErrorConfig,
    PosteriorUncertaintyConfig,
    PosteriorUncertaintyKind,
    SurrogateErrorConfig,
    compose_toy_likelihood,
    get_model_support,
    build_model_config,
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
    assert set(list_model_ids()) == {"emb/compression", "gv/stretching"}
    assert len(list_model_support_metadata()) == 2

    emb = get_model_support("emb/compression")
    gv = get_model_support("gv/stretching")

    assert emb.model_id == "emb/compression"
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
