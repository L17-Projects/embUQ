import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from meso_uq.noise import SyntheticRecoveryInputs, SyntheticRecoveryThresholds, evaluate_synthetic_recovery


def _tuple_vector(values):
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _tuple_matrix(values):
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _inputs(*, observation_shift=0.0):
    x = np.linspace(-1.0, 1.0, 16)
    design = np.column_stack((np.ones_like(x), x))
    truth = np.asarray((1.0, 0.25))
    covariance = 0.01 * np.eye(x.size)
    observations = design @ truth + observation_shift
    return SyntheticRecoveryInputs(
        scenario_id="fixture",
        seed=123,
        design_matrix=_tuple_matrix(design),
        observations=_tuple_vector(observations),
        true_parameters=_tuple_vector(truth),
        parameter_names=("offset", "slope"),
        total_covariance=_tuple_matrix(covariance),
        covariance_components={"observation:additive_relative": _tuple_matrix(covariance)},
        expected_observables=_tuple_vector(design @ truth),
    )


def test_synthetic_recovery_recovers_exact_linear_fixture():
    result = evaluate_synthetic_recovery(_inputs())

    assert result.gate_status == "pass"
    assert result.parameter_estimates["offset"] == pytest.approx(1.0)
    assert result.parameter_estimates["slope"] == pytest.approx(0.25)
    assert result.metrics["finite_observables"] is True
    assert result.metrics["parameter_interval_coverage"] == pytest.approx(1.0)
    assert result.covariance_group_trace_shares["observation"] == pytest.approx(1.0)


def test_synthetic_recovery_thresholds_fail_biased_fixture():
    result = evaluate_synthetic_recovery(
        _inputs(observation_shift=0.3),
        SyntheticRecoveryThresholds(max_abs_parameter_bias=0.05, max_parameter_z_error=10.0),
    )

    assert result.gate_status == "fail"
    assert any("parameter bias" in failure for failure in result.failures)


def test_synthetic_recovery_accepts_signed_child_covariance_components():
    signed_cross = ((0.0, 0.2), (0.2, 0.0))

    inputs = SyntheticRecoveryInputs(
        scenario_id="signed_components",
        seed=1,
        design_matrix=((1.0,), (2.0,)),
        observations=(1.0, 2.0),
        true_parameters=(1.0,),
        parameter_names=("slope",),
        total_covariance=((1.0, 0.0), (0.0, 1.0)),
        covariance_components={
            "observation:additive": ((1.0, 0.0), (0.0, 1.0)),
            "model_discrepancy_cross:signed": signed_cross,
        },
    )

    assert inputs.covariance_components["model_discrepancy_cross:signed"] == signed_cross


def test_synthetic_recovery_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="row count"):
        SyntheticRecoveryInputs(
            scenario_id="bad",
            seed=1,
            design_matrix=((1.0,),),
            observations=(1.0, 2.0),
            true_parameters=(1.0,),
            parameter_names=("offset",),
            total_covariance=((1.0, 0.0), (0.0, 1.0)),
            covariance_components={"observation:additive_relative": ((1.0, 0.0), (0.0, 1.0))},
        )
    with pytest.raises(ValueError, match="positive definite"):
        SyntheticRecoveryInputs(
            scenario_id="bad",
            seed=1,
            design_matrix=((1.0,), (1.0,)),
            observations=(1.0, 2.0),
            true_parameters=(1.0,),
            parameter_names=("offset",),
            total_covariance=((1.0, 1.0), (1.0, 1.0)),
            covariance_components={"observation:additive_relative": ((1.0, 0.0), (0.0, 1.0))},
        )
    with pytest.raises(ValueError, match="min_interval_coverage"):
        SyntheticRecoveryThresholds(min_interval_coverage=1.2)


def test_synthetic_recovery_diagnostics_script_writes_required_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "m6"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_m6_synthetic_recovery_diagnostics.py"),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo_root,
    )

    manifest = json.loads((output_root / "synthetic_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "synthetic_recovery_metrics.json").read_text(encoding="utf-8"))
    assert manifest["required_scenarios"] == [
        "legacy_noise_only",
        "measurement_uncertainty",
        "surrogate_covariance",
        "discrepancy_enabled",
    ]
    assert metrics["all_scenarios_passed"] is True
    for scenario_id in manifest["required_scenarios"]:
        artifacts = manifest["scenario_artifacts"][scenario_id]
        for path in artifacts.values():
            assert Path(path).exists()
        scenario_config = json.loads(Path(artifacts["synthetic_config"]).read_text(encoding="utf-8"))
        assert scenario_config["seed"] > 0
        assert "streams" in scenario_config
    for key in (
        "recovery_parameter_intervals",
        "synthetic_observable_overlay",
        "residual_whitened_hist",
        "covariance_heatmap",
    ):
        path = output_root / manifest["artifacts"][key]
        assert path.exists()
        assert path.stat().st_size > 0
