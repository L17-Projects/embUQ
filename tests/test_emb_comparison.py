import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from meso_uq.noise import EmbComparisonInputs, EmbComparisonThresholds, evaluate_emb_comparison


def _tuple_vector(values):
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _tuple_matrix(values):
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _inputs(*, upgraded_scale=1.3):
    observed = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=float)
    legacy_predictions = np.asarray((0.92, 2.10, 2.88, 4.12), dtype=float)
    upgraded_predictions = np.asarray((0.97, 2.03, 2.97, 4.02), dtype=float)
    legacy_sd = np.asarray((0.16, 0.18, 0.20, 0.22), dtype=float)
    upgraded_covariance = np.diag((upgraded_scale * legacy_sd) ** 2)
    rng = np.random.default_rng(371)
    legacy_samples = rng.normal((1.0, 0.8), (0.10, 0.08), size=(64, 2))
    upgraded_samples = rng.normal((1.03, 0.79), (0.14, 0.11), size=(64, 2))
    return EmbComparisonInputs(
        scenario_id="fixture",
        modality="emb_compression",
        dataset_path="emb/compression/evalkit/data/data_1.csv",
        config_id="fixture_config",
        axis_name="deformation_nm",
        observable_name="force_nN",
        axis_values=(0.0, 1.0, 2.0, 3.0),
        observations=_tuple_vector(observed),
        legacy_predictions=_tuple_vector(legacy_predictions),
        upgraded_predictions=_tuple_vector(upgraded_predictions),
        legacy_standard_deviation=_tuple_vector(legacy_sd),
        upgraded_covariance=_tuple_matrix(upgraded_covariance),
        covariance_components={"observation:additive_relative": _tuple_matrix(upgraded_covariance)},
        parameter_names=("elastic_scale", "bending_scale"),
        legacy_posterior_samples=_tuple_matrix(legacy_samples),
        upgraded_posterior_samples=_tuple_matrix(upgraded_samples),
    )


def test_emb_comparison_passes_paired_fixture():
    result = evaluate_emb_comparison(_inputs())

    assert result.gate_status == "pass"
    assert result.metrics["legacy_mode_recoverable"] is True
    assert result.metrics["upgraded_mode_recoverable"] is True
    assert result.predictive_metrics["mean_upgraded_to_legacy_std_ratio"] > 1.0
    assert result.posterior_metrics["parameter_count"] == 2
    assert result.interpretation


def test_emb_comparison_fails_when_upgraded_uncertainty_is_narrower():
    result = evaluate_emb_comparison(_inputs(upgraded_scale=0.6), EmbComparisonThresholds(min_upgraded_to_legacy_uncertainty_ratio=1.0))

    assert result.gate_status == "fail"
    assert any("narrower" in failure for failure in result.failures)


def test_emb_comparison_rejects_empty_parameter_names():
    inputs = _inputs()
    with pytest.raises(ValueError, match="parameter_names must contain at least one"):
        EmbComparisonInputs(
            scenario_id=inputs.scenario_id,
            modality=inputs.modality,
            dataset_path=inputs.dataset_path,
            config_id=inputs.config_id,
            axis_name=inputs.axis_name,
            axis_values=inputs.axis_values,
            observable_name=inputs.observable_name,
            observations=inputs.observations,
            legacy_predictions=inputs.legacy_predictions,
            legacy_standard_deviation=inputs.legacy_standard_deviation,
            upgraded_predictions=inputs.upgraded_predictions,
            upgraded_covariance=inputs.upgraded_covariance,
            covariance_components=inputs.covariance_components,
            parameter_names=(),
            legacy_posterior_samples=((), (), ()),
            upgraded_posterior_samples=((), (), ()),
        )


def test_emb_comparison_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="covariance_components"):
        base = _inputs()
        EmbComparisonInputs(
            scenario_id=base.scenario_id,
            modality=base.modality,
            dataset_path=base.dataset_path,
            config_id=base.config_id,
            axis_name=base.axis_name,
            observable_name=base.observable_name,
            axis_values=base.axis_values,
            observations=base.observations,
            legacy_predictions=base.legacy_predictions,
            upgraded_predictions=base.upgraded_predictions,
            legacy_standard_deviation=base.legacy_standard_deviation,
            upgraded_covariance=base.upgraded_covariance,
            covariance_components={},
            parameter_names=base.parameter_names,
            legacy_posterior_samples=base.legacy_posterior_samples,
            upgraded_posterior_samples=base.upgraded_posterior_samples,
        )


def test_emb_comparison_diagnostics_script_writes_required_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "m7"
    subprocess.run(
        [sys.executable, str(repo_root / "scripts/qa/noise_m7_emb_comparison_diagnostics.py"), "--output-root", str(output_root)],
        check=True,
        cwd=repo_root,
    )

    manifest = json.loads((output_root / "emb_comparison_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "emb_comparison_metrics.json").read_text(encoding="utf-8"))
    assert manifest["evidence_class"] == "validation_fixture"
    assert manifest["production_claim"] is False
    assert manifest["required_scenarios"] == ["emb_compression_reference", "emb_indentation_reference"]
    assert metrics["all_scenarios_passed"] is True
    assert set(metrics["scenario_gate_statuses"].values()) == {"pass"}
    assert Path(manifest["artifacts"]["report_md"]).exists()
    for scenario_id in manifest["required_scenarios"]:
        for path in manifest["scenario_artifacts"][scenario_id].values():
            assert Path(path).exists()
    for key in ("posterior_intervals", "predictive_bands", "residual_diagnostics", "metrics_table"):
        path = Path(manifest["artifacts"][key])
        assert path.exists()
        assert path.stat().st_size > 0
