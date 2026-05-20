import json
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.noise import (
    resolve_noise_mode,
    supported_noise_modes,
    validate_noise_config_document,
    validate_noise_config_file,
)


def _write_evidence(root: Path, label: str, *, clean: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / f"{label}_metrics.json"
    manifest_path = root / f"{label}_manifest.json"
    report_path = root / f"{label}_report.md"
    report_path.write_text(f"# {label}\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "all_scenarios_passed": True,
                "scenario_gate_statuses": {"fixture": "pass"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "commands": {"regenerate": f"python {label}.py"},
                "provenance": {
                    "git_commit": "abc123",
                    "git_branch": "feature/noise-release-readiness",
                    "git_status_clean": clean,
                },
                "evidence_class": "validation_fixture",
                "production_claim": False,
                "required_scenarios": ["fixture"],
                "configs": {"primary": "configs/noise/full_hierarchy.example.yaml"},
                "artifacts": {
                    "metrics": metrics_path.as_posix(),
                    "summary_csv": report_path.as_posix(),
                    "calibration_summary": report_path.as_posix(),
                    "covariance_heatmap": report_path.as_posix(),
                    "gate06_summary": report_path.as_posix(),
                    "metrics_table": report_path.as_posix(),
                    "posterior_intervals": report_path.as_posix(),
                    "ppc_observable_overlay": report_path.as_posix(),
                    "ppc_summary_intervals": report_path.as_posix(),
                    "predictive_bands": report_path.as_posix(),
                    "recovery_parameter_intervals": report_path.as_posix(),
                    "report_md": report_path.as_posix(),
                    "residual_diagnostics": report_path.as_posix(),
                    "residual_whitened_hist": report_path.as_posix(),
                    "sbc_rank_histogram": report_path.as_posix(),
                    "synthetic_observable_overlay": report_path.as_posix(),
                },
                "residual_risk": "fixture residual risk",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_noise_config_examples_validate():
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = sorted((repo_root / "configs/noise").glob("*.example.yaml"))

    results = [validate_noise_config_file(path) for path in config_paths]

    assert config_paths
    assert all(result.passed for result in results), [result.as_dict() for result in results if not result.passed]
    names = {result.config_name for result in results}
    assert {"legacy", "synthetic_recovery", "predictive_checks", "emb_comparison"}.issubset(names)


def test_noise_config_validation_rejects_missing_metadata_and_unknown_family():
    missing = validate_noise_config_document({"schema_version": 1}, source="missing.yaml")

    assert missing.passed is False
    assert any("name" in error for error in missing.errors)
    assert any("family" in error for error in missing.errors)

    unsupported = validate_noise_config_document(
        {
            "schema_version": 1,
            "kind": "noise",
            "metadata": {"id": "bad", "name": "Bad"},
            "spec": {"family": "unsupported_family"},
        },
        source="bad.yaml",
    )

    assert unsupported.passed is False
    assert any("unsupported spec.family" in error for error in unsupported.errors)


def test_noise_config_validation_rejects_invalid_likelihood_components():
    invalid = validate_noise_config_document(
        {
            "schema_version": 1,
            "kind": "noise",
            "metadata": {"id": "bad_components", "name": "Bad Components"},
            "spec": {
                "family": "measurement_uncertainty",
                "likelihood": {
                    "stage": "M2",
                    "components": ["total_covariance"],
                },
            },
        },
        source="bad_components.yaml",
    )

    assert invalid.passed is False
    assert any("spec.likelihood" in error and "not available for stage M2" in error for error in invalid.errors)


def test_noise_config_validation_rejects_family_stage_mismatch():
    invalid = validate_noise_config_document(
        {
            "schema_version": 1,
            "kind": "noise",
            "metadata": {"id": "bad_family_stage", "name": "Bad Family Stage"},
            "spec": {
                "family": "gaussian",
                "likelihood": {
                    "stage": "M5",
                    "components": ["additive_noise", "relative_noise", "total_covariance"],
                },
            },
        },
        source="bad_family_stage.yaml",
    )

    assert invalid.passed is False
    assert any("must be M2 for family 'gaussian'" in error for error in invalid.errors)



def test_noise_mode_resolution_covers_legacy_staged_and_full_modes():
    assert supported_noise_modes()[0] == "legacy"
    legacy = resolve_noise_mode("legacy")
    full = resolve_noise_mode("full-hierarchy")
    emb = resolve_noise_mode("emb_comparison")

    assert legacy.stage == "M1"
    assert legacy.components == ("legacy",)
    assert legacy.legacy_mode == "emb_legacy"
    discrepancy = resolve_noise_mode("discrepancy")
    synthetic = resolve_noise_mode("synthetic_recovery")
    predictive = resolve_noise_mode("predictive_checks")
    assert full.stage == "M5"
    assert "total_covariance" in full.components
    assert "model_discrepancy" in full.components
    assert discrepancy.components == ("model_discrepancy", "total_covariance")
    assert discrepancy.components != full.components
    assert synthetic.components == ("synthetic_recovery",)
    assert predictive.components == ("predictive_checks",)
    assert emb.stage == "M7"
    assert emb.components == ("emb_comparison", "total_covariance")
    with pytest.raises(ValueError, match="Unsupported noise hierarchy mode"):
        resolve_noise_mode("missing-mode")


def test_release_readiness_and_gate07_scripts_write_required_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    synthetic = _write_evidence(tmp_path / "synthetic", "synthetic")
    predictive = _write_evidence(tmp_path / "predictive", "predictive")
    emb = _write_evidence(tmp_path / "emb", "emb")
    gate06 = tmp_path / "gate06.json"
    gate06.write_text(json.dumps({"pass": True, "failures": [], "warnings": []}), encoding="utf-8")
    output_root = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_hierarchy_release_readiness.py"),
            "--mode",
            "full_hierarchy",
            "--synthetic-manifest",
            str(synthetic),
            "--predictive-manifest",
            str(predictive),
            "--emb-manifest",
            str(emb),
            "--gate06-manifest",
            str(gate06),
            "--output-root",
            str(output_root),
            "--confirm-no-karolina-interaction",
        ],
        check=True,
        cwd=repo_root,
    )

    manifest_path = output_root / "noise_release_readiness_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["git_status_clean"] = True
    manifest["provenance"]["git_status_short"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest["gate07"]["pass"] is True
    assert manifest["config_validation"]["passed"] is True
    assert manifest["report_template"] == "full"
    assert manifest["merge_boundary"] == "human_review_required"
    assert manifest["no_karolina_interaction"] is True
    assert "python_version" in manifest["provenance"]
    assert {mode["mode"] for mode in manifest["modes"]} == set(supported_noise_modes())
    for path in manifest["artifacts"].values():
        assert Path(path).exists()

    gate_root = tmp_path / "gate07"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(manifest_path),
            "--output-root",
            str(gate_root),
            "--allow-missing-github-checks",
        ],
        check=True,
        cwd=repo_root,
    )
    gate_manifest = json.loads((gate_root / "noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert gate_manifest["pass"] is True
    assert gate_manifest["source_release_manifest"] == manifest_path.as_posix()
    assert (gate_root / "noise_gate07_report.md").exists()


def test_release_readiness_cli_reports_actionable_invalid_mode(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_hierarchy_release_readiness.py"),
            "--mode",
            "not-a-mode",
            "--config",
            str(repo_root / "configs/noise/legacy.example.yaml"),
            "--output-root",
            str(tmp_path / "bad"),
            "--confirm-no-karolina-interaction",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Unsupported noise hierarchy mode" in completed.stderr


def test_gate07_rejects_karolina_interaction_claim(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_manifest = tmp_path / "release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {"passed": True},
                "artifact_index": {"entries": {"synthetic_recovery": {}}},
                "merge_boundary": "human_review_required",
                "no_karolina_interaction": False,
                "provenance": {"git_commit": "abc123"},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(release_manifest),
            "--output-root",
            str(tmp_path / "gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    gate_manifest = json.loads((tmp_path / "gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("Karolina" in failure for failure in gate_manifest["failures"])


def test_gate07_rejects_forged_incomplete_release_manifest(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_manifest = tmp_path / "forged.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {"passed": True, "results": [{"config_name": "legacy", "passed": True}]},
                "artifact_index": {"entries": {"synthetic_recovery": {"exists": True}}},
                "merge_boundary": "human_review_required",
                "no_karolina_interaction": True,
                "karolina_interaction_confirmation": {"operator_confirmed": True},
                "provenance": {"git_commit": "abc123", "git_status_clean": True},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(release_manifest),
            "--output-root",
            str(tmp_path / "forged_gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    gate_manifest = json.loads((tmp_path / "forged_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("missing required evidence entries" in failure for failure in gate_manifest["failures"])
    assert any("Gate06 manifest" in failure for failure in gate_manifest["failures"])
