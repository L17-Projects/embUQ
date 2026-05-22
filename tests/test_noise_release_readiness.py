import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from meso_uq.noise import (
    build_noise_artifact_index,
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


def test_artifact_index_records_malformed_evidence_json(tmp_path):
    malformed_manifest = tmp_path / "synthetic_manifest.json"
    malformed_manifest.write_text("{not json", encoding="utf-8")

    index = build_noise_artifact_index({"synthetic_recovery": malformed_manifest})

    entry = index["entries"]["synthetic_recovery"]
    assert entry["exists"] is False
    assert entry["all_scenarios_passed"] is False
    assert "invalid JSON" in entry["read_error"]


def test_artifact_index_records_malformed_metrics_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{not json", encoding="utf-8")
    manifest_path = tmp_path / "synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "provenance": {"git_commit": "abc123", "git_status_clean": True},
                "artifacts": {"metrics": metrics_path.name},
            }
        ),
        encoding="utf-8",
    )

    index = build_noise_artifact_index({"synthetic_recovery": manifest_path})

    entry = index["entries"]["synthetic_recovery"]
    assert entry["exists"] is True
    assert entry["metrics_exists"] is True
    assert entry["all_scenarios_passed"] is False
    assert "invalid JSON" in entry["metrics_error"]


def test_release_readiness_script_reports_malformed_metrics_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    synthetic = _write_evidence(tmp_path / "synthetic", "synthetic")
    (synthetic.parent / "synthetic_metrics.json").write_text("{not json", encoding="utf-8")
    predictive = _write_evidence(tmp_path / "predictive", "predictive")
    emb = _write_evidence(tmp_path / "emb", "emb")
    gate06 = tmp_path / "gate06.json"
    gate06.write_text(json.dumps({"pass": True, "failures": [], "warnings": []}), encoding="utf-8")
    output_root = tmp_path / "release"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_hierarchy_release_readiness.py"),
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
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    manifest = json.loads((output_root / "noise_release_readiness_manifest.json").read_text(encoding="utf-8"))
    failures = manifest["gate07"]["failures"]
    assert any("synthetic_recovery metrics could not be parsed" in failure for failure in failures)


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


def test_noise_config_validation_requires_likelihood_for_hierarchy_modes():
    invalid = validate_noise_config_document(
        {
            "schema_version": "1.0",
            "kind": "noise",
            "metadata": {"id": "synthetic_recovery", "name": "Synthetic Recovery"},
            "spec": {
                "family": "noise_hierarchy",
                "required_scenarios": [{"id": "fixture"}],
                "artifacts": {"metrics": "metrics.json"},
            },
        },
        source="synthetic_recovery.yaml",
    )

    assert invalid.passed is False
    assert any("spec.likelihood is required" in error for error in invalid.errors)


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
    assert manifest["artifacts"]["release_report"] == "noise_release_readiness_report.md"
    for path in manifest["artifacts"].values():
        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        assert artifact_path.exists()

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
        cwd=tmp_path,
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


def test_gate07_resolves_release_manifest_relative_paths_from_other_cwd(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_root = tmp_path / "release"
    release_root.mkdir()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    required_artifacts = {
        "synthetic_recovery": {
            "metrics",
            "summary_csv",
            "covariance_heatmap",
            "recovery_parameter_intervals",
            "residual_whitened_hist",
            "synthetic_observable_overlay",
        },
        "predictive_checks": {
            "metrics",
            "summary_csv",
            "calibration_summary",
            "ppc_observable_overlay",
            "ppc_summary_intervals",
            "sbc_rank_histogram",
        },
        "emb_comparison": {
            "metrics",
            "summary_csv",
            "gate06_summary",
            "report_md",
            "metrics_table",
            "posterior_intervals",
            "predictive_bands",
            "residual_diagnostics",
        },
    }
    entries = {}
    for label, artifact_names in required_artifacts.items():
        metrics_path = release_root / f"{label}_metrics.json"
        metrics_path.write_text(
            json.dumps({"all_scenarios_passed": True, "scenario_gate_statuses": {"fixture": "pass"}}),
            encoding="utf-8",
        )
        sidecar_path = release_root / f"{label}_sidecar.txt"
        sidecar_path.write_text("sidecar\n", encoding="utf-8")
        artifacts = {name: sidecar_path.name for name in artifact_names}
        artifacts["metrics"] = metrics_path.name
        manifest_path = release_root / f"{label}_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "commands": {"regenerate": f"python {label}.py"},
                    "provenance": {"git_commit": "abc123", "git_status_clean": True},
                    "configs": {"primary": "configs/noise/full_hierarchy.example.yaml"},
                    "artifacts": artifacts,
                    "residual_risk": "fixture residual risk",
                }
            ),
            encoding="utf-8",
        )
        entries[label] = {
            "path": manifest_path.name,
            "exists": True,
            "all_scenarios_passed": True,
            "scenario_gate_statuses": {"fixture": "pass"},
            "git_status_clean": True,
            "git_commit": "abc123",
            "commands": {"regenerate": f"python {label}.py"},
            "configs": {"primary": "configs/noise/full_hierarchy.example.yaml"},
            "residual_risk": "fixture residual risk",
            "artifact_existence": {name: True for name in artifact_names},
        }
    gate06_path = release_root / "gate06.json"
    gate06_path.write_text(json.dumps({"pass": True}), encoding="utf-8")
    release_artifacts = {}
    for artifact_name, file_name in {
        "artifact_index": "artifact_index.json",
        "config_validation": "config_validation.json",
        "release_manifest": "release.json",
        "release_report": "report.md",
    }.items():
        artifact_path = release_root / file_name
        artifact_path.write_text("{}\n" if file_name.endswith(".json") else "# report\n", encoding="utf-8")
        release_artifacts[artifact_name] = file_name
    release_manifest = release_root / "release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {
                    "passed": True,
                    "results": [
                        {"config_name": name, "passed": True, "path": f"{name}.yaml"}
                        for name in ("legacy", "synthetic_recovery", "predictive_checks", "emb_comparison")
                    ],
                },
                "artifact_index": {"entries": entries},
                "gate06_manifest": gate06_path.name,
                "artifacts": release_artifacts,
                "merge_boundary": "human_review_required",
                "no_karolina_interaction": True,
                "karolina_interaction_confirmation": {"operator_confirmed": True},
                "provenance": {"git_commit": "abc123", "git_status_clean": True},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(release_manifest),
            "--output-root",
            str(tmp_path / "gate07_relative"),
            "--allow-missing-github-checks",
        ],
        cwd=other_cwd,
        check=True,
    )


def test_gate07_requires_release_artifact_manifest_entries(tmp_path):
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
    manifest.pop("artifacts")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "gate07_missing_artifacts"),
            "--allow-missing-github-checks",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    gate_manifest = json.loads((tmp_path / "gate07_missing_artifacts/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("release manifest artifacts" in failure for failure in gate_manifest["failures"])


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


def test_gate07_rejects_non_mapping_artifact_entry_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_manifest = tmp_path / "forged_scalar_entry.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {
                    "passed": True,
                    "results": [
                        {"config_name": name, "passed": True, "path": f"{name}.yaml"}
                        for name in ("legacy", "synthetic_recovery", "predictive_checks", "emb_comparison")
                    ],
                },
                "artifact_index": {
                    "entries": {
                        "synthetic_recovery": "not an object",
                        "predictive_checks": {},
                        "emb_comparison": {},
                    }
                },
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
            str(tmp_path / "scalar_entry_gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    gate_manifest = json.loads((tmp_path / "scalar_entry_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("artifact index entry synthetic_recovery is not an object" in failure for failure in gate_manifest["failures"])


def test_gate07_reports_malformed_release_manifest_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_manifest = tmp_path / "malformed_release.json"
    release_manifest.write_text("{not json", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(release_manifest),
            "--output-root",
            str(tmp_path / "malformed_release_gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    gate_manifest = json.loads((tmp_path / "malformed_release_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("release manifest could not be read" in failure for failure in gate_manifest["failures"])


def test_gate07_rejects_non_mapping_evidence_manifest_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    evidence_manifest = tmp_path / "synthetic_evidence.json"
    evidence_manifest.write_text("[]", encoding="utf-8")
    release_manifest = tmp_path / "forged_evidence_root.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {
                    "passed": True,
                    "results": [
                        {"config_name": name, "passed": True, "path": f"{name}.yaml"}
                        for name in ("legacy", "synthetic_recovery", "predictive_checks", "emb_comparison")
                    ],
                },
                "artifact_index": {
                    "entries": {
                        "synthetic_recovery": {
                            "exists": True,
                            "path": evidence_manifest.name,
                            "all_scenarios_passed": True,
                            "scenario_gate_statuses": {"fixture": "pass"},
                            "git_status_clean": True,
                            "git_commit": "abc123",
                            "commands": {"regenerate": "python synthetic.py"},
                            "configs": {"primary": "configs/noise/full_hierarchy.example.yaml"},
                            "residual_risk": "fixture residual risk",
                        },
                        "predictive_checks": {},
                        "emb_comparison": {},
                    }
                },
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
            str(tmp_path / "evidence_root_gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    gate_manifest = json.loads((tmp_path / "evidence_root_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("evidence manifest synthetic_recovery root is not an object" in failure for failure in gate_manifest["failures"])


def test_gate07_reports_malformed_gate06_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    gate06 = tmp_path / "gate06.json"
    gate06.write_text("{not json", encoding="utf-8")
    release_manifest = tmp_path / "malformed_gate06_release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {
                    "passed": True,
                    "results": [
                        {"config_name": name, "passed": True, "path": f"{name}.yaml"}
                        for name in ("legacy", "synthetic_recovery", "predictive_checks", "emb_comparison")
                    ],
                },
                "artifact_index": {"entries": {"synthetic_recovery": {}, "predictive_checks": {}, "emb_comparison": {}}},
                "gate06_manifest": gate06.name,
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
            str(tmp_path / "malformed_gate06_gate"),
            "--allow-missing-github-checks",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    gate_manifest = json.loads((tmp_path / "malformed_gate06_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("Gate06 manifest could not be read" in failure for failure in gate_manifest["failures"])


def test_gate07_reports_missing_gh_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    release_manifest = tmp_path / "github_release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "gate07": {"pass": True},
                "config_validation": {"passed": True, "results": []},
                "artifact_index": {"entries": {}},
                "merge_boundary": "human_review_required",
                "no_karolina_interaction": True,
                "karolina_interaction_confirmation": {"operator_confirmed": True},
                "provenance": {"git_commit": "abc123", "git_status_clean": True},
            }
        ),
        encoding="utf-8",
    )
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    env = {**os.environ, "PATH": empty_path.as_posix()}

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate07_release_checks.py"),
            "--release-manifest",
            str(release_manifest),
            "--output-root",
            str(tmp_path / "missing_gh_gate"),
            "--github-pr",
            "202",
            "--repo",
            "BrieucB/MesoUQ",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    gate_manifest = json.loads((tmp_path / "missing_gh_gate/noise_gate07_manifest.json").read_text(encoding="utf-8"))
    assert any("GitHub PR head commit could not be read" in failure for failure in gate_manifest["failures"])
    assert any("GitHub PR checks could not be read" in failure for failure in gate_manifest["failures"])
