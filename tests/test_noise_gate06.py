import json
import subprocess
import sys
from pathlib import Path


def _write_evidence(root: Path, name: str, *, evidence_class: str | None = None, production_claim: bool = False, clean: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / f"{name}_metrics.json"
    manifest_path = root / f"{name}_manifest.json"
    metrics_path.write_text(
        json.dumps({"schema_version": 1, "all_scenarios_passed": True, "scenario_gate_statuses": {"scenario": "pass"}}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "commands": {"regenerate": "python fixture.py"},
        "provenance": {"git_commit": "abc123", "git_branch": "feature/test", "git_status_clean": clean},
        "required_scenarios": ["scenario"],
        "scenario_artifacts": {"scenario": {"metrics": metrics_path.as_posix()}},
        "artifacts": {"metrics": metrics_path.as_posix()},
    }
    if evidence_class is not None:
        manifest["evidence_class"] = evidence_class
        manifest["production_claim"] = production_claim
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_gate06_accepts_validation_fixture_without_production_claim(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    synthetic = _write_evidence(tmp_path / "synthetic", "synthetic")
    predictive = _write_evidence(tmp_path / "predictive", "predictive")
    emb = _write_evidence(tmp_path / "emb", "emb", evidence_class="validation_fixture", production_claim=False)
    output_root = tmp_path / "gate"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate06_integrated_emb.py"),
            "--synthetic-manifest",
            str(synthetic),
            "--predictive-manifest",
            str(predictive),
            "--emb-manifest",
            str(emb),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo_root,
    )

    manifest = json.loads((output_root / "noise_gate06_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pass"] is True
    assert manifest["evidence_class"] == "validation_fixture"
    assert manifest["production_claim"] is False
    assert manifest["warnings"]
    assert (output_root / "noise_gate06_report.md").exists()


def test_gate06_rejects_fixture_when_production_is_required(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    synthetic = _write_evidence(tmp_path / "synthetic", "synthetic")
    predictive = _write_evidence(tmp_path / "predictive", "predictive")
    emb = _write_evidence(tmp_path / "emb", "emb", evidence_class="validation_fixture", production_claim=False)
    output_root = tmp_path / "gate"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/qa/noise_gate06_integrated_emb.py"),
            "--synthetic-manifest",
            str(synthetic),
            "--predictive-manifest",
            str(predictive),
            "--emb-manifest",
            str(emb),
            "--output-root",
            str(output_root),
            "--require-production",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    manifest = json.loads((output_root / "noise_gate06_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pass"] is False
    assert any("require-production" in failure for failure in manifest["failures"])
