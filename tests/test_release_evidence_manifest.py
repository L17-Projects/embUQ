from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "release" / "validate_release_evidence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_release_evidence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_release_manifest_is_structurally_valid() -> None:
    module = _load_module()
    example_path = REPO_ROOT / "examples" / "reports" / "release_evidence_manifest.example.json"
    manifest = json.loads(example_path.read_text(encoding="utf-8"))

    assert module.validate_manifest(manifest) == []


def test_validator_enforces_v010_native_cuda_non_claim() -> None:
    module = _load_module()
    manifest = {
        "release": "v0.1.0",
        "status": "ready",
        "candidate_commit": "abc123",
        "claims": {"native_cuda_phase2_public_claim": True},
        "github_ci": {
            "ci_workflow_url": "https://github.com/L17-Projects/embUQ/actions/runs/1",
            "release_smoke_url": "https://github.com/L17-Projects/embUQ/actions/runs/2",
        },
        "reports": {
            "vega_validation_matrix": "/tmp/validation.json",
            "vega_acceptance": "/tmp/acceptance.json",
            "vega_production_sanity": "/tmp/production.json",
            "workstation_acceptance": "/tmp/workstation.json",
        },
        "documents": {
            "release_notes": "docs/RELEASE_NOTES_v0.1.0.md",
            "validation_matrix": "docs/VALIDATION_MATRIX.md",
            "workstation_acceptance_checklist": "docs/WORKSTATION_ACCEPTANCE_CHECKLIST.md",
            "security_policy": "SECURITY.md",
        },
    }

    errors = module.validate_manifest(manifest)
    assert any("native_cuda_phase2_public_claim" in error for error in errors)


def test_validator_accepts_existing_paths(tmp_path: Path) -> None:
    module = _load_module()
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    validation_report = reports_root / "validation.json"
    acceptance_report = reports_root / "acceptance.json"
    production_report = reports_root / "production.json"
    workstation_report = reports_root / "workstation.json"
    for path in (validation_report, acceptance_report, production_report, workstation_report):
        path.write_text("{}", encoding="utf-8")

    manifest = {
        "release": "v0.1.0",
        "status": "ready",
        "candidate_commit": "3be8f4ccbc55d90d4ed58141c8268718fbd841d6",
        "claims": {"native_cuda_phase2_public_claim": False},
        "github_ci": {
            "ci_workflow_url": "https://github.com/L17-Projects/embUQ/actions/runs/24182978893",
            "release_smoke_url": "https://github.com/L17-Projects/embUQ/actions/runs/24182978877",
        },
        "reports": {
            "vega_validation_matrix": str(validation_report),
            "vega_acceptance": str(acceptance_report),
            "vega_production_sanity": str(production_report),
            "workstation_acceptance": str(workstation_report),
        },
        "documents": {
            "release_notes": "docs/RELEASE_NOTES_v0.1.0.md",
            "validation_matrix": "docs/VALIDATION_MATRIX.md",
            "workstation_acceptance_checklist": "docs/WORKSTATION_ACCEPTANCE_CHECKLIST.md",
            "security_policy": "SECURITY.md",
        },
    }

    assert module.validate_manifest(manifest, must_exist=True) == []
