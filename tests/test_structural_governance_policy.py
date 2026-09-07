from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.support.governance import (
    DEFAULT_ALLOWLIST_PATH,
    load_governance_allowlist,
    validate_artifact_manifest_example,
    validate_ci_path_references,
    validate_config_schema_versions,
    validate_docs_retired_command_markers,
    validate_governance_allowlist_document,
    validate_import_cycle_allowlist_structure,
    validate_public_api_exports,
    validate_repo_structural_governance_policy,
    validate_tracked_generated_roots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_governance_allowlist_example_has_required_structure() -> None:
    allowlist = load_governance_allowlist(DEFAULT_ALLOWLIST_PATH)

    assert validate_governance_allowlist_document(allowlist, source=DEFAULT_ALLOWLIST_PATH) == []
    assert {entry["check"] for entry in allowlist["entries"]} == {
        "public_api_exports",
        "ci_path_references",
        "docs_retired_markers",
        "import_cycles",
    }
    assert all(entry["owner"] and entry["reason"] and entry["review_condition"] for entry in allowlist["entries"])


def test_governance_allowlist_validator_reports_missing_review_condition(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "manifest_id": "broken-allowlist",
        "generated_at": "2026-05-13T00:00:00Z",
        "entries": [
            {
                "check": "public_api_exports",
                "owner": "core-contracts",
                "reason": "missing review condition",
                "expected_exports": ["alpha"],
            }
        ],
    }

    errors = validate_governance_allowlist_document(payload, source=tmp_path / "governance.json")

    assert any("review_condition" in error for error in errors)


def test_config_schema_version_helper_reports_missing_schema_version(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "sample.yaml").write_text(
        "kind: dataset\nmetadata:\n  id: sample\n  name: Sample\n",
        encoding="utf-8",
    )

    errors = validate_config_schema_versions(tmp_path)

    assert any("sample.yaml" in error for error in errors)
    assert any("schema_version" in error for error in errors)
    assert any("Remediation" in error for error in errors)


def test_artifact_manifest_example_validates_via_helper() -> None:
    assert validate_artifact_manifest_example(REPO_ROOT) == []


def test_artifact_manifest_helper_reports_bad_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "configs" / "artifacts" / "artifact_manifest.example.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_id": "bad-manifest",
                "generated_at": "2026-05-13T00:00:00Z",
                "cleanup_policy": {"protected_roots": ["extern/korali"]},
                "artifacts": [
                    {
                        "artifact_id": "bad-path",
                        "artifact_class": "generated",
                        "path": "/tmp/private/output.bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from meso_uq.artifacts.policy import validate_artifact_manifest_file

    errors = validate_artifact_manifest_file(manifest_path)

    assert any("artifact_manifest.example.json" in error for error in errors)
    assert any("private path" in error or "relative or placeholder-based" in error for error in errors)


def test_generated_root_tracked_file_detection_uses_git_ls_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path, check=True, capture_output=True)

    tracked_root = tmp_path / "_runs" / "job-1"
    tracked_root.mkdir(parents=True)
    (tracked_root / "output.txt").write_text("payload", encoding="utf-8")
    safe_file = tmp_path / "docs" / "note.md"
    safe_file.parent.mkdir(parents=True)
    safe_file.write_text("ok", encoding="utf-8")

    subprocess.run(["git", "add", "_runs/job-1/output.txt", "docs/note.md"], cwd=tmp_path, check=True, capture_output=True)

    errors = validate_tracked_generated_roots(tmp_path)

    assert any("_runs/job-1/output.txt" in error for error in errors)
    assert any("Remediation" in error for error in errors)


def test_generated_root_tracker_accepts_clean_tracked_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path, check=True, capture_output=True)

    safe_file = tmp_path / "docs" / "note.md"
    safe_file.parent.mkdir(parents=True)
    safe_file.write_text("ok", encoding="utf-8")
    subprocess.run(["git", "add", "docs/note.md"], cwd=tmp_path, check=True, capture_output=True)

    assert validate_tracked_generated_roots(tmp_path) == []


def test_public_api_exports_match_allowlist() -> None:
    allowlist = load_governance_allowlist(DEFAULT_ALLOWLIST_PATH)

    assert validate_public_api_exports(allowlist) == []


def test_public_api_exports_helper_reports_mismatch() -> None:
    allowlist = {
        "entries": [
            {
                "check": "public_api_exports",
                "owner": "core-contracts",
                "reason": "fixture",
                "review_condition": "fixture",
                "expected_exports": ["alpha", "beta"],
            }
        ]
    }

    errors = validate_public_api_exports(allowlist)

    assert errors
    assert "meso_uq.public_api.__all__" in errors[0]
    assert "missing" in errors[0] or "extra" in errors[0]


def test_ci_path_reference_helper_reports_unallowlisted_path(tmp_path: Path) -> None:
    repo_root = tmp_path
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  example:\n"
        "    steps:\n"
        "      - run: echo _ci/unlisted/path\n",
        encoding="utf-8",
    )
    allowlist = {
        "entries": [
            {
                "check": "ci_path_references",
                "owner": "ci-governance",
                "reason": "fixture",
                "review_condition": "fixture",
                "files": [{"path": ".github/workflows/ci.yml", "fragments": ["_ci/allowed"]}],
            }
        ]
    }

    errors = validate_ci_path_references(repo_root, allowlist)

    assert any(".github/workflows/ci.yml:4" in error for error in errors)
    assert any("unallowlisted" in error or "not allowlisted" in error for error in errors)


def test_docs_retired_marker_helper_reports_unallowlisted_marker(tmp_path: Path) -> None:
    repo_root = tmp_path
    doc = repo_root / "docs" / "guide.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("This section is deprecated until migration finishes.\n", encoding="utf-8")
    allowlist = {
        "entries": [
            {
                "check": "docs_retired_markers",
                "owner": "docs-governance",
                "reason": "fixture",
                "review_condition": "fixture",
                "files": [{"path": "docs/guide.md", "fragments": ["legacy compatibility"]}],
            }
        ]
    }

    errors = validate_docs_retired_command_markers(repo_root, allowlist)

    assert any("docs/guide.md:1" in error for error in errors)
    assert any("deprecated" in error for error in errors)


def test_import_cycle_allowlist_structure_matches_expected_sequences() -> None:
    allowlist = load_governance_allowlist(DEFAULT_ALLOWLIST_PATH)

    assert validate_import_cycle_allowlist_structure(allowlist) == []
    import_cycles = next(entry for entry in allowlist["entries"] if entry["check"] == "import_cycles")
    assert {
        tuple(sequence)
        for sequence in import_cycles["sequences"]
    } == {
        ("meso_uq.surrogate.catalogs", "meso_uq.surrogate.emb_catalog"),
        ("meso_uq.surrogate.emb_catalog", "meso_uq.surrogate.catalogs"),
    }


def test_import_cycle_allowlist_structure_rejects_short_sequence() -> None:
    allowlist = {
        "entries": [
            {
                "check": "import_cycles",
                "owner": "import-governance",
                "reason": "fixture",
                "review_condition": "fixture",
                "sequences": [["meso_uq.structures"]],
            }
        ]
    }

    errors = validate_import_cycle_allowlist_structure(allowlist)

    assert any("at least two module names" in error for error in errors)


def test_repo_structural_governance_policy_passes_current_repo() -> None:
    assert validate_repo_structural_governance_policy(REPO_ROOT) == []
