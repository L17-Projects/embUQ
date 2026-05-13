from __future__ import annotations

from pathlib import Path

from meso_uq.artifacts import validate_artifact_manifest_document, validate_artifact_manifest_file


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs" / "artifacts" / "artifact_manifest.example.json"


def test_artifact_manifest_example_validates() -> None:
    assert validate_artifact_manifest_file(EXAMPLE_MANIFEST) == []


def test_artifact_manifest_rejects_unknown_artifact_class() -> None:
    errors = validate_artifact_manifest_document(
        {
            "schema_version": "1.0",
            "manifest_id": "bad-manifest",
            "generated_at": "2026-05-13T00:00:00Z",
            "cleanup_policy": {"protected_roots": ["extern/korali"]},
            "artifacts": [
                {
                    "artifact_id": "bad-item",
                    "artifact_class": "mystery",
                    "path": "paper_data/output.bin",
                }
            ],
        }
    )

    assert any("artifact_class" in error for error in errors)


def test_artifact_manifest_rejects_private_absolute_paths() -> None:
    errors = validate_artifact_manifest_document(
        {
            "schema_version": "1.0",
            "manifest_id": "bad-path",
            "generated_at": "2026-05-13T00:00:00Z",
            "cleanup_policy": {"protected_roots": ["extern/korali"]},
            "artifacts": [
                {
                    "artifact_id": "private-item",
                    "artifact_class": "generated",
                    "path": "/ceph/hpc/home/eubrieucb/private/output",
                }
            ],
        }
    )

    assert any("forbidden private path" in error for error in errors)
