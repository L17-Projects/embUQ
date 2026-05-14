from __future__ import annotations

from pathlib import Path

from meso_uq.artifacts import (
    ArtifactManifestRecord,
    ArtifactValidationStatus,
    class_policy_for_artifact_class,
    is_source_tree_generated_root,
    validate_artifact_manifest_document,
    validate_artifact_manifest_file,
)


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


def test_artifact_manifest_accepts_release_critical_surrogate_artifacts() -> None:
    errors = validate_artifact_manifest_document(
        {
            "schema_version": "1.0",
            "manifest_id": "release-critical-slice",
            "generated_at": "2026-05-13T00:00:00Z",
            "cleanup_policy": {"protected_roots": ["extern/korali"]},
            "artifacts": [
                {
                    "artifact_id": "surrogate-checkpoint-2026",
                    "artifact_class": "surrogate_checkpoint",
                    "path": "_runs/release_critical/surrogate.pt",
                }
            ],
        }
    )

    assert errors == []


def test_artifact_manifest_rejects_generated_root_for_non_curated_artifacts() -> None:
    errors = validate_artifact_manifest_document(
        {
            "schema_version": "1.0",
            "manifest_id": "generated-root-run",
            "generated_at": "2026-05-13T00:00:00Z",
            "cleanup_policy": {"protected_roots": ["extern/korali"]},
            "artifacts": [
                {
                    "artifact_id": "generated-run-artifact",
                    "artifact_class": "generated",
                    "path": "_runs/validation/run_001/output.bin",
                }
            ],
        }
    )

    assert any("source-tree generated root" in error for error in errors)


def test_surrogate_checkpoint_is_release_critical_by_policy() -> None:
    policy = class_policy_for_artifact_class("surrogate_checkpoint")
    assert policy.release_critical


def test_generated_root_detector_matches_dirtree_hints() -> None:
    assert is_source_tree_generated_root("_runs/something")
    assert is_source_tree_generated_root("_init_compression_2.1um/results")
    assert not is_source_tree_generated_root("paper_data/surrogates/model.pt")


def test_generated_root_detector_preserves_placeholder_roots() -> None:
    assert not is_source_tree_generated_root("${MESOUQ_RUNS_ROOT}/logs/driver.log")
    assert not is_source_tree_generated_root("$MESOUQ_RUNS_ROOT/_runs/run_001/output.bin")


def test_artifact_manifest_accepts_generated_placeholder_roots() -> None:
    errors = validate_artifact_manifest_document(
        {
            "schema_version": "1.0",
            "manifest_id": "placeholder-root-run",
            "generated_at": "2026-05-13T00:00:00Z",
            "cleanup_policy": {"protected_roots": ["extern/korali"]},
            "artifacts": [
                {
                    "artifact_id": "generated-placeholder-log",
                    "artifact_class": "log",
                    "path": "${MESOUQ_RUNS_ROOT}/logs/driver.log",
                    "storage_location": "hpc_output",
                    "retention_policy": "generated",
                    "release_critical": False,
                }
            ],
        }
    )

    assert errors == []


def test_artifact_manifest_record_type_accepts_metadata_shape() -> None:
    record_payload = {
        "artifact_id": "record-001",
        "artifact_class": "surrogate_checkpoint",
        "path": "paper_data/surrogates/example.pt",
        "checksum": {"algorithm": "sha256", "value": "abc123"},
        "provenance": {"platform": "karolina", "metadata": {"campaign": "baseline"}},
        "retention_policy": "curated",
        "storage_location": "source_tree",
        "validation_status": "valid",
    }

    record = ArtifactManifestRecord.from_dict(record_payload)

    assert record.artifact_id == "record-001"
    assert record.validation_status == ArtifactValidationStatus.VALID
    assert record.checksum is not None


def test_artifact_manifest_record_defaults_validation_status() -> None:
    record = ArtifactManifestRecord.from_dict(
        {
            "artifact_id": "record-defaults",
            "artifact_class": "reference",
            "path": "data/reference.csv",
        }
    )

    assert record.validation_status == ArtifactValidationStatus.UNKNOWN
