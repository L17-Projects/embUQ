from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.artifacts.relocation import (  # noqa: E402
    GeneratedArtifactOwnerDecision,
    GeneratedArtifactPathKind,
    GeneratedArtifactPlanAction,
    build_generated_artifact_plan,
    classify_generated_artifact_path,
    is_generated_artifact_root,
    is_root_level_slurm_log,
)


def test_classifies_approved_generated_roots_as_archive_delete_candidates() -> None:
    record = classify_generated_artifact_path("_runs/karolina/flow/run-01")

    assert record.kind is GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT
    assert record.root_segment == "_runs"
    assert record.can_archive is True
    assert record.can_delete is True
    assert record.owner_confirmation_required is True
    assert is_generated_artifact_root("_runs/karolina/flow/run-01") is True

    plan = build_generated_artifact_plan(
        ["_runs/karolina/flow/run-01"],
        owner_decisions={"_runs/karolina/flow/run-01": "delete"},
    )[0]

    assert plan.owner_decision is GeneratedArtifactOwnerDecision.DELETE
    assert plan.planned_action is GeneratedArtifactPlanAction.DELETE
    assert plan.decision_valid is True
    assert plan.decision_applied is True

    archive_plan = build_generated_artifact_plan(
        ["_init_compression_case-17"],
        owner_decisions={"_init_compression_case-17": "archive"},
    )[0]

    assert archive_plan.kind is GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT
    assert archive_plan.planned_action is GeneratedArtifactPlanAction.ARCHIVE
    assert archive_plan.decision_valid is True


def test_rejects_generated_root_traversal_before_approval() -> None:
    traversal_path = "_runs/../src/meso_uq"
    record = classify_generated_artifact_path(traversal_path)

    assert record.kind is GeneratedArtifactPathKind.PATH_TRAVERSAL
    assert record.is_generated_root_candidate is False
    assert record.can_archive is False
    assert record.can_delete is False
    assert record.forbidden is True
    assert any("traversal" in reason for reason in record.reasons)

    plan = build_generated_artifact_plan(
        [traversal_path],
        owner_decisions={traversal_path: "delete"},
    )[0]
    assert plan.planned_action is GeneratedArtifactPlanAction.REJECT
    assert plan.decision_valid is False


def test_classifies_root_level_slurm_logs_and_keeps_nested_logs_investigation_needed() -> None:
    root_log = classify_generated_artifact_path("job-42.out")
    nested_log = classify_generated_artifact_path("runs/job-42.err")

    assert root_log.kind is GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG
    assert root_log.can_archive is True
    assert root_log.can_delete is True
    assert root_log.owner_confirmation_required is True
    assert is_root_level_slurm_log("job-42.out") is True

    root_plan = build_generated_artifact_plan(
        ["job-42.out"],
        owner_decisions={"job-42.out": "archive"},
    )[0]
    assert root_plan.planned_action is GeneratedArtifactPlanAction.ARCHIVE
    assert root_plan.decision_valid is True

    assert nested_log.kind is GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG
    assert nested_log.can_archive is False
    assert nested_log.can_delete is False

    nested_plan = build_generated_artifact_plan(["runs/job-42.err"])[0]
    assert nested_plan.planned_action is GeneratedArtifactPlanAction.INVESTIGATE
    assert nested_plan.decision_applied is False


def test_protected_roots_and_checkpoint_like_paths_are_not_marked_safe_to_delete() -> None:
    protected_record = classify_generated_artifact_path("extern/korali/checkpoints/latest.json")
    checkpoint_record = classify_generated_artifact_path("emb/compression/trained_checkpoint/run-17.ckpt")

    assert protected_record.kind is GeneratedArtifactPathKind.PROTECTED_ROOT
    assert protected_record.can_archive is False
    assert protected_record.can_delete is False
    assert protected_record.owner_confirmation_required is True

    protected_plan = build_generated_artifact_plan(
        ["extern/korali/checkpoints/latest.json"],
        owner_decisions={"extern/korali/checkpoints/latest.json": "delete"},
    )[0]
    assert protected_plan.planned_action is GeneratedArtifactPlanAction.HOLD
    assert protected_plan.decision_valid is False

    assert checkpoint_record.kind is GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT
    assert checkpoint_record.can_archive is False
    assert checkpoint_record.can_delete is False
    assert checkpoint_record.owner_confirmation_required is True


def test_forbidden_private_paths_are_rejected() -> None:
    forbidden_path = "/ceph/hpc/home/eubrieucb/_runs/project/job-1.out"
    record = classify_generated_artifact_path(forbidden_path)

    assert record.kind is GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH
    assert record.forbidden is True
    assert record.can_archive is False
    assert record.can_delete is False
    assert any("forbidden" in note.lower() for note in record.validation_notes)

    plan = build_generated_artifact_plan([forbidden_path], owner_decisions={forbidden_path: "delete"})[0]
    assert plan.planned_action is GeneratedArtifactPlanAction.REJECT
    assert plan.decision_valid is False


def test_module_import_is_dependency_light() -> None:
    script = """
import sys
from pathlib import Path

repo_root = Path.cwd()
src_root = repo_root / "src"
sys.path.insert(0, str(src_root))

import meso_uq.artifacts.relocation  # noqa: F401

print("numpy" in sys.modules, "pandas" in sys.modules, "torch" in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False False False"
