"""Dependency-light planning primitives for generated artifact relocation.

This module classifies generated-path candidates into conservative plan records.
It is intentionally non-destructive: it never moves, deletes, or rewrites files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS

APPROVED_GENERATED_ROOTS = ("_out", "_runs", "_ci", "out_hierarchical")
ROOT_LEVEL_LOG_SUFFIXES = (".out", ".err")
PROTECTED_TOP_LEVEL_ROOTS = ("extern/korali", "src", "configs", "docs", "tests", "examples")
PROTECTED_CHECKPOINT_HINTS = ("compression", "indentation")
CHECKPOINT_MARKERS = ("checkpoint", "checkpoints", "ckpt", "trained", "training")


class GeneratedArtifactPathKind(str, Enum):
    APPROVED_GENERATED_ROOT = "approved_generated_root"
    ROOT_LEVEL_SLURM_LOG = "root_level_slurm_log"
    AMBIGUOUS_NESTED_LOG = "ambiguous_nested_log"
    PROTECTED_ROOT = "protected_root"
    OWNER_DECISION_ONLY_CHECKPOINT = "owner_decision_only_checkpoint"
    PATH_TRAVERSAL = "path_traversal"
    FORBIDDEN_PRIVATE_PATH = "forbidden_private_path"
    OTHER = "other"


class GeneratedArtifactPlanAction(str, Enum):
    ARCHIVE = "archive"
    DELETE = "delete"
    HOLD = "hold"
    REJECT = "reject"
    INVESTIGATE = "investigate"


class GeneratedArtifactOwnerDecision(str, Enum):
    ARCHIVE = "archive"
    DELETE = "delete"
    KEEP = "keep"
    INVESTIGATE = "investigate"


@dataclass(frozen=True)
class GeneratedArtifactPathRecord:
    path: str
    normalized_path: str
    kind: GeneratedArtifactPathKind
    root_segment: str | None
    basename: str
    segments: tuple[str, ...]
    is_absolute: bool
    is_generated_root_candidate: bool
    is_root_log_candidate: bool
    protected: bool
    owner_decision_only: bool
    forbidden: bool
    can_archive: bool
    can_delete: bool
    owner_confirmation_required: bool
    reasons: tuple[str, ...]
    validation_notes: tuple[str, ...]
    rollback_notes: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedArtifactPlanRecord:
    path: str
    normalized_path: str
    kind: GeneratedArtifactPathKind
    owner_decision: GeneratedArtifactOwnerDecision | None
    planned_action: GeneratedArtifactPlanAction
    decision_applied: bool
    decision_valid: bool
    owner_confirmation_required: bool
    can_archive: bool
    can_delete: bool
    protected: bool
    forbidden: bool
    reasons: tuple[str, ...]
    validation_notes: tuple[str, ...]
    rollback_notes: tuple[str, ...]


def _normalize_path_text(path_value: str | Path) -> str:
    text = str(path_value).strip().replace("\\", "/")
    text = re.sub(r"^(?:\./)+", "", text)
    while text.endswith("/") and text != "/":
        text = text[:-1]
    return text


def _split_path_text(normalized_path: str) -> tuple[str, ...]:
    return tuple(part for part in normalized_path.split("/") if part and part != ".")


def _root_segment(segments: tuple[str, ...]) -> str | None:
    return segments[0] if segments else None


def _has_forbidden_private_literal(normalized_path: str) -> bool:
    return any(fragment in normalized_path for fragment in FORBIDDEN_PRIVATE_PATHS)


def _has_path_traversal(segments: tuple[str, ...]) -> bool:
    return any(part == ".." for part in segments)


def _is_approved_generated_root(normalized_path: str, segments: tuple[str, ...]) -> bool:
    if not segments:
        return False
    root = segments[0]
    return root in APPROVED_GENERATED_ROOTS or bool(re.fullmatch(r"_init_compression_[^/]+", root))


def _is_root_level_slurm_log(normalized_path: str, basename: str, segments: tuple[str, ...]) -> bool:
    return len(segments) == 1 and any(basename.endswith(suffix) for suffix in ROOT_LEVEL_LOG_SUFFIXES)


def _is_nested_slurm_log(basename: str, segments: tuple[str, ...]) -> bool:
    return len(segments) > 1 and any(basename.endswith(suffix) for suffix in ROOT_LEVEL_LOG_SUFFIXES)


def _is_protected_root(normalized_path: str, segments: tuple[str, ...]) -> bool:
    if not segments:
        return False
    candidate = normalized_path.lstrip("/")
    for protected_root in PROTECTED_TOP_LEVEL_ROOTS:
        if candidate == protected_root or candidate.startswith(f"{protected_root}/"):
            return True
    return False


def _is_checkpoint_like(segments: tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in segments)
    has_hint = any(any(part == hint or part.startswith(f"{hint}_") for part in lowered) for hint in PROTECTED_CHECKPOINT_HINTS)
    has_marker = any(any(marker in part for marker in CHECKPOINT_MARKERS) for part in lowered)
    return has_hint and has_marker


def _validation_notes_for_kind(kind: GeneratedArtifactPathKind) -> tuple[str, ...]:
    if kind == GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT:
        return (
            "Confirm the owner marked this generated root disposable before any relocation or deletion.",
            "Validate that the path is a top-level generated root and not a protected source tree subtree.",
        )
    if kind == GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG:
        return (
            "Root-level Slurm logs can be archived after owner confirmation.",
            "Verify the log is not the only trace of a failing run before deleting it.",
        )
    if kind == GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG:
        return (
            "Nested Slurm logs need provenance review before archival or deletion.",
            "Do not assume a nested .out/.err file is disposable just because it has a Slurm suffix.",
        )
    if kind == GeneratedArtifactPathKind.PROTECTED_ROOT:
        return (
            "Protected roots are investigation-only unless the owner explicitly overrides the policy.",
            "Do not mark source, docs, tests, examples, or vendor trees safe to delete.",
        )
    if kind == GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT:
        return (
            "Compression/indentation trained checkpoints require owner decision before any destructive action.",
            "Keep these artifacts intact until the checkpoint lineage is confirmed.",
        )
    if kind == GeneratedArtifactPathKind.PATH_TRAVERSAL:
        return (
            "Path traversal segments are rejected before generated-root approval.",
            "Do not relocate or delete paths that escape their apparent top-level root.",
        )
    if kind == GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH:
        return (
            "Karolina-forbidden private paths must be rejected.",
            "Do not touch paths rooted under the forbidden private literal.",
        )
    return (
        "No generated-root policy match was found for this path.",
        "Treat the path as investigation-needed unless the owner provides a specific decision.",
    )


def _rollback_notes_for_kind(kind: GeneratedArtifactPathKind) -> tuple[str, ...]:
    if kind in {
        GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT,
        GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG,
    }:
        return (
            "Archive the current path contents before any delete request is executed.",
            "Keep the original path mapping so the artifact can be restored if the owner later needs it.",
        )
    if kind == GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG:
        return (
            "Do not delete until the parent workflow and provenance are understood.",
            "If the file is later deemed disposable, archive it first so the run can be reconstructed.",
        )
    if kind == GeneratedArtifactPathKind.PROTECTED_ROOT:
        return (
            "No destructive action should be scheduled for protected roots without a separate owner-approved exception.",
            "Record the decision trail rather than attempting rollback on the file system.",
        )
    if kind == GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT:
        return (
            "Checkpoint rollback must preserve the latest trained state until the owner confirms a replacement.",
            "Snapshot the checkpoint directory before any relocation plan is executed.",
        )
    if kind == GeneratedArtifactPathKind.PATH_TRAVERSAL:
        return (
            "Reject the path and stop planning; traversal makes the generated-root classification unsafe.",
            "Ask the owner for a normalized, repo-relative path before any future action.",
        )
    if kind == GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH:
        return (
            "Reject the path and stop planning; the forbidden location is out of scope for relocation.",
            "Escalate the private path reference instead of trying to recover from it in place.",
        )
    return (
        "No delete or relocate action should be performed without a separate confirmed owner decision.",
        "Keep a copy of the current inventory entry so later policy updates can be replayed.",
    )


def _classify_kind(path_value: str | Path) -> GeneratedArtifactPathKind:
    normalized_path = _normalize_path_text(path_value)
    segments = _split_path_text(normalized_path)
    basename = segments[-1] if segments else normalized_path

    if _has_forbidden_private_literal(normalized_path):
        return GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH
    if _has_path_traversal(segments):
        return GeneratedArtifactPathKind.PATH_TRAVERSAL
    if _is_protected_root(normalized_path, segments):
        return GeneratedArtifactPathKind.PROTECTED_ROOT
    if _is_approved_generated_root(normalized_path, segments):
        return GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT
    if _is_root_level_slurm_log(normalized_path, basename, segments):
        return GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG
    if _is_nested_slurm_log(basename, segments):
        return GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG
    if _is_checkpoint_like(segments):
        return GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT
    return GeneratedArtifactPathKind.OTHER


def _is_owner_decision_only(kind: GeneratedArtifactPathKind) -> bool:
    return kind in {
        GeneratedArtifactPathKind.PROTECTED_ROOT,
        GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT,
    }


def _can_archive(kind: GeneratedArtifactPathKind) -> bool:
    return kind in {
        GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT,
        GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG,
    }


def _can_delete(kind: GeneratedArtifactPathKind) -> bool:
    return _can_archive(kind)


def _default_planned_action(kind: GeneratedArtifactPathKind) -> GeneratedArtifactPlanAction:
    if kind in {
        GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT,
        GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG,
    }:
        return GeneratedArtifactPlanAction.HOLD
    if kind == GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG:
        return GeneratedArtifactPlanAction.INVESTIGATE
    if kind in {
        GeneratedArtifactPathKind.PROTECTED_ROOT,
        GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT,
    }:
        return GeneratedArtifactPlanAction.HOLD
    if kind in {
        GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH,
        GeneratedArtifactPathKind.PATH_TRAVERSAL,
    }:
        return GeneratedArtifactPlanAction.REJECT
    return GeneratedArtifactPlanAction.HOLD


def _coerce_owner_decision(value: Any) -> GeneratedArtifactOwnerDecision | None:
    if value is None:
        return None
    if isinstance(value, GeneratedArtifactOwnerDecision):
        return value
    return GeneratedArtifactOwnerDecision(str(value))


def _lookup_owner_decision(
    path_record: GeneratedArtifactPathRecord,
    owner_decisions: Mapping[str, Any] | None,
) -> GeneratedArtifactOwnerDecision | None:
    if not owner_decisions:
        return None

    candidates = (
        path_record.path,
        path_record.normalized_path,
        Path(path_record.normalized_path).as_posix(),
    )
    for candidate in candidates:
        if candidate in owner_decisions:
            return _coerce_owner_decision(owner_decisions[candidate])
    return None


def classify_generated_artifact_path(path_value: str | Path) -> GeneratedArtifactPathRecord:
    normalized_path = _normalize_path_text(path_value)
    segments = _split_path_text(normalized_path)
    basename = segments[-1] if segments else normalized_path
    kind = _classify_kind(path_value)
    protected = kind == GeneratedArtifactPathKind.PROTECTED_ROOT
    owner_decision_only = _is_owner_decision_only(kind)
    forbidden = kind in {
        GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH,
        GeneratedArtifactPathKind.PATH_TRAVERSAL,
    }
    is_root_log = kind == GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG
    is_generated_root = kind == GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT

    reasons: list[str] = []
    if kind == GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT:
        reasons.append("top-level generated root")
    elif kind == GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG:
        reasons.append("root-level Slurm log")
    elif kind == GeneratedArtifactPathKind.AMBIGUOUS_NESTED_LOG:
        reasons.append("nested Slurm log needs provenance review")
    elif kind == GeneratedArtifactPathKind.PROTECTED_ROOT:
        reasons.append("protected source/tree root")
    elif kind == GeneratedArtifactPathKind.OWNER_DECISION_ONLY_CHECKPOINT:
        reasons.append("compression/indentation checkpoint lineage needs owner decision")
    elif kind == GeneratedArtifactPathKind.PATH_TRAVERSAL:
        reasons.append("path traversal is rejected before generated-root approval")
    elif kind == GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH:
        reasons.append("forbidden private Karolina path")
    else:
        reasons.append("no generated relocation policy match")

    if normalized_path.startswith("/"):
        reasons.append("absolute path requires ownership review")
    if kind == GeneratedArtifactPathKind.FORBIDDEN_PRIVATE_PATH:
        reasons.append("private path literal is blocked")
    if kind == GeneratedArtifactPathKind.PATH_TRAVERSAL:
        reasons.append("path traversal is blocked")

    validation_notes = _validation_notes_for_kind(kind)
    if normalized_path.startswith("/") and not forbidden:
        validation_notes = validation_notes + ("Absolute paths should be confirmed against the repo inventory before any action.",)
    rollback_notes = _rollback_notes_for_kind(kind)

    return GeneratedArtifactPathRecord(
        path=str(path_value),
        normalized_path=normalized_path,
        kind=kind,
        root_segment=_root_segment(segments),
        basename=basename,
        segments=segments,
        is_absolute=normalized_path.startswith("/"),
        is_generated_root_candidate=is_generated_root,
        is_root_log_candidate=is_root_log,
        protected=protected,
        owner_decision_only=owner_decision_only,
        forbidden=forbidden,
        can_archive=_can_archive(kind),
        can_delete=_can_delete(kind),
        owner_confirmation_required=True,
        reasons=tuple(reasons),
        validation_notes=validation_notes,
        rollback_notes=rollback_notes,
    )


def _finalize_planned_action(
    path_record: GeneratedArtifactPathRecord,
    owner_decision: GeneratedArtifactOwnerDecision | None,
) -> tuple[GeneratedArtifactPlanAction, bool]:
    if path_record.forbidden:
        return GeneratedArtifactPlanAction.REJECT, owner_decision is None

    default_action = _default_planned_action(path_record.kind)
    if owner_decision is None:
        return default_action, True

    if owner_decision == GeneratedArtifactOwnerDecision.KEEP:
        return GeneratedArtifactPlanAction.HOLD, True
    if owner_decision == GeneratedArtifactOwnerDecision.INVESTIGATE:
        return GeneratedArtifactPlanAction.INVESTIGATE, True

    if owner_decision == GeneratedArtifactOwnerDecision.ARCHIVE:
        if path_record.can_archive:
            return GeneratedArtifactPlanAction.ARCHIVE, True
        return default_action, False

    if owner_decision == GeneratedArtifactOwnerDecision.DELETE:
        if path_record.can_delete:
            return GeneratedArtifactPlanAction.DELETE, True
        return default_action, False

    return default_action, False


def build_generated_artifact_plan(
    path_values: Iterable[str | Path],
    *,
    owner_decisions: Mapping[str, Any] | None = None,
) -> tuple[GeneratedArtifactPlanRecord, ...]:
    records: list[GeneratedArtifactPlanRecord] = []
    for path_value in path_values:
        path_record = classify_generated_artifact_path(path_value)
        owner_decision = _lookup_owner_decision(path_record, owner_decisions)
        planned_action, decision_valid = _finalize_planned_action(path_record, owner_decision)
        decision_applied = owner_decision is not None and decision_valid
        records.append(
            GeneratedArtifactPlanRecord(
                path=path_record.path,
                normalized_path=path_record.normalized_path,
                kind=path_record.kind,
                owner_decision=owner_decision,
                planned_action=planned_action,
                decision_applied=decision_applied,
                decision_valid=decision_valid,
                owner_confirmation_required=path_record.owner_confirmation_required,
                can_archive=path_record.can_archive,
                can_delete=path_record.can_delete,
                protected=path_record.protected,
                forbidden=path_record.forbidden,
                reasons=path_record.reasons,
                validation_notes=path_record.validation_notes,
                rollback_notes=path_record.rollback_notes,
            )
        )
    return tuple(records)


def is_generated_artifact_root(path_value: str | Path) -> bool:
    record = classify_generated_artifact_path(path_value)
    return record.kind == GeneratedArtifactPathKind.APPROVED_GENERATED_ROOT


def is_root_level_slurm_log(path_value: str | Path) -> bool:
    record = classify_generated_artifact_path(path_value)
    return record.kind == GeneratedArtifactPathKind.ROOT_LEVEL_SLURM_LOG
