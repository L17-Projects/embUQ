from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from meso_uq.core import ArtifactClass, coerce_artifact_class
from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS, load_structured_document

SUPPORTED_ARTIFACT_CLASSES = frozenset(item.value for item in ArtifactClass)
PROTECTED_VENDOR_ROOTS = ("extern/korali",)
SOURCE_TREE_GENERATED_ROOTS = ("_out", "_runs", "_ci", "out_hierarchical", "logs")


class ArtifactRetentionPolicy(str, Enum):
    CURATED = "curated"
    GENERATED = "generated"


class ArtifactStorageLocation(str, Enum):
    SOURCE_TREE = "source_tree"
    GENERATED_ROOT = "generated_root"
    HPC_OUTPUT = "hpc_output"
    EXTERNAL = "external"


class ArtifactValidationStatus(str, Enum):
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class ChecksumMetadata:
    algorithm: str
    value: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChecksumMetadata":
        return cls(
            algorithm=str(payload["algorithm"]).strip(),
            value=str(payload["value"]).strip(),
        )


@dataclass(frozen=True)
class ArtifactProvenance:
    generated_by: str | None = None
    generated_at_tool: str | None = None
    platform: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactProvenance":
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("provenance.metadata must be a mapping.")
        return cls(
            generated_by=payload.get("generated_by"),
            generated_at_tool=payload.get("generated_at_tool"),
            platform=payload.get("platform"),
            metadata={str(key): value for key, value in metadata.items()},
        )


@dataclass(frozen=True)
class ArtifactClassPolicy:
    artifact_class: ArtifactClass
    retention: ArtifactRetentionPolicy
    default_storage: ArtifactStorageLocation
    release_critical: bool = False


_ARTIFACT_CLASS_POLICIES: dict[ArtifactClass, ArtifactClassPolicy] = {
    ArtifactClass.RAW: ArtifactClassPolicy(
        artifact_class=ArtifactClass.RAW,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.REFERENCE: ArtifactClassPolicy(
        artifact_class=ArtifactClass.REFERENCE,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.PROCESSED: ArtifactClassPolicy(
        artifact_class=ArtifactClass.PROCESSED,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.GENERATED: ArtifactClassPolicy(
        artifact_class=ArtifactClass.GENERATED,
        retention=ArtifactRetentionPolicy.GENERATED,
        default_storage=ArtifactStorageLocation.GENERATED_ROOT,
        release_critical=False,
    ),
    ArtifactClass.SURROGATE: ArtifactClassPolicy(
        artifact_class=ArtifactClass.SURROGATE,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.SURROGATE_CHECKPOINT: ArtifactClassPolicy(
        artifact_class=ArtifactClass.SURROGATE_CHECKPOINT,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.POSTERIOR: ArtifactClassPolicy(
        artifact_class=ArtifactClass.POSTERIOR,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.POSTERIOR_SAMPLE: ArtifactClassPolicy(
        artifact_class=ArtifactClass.POSTERIOR_SAMPLE,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.CONFIG: ArtifactClassPolicy(
        artifact_class=ArtifactClass.CONFIG,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.REPORT: ArtifactClassPolicy(
        artifact_class=ArtifactClass.REPORT,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.TRAINING_MANIFEST: ArtifactClassPolicy(
        artifact_class=ArtifactClass.TRAINING_MANIFEST,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.GENERATED_ROOT,
        release_critical=True,
    ),
    ArtifactClass.RUNTIME_MANIFEST: ArtifactClassPolicy(
        artifact_class=ArtifactClass.RUNTIME_MANIFEST,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.GENERATED_ROOT,
        release_critical=True,
    ),
    ArtifactClass.RUN_MANIFEST: ArtifactClassPolicy(
        artifact_class=ArtifactClass.RUN_MANIFEST,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.GENERATED_ROOT,
        release_critical=True,
    ),
    ArtifactClass.METADATA: ArtifactClassPolicy(
        artifact_class=ArtifactClass.METADATA,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.FIGURE: ArtifactClassPolicy(
        artifact_class=ArtifactClass.FIGURE,
        retention=ArtifactRetentionPolicy.CURATED,
        default_storage=ArtifactStorageLocation.SOURCE_TREE,
        release_critical=True,
    ),
    ArtifactClass.LOG: ArtifactClassPolicy(
        artifact_class=ArtifactClass.LOG,
        retention=ArtifactRetentionPolicy.GENERATED,
        default_storage=ArtifactStorageLocation.HPC_OUTPUT,
        release_critical=False,
    ),
    ArtifactClass.SIMULATION_OUTPUT: ArtifactClassPolicy(
        artifact_class=ArtifactClass.SIMULATION_OUTPUT,
        retention=ArtifactRetentionPolicy.GENERATED,
        default_storage=ArtifactStorageLocation.HPC_OUTPUT,
        release_critical=False,
    ),
}


def class_policy_for_artifact_class(
    value: ArtifactClass | str,
) -> ArtifactClassPolicy:
    return _ARTIFACT_CLASS_POLICIES[coerce_artifact_class(value)]


def is_source_tree_generated_root(path_value: str) -> bool:
    normalized = path_value.replace("\\", "/").strip()
    normalized = re.sub(r"^\$\{[^}]+\}/?", "", normalized).strip("/")
    if not normalized:
        return False

    root_segment = normalized.split("/")[0]
    return (
        root_segment in SOURCE_TREE_GENERATED_ROOTS
        or bool(re.fullmatch(r"_init_compression_[^/]+", root_segment))
    )


@dataclass(frozen=True)
class ArtifactManifestRecord:
    artifact_id: str
    artifact_class: ArtifactClass
    path: str
    storage_location: ArtifactStorageLocation
    retention_policy: ArtifactRetentionPolicy
    release_critical: bool
    checksum: ChecksumMetadata | None = None
    provenance: ArtifactProvenance | None = None
    validation_status: ArtifactValidationStatus = ArtifactValidationStatus.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifestRecord":
        artifact_class = coerce_artifact_class(payload["artifact_class"])
        policy = class_policy_for_artifact_class(artifact_class)
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_class=artifact_class,
            path=str(payload["path"]),
            storage_location=_coerce_storage_location(
                payload.get("storage_location"),
                default=policy.default_storage,
            ),
            retention_policy=_coerce_retention_policy(
                payload.get("retention_policy"),
                default=policy.retention,
            ),
            release_critical=_coerce_release_critical(
                payload.get("release_critical"),
                default=policy.release_critical,
            ),
            checksum=(
                ChecksumMetadata.from_dict(payload["checksum"]) if payload.get("checksum") is not None else None
            ),
            provenance=(
                ArtifactProvenance.from_dict(payload["provenance"])
                if payload.get("provenance") is not None
                else None
            ),
            validation_status=_coerce_validation_status(payload.get("validation_status")),
            metadata={str(key): value for key, value in payload.get("metadata", {}).items()},
        )


@dataclass(frozen=True)
class ArtifactManifestCleanupPolicy:
    generated_roots: tuple[str, ...] = ()
    protected_roots: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifestCleanupPolicy":
        return cls(
            generated_roots=tuple(str(path) for path in payload.get("generated_roots", ())),
            protected_roots=tuple(str(path) for path in payload.get("protected_roots", ())),
        )


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    manifest_id: str
    generated_at: str
    cleanup_policy: ArtifactManifestCleanupPolicy
    artifacts: tuple[ArtifactManifestRecord, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            manifest_id=str(payload["manifest_id"]),
            generated_at=str(payload["generated_at"]),
            cleanup_policy=ArtifactManifestCleanupPolicy.from_dict(payload.get("cleanup_policy", {})),
            artifacts=tuple(ArtifactManifestRecord.from_dict(item) for item in payload["artifacts"]),
        )


def validate_artifact_manifest_file(path: str | Path) -> list[str]:
    document_path = Path(path)
    return validate_artifact_manifest_document(
        load_structured_document(document_path),
        source=document_path,
    )


def validate_artifact_manifest_document(
    document: Any,
    *,
    source: str | Path | None = None,
) -> list[str]:
    label = str(source) if source is not None else "<artifact-manifest>"
    errors: list[str] = []

    if not isinstance(document, dict):
        return [f"{label}: manifest must be a mapping."]

    _require_string(document.get("schema_version"), f"{label}: schema_version", errors)
    _require_string(document.get("manifest_id"), f"{label}: manifest_id", errors)
    _require_string(document.get("generated_at"), f"{label}: generated_at", errors)

    cleanup_policy = document.get("cleanup_policy")
    if not isinstance(cleanup_policy, dict):
        errors.append(f"{label}: cleanup_policy must be a mapping.")
    else:
        generated_roots = cleanup_policy.get("generated_roots")
        if generated_roots is not None:
            _validate_roots(
                generated_roots,
                f"{label}: cleanup_policy.generated_roots",
                errors,
            )
        protected_roots = cleanup_policy.get("protected_roots")
        if not isinstance(protected_roots, list) or not protected_roots:
            errors.append(f"{label}: cleanup_policy.protected_roots must be a non-empty list.")
        else:
            for protected_root in PROTECTED_VENDOR_ROOTS:
                if protected_root not in protected_roots:
                    errors.append(
                        f"{label}: cleanup_policy.protected_roots must include '{protected_root}'."
                    )

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{label}: artifacts must be a non-empty list.")
        return errors

    for index, artifact in enumerate(artifacts):
        item_label = f"{label}: artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{item_label} must be a mapping.")
            continue

        _require_string(artifact.get("artifact_id"), f"{item_label}.artifact_id", errors)
        artifact_class = artifact.get("artifact_class")
        if not isinstance(artifact_class, str):
            errors.append(
                f"{item_label}.artifact_class must be one of {sorted(SUPPORTED_ARTIFACT_CLASSES)}, got {artifact_class!r}."
            )
            continue
        if artifact_class not in SUPPORTED_ARTIFACT_CLASSES:
            errors.append(
                f"{item_label}.artifact_class must be one of {sorted(SUPPORTED_ARTIFACT_CLASSES)}, got {artifact_class!r}."
            )
            continue

        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            errors.append(f"{item_label}.path must be a non-empty string.")
        else:
            if any(fragment in path_value for fragment in FORBIDDEN_PRIVATE_PATHS):
                errors.append(f"{item_label}.path contains a forbidden private path literal.")
            if path_value.startswith("/"):
                errors.append(f"{item_label}.path must be relative or placeholder-based, got '{path_value}'.")
            release_critical = artifact.get("release_critical")
            if "release_critical" in artifact and not isinstance(release_critical, bool):
                errors.append(f"{item_label}.release_critical must be a boolean.")
            policy = class_policy_for_artifact_class(artifact_class)
            release_critical_value = (
                release_critical if isinstance(release_critical, bool) else policy.release_critical
            )
            if is_source_tree_generated_root(path_value) and not release_critical_value:
                errors.append(
                    f"{item_label}.path points into a source-tree generated root and requires release_critical provenance."
                )

            storage_location = artifact.get("storage_location")
            if storage_location is not None:
                try:
                    ArtifactStorageLocation(storage_location)
                except ValueError:
                    accepted = ", ".join(sorted(item.value for item in ArtifactStorageLocation))
                    errors.append(
                        f"{item_label}.storage_location must be one of {{{accepted}}}, got {storage_location!r}."
                    )
            retention_policy = artifact.get("retention_policy")
            if retention_policy is not None:
                try:
                    ArtifactRetentionPolicy(retention_policy)
                except ValueError:
                    accepted = ", ".join(sorted(item.value for item in ArtifactRetentionPolicy))
                    errors.append(
                        f"{item_label}.retention_policy must be one of {{{accepted}}}, got {retention_policy!r}."
                    )
            validation_status = artifact.get("validation_status")
            if validation_status is not None:
                try:
                    ArtifactValidationStatus(validation_status)
                except ValueError:
                    accepted = ", ".join(sorted(item.value for item in ArtifactValidationStatus))
                    errors.append(
                        f"{item_label}.validation_status must be one of {{{accepted}}}, got {validation_status!r}."
                    )
            checksum = artifact.get("checksum")
            if checksum is not None:
                if not isinstance(checksum, dict):
                    errors.append(f"{item_label}.checksum must be a mapping.")
                else:
                    _validate_checksum(checksum, item_label, errors)
            provenance = artifact.get("provenance")
            if provenance is not None:
                if not isinstance(provenance, dict):
                    errors.append(f"{item_label}.provenance must be a mapping.")
                elif not isinstance(provenance.get("metadata", {}), dict):
                    errors.append(f"{item_label}.provenance.metadata must be a mapping.")

            metadata = artifact.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                errors.append(f"{item_label}.metadata must be a mapping.")

    return errors


def _require_string(value: Any, field_label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_label} must be a non-empty string.")


def _validate_roots(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list.")
        return

    for item in value:
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label} entries must be non-empty strings.")


def _validate_checksum(
    payload: Mapping[str, Any],
    item_label: str,
    errors: list[str],
) -> None:
    algorithm = payload.get("algorithm")
    value = payload.get("value")
    if not isinstance(algorithm, str) or not algorithm.strip():
        errors.append(f"{item_label}.checksum.algorithm must be a non-empty string.")
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{item_label}.checksum.value must be a non-empty string.")


def _coerce_retention_policy(
    value: Any,
    *,
    default: ArtifactRetentionPolicy,
) -> ArtifactRetentionPolicy:
    if value is None:
        return default
    if isinstance(value, ArtifactRetentionPolicy):
        return value
    return ArtifactRetentionPolicy(str(value))


def _coerce_storage_location(
    value: Any,
    *,
    default: ArtifactStorageLocation | None = None,
) -> ArtifactStorageLocation:
    if value is None:
        if default is None:
            return ArtifactStorageLocation.SOURCE_TREE
        return default
    if isinstance(value, ArtifactStorageLocation):
        return value
    return ArtifactStorageLocation(str(value))


def _coerce_release_critical(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError("release_critical must be a boolean.")
    return value


def _coerce_validation_status(value: Any) -> ArtifactValidationStatus:
    if value is None:
        return ArtifactValidationStatus.UNKNOWN
    if isinstance(value, ArtifactValidationStatus):
        return value
    return ArtifactValidationStatus(str(value))
