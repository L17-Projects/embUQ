from __future__ import annotations

from pathlib import Path
from typing import Any

from meso_uq.core import ArtifactClass
from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS, load_structured_document

SUPPORTED_ARTIFACT_CLASSES = frozenset(item.value for item in ArtifactClass)
PROTECTED_VENDOR_ROOTS = ("extern/korali",)


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
        if not isinstance(artifact_class, str) or artifact_class not in SUPPORTED_ARTIFACT_CLASSES:
            errors.append(
                f"{item_label}.artifact_class must be one of {sorted(SUPPORTED_ARTIFACT_CLASSES)}, got {artifact_class!r}."
            )

        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            errors.append(f"{item_label}.path must be a non-empty string.")
        else:
            if any(fragment in path_value for fragment in FORBIDDEN_PRIVATE_PATHS):
                errors.append(f"{item_label}.path contains a forbidden private path literal.")
            if path_value.startswith("/"):
                errors.append(f"{item_label}.path must be relative or placeholder-based, got '{path_value}'.")

    return errors


def _require_string(value: Any, field_label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_label} must be a non-empty string.")
