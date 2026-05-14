from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS


EXAMPLE_CONFIG_ROOT = Path("examples/configs")
ALLOWED_EXAMPLE_CONFIG_SUFFIXES = frozenset({".yaml", ".yml"})


class ExampleConfigStatus(str, Enum):
    MAINTAINED = "maintained"
    ARCHIVE_CANDIDATE = "archive_candidate"
    DUPLICATE = "duplicate"
    NEEDS_REFRESH = "needs_refresh"
    OWNER_DECISION = "owner_decision"


@dataclass(frozen=True)
class ExampleConfigInventoryEntry:
    path: str
    canonical_path: str | None = None


@dataclass(frozen=True)
class ExampleConfigManifestEntry:
    path: str
    status: ExampleConfigStatus
    reason: str
    validation_notes: tuple[str, ...] = ()
    canonical_path: str | None = None


@dataclass(frozen=True)
class ExampleConfigManifest:
    schema_version: str
    manifest_id: str
    generated_at: str
    examples: tuple[ExampleConfigManifestEntry, ...]


_CURRENT_EXAMPLE_CONFIGS: tuple[ExampleConfigInventoryEntry, ...] = (
    ExampleConfigInventoryEntry(
        path="examples/configs/compression_full.yaml",
        canonical_path="inference/configs/production/inference_config_compression.yaml",
    ),
    ExampleConfigInventoryEntry(
        path="examples/configs/compression_reduced.yaml",
        canonical_path="reduced/configs/production/reduced_config_compression.yaml",
    ),
    ExampleConfigInventoryEntry(
        path="examples/configs/indentation_full.yaml",
        canonical_path="inference/configs/production/inference_config_indentation.yaml",
    ),
    ExampleConfigInventoryEntry(
        path="examples/configs/indentation_reduced.yaml",
        canonical_path="reduced/configs/production/reduced_config_indentation.yaml",
    ),
)


def current_example_config_inventory(repo_root: str | Path) -> tuple[ExampleConfigManifestEntry, ...]:
    repo_root_path = Path(repo_root)
    seen_fingerprints: dict[str, str] = {}
    return tuple(
        _classify_example_config_entry(
            entry,
            repo_root=repo_root_path,
            seen_fingerprints=seen_fingerprints,
        )
        for entry in _CURRENT_EXAMPLE_CONFIGS
    )


def classify_example_config_record(
    record: ExampleConfigInventoryEntry,
    *,
    repo_root: str | Path,
) -> ExampleConfigManifestEntry:
    return _classify_example_config_entry(record, repo_root=Path(repo_root), seen_fingerprints={})


def load_example_config_manifest(path: str | Path) -> ExampleConfigManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = tuple(
        ExampleConfigManifestEntry(
            path=str(item["path"]),
            canonical_path=item.get("canonical_path"),
            status=ExampleConfigStatus(item["status"]),
            reason=str(item["reason"]),
            validation_notes=tuple(str(note) for note in item.get("validation_notes", ())),
        )
        for item in payload.get("examples", [])
    )
    return ExampleConfigManifest(
        schema_version=str(payload["schema_version"]),
        manifest_id=str(payload["manifest_id"]),
        generated_at=str(payload["generated_at"]),
        examples=examples,
    )


def validate_example_config_manifest(
    manifest: ExampleConfigManifest,
    *,
    repo_root: str | Path,
) -> list[str]:
    repo_root_path = Path(repo_root)
    errors: list[str] = []
    seen_fingerprints: dict[str, str] = {}

    _require_non_empty_string(manifest.schema_version, "manifest.schema_version", errors)
    _require_non_empty_string(manifest.manifest_id, "manifest.manifest_id", errors)
    _require_non_empty_string(manifest.generated_at, "manifest.generated_at", errors)

    for entry in manifest.examples:
        errors.extend(
            validate_example_config_manifest_entry(
                entry,
                repo_root=repo_root_path,
                seen_fingerprints=seen_fingerprints,
            )
        )

    return errors


def validate_example_config_manifest_entry(
    entry: ExampleConfigManifestEntry,
    *,
    repo_root: str | Path,
    seen_fingerprints: dict[str, str] | None = None,
) -> list[str]:
    repo_root_path = Path(repo_root)
    errors = _validate_example_config_path(entry.path, repo_root_path)
    if errors:
        return errors

    resolved_path = _resolve_repo_path(repo_root_path, entry.path)
    if not resolved_path.exists():
        return [f"{entry.path}: example config file not found."]

    try:
        document = _load_structured_yaml(resolved_path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted fixtures.
        return [f"{entry.path}: unable to load YAML: {exc}"]

    notes = tuple(entry.validation_notes)
    if entry.status is ExampleConfigStatus.MAINTAINED:
        canonical_errors = _validate_canonical_match(entry, repo_root_path, document)
        errors.extend(canonical_errors)
    elif entry.canonical_path is not None:
        canonical_path = _resolve_repo_path(repo_root_path, entry.canonical_path)
        if canonical_path.exists():
            canonical_document = _load_structured_yaml(canonical_path)
            if _document_fingerprint(document) == _document_fingerprint(canonical_document):
                errors.append(
                    f"{entry.path}: status {entry.status.value} conflicts with an unchanged canonical copy."
                )

    for value in _iter_strings(document):
        if any(fragment in value for fragment in FORBIDDEN_PRIVATE_PATHS):
            errors.append(f"{entry.path}: contains forbidden private path literal.")

    fingerprint = _document_fingerprint(document)
    if seen_fingerprints is not None:
        prior_path = seen_fingerprints.get(fingerprint)
        if prior_path is not None and prior_path != entry.path:
            if entry.status is not ExampleConfigStatus.DUPLICATE:
                errors.append(f"{entry.path}: duplicates {prior_path} but is not marked duplicate.")
        else:
            seen_fingerprints[fingerprint] = entry.path

    if entry.status is ExampleConfigStatus.OWNER_DECISION and not notes:
        errors.append(f"{entry.path}: owner decision entries should carry validation notes.")

    return errors


def _classify_example_config_entry(
    record: ExampleConfigInventoryEntry,
    *,
    repo_root: Path,
    seen_fingerprints: dict[str, str],
) -> ExampleConfigManifestEntry:
    path_errors = _validate_example_config_path(record.path, repo_root)
    if path_errors:
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=record.canonical_path,
            status=ExampleConfigStatus.OWNER_DECISION,
            reason="; ".join(path_errors),
            validation_notes=tuple(path_errors),
        )

    resolved_path = _resolve_repo_path(repo_root, record.path)
    if not resolved_path.exists():
        note = f"{record.path}: missing from repository checkout."
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=record.canonical_path,
            status=ExampleConfigStatus.OWNER_DECISION,
            reason=note,
            validation_notes=(note,),
        )

    document = _load_structured_yaml(resolved_path)
    notes = [
        "YAML parses as a structured document",
        "path is relative under examples/configs",
        "no private path literals were found",
    ]
    if record.canonical_path is None:
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=None,
            status=ExampleConfigStatus.OWNER_DECISION,
            reason="No canonical counterpart supplied for comparison.",
            validation_notes=tuple(notes),
        )

    canonical_path = _resolve_repo_path(repo_root, record.canonical_path)
    if not canonical_path.exists():
        notes.append(f"canonical counterpart missing: {record.canonical_path}")
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=record.canonical_path,
            status=ExampleConfigStatus.OWNER_DECISION,
            reason=f"Canonical counterpart is missing: {record.canonical_path}",
            validation_notes=tuple(notes),
        )

    canonical_document = _load_structured_yaml(canonical_path)
    fingerprint = _document_fingerprint(document)
    prior_path = seen_fingerprints.get(fingerprint)
    if prior_path is not None and prior_path != record.path:
        notes.append(f"duplicates {prior_path}")
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=record.canonical_path,
            status=ExampleConfigStatus.DUPLICATE,
            reason=f"Duplicate structured payload of {prior_path}.",
            validation_notes=tuple(notes),
        )

    seen_fingerprints[fingerprint] = record.path

    if _document_fingerprint(document) == _document_fingerprint(canonical_document):
        notes.append("matches the canonical workflow config")
        return ExampleConfigManifestEntry(
            path=record.path,
            canonical_path=record.canonical_path,
            status=ExampleConfigStatus.MAINTAINED,
            reason="Maintained copy matches the canonical workflow config.",
            validation_notes=tuple(notes),
        )

    notes.append(f"differs from {record.canonical_path}")
    return ExampleConfigManifestEntry(
        path=record.path,
        canonical_path=record.canonical_path,
        status=ExampleConfigStatus.NEEDS_REFRESH,
        reason=(
            "Convenience copy diverges from the canonical workflow config and should be refreshed."
        ),
        validation_notes=tuple(notes),
    )


def _validate_canonical_match(
    entry: ExampleConfigManifestEntry,
    repo_root: Path,
    example_document: Any,
) -> list[str]:
    if entry.canonical_path is None:
        return [f"{entry.path}: maintained entries require canonical_path."]
    canonical_path = _resolve_repo_path(repo_root, entry.canonical_path)
    if not canonical_path.exists():
        return [f"{entry.path}: canonical counterpart missing at {entry.canonical_path}."]

    canonical_document = _load_structured_yaml(canonical_path)
    if _document_fingerprint(example_document) != _document_fingerprint(canonical_document):
        return [f"{entry.path}: maintained entry does not match {entry.canonical_path}."]
    return []


def _validate_example_config_path(path: str, repo_root: Path) -> list[str]:
    errors: list[str] = []
    candidate = Path(path)

    if candidate.is_absolute():
        errors.append(f"{path}: path must be repository-relative.")
    if ".." in candidate.parts:
        errors.append(f"{path}: path traversal is not allowed.")
    if candidate.parts[: len(EXAMPLE_CONFIG_ROOT.parts)] != EXAMPLE_CONFIG_ROOT.parts:
        errors.append(f"{path}: path must live under examples/configs.")
    if candidate.suffix.lower() not in ALLOWED_EXAMPLE_CONFIG_SUFFIXES:
        errors.append(
            f"{path}: expected a YAML suffix from {sorted(ALLOWED_EXAMPLE_CONFIG_SUFFIXES)}."
        )

    resolved = _resolve_repo_path(repo_root, path)
    if resolved.exists():
        try:
            document = _load_structured_yaml(resolved)
        except Exception as exc:  # pragma: no cover - defensive for corrupted fixtures.
            errors.append(f"{path}: YAML could not be loaded: {exc}")
        else:
            for value in _iter_strings(document):
                if any(fragment in value for fragment in FORBIDDEN_PRIVATE_PATHS):
                    errors.append(f"{path}: contains forbidden private path literal.")
    return errors


def _load_structured_yaml(path: Path) -> Any:
    from meso_uq.config.loader import load_yaml

    return load_yaml(path)


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    return repo_root / Path(relative_path)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_strings(nested)
    elif isinstance(value, str):
        yield value


def _document_fingerprint(document: Any) -> str:
    normalized = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(normalized.encode("utf-8")).hexdigest()


def _require_non_empty_string(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string.")


__all__ = [
    "ALLOWED_EXAMPLE_CONFIG_SUFFIXES",
    "EXAMPLE_CONFIG_ROOT",
    "ExampleConfigInventoryEntry",
    "ExampleConfigManifest",
    "ExampleConfigManifestEntry",
    "ExampleConfigStatus",
    "classify_example_config_record",
    "current_example_config_inventory",
    "load_example_config_manifest",
    "validate_example_config_manifest",
    "validate_example_config_manifest_entry",
]
