from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_CONFIG_KINDS = frozenset(
    {
        "active_learning",
        "agent",
        "dataset",
        "inference",
        "modality",
        "noise",
        "platform",
        "report",
        "surrogate",
    }
)
SUPPORTED_PLATFORM_KEYS = frozenset({"generic_slurm", "karolina", "vega", "workstation"})
FORBIDDEN_PRIVATE_PATHS = frozenset({"/ceph/hpc/home/eubrieucb"})
_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


def load_structured_document(path: str | Path) -> Any:
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    text = document_path.read_text(encoding="utf-8")

    if suffix in _YAML_SUFFIXES:
        return yaml.safe_load(text) or {}
    if suffix in _JSON_SUFFIXES:
        return json.loads(text)

    raise ValueError(f"Unsupported config format for '{document_path}'. Expected YAML or JSON.")


def validate_config_file(path: str | Path) -> list[str]:
    document_path = Path(path)
    return validate_config_document(load_structured_document(document_path), source=document_path)


def validate_config_document(document: Any, *, source: str | Path | None = None) -> list[str]:
    label = str(source) if source is not None else "<config>"
    errors: list[str] = []

    if not isinstance(document, dict):
        return [f"{label}: config must be a mapping."]

    _require_non_empty_string(document.get("schema_version"), f"{label}: schema_version", errors)

    kind = document.get("kind")
    if not isinstance(kind, str) or kind not in SUPPORTED_CONFIG_KINDS:
        errors.append(
            f"{label}: kind must be one of {sorted(SUPPORTED_CONFIG_KINDS)}, got {kind!r}."
        )

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"{label}: metadata must be a mapping with id and name fields.")
    else:
        _require_non_empty_string(metadata.get("id"), f"{label}: metadata.id", errors)
        _require_non_empty_string(metadata.get("name"), f"{label}: metadata.name", errors)

    if kind == "platform":
        _validate_platform_document(document, label, errors)

    for trail, value in _iter_strings(document):
        _validate_string_value(label, trail, value, errors)

    return errors


def _validate_platform_document(document: dict[str, Any], label: str, errors: list[str]) -> None:
    platform_key = document.get("platform_key")
    if not isinstance(platform_key, str) or platform_key not in SUPPORTED_PLATFORM_KEYS:
        errors.append(
            f"{label}: platform_key must be one of {sorted(SUPPORTED_PLATFORM_KEYS)}, got {platform_key!r}."
        )

    path_policy = document.get("path_policy")
    if not isinstance(path_policy, dict):
        errors.append(f"{label}: platform configs must define a path_policy mapping.")
        return

    if path_policy.get("allow_absolute_paths") is not False:
        errors.append(f"{label}: path_policy.allow_absolute_paths must be false.")

    required_placeholders = path_policy.get("required_placeholders")
    if not isinstance(required_placeholders, list) or not required_placeholders:
        errors.append(f"{label}: path_policy.required_placeholders must be a non-empty list.")
    else:
        for placeholder in required_placeholders:
            if not isinstance(placeholder, str) or "${" not in placeholder:
                errors.append(
                    f"{label}: placeholder entry {placeholder!r} must reference an environment placeholder."
                )

    if platform_key == "karolina":
        forbidden_literals = path_policy.get("forbidden_literals")
        if not isinstance(forbidden_literals, list) or "/ceph/hpc/home/eubrieucb" not in forbidden_literals:
            errors.append(
                f"{label}: Karolina policy must explicitly forbid '/ceph/hpc/home/eubrieucb'."
            )


def _require_non_empty_string(value: Any, field_label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_label} must be a non-empty string.")


def _iter_strings(value: Any, trail: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    items: list[tuple[tuple[Any, ...], str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            items.extend(_iter_strings(nested, trail + (key,)))
        return items
    if isinstance(value, list):
        for index, nested in enumerate(value):
            items.extend(_iter_strings(nested, trail + (index,)))
        return items
    if isinstance(value, str):
        return [(trail, value)]
    return items


def _validate_string_value(
    label: str,
    trail: tuple[Any, ...],
    value: str,
    errors: list[str],
) -> None:
    is_forbidden_slot = (
        len(trail) >= 2
        and trail[-2] in {"forbidden_literals", "forbidden_path_prefixes"}
    )
    joined_trail = ".".join(str(part) for part in trail)

    if any(fragment in value for fragment in FORBIDDEN_PRIVATE_PATHS) and not is_forbidden_slot:
        errors.append(f"{label}: {joined_trail} contains a forbidden private path literal.")

    if value.startswith("/") and not is_forbidden_slot:
        errors.append(
            f"{label}: {joined_trail} uses absolute path '{value}'. Use env placeholders instead."
        )
