from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.artifacts.policy import validate_artifact_manifest_file
from meso_uq.configs.policy import load_structured_document


DEFAULT_ALLOWLIST_PATH = REPO_ROOT / "configs" / "governance_allowlist.example.json"
CONFIG_ROOT = REPO_ROOT / "configs"
DOCS_ROOT = REPO_ROOT / "docs"
WORKFLOWS_ROOT = REPO_ROOT / ".github" / "workflows"
GENERATED_ROOT_PREFIXES = ("_out", "_runs", "_ci", "out_hierarchical")
GENERATED_ROOT_PATTERN = re.compile(r"^_init_compression_[^/]+$")
RETIRED_MARKER_PATTERN = re.compile(r"\b(deprecated|retired)\b", re.IGNORECASE)


def load_governance_allowlist(path: str | Path = DEFAULT_ALLOWLIST_PATH) -> dict[str, Any]:
    document_path = Path(path)
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    errors = validate_governance_allowlist_document(payload, source=document_path)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_governance_allowlist_document(
    document: Any,
    *,
    source: str | Path | None = None,
) -> list[str]:
    label = str(source) if source is not None else "<governance-allowlist>"
    errors: list[str] = []

    if not isinstance(document, dict):
        return [f"{label}: governance allowlist must be a mapping."]

    _require_non_empty_string(document.get("schema_version"), f"{label}: schema_version", errors)
    _require_non_empty_string(document.get("manifest_id"), f"{label}: manifest_id", errors)
    _require_non_empty_string(document.get("generated_at"), f"{label}: generated_at", errors)

    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label}: entries must be a non-empty list.")
        return errors

    for index, entry in enumerate(entries):
        entry_label = f"{label}: entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} must be a mapping.")
            continue

        _require_non_empty_string(entry.get("check"), f"{entry_label}.check", errors)
        _require_non_empty_string(entry.get("owner"), f"{entry_label}.owner", errors)
        _require_non_empty_string(entry.get("reason"), f"{entry_label}.reason", errors)
        _require_non_empty_string(
            entry.get("review_condition"), f"{entry_label}.review_condition", errors
        )

        check = entry.get("check")
        if check == "public_api_exports":
            _require_non_empty_string_list(
                entry.get("expected_exports"),
                f"{entry_label}.expected_exports",
                errors,
            )
        elif check in {"ci_path_references", "docs_retired_markers"}:
            _validate_file_fragment_entries(
                entry.get("files"),
                f"{entry_label}.files",
                errors,
            )
        elif check == "import_cycles":
            _validate_import_cycle_entries(
                entry.get("sequences"),
                f"{entry_label}.sequences",
                errors,
            )
        else:
            errors.append(
                f"{entry_label}.check must be one of "
                "['ci_path_references', 'docs_retired_markers', 'import_cycles', 'public_api_exports']."
            )

    return errors


def validate_config_schema_versions(repo_root: str | Path = REPO_ROOT) -> list[str]:
    repo_root = Path(repo_root)
    errors: list[str] = []

    for path in _structured_config_paths(repo_root):
        relative = path.relative_to(repo_root)
        try:
            document = load_structured_document(path)
        except Exception as exc:  # pragma: no cover - exercised through error paths
            errors.append(
                f"{relative}: could not parse structured config. Remediation: fix the document syntax. "
                f"Details: {exc}"
            )
            continue

        if not isinstance(document, dict):
            errors.append(
                f"{relative}: config must be a mapping with a schema_version field. "
                "Remediation: convert the document to a mapping and add schema_version: '1.0'."
            )
            continue

        schema_version = document.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            errors.append(
                f"{relative}: missing schema_version. "
                "Remediation: add schema_version: '1.0' at the document root."
            )

    return errors


def validate_artifact_manifest_example(repo_root: str | Path = REPO_ROOT) -> list[str]:
    repo_root = Path(repo_root)
    manifest_path = repo_root / "configs" / "artifacts" / "artifact_manifest.example.json"
    return validate_artifact_manifest_file(manifest_path)


def validate_tracked_generated_roots(repo_root: str | Path = REPO_ROOT) -> list[str]:
    repo_root = Path(repo_root)
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    errors: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        path = Path(raw_path)
        if _is_tracked_generated_root(path):
            errors.append(
                f"{path.as_posix()}: tracked generated-root path. "
                "Remediation: move the file under an ignored runtime directory or remove it from git."
            )

    return errors


def validate_public_api_exports(
    allowlist: Mapping[str, Any],
) -> list[str]:
    from meso_uq import public_api

    entry = _single_allowlist_entry(allowlist, "public_api_exports")
    if entry is None:
        return ["<governance-allowlist>: missing public_api_exports allowlist entry."]

    expected_exports = tuple(entry["expected_exports"])
    actual_exports = tuple(public_api.__all__)

    missing = sorted(set(expected_exports) - set(actual_exports))
    extra = sorted(set(actual_exports) - set(expected_exports))
    if not missing and not extra:
        return []

    details = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"extra {extra}")

    return [
        "meso_uq.public_api.__all__ does not match the governance allowlist: "
        + ", ".join(details)
        + ". Remediation: update configs/governance_allowlist.example.json or the public API export list."
    ]


def validate_ci_path_references(
    repo_root: str | Path = REPO_ROOT,
    allowlist: Mapping[str, Any] | None = None,
) -> list[str]:
    repo_root = Path(repo_root)
    allowlist = allowlist or load_governance_allowlist(repo_root / "configs" / "governance_allowlist.example.json")
    allowed = _file_fragment_allowlist(allowlist, "ci_path_references")
    errors: list[str] = []

    workflows_root = repo_root / ".github" / "workflows"
    for path in sorted(workflows_root.glob("*.yml")) + sorted(workflows_root.glob("*.yaml")):
        relative = path.relative_to(repo_root)
        fragments = allowed.get(relative.as_posix(), set())
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "_ci/" not in line:
                continue
            references = sorted(set(_CI_PATH_RE.findall(line)))
            if not references:
                continue
            unexpected = [reference for reference in references if reference not in fragments]
            if unexpected:
                errors.append(
                    f"{relative}:{line_no}: CI path reference {unexpected} is not allowlisted. "
                    "Remediation: either avoid repo-tree CI artifact paths or add an allowlist entry "
                    "with owner/reason/review_condition."
                )

    return errors


def validate_docs_retired_command_markers(
    repo_root: str | Path = REPO_ROOT,
    allowlist: Mapping[str, Any] | None = None,
) -> list[str]:
    repo_root = Path(repo_root)
    allowlist = allowlist or load_governance_allowlist(repo_root / "configs" / "governance_allowlist.example.json")
    allowed = _file_fragment_allowlist(allowlist, "docs_retired_markers")
    errors: list[str] = []

    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.md")):
        relative = path.relative_to(repo_root)
        fragments = allowed.get(relative.as_posix())
        if fragments is None:
            fragments = set()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not RETIRED_MARKER_PATTERN.search(line):
                continue
            if not any(fragment in line for fragment in fragments):
                errors.append(
                    f"{relative}:{line_no}: retired/deprecated marker is not allowlisted. "
                    "Remediation: either remove the retired-command text or add a docs allowlist entry "
                    "with owner/reason/review_condition."
                )

    return errors


def validate_import_cycle_allowlist_structure(allowlist: Mapping[str, Any]) -> list[str]:
    entry = _single_allowlist_entry(allowlist, "import_cycles")
    if entry is None:
        return ["<governance-allowlist>: missing import_cycles allowlist entry."]

    errors: list[str] = []
    _validate_import_cycle_entries(entry.get("sequences"), "<allowlist>.sequences", errors)
    return errors


def validate_repo_structural_governance_policy(
    repo_root: str | Path = REPO_ROOT,
    *,
    allowlist_path: str | Path = DEFAULT_ALLOWLIST_PATH,
) -> list[str]:
    repo_root = Path(repo_root)
    allowlist = load_governance_allowlist(allowlist_path)

    errors: list[str] = []
    errors.extend(validate_config_schema_versions(repo_root))
    errors.extend(validate_artifact_manifest_example(repo_root))
    errors.extend(validate_tracked_generated_roots(repo_root))
    errors.extend(validate_public_api_exports(allowlist))
    errors.extend(validate_ci_path_references(repo_root, allowlist))
    errors.extend(validate_docs_retired_command_markers(repo_root, allowlist))
    errors.extend(validate_import_cycle_allowlist_structure(allowlist))
    return errors


def _structured_config_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(repo_root.joinpath("configs").rglob("*")):
        if not path.is_file():
            continue
        if path.name == "governance_allowlist.example.json":
            paths.append(path)
            continue
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            paths.append(path)
    return paths


def _is_tracked_generated_root(path: Path) -> bool:
    if not path.parts:
        return False

    root = path.parts[0]
    if root in GENERATED_ROOT_PREFIXES:
        return True
    if GENERATED_ROOT_PATTERN.fullmatch(root):
        return True
    if len(path.parts) == 1 and path.suffix in {".out", ".err"}:
        return True
    return False


def _single_allowlist_entry(
    allowlist: Mapping[str, Any],
    check_name: str,
) -> Mapping[str, Any] | None:
    entries = allowlist.get("entries")
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("check") == check_name:
            return entry
    return None


def _file_fragment_allowlist(
    allowlist: Mapping[str, Any],
    check_name: str,
) -> dict[str, set[str]]:
    entry = _single_allowlist_entry(allowlist, check_name)
    if entry is None:
        return {}

    files = entry.get("files")
    if not isinstance(files, list):
        return {}

    mapping: dict[str, set[str]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        fragments = item.get("fragments")
        if not isinstance(path, str) or not isinstance(fragments, list):
            continue
        mapping[path] = {str(fragment) for fragment in fragments if str(fragment).strip()}
    return mapping


def _require_non_empty_string(value: Any, field_label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_label} must be a non-empty string.")


def _require_non_empty_string_list(value: Any, field_label: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field_label} must be a non-empty list of strings.")
        return

    for item in value:
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{field_label} must contain only non-empty strings.")
            return


def _validate_file_fragment_entries(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list.")
        return

    for index, entry in enumerate(value):
        entry_label = f"{label}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{entry_label} must be a mapping.")
            continue
        _require_non_empty_string(entry.get("path"), f"{entry_label}.path", errors)
        _require_non_empty_string_list(entry.get("fragments"), f"{entry_label}.fragments", errors)


def _validate_import_cycle_entries(value: Any, label: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list of import sequences.")
        return errors

    for index, sequence in enumerate(value):
        sequence_label = f"{label}[{index}]"
        if not isinstance(sequence, list) or len(sequence) < 2:
            errors.append(
                f"{sequence_label} must be a list with at least two module names. "
                "Remediation: spell out the import order explicitly."
            )
            continue
        for module_index, module_name in enumerate(sequence):
            if not isinstance(module_name, str) or not module_name.strip():
                errors.append(
                    f"{sequence_label}[{module_index}] must be a non-empty module name."
                )
    return errors


_CI_PATH_RE = re.compile(r"(_ci/[A-Za-z0-9_./-]+)")
