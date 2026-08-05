#!/usr/bin/env python3
"""Plan, stage, and verify immutable external UQ_EMB artifact sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
PAPER_ID = "UQ_EMB"
MIN_FREE_HEADROOM_BYTES = 512 * 1024 * 1024
ENVIRONMENT_VARIABLE_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_alias(path: Path, *, label: str) -> Path:
    """Reject direct and ancestor symlinks without resolving away their spelling."""
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"{label} cannot be symlinks or contain symlinked path components: {current}"
            )
    return absolute


def _write_json_temporary(path: Path, payload: dict[str, Any]) -> Path:
    path = _reject_symlink_alias(path, label="JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_alias(path.parent, label="JSON output parent")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f".{uuid.uuid4().hex}.tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = _write_json_temporary(path, payload)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _expand(value: str) -> Path:
    for match in ENVIRONMENT_VARIABLE_PATTERN.finditer(value):
        name = match.group("braced") or match.group("plain")
        if name in os.environ and not os.environ[name].strip():
            raise ValueError(f"Empty environment variable in path: {name}")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    if not expanded.strip():
        raise ValueError(f"Expanded path is empty: {value}")
    return Path(expanded).expanduser().absolute()


def _reject_symlinks(root: Path) -> None:
    root = _reject_symlink_alias(root, label="Artifact roots")
    links = sorted(path for path in root.rglob("*") if path.is_symlink())
    if links:
        rendered = ", ".join(str(path.relative_to(root)) for path in links[:20])
        raise ValueError(f"Artifact selections cannot contain symlinks: {rendered}")


def _reject_path_within_root(*, path: Path, root: Path, label: str) -> None:
    _reject_symlink_alias(path, label=label)
    _reject_symlink_alias(root, label="Artifact roots")
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        raise ValueError(f"{label} must be outside the artifact root: {path}")


def _reject_same_path(*, path: Path, protected_path: Path, label: str) -> None:
    same_existing_file = (
        path.exists()
        and protected_path.exists()
        and os.path.samefile(path, protected_path)
    )
    if path.resolve() == protected_path.resolve() or same_existing_file:
        raise ValueError(f"{label} must not overwrite {protected_path}: {path}")


def _reject_existing_hardlink_within_root(*, path: Path, root: Path, label: str) -> None:
    if not path.exists():
        return
    path_stat = path.stat()
    for artifact_path in root.rglob("*"):
        if not artifact_path.is_file():
            continue
        artifact_stat = artifact_path.stat()
        if (path_stat.st_dev, path_stat.st_ino) == (
            artifact_stat.st_dev,
            artifact_stat.st_ino,
        ):
            raise ValueError(
                f"{label} must not be a hardlink to an artifact file: {artifact_path}"
            )


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.is_symlink():
            raise ValueError(f"Artifact selections cannot contain symlinks: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Artifact source does not exist: {path}")
    _reject_symlinks(path)
    return sorted(item for item in path.rglob("*") if item.is_file())


def _load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported staging schema: {payload.get('schema_version')!r}")
    if payload.get("paper_id") != PAPER_ID:
        raise ValueError(f"Unexpected paper_id: {payload.get('paper_id')!r}")
    if payload.get("locked") is not True:
        raise ValueError("Artifact staging specs must set locked=true")
    if not isinstance(payload.get("entries"), list) or not payload["entries"]:
        raise ValueError("Artifact staging specs must contain a non-empty entries list")
    return payload


def _selection(spec: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    destination_parent = _expand(str(spec["destination_root"]))
    set_name = str(spec["artifact_set_dir"])
    if not set_name or Path(set_name).is_absolute() or ".." in Path(set_name).parts:
        raise ValueError(f"Invalid artifact_set_dir: {set_name!r}")
    final_root = destination_parent / set_name

    selected: list[dict[str, Any]] = []
    destinations: set[str] = set()
    artifact_ids: set[str] = set()
    for raw in spec["entries"]:
        if not isinstance(raw, dict):
            raise ValueError("Artifact staging entries must be mappings")
        artifact_id = str(raw["artifact_id"])
        if not artifact_id or artifact_id in artifact_ids:
            raise ValueError(f"Duplicate or empty artifact_id: {artifact_id!r}")
        artifact_ids.add(artifact_id)
        source = _expand(str(raw["source"]))
        destination = Path(str(raw["destination"]))
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError(f"Artifact destination must be relative: {destination}")
        destination_key = destination.as_posix()
        if destination_key in destinations:
            raise ValueError(f"Duplicate artifact destination: {destination_key}")
        destinations.add(destination_key)
        files = _source_files(source)
        if not files:
            raise ValueError(f"Artifact source is empty: {source}")
        selected.append(
            {
                "artifact_id": artifact_id,
                "source": source,
                "source_hint": str(raw["source"]),
                "destination": destination,
                "files": files,
            }
        )
    _reject_overlapping_targets(selected)
    return destination_parent, final_root, selected


def _relative_source_path(source: Path, file_path: Path) -> Path:
    return Path(file_path.name) if source.is_file() else file_path.relative_to(source)


def _target_path(item: dict[str, Any], file_path: Path) -> Path:
    source = item["source"]
    destination = item["destination"]
    relative = _relative_source_path(source, file_path)
    return destination / relative if source.is_dir() else destination


def _reject_overlapping_targets(selected: list[dict[str, Any]]) -> None:
    targets: dict[Path, str] = {}
    for item in selected:
        artifact_id = str(item["artifact_id"])
        for file_path in item["files"]:
            target = _target_path(item, file_path)
            for existing, existing_artifact_id in targets.items():
                if (
                    target == existing
                    or target in existing.parents
                    or existing in target.parents
                ):
                    raise ValueError(
                        "Overlapping artifact targets: "
                        f"{existing_artifact_id!r} and {artifact_id!r} both map through "
                        f"{existing.as_posix()!r} / {target.as_posix()!r}"
                    )
            targets[target] = artifact_id


def plan_staging(spec_path: Path) -> dict[str, Any]:
    spec = _load_spec(spec_path)
    destination_parent, final_root, selected = _selection(spec)
    source_inodes: dict[tuple[int, int], int] = {}
    logical_bytes = 0
    unique_bytes = 0
    file_count = 0
    entries: list[dict[str, Any]] = []
    for item in selected:
        item_bytes = 0
        for file_path in item["files"]:
            stat = file_path.stat()
            file_count += 1
            logical_bytes += stat.st_size
            item_bytes += stat.st_size
            inode = (stat.st_dev, stat.st_ino)
            if inode not in source_inodes:
                source_inodes[inode] = stat.st_size
                unique_bytes += stat.st_size
        entries.append(
            {
                "artifact_id": item["artifact_id"],
                "source": item["source_hint"],
                "destination": item["destination"].as_posix(),
                "file_count": len(item["files"]),
                "logical_size_bytes": item_bytes,
            }
        )
    return {
        "status": "PASS",
        "artifact_set_id": spec["artifact_set_id"],
        "destination_parent": str(destination_parent),
        "final_root": str(final_root),
        "file_count": file_count,
        "logical_size_bytes": logical_bytes,
        "unique_inode_size_bytes": unique_bytes,
        "internal_hardlink_savings_bytes": logical_bytes - unique_bytes,
        "entries": entries,
    }


def _copy_selection(selected: list[dict[str, Any]], temporary_root: Path) -> None:
    copied_inodes: dict[tuple[int, int], Path] = {}
    for item in selected:
        source = item["source"]
        destination = temporary_root / item["destination"]
        for file_path in item["files"]:
            relative_target = _target_path(item, file_path)
            target = temporary_root / relative_target
            target.parent.mkdir(parents=True, exist_ok=True)
            stat = file_path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in copied_inodes:
                os.link(copied_inodes[inode], target)
            else:
                shutil.copy2(file_path, target)
                copied_inodes[inode] = target


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    _reject_symlinks(root)
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _total_size(entries: Iterable[dict[str, Any]]) -> int:
    return sum(int(entry["size_bytes"]) for entry in entries)


def verify_staged(*, root: Path, manifest_path: Path) -> dict[str, Any]:
    root = _reject_symlink_alias(root, label="Artifact roots")
    manifest_path = _reject_symlink_alias(manifest_path, label="Artifact manifest")
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact root is not a directory: {root}")
    _reject_path_within_root(
        path=manifest_path,
        root=root,
        label="Artifact manifest",
    )
    _reject_existing_hardlink_within_root(
        path=manifest_path,
        root=root,
        label="Artifact manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("paper_id") != PAPER_ID:
        raise ValueError("Unexpected external-artifact manifest schema or paper_id")
    if manifest.get("locked") is not True:
        raise ValueError("External-artifact manifests must set locked=true")
    actual = _manifest_entries(root)
    expected = manifest.get("files")
    if not isinstance(expected, list):
        raise ValueError("External-artifact manifest files must be a list")
    actual_by_path = {entry["path"]: entry for entry in actual}
    expected_by_path = {entry["path"]: entry for entry in expected}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    unexpected = sorted(set(actual_by_path) - set(expected_by_path))
    mismatches: list[dict[str, Any]] = []
    for path in sorted(set(actual_by_path) & set(expected_by_path)):
        for field in ("size_bytes", "sha256"):
            if actual_by_path[path].get(field) != expected_by_path[path].get(field):
                mismatches.append(
                    {
                        "path": path,
                        "field": field,
                        "expected": expected_by_path[path].get(field),
                        "actual": actual_by_path[path].get(field),
                    }
                )
    count_matches = int(manifest.get("file_count", -1)) == len(actual)
    size_matches = int(manifest.get("logical_size_bytes", -1)) == _total_size(actual)
    passed = not missing and not unexpected and not mismatches and count_matches and size_matches
    return {
        "status": "PASS" if passed else "FAIL",
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "file_count": len(actual),
        "logical_size_bytes": _total_size(actual),
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "count_matches": count_matches,
        "size_matches": size_matches,
    }


def stage_artifacts(*, spec_path: Path, manifest_path: Path) -> dict[str, Any]:
    spec_path = _reject_symlink_alias(spec_path, label="Artifact staging spec")
    manifest_path = _reject_symlink_alias(manifest_path, label="Artifact manifest")
    spec = _load_spec(spec_path)
    plan = plan_staging(spec_path)
    destination_parent, final_root, selected = _selection(spec)
    _reject_symlinks(final_root)
    _reject_path_within_root(
        path=manifest_path,
        root=final_root,
        label="Artifact manifest",
    )
    if final_root.exists():
        raise ValueError(f"Artifact-set destination already exists: {final_root}")
    destination_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination_parent).free
    required_bytes = int(plan["unique_inode_size_bytes"]) + MIN_FREE_HEADROOM_BYTES
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient free space: free={free_bytes}, required={required_bytes}")

    temporary_root = destination_parent / f".staging-{spec['artifact_set_dir']}-{uuid.uuid4().hex}"
    temporary_manifest: Path | None = None
    published = False
    completed = False
    try:
        temporary_root.mkdir(parents=False)
        _copy_selection(selected, temporary_root)
        files = _manifest_entries(temporary_root)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "paper_id": PAPER_ID,
            "artifact_set_id": spec["artifact_set_id"],
            "artifact_set_dir": spec["artifact_set_dir"],
            "locked": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_spec": spec.get("source_spec_hint", spec_path.as_posix()),
            "artifact_root_hint": spec["destination_root"],
            "selections": [
                {
                    "artifact_id": item["artifact_id"],
                    "source": item["source_hint"],
                    "destination": item["destination"].as_posix(),
                }
                for item in selected
            ],
            "file_count": len(files),
            "logical_size_bytes": _total_size(files),
            "files": files,
        }
        if manifest_path.exists():
            locked_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if locked_manifest.get("artifact_set_id") != spec["artifact_set_id"]:
                raise ValueError("Locked manifest artifact_set_id differs from staging spec")
            if locked_manifest.get("artifact_set_dir") != spec["artifact_set_dir"]:
                raise ValueError("Locked manifest artifact_set_dir differs from staging spec")
            manifest_for_verification = manifest_path
        else:
            temporary_manifest = _write_json_temporary(manifest_path, payload)
            manifest_for_verification = temporary_manifest
        preflight = verify_staged(
            root=temporary_root,
            manifest_path=manifest_for_verification,
        )
        if preflight["status"] != "PASS":
            raise RuntimeError(f"Staged artifact verification failed: {preflight}")
        temporary_root.rename(final_root)
        published = True
        if temporary_manifest is not None:
            os.replace(temporary_manifest, manifest_path)
            temporary_manifest = None
        report = verify_staged(root=final_root, manifest_path=manifest_path)
        if report["status"] != "PASS":
            raise RuntimeError(f"Final artifact verification failed: {report}")
        completed = True
        return report
    except Exception:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if published and not completed and final_root.exists() and not final_root.is_symlink():
            shutil.rmtree(final_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Validate and size an artifact staging spec")
    plan.add_argument("--spec", type=Path, required=True)

    stage = subparsers.add_parser("stage", help="Copy and checksum an immutable artifact set")
    stage.add_argument("--spec", type=Path, required=True)
    stage.add_argument("--manifest", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify an immutable artifact set")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        report = plan_staging(args.spec.resolve())
    elif args.command == "stage":
        report = stage_artifacts(
            spec_path=args.spec.absolute(),
            manifest_path=args.manifest.absolute(),
        )
    else:
        root = args.root.absolute()
        manifest_path = args.manifest.absolute()
        report_path = args.report.absolute() if args.report else None
        if report_path is not None:
            _reject_path_within_root(
                path=report_path,
                root=root,
                label="Verification report",
            )
            _reject_same_path(
                path=report_path,
                protected_path=manifest_path,
                label="Verification report",
            )
            _reject_existing_hardlink_within_root(
                path=report_path,
                root=root,
                label="Verification report",
            )
        report = verify_staged(root=root, manifest_path=manifest_path)
        if report_path is not None:
            _publish_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
