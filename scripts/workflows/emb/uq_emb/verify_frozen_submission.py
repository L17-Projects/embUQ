#!/usr/bin/env python3
"""Create or verify an immutable UQ_EMB editor-submission snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
PAPER_ID = "UQ_EMB"
MIN_FREE_HEADROOM_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_TOTAL_BYTES = 20 * 1024 * 1024
MAX_SNAPSHOT_FILE_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


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


def _reject_symlinks(root: Path) -> None:
    root = _reject_symlink_alias(root, label="Frozen snapshot roots")
    links = sorted(path for path in root.rglob("*") if path.is_symlink())
    if links:
        rendered = ", ".join(str(path.relative_to(root)) for path in links)
        raise ValueError(f"Frozen snapshots cannot contain symlinks: {rendered}")


def _entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _entries(root: Path) -> list[dict[str, Any]]:
    _reject_symlinks(root)
    return [_entry(root, path) for path in _files(root)]


def _total_size(entries: Iterable[dict[str, Any]]) -> int:
    return sum(int(entry["size_bytes"]) for entry in entries)


def _enforce_snapshot_size_limits(entries: list[dict[str, Any]]) -> None:
    oversized = [
        entry
        for entry in entries
        if int(entry["size_bytes"]) > MAX_SNAPSHOT_FILE_BYTES
    ]
    if oversized:
        rendered = ", ".join(
            f"{entry['path']} ({entry['size_bytes']} bytes)" for entry in oversized
        )
        raise ValueError(
            "Frozen snapshot files exceed the per-file limit "
            f"of {MAX_SNAPSHOT_FILE_BYTES} bytes: {rendered}"
        )

    total_size = _total_size(entries)
    if total_size > MAX_SNAPSHOT_TOTAL_BYTES:
        raise ValueError(
            "Frozen snapshot exceeds the total size limit: "
            f"size={total_size}, limit={MAX_SNAPSHOT_TOTAL_BYTES}"
        )


def _reject_path_within_root(*, path: Path, root: Path, label: str) -> None:
    _reject_symlink_alias(path, label=label)
    _reject_symlink_alias(root, label="Frozen snapshot roots")
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        raise ValueError(f"{label} must be outside the frozen snapshot root: {path}")


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
    for member in _files(root):
        member_stat = member.stat()
        if (path_stat.st_dev, path_stat.st_ino) == (member_stat.st_dev, member_stat.st_ino):
            raise ValueError(
                f"{label} must not be a hardlink to a frozen snapshot member: {member}"
            )


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


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if payload.get("paper_id") != PAPER_ID:
        raise ValueError(f"Unexpected paper_id {payload.get('paper_id')!r}")
    if payload.get("locked") is not True:
        raise ValueError("Frozen submission manifest must set locked=true")
    return payload


def verify_snapshot(*, root: Path, manifest_path: Path) -> dict[str, Any]:
    root = _reject_symlink_alias(root, label="Frozen snapshot roots")
    manifest_path = _reject_symlink_alias(manifest_path, label="Snapshot manifest")
    if not root.is_dir():
        raise FileNotFoundError(f"Frozen snapshot root is not a directory: {root}")
    _reject_path_within_root(path=manifest_path, root=root, label="Snapshot manifest")
    _reject_existing_hardlink_within_root(
        path=manifest_path,
        root=root,
        label="Snapshot manifest",
    )
    manifest = _load_manifest(manifest_path)
    actual_entries = _entries(root)
    expected_entries = manifest.get("files")
    if not isinstance(expected_entries, list):
        raise ValueError("Manifest files must be a list")

    expected_by_path = {entry["path"]: entry for entry in expected_entries}
    actual_by_path = {entry["path"]: entry for entry in actual_entries}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    unexpected = sorted(set(actual_by_path) - set(expected_by_path))
    mismatches: list[dict[str, Any]] = []
    for relative_path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected = expected_by_path[relative_path]
        actual = actual_by_path[relative_path]
        for field in ("size_bytes", "sha256"):
            if expected.get(field) != actual.get(field):
                mismatches.append(
                    {
                        "path": relative_path,
                        "field": field,
                        "expected": expected.get(field),
                        "actual": actual.get(field),
                    }
                )

    expected_count = int(manifest.get("file_count", -1))
    expected_total = int(manifest.get("total_size_bytes", -1))
    count_ok = expected_count == len(actual_entries)
    size_ok = expected_total == _total_size(actual_entries)
    passed = not missing and not unexpected and not mismatches and count_ok and size_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "file_count": len(actual_entries),
        "total_size_bytes": _total_size(actual_entries),
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "count_matches": count_ok,
        "size_matches": size_ok,
    }


def create_snapshot(
    *,
    source: Path,
    destination: Path,
    manifest_path: Path,
    snapshot_id: str,
    source_hint: str,
) -> dict[str, Any]:
    source = _reject_symlink_alias(source, label="Snapshot source")
    destination = _reject_symlink_alias(destination, label="Snapshot destination")
    manifest_path = _reject_symlink_alias(manifest_path, label="Snapshot manifest")
    if not source.is_dir():
        raise ValueError(f"Snapshot source is not a directory: {source}")
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if (
        source_resolved == destination_resolved
        or source_resolved in destination_resolved.parents
        or destination_resolved in source_resolved.parents
    ):
        raise ValueError(
            "Snapshot source and destination must not overlap: "
            f"source={source}, destination={destination}"
        )
    _reject_path_within_root(
        path=manifest_path,
        root=destination,
        label="Snapshot manifest",
    )
    _reject_path_within_root(
        path=manifest_path,
        root=source,
        label="Snapshot manifest",
    )
    _reject_symlinks(source)
    source_entries = _entries(source)
    if not source_entries:
        raise ValueError(f"Snapshot source is empty: {source}")
    _enforce_snapshot_size_limits(source_entries)
    source_total = _total_size(source_entries)

    if destination.exists():
        raise ValueError(f"Snapshot destination must be absent: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_alias(destination.parent, label="Snapshot destination parent")
    free_bytes = shutil.disk_usage(destination.parent).free
    required_bytes = source_total + MIN_FREE_HEADROOM_BYTES
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space for snapshot: free={free_bytes}, required={required_bytes}"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=f".{uuid.uuid4().hex}.tmp",
        )
    )
    temporary_manifest: Path | None = None
    published = False
    completed = False
    try:
        shutil.copytree(source, temporary_root, copy_function=shutil.copy2, dirs_exist_ok=True)
        copied_entries = _entries(temporary_root)
        if source_entries != copied_entries:
            raise RuntimeError("Copied snapshot does not match source checksums, sizes, and paths")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "paper_id": PAPER_ID,
            "snapshot_id": snapshot_id,
            "locked": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_hint": source_hint,
            "snapshot_root": destination.as_posix(),
            "file_count": len(copied_entries),
            "total_size_bytes": _total_size(copied_entries),
            "files": copied_entries,
        }
        temporary_manifest = _write_json_temporary(manifest_path, payload)
        preflight = verify_snapshot(root=temporary_root, manifest_path=temporary_manifest)
        if preflight["status"] != "PASS":
            raise RuntimeError(f"Post-copy verification failed: {json.dumps(preflight, sort_keys=True)}")
        temporary_root.rename(destination)
        published = True
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
        report = verify_snapshot(root=destination, manifest_path=manifest_path)
        if report["status"] != "PASS":
            raise RuntimeError(f"Post-publication verification failed: {json.dumps(report, sort_keys=True)}")
        completed = True
        return report
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if published and not completed and destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Copy and checksum a frozen submission")
    snapshot.add_argument("--source", type=Path, required=True)
    snapshot.add_argument("--destination", type=Path, required=True)
    snapshot.add_argument("--manifest", type=Path, required=True)
    snapshot.add_argument("--snapshot-id", required=True)
    snapshot.add_argument("--source-hint", required=True)

    verify = subparsers.add_parser("verify", help="Verify a frozen submission")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "snapshot":
        report = create_snapshot(
            source=args.source.absolute(),
            destination=args.destination,
            manifest_path=args.manifest.absolute(),
            snapshot_id=args.snapshot_id,
            source_hint=args.source_hint,
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
        report = verify_snapshot(root=root, manifest_path=manifest_path)
        if report_path is not None:
            _publish_json(report_path, report)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
