"""Provenance helpers shared by the frozen UQ_EMB replay entrypoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CLOSEOUT_MANIFESTS = (
    "accepted_production_outputs_202607.files.json",
    "editor_submission_review2_v1.json",
    "frozen_legacy_paper_runtime_complete_202606.files.json",
    "frozen_plotting_dependencies_202607.files.json",
    "frozen_runtime_dependencies_202607.files.json",
    "frozen_tinytex_runtime_202606.files.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Locked UQ_EMB manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("paper_id") != "UQ_EMB" or payload.get("locked") is not True:
        raise ValueError(f"Unexpected or unlocked UQ_EMB manifest: {path}")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Locked UQ_EMB manifest has no files list: {path}")
    paths = [str(entry.get("path", "")) for entry in files if isinstance(entry, dict)]
    if len(paths) != len(files) or any(not item for item in paths) or len(set(paths)) != len(paths):
        raise ValueError(f"Locked UQ_EMB manifest has invalid or duplicate paths: {path}")
    return payload


def _manifest_entries(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(entry["path"]): entry for entry in payload["files"]}


def verify_locked_artifact_root(*, root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every file in a locked artifact set, including absence of extras."""
    root = root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    payload = _locked_manifest(manifest_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Locked UQ_EMB artifact root is missing: {root}")
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError(f"Locked UQ_EMB artifact root contains symlinks: {symlinks[0]}")

    expected = _manifest_entries(payload)
    actual_paths = sorted(path for path in root.rglob("*") if path.is_file())
    actual_relatives = {path.relative_to(root).as_posix(): path for path in actual_paths}
    missing = sorted(set(expected) - set(actual_relatives))
    unexpected = sorted(set(actual_relatives) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Locked UQ_EMB artifact inventory mismatch for {root}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    logical_size = 0
    for relative, path in actual_relatives.items():
        record = expected[relative]
        size = path.stat().st_size
        logical_size += size
        if int(record.get("size_bytes", -1)) != size or record.get("sha256") != sha256(path):
            raise ValueError(f"Locked UQ_EMB artifact content mismatch: {path}")

    expected_count = int(payload.get("file_count", -1))
    expected_size = int(
        payload.get("logical_size_bytes", payload.get("total_size_bytes", -1))
    )
    if expected_count != len(actual_paths) or expected_size != logical_size:
        raise ValueError(
            f"Locked UQ_EMB artifact totals mismatch for {root}: "
            f"count={len(actual_paths)}/{expected_count}, size={logical_size}/{expected_size}"
        )
    return {
        "status": "PASS",
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "file_count": len(actual_paths),
        "logical_size_bytes": logical_size,
    }


def verify_locked_manifest_members(
    *,
    root: Path,
    manifest_path: Path,
    members: Iterable[Path],
) -> dict[str, Any]:
    """Verify selected consumed files against a locked artifact manifest."""
    root = root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    payload = _locked_manifest(manifest_path)
    expected = _manifest_entries(payload)
    verified: list[dict[str, Any]] = []
    for member in members:
        path = member.expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Consumed artifact escapes its locked root: {path}") from exc
        record = expected.get(relative)
        if record is None:
            raise ValueError(f"Consumed artifact is absent from the locked manifest: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Consumed locked artifact is missing: {path}")
        size = path.stat().st_size
        digest = sha256(path)
        if int(record.get("size_bytes", -1)) != size or record.get("sha256") != digest:
            raise ValueError(f"Consumed locked artifact content mismatch: {path}")
        verified.append({"path": str(path), "size_bytes": size, "sha256": digest})
    logical_size = sum(int(item["size_bytes"]) for item in verified)
    return {
        "status": "PASS",
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "file_count": len(verified),
        "logical_size_bytes": logical_size,
        "members": verified,
    }


def _artifact_root_for_path(path: Path, artifact_set_dir: str) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == artifact_set_dir:
            return candidate
    return None


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def runtime_provenance(
    *,
    repo_root: Path,
    site: str,
    requested_python_bin: str | None = None,
) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    return {
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current") or None,
        "git_status_clean": not bool(status),
        "site": site,
        "hostname": socket.getfqdn(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "requested_python_bin": requested_python_bin,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "site_runtime_root": os.environ.get("MESOUQ_SITE_RUNTIME_ROOT"),
        "repo_root": str(repo_root.resolve()),
    }


def replay_receipt_provenance(
    *,
    repo_root: Path,
    runner: Path,
    site: str | None = None,
    consumed_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Bind a replay receipt to Git and verify every consumed locked artifact set."""
    repo_root = repo_root.expanduser().resolve()
    runner = runner.expanduser().resolve()
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifests: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in DEFAULT_CLOSEOUT_MANIFESTS:
        path = manifest_root / name
        payloads[name] = _locked_manifest(path)
        manifests[name] = {"path": str(path), "sha256": sha256(path)}

    resolved_consumed = [path.expanduser().resolve() for path in consumed_paths]
    roots_to_verify: dict[tuple[str, Path], None] = {}
    editor_root = repo_root / "papers" / "UQ_EMB" / "editor_submission" / "review2_v1"
    for consumed in resolved_consumed:
        if consumed == editor_root or editor_root in consumed.parents:
            roots_to_verify[("editor_submission_review2_v1.json", editor_root)] = None
            continue
        if consumed == repo_root or repo_root in consumed.parents:
            continue
        matched = False
        for name, payload in payloads.items():
            artifact_set_dir = payload.get("artifact_set_dir")
            if not isinstance(artifact_set_dir, str) or not artifact_set_dir:
                continue
            root = _artifact_root_for_path(consumed, artifact_set_dir)
            if root is not None:
                roots_to_verify[(name, root)] = None
                matched = True
                break
        if not matched:
            raise ValueError(
                f"External replay input is not covered by a locked UQ_EMB manifest: {consumed}"
            )

    verified_sets = {
        f"{name}:{root}": verify_locked_artifact_root(
            root=root,
            manifest_path=manifest_root / name,
        )
        for name, root in roots_to_verify
    }
    return {
        "runtime": runtime_provenance(
            repo_root=repo_root,
            site=site or os.environ.get("MESOUQ_SITE", "local"),
        ),
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "locked_manifests": manifests,
        "verified_artifact_sets": verified_sets,
    }


def load_materialization_binding(
    config_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    receipt_path = config_path.with_suffix(".materialization.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"UQ_EMB materialization receipt is missing for {config_path}: {receipt_path}"
        )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("paper_id") != "UQ_EMB":
        raise ValueError(f"Unexpected materialization receipt paper_id: {receipt_path}")
    if Path(str(payload.get("materialized_config", ""))).resolve() != config_path:
        raise ValueError(f"Materialization receipt points at a different config: {receipt_path}")
    actual_config_sha = sha256(config_path)
    if payload.get("materialized_config_sha256") != actual_config_sha:
        raise ValueError(f"Materialized config hash mismatch: {config_path}")
    agent = str(payload.get("agent", "")).strip().lower()
    artifact_root = Path(str(payload.get("artifact_root", ""))).expanduser().resolve()
    from materialize_hbi_config import (  # Imported lazily to avoid a module cycle.
        ACCEPTED_SET,
        DEPENDENCY_SET,
        _semantic_config_sha256,
    )

    dependency_root = artifact_root / DEPENDENCY_SET
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Materialized config is not a mapping: {config_path}")
    semantic_sha = _semantic_config_sha256(
        config,
        agent=agent,
        dependency_root=dependency_root,
    )
    if payload.get("semantic_config_sha256") != semantic_sha:
        raise ValueError(f"Materialized semantic config hash mismatch: {config_path}")

    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifest_bindings = (
        (
            "accepted_manifest",
            "accepted_manifest_sha256",
            manifest_root / f"{ACCEPTED_SET}.files.json",
        ),
        (
            "dependency_manifest",
            "dependency_manifest_sha256",
            manifest_root / f"{DEPENDENCY_SET}.files.json",
        ),
    )
    for path_key, hash_key, expected_path in manifest_bindings:
        recorded_path = Path(str(payload.get(path_key, ""))).expanduser().resolve()
        if recorded_path != expected_path.resolve():
            raise ValueError(
                f"Materialization receipt {path_key} is not the current locked manifest: "
                f"{recorded_path} != {expected_path.resolve()}"
            )
        if not expected_path.is_file() or payload.get(hash_key) != sha256(expected_path):
            raise ValueError(f"Materialization receipt {hash_key} mismatch: {expected_path}")

    dependency_verification = verify_locked_artifact_root(
        root=dependency_root,
        manifest_path=manifest_root / f"{DEPENDENCY_SET}.files.json",
    )

    source_config = Path(str(payload.get("source_config", ""))).expanduser().resolve()
    expected_source_root = artifact_root / ACCEPTED_SET
    try:
        source_config.relative_to(expected_source_root)
    except ValueError as exc:
        raise ValueError(
            f"Materialization source config escapes the accepted artifact root: {source_config}"
        ) from exc
    if not source_config.is_file() or payload.get("source_config_sha256") != sha256(source_config):
        raise ValueError(f"Materialization source config hash mismatch: {source_config}")
    accepted_source_verification = verify_locked_manifest_members(
        root=expected_source_root,
        manifest_path=manifest_root / f"{ACCEPTED_SET}.files.json",
        members=[source_config],
    )

    materialization_provenance = payload.get("provenance")
    if not isinstance(materialization_provenance, dict):
        raise ValueError(f"Materialization receipt lacks Git provenance: {receipt_path}")
    current_commit = _git(repo_root, "rev-parse", "HEAD")
    if materialization_provenance.get("git_commit") != current_commit:
        raise ValueError(
            "Materialization commit does not match runtime HEAD: "
            f"{materialization_provenance.get('git_commit')} != {current_commit}"
        )
    if materialization_provenance.get("git_status_clean") is not True:
        raise ValueError(f"Materialization was not produced from a clean worktree: {receipt_path}")
    if _git(repo_root, "status", "--porcelain"):
        raise ValueError(f"UQ_EMB replay requires a clean runtime worktree: {repo_root}")
    return {
        "materialization_receipt": str(receipt_path),
        "materialization_receipt_sha256": sha256(receipt_path),
        "config_sha256": actual_config_sha,
        "config_semantic_sha256": semantic_sha,
        "source_config_sha256": payload.get("source_config_sha256"),
        "accepted_manifest_sha256": payload.get("accepted_manifest_sha256"),
        "dependency_manifest_sha256": payload.get("dependency_manifest_sha256"),
        "accepted_source_verification": accepted_source_verification,
        "dependency_verification": dependency_verification,
    }
