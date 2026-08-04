"""Provenance helpers shared by the frozen UQ_EMB replay entrypoints."""

from __future__ import annotations

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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict[str, Any]:
    """Bind a replay receipt to its runner, Git state, and locked manifests."""
    repo_root = repo_root.expanduser().resolve()
    runner = runner.expanduser().resolve()
    manifest_root = repo_root / "papers" / "UQ_EMB" / "manifests"
    manifests = {}
    for name in DEFAULT_CLOSEOUT_MANIFESTS:
        path = manifest_root / name
        if path.is_file():
            manifests[name] = {"path": str(path), "sha256": sha256(path)}
    return {
        "runtime": runtime_provenance(
            repo_root=repo_root,
            site=site or os.environ.get("MESOUQ_SITE", "local"),
        ),
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "locked_manifests": manifests,
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
    }
