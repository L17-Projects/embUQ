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


def load_materialization_binding(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
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
    semantic_sha = str(payload.get("semantic_config_sha256", ""))
    if len(semantic_sha) != 64:
        raise ValueError(f"Materialization receipt lacks a semantic config digest: {receipt_path}")
    return {
        "materialization_receipt": str(receipt_path),
        "materialization_receipt_sha256": sha256(receipt_path),
        "config_sha256": actual_config_sha,
        "config_semantic_sha256": semantic_sha,
        "source_config_sha256": payload.get("source_config_sha256"),
        "accepted_manifest_sha256": payload.get("accepted_manifest_sha256"),
        "dependency_manifest_sha256": payload.get("dependency_manifest_sha256"),
    }
