"""Quota-guarded transient particle-dump staging for EMB workflows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def directory_rbytes(path: Path) -> int:
    try:
        return int(os.getxattr(path, "ceph.dir.rbytes").decode("ascii"))
    except (AttributeError, OSError, ValueError):
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def reserve_particle_staging(
    root: Path,
    *,
    case_index: int,
    stage: str,
    limit_bytes: int,
    reservation_bytes: int,
    namespace: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if int(limit_bytes) <= 0 or int(reservation_bytes) <= 0:
        raise ValueError("Particle staging limit and reservation must be positive.")
    if int(reservation_bytes) > int(limit_bytes):
        raise ValueError("Particle staging reservation cannot exceed the staging limit.")
    root.mkdir(parents=True, exist_ok=True)
    reservations = root / ".reservations"
    reservations.mkdir(exist_ok=True)
    lock_path = root / ".reservation.lock"
    namespace_component = "" if namespace is None else f"-{namespace}"
    reservation_id = (
        f"job-{os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID') or os.getpid()}"
        f"-task-{os.environ.get('SLURM_ARRAY_TASK_ID') or 'none'}"
        f"{namespace_component}-{stage}-case-{int(case_index):04d}"
    )
    reservation_path = reservations / f"{reservation_id}.json"
    case_root = root / "cases"
    if namespace is not None:
        case_root /= namespace
    case_root /= f"{stage}-case-{int(case_index):04d}"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        reserved = 0
        for path in reservations.glob("*.json"):
            try:
                reserved += int(json.loads(path.read_text(encoding="utf-8"))["reservation_bytes"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid particle staging reservation: {path}") from exc
        used = directory_rbytes(root)
        projected = used + reserved + int(reservation_bytes)
        if projected > int(limit_bytes):
            raise RuntimeError(
                "Particle staging high-water guard rejected the case: "
                f"used={used}, active_reservations={reserved}, requested={reservation_bytes}, "
                f"limit={limit_bytes}."
            )
        if reservation_path.exists():
            raise FileExistsError(f"Particle staging reservation already exists: {reservation_path}")
        if case_root.exists():
            raise FileExistsError(f"Particle staging case root already exists: {case_root}")
        case_root.mkdir(parents=True)
        payload = {
            "schema": "mesouq.emb_particle_staging_reservation.v1",
            "created_utc": _utc_stamp(),
            "reservation_id": reservation_id,
            "case_index": int(case_index),
            "stage": stage,
            "namespace": namespace,
            "root": str(root),
            "case_root": str(case_root),
            "reservation_path": str(reservation_path),
            "limit_bytes": int(limit_bytes),
            "reservation_bytes": int(reservation_bytes),
            "used_bytes_at_reservation": int(used),
            "active_reservation_bytes_before": int(reserved),
            "projected_guard_bytes": int(projected),
            "hostname": socket.getfqdn(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        reservation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (case_root / "STAGING_OWNER.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload


def particle_staging_namespace(campaign_root: Path) -> str:
    digest = hashlib.sha256(str(campaign_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"campaign-{digest}"


def release_particle_staging(allocation: dict[str, Any] | None) -> dict[str, Any]:
    if allocation is None:
        return {"status": "not_configured"}
    root = Path(str(allocation["root"])).resolve()
    case_root = Path(str(allocation["case_root"])).resolve()
    reservation_path = Path(str(allocation["reservation_path"])).resolve()
    if root not in case_root.parents or root not in reservation_path.parents:
        raise RuntimeError("Refusing to clean particle staging outside its configured root.")
    def ignore_missing(
        function: Any,
        path: str,
        exc_info: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        del function, path
        if isinstance(exc_info[1], FileNotFoundError):
            return
        raise exc_info[1]

    if case_root.is_dir():
        shutil.rmtree(case_root, onerror=ignore_missing)
    lock_path = root / ".reservation.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        reservation_path.unlink(missing_ok=True)
    return {
        "status": "cleaned",
        "case_root": str(case_root),
        "reservation_path": str(reservation_path),
        "case_root_exists_after": case_root.exists(),
        "reservation_exists_after": reservation_path.exists(),
        "staging_root_rbytes_after": directory_rbytes(root),
    }
