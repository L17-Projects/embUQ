from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION = "meso_uq.dpd_sampling.production_preflight.v1"

_CAPTURE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "MESOUQ_RUNS_ROOT",
    "MESOUQ_SCRATCH_ROOT",
    "MESOUQ_SITE",
    "PYTHONPATH",
    "PYTHON_EXECUTABLE",
    "REPO_ROOT",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_NODELIST",
    "SLURM_NTASKS",
    "SLURM_SUBMIT_DIR",
    "SLURM_TRES_PER_TASK",
)


@dataclass(frozen=True)
class DPDProductionPreflightConfig:
    output_root: Path
    manifest_path: Path | None = None
    candidate_manifest: Path | None = None
    require_scratch: bool = True
    allowed_scratch_roots: tuple[Path, ...] = ()
    min_free_bytes: int = 1_000_000_000
    hdf5_smoke_test: bool = False
    hdf5_driver: str = "serial"
    env: Mapping[str, str] | None = None


def _resolve_output_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def output_root_from_candidate_manifest(candidate_manifest: str | Path) -> Path:
    path = Path(candidate_manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_payload = payload.get("rendered_payload", {}).get("request_payload", {})
    output_root = str(request_payload.get("output_root") or payload.get("output_root") or "").strip()
    if not output_root:
        raise ValueError(f"Candidate manifest does not define an output_root: {path}")
    return _resolve_output_root(output_root)


def default_preflight_manifest_path(output_root: str | Path) -> Path:
    return _resolve_output_root(output_root) / "dpd_production_preflight.json"


def _is_home_backed(path: Path) -> bool:
    return len(path.parts) > 1 and path.parts[1].lower() == "home"


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _split_path_list(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    return tuple(Path(item).expanduser().resolve() for item in value.split(os.pathsep) if item.strip())


def _allowed_scratch_roots(
    *,
    explicit_roots: tuple[Path, ...],
    env: Mapping[str, str],
) -> tuple[Path, ...]:
    roots = [root.expanduser().resolve() for root in explicit_roots]
    roots.extend(_split_path_list(env.get("MESOUQ_SCRATCH_ROOT")))
    roots.extend(_split_path_list(env.get("SCRATCH_ROOT")))
    if not roots:
        roots.append(Path("/scratch").resolve())
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return tuple(deduped)


def _check_output_root_policy(
    output_root: Path,
    *,
    require_scratch: bool,
    allowed_scratch_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    scratch_backed = (
        not _is_home_backed(output_root)
        and any(_path_is_relative_to(output_root, root) for root in allowed_scratch_roots)
    )
    if require_scratch and not scratch_backed:
        roots_text = ", ".join(str(root) for root in allowed_scratch_roots)
        checks.append(
            {
                "name": "scratch_output_root",
                "status": "failed",
                "message": f"DPD production output root must be under an allowed scratch root ({roots_text}): {output_root}",
            }
        )
    else:
        checks.append(
            {
                "name": "scratch_output_root",
                "status": "passed",
                "message": "output root is accepted for this preflight policy",
            }
        )
    return checks


def _check_writable_output_root(output_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".dpd_preflight_write_", dir=output_root, delete=True) as handle:
        handle.write(b"mesouq-dpd-preflight\n")
        handle.flush()
        os.fsync(handle.fileno())
    checks.append({"name": "output_root_writable", "status": "passed", "message": "write/fsync succeeded"})
    return checks


def _check_free_space(output_root: Path, *, min_free_bytes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usage = shutil.disk_usage(output_root)
    payload = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "min_free_bytes": int(min_free_bytes),
    }
    if usage.free < min_free_bytes:
        return (
            [
                {
                    "name": "free_space",
                    "status": "failed",
                    "message": f"free bytes {usage.free} is below required {min_free_bytes}",
                }
            ],
            payload,
        )
    return (
        [
            {
                "name": "free_space",
                "status": "passed",
                "message": f"free bytes {usage.free} meets required {min_free_bytes}",
            }
        ],
        payload,
    )


def _check_nested_srun_env(env: Mapping[str, str]) -> list[dict[str, Any]]:
    cpus = str(env.get("SLURM_CPUS_PER_TASK", "")).strip()
    tres = str(env.get("SLURM_TRES_PER_TASK", "")).strip()
    if cpus and tres:
        return [
            {
                "name": "nested_srun_cpu_environment",
                "status": "failed",
                "message": f"conflicting nested srun CPU env: SLURM_CPUS_PER_TASK={cpus}, SLURM_TRES_PER_TASK={tres}",
            }
        ]
    return [
        {
            "name": "nested_srun_cpu_environment",
            "status": "passed",
            "message": "nested srun CPU env is clean",
        }
    ]


def _run_hdf5_smoke_test(output_root: Path, *, driver: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    smoke_path = output_root / ".dpd_preflight_hdf5_smoke.h5"
    payload: dict[str, Any] = {"path": str(smoke_path), "driver": driver}
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - exercised in environments without h5py
        return (
            [
                {
                    "name": "hdf5_smoke_test",
                    "status": "failed",
                    "message": f"unable to import h5py: {exc}",
                }
            ],
            payload,
        )

    try:
        if driver == "serial":
            with h5py.File(smoke_path, "w") as h5:
                h5.attrs["schema_version"] = DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION
                h5.create_dataset("smoke", data=[1, 2, 3])
            with h5py.File(smoke_path, "r") as h5:
                payload["dataset_shape"] = list(h5["smoke"].shape)
        elif driver == "mpio":
            from mpi4py import MPI

            with h5py.File(smoke_path, "w", driver="mpio", comm=MPI.COMM_WORLD) as h5:
                h5.attrs["schema_version"] = DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION
                h5.create_dataset("smoke", data=[1, 2, 3])
            with h5py.File(smoke_path, "r", driver="mpio", comm=MPI.COMM_WORLD) as h5:
                payload["dataset_shape"] = list(h5["smoke"].shape)
            payload["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
            payload["mpi_size"] = MPI.COMM_WORLD.Get_size()
        else:
            raise ValueError(f"Unsupported HDF5 smoke-test driver: {driver}")
    except Exception as exc:
        return (
            [
                {
                    "name": "hdf5_smoke_test",
                    "status": "failed",
                    "message": f"HDF5 smoke test failed: {exc}",
                }
            ],
            payload,
        )
    finally:
        try:
            smoke_path.unlink()
        except FileNotFoundError:
            pass

    return (
        [
            {
                "name": "hdf5_smoke_test",
                "status": "passed",
                "message": "HDF5 write/read smoke test succeeded",
            }
        ],
        payload,
    )


def _capture_environment(env: Mapping[str, str]) -> dict[str, str]:
    return {key: str(env[key]) for key in _CAPTURE_ENV_KEYS if key in env}


def run_dpd_production_preflight(config: DPDProductionPreflightConfig) -> dict[str, Any]:
    env = config.env if config.env is not None else os.environ
    output_root = _resolve_output_root(config.output_root)
    allowed_scratch_roots = _allowed_scratch_roots(
        explicit_roots=config.allowed_scratch_roots,
        env=env,
    )
    started = time.time()
    checks: list[dict[str, Any]] = []
    storage: dict[str, Any] = {}
    hdf5: dict[str, Any] | None = None

    try:
        checks.extend(
            _check_output_root_policy(
                output_root,
                require_scratch=config.require_scratch,
                allowed_scratch_roots=allowed_scratch_roots,
            )
        )
        if all(item["status"] == "passed" for item in checks):
            checks.extend(_check_writable_output_root(output_root))
            free_space_checks, storage = _check_free_space(output_root, min_free_bytes=config.min_free_bytes)
            checks.extend(free_space_checks)
            checks.extend(_check_nested_srun_env(env))
            if config.hdf5_smoke_test:
                hdf5_checks, hdf5 = _run_hdf5_smoke_test(output_root, driver=config.hdf5_driver)
                checks.extend(hdf5_checks)
    except Exception as exc:
        checks.append({"name": "preflight_exception", "status": "failed", "message": str(exc)})

    finished = time.time()
    status = "passed" if checks and all(item["status"] == "passed" for item in checks) else "failed"
    payload: dict[str, Any] = {
        "schema_version": DPD_PRODUCTION_PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "output_root": str(output_root),
        "candidate_manifest": str(config.candidate_manifest) if config.candidate_manifest else None,
        "manifest_path": str(config.manifest_path) if config.manifest_path else None,
        "require_scratch": config.require_scratch,
        "allowed_scratch_roots": [str(root) for root in allowed_scratch_roots],
        "hostname": socket.gethostname(),
        "started_epoch": started,
        "finished_epoch": finished,
        "runtime_seconds": max(0.0, finished - started),
        "checks": checks,
        "storage": storage,
        "environment": _capture_environment(env),
    }
    if hdf5 is not None:
        payload["hdf5_smoke_test"] = hdf5
    return payload


def write_preflight_manifest(payload: Mapping[str, Any], manifest_path: str | Path) -> Path:
    path = Path(manifest_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path
