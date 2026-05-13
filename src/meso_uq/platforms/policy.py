"""Dependency-light immutable platform policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS
from meso_uq.core import Platform

PlatformPolicy = str


@dataclass(frozen=True)
class PlatformPolicyRecord:
    site: str
    scheduler: str
    gpu_resource_syntax: tuple[str, ...]
    canonical_runs_root: str
    forbidden_path_prefixes: tuple[str, ...]
    environment_scripts: tuple[str, ...]
    allow_absolute_paths: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlatformPolicyRecord":
        return cls(
            site=str(payload["site"]),
            scheduler=str(payload["scheduler"]),
            gpu_resource_syntax=tuple(str(item) for item in payload["gpu_resource_syntax"]),
            canonical_runs_root=str(payload["canonical_runs_root"]),
            forbidden_path_prefixes=tuple(str(item) for item in payload["forbidden_path_prefixes"]),
            environment_scripts=tuple(str(item) for item in payload["environment_scripts"]),
            allow_absolute_paths=bool(payload.get("allow_absolute_paths", False)),
        )


@dataclass(frozen=True)
class PlatformPolicyLookupError(KeyError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


_PLATFORM_POLICIES: dict[PlatformPolicy, PlatformPolicyRecord] = {
    "karolina": PlatformPolicyRecord(
        site="karolina",
        scheduler="slurm",
        gpu_resource_syntax=("--gpus", "--gpus-per-task"),
        canonical_runs_root="${MESOUQ_SCRATCH_ROOT}/runs",
        forbidden_path_prefixes=("/ceph/hpc/home/eubrieucb",),
        environment_scripts=(
            "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh",
        ),
    ),
    "vega": PlatformPolicyRecord(
        site="vega",
        scheduler="slurm",
        gpu_resource_syntax=("--gres=gpu:1",),
        canonical_runs_root="${MESOUQ_RUNS_ROOT}",
        forbidden_path_prefixes=(),
        environment_scripts=(
            "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh",
        ),
    ),
    "workstation": PlatformPolicyRecord(
        site="workstation",
        scheduler="local",
        gpu_resource_syntax=(),
        canonical_runs_root="${MESOUQ_RUNS_ROOT}",
        forbidden_path_prefixes=(),
        environment_scripts=(),
    ),
    "generic_slurm": PlatformPolicyRecord(
        site="generic_slurm",
        scheduler="slurm",
        gpu_resource_syntax=("--gres=gpu:1",),
        canonical_runs_root="${MESOUQ_RUNS_ROOT}",
        forbidden_path_prefixes=(),
        environment_scripts=(
            "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh",
            "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh",
        ),
    ),
}


def list_platform_policies() -> tuple[PlatformPolicy, ...]:
    """Return the known platform policy names."""

    return tuple(_PLATFORM_POLICIES.keys())


def lookup_platform_policy(platform: Platform | PlatformPolicy) -> PlatformPolicyRecord:
    """Return immutable policy metadata for a known platform."""

    platform_key = platform.value if isinstance(platform, Platform) else platform
    try:
        return _PLATFORM_POLICIES[platform_key]
    except KeyError as exc:
        known = ", ".join(sorted(_PLATFORM_POLICIES.keys()))
        raise PlatformPolicyLookupError(
            f"Unknown platform: {platform_key!r}. Known platforms: [{known}]"
        ) from exc


def validate_platform_path_policy(
    platform: Platform | PlatformPolicy,
    path_values: Mapping[str, Any],
    *,
    label: str | None = None,
) -> list[str]:
    """Validate path-like values against a platform policy."""

    policy = lookup_platform_policy(platform)
    record_label = label or platform
    errors: list[str] = []

    for field_name, path_value in path_values.items():
        if not isinstance(path_value, str):
            errors.append(f"{record_label}.{field_name}: path must be a string.")
            continue

        if path_value.startswith("/") and not policy.allow_absolute_paths:
            errors.append(
                f"{record_label}.{field_name}: absolute paths are forbidden for {platform!r} policy."
            )

        for prefix in policy.forbidden_path_prefixes:
            if path_value.startswith(prefix):
                errors.append(
                    f"{record_label}.{field_name}: path uses forbidden prefix {prefix!r}."
                )

        for forbidden in FORBIDDEN_PRIVATE_PATHS:
            if forbidden in path_value:
                errors.append(
                    f"{record_label}.{field_name}: path contains forbidden private literal {forbidden!r}."
                )

    return errors
