from __future__ import annotations

import pytest

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS
from meso_uq.core import Platform
from meso_uq.platforms import (
    PlatformPolicyLookupError,
    PlatformPolicyRecord,
    list_platform_policies,
    lookup_platform_policy,
    validate_platform_path_policy,
)


def test_list_platform_policies_is_stable() -> None:
    assert set(list_platform_policies()) == {
        "generic_slurm",
        "karolina",
        "vega",
        "workstation",
    }


def test_lookup_known_platform_policy() -> None:
    policy = lookup_platform_policy("vega")
    assert isinstance(policy, PlatformPolicyRecord)
    assert policy.site == "vega"
    assert policy.scheduler == "slurm"


def test_lookup_unknown_platform_policy_raises() -> None:
    with pytest.raises(PlatformPolicyLookupError):
        lookup_platform_policy("moonbase")


def test_karolina_platform_policy_forbids_private_root() -> None:
    policy = lookup_platform_policy("karolina")
    assert "/ceph/hpc/home/eubrieucb" in policy.forbidden_path_prefixes
    assert policy.canonical_runs_root == "${MESOUQ_SCRATCH_ROOT}/runs"
    assert policy.environment_scripts == (
        "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh",
    )


def test_generic_slurm_policy_has_no_private_root_forbidden_prefix() -> None:
    policy = lookup_platform_policy("generic_slurm")
    assert all(
        forbidden not in FORBIDDEN_PRIVATE_PATHS for forbidden in policy.forbidden_path_prefixes
    )


def test_generic_platform_enum_uses_generic_slurm_baseline() -> None:
    policy = lookup_platform_policy(Platform.GENERIC)

    assert policy.site == "generic_slurm"
    assert validate_platform_path_policy(
        Platform.GENERIC,
        {"runs_root": "${MESOUQ_RUNS_ROOT}/validation"},
        label="generic",
    ) == []


def test_generic_slurm_path_validation_rejects_private_path_literals() -> None:
    errors = validate_platform_path_policy(
        "generic_slurm",
        {"runs_root": "/ceph/hpc/home/eubrieucb/mesouq/runs"},
        label="generic_slurm",
    )

    assert any("forbidden private literal" in error for error in errors)


def test_platform_path_policy_rejects_private_root_paths() -> None:
    errors = validate_platform_path_policy(
        "karolina",
        {"runs_root": "/ceph/hpc/home/eubrieucb/mesouq/runs"},
        label="karolina",
    )

    assert any("forbidden private literal" in error or "forbidden prefix" in error for error in errors)
