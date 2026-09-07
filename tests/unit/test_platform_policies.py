from __future__ import annotations

import subprocess
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SITE_ENV = "HPC" "_SITE"


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
        "${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/korali/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/env.sh",
        "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh",
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


def test_tracked_files_do_not_reintroduce_legacy_site_env() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    offenders = []
    for relative in completed.stdout.splitlines():
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LEGACY_SITE_ENV in text:
            offenders.append(relative)

    assert offenders == []


def test_mirheo_bootstrap_defaults_to_full_double_precision() -> None:
    script = REPO_ROOT / "scripts" / "platforms" / "hpc" / "bootstrap_mirheo.sh"
    text = script.read_text(encoding="utf-8")

    assert '"$runtime_python" -m pip install h5py MDAnalysis PyYAML pydantic' in text
    assert "-DMIR_DOUBLE_PRECISION=ON" in text
    assert "-DMIR_MEMBRANE_DOUBLE=ON" in text
    assert "-DMIR_ROD_DOUBLE=ON" in text
    assert "-DMIR_DOUBLE_PRECISION=OFF" not in text
    assert "-DMIR_MEMBRANE_DOUBLE=OFF" not in text
    assert "-DMIR_ROD_DOUBLE=OFF" not in text


def test_vega_sbatch_scripts_use_rebuilt_openmpi_module() -> None:
    sbatch_root = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch"
    offenders = []
    for path in sorted(sbatch_root.rglob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if "openmpi/4.1.2.1" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []

    validation_matrix = sbatch_root / "validation_matrix.sbatch"
    assert "OpenMPI/4.1.4-GCC-12.2.0" in validation_matrix.read_text(encoding="utf-8")


def test_vega_gpu_sbatch_scripts_exclude_known_bad_gpu_node() -> None:
    sbatch_root = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch"
    offenders = []
    for path in sorted(sbatch_root.rglob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if "#SBATCH --gres=gpu:1" in text and "#SBATCH --exclude=gn10" not in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_tracked_files_do_not_reintroduce_split_site_env_paths() -> None:
    retired_tokens = [
        f"{site}/{env_name}"
        for site in ("_vega", "_karolina")
        for env_name in ("venv", "gv" + "_venv")
    ]
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    offenders = []
    for relative in completed.stdout.splitlines():
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = [token for token in retired_tokens if token in text]
        if found:
            offenders.append((relative, found))

    assert offenders == []
