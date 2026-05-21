from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from meso_uq.platforms.korali_runtime import (
    KORALI_VENDOR_RELATIVE_ROOT,
    build_korali_runtime_validation_command,
    validate_korali_runtime_contract,
)


HEAVY_OPTIONAL_MODULES = ("torch", "pyro", "matplotlib", "mpi4py", "korali")


def _make_repo(tmp_path: Path, *, vendor: bool = False) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "extern").mkdir()
    if vendor:
        (repo_root / KORALI_VENDOR_RELATIVE_ROOT).mkdir(parents=True)
    return repo_root


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_korali_runtime_contract_rejects_missing_vendor_root(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=False)

    report = validate_korali_runtime_contract(repo_root, env={})

    assert not report.ok
    assert any("Missing vendored Korali source root" in error for error in report.errors)


def test_validate_korali_runtime_contract_accepts_existing_vendor_root(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=True)
    pythonpath_hint = f"{repo_root / 'src'}:{repo_root}"
    path_hint = "/usr/bin:/bin"
    build_root = tmp_path / "build"
    build_root.mkdir()
    library_path = tmp_path / "lib"
    library_path.mkdir()

    report = validate_korali_runtime_contract(
        repo_root,
        env={"MESOUQ_SITE": "karolina", "MESOUQ_SITE_RUNTIME_ROOT": str(tmp_path / "runtime")},
        build_root=build_root,
        library_paths=(library_path,),
        pythonpath_hint=pythonpath_hint,
        path_hint=path_hint,
    )

    assert report.ok
    assert report.vendor_root == repo_root / KORALI_VENDOR_RELATIVE_ROOT


def test_validate_korali_runtime_contract_reports_missing_env_hints(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=True)

    report = validate_korali_runtime_contract(repo_root, env={})

    assert not report.ok
    assert any("MESOUQ_SITE was not supplied" in warning for warning in report.warnings)
    assert any("MESOUQ_SITE_RUNTIME_ROOT was not supplied" in error for error in report.errors)
    assert any("No PYTHONPATH hint was supplied" in warning for warning in report.warnings)
    assert any("No PATH hint was supplied" in warning for warning in report.warnings)


def test_validate_korali_runtime_contract_rejects_private_path_hints(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=True)

    report = validate_korali_runtime_contract(
        repo_root,
        env={"MESOUQ_SITE": "karolina"},
        pythonpath_hint=f"/ceph/hpc/home/eubrieucb/mesouq/src:{repo_root / 'src'}",
    )

    assert not report.ok
    assert any("forbidden private path prefix" in error for error in report.errors)


def test_build_korali_runtime_validation_command_shape(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=True)
    build_root = tmp_path / "build"
    build_root.mkdir()
    library_path = tmp_path / "lib"
    library_path.mkdir()

    command = build_korali_runtime_validation_command(
        repo_root,
        env={"MESOUQ_SITE": "karolina"},
        build_root=build_root,
        library_paths=(library_path,),
        pythonpath_hint=f"{repo_root / 'src'}:{repo_root}",
        path_hint="/usr/bin:/bin",
    )

    assert command.argv[:3] == (sys.executable, "-m", "meso_uq.platforms.korali_runtime")
    assert "--repo-root" in command.argv
    assert "--vendor-root" in command.argv
    assert "--strict" in command.argv
    assert command.cwd == repo_root
    assert command.env


def test_korali_runtime_module_executes_cli(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path, vendor=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["MESOUQ_SITE_RUNTIME_ROOT"] = str(tmp_path / "runtime")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "meso_uq.platforms.korali_runtime",
            "--repo-root",
            str(repo_root),
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"ok": true' in result.stdout
    assert "Korali runtime validation" not in result.stdout


def test_korali_runtime_module_import_is_light() -> None:
    code = """
import sys
import meso_uq.platforms.korali_runtime
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'korali') if name in sys.modules]
assert loaded == [], loaded
"""

    result = _run_import_probe(code)

    assert result.returncode == 0, result.stderr
