from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_installed_package_smoke.py"
    module_name = "run_installed_package_smoke_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class _FakeTemporaryDirectory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __enter__(self) -> str:
        return str(self.root)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_parser_defaults_and_smoke_surface_modules():
    module = _load_module()

    parsed = module._build_arg_parser().parse_args([])
    assert parsed.repo_root == str(module.REPO_ROOT)
    assert parsed.python_bin
    assert parsed.wheel is None
    assert "meso_uq.public_api" in module.SMOKE_MODULES
    assert "meso_uq.config.loader" in module.SMOKE_MODULES
    assert "meso_uq.artifacts.policy" in module.SMOKE_MODULES
    assert module.HEAVY_OPTIONAL_MODULES == ("torch", "pyro", "matplotlib", "mpi4py", "mirheo", "korali", "slurm")


def test_wheel_resolution_prefers_the_latest_candidate(tmp_path):
    module = _load_module()
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    first = wheel_dir / "mesouq-0.1.0-py3-none-any.whl"
    second = wheel_dir / "mesouq-0.1.1-py3-none-any.whl"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert module._resolve_wheel_path(wheel_dir) == second.resolve()


def test_repo_path_filtering_removes_repo_and_src_entries(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    repo_root.mkdir()
    src_root.mkdir()
    entries = [str(repo_root), str(src_root), "/opt/venv/lib/python3.11/site-packages"]

    kept, removed = module._filter_repo_paths(entries, repo_root, src_root)

    assert removed == [str(repo_root), str(src_root)]
    assert kept == ["/opt/venv/lib/python3.11/site-packages"]


def test_dependency_paths_preserve_repo_local_virtualenv_site_packages(monkeypatch, tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    site_packages = repo_root / "runtime" / "venv" / "lib" / "python3.11" / "site-packages"
    user_site = repo_root / ".local" / "lib" / "python3.11" / "site-packages"
    external_site = tmp_path / "external" / "lib" / "python3.11" / "site-packages"
    for path in (src_root, site_packages, user_site, external_site):
        path.mkdir(parents=True)

    monkeypatch.setattr(module.site, "getsitepackages", lambda: [str(repo_root), str(src_root), str(site_packages), str(external_site)])
    monkeypatch.setattr(module.site, "getusersitepackages", lambda: str(user_site))

    entries = module._dependency_path_entries(repo_root)

    assert str(site_packages.resolve()) in entries
    assert str(user_site.resolve()) in entries
    assert str(external_site.resolve()) in entries
    assert str(repo_root.resolve()) not in entries
    assert str(src_root.resolve()) not in entries


def test_rendered_probe_reports_repo_root_and_heavy_module_guards(tmp_path):
    module = _load_module()
    probe = module._render_smoke_probe(repo_root=tmp_path)

    assert "repository-root import leakage detected" in probe
    assert "lightweight smoke loaded optional heavy modules" in probe
    assert "_is_site_packages_path" in probe
    assert "meso_uq.public_api" in probe
    assert "torch" in probe


def test_run_installed_package_smoke_uses_existing_wheel_without_building(monkeypatch, tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wheel = tmp_path / "mesouq-0.1.0-py3-none-any.whl"
    wheel.write_text("wheel", encoding="utf-8")

    calls: list[tuple[str, object]] = []

    def fail_build(*args, **kwargs):
        raise AssertionError("build should not be called when a wheel is supplied")

    def fake_create_venv(venv_root: Path, python_bin: str) -> Path:
        calls.append(("venv", venv_root))
        venv_root.mkdir(parents=True, exist_ok=True)
        python = venv_root / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        return python

    def fake_install(python_bin: Path, wheel_path: Path) -> None:
        calls.append(("install", wheel_path))

    def fake_probe(python_bin: Path, *, repo_root: Path, workdir: Path, modules, heavy_modules):
        calls.append(("probe", tuple(modules)))
        workdir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(stdout=json.dumps({"imported": list(modules)}), stderr="", returncode=0)

    monkeypatch.setattr(module, "_build_wheel", fail_build)
    monkeypatch.setattr(module, "_create_venv", fake_create_venv)
    monkeypatch.setattr(module, "_install_wheel", fake_install)
    monkeypatch.setattr(module, "_run_smoke_probe", fake_probe)

    @contextmanager
    def fake_tempdir(prefix: str):
        root = tmp_path / "temp"
        root.mkdir(exist_ok=True)
        yield str(root)

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", fake_tempdir)

    result = module.run_installed_package_smoke(repo_root=repo_root, wheel=wheel)

    assert result.wheel_path == wheel.resolve()
    assert result.imported_modules[0] == "meso_uq"
    assert ("install", wheel.resolve()) in calls
    assert any(item[0] == "probe" for item in calls)


def test_run_installed_package_smoke_builds_when_no_wheel_is_supplied(monkeypatch, tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    built_wheel = tmp_path / "wheelhouse" / "mesouq-0.1.0-py3-none-any.whl"
    built_wheel.parent.mkdir()
    built_wheel.write_text("wheel", encoding="utf-8")

    calls: list[tuple[str, object]] = []

    def fake_build(repo_root_arg: Path, wheelhouse: Path, python_bin: str) -> Path:
        calls.append(("build", wheelhouse))
        return built_wheel

    def fake_create_venv(venv_root: Path, python_bin: str) -> Path:
        calls.append(("venv", venv_root))
        venv_root.mkdir(parents=True, exist_ok=True)
        python = venv_root / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        return python

    def fake_install(python_bin: Path, wheel_path: Path) -> None:
        calls.append(("install", wheel_path))

    def fake_probe(python_bin: Path, *, repo_root: Path, workdir: Path, modules, heavy_modules):
        calls.append(("probe", tuple(modules)))
        workdir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(stdout=json.dumps({"imported": list(modules)}), stderr="", returncode=0)

    monkeypatch.setattr(module, "_build_wheel", fake_build)
    monkeypatch.setattr(module, "_create_venv", fake_create_venv)
    monkeypatch.setattr(module, "_install_wheel", fake_install)
    monkeypatch.setattr(module, "_run_smoke_probe", fake_probe)

    @contextmanager
    def fake_tempdir(prefix: str):
        root = tmp_path / "temp"
        root.mkdir(exist_ok=True)
        yield str(root)

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", fake_tempdir)

    result = module.run_installed_package_smoke(repo_root=repo_root)

    assert result.wheel_path == built_wheel
    assert any(item[0] == "build" for item in calls)
    assert any(item[0] == "install" and item[1] == built_wheel for item in calls)


def test_logged_command_raises_with_captured_failure_output(monkeypatch, tmp_path):
    module = _load_module()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="repository-root import leakage detected: /tmp/repo/src/meso_uq/public_api.py",
        ),
    )

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        module._run_logged_command(["python", "-c", "pass"], cwd=tmp_path)

    assert "repository-root import leakage detected" in str(excinfo.value.stderr)
