from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

def _load_doctor_module():
    spec = importlib.util.spec_from_file_location(
        "mesouq_test_vega_doctor_runtime",
        Path("scripts/platforms/hpc/doctor_runtime.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_karolina_doctor_module():
    spec = importlib.util.spec_from_file_location(
        "mesouq_test_karolina_doctor_runtime",
        Path("scripts/platforms/karolina/doctor_karolina.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_doctor_gv_runtime_diagnostics_includes_scale_space_and_mirheo_checks(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    (repo_root / "extern" / "korali").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))
    (runtime_root / "mirheo").mkdir(parents=True, exist_ok=True)
    (runtime_root / "mirheo" / "env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (runtime_root / "mirheo" / "source_snapshot.json").write_text("{}", encoding="utf-8")
    (runtime_root / "env").mkdir(parents=True, exist_ok=True)
    (runtime_root / "env" / "env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    mirheo_install = runtime_root / "mirheo" / "install" / "lib"
    mirheo_install.mkdir(parents=True)
    (mirheo_install / "libmirheo-test.so").write_text("binary", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    import_map = {
        "meso_uq": str(repo_root / "src" / "meso_uq.py"),
        "mpi4py": "/x/mpi4py.py",
        "pybind11": "/x/pybind11.py",
        "mesonbuild": "/x/mesonbuild.py",
        "h5py": "/x/h5py.py",
        "mirheo": "/x/mirheo.py",
        "MDAnalysis": "/x/MDAnalysis.py",
    }

    monkeypatch.setattr(module, "_command_path", lambda _: "/bin/cmd")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _: "1.0")
    monkeypatch.setattr(module, "_python_module_spec", lambda name: import_map.get(name, ""))
    monkeypatch.setenv("MESOUQ_OPENMPI_LIB_DIR", "/usr/local/lib/openmpi")
    monkeypatch.setattr(module, "_resolve_scale_space_binary", lambda: ("/usr/bin/scale_space", "PATH"))
    monkeypatch.setattr(module, "_resolve_scale_space_dynamic_libs", lambda _binary: (True, "ok"))

    report = module.collect_diagnostics("python", with_gv_runtime=True)

    check_map = {entry["name"]: entry for entry in report["checks"]}
    assert report["with_gv_runtime"] is True
    assert check_map["repo_local_unified_env_script"]["status"] == "ok"
    assert check_map["mirheo_libmirheo"]["status"] == "ok"
    assert check_map["scale_space_binary:PATH"]["status"] == "ok"
    assert check_map["python:MDAnalysis"]["status"] == "ok"
    assert check_map["openmpi_lib_path"]["status"] == "ok"


def test_scale_space_resolution_checks_env_root_path_and_missing_libs(tmp_path, monkeypatch):
    module = _load_doctor_module()
    direct = tmp_path / "direct_scale_space"
    direct.write_text("binary", encoding="utf-8")
    monkeypatch.setenv("GV_SCALE_SPACE_BINARY", str(direct))
    assert module._resolve_scale_space_binary() == (str(direct), "GV_SCALE_SPACE_BINARY")

    monkeypatch.delenv("GV_SCALE_SPACE_BINARY")
    cgal_root = tmp_path / "cgal"
    direct_cgal = cgal_root / "scale_space"
    direct_cgal.parent.mkdir(parents=True)
    direct_cgal.write_text("binary", encoding="utf-8")
    monkeypatch.setenv("GV_CGAL_TOOLS_ROOT", str(cgal_root))
    assert module._resolve_scale_space_binary() == (str(direct_cgal), "GV_CGAL_TOOLS_ROOT")

    direct_cgal.unlink()
    nested = cgal_root / "build" / "cgal_scripts" / "scale_space"
    nested.parent.mkdir(parents=True)
    nested.write_text("binary", encoding="utf-8")
    assert module._resolve_scale_space_binary() == (str(nested), "GV_CGAL_TOOLS_ROOT")

    monkeypatch.setattr(module, "_command_path", lambda name: "/usr/bin/scale_space" if name == "scale_space" else "")
    monkeypatch.delenv("GV_CGAL_TOOLS_ROOT")
    assert module._resolve_scale_space_binary() == ("/usr/bin/scale_space", "PATH")

    assert module._resolve_scale_space_dynamic_libs("") == (False, "scale_space binary was not resolved")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="ldd boom"),
    )
    assert module._resolve_scale_space_dynamic_libs("/bad")[1] == "ldd failed: ldd boom"

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="libmpfr.so.6 => not found\n", stderr=""),
    )
    ok, details = module._resolve_scale_space_dynamic_libs("/bad")
    assert ok is False
    assert "missing dynamic libs" in details

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="libok.so => /x/libok.so\n", stderr=""),
    )
    assert module._resolve_scale_space_dynamic_libs("/good") == (True, "all dynamic libs resolved")

    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    assert module._resolve_scale_space_binary() == ("", "")


def test_doctor_core_and_tex_diagnostics_cover_warning_paths(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.delenv("LOADEDMODULES", raising=False)
    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")
    monkeypatch.setattr(module, "_kpsewhich", lambda _name: "")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    report = module.collect_diagnostics("python", with_tex=True)
    checks = {entry["name"]: entry for entry in report["checks"]}

    assert report["with_gv_runtime"] is False
    assert checks["loaded_modules"]["status"] == "warn"
    assert checks["command:meson"]["status"] == "warn"
    assert checks["command:ninja"]["status"] == "warn"
    assert checks["python:korali"]["status"] == "warn"
    assert checks["pythonpath:external_korali"]["status"] == "ok"
    assert checks["repo_local_tinytex_env_script"]["status"] == "warn"
    assert checks["tex:helvet.sty"]["status"] == "warn"


def test_doctor_collect_diagnostics_can_target_karolina_site(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    (repo_root / "extern" / "korali").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    runtime_root = tmp_path / "karolina-runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")

    report = module.collect_diagnostics("python", site="karolina", with_gv_runtime=True)

    assert report["site"] == "karolina"
    assert "CUDA/12.4.0" in report["recommended_modules"]
    assert report["paths"]["site_root"] == str(runtime_root.resolve())
    assert report["paths"]["scale_space_binary"] == str((runtime_root / "gv_cgal_tools" / "bin" / "scale_space").resolve())


def test_doctor_uses_env_site_and_karolina_module_profiles(monkeypatch):
    module = _load_doctor_module()
    monkeypatch.setenv("MESOUQ_SITE", "karolina")

    assert module.resolve_hpc_site(env={"MESOUQ_SITE": "karolina"}) == "karolina"
    assert module._recommended_modules("karolina", with_mirheo=False, with_gv_runtime=False) == list(
        module.DEFAULT_KAROLINA_MODULES
    )
    assert module._recommended_modules("karolina", with_mirheo=True, with_gv_runtime=False) == list(
        module.DEFAULT_KAROLINA_MIRHEO_MODULES
    )


def test_karolina_doctor_propagates_site_env(monkeypatch):
    module = _load_karolina_doctor_module()
    captured = {}

    def fake_call(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)

    assert module.main(["--strict", "--with-gv-runtime"]) == 0

    assert captured["args"][2:4] == ["--site", "karolina"]
    assert captured["args"][-2:] == ["--strict", "--with-gv-runtime"]
    assert captured["args"][1].endswith("scripts/platforms/hpc/doctor_hpc.py")


def test_doctor_mirheo_diagnostics_cover_source_warnings(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOADEDMODULES", "Python/3.10")
    monkeypatch.setattr(module, "_command_path", lambda name: "/bin/mpicxx" if name == "mpicxx" else "")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")
    monkeypatch.setattr(module, "load_mirheo_source_lock", lambda _root, **_kwargs: {"source_path": "/missing/mirheo"})
    monkeypatch.setattr(
        module,
        "resolve_mirheo_source",
        lambda _root, **_kwargs: (_ for _ in ()).throw(RuntimeError("no source")),
    )
    monkeypatch.delenv("MESOUQ_OPENMPI_LIB_DIR", raising=False)

    report = module.collect_diagnostics("python", with_mirheo=True)
    checks = {entry["name"]: entry for entry in report["checks"]}

    assert checks["mirheo_source"]["status"] == "warn"
    assert checks["mirheo_source"]["details"] == "no source"
    assert "scale_space_binary" not in checks
    assert checks["openmpi_lib_path"]["status"] == "warn"


def test_doctor_main_reports_json_and_strict_failure(monkeypatch, capsys):
    module = _load_doctor_module()
    report = {
        "repo_root": "/repo",
        "python_bin": "python",
        "python_version": "3.10",
        "hostname": "host",
        "recommended_modules": ["Python/3.10"],
        "checks": [{"name": "x", "status": "warn", "details": "missing"}],
    }
    monkeypatch.setattr(module, "collect_diagnostics", lambda *_args, **_kwargs: report)

    assert module.main(["--json"]) == 0
    assert '"repo_root": "/repo"' in capsys.readouterr().out
    assert module.main(["--strict"]) == 1
    assert "[WARN] x: missing" in capsys.readouterr().out


def test_doctor_core_diagnostics_warn_when_korali_import_is_not_repo_local(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setenv("LOADEDMODULES", "Python/3.10")
    monkeypatch.setattr(module, "_command_path", lambda name: "/bin/tool" if name in {"python", "mpicxx", "nvcc", "pkg-config", "meson", "ninja"} else "")
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "1.0")

    def fake_spec(name: str) -> str:
        if name == "meso_uq":
            return str(repo_root / "src" / "meso_uq.py")
        if name == "korali":
            return "/opt/korali/korali/__init__.py"
        return f"/x/{name}.py"

    monkeypatch.setattr(module, "_python_module_spec", fake_spec)

    report = module.collect_diagnostics("python")
    checks = {entry["name"]: entry for entry in report["checks"]}

    assert checks["python:korali"]["status"] == "warn"
    assert checks["python:korali"]["details"] == "/opt/korali/korali/__init__.py"


def test_doctor_core_diagnostics_accepts_repo_local_korali_via_path_alias(tmp_path, monkeypatch):
    module = _load_doctor_module()
    real_repo = tmp_path / "real" / "repo"
    real_repo.mkdir(parents=True)
    alias_repo = tmp_path / "alias_repo"
    alias_repo.symlink_to(real_repo, target_is_directory=True)
    (real_repo / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (real_repo / "extern" / "korali").mkdir(parents=True)
    runtime_root = real_repo / "runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))
    korali_init = (
        runtime_root
        / "korali"
        / "install"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "korali"
        / "__init__.py"
    )
    korali_init.parent.mkdir(parents=True)
    korali_init.write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", alias_repo)
    monkeypatch.setenv("LOADEDMODULES", "Python/3.10")
    monkeypatch.setattr(module, "_command_path", lambda _name: "/bin/tool")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "1.0")

    def fake_spec(name: str) -> str:
        if name == "meso_uq":
            return str(real_repo / "src" / "meso_uq.py")
        if name == "korali":
            return str(korali_init)
        return f"/x/{name}.py"

    monkeypatch.setattr(module, "_python_module_spec", fake_spec)

    report = module.collect_diagnostics("python")
    checks = {entry["name"]: entry for entry in report["checks"]}

    assert checks["python:korali"]["status"] == "ok"
    assert checks["python:korali"]["details"] == str(korali_init)


def test_doctor_core_diagnostics_omits_openmpi_lib_check_when_mpicxx_is_absent(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")

    report = module.collect_diagnostics("python")
    check_names = {entry["name"] for entry in report["checks"]}

    assert "openmpi_lib_path" not in check_names
