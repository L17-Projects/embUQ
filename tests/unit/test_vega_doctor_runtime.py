from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

def _load_doctor_module():
    spec = importlib.util.spec_from_file_location(
        "mesouq_test_vega_doctor_runtime",
        Path("scripts/platforms/vega/doctor_vega.py"),
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
    (repo_root / "_vega").mkdir()
    (repo_root / "_vega" / "mirheo").mkdir(parents=True, exist_ok=True)
    (repo_root / "_vega" / "mirheo" / "env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (repo_root / "_vega" / "mirheo" / "source_snapshot.json").write_text("{}", encoding="utf-8")
    (repo_root / "_vega" / "gv_venv").mkdir(parents=True, exist_ok=True)
    (repo_root / "_vega" / "gv_venv" / "env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    mirheo_install = repo_root / "_vega" / "mirheo" / "install" / "lib"
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
    assert check_map["repo_local_gv_venv_env_script"]["status"] == "ok"
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
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    monkeypatch.setattr(module, "_pkg_config_version", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")

    report = module.collect_diagnostics("python", site="karolina", with_gv_runtime=True)

    assert report["site"] == "karolina"
    assert "CUDA/12.4.0" in report["recommended_modules"]
    assert report["paths"]["site_root"].endswith("_karolina")
    assert report["paths"]["scale_space_binary"].endswith("_karolina/gv_cgal_tools/bin/scale_space")


def test_doctor_uses_env_site_and_karolina_module_profiles(monkeypatch):
    module = _load_doctor_module()
    monkeypatch.setenv("MESOUQ_SITE", "karolina")

    assert module._default_runtime_site() == "karolina"
    assert module._recommended_modules("karolina", with_mirheo=False, with_gv_runtime=False) == list(
        module.DEFAULT_KAROLINA_MODULES
    )
    assert module._recommended_modules("karolina", with_mirheo=True, with_gv_runtime=False) == list(
        module.DEFAULT_KAROLINA_MIRHEO_MODULES
    )


def test_karolina_doctor_propagates_site_env(monkeypatch):
    module = _load_karolina_doctor_module()
    captured = {}

    def fake_call(args, env):
        captured["args"] = args
        captured["env"] = env
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)

    assert module.main(["--strict", "--with-gv-runtime"]) == 0

    assert captured["env"]["HPC_SITE"] == "karolina"
    assert captured["env"]["MESOUQ_SITE"] == "karolina"
    assert captured["args"][-2:] == ["--strict", "--with-gv-runtime"]
    assert captured["args"][1].endswith("scripts/platforms/vega/doctor_vega.py")


def test_doctor_mirheo_diagnostics_cover_source_and_scale_space_warnings(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
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
    monkeypatch.setattr(module, "_resolve_scale_space_binary", lambda: ("", ""))
    monkeypatch.delenv("MESOUQ_OPENMPI_LIB_DIR", raising=False)

    report = module.collect_diagnostics("python", with_mirheo=True)
    checks = {entry["name"]: entry for entry in report["checks"]}

    assert checks["mirheo_source"]["status"] == "warn"
    assert checks["mirheo_source"]["details"] == "no source"
    assert checks["scale_space_binary"]["status"] == "warn"
    assert "GV_SCALE_SPACE_BINARY" in checks["scale_space_binary"]["details"]
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


def test_doctor_core_diagnostics_omits_openmpi_lib_check_when_mpicxx_is_absent(tmp_path, monkeypatch):
    module = _load_doctor_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='mesouq'\n", encoding="utf-8")
    (repo_root / "extern" / "korali").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(module, "_command_path", lambda _name: "")
    monkeypatch.setattr(module, "_python_module_spec", lambda _name: "")

    report = module.collect_diagnostics("python")
    check_names = {entry["name"] for entry in report["checks"]}

    assert "openmpi_lib_path" not in check_names
