from __future__ import annotations

import importlib.util
from pathlib import Path

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
