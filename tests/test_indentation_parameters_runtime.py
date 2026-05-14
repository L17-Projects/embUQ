from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import yaml


def _load_module(path: Path, key: str):
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_parameters_emb_runtime_outputs(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "emb" / "indentation" / "src" / "parameters.py",
        "mesouq_indentation_parameters",
    )

    simu_dir = tmp_path / "sim"
    (simu_dir / "parameter").mkdir(parents=True)
    (simu_dir / "microbubble").mkdir(parents=True)
    (simu_dir / "mesh").mkdir(parents=True)

    defaults_path = repo_root / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    defaults["numObjects"] = 2
    (simu_dir / "parameter" / "parameters-default00001.yaml").write_text(
        yaml.safe_dump(defaults),
        encoding="utf-8",
    )

    class _FakeMesh:
        vertices = [(0.0, 0.0, 0.0)] * 12
        area = 2.0
        volume = 1.0

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    fake_trimesh = types.ModuleType("trimesh")
    fake_trimesh.load = lambda path: _FakeMesh()
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    module.write_parameters(source_path="", simu_path=str(simu_dir) + "/", simnum="00001")

    assert (simu_dir / "parameter" / "parameters00001.yaml").exists()
    assert (simu_dir / "parameter" / "parameters.prms00001.yaml").exists()
    assert (simu_dir / "posq.txt").exists()
