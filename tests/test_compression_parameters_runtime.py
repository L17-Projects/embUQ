from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from compression.src import parameters


def _write_runtime_config(root: Path) -> None:
    config_dir = root / "inference" / "configs" / "production"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "inference_config_compression.yaml").write_text(
        yaml.dump({"debug": 0}),
        encoding="utf-8",
    )


def _write_default_parameters(path: Path, *, obj_file: str = "emb.off", num_objects: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            {
                "objFile": obj_file,
                "rho_water": 1000.0,
                "rho_gas": 1.2,
                "rhow": 3.0,
                "rhog": 0.1,
                "energyFactor": 1.0,
                "kbol": 1.38e-23,
                "t0": 300.0,
                "Lx": 20.0,
                "Ly": 22.0,
                "Lz": 24.0,
                "ul": 1e-7,
                "visw": 8.9e-4,
                "visg": 1.8e-5,
                "ka_tot": 100.0,
                "kv_tot": 100.0,
                "gamma_dpd": 10.0,
                "fscale": 1.0,
                "yt_fac": 1.0,
                "Yt": 2e7,
                "Yl": 2e7,
                "nu": 0.5,
                "s": 0.25,
                "s_g": 0.25,
                "rc": 1.0,
                "k_fsi": 1.0,
                "kb_fac": 1.0,
                "a3": 0.0,
                "a4": 2.0,
                "b1": 1.5,
                "b2": 4.0,
                "radp": 10.0,
                "numObjects": num_objects,
                "rho_shell": 1000.0,
                "th_fac": 1.0,
                "shell_th": 0.01,
                "subDiv": 2,
                "numsteps": 1000,
                "numsteps_eq": 500,
                "stslik": 100,
                "stslik_eq": 50,
            }
        ),
        encoding="utf-8",
    )


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch, sample: np.ndarray) -> list[str]:
    commands: list[str] = []

    class _Mesh:
        vertices = np.zeros((4, 3))
        area = 12.0
        volume = -8.0

    trimesh_mod = types.ModuleType("trimesh")
    trimesh_mod.load = lambda _path: _Mesh()

    qmc_mod = types.SimpleNamespace(
        Sobol=lambda d, scramble: types.SimpleNamespace(random_base2=lambda m: sample.copy())
    )
    scipy_mod = types.ModuleType("scipy")
    scipy_stats_mod = types.ModuleType("scipy.stats")
    scipy_stats_mod.qmc = qmc_mod
    scipy_mod.stats = scipy_stats_mod

    monkeypatch.setitem(sys.modules, "trimesh", trimesh_mod)
    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stats_mod)
    monkeypatch.setitem(sys.modules, "scipy.stats.qmc", qmc_mod)
    monkeypatch.setattr(parameters.os, "system", lambda cmd: commands.append(cmd) or 0)
    monkeypatch.setattr(np.random, "random", lambda: 0.25)

    return commands


def test_write_parameters_emb_single_object_writes_public_runtime_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_runtime_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    simu_path = tmp_path / "simu"
    (simu_path / "microbubble").mkdir(parents=True)
    (simu_path / "mesh").mkdir(parents=True)
    _write_default_parameters(simu_path / "parameter" / "parameters-default00001.yaml", num_objects=1)

    commands = _install_runtime_stubs(monkeypatch, sample=np.array([[0.1, 0.2, 0.3]], dtype=float))

    parameters.write_parameters(
        source_path=f"{simu_path}/",
        simu_path=f"{simu_path}/",
        simnum="00001",
    )

    assert any("sphere_icosphere.py" in command for command in commands)
    assert (simu_path / "parameter" / "parameters.prms00001.yaml").exists()
    assert (simu_path / "parameter" / "parameters00001.yaml").exists()

    posq = np.loadtxt(simu_path / "posq.txt", ndmin=2)
    assert posq.shape == (1, 7)
    assert posq[0].tolist() == pytest.approx([10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0])


def test_write_parameters_multiple_objects_generate_quaternions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_runtime_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    simu_path = tmp_path / "simu"
    (simu_path / "microbubble").mkdir(parents=True)
    (simu_path / "mesh").mkdir(parents=True)
    _write_default_parameters(simu_path / "parameter" / "parameters-default00001.yaml", num_objects=4)

    _install_runtime_stubs(
        monkeypatch,
        sample=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.3, 0.4],
            ],
            dtype=float,
        ),
    )

    parameters.write_parameters(
        source_path=f"{simu_path}/",
        simu_path=f"{simu_path}/",
        simnum="00001",
    )

    posq = np.loadtxt(simu_path / "posq.txt", ndmin=2)
    assert posq.shape == (4, 7)
    assert np.all(posq[:, :3] > 0.0)
    assert np.allclose(np.linalg.norm(posq[:, 3:], axis=1), 1.0)


def test_write_parameters_rejects_non_emb_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_runtime_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    simu_path = tmp_path / "simu"
    (simu_path / "microbubble").mkdir(parents=True)
    (simu_path / "mesh").mkdir(parents=True)
    _write_default_parameters(
        simu_path / "parameter" / "parameters-default00001.yaml",
        obj_file="gv.off",
        num_objects=1,
    )

    _install_runtime_stubs(monkeypatch, sample=np.array([[0.1, 0.2, 0.3]], dtype=float))

    with pytest.raises(NotImplementedError, match="supports emb object preparation only"):
        parameters.write_parameters(
            source_path=f"{simu_path}/",
            simu_path=f"{simu_path}/",
            simnum="00001",
        )


def test_write_parameters_raises_when_no_runtime_config_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    simu_path = tmp_path / "simu"
    (simu_path / "microbubble").mkdir(parents=True)
    (simu_path / "mesh").mkdir(parents=True)
    _write_default_parameters(simu_path / "parameter" / "parameters-default00001.yaml", num_objects=1)

    real_exists = parameters.os.path.exists

    def fake_exists(path: str) -> bool:
        if "inference_config" in path or "baseline_config" in path:
            return False
        return real_exists(path)

    monkeypatch.setattr(parameters.os.path, "exists", fake_exists)

    with pytest.raises(FileNotFoundError, match="Could not find config file"):
        parameters.write_parameters(
            source_path=f"{simu_path}/",
            simu_path=f"{simu_path}/",
            simnum="00001",
        )
