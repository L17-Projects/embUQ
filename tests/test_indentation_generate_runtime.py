from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load_module(path: Path, key: str):
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_generate_sim_parallel_writes_commands_and_sbatch(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "indentation" / "src" / "generate.py",
        "mesouq_indentation_generate_parallel",
    )

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"dummy": True}), encoding="utf-8")
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir(parents=True)

    monkeypatch.setattr(module, "_resolve_config_file", lambda: str(config))
    monkeypatch.setattr(module.os, "system", lambda cmd: 0)

    module.generate_sim(
        source_path=str(tmp_path) + "/",
        simu_path=str(sim_dir) + "/",
        par=None,
        obj="emb",
        forward=False,
        hysteresis=False,
        parallel=True,
        g=2,
        N=1,
        first=False,
        numJobs=1,
    )

    commands = (sim_dir / "commands.txt").read_text(encoding="utf-8")
    sbatch = (sim_dir / "run_HPC.sbatch").read_text(encoding="utf-8")
    assert "bash run.sh --equil 00001eq 4" in commands
    assert "#SBATCH --gres=gpu:2" in sbatch


def test_generate_sim_parameter_loop_writes_parameter_files(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "indentation" / "src" / "generate.py",
        "mesouq_indentation_generate_parameter_loop",
    )

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"dummy": True}), encoding="utf-8")
    sim_dir = tmp_path / "sim"
    parameter_dir = sim_dir / "parameter"
    sim_dir.mkdir(parents=True)
    parameter_dir.mkdir(parents=True)
    (sim_dir / "parameters-default.yaml").write_text(
        yaml.safe_dump({"buck": 1}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_resolve_config_file", lambda: str(config))
    monkeypatch.setattr(module.os, "system", lambda cmd: 0)

    module.generate_sim(
        source_path=str(tmp_path) + "/",
        simu_path=str(sim_dir) + "/",
        par=[["buck", "1.0", "2.0", "2"]],
        obj="emb",
        forward=True,
        hysteresis=False,
        parallel=False,
        g=1,
        N=1,
        first=True,
        numJobs=1,
    )

    assert (parameter_dir / "parameters-default00001.yaml").exists()
    assert (parameter_dir / "parameters-default00002.yaml").exists()
    commands = (sim_dir / "commands.txt").read_text(encoding="utf-8")
    assert "--restart 00001" in commands
