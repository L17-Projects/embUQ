from __future__ import annotations

import sys
from pathlib import Path

import yaml

from emb.compression.src import generate


def _write_config(root: Path) -> None:
    config_dir = root / "inference" / "configs" / "production"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "inference_config_compression.yaml").write_text(
        yaml.dump({"debug": 0}),
        encoding="utf-8",
    )


def _write_source_parameters(source_path: Path, obj: str, params: dict) -> None:
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / f"parameters-default.{obj}.yaml").write_text(
        yaml.dump(params),
        encoding="utf-8",
    )


def _write_source_parameter_files(source_path: Path, params: dict, count: int) -> None:
    param_dir = source_path / "parameter"
    param_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        param_dir.joinpath(f"parameters-default{idx:05d}.yaml").write_text(
            yaml.dump(params),
            encoding="utf-8",
        )


def test_generate_sim_parallel_creates_public_artifacts(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    source_path = tmp_path / "source"
    simu_path = tmp_path / "simu"
    source_path.mkdir()
    simu_path.mkdir()
    _write_source_parameters(source_path, "emb", {"buck": 10})

    generate.generate_sim(
        source_path=f"{source_path}/",
        simu_path=f"{simu_path}/",
        par=None,
        obj="emb",
        forward=None,
        hysteresis=None,
        parallel=True,
        g=1,
        N=1,
        first=None,
        numJobs=1,
    )

    assert (simu_path / "parameter" / "parameters-default00001.yaml").exists()
    assert (simu_path / "commands.txt").exists()
    assert (simu_path / "run_HPC.sbatch").exists()


def test_generate_sim_parameter_loop_writes_expected_grid(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    source_path = tmp_path / "source"
    simu_path = tmp_path / "simu"
    source_path.mkdir()
    simu_path.mkdir()
    _write_source_parameters(source_path, "emb", {"buck": 10, "steps": 10})

    generate.generate_sim(
        source_path=f"{source_path}/",
        simu_path=f"{simu_path}/",
        par=[["buck", "1.0", "2.0", "2"], ["steps", "3", "5", "2"]],
        obj="emb",
        forward=None,
        hysteresis=None,
        parallel=True,
        g=1,
        N=1,
        first=None,
        numJobs=1,
    )

    generated = sorted((simu_path / "parameter").glob("parameters-default*.yaml"))
    non_eq = [path for path in generated if "eq" not in path.name]

    assert len(non_eq) == 4
    first_payload = yaml.safe_load(non_eq[0].read_text(encoding="utf-8"))
    assert isinstance(first_payload["steps"], int)


def test_generate_sim_forward_first_writes_equil_and_restart(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    source_path = tmp_path / "source"
    simu_path = tmp_path / "simu"
    source_path.mkdir()
    simu_path.mkdir()
    _write_source_parameters(source_path, "emb", {"buck": 10})
    _write_source_parameter_files(source_path, {"buck": 10}, count=1)

    generate.generate_sim(
        source_path=f"{source_path}/",
        simu_path=f"{simu_path}/",
        par=None,
        obj="emb",
        forward=True,
        hysteresis=None,
        parallel=None,
        g=1,
        N=1,
        first=True,
        numJobs=1,
    )

    commands = (simu_path / "commands.txt").read_text(encoding="utf-8").strip().splitlines()
    assert commands == [
        "bash run.sh --equil 00001eq 2",
        "bash run.sh --restart 00001 2",
    ]
    assert (simu_path / "parameter" / "parameters-default00001eq.yaml").exists()


def test_generate_sim_hysteresis_adds_reverse_sweep(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    source_path = tmp_path / "source"
    simu_path = tmp_path / "simu"
    source_path.mkdir()
    simu_path.mkdir()
    _write_source_parameters(source_path, "emb", {"buck": 10})
    _write_source_parameter_files(source_path, {"buck": 10}, count=2)

    generate.generate_sim(
        source_path=f"{source_path}/",
        simu_path=f"{simu_path}/",
        par=[["buck", "1.0", "2.0", "2"]],
        obj="emb",
        forward=None,
        hysteresis=True,
        parallel=None,
        g=1,
        N=1,
        first=True,
        numJobs=1,
    )

    commands = (simu_path / "commands.txt").read_text(encoding="utf-8").strip().splitlines()
    assert commands[-1] == "bash run.sh --restart 00003 2"
    assert (simu_path / "parameter" / "parameters-default00003.yaml").exists()


def test_generate_main_parses_cli_and_delegates(monkeypatch) -> None:
    captured = {}

    def fake_generate_sim(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generate, "generate_sim", fake_generate_sim)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--parallel",
            "-o",
            "emb",
            "-g",
            "2",
            "-N",
            "3",
            "--first",
            "-j",
            "4",
        ],
    )

    generate.main(sys.argv[1:])

    assert captured == {
        "source_path": "",
        "simu_path": "",
        "par": None,
        "obj": "emb",
        "forward": None,
        "hysteresis": None,
        "parallel": True,
        "g": 2,
        "N": 3,
        "first": True,
        "numJobs": 4,
    }
