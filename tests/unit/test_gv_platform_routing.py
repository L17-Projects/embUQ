from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gv_platform_vega_runtime_wrapper_forwards_platform_flag(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_gv_runtime.py"),
        "gv_platform_vega_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)

    rc = module.main(["--selection", "gv:stretching"])

    assert rc == 0
    assert captured
    rendered = " ".join(captured[0])
    assert "scripts/workflows/gv/run_gv_runtime.py" in rendered
    assert "--platform vega" in rendered
    assert "--selection gv:stretching" in rendered


def test_gv_platform_karolina_runtime_wrapper_forwards_platform_flag(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_gv_runtime.py"),
        "gv_platform_karolina_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)

    rc = module.main(["--selection", "gv:torsion"])

    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert "scripts/workflows/gv/run_gv_runtime.py" in joined
    assert "--platform karolina" in joined
    assert "--selection gv:torsion" in joined


def test_gv_platform_hpc_runtime_wrapper_dispatches_by_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_gv_runtime.py"),
        "gv_platform_hpc_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setenv("HPC_SITE", "karolina")

    rc = module.main(["--selection", "gv:shear_flow"])

    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert "scripts/platforms/karolina/run_gv_runtime.py" in joined
    assert "--selection gv:shear_flow" in joined


def test_gv_runtime_rendering_targets_staged_work_dir_and_generates_scheduler(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_test",
    )

    def _fake_dry_run(argv: list[str]) -> int:
        parsed_root = Path(argv[argv.index("--output-root") + 1]).expanduser().resolve()
        parsed_root.mkdir(parents=True, exist_ok=True)
        work_dir = (
            parsed_root
            / "shear_flow"
            / "gv_rad2_height14_28"
            / "ptan_0_4__afsi_0__bpress_-91"
            / "work"
        )
        manifest = {
            "structure": "gv",
            "experiment": "shear_flow",
            "geometry": "gv_rad2_height14_28",
            "controls": {
                "ptan": 0.4,
                "afsi": 0.0,
                "bpress": -91.0,
            },
            "control_id": "ptan_0_4__afsi_0__bpress_-91",
            "dataset_id": "gv:shear_flow:gv_rad2_height14_28:ptan_0_4__afsi_0__bpress_-91",
            "output_root": str(parsed_root),
            "work_dir": str(work_dir),
            "commands": [
                {
                    "argv": ["python3", "generate.py", "--object", "gv", "--parallel", "--first"],
                    "cwd": "{work_dir}",
                },
                {"argv": ["sbatch", "run_HPC.sbatch"], "cwd": "{work_dir}"},
            ],
            "analysis_commands": [
                {"argv": ["bash", "commands.txt"], "cwd": "analysis"},
            ],
            "generated_subdirs": ["logs", "analysis", "mesh"],
            "runtime_package": "mirheoOBMD",
        }
        (parsed_root / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)

    captured_runs: list[tuple[tuple[str, ...], str]] = []

    def _run(commands, **kwargs) -> None:
        captured_runs.append((tuple(commands), kwargs.get("cwd", "")))
        raise AssertionError("runtime commands should not execute under --dry-run")

    monkeypatch.setattr(module.subprocess, "run", _run)

    rc = module.main(
        [
            "--selection",
            "gv:shear_flow",
            "--output-root",
            str(tmp_path / "runtime"),
            "--include-experimental",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not captured_runs
    runtime_output_root = tmp_path / "runtime"
    manifest = json.loads((runtime_output_root / "gv_runtime_render_manifest.json").read_text(encoding="utf-8"))
    work_dir = runtime_output_root / "shear_flow" / "gv_rad2_height14_28" / "ptan_0_4__afsi_0__bpress_-91" / "work"
    assert work_dir.is_dir()
    commands_txt = work_dir / "commands.txt"
    assert commands_txt.is_file()
    assert (work_dir / "run_HPC.sbatch").is_file()
    assert manifest["dry_run"] is True
    assert manifest["commands"][0]["status"] == "skipped-dry-run"
    assert manifest["generated_scheduler_scripts"] == ["run_HPC.sbatch"]
    contents = commands_txt.read_text(encoding="utf-8")
    assert "generate.py" in contents
    assert "run_HPC.sbatch" in contents
