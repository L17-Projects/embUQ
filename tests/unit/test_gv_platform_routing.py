from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


def test_gv_platform_hpc_runtime_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_gv_runtime.py"),
        "gv_platform_hpc_runtime_wrapper_unknown_site_test",
    )
    monkeypatch.setenv("HPC_SITE", "unknown")

    with pytest.raises(SystemExit, match="Unsupported HPC_SITE"):
        module.main(["--selection", "gv:stretching"])


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


def test_gv_runtime_workflow_helper_edge_cases(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_helper_edges_test",
    )

    args = Namespace(
        structure="gv",
        selection=None,
        experiment="torsion",
        geometry_id="gv_rad2_height14_28",
        radius=2.0,
        height=14.28,
        control=["theta=0.03"],
        output_root=None,
        run_tag="tagged-run",
        platform="vega",
        include_experimental=True,
    )

    assert module._resolve_output_root(args).name == "tagged-run"
    assert module._build_runtime_argv(args) == [
        "--structure",
        "gv",
        "--experiment",
        "torsion",
        "--geometry-id",
        "gv_rad2_height14_28",
        "--radius",
        "2.0",
        "--height",
        "14.28",
        "--control",
        "theta=0.03",
        "--run-tag",
        "tagged-run",
        "--include-experimental",
    ]

    explicit_root = tmp_path / "explicit-root"
    args.output_root = str(explicit_root)
    assert module._resolve_output_root(args) == explicit_root.resolve()

    args.selection = "gv:torsion"
    assert module._build_runtime_argv(args)[:4] == ["--structure", "gv", "--selection", "gv:torsion"]

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        module._load_runtime_manifest(list_json)

    missing_json = tmp_path / "missing.json"
    missing_json.write_text(json.dumps({"structure": "gv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        module._load_runtime_manifest(missing_json)

    bad_controls_json = tmp_path / "bad-controls.json"
    bad_controls_json.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": [],
                "control_id": "theta_0_03",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
                "work_dir": str(tmp_path / "work"),
                "output_root": str(tmp_path),
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controls"):
        module._load_runtime_manifest(bad_controls_json)


def test_gv_runtime_render_manifest_records_classified_known_issues(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_known_issues_render_test",
    )

    manifest = {
        "structure": "gv",
        "experiment": "shear_flow",
        "geometry": "gv_rad2_height14_28",
        "control_id": "theta_0_03",
        "controls": {"theta": 0.03},
        "dataset_id": "gv:shear_flow:gv_rad2_height14_28:theta_0_03",
        "output_root": str(tmp_path / "runtime"),
        "work_dir": str(tmp_path / "runtime" / "work"),
        "geometry_spec": {"id": "gv_rad2_height14_28", "parameters": {"radius": 2.0, "height": 14.28}},
        "source_root": str(tmp_path / "sources" / "shear_flow"),
        "runtime_package": "mirheoOBMD",
        "experimental": True,
        "known_issues": [
            {
                "id": "bouncer_overflow",
                "summary": "Observed triangle overlap candidates.",
                "evidence": "gv/shear_flow/src/fixtures/bouncer.txt",
                "severity": "Blocking",
            },
            {
                "id": "artifact_stale",
                "summary": "Reference artifact timestamp drift.",
                "evidence": "artifacts/state.csv",
                "severity": "warning",
            },
        ],
        "commands": [],
        "analysis_commands": [],
        "generated_subdirs": [],
    }
    manifest_path = tmp_path / "runtime" / module.GV_RUNTIME_MANIFEST

    def _fake_dry_run(argv: list[str]) -> int:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        (manifest_path.parent / "work").mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    def _forbidden_run(*args, **kwargs) -> None:
        raise AssertionError("runtime commands should not execute under --dry-run")

    monkeypatch.setattr(module.subprocess, "run", _forbidden_run)

    rc = module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path / "runtime"), "--dry-run"])
    assert rc == 0

    rendered = json.loads((tmp_path / "runtime" / module.GV_RUNTIME_RENDER_MANIFEST).read_text(encoding="utf-8"))
    runtime_stage = rendered["runtime_stage"]
    assert runtime_stage["experimental"] is True
    assert runtime_stage["runtime_package"] == "mirheoOBMD"
    assert runtime_stage["runtime_package_source"] == str(manifest["source_root"])
    assert runtime_stage["blocked_issue_count"] == 1
    assert runtime_stage["known_issues"] == [
        {
            "id": "bouncer_overflow",
            "summary": "Observed triangle overlap candidates.",
            "evidence": "gv/shear_flow/src/fixtures/bouncer.txt",
            "severity": "blocking",
            "classification": "experimental_blocked",
        },
        {
            "id": "artifact_stale",
            "summary": "Reference artifact timestamp drift.",
            "evidence": "artifacts/state.csv",
            "severity": "warning",
            "classification": "observed",
        },
    ]

    assert module._normalize_command_value(("python3", "generate.py")) == ["python3", "generate.py"]
    with pytest.raises(ValueError, match="Expected a command list"):
        module._normalize_command_value("python3 generate.py")

    work_dir = tmp_path / "work"
    assert module._normalize_cwd(None, work_dir=work_dir) == work_dir
    assert module._normalize_cwd("analysis", work_dir=work_dir) == work_dir / "analysis"
    with pytest.raises(ValueError, match="string cwd"):
        module._normalize_cwd(3, work_dir=work_dir)

    empty_manifest = {"commands": [], "analysis_commands": [], "work_dir": str(work_dir)}
    assert module._as_command_list(empty_manifest) == []

    malformed_manifest = {"commands": ["python3 generate.py"], "work_dir": str(work_dir)}
    with pytest.raises(ValueError, match="malformed"):
        module._as_command_list(malformed_manifest)

    empty_command_manifest = {"commands": [{"argv": []}], "work_dir": str(work_dir)}
    with pytest.raises(ValueError, match="is empty"):
        module._as_command_list(empty_command_manifest)

    scheduled = module._ensure_scheduler_scripts(
        command_list=[
            ((), work_dir),
            (("python3", "generate.py"), work_dir),
            (("sbatch",), work_dir),
        ],
        manifest={"work_dir": str(work_dir)},
        platform="vega",
    )
    assert scheduled == []

    existing_scheduler = work_dir / "existing.sbatch"
    existing_scheduler.parent.mkdir(parents=True, exist_ok=True)
    existing_scheduler.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    assert module._ensure_scheduler_scripts(
        command_list=[(("sbatch", "existing.sbatch"), work_dir)],
        manifest={"work_dir": str(work_dir)},
        platform="vega",
    ) == []

    module._ensure_generated_directories(work_dir, {"generated_subdirs": ["logs", 3]})
    assert (work_dir / "logs").is_dir()

    calls: list[tuple[tuple[str, ...], str]] = []

    def _fake_run(command, *, cwd, capture_output, text, check):
        calls.append((tuple(command), cwd))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    records, rc = module._run_commands([(("python3", "generate.py"), work_dir)], dry_run=False)
    assert rc == 0
    assert records == [
        {
            "argv": ["python3", "generate.py"],
            "cwd": str(work_dir),
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "status": "completed",
        }
    ]
    assert calls == [(("python3", "generate.py"), str(work_dir))]

    def _fake_failed_run(command, *, cwd, capture_output, text, check):
        return SimpleNamespace(returncode=17, stdout="", stderr="failed")

    monkeypatch.setattr(module.subprocess, "run", _fake_failed_run)
    records, rc = module._run_commands([(("bash", "run.sh"), work_dir)], dry_run=False)
    assert rc == 17
    assert records[0]["status"] == "failed"


def test_gv_runtime_workflow_raises_for_failed_dry_run(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_failed_dry_run_test",
    )
    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", lambda _argv: 2)

    with pytest.raises(RuntimeError, match="dry-run failed"):
        module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path)])


def test_gv_runtime_workflow_raises_for_failed_command(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_failed_command_test",
    )

    def _fake_dry_run(argv: list[str]) -> int:
        parsed_root = Path(argv[argv.index("--output-root") + 1]).expanduser().resolve()
        parsed_root.mkdir(parents=True, exist_ok=True)
        work_dir = parsed_root / "torsion" / "gv_rad2_height14_28" / "theta_0_03" / "work"
        manifest = {
            "structure": "gv",
            "experiment": "torsion",
            "geometry": "gv_rad2_height14_28",
            "controls": {"theta": 0.03},
            "control_id": "theta_0_03",
            "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
            "output_root": str(parsed_root),
            "work_dir": str(work_dir),
            "commands": [{"argv": ["bash", "run.sh"], "cwd": "{work_dir}"}],
        }
        (parsed_root / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="command execution failed"):
        module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path / "runtime")])
