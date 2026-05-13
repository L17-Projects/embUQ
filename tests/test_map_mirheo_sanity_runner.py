import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_map_mirheo_sanity_runner_writes_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_test",
    )

    run_root = (
        tmp_path
        / "gpu_canaries"
        / "compression_full_retry"
        / "runs"
        / "compression"
        / "full-model"
        / "production"
    )
    (run_root / "map_phase3b").mkdir(parents=True)
    (run_root / "map_phase3b" / "phase3b_map_manifest.json").write_text(
        json.dumps({"datasets": {"compression_2.1um": {"diameter_um": 2.1}}}),
        encoding="utf-8",
    )
    (run_root / "map_mirheo").mkdir(parents=True)
    (run_root / "map_mirheo" / "map_mirheo_manifest.json").write_text(
        json.dumps({"status": "passed", "diameters": [{"status": "passed"}]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        if command[0] == "sbatch":
            captured["sbatch_command"] = command
            captured["sbatch_env"] = env
            return _Result(stdout="12345\n")
        if command[0] == "squeue":
            return _Result(stdout="")
        if command[0] == "sacct":
            return _Result(stdout="12345|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    report_root = tmp_path / "report"
    rc = module.main(
        [
            "--selection",
            "compression:full-model:production",
            "--canary-root",
            str(tmp_path / "gpu_canaries"),
            "--report-root",
            str(report_root),
        ]
    )

    assert rc == 0
    report_path = report_root / "map_mirheo_sanity_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["numsteps"] == 200
    assert report["numsteps_eq"] == 200
    assert report["init_directory_policy"]["preexisting_init_dirs_required"] is False
    assert report["init_directory_policy"]["scratch_root_pattern"] == (
        "<lane output>/map_mirheo/_scratch/<dataset_name>"
    )
    assert report["skip_policy"]["missing_init_inputs"].startswith("not skipped")
    assert report["lanes"][0]["selection"] == "compression:full-model:production"
    assert report["lanes"][0]["job_id"] == "12345"
    assert report["lanes"][0]["init_directory_policy"]["scratch_root_pattern"] == str(
        run_root / "map_mirheo" / "_scratch" / "<dataset_name>"
    )
    sbatch_command = captured["sbatch_command"]
    assert isinstance(sbatch_command, list)
    assert "--export" in sbatch_command
    export_arg = sbatch_command[sbatch_command.index("--export") + 1]
    assert "OUTPUT_DIR=" in export_arg
    assert "GPU_TIME_LIMIT=" in export_arg
    assert "MAP_MIRHEO_NUMSTEPS=" in export_arg


def test_map_mirheo_sanity_runner_requires_phase3b_manifest(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_missing_manifest_test",
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("sbatch should not run without a phase3b MAP manifest")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    with pytest.raises(FileNotFoundError, match="Missing phase3b MAP manifest"):
        module.main(
            [
                "--selection",
                "compression:full-model:production",
                "--canary-root",
                str(tmp_path / "gpu_canaries"),
                "--report-root",
                str(tmp_path / "report"),
            ]
        )


def test_map_mirheo_sanity_selection_resolution_covers_new_policy_branches():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_selection_test",
    )

    default = module.resolve_map_mirheo_sanity_selections([])
    assert [module.selection_key(item) for item in default] == [
        "compression:full-model:production"
    ]

    all_lanes = module.resolve_map_mirheo_sanity_selections([], all_lanes=True)
    assert [module.selection_key(item) for item in all_lanes] == [
        "compression:full-model:production",
        "compression:reduced-model:production",
        "indentation:full-model:production",
        "indentation:reduced-model:production",
    ]

    deduped = module.resolve_map_mirheo_sanity_selections(
        [
            "compression:full-model:production",
            "compression:full-model:production",
        ]
    )
    assert [module.selection_key(item) for item in deduped] == [
        "compression:full-model:production"
    ]

    with pytest.raises(ValueError, match="only supports production"):
        module.resolve_map_mirheo_sanity_selections(
            ["compression:full-model:validation"]
        )


def test_map_mirheo_sanity_selection_run_roots_cover_all_supported_lanes(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_roots_test",
    )

    selections = module.resolve_map_mirheo_sanity_selections([], all_lanes=True)
    roots = {
        module.selection_key(selection): module._selection_run_root(tmp_path, selection)
        for selection in selections
    }

    assert roots["compression:full-model:production"] == (
        tmp_path / "compression_full_retry" / "runs" / "compression" / "full-model" / "production"
    )
    assert roots["compression:reduced-model:production"] == (
        tmp_path / "compression_reduced" / "runs" / "compression" / "reduced-model" / "production"
    )
    assert roots["indentation:full-model:production"] == (
        tmp_path / "indentation_full" / "runs" / "indentation" / "full-model" / "production"
    )
    assert roots["indentation:reduced-model:production"] == (
        tmp_path / "indentation_reduced" / "runs" / "indentation" / "reduced-model" / "production"
    )


def test_map_mirheo_sanity_submit_failure_and_relative_path(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_submit_failure_test",
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    assert module._resolve_path("relative/report") == (tmp_path / "relative" / "report").resolve()

    def fake_run(command, **kwargs):
        assert command[0] == "sbatch"
        return _Result(returncode=2, stdout="bad\n", stderr="worse\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="sbatch failed"):
        module._submit_job(
            selection=module.VegaWorkflowSelection("compression", "full-model", "production"),
            output_dir=tmp_path / "run",
            python_bin="python",
            n_displacements=1,
            numsteps=2,
            numsteps_eq=3,
            time_limit="00:01:00",
        )


def test_map_mirheo_sanity_waits_for_active_jobs_and_skips_blank_sacct(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "platforms" / "vega" / "run_map_mirheo_sanity.py",
        "run_map_mirheo_sanity_wait_test",
    )

    squeue_outputs = iter(["123 RUNNING\n", ""])
    slept: list[int] = []

    def fake_run(command, **kwargs):
        if command[0] == "squeue":
            return _Result(stdout=next(squeue_outputs))
        if command[0] == "sacct":
            return _Result(stdout="\n123|COMPLETED|0:0\n123.batch|COMPLETED|0:0\n")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: slept.append(seconds))

    records = module._wait_for_jobs(["123"], timeout_seconds=30, poll_seconds=4)

    assert slept == [4]
    assert records == {"123": {"state": "COMPLETED", "exit_code": "0:0"}}
