import importlib.util
import json
from pathlib import Path


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
    assert report["lanes"][0]["selection"] == "compression:full-model:production"
    assert report["lanes"][0]["job_id"] == "12345"
    sbatch_command = captured["sbatch_command"]
    assert isinstance(sbatch_command, list)
    assert "--export" in sbatch_command
    export_arg = sbatch_command[sbatch_command.index("--export") + 1]
    assert "OUTPUT_DIR=" in export_arg
    assert "GPU_TIME_LIMIT=" in export_arg
    assert "MAP_MIRHEO_NUMSTEPS=" in export_arg
