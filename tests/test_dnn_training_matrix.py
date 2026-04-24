from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_dnn_training_matrix_runner_writes_report_and_invokes_multi_arch(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_dnn_surrogate_training.py",
        "run_dnn_surrogate_training_test",
    )
    spec = dict(next(item for item in module.SPECS if item["name"] == "compression_2.1um"))
    spec["best"] = tmp_path / "trained" / "microbubble_force_BEST.pkl"
    monkeypatch.setattr(module, "SPECS", [spec])
    captured: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        captured.append(command)
        if command[0] == "sbatch":
            return _Result(returncode=0, stdout="12345\n")
        if command[0] == "squeue":
            return _Result(returncode=0, stdout="")
        if command[0] == "sacct":
            rows = [f"12345_{idx}|COMPLETED|0:0" for idx in range(module.ARRAY_SIZE)]
            return _Result(returncode=0, stdout="\n".join(rows) + "\n")
        report_path = Path(command[command.index("--report-json") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dest = Path(spec["best"])
        best_dest.parent.mkdir(parents=True, exist_ok=True)
        best_dest.write_text("model", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "best_dest": str(best_dest),
                    "best": {"name": "w32_d2"},
                }
            ),
            encoding="utf-8",
        )
        assert env is None or env["EXPERIMENT"] == "compression"
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rc = module.main(["--output-root", str(tmp_path), "--only", "compression_2.1um"])
    assert rc == 0
    assert captured[0][0] == "sbatch"
    assert Path(captured[0][-1]).name == "train_dnn_arch_array.sbatch"
    assert Path(captured[-1][1]).name == "train_multi_arch.py"
    report = json.loads((tmp_path / "dnn_training_matrix_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["runs"][0]["status"] == "passed"
    assert report["runs"][0]["array_job_id"] == "12345"
    manifest_path = Path(report["runs"][0]["manifest"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["array_size"] == module.ARRAY_SIZE
    assert manifest["slurm"]["array_job_id"] == "12345"
    assert manifest["logs"]["array"]["stdout_pattern"].endswith("array_%A_%a.out")
    assert manifest["artifacts"]["best_artifact"]["sha256"]
    assert manifest["artifacts"]["report"]["sha256"]


def test_dnn_training_matrix_runner_resumes_from_existing_array_results(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "vega" / "run_dnn_surrogate_training.py",
        "run_dnn_surrogate_training_resume_test",
    )

    spec = dict(next(item for item in module.SPECS if item["name"] == "compression_2.1um"))
    spec["best"] = tmp_path / "trained" / "microbubble_force_BEST.pkl"
    monkeypatch.setattr(module, "SPECS", [spec])
    models_dir = tmp_path / "compression_2.1um" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(module.ARRAY_SIZE):
        (models_dir / f"result_{idx}.json").write_text(
            json.dumps({"name": f"arch_{idx}", "model_path": str(models_dir / f"arch_{idx}.pkl")}),
            encoding="utf-8",
        )

    best_dest = Path(spec["best"])
    best_dest.parent.mkdir(parents=True, exist_ok=True)

    captured: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check, env
        captured.append(command)
        if command[0] == "sbatch":
            raise AssertionError("resume path should not resubmit arrays when complete results already exist")
        report_path = Path(command[command.index("--report-json") + 1])
        report_path.write_text(
            json.dumps(
                {
                    "best_dest": str(best_dest),
                    "best": {"name": "arch_0"},
                }
            ),
            encoding="utf-8",
        )
        best_dest.write_text("model", encoding="utf-8")
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rc = module.main(["--output-root", str(tmp_path), "--only", "compression_2.1um"])
    assert rc == 0
    assert len(captured) == 1
    assert Path(captured[0][1]).name == "train_multi_arch.py"
    report = json.loads((tmp_path / "dnn_training_matrix_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["runs"][0]["status"] == "passed"
    assert report["runs"][0]["resumed_from_existing_results"] is True
