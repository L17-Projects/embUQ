import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_canary_runner_writes_report_and_checks_artifacts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "run_workflow_canary.py",
        "run_workflow_canary_test",
    )
    captured = {}

    def fake_run(command, cwd=None, capture_output=False, text=False):
        captured["command"] = command
        output_root = Path(command[command.index("--output-root") + 1])
        workflow_dir = output_root / "compression__reduced-model__validation"
        dataset = "compression_2.1um"

        (output_root / "workflow_suite_summary.json").write_text("[]", encoding="utf-8")
        (workflow_dir / "results" / "results_phase_1" / dataset).mkdir(parents=True, exist_ok=True)
        (workflow_dir / "results" / "results_phase_2").mkdir(parents=True, exist_ok=True)
        (workflow_dir / "results" / "results_phase_3b" / dataset).mkdir(parents=True, exist_ok=True)
        (workflow_dir / "results" / "propagation_phase3b" / dataset).mkdir(parents=True, exist_ok=True)
        (workflow_dir / "map_phase3b").mkdir(parents=True, exist_ok=True)
        (workflow_dir / "overlay_uq_ref").mkdir(parents=True, exist_ok=True)
        (workflow_dir / "posteriors_phase3b").mkdir(parents=True, exist_ok=True)

        (workflow_dir / "results" / "results_phase_1" / dataset / "latest").write_text("{}", encoding="utf-8")
        (workflow_dir / "results" / "results_phase_2" / "latest").write_text("{}", encoding="utf-8")
        (workflow_dir / "results" / "results_phase_3b" / dataset / "latest").write_text("{}", encoding="utf-8")
        (workflow_dir / "results" / "propagation_phase3b" / dataset / "summary.csv").write_text(
            "x,mean\n0.0,0.0\n",
            encoding="utf-8",
        )
        (workflow_dir / "map_phase3b" / "all_diameters_map.json").write_text("{}", encoding="utf-8")
        (workflow_dir / "overlay_uq_ref" / "uq_overlay_2.1um.png").write_text("png", encoding="utf-8")
        (workflow_dir / "posteriors_phase3b" / "posterior_marginals_2.1um.png").write_text("png", encoding="utf-8")
        (workflow_dir / "summary.json").write_text(
            json.dumps({"selection": "compression:reduced-model:validation"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="workflow ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--output-root", str(tmp_path)])

    assert rc == 0
    assert "compression:reduced-model:validation" in captured["command"]
    report = json.loads((tmp_path / "workflow_canary_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["selection"] == "compression:reduced-model:validation"
    assert Path(report["artifacts"]["posterior_plot"]).exists()
    assert Path(report["artifacts"]["stdout_log"]).read_text(encoding="utf-8") == "workflow ok\n"
    assert Path(report["artifacts"]["stderr_log"]).read_text(encoding="utf-8") == ""


def test_ci_workflow_canary_config_uses_multiple_public_diameters():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "reduced" / "configs" / "ci" / "ci_canary_config_compression.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    diameters = config["emb_diameters"]
    assert len(diameters) >= 2

    for diameter in diameters:
        data_file = repo_root / "compression" / "evalkit" / "data" / f"compression_data_{diameter}um.dat"
        surrogate_dir = repo_root / "compression" / "surrogate" / "diameters" / f"{diameter}um"
        assert data_file.exists()
        assert surrogate_dir.is_dir()


def test_workflow_canary_reports_validation_failures(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "run_workflow_canary.py",
        "run_workflow_canary_validation_errors_test",
    )

    assert module._resolve_repo_path(tmp_path.name).is_absolute()

    config_path = tmp_path / "empty.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "load_experiments", lambda _config, _root: [])
    with pytest.raises(ValueError, match="No enabled datasets"):
        module._load_datasets(config_path, "compression")

    with pytest.raises(FileNotFoundError, match="required artifacts are missing"):
        module._assert_artifacts_exist({"missing": str(tmp_path / "missing.txt")})

    with pytest.raises(ValueError, match="Expected selection"):
        module.main(["--selection", "bad", "--output-root", str(tmp_path / "out")])


def test_workflow_canary_command_and_summary_errors(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "run_workflow_canary.py",
        "run_workflow_canary_command_errors_test",
    )

    def failing_run(command, cwd=None, capture_output=False, text=False):
        return subprocess.CompletedProcess(command, 2, stdout="stdout text\n", stderr="stderr text\n")

    monkeypatch.setattr(module.subprocess, "run", failing_run)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        module._run_logged_command(["python", "-c", "fail"], repo_root, tmp_path)
    assert excinfo.value.returncode == 2
    assert (tmp_path / "workflow_canary.stdout.log").read_text(encoding="utf-8") == "stdout text\n"
    assert (tmp_path / "workflow_canary.stderr.log").read_text(encoding="utf-8") == "stderr text\n"

    def mismatched_summary_run(command, cwd=None, capture_output=False, text=False):
        output_root = Path(command[command.index("--output-root") + 1])
        workflow_dir = output_root / "compression__reduced-model__validation"
        dataset = "compression_2.1um"
        (output_root / "workflow_suite_summary.json").write_text("[]", encoding="utf-8")
        for path in [
            workflow_dir / "results" / "results_phase_1" / dataset / "latest",
            workflow_dir / "results" / "results_phase_2" / "latest",
            workflow_dir / "results" / "results_phase_3b" / dataset / "latest",
            workflow_dir / "results" / "propagation_phase3b" / dataset / "summary.csv",
            workflow_dir / "map_phase3b" / "all_diameters_map.json",
            workflow_dir / "overlay_uq_ref" / "uq_overlay_2.1um.png",
            workflow_dir / "posteriors_phase3b" / "posterior_marginals_2.1um.png",
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact", encoding="utf-8")
        (workflow_dir / "summary.json").write_text(json.dumps({"selection": "wrong"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", mismatched_summary_run)
    with pytest.raises(ValueError, match="selection mismatch"):
        module.main(["--output-root", str(tmp_path / "summary-mismatch")])
