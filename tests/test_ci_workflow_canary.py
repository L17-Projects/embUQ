import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
    module = _load_module(repo_root / "scripts" / "ci" / "run_workflow_canary.py", "run_workflow_canary_test")
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
