import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vega_acceptance_wrapper_writes_machine_readable_report(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "run_vega_acceptance.py", "run_vega_acceptance_test")

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=False):
        if any("run_gpu_validation_suite.py" in str(part) for part in command):
            out_idx = command.index("--output-root") + 1
            runner_output = Path(command[out_idx])
            runner_output.mkdir(parents=True, exist_ok=True)
            summary_path = runner_output / "workflow_suite_summary.json"
            summary_path.write_text(json.dumps([{"workflow": "compression_reduced", "elapsed_seconds": 1.23}]), encoding="utf-8")
            return Result(returncode=0, stdout="validation ok", stderr="")
        return Result(returncode=0, stdout="stub", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.platform, "node", lambda: "vega-node")

    output_root = tmp_path / "acceptance"
    monkeypatch.setattr(sys, "argv", [
        "run_vega_acceptance.py",
        "--output-root", str(output_root),
        "--workflows", "compression_reduced",
    ])

    rc = module.main()
    assert rc == 0

    report_path = output_root / "vega_acceptance_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["target"] == "vega"
    assert report["steps"][0]["returncode"] == 0
    assert Path(report["artifacts"]["workflow_suite_summary"]).exists()
