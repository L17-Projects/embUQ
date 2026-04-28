import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_retraining_canary_runner_writes_report_and_checks_outputs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "run_retraining_canary.py",
        "run_retraining_canary_test",
    )
    captured = {}

    def fake_run(command, cwd=None, capture_output=False, text=False):
        captured["command"] = command
        model_path = Path(command[command.index("--out") + 1])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("model", encoding="utf-8")
        model_path.with_name(f"{model_path.stem}_loss_hist.pkl").write_text("loss", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="retraining ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(["--output-root", str(tmp_path)])

    assert rc == 0
    assert str(repo_root / "compression" / "surrogate" / "scripts" / "emb_train.py") in captured["command"]
    report = json.loads((tmp_path / "retraining_canary_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert Path(report["artifacts"]["model"]).exists()
    assert Path(report["artifacts"]["loss_history"]).exists()
    assert Path(report["artifacts"]["stdout_log"]).read_text(encoding="utf-8") == "retraining ok\n"
    assert Path(report["artifacts"]["stderr_log"]).read_text(encoding="utf-8") == ""
