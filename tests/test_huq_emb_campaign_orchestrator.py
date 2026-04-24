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


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_huq_emb_orchestrator_runs_workflow_and_manifest_postprocess(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py",
        "huq_emb_campaign_orchestrator_test",
    )
    captured: list[list[str]] = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del cwd, text, capture_output, check
        captured.append(command)
        script_name = Path(command[1]).name
        if script_name == "run_workflow_matrix.py":
            workflow_root = Path(_arg_value(command, "--output-root"))
            lane_root = workflow_root / "manifests" / "lanes"
            lane_root.mkdir(parents=True, exist_ok=True)
            lane_payload = {
                "lane": "compression:full-model:production",
                "selection": {
                    "experiment": "compression",
                    "model_family": "full-model",
                    "profile": "production",
                },
                "stage_statuses": [
                    {"stage": "phase1", "status": "passed"},
                    {"stage": "phase2", "status": "passed"},
                    {"stage": "phase3b", "status": "passed"},
                    {"stage": "propagation_phase3b", "status": "passed"},
                    {"stage": "map_phase3b", "status": "passed"},
                    {"stage": "map_mirheo", "status": "passed"},
                ],
                "artifacts": {},
                "job_manifests": [],
                "policy": {"status": "pass", "violations": []},
            }
            (lane_root / "compression.json").write_text(
                json.dumps(lane_payload), encoding="utf-8"
            )
        elif script_name == "run_postprocess_figures.py":
            manifest_output = Path(_arg_value(command, "--manifest-output"))
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(
                json.dumps({"release_status": "PASS", "hard_failures": []}),
                encoding="utf-8",
            )
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    internal_steps: list[str] = []

    def fake_internal_step(*, name, logs_root, fn):
        del logs_root, fn
        internal_steps.append(name)
        return {
            "name": name,
            "command": ["python:<internal>", name],
            "returncode": 0,
            "start_utc": "2026-01-01T00:00:00+00:00",
            "end_utc": "2026-01-01T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "stdout_log": "internal.stdout.log",
            "stderr_log": "internal.stderr.log",
        }

    monkeypatch.setattr(module, "_record_python_step", fake_internal_step)

    paper_root = tmp_path / "paper_data"
    rc = module.main(["--paper-data-root", str(paper_root), "--campaign-id", "camp1", "--site", "karolina"])
    assert rc == 0

    assert any(Path(cmd[1]).name == "run_workflow_matrix.py" for cmd in captured)
    assert any(Path(cmd[1]).name == "run_postprocess_figures.py" for cmd in captured)

    workflow_cmd = next(cmd for cmd in captured if Path(cmd[1]).name == "run_workflow_matrix.py")
    assert "--allow-release-fail" in workflow_cmd
    assert "--selection" in workflow_cmd
    assert workflow_cmd.count("--selection") == 4

    post_cmd = next(cmd for cmd in captured if Path(cmd[1]).name == "run_postprocess_figures.py")
    assert "--emit-release-manifest" in post_cmd
    assert "--manifest-only" in post_cmd
    assert internal_steps == ["mandatory_asset_graph"]

    report_path = paper_root / "logs" / "camp1" / "huq_emb_campaign_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["release_status"] == "PASS"


def test_huq_emb_orchestrator_fails_when_release_manifest_reports_fail(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py",
        "huq_emb_campaign_orchestrator_fail_test",
    )

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False):
        del cwd, text, capture_output, check
        script_name = Path(command[1]).name
        if script_name == "run_postprocess_figures.py":
            manifest_output = Path(_arg_value(command, "--manifest-output"))
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(
                json.dumps({"release_status": "FAIL", "hard_failures": ["missing asset"]}),
                encoding="utf-8",
            )
        elif script_name == "run_workflow_matrix.py":
            workflow_root = Path(_arg_value(command, "--output-root"))
            lane_root = workflow_root / "manifests" / "lanes"
            lane_root.mkdir(parents=True, exist_ok=True)
            (lane_root / "lane.json").write_text(
                json.dumps(
                    {
                        "lane": "compression:full-model:production",
                        "selection": {
                            "experiment": "compression",
                            "model_family": "full-model",
                            "profile": "production",
                        },
                        "stage_statuses": [
                            {"stage": "phase1", "status": "passed"},
                            {"stage": "phase2", "status": "passed"},
                            {"stage": "phase3b", "status": "passed"},
                            {"stage": "propagation_phase3b", "status": "passed"},
                            {"stage": "map_phase3b", "status": "passed"},
                            {"stage": "map_mirheo", "status": "passed"},
                        ],
                        "artifacts": {},
                        "job_manifests": [],
                        "policy": {"status": "pass", "violations": []},
                    }
                ),
                encoding="utf-8",
            )
        return _Result(returncode=0, stdout="ok\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "_record_python_step",
        lambda **_: {
            "name": "mandatory_asset_graph",
            "command": ["python:<internal>", "mandatory_asset_graph"],
            "returncode": 0,
            "start_utc": "2026-01-01T00:00:00+00:00",
            "end_utc": "2026-01-01T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "stdout_log": "internal.stdout.log",
            "stderr_log": "internal.stderr.log",
        },
    )

    rc = module.main(["--paper-data-root", str(tmp_path / "paper_data"), "--campaign-id", "camp2", "--site", "karolina"])
    assert rc == 1


def test_huq_emb_orchestrator_rejects_vega_full_rebuild_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py",
        "huq_emb_campaign_orchestrator_guard_test",
    )

    try:
        module.main(["--paper-data-root", str(tmp_path / "paper_data"), "--campaign-id", "camp3"])
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected Vega full rebuild guard to abort the legacy runner.")

    assert "run_vega_50k_campaign.py" in message
    assert "--skip-workflow" in message
