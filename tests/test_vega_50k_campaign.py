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
    def __init__(self, returncode: int = 0, stdout: str = "12345\n", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_vega_50k_campaign_uses_sbatch_and_postprocess(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "workflows" / "emb" / "huq_emb" / "run_vega_50k_campaign.py",
        "run_vega_50k_campaign_test",
    )
    calls = []

    def fake_run(command, cwd=None, text=False, capture_output=False, check=False, env=None):
        del cwd, text, capture_output, check
        calls.append((command, env))
        if len(command) > 1 and Path(command[1]).name == "run_paper_data_campaign.py":
            report_path = tmp_path / "paper_data" / "logs" / "camp1" / "huq_emb_campaign_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"status": "passed", "release_status": "PASS"}), encoding="utf-8")
        return _Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "_snapshot_inputs",
        lambda *, snapshot_root, selections: {
            "manifest": str(snapshot_root / "inputs_snapshot_manifest.json"),
            "files": [module.selection_key(selection) for selection in selections],
        },
    )
    monkeypatch.setattr(module, "_batch_wait_and_validate", lambda submissions, report, logs_root: True)
    rc = module.main(["--paper-data-root", str(tmp_path / "paper_data"), "--campaign-id", "camp1", "--selection", "compression:full-model:production", "--selection", "indentation:reduced-model:production"])
    assert rc == 0

    sbatch_calls = [call for call in calls if call[0][0] == "sbatch"]
    postprocess_call = next(call for call in calls if len(call[0]) > 1 and Path(call[0][1]).name == "run_paper_data_campaign.py")
    assert "--skip-workflow" in postprocess_call[0]
    assert any(Path(call[0][-1]).name == "train_dnn_surrogates.sbatch" for call in sbatch_calls)
    assert any(Path(call[0][-1]).name == "complete_inference_compression.sbatch" for call in sbatch_calls)
    assert any(Path(call[0][-1]).name == "complete_reduced_indentation.sbatch" for call in sbatch_calls)
    assert any(Path(call[0][-1]).name == "workflow_map.sbatch" for call in sbatch_calls)
    assert any(Path(call[0][-1]).name == "workflow_map_mirheo.sbatch" for call in sbatch_calls)
    complete_env = next(env for cmd, env in sbatch_calls if Path(cmd[-1]).name == "complete_inference_compression.sbatch")
    mirheo_env = next(env for cmd, env in sbatch_calls if Path(cmd[-1]).name == "workflow_map_mirheo.sbatch")
    assert complete_env["PHASE2_BACKEND"] == "native-cuda"
    assert complete_env["LOGS_DIR"].endswith("compression__full-model__production")
    assert mirheo_env["GPU_TIME_LIMIT"] == "02:00:00"
