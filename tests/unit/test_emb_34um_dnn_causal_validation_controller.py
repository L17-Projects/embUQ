from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_controller_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "run_emb_34um_dnn_causal_validation_controller.py"
    )
    spec = importlib.util.spec_from_file_location("run_emb_34um_dnn_causal_validation_controller", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_design_manifest(campaign_root: Path, module: Any) -> Path:
    entries = [
        {
            "mode": "unseen_test",
            "replica": 0,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "unseen_test" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 ...",
        },
        {
            "mode": "shared_initial",
            "replica": 1,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "shared_initial" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "shared_initial",
            "replica": 2,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-002" / "shared_initial" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "al-step-01",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "al-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "lhs-step-01",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "lhs-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "al-step-01",
            "replica": 2,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-002" / "al-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "lhs-step-01",
            "replica": 2,
            "cycle": 1,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-002" / "lhs-step-01" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "al-step-02",
            "replica": 1,
            "cycle": 2,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "al-step-02" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "lhs-step-02",
            "replica": 1,
            "cycle": 2,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-001" / "lhs-step-02" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "al-step-02",
            "replica": 2,
            "cycle": 2,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-002" / "al-step-02" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
        {
            "mode": "lhs-step-02",
            "replica": 2,
            "cycle": 2,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "replica-002" / "lhs-step-02" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-99%30 --export=EXECUTION_MODE=render-only,...",
        },
    ]
    design_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1",
        "timestamp": "20260521_130000",
        "scratch_root": str(campaign_root.parent),
        "vault_root_timestamp": str(campaign_root.parent / "vault" / "20260521_130000"),
        "run_id_prefix": "emb-34um-dnn-causal-test",
        "force_grid_source": "protocol_default",
        "campaign_root": str(campaign_root),
        "command_inventory": {
            "count": len(entries),
            "walltime": "00:30:00",
            "concurrent_jobs": 30,
            "retry_limit": 3,
            "entries": entries,
        },
    }
    manifest_path = campaign_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(design_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_controller_builds_stage_graph_and_wires_train_render_analyze_commands(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    manifest_path = _write_design_manifest(campaign_root, module)

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
    )

    output_path = Path(result["manifest_path"])
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert manifest["schema_version"] == module.CONTROLLER_SCHEMA_VERSION
    assert manifest["execution_mode"] == "render-only"
    assert manifest["render_only"] is True
    assert manifest["dry_run"] is False
    assert manifest["command_inventory"]["count"] == 11
    assert manifest["replicas"] == [1, 2]
    assert manifest["cycle_count"] == 2
    assert manifest["stage_order"] == [
        "prepare_design",
        "unseen_test_submit",
        "shared_initial_submit",
        "lhs-step-01-submit",
        "al-step-01-train-score-select",
        "al-step-01-render-selected",
        "al-step-01-submit",
        "lhs-step-02-submit",
        "al-step-02-train-score-select",
        "al-step-02-render-selected",
        "al-step-02-submit",
        "final_ingest",
        "final_analyze",
    ]
    assert len(manifest["parallel_arrays"]) == 11
    assert all(entry["execution_mode"] in {"render-only", "execute", "dry-run"} for entry in manifest["parallel_arrays"])
    assert all(entry["array"] == "0-99%30" for entry in manifest["parallel_arrays"])
    stages = {stage["name"]: stage for stage in manifest["stages"]}

    assert stages["al-step-01-train-score-select"]["dependencies"] == ["shared_initial_submit"]
    assert stages["al-step-02-train-score-select"]["dependencies"] == ["al-step-01-submit"]
    assert stages["al-step-02-render-selected"]["dependencies"] == ["al-step-02-train-score-select"]
    assert stages["final_analyze"]["dependencies"] == ["final_ingest"]

    train_commands = [item["command"] for item in stages["al-step-01-train-score-select"]["commands"]]
    assert any("--stage-action al-build-inputs" in command for command in train_commands)
    assert any("run_emb_34um_dnn_train_score_select.py" in command for command in train_commands)
    assert any("--timing-canary" in command for command in train_commands)
    assert any("--dry-run" in command for command in train_commands)

    render_commands = [item["command"] for item in stages["al-step-01-render-selected"]["commands"]]
    assert all("--stage-action al-render-selected" in command for command in render_commands)

    analyze_command = stages["final_analyze"]["commands"][0]["command"]
    assert "analyze_emb_34um_dnn_causal_validation.py" in analyze_command
    assert "--rows" in analyze_command
    assert module.INGEST_ROWS_FILENAME in analyze_command
    assert "--allow-blocked" in analyze_command

    assert manifest["submission"]["submitted"] is False
    assert manifest["submission"]["submission_commands"] == []


def test_controller_execute_mode_rewrites_submission_commands_and_dispatch_submit_guard(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    manifest_path = _write_design_manifest(campaign_root, module)

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
        execution_mode="execute",
    )
    manifest = result["manifest"]
    assert manifest["execution_mode"] == "execute"
    assert manifest["render_only"] is False
    assert manifest["dry_run"] is False

    stages = {stage["name"]: stage for stage in manifest["stages"]}
    submit_stage = stages["shared_initial_submit"]["commands"]
    assert submit_stage
    assert all("EXECUTION_MODE=execute" in item["command"] for item in submit_stage)
    assert not any("EXECUTION_MODE=render-only" in item["command"] for item in submit_stage)

    train_commands = [item["command"] for item in stages["al-step-01-train-score-select"]["commands"]]
    assert any("emb_34um_dnn_train_score_select.sbatch" in command for command in train_commands)
    assert any("sbatch --parsable" in command for command in train_commands)
    assert not any("--dry-run" in command for command in train_commands)

    final_ingest_command = stages["final_ingest"]["commands"][0]["command"]
    assert "emb_34um_dnn_causal_validation_metrics.sbatch" in final_ingest_command
    assert "RUN_ANALYZE=1" in final_ingest_command

    with pytest.raises(ValueError, match="--submit requires --execution-mode execute"):
        module.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--stage-action",
                "shared_initial_submit",
                "--execution-mode",
                "render-only",
                "--submit",
            ]
        )


def test_controller_dispatch_runs_stage_commands_only_in_execute_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    manifest_path = _write_design_manifest(campaign_root, module)

    module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
        execution_mode="execute",
    )

    runs: list[tuple[str, bool, bool]] = []

    def fake_run(command: str, *, shell: bool, check: bool) -> subprocess.CompletedProcess[str]:
        runs.append((command, shell, check))
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--stage-action",
            "shared_initial_submit",
            "--execution-mode",
            "execute",
            "--submit",
        ]
    )

    assert exit_code == 0
    assert runs
    assert all(shell is True and check is True for _, shell, check in runs)
    assert all("EXECUTION_MODE=execute" in command for command, _, _ in runs)
