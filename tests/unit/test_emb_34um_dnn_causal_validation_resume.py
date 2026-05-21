from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESULT_FILENAME = "emb_34um_result.json"
BATCH_SUMMARY_FILENAME = "emb_34um_batch_summary.json"


def _load_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "resume_emb_34um_dnn_causal_validation.py"
    )
    spec = importlib.util.spec_from_file_location("resume_emb_34um_dnn_causal_validation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_campaign(
    campaign_root: Path,
    *,
    module: Any,
    missing_indices: set[int] | None = None,
) -> Path:
    missing_indices = missing_indices or set()
    stage_one_summary = campaign_root / "replica-001" / "shared_initial" / BATCH_SUMMARY_FILENAME
    stage_two_summary = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    stage_one_outputs = []
    for index in range(3):
        root = campaign_root / "replica-001" / "shared_initial" / "emb" / f"shared-{index:03d}"
        stage_one_outputs.append(str(root))
        if index not in missing_indices:
            root.mkdir(parents=True, exist_ok=True)
            (root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": root.name}), encoding="utf-8")
    _write_json(
        stage_one_summary,
        {
            "status": "selection_rendered",
            "candidate_count": 3,
            "expected_output_roots": stage_one_outputs,
        },
    )

    stage_two_outputs = []
    for index in range(4):
        root = campaign_root / "replica-001" / "al-step-01" / "emb" / f"al-{index:03d}"
        stage_two_outputs.append(str(root))
        root.mkdir(parents=True, exist_ok=True)
        (root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": root.name}), encoding="utf-8")
    _write_json(
        stage_two_summary,
        {
            "status": "selection_rendered",
            "candidate_count": 4,
            "expected_output_roots": stage_two_outputs,
        },
    )

    manifest_path = campaign_root / module.CONTROLLER_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "schema_version": module.RESUME_SCHEMA_VERSION,
            "execution_mode": "execute",
            "command_inventory": {"concurrent_jobs": 4},
            "stages": [
                {
                    "name": "shared_initial_submit",
                    "command_type": "submission",
                    "commands": [
                        {"command": "sbatch --parsable shared_initial_001"},
                    ],
                    "expected_output_roots": [str(stage_one_summary)],
                },
                {
                    "name": "al-step-01-submit",
                    "command_type": "submission",
                    "commands": [
                        {"command": "sbatch --parsable al_step_01_001"},
                    ],
                    "expected_output_roots": [str(stage_two_summary)],
                },
                {
                    "name": "final_ingest",
                    "command_type": "ingest",
                    "commands": [
                        {"command": "python final_ingest.py"},
                    ],
                    "expected_output_roots": [str(campaign_root / "ingest" / "rows.json")],
                },
            ],
        },
    )
    return manifest_path


def test_resume_helper_reports_missing_indices_and_plans_only_incomplete_batches(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["submission_stage_count"] == 2
    assert plan["complete_batch_count"] == 1
    assert plan["missing_batch_count"] == 1
    assert plan["completed_candidate_count"] == 6
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["stage_name"] == "shared_initial_submit"
    assert plan["stages"][0]["commands"][0]["batch_report"]["missing_candidate_indices"] == [1]
    assert plan["stages"][1]["commands"][0]["batch_report"]["missing_candidate_indices"] == []


def test_resume_helper_submits_only_until_max_active_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={0, 2})
    (campaign_root / "replica-001" / "al-step-01" / "emb" / "al-002" / RESULT_FILENAME).unlink()

    calls: list[list[str] | str] = []
    squeue_outputs = iter(["\n".join(["1001", "1002"]) + "\n", "\n".join(["1001", "1002"]) + "\n"])

    def fake_run(command, **kwargs):
        if isinstance(command, list) and command[0] == "squeue":
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": next(squeue_outputs), "stderr": ""})()
        if isinstance(command, str) and command.startswith("sbatch"):
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "12345\n", "stderr": ""})()
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    plan = module.build_resume_plan(campaign_root=campaign_root)
    assert len(plan["commands_to_submit"]) == 2
    submission = module._submit_missing_commands(
        plan["commands_to_submit"],
        user="test-user",
        max_active_jobs=3,
    )

    assert submission["submitted_count"] == 1
    assert submission["deferred_count"] == 1
    assert len([call for call in calls if isinstance(call, str) and call.startswith("sbatch")]) == 1
    assert submission["submitted_commands"][0]["stage_name"] == "shared_initial_submit"
    assert submission["deferred_commands"][0]["deferred_reason"] == "active_job_limit_reached"


def test_resume_helper_blocks_unrendered_adaptive_placeholders(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    placeholder_summary = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    _write_json(
        placeholder_summary,
        {
            "status": "adaptive_placeholder_not_rendered",
            "candidate_count": 100,
            "expected_output_roots": [str(placeholder_summary.parent)],
            "rendered_candidate_manifests": [],
        },
    )
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "execution_mode": "execute",
            "command_inventory": {"concurrent_jobs": 4},
            "stages": [
                {
                    "name": "al-step-01-submit",
                    "command_type": "submission",
                    "commands": [{"command": "sbatch --parsable al_step_01_001"}],
                    "expected_output_roots": [str(placeholder_summary)],
                },
            ],
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["blocked_batch_count"] == 1
    assert plan["missing_batch_count"] == 0
    assert plan["commands_to_submit"] == []
    command = plan["stages"][0]["commands"][0]
    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["blocked_reason"] == "batch_not_fully_rendered"
    assert command["batch_report"]["completed_candidate_count"] == 0
