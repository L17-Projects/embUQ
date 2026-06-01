from __future__ import annotations

import importlib.util
import json
import math
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
SELECTION_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_selection_manifest.json"
SELECTION_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_selection_batch_summary.json"


def _load_repair_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "repair_emb_34um_dnn_causal_validation_resume.py"
    )
    spec = importlib.util.spec_from_file_location("repair_emb_34um_dnn_causal_validation_resume", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _write_pilot_summary(campaign_root: Path, *, passed: bool) -> Path:
    summary_path = campaign_root / "pilot" / "validation" / "emb_34um_dnn_causal_validation_pilot_summary.json"
    _write_json(
        summary_path,
        {
            "decision": {"passed": passed},
            "promotion_readiness": {"passed": passed},
        },
    )
    return summary_path


def _runtime_fingerprint(
    *,
    radp: float,
    shell_th: float,
    bpress: float = -91.0,
) -> dict[str, object]:
    lx = float(math.ceil(2.0 * radp + 6.0))
    lz = float(math.ceil(2.0 * radp + 10.0))
    return {
        "radp": radp,
        "shell_th": shell_th,
        "fscale": 0.0074,
        "numsteps": 5000,
        "numsteps_eq": 10000,
        "Lx": lx,
        "Ly": lx,
        "Lz": lz,
        "L": lz,
        "direct_stiffness_override": True,
        "bpress": bpress,
    }


def _rendered_candidate_manifest_payload(candidate: Any, *, output_root: Path) -> dict[str, Any]:
    runtime_fingerprint = dict(candidate.parameters["runtime_fingerprint"])
    parameters = {
        "ka": candidate.parameters["ka"],
        "kb": candidate.parameters["kb"],
        "radp": candidate.parameters["radp"],
        "shell_th": candidate.parameters["shell_th"],
        "bpress": candidate.parameters["bpress"],
    }
    return {
        "candidate_id": candidate.candidate_id,
        "output_root": str(output_root),
        "normalized_payload": {
            "candidate_id": candidate.candidate_id,
            "output_root": str(output_root),
            "parameters": parameters,
            "force_grid": candidate.parameters["force_grid"],
            "fingerprint": runtime_fingerprint,
        },
        "rendered_payload": {
            "request_payload": {
                "candidate_id": candidate.candidate_id,
                "output_root": str(output_root),
                "parameters": parameters,
                "force_grid": candidate.parameters["force_grid"],
                "fingerprint": runtime_fingerprint,
            }
        },
        "active_learning_metadata": dict(candidate.metadata),
    }


def _seed_campaign(
    campaign_root: Path,
    *,
    module: Any,
    missing_indices: set[int] | None = None,
) -> Path:
    missing_indices = missing_indices or set()
    _write_pilot_summary(campaign_root, passed=True)
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


def _seed_al_replacement_stage(campaign_root: Path, *, module: Any) -> None:
    _write_pilot_summary(campaign_root, passed=True)
    stage_root = campaign_root / "replica-001" / "al-step-01"
    runtime_fingerprint = _runtime_fingerprint(radp=6.62, shell_th=3.75e-9)
    output_roots: list[str] = []
    candidate_records: list[dict[str, object]] = []
    for index in range(3):
        candidate_id = f"al-{index:03d}"
        output_root = stage_root / "emb" / candidate_id
        output_roots.append(str(output_root))
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "family": "emb",
                "experiment": "indentation",
                "ka": 1.0e3 + index,
                "kb": 2.0e3 + index,
                "radp": runtime_fingerprint["radp"],
                "shell_th": runtime_fingerprint["shell_th"],
                "bpress": runtime_fingerprint["bpress"],
                "selection_seed": 10_001,
                "selection_mode": "dnn_causal_fresh_only",
                "selection_source": "dnn_ensemble_disagreement_diversity",
                "selection_status": "rendered",
                "selection_payload": {"candidate_pool_id": f"pool-{index:03d}"},
                "runtime_fingerprint": dict(runtime_fingerprint),
            }
        )
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(
            manifest_path,
            {
                "candidate_id": candidate_id,
                "output_root": str(output_root),
                "normalized_payload": {
                    "candidate_id": candidate_id,
                    "output_root": str(output_root),
                    "parameters": {
                        "ka": 1.0e3 + index,
                        "kb": 2.0e3 + index,
                        "radp": runtime_fingerprint["radp"],
                        "shell_th": runtime_fingerprint["shell_th"],
                        "bpress": runtime_fingerprint["bpress"],
                    },
                    "force_grid": [5000.0 * step / 7.0 for step in range(8)],
                },
                "rendered_payload": {
                    "request_payload": {
                        "candidate_id": candidate_id,
                        "output_root": str(output_root),
                        "parameters": {
                            "ka": 1.0e3 + index,
                            "kb": 2.0e3 + index,
                            "radp": runtime_fingerprint["radp"],
                            "shell_th": runtime_fingerprint["shell_th"],
                            "bpress": runtime_fingerprint["bpress"],
                        },
                        "force_grid": [5000.0 * step / 7.0 for step in range(8)],
                    }
                },
                "active_learning_metadata": {},
            },
        )
        if index != 0:
            (output_root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": candidate_id}), encoding="utf-8")

    _write_json(
        stage_root / BATCH_SUMMARY_FILENAME,
        {
            "status": "selection_rendered",
            "candidate_count": 3,
            "rendered_candidate_manifests": [
                str(stage_root / "emb" / f"al-{index:03d}" / "dpd_sampling_candidate_manifest.json")
                for index in range(3)
            ],
            "expected_output_roots": output_roots,
        },
    )
    _write_json(
        stage_root / SELECTION_MANIFEST_FILENAME,
        {
            "status": "selection_rendered",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 3,
            "candidate_records": candidate_records,
            "candidate_reserve": [
                {
                    "candidate_pool_id": "reserve-001",
                    "ka": 9.1e3,
                    "kb": 8.2e3,
                    "radp": runtime_fingerprint["radp"],
                    "shell_th": runtime_fingerprint["shell_th"],
                    "bpress": runtime_fingerprint["bpress"],
                    "acquisition_score": 0.91,
                    "ensemble_disagreement": 0.33,
                    "diversity_term": 0.22,
                    "runtime_fingerprint": dict(runtime_fingerprint),
                }
            ],
            "candidate_reserve_count": 1,
            "force_grid": [5000.0 * step / 7.0 for step in range(8)],
        },
    )
    _write_json(
        stage_root / SELECTION_SUMMARY_FILENAME,
        {
            "status": "selection_rendered",
            "replica": 1,
            "cycle": 1,
            "candidate_count": 3,
            "candidate_reserve_count": 1,
        },
    )
    _write_json(
        campaign_root / "emb_34um_dnn_causal_validation_manifest.json",
        {
            "run_id_prefix": "emb-34um-dnn-causal-validation",
            "command_inventory": {"walltime": "00:12:00", "concurrent_jobs": 30, "retry_limit": 0},
        },
    )
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "schema_version": module.RESUME_SCHEMA_VERSION,
            "execution_mode": "execute",
            "command_inventory": {"concurrent_jobs": 4},
            "stages": [
                {
                    "name": "al-step-01-submit",
                    "command_type": "submission",
                    "commands": [
                        {"command": "sbatch --parsable original_al_step_01"},
                    ],
                    "expected_output_roots": [str(stage_root / BATCH_SUMMARY_FILENAME)],
                },
            ],
        },
    )


def _seed_non_al_repair_stage(
    campaign_root: Path,
    *,
    module: Any,
    stage_name: str,
    shared_size: int = 3,
    lhs_size: int = 4,
    unseen_size: int = 2,
) -> tuple[Path, int]:
    _write_pilot_summary(campaign_root, passed=True)
    force_grid = [5000.0 * step / 7.0 for step in range(8)]
    runtime_fingerprint = _runtime_fingerprint(radp=6.52, shell_th=3.75e-9)

    def _candidate_record(*, candidate_id: str, ka: float, kb: float, selection_source: str, stage_label: str) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "family": "emb",
            "experiment": "indentation",
            "ka": ka,
            "kb": kb,
            "radp": runtime_fingerprint["radp"],
            "shell_th": runtime_fingerprint["shell_th"],
            "bpress": runtime_fingerprint["bpress"],
            "selection_seed": 42,
            "selection_mode": "dnn_causal_fresh_only",
            "selection_source": selection_source,
            "selection_status": "rendered",
            "selection_payload": {"candidate_pool_id": f"pool-{candidate_id}"},
            "stage": stage_label,
            "runtime_fingerprint": dict(runtime_fingerprint),
        }

    shared_records = [
            _candidate_record(
                candidate_id=f"shared-{index:03d}",
                ka=1.0e3 + index,
                kb=2.0e3 + index,
                selection_source="fresh_dpd",
                stage_label="shared_initial",
            )
        for index in range(shared_size)
    ]
    lhs_records = [
            _candidate_record(
                candidate_id=f"lhs-{index:03d}",
                ka=2.0e3 + index,
                kb=3.0e3 + index,
                selection_source="fresh_dpd",
                stage_label=stage_name,
            )
        for index in range(lhs_size)
    ]
    unseen_records = [
            _candidate_record(
                candidate_id=f"unseen-{index:03d}",
                ka=3.0e3 + index,
                kb=4.0e3 + index,
                selection_source="fresh_dpd_shared_unseen",
            stage_label="unseen_test",
        )
        for index in range(unseen_size)
    ]

    design_manifest_path = campaign_root / "emb_34um_dnn_causal_validation_manifest.json"
    _write_json(
        design_manifest_path,
        {
            "run_id_prefix": "emb-34um-dnn-causal-validation",
            "command_inventory": {"walltime": "00:12:00", "concurrent_jobs": 30, "retry_limit": 0},
            "policy": {"force_grid": force_grid},
            "unseen_test": {"candidate_records": unseen_records},
            "seeds": [
                {
                    "replica": 1,
                    "shared_initial": {"candidate_records": shared_records},
                    "lhs_steps": [{"candidate_records": lhs_records}],
                }
            ],
        },
    )

    if stage_name == "shared_initial":
        stage_root = campaign_root / "replica-001" / "shared_initial"
        candidate_records = shared_records
    elif stage_name == "unseen_test":
        stage_root = campaign_root / "unseen_test"
        candidate_records = unseen_records
    elif stage_name == "lhs-step-01":
        stage_root = campaign_root / "replica-001" / "lhs-step-01"
        candidate_records = lhs_records
    else:
        raise AssertionError(stage_name)

    output_roots: list[str] = []
    for index, record in enumerate(candidate_records):
        output_root = stage_root / "emb" / str(record["candidate_id"])
        output_roots.append(str(output_root))
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(
            manifest_path,
            {
                "candidate_id": record["candidate_id"],
                "output_root": str(output_root),
                "normalized_payload": {
                    "candidate_id": record["candidate_id"],
                    "output_root": str(output_root),
                    "parameters": {
                        "ka": record["ka"],
                        "kb": record["kb"],
                        "radp": record["radp"],
                        "shell_th": record["shell_th"],
                        "bpress": record["bpress"],
                    },
                    "force_grid": force_grid,
                },
                "rendered_payload": {
                    "request_payload": {
                        "candidate_id": record["candidate_id"],
                        "output_root": str(output_root),
                        "parameters": {
                            "ka": record["ka"],
                            "kb": record["kb"],
                            "radp": record["radp"],
                            "shell_th": record["shell_th"],
                            "bpress": record["bpress"],
                        },
                        "force_grid": force_grid,
                    }
                },
                "active_learning_metadata": {},
            },
        )
        if index != 0:
            (output_root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": record["candidate_id"]}), encoding="utf-8")

    _write_json(
        stage_root / BATCH_SUMMARY_FILENAME,
        {
            "status": "selection_rendered",
            "candidate_count": len(candidate_records),
            "rendered_candidate_manifests": [
                str(stage_root / "emb" / str(record["candidate_id"]) / "dpd_sampling_candidate_manifest.json")
                for record in candidate_records
            ],
            "expected_output_roots": output_roots,
        },
    )

    stage_submit = "unseen_test_submit" if stage_name == "unseen_test" else f"{stage_name}-submit"
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "schema_version": module.RESUME_SCHEMA_VERSION,
            "execution_mode": "execute",
            "command_inventory": {"concurrent_jobs": 4},
            "stages": [
                {
                    "name": stage_submit,
                    "command_type": "submission",
                    "commands": [{"command": f"sbatch --parsable {stage_name}_001"}],
                    "expected_output_roots": [str(stage_root / BATCH_SUMMARY_FILENAME)],
                }
            ],
        },
    )

    return stage_root, len(candidate_records)


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


def test_resume_helper_blocks_running_candidates_instead_of_resubmitting(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    running_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        running_root / "emb_34um_runtime_status.json",
        {
            "status": "running",
            "candidate_id": "shared-001",
            "slurm_job_id": "33779001",
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    report = command["batch_report"]
    status = report["candidate_runtime_statuses"][1]

    assert plan["commands_to_submit"] == []
    assert plan["blocked_batch_count"] == 1
    assert plan["missing_batch_count"] == 0
    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert report["blocked_reason"] == "candidate_runtime_in_progress"
    assert report["active_candidate_indices"] == [1]
    assert report["raw_missing_candidate_indices"] == []
    assert status["runtime_status"] == "running"
    assert status["runtime_failure_kind"] == "running"
    assert status["active_runtime"] is True


def test_runtime_failure_guard_blocks_resume_commands_until_investigated() -> None:
    module = _load_module()
    plan = {
        "commands_to_submit": [{"stage_name": "shared_initial_submit", "command": "sbatch --parsable unsafe"}],
        "stages": [
            {
                "stage_name": "shared_initial_submit",
                "commands": [
                    {
                        "command_index": 0,
                        "batch_report": {
                            "batch_summary_path": "/scratch/campaign/replica-003/shared_initial/emb_34um_batch_summary.json",
                            "candidate_runtime_statuses": [
                                {
                                    "candidate_index": 7,
                                    "output_root": "/scratch/campaign/replica-003/shared_initial/emb/candidate-008",
                                    "runtime_status": "failed",
                                    "runtime_status_path": "/scratch/campaign/replica-003/shared_initial/emb/candidate-008/emb_34um_runtime_status.json",
                                    "error_message": "Candidate interrupted by Slurm signal TERM in job 4394633.",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    guard = module._apply_runtime_failure_guard(plan)

    assert guard["status"] == "blocked"
    assert guard["triggered"] is True
    assert guard["failure_count"] == 1
    assert guard["records"][0]["candidate_index"] == 7
    assert plan["commands_to_submit"] == []
    assert plan["commands_blocked_by_runtime_failure_guard"][0]["command"] == "sbatch --parsable unsafe"


def test_runtime_failure_guard_allows_explicit_post_investigation_override() -> None:
    module = _load_module()
    plan = {
        "commands_to_submit": [{"stage_name": "shared_initial_submit", "command": "sbatch --parsable reviewed"}],
        "stages": [
            {
                "stage_name": "shared_initial_submit",
                "commands": [
                    {
                        "command_index": 0,
                        "batch_report": {
                            "batch_summary_path": "/scratch/campaign/replica-003/shared_initial/emb_34um_batch_summary.json",
                            "candidate_runtime_statuses": [
                                {
                                    "candidate_index": 7,
                                    "output_root": "/scratch/campaign/replica-003/shared_initial/emb/candidate-008",
                                    "runtime_status": "failed",
                                    "runtime_status_path": "/scratch/campaign/replica-003/shared_initial/emb/candidate-008/emb_34um_runtime_status.json",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    guard = module._apply_runtime_failure_guard(plan, allow_runtime_failure_resume=True)

    assert guard["status"] == "override_allowed"
    assert guard["triggered"] is True
    assert plan["commands_to_submit"][0]["command"] == "sbatch --parsable reviewed"
    assert plan["commands_blocked_by_runtime_failure_guard"] == []


def test_resume_helper_retries_slurm_cancelled_candidate_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    failed_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        failed_root / "emb_34um_runtime_status.json",
        {
            "status": "failed",
            "error_message": "Candidate interrupted by Slurm signal TERM in job 12345.",
        },
    )
    monkeypatch.setattr(
        module,
        "_slurm_accounting_for_job",
        lambda job_id: {
            "job_id": job_id,
            "state": "CANCELLED by 11515",
            "reason": "None",
            "elapsed": "00:01:41",
            "exit_code": "0:0",
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    status = command["batch_report"]["candidate_runtime_statuses"][1]

    assert command["blocked"] is False
    assert command["needs_submission"] is True
    assert command["batch_report"]["missing_candidate_indices"] == [1]
    assert command["batch_report"]["failed_candidate_indices"] == []
    assert command["batch_report"]["replacement_required_candidate_indices"] == []
    assert status["runtime_status"] == "failed"
    assert status["runtime_failure_kind"] == "manual_cancellation"
    assert status["manual_cancellation"] is True
    assert status["retry_same_candidate"] is True
    assert plan["commands_to_submit"][0]["stage_array_candidate_indices"] == [1]
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "clear"
    assert guard["triggered"] is False


def test_resume_helper_keeps_explicit_cancelled_status_out_of_failure_guard_without_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    cancelled_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        cancelled_root / "emb_34um_runtime_status.json",
        {
            "status": "cancelled",
            "error_message": "Candidate interrupted by manual campaign pause.",
        },
    )
    monkeypatch.setattr(module, "_slurm_accounting_for_job", lambda job_id: None)

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    status = command["batch_report"]["candidate_runtime_statuses"][1]

    assert command["blocked"] is False
    assert command["needs_submission"] is True
    assert command["batch_report"]["missing_candidate_indices"] == [1]
    assert command["batch_report"]["failed_candidate_indices"] == []
    assert status["runtime_status"] == "cancelled"
    assert status["runtime_failure_kind"] == "manual_cancellation"
    assert status["manual_cancellation"] is True
    assert status["retry_same_candidate"] is True
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "clear"
    assert guard["triggered"] is False


def test_resume_helper_keeps_slurm_timeout_replacement_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    failed_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        failed_root / "emb_34um_runtime_status.json",
        {
            "status": "failed",
            "error_message": "Candidate interrupted by Slurm signal TERM in job 12345.",
        },
    )
    monkeypatch.setattr(
        module,
        "_slurm_accounting_for_job",
        lambda job_id: {
            "job_id": job_id,
            "state": "TIMEOUT",
            "reason": "None",
            "elapsed": "00:12:22",
            "exit_code": "0:0",
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    status = command["batch_report"]["candidate_runtime_statuses"][1]

    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["failed_candidate_indices"] == [1]
    assert command["batch_report"]["replacement_required_candidate_indices"] == [1]
    assert status["runtime_failure_kind"] == "slurm_timeout"
    assert status["retry_same_candidate"] is False
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "blocked"
    assert guard["triggered"] is True


def test_resume_helper_blocks_completed_runtime_missing_final_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    missing_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        missing_root / "emb_34um_runtime_status.json",
        {
            "status": "completed",
            "hdf5": str(missing_root / "emb_34um_missing_indentation.h5"),
        },
    )
    monkeypatch.setattr(module, "_slurm_accounting_for_job", lambda job_id: None)

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    status = command["batch_report"]["candidate_runtime_statuses"][1]

    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["failed_candidate_indices"] == [1]
    assert command["batch_report"]["replacement_required_candidate_indices"] == [1]
    assert status["runtime_status"] == "completed"
    assert status["runtime_failure_kind"] == "missing_final_output_after_completed_runtime"
    assert status["guard_blocking_failure"] is True
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "blocked"
    assert guard["triggered"] is True
    assert guard["records"][0]["runtime_failure_kind"] == "missing_final_output_after_completed_runtime"


def test_runtime_failure_guard_sees_original_failure_after_completed_replacement(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={1})
    failed_root = campaign_root / "replica-001" / "shared_initial" / "emb" / "shared-001"
    _write_json(
        failed_root / "emb_34um_runtime_status.json",
        {
            "status": "timeout",
            "error_message": "Candidate exceeded walltime.",
        },
    )

    replacement_root = campaign_root / "replica-001" / "shared_initial" / "replacement" / "batch-001" / "emb" / "shared-replacement-001"
    replacement_root.mkdir(parents=True, exist_ok=True)
    (replacement_root / RESULT_FILENAME).write_text(
        json.dumps({"candidate_id": replacement_root.name}),
        encoding="utf-8",
    )
    replacement_summary_path = (
        campaign_root
        / "replica-001"
        / "shared_initial"
        / "replacement"
        / "batch-001"
        / module.REPLACEMENT_BATCH_SUMMARY_FILENAME
    )
    _write_json(
        replacement_summary_path,
        {
            "status": "replacement_rendered",
            "failed_candidate_indices": [1],
            "expected_output_roots": [str(replacement_root)],
            "replacement_submission_commands": ["sbatch --parsable replacement_batch_001"],
        },
    )
    batch_summary_path = campaign_root / "replica-001" / "shared_initial" / BATCH_SUMMARY_FILENAME
    batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    batch_summary["replacement_batches"] = [
        {
            "replacement_batch_summary_path": str(replacement_summary_path),
        }
    ]
    _write_json(batch_summary_path, batch_summary)

    plan = module.build_resume_plan(campaign_root=campaign_root)
    batch_report = plan["stages"][0]["commands"][0]["batch_report"]

    assert batch_report["complete"] is True
    assert batch_report["candidate_runtime_statuses"][1]["runtime_status"] == "missing"
    assert batch_report["original_candidate_runtime_statuses"][1]["runtime_failure_kind"] == "timeout"
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "blocked"
    assert guard["triggered"] is True
    assert any(
        record["source"] == "original_stage_batch" and record["candidate_index"] == 1
        for record in guard["records"]
    )


def test_runtime_failure_guard_scans_campaign_status_files_outside_current_batches(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices=set())
    orphan_root = campaign_root / "diagnostics" / "orphaned-failed-candidate"
    _write_json(
        orphan_root / "emb_34um_runtime_status.json",
        {
            "candidate_id": "orphaned-failed-candidate",
            "status": "timeout",
            "error_message": "Candidate exceeded walltime.",
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["commands_to_submit"] == []
    guard = module._apply_runtime_failure_guard(plan)
    assert guard["status"] == "blocked"
    assert guard["triggered"] is True
    assert guard["records"][0]["source"] == "campaign_runtime_status_scan"
    assert guard["records"][0]["candidate_id"] == "orphaned-failed-candidate"


def test_resume_helper_submits_only_until_max_active_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _seed_campaign(campaign_root, module=module, missing_indices={0, 2})
    (campaign_root / "replica-001" / "al-step-01" / "emb" / "al-002" / RESULT_FILENAME).unlink()

    calls: list[list[str] | str] = []
    squeue_outputs = iter(["\n".join(["1001", "1002"]) + "\n", "\n".join(["1001", "1002"]) + "\n"])

    def fake_run(command, **kwargs):
        if isinstance(command, list) and command[0] == "squeue":
            assert command[4] == "-t"
            assert command[5] == "PENDING,RUNNING,CONFIGURING"
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": next(squeue_outputs), "stderr": ""})()
        if isinstance(command, list) and command[0] == "sbatch":
            assert kwargs.get("shell") is False
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
    assert len([call for call in calls if isinstance(call, list) and call[0] == "sbatch"]) == 1
    assert submission["submitted_commands"][0]["stage_name"] == "shared_initial_submit"
    assert submission["deferred_commands"][0]["deferred_reason"] == "active_job_limit_reached"


def test_submit_missing_commands_moves_sbatch_export_values_into_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        if isinstance(command, list) and command[0] == "squeue":
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if isinstance(command, list) and command[0] == "sbatch":
            calls.append((command, kwargs))
            assert kwargs.get("shell") is False
            assert "--export=ALL" in command
            assert all("TIMESTAMP=" not in item for item in command)
            env = kwargs.get("env")
            assert isinstance(env, dict)
            assert env["TIMESTAMP"] == "20260529T000000Z"
            assert env["VAULT_ROOT"] == "/tmp/vault root"
            assert env["MODE"] == "lhs-step-03"
            return type("Result", (), {"returncode": 0, "stdout": "4409999\n", "stderr": ""})()
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    submission = module._submit_missing_commands(
        [
            {
                "stage_name": "lhs-step-03-submit",
                "command_index": 0,
                "command": (
                    "sbatch --parsable --time=00:12:00 "
                    "--export=ALL,TIMESTAMP=20260529T000000Z,VAULT_ROOT='/tmp/vault root',MODE=lhs-step-03 "
                    "/tmp/wrapper.sbatch"
                ),
            }
        ],
        user="test-user",
        max_active_jobs=1,
    )

    assert len(calls) == 1
    assert submission["submitted_count"] == 1
    submitted = submission["submitted_commands"][0]
    assert submitted["stdout"] == "4409999"
    assert submitted["launch_export_strategy"] == "environment_plus_export_all"
    assert submitted["launch_environment_keys"] == ["MODE", "TIMESTAMP", "VAULT_ROOT"]
    assert "--export=ALL," not in submitted["launch_command"]


def test_resume_helper_filters_commands_by_stage_and_submission_kind(tmp_path: Path) -> None:
    module = _load_module()
    commands = [
        {"stage_name": "shared_initial_submit", "submission_kind": "stage_batch", "command": "sbatch shared"},
        {"stage_name": "shared_initial_submit", "submission_kind": "replacement_batch", "command": "sbatch replacement"},
        {"stage_name": "lhs-step-01-submit", "submission_kind": "stage_batch", "command": "sbatch lhs"},
    ]

    selected, filtered = module._filter_commands_to_submit(
        commands,
        stage_names={"shared_initial_submit"},
        submission_kinds={"replacement_batch"},
    )

    assert selected == [commands[1]]
    assert len(filtered) == 2
    assert {item["filtered_reason"] for item in filtered} == {
        "submission_kind_filter",
        "stage_name_filter",
    }


def test_resume_helper_blocks_unrendered_adaptive_placeholders(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _write_pilot_summary(campaign_root, passed=True)
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


def test_repair_helper_renders_reserve_replacement_and_resume_uses_replacement_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        assert candidate.parameters["runtime_fingerprint"]["Lx"] == pytest.approx(20.0)
        assert candidate.parameters["runtime_fingerprint"]["Ly"] == pytest.approx(20.0)
        assert candidate.parameters["runtime_fingerprint"]["Lz"] == pytest.approx(24.0)
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    assert result["replacement_submission_commands"] == ["sbatch --parsable replacement_batch_001"]
    replacement_record = result["replacement_records"][0]
    assert replacement_record["sampler"] == "AL"
    assert replacement_record["replica"] == 1
    assert replacement_record["cycle"] == 1
    assert replacement_record["reason"] == "runtime_failure_or_quarantine"
    assert replacement_record["attempt_count"] == 1
    assert replacement_record["original_candidate_id"] == "al-000"
    assert replacement_record["replacement_candidate_id"]
    assert isinstance(replacement_record["skipped_by_feasibility_filter"], bool)
    assert replacement_record["radp"] == pytest.approx(6.62)
    assert replacement_record["shell_th"] == pytest.approx(3.75e-9)
    assert replacement_record["bpress"] == pytest.approx(-91.0)
    assert replacement_record["runtime_fingerprint"]["radp"] == pytest.approx(6.62)
    assert replacement_record["runtime_fingerprint"]["shell_th"] == pytest.approx(3.75e-9)
    assert replacement_record["runtime_fingerprint"]["Lx"] == pytest.approx(20.0)
    assert replacement_record["runtime_fingerprint"]["Lz"] == pytest.approx(24.0)

    batch_summary = json.loads(
        (campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert batch_summary["expected_output_roots"][0].endswith("/replacement/batch-001/emb/emb-34um-dnn-causal-validation-rep001-al-step01-r01-001")
    assert batch_summary["replacement_count"] == 1
    selection_manifest = json.loads(
        (campaign_root / "replica-001" / "al-step-01" / SELECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert selection_manifest["candidate_records"][0]["replacement"] is True
    assert selection_manifest["candidate_records"][0]["replacement_for"] == "al-000"
    assert selection_manifest["candidate_records"][0]["radp"] == pytest.approx(6.62)
    assert selection_manifest["candidate_records"][0]["shell_th"] == pytest.approx(3.75e-9)
    assert selection_manifest["candidate_records"][0]["runtime_fingerprint"]["Lx"] == pytest.approx(20.0)
    assert selection_manifest["candidate_reserve_count"] == 0
    assert batch_summary["replacement_batches"][0]["expected_output_roots"] == [batch_summary["expected_output_roots"][0]]
    replacement_manifest = json.loads(Path(replacement_record["replacement_candidate_manifest_path"]).read_text(encoding="utf-8"))
    assert replacement_manifest["normalized_payload"]["parameters"]["radp"] == pytest.approx(6.62)
    assert replacement_manifest["normalized_payload"]["parameters"]["shell_th"] == pytest.approx(3.75e-9)
    assert replacement_manifest["normalized_payload"]["parameters"]["bpress"] == pytest.approx(-91.0)
    assert replacement_manifest["normalized_payload"]["fingerprint"]["radp"] == pytest.approx(6.62)
    assert replacement_manifest["rendered_payload"]["request_payload"]["fingerprint"]["Lx"] == pytest.approx(20.0)

    plan = module.build_resume_plan(campaign_root=campaign_root)
    assert plan["blocked_batch_count"] == 0
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["submission_kind"] == "replacement_batch"
    assert plan["stages"][0]["commands"][0]["blocked"] is False
    assert plan["stages"][0]["commands"][0]["needs_submission"] is True
    assert plan["stages"][0]["commands"][0]["batch_report"]["submission_strategy"] == "replacement_batches"

    replacement_output_root = Path(batch_summary["expected_output_roots"][0])
    (replacement_output_root / RESULT_FILENAME).write_text(
        json.dumps({"candidate_id": replacement_output_root.name}),
        encoding="utf-8",
    )
    completed_plan = module.build_resume_plan(campaign_root=campaign_root)
    assert completed_plan["missing_batch_count"] == 0
    assert completed_plan["complete_batch_count"] == 1


def test_repair_helper_derives_al_reserve_from_scored_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)
    stage_root = campaign_root / "replica-001" / "al-step-01"
    selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selected_rows = list(selection_manifest["candidate_records"])
    selection_manifest["candidate_reserve"] = []
    selection_manifest["candidate_reserve_count"] = 0
    train_manifest_path = stage_root / "train_score_select" / "emb_34um_dnn_train_score_select_manifest.json"
    selection_manifest["train_score_select_manifest_path"] = str(train_manifest_path)
    selection_manifest_path.write_text(json.dumps(selection_manifest, indent=2), encoding="utf-8")

    scored_rows = [
        {
            "candidate_id": row["selection_payload"]["candidate_pool_id"],
            "ka": row["ka"],
            "kb": row["kb"],
            "radp": row["radp"],
            "shell_th": row["shell_th"],
            "runtime_fingerprint": row["runtime_fingerprint"],
            "candidate_log10_point": [4.0, 4.0, 0.5, 0.5],
            "ensemble_disagreement": 0.1,
            "ensemble_disagreement_norm": 0.1,
            "diversity_term": 0.0,
            "acquisition_score": 0.1,
        }
        for row in selected_rows
    ]
    scored_rows.append(
        {
            "candidate_id": "derived-reserve-001",
            "ka": 9100.0,
            "kb": 8200.0,
            "radp": 6.58,
            "shell_th": 3.9e-9,
            "runtime_fingerprint": _runtime_fingerprint(radp=6.58, shell_th=3.9e-9),
            "candidate_log10_point": [3.95, 3.91, 0.85, 0.6],
            "ensemble_disagreement": 0.9,
            "ensemble_disagreement_norm": 0.9,
            "diversity_term": 0.4,
            "acquisition_score": 1.0,
        }
    )
    _write_json(
        train_manifest_path,
        {
            "candidate_space": "d4",
            "candidate_scores": scored_rows,
            "existing_points": {"points": [], "diversity_weight": 0.25},
        },
    )

    captured: dict[str, object] = {}

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        captured["candidate"] = candidates[0]
        output_root = batch_root / "emb" / candidates[0].candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidates[0], output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {"scheduler_boundary": {"submission": {"submitted": False, "submission_commands": ["sbatch --parsable derived"]}}},
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)
    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    assert result["replacement_submission_commands"] == ["sbatch --parsable derived"]
    candidate = captured["candidate"]
    assert candidate.parameters["ka"] == pytest.approx(9100.0)
    assert candidate.parameters["kb"] == pytest.approx(8200.0)
    assert candidate.parameters["radp"] == pytest.approx(6.58)
    updated_selection = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    assert updated_selection["replacement_records"][0]["candidate_pool_id"] == "derived-reserve-001"


def test_repair_helper_derives_al_reserve_without_duplicating_current_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)
    stage_root = campaign_root / "replica-001" / "al-step-01"
    selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    train_manifest_path = stage_root / "train_score_select" / "emb_34um_dnn_train_score_select_manifest.json"
    selection_manifest["train_score_select_manifest_path"] = str(train_manifest_path)
    selection_manifest_path.write_text(json.dumps(selection_manifest, indent=2), encoding="utf-8")

    runtime_fingerprint = _runtime_fingerprint(radp=6.62, shell_th=3.75e-9)
    _write_json(
        train_manifest_path,
        {
            "candidate_space": "d4",
            "candidate_scores": [
                {
                    "candidate_id": "reserve-001",
                    "ka": 9100.0,
                    "kb": 8200.0,
                    "radp": 6.62,
                    "shell_th": 3.75e-9,
                    "runtime_fingerprint": runtime_fingerprint,
                    "candidate_log10_point": [3.95, 3.91, 0.5, 0.5],
                    "ensemble_disagreement": 1.0,
                    "ensemble_disagreement_norm": 1.0,
                    "diversity_term": 0.5,
                    "acquisition_score": 1.0,
                },
                {
                    "candidate_id": "reserve-002",
                    "ka": 9300.0,
                    "kb": 8400.0,
                    "radp": 6.64,
                    "shell_th": 3.85e-9,
                    "runtime_fingerprint": _runtime_fingerprint(radp=6.64, shell_th=3.85e-9),
                    "candidate_log10_point": [3.97, 3.92, 0.6, 0.55],
                    "ensemble_disagreement": 0.9,
                    "ensemble_disagreement_norm": 0.9,
                    "diversity_term": 0.4,
                    "acquisition_score": 0.9,
                },
            ],
            "existing_points": {"points": [], "diversity_weight": 0.25},
        },
    )

    captured: dict[str, object] = {}

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        captured["candidate_pool_ids"] = [candidate.metadata["candidate_pool_id"] for candidate in candidates]
        expected_output_roots = []
        manifest_paths = []
        for candidate in candidates:
            output_root = batch_root / "emb" / candidate.candidate_id
            output_root.mkdir(parents=True, exist_ok=True)
            manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
            _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
            expected_output_roots.append(str(output_root))
            manifest_paths.append(str(manifest_path))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": len(candidates),
                "rendered_candidate_manifests": manifest_paths,
                "expected_output_roots": expected_output_roots,
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {"scheduler_boundary": {"submission": {"submitted": False, "submission_commands": ["sbatch --parsable derived"]}}},
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": manifest_paths,
            "expected_output_roots": expected_output_roots,
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)
    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0, 1),
    )

    pool_ids = [record["candidate_pool_id"] for record in result["replacement_records"]]
    assert captured["candidate_pool_ids"] == ["reserve-001", "reserve-002"]
    assert pool_ids == ["reserve-001", "reserve-002"]
    assert len(set(pool_ids)) == 2


@pytest.mark.parametrize("runtime_status", ["timeout", "timed_out", "failed"])
@pytest.mark.parametrize("result_exists", [False, True])
def test_resume_helper_renders_replacement_for_failed_or_timed_out_runtime_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_status: str,
    result_exists: bool,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)
    failed_root = campaign_root / "replica-001" / "al-step-01" / "emb" / "al-000"
    if result_exists:
        (failed_root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": "stale-result"}), encoding="utf-8")
    _write_json(
        failed_root / "emb_34um_runtime_status.json",
        {"status": runtime_status, "retry_count": 0, "retry_limit": 0},
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)
    command = plan["stages"][0]["commands"][0]
    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["blocked_reason"] == "replacement_required_for_failed_candidates"
    assert command["batch_report"]["quarantined_candidate_indices"] == [0]
    assert plan["commands_to_submit"] == []

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert retry_limit == 0
        candidate = candidates[0]
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)
    rendered = module.render_replacements_for_resume_plan(plan, repair_module=repair)
    assert rendered["rendered_count"] == 1
    assert rendered["rendered"][0]["failed_candidate_indices"] == [0]

    refreshed = module.build_resume_plan(campaign_root=campaign_root)
    assert refreshed["blocked_batch_count"] == 0
    assert refreshed["missing_batch_count"] == 1
    assert refreshed["commands_to_submit"][0]["submission_kind"] == "replacement_batch"


def test_resume_helper_subsets_retryable_replacement_batch_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    stage_root, _ = _seed_non_al_repair_stage(campaign_root, module=module, stage_name="shared_initial")
    for index in range(3):
        root = stage_root / "emb" / f"shared-{index:03d}"
        result = root / RESULT_FILENAME
        if result.exists():
            result.unlink()
        _write_json(root / "emb_34um_runtime_status.json", {"status": "timeout"})

    replacement_root = stage_root / "replacement" / "batch-001"
    replacement_outputs = []
    for index in range(3):
        output_root = replacement_root / "emb" / f"replacement-{index:03d}"
        replacement_outputs.append(str(output_root))
        output_root.mkdir(parents=True, exist_ok=True)
        if index == 0:
            (output_root / RESULT_FILENAME).write_text(json.dumps({"candidate_id": output_root.name}), encoding="utf-8")
        elif index == 1:
            _write_json(
                output_root / "emb_34um_runtime_status.json",
                {
                    "status": "failed",
                    "error_message": "Candidate interrupted by Slurm signal TERM in job 54321.",
                },
            )
    _write_json(
        replacement_root / module.REPLACEMENT_BATCH_SUMMARY_FILENAME,
        {
            "status": "replacement_rendered",
            "failed_candidate_indices": [0, 1, 2],
            "expected_output_roots": replacement_outputs,
            "replacement_submission_commands": ["sbatch --parsable --array=0-2%3 replacement_batch"],
        },
    )
    monkeypatch.setattr(
        module,
        "_slurm_accounting_for_job",
        lambda job_id: {
            "job_id": job_id,
            "state": "CANCELLED by 11515",
            "reason": "None",
            "elapsed": "00:01:41",
            "exit_code": "0:0",
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert len(plan["commands_to_submit"]) == 1
    command = plan["commands_to_submit"][0]
    assert command["submission_kind"] == "replacement_batch"
    assert command["replacement_array_subset"] is True
    assert command["replacement_array_candidate_indices"] == [1, 2]
    assert "--array=1-2%3" in command["command"]
    replacement_batch = command["batch_report"]["replacement_batches"][0]
    assert replacement_batch["failed_replacement_candidate_indices"] == []
    assert replacement_batch["missing_candidate_indices"] == [1, 2]
    assert replacement_batch["replacement_runtime_statuses"][1]["retry_same_candidate"] is True


def test_resume_helper_detects_completed_replacement_batches_via_summary_path_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    original_batch_summary_path = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_output_root = original_batch_summary["expected_output_roots"][0]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    replacement_output_root = Path(result["replacement_records"][0]["replacement_output_root"])
    (replacement_output_root / RESULT_FILENAME).write_text(
        json.dumps({"candidate_id": replacement_output_root.name}),
        encoding="utf-8",
    )

    patched_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    patched_batch_summary["expected_output_roots"][0] = original_output_root
    patched_batch_summary["replacement_batches"][0].pop("expected_output_roots", None)
    _write_json(original_batch_summary_path, patched_batch_summary)

    batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    assert batch_summary["replacement_batches"][0]["replacement_batch_summary_path"].endswith("emb_34um_dnn_causal_validation_replacement_batch_summary.json")

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["complete_batch_count"] == 1
    assert plan["missing_batch_count"] == 0
    assert plan["blocked_batch_count"] == 0
    assert plan["commands_to_submit"] == []
    assert plan["stages"][0]["commands"][0]["batch_report"]["complete"] is True


def test_resume_helper_submits_uncovered_original_indices_after_replacement_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    controller_path = campaign_root / module.CONTROLLER_MANIFEST_FILENAME
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["stages"][0]["commands"][0]["command"] = "sbatch --parsable --array=0-2%3 original_al_step_01"
    _write_json(controller_path, controller)

    original_batch_summary_path = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_output_root = original_batch_summary["expected_output_roots"][0]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )
    replacement_output_root = Path(result["replacement_records"][0]["replacement_output_root"])
    (replacement_output_root / RESULT_FILENAME).write_text(
        json.dumps({"candidate_id": replacement_output_root.name}),
        encoding="utf-8",
    )

    Path(original_batch_summary["expected_output_roots"][2], RESULT_FILENAME).unlink()
    patched_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    patched_batch_summary["expected_output_roots"][0] = original_output_root
    patched_batch_summary["replacement_batches"][0].pop("expected_output_roots", None)
    _write_json(original_batch_summary_path, patched_batch_summary)

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["complete_batch_count"] == 0
    assert plan["missing_batch_count"] == 1
    assert plan["blocked_batch_count"] == 0
    assert len(plan["commands_to_submit"]) == 1
    command = plan["commands_to_submit"][0]
    assert command["submission_kind"] == "stage_batch"
    assert command["stage_array_subset"] is True
    assert command["stage_array_candidate_indices"] == [2]
    assert "--array=2%3" in command["command"]
    report = plan["stages"][0]["commands"][0]["batch_report"]
    assert report["replacement_rendered_candidate_indices"] == [0]
    assert report["covered_missing_candidate_indices"] == [0]
    assert report["uncovered_missing_candidate_indices"] == [2]


def test_resume_helper_blocks_running_replacement_batches_instead_of_resubmitting_original_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    original_batch_summary_path = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_output_root = original_batch_summary["expected_output_roots"][0]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    replacement_summary_path = (
        campaign_root
        / "replica-001"
        / "al-step-01"
        / "replacement"
        / "batch-001"
        / "emb_34um_dnn_causal_validation_replacement_batch_summary.json"
    )
    replacement_summary = json.loads(replacement_summary_path.read_text(encoding="utf-8"))
    replacement_summary["status"] = "running"
    replacement_summary.pop("expected_output_roots", None)
    _write_json(replacement_summary_path, replacement_summary)

    patched_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    patched_batch_summary["expected_output_roots"][0] = original_output_root
    patched_batch_summary["replacement_batches"][0].pop("expected_output_roots", None)
    _write_json(original_batch_summary_path, patched_batch_summary)

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["blocked_batch_count"] == 1
    assert plan["missing_batch_count"] == 0
    assert plan["commands_to_submit"] == []
    command = plan["stages"][0]["commands"][0]
    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["blocked_reason"] == "replacement_batches_incomplete"
    assert command["batch_report"]["ready_for_submission"] is False


def test_resume_helper_synthesizes_submission_for_render_only_replacement_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    original_batch_summary_path = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_output_root = original_batch_summary["expected_output_roots"][0]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        _write_json(
            batch_root / BATCH_SUMMARY_FILENAME,
            {
                "candidate_count": 1,
                "rendered_candidate_manifests": [str(manifest_path)],
                "expected_output_roots": [str(output_root)],
            },
        )
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": [],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    replacement_summary_path = Path(result["replacement_batch_summary_path"])
    replacement_summary = json.loads(replacement_summary_path.read_text(encoding="utf-8"))
    replacement_summary["replacement_submission_commands"] = []
    batch_payload = replacement_summary.get("batch_payload", {})
    if isinstance(batch_payload, dict):
        scheduler_boundary = batch_payload.setdefault("scheduler_boundary", {})
        if isinstance(scheduler_boundary, dict):
            submission = scheduler_boundary.setdefault("submission", {})
            if isinstance(submission, dict):
                submission["submission_commands"] = []
    _write_json(replacement_summary_path, replacement_summary)

    patched_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    patched_batch_summary["expected_output_roots"][0] = original_output_root
    _write_json(original_batch_summary_path, patched_batch_summary)

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["blocked_batch_count"] == 0
    assert plan["missing_batch_count"] == 1
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["submission_kind"] == "replacement_batch"
    assert plan["commands_to_submit"][0]["command"].startswith("sbatch --parsable --time=00:12:00 --array=0%1")
    assert "BATCH_DIR_OVERRIDE=" in plan["commands_to_submit"][0]["command"]
    assert plan["stages"][0]["commands"][0]["batch_report"]["submission_strategy"] == "replacement_batches"
    assert plan["stages"][0]["commands"][0]["batch_report"]["replacement_batches"][0]["synthesized_submission_commands"] is True
    assert plan["stages"][0]["commands"][0]["blocked"] is False
    assert plan["stages"][0]["commands"][0]["needs_submission"] is True


def test_resume_helper_discovers_replacement_batch_summary_without_stage_manifest_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    _seed_al_replacement_stage(campaign_root, module=module)

    original_batch_summary_path = campaign_root / "replica-001" / "al-step-01" / BATCH_SUMMARY_FILENAME
    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_output_root = original_batch_summary["expected_output_roots"][0]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": ["sbatch --parsable replacement_batch_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=1,
        failed_candidate_indices=(0,),
    )

    original_batch_summary = json.loads(original_batch_summary_path.read_text(encoding="utf-8"))
    original_batch_summary["expected_output_roots"][0] = original_output_root
    original_batch_summary.pop("replacement_batches", None)
    original_batch_summary.pop("replacement_manifest_path", None)
    _write_json(original_batch_summary_path, original_batch_summary)

    replacement_manifest_path = (
        campaign_root
        / "replica-001"
        / "al-step-01"
        / "emb_34um_dnn_causal_validation_replacement_manifest.json"
    )
    replacement_manifest = json.loads(replacement_manifest_path.read_text(encoding="utf-8"))
    replacement_manifest["replacement_batches"] = []
    _write_json(replacement_manifest_path, replacement_manifest)

    replacement_summary_path = Path(result["replacement_batch_summary_path"])
    assert replacement_summary_path.is_file()

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["blocked_batch_count"] == 0
    assert plan["missing_batch_count"] == 1
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["submission_kind"] == "replacement_batch"
    assert plan["commands_to_submit"][0]["command"] == "sbatch --parsable replacement_batch_001"
    assert plan["commands_to_submit"][0]["replacement_batch_summary_path"] == str(replacement_summary_path)
    command = plan["stages"][0]["commands"][0]
    assert command["blocked"] is False
    assert command["needs_submission"] is True
    assert command["batch_report"]["submission_strategy"] == "replacement_batches"


@pytest.mark.parametrize(
    ("stage_name", "expected_source_index"),
    [
        ("shared_initial", 3),
        ("lhs-step-01", 7),
        ("unseen_test", 2),
    ],
)
def test_repair_helper_replaces_non_al_stages_with_next_unused_design_stream_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    expected_source_index: int,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / stage_name
    stage_root, candidate_count = _seed_non_al_repair_stage(campaign_root, module=module, stage_name=stage_name)

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.parameters["radp"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["radp"])
        assert candidate.parameters["shell_th"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["shell_th"])
        assert candidate.parameters["bpress"] == pytest.approx(candidate.parameters["runtime_fingerprint"]["bpress"])
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": [f"sbatch --parsable replacement_{stage_name}_001"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    result = repair.render_al_stage_replacements(
        campaign_root=campaign_root,
        replica=1,
        cycle=None,
        stage=stage_name,
        failed_candidate_indices=(0,),
    )

    assert result["replacement_policy"] == "deterministic_next_unused_fresh_dpd"
    assert f"src{result['replacement_records'][0]['source_index']:06d}" in result["replacement_records"][0]["replacement_candidate_id"]
    assert result["replacement_records"][0]["source_index"] >= expected_source_index
    assert result["replacement_records"][0]["requested_source_index"] == expected_source_index
    assert result["replacement_records"][0]["sampler"] == "LHS"
    assert result["replacement_records"][0]["reason"] == "runtime_failure_or_quarantine"
    assert "attempt_count" in result["replacement_records"][0]
    assert "skipped_by_feasibility_filter" in result["replacement_records"][0]

    from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import _sample_points_with_source_indices_4d
    from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import is_dnn_causal_low_corner_excluded

    accepted_source_index, expected_ka, expected_kb, expected_radp, expected_shell_th = _sample_points_with_source_indices_4d(
        start_index=expected_source_index,
        count=1,
    )[0]
    assert result["replacement_records"][0]["source_index"] == accepted_source_index
    replacement_manifest = json.loads(Path(result["replacement_records"][0]["replacement_candidate_manifest_path"]).read_text(encoding="utf-8"))
    selection_manifest = json.loads(
        (stage_root / "emb_34um_dnn_causal_validation_selection_manifest.json").read_text(encoding="utf-8")
    )
    failed_record = selection_manifest["original_candidate_records"][0]
    assert replacement_manifest["normalized_payload"]["parameters"]["ka"] == pytest.approx(expected_ka)
    assert replacement_manifest["normalized_payload"]["parameters"]["kb"] == pytest.approx(expected_kb)
    assert replacement_manifest["normalized_payload"]["parameters"]["radp"] == pytest.approx(expected_radp)
    assert replacement_manifest["normalized_payload"]["parameters"]["shell_th"] == pytest.approx(expected_shell_th)
    assert replacement_manifest["normalized_payload"]["parameters"]["bpress"] == pytest.approx(-91.0)
    assert replacement_manifest["normalized_payload"]["fingerprint"]["radp"] == pytest.approx(expected_radp)
    assert replacement_manifest["normalized_payload"]["fingerprint"]["shell_th"] == pytest.approx(expected_shell_th)
    assert not is_dnn_causal_low_corner_excluded(
        replacement_manifest["normalized_payload"]["parameters"]["ka"],
        replacement_manifest["normalized_payload"]["parameters"]["kb"],
    )
    assert replacement_manifest["normalized_payload"]["parameters"]["ka"] != pytest.approx(failed_record["ka"])
    assert replacement_manifest["normalized_payload"]["parameters"]["kb"] != pytest.approx(failed_record["kb"])
    assert selection_manifest["replacement_policy"] == "deterministic_next_unused_fresh_dpd"
    assert selection_manifest["replacement_records"][0]["source_index"] == accepted_source_index
    assert selection_manifest["replacement_records"][0]["requested_source_index"] == expected_source_index
    assert selection_manifest["replacement_records"][0]["failed_candidate_id"] == selection_manifest["original_candidate_records"][0]["candidate_id"]
    assert selection_manifest["replacement_records"][0]["radp"] == pytest.approx(expected_radp)
    assert selection_manifest["replacement_records"][0]["shell_th"] == pytest.approx(expected_shell_th)
    assert selection_manifest["replacement_records"][0]["runtime_fingerprint"]["radp"] == pytest.approx(expected_radp)
    assert selection_manifest["replacement_records"][0]["runtime_fingerprint"]["shell_th"] == pytest.approx(expected_shell_th)
    assert selection_manifest["candidate_count"] == candidate_count

    plan = module.build_resume_plan(campaign_root=campaign_root)
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["stage_name"] == (
        "unseen_test_submit" if stage_name == "unseen_test" else f"{stage_name}-submit"
    )
    assert plan["commands_to_submit"][0]["submission_kind"] == "replacement_batch"


def test_resume_helper_renders_late_failures_after_existing_replacement_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    stage_root, _ = _seed_non_al_repair_stage(campaign_root, module=module, stage_name="shared_initial")

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        candidate = candidates[0]
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": [f"sbatch --parsable {batch_id}"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    first_failed_root = stage_root / "emb" / "shared-000"
    _write_json(first_failed_root / "emb_34um_runtime_status.json", {"status": "failed"})
    first_plan = module.build_resume_plan(campaign_root=campaign_root)
    first_rendered = module.render_replacements_for_resume_plan(first_plan, repair_module=repair)
    assert first_rendered["rendered_count"] == 1
    assert first_rendered["rendered"][0]["failed_candidate_indices"] == [0]

    second_failed_root = stage_root / "emb" / "shared-001"
    _write_json(second_failed_root / "emb_34um_runtime_status.json", {"status": "timeout"})
    second_plan = module.build_resume_plan(campaign_root=campaign_root)
    command = second_plan["stages"][0]["commands"][0]

    assert command["blocked"] is True
    assert command["batch_report"]["submission_strategy"] == "replacement_required"
    assert command["batch_report"]["replacement_rendered_candidate_indices"] == [0]
    assert command["batch_report"]["replacement_required_candidate_indices"] == [1]

    second_rendered = module.render_replacements_for_resume_plan(second_plan, repair_module=repair)
    assert second_rendered["rendered_count"] == 1
    assert second_rendered["rendered"][0]["failed_candidate_indices"] == [1]
    assert second_rendered["rendered"][0]["replacement_batch_summary_path"].endswith(
        "replacement/batch-002/emb_34um_dnn_causal_validation_replacement_batch_summary.json"
    )


def test_resume_helper_honors_stage_filter_when_rendering_replacements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    stage_root, _ = _seed_non_al_repair_stage(campaign_root, module=module, stage_name="shared_initial")

    failed_root = stage_root / "emb" / "shared-000"
    _write_json(failed_root / "emb_34um_runtime_status.json", {"status": "failed"})
    plan = module.build_resume_plan(campaign_root=campaign_root)

    filtered = module.render_replacements_for_resume_plan(
        plan,
        repair_module=repair,
        stage_names={"unseen_test_submit"},
    )
    assert filtered["rendered_count"] == 0
    assert filtered["stage_name_filter"] == ["unseen_test_submit"]

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        candidate = candidates[0]
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": [f"sbatch --parsable {batch_id}"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)
    rendered = module.render_replacements_for_resume_plan(
        plan,
        repair_module=repair,
        stage_names={"shared_initial-submit"},
    )
    assert rendered["rendered_count"] == 1


def test_resume_helper_renders_next_replacement_when_replacement_candidate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repair = _load_repair_module()
    campaign_root = tmp_path / "campaign"
    stage_root, _ = _seed_non_al_repair_stage(campaign_root, module=module, stage_name="shared_initial")

    def fake_render_replacement_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
        candidate = candidates[0]
        output_root = batch_root / "emb" / candidate.candidate_id
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
        _write_json(manifest_path, _rendered_candidate_manifest_payload(candidate, output_root=output_root))
        batch_manifest_path = batch_root / "replacement_gate_manifest.json"
        _write_json(
            batch_manifest_path,
            {
                "scheduler_boundary": {
                    "submission": {
                        "submitted": False,
                        "submission_commands": [f"sbatch --parsable {batch_id}"],
                    }
                }
            },
        )
        return {
            "manifest_path": str(batch_manifest_path),
            "rendered_candidate_manifests": [str(manifest_path)],
            "expected_output_roots": [str(output_root)],
        }

    monkeypatch.setattr(repair, "_render_replacement_batch", fake_render_replacement_batch)

    original_failed_root = stage_root / "emb" / "shared-000"
    _write_json(original_failed_root / "emb_34um_runtime_status.json", {"status": "failed"})
    first_rendered = module.render_replacements_for_resume_plan(
        module.build_resume_plan(campaign_root=campaign_root),
        repair_module=repair,
    )
    first_replacement_summary = json.loads(
        Path(first_rendered["rendered"][0]["replacement_batch_summary_path"]).read_text(encoding="utf-8")
    )
    first_replacement_root = Path(first_replacement_summary["expected_output_roots"][0])
    _write_json(first_replacement_root / "emb_34um_runtime_status.json", {"status": "timeout"})

    retry_plan = module.build_resume_plan(campaign_root=campaign_root)
    command = retry_plan["stages"][0]["commands"][0]
    assert command["blocked"] is True
    assert command["batch_report"]["replacement_rendered_candidate_indices"] == [0]
    assert command["batch_report"]["replacement_required_candidate_indices"] == [0]
    assert command["batch_report"]["replacement_batches"][0]["failed_original_candidate_indices"] == [0]
    assert retry_plan["commands_to_submit"] == []

    second_rendered = module.render_replacements_for_resume_plan(retry_plan, repair_module=repair)
    assert second_rendered["rendered_count"] == 1
    assert second_rendered["rendered"][0]["failed_candidate_indices"] == [0]
    assert second_rendered["rendered"][0]["replacement_batch_summary_path"].endswith(
        "replacement/batch-002/emb_34um_dnn_causal_validation_replacement_batch_summary.json"
    )

    ready_plan = module.build_resume_plan(campaign_root=campaign_root)
    assert len(ready_plan["commands_to_submit"]) == 1
    assert ready_plan["commands_to_submit"][0]["submission_kind"] == "replacement_batch"
    assert ready_plan["commands_to_submit"][0]["replacement_batch_summary_path"].endswith(
        "replacement/batch-002/emb_34um_dnn_causal_validation_replacement_batch_summary.json"
    )


def test_resume_plan_skips_replicas_beyond_active_replicate_count(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _write_pilot_summary(campaign_root, passed=True)
    active_summary = campaign_root / "replica-001" / "shared_initial" / BATCH_SUMMARY_FILENAME
    inactive_summary = campaign_root / "replica-004" / "shared_initial" / BATCH_SUMMARY_FILENAME
    _write_json(
        active_summary,
        {"status": "selection_rendered", "candidate_count": 1, "expected_output_roots": [str(active_summary.parent / "emb" / "a-001")]},
    )
    _write_json(
        inactive_summary,
        {"status": "selection_rendered", "candidate_count": 1, "expected_output_roots": [str(inactive_summary.parent / "emb" / "a-004")]},
    )
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "execution_mode": "execute",
            "active_replicate_count": 3,
            "stages": [
                {
                    "name": "shared_initial_submit",
                    "command_type": "submission",
                    "commands": [
                        {"command": "sbatch --parsable shared_001"},
                        {"command": "sbatch --parsable shared_004"},
                    ],
                    "expected_output_roots": [str(active_summary), str(inactive_summary)],
                }
            ],
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)
    assert plan["active_replicate_count"] == 3
    assert plan["stages"][0]["skipped_inactive_replica_count"] == 1
    assert len(plan["commands_to_submit"]) == 1
    assert plan["commands_to_submit"][0]["command"] == "sbatch --parsable shared_001"


def test_resume_helper_blocks_production_submissions_without_pilot_verification_summary(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    shared_summary = campaign_root / "replica-001" / "shared_initial" / BATCH_SUMMARY_FILENAME
    _write_json(
        shared_summary,
        {
            "status": "selection_rendered",
            "candidate_count": 2,
            "expected_output_roots": [
                str(shared_summary.parent / "emb" / "shared-001"),
                str(shared_summary.parent / "emb" / "shared-002"),
            ],
        },
    )
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "execution_mode": "execute",
            "stages": [
                {"name": "prepare_design", "command_type": "prepare", "commands": [{"command": "python prepare.py"}], "expected_output_roots": []},
                {
                    "name": "pilot_submit",
                    "command_type": "submission",
                    "commands": [{"command": "sbatch --parsable pilot_001"}],
                    "expected_output_roots": [str(campaign_root / "pilot" / BATCH_SUMMARY_FILENAME)],
                },
                {"name": "pilot_verify", "command_type": "verification", "commands": [{"command": "python pilot_verify.py"}], "expected_output_roots": [str(campaign_root / "pilot" / "validation" / "emb_34um_dnn_causal_validation_pilot_summary.json")]},
                {
                    "name": "shared_initial_submit",
                    "command_type": "submission",
                    "commands": [{"command": "sbatch --parsable shared_001"}],
                    "expected_output_roots": [str(shared_summary)],
                },
            ],
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["pilot_verification"]["passed"] is False
    assert plan["pilot_verification"]["status"] == "missing"
    assert plan["commands_to_submit"] == []
    stage_report = next(item for item in plan["stages"] if item["stage_name"] == "shared_initial_submit")
    command = stage_report["commands"][0]
    assert command["blocked"] is True
    assert command["needs_submission"] is False
    assert command["batch_report"]["blocked_reason"] == "pilot_validation_summary_missing"


def test_resume_helper_blocks_production_submissions_when_pilot_verification_failed(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    _write_pilot_summary(campaign_root, passed=False)
    shared_summary = campaign_root / "replica-001" / "shared_initial" / BATCH_SUMMARY_FILENAME
    _write_json(
        shared_summary,
        {
            "status": "selection_rendered",
            "candidate_count": 1,
            "expected_output_roots": [str(shared_summary.parent / "emb" / "shared-001")],
        },
    )
    _write_json(
        campaign_root / module.CONTROLLER_MANIFEST_FILENAME,
        {
            "execution_mode": "execute",
            "stages": [
                {
                    "name": "shared_initial_submit",
                    "command_type": "submission",
                    "commands": [{"command": "sbatch --parsable shared_001"}],
                    "expected_output_roots": [str(shared_summary)],
                }
            ],
        },
    )

    plan = module.build_resume_plan(campaign_root=campaign_root)

    assert plan["pilot_verification"]["passed"] is False
    assert plan["pilot_verification"]["status"] == "failed"
    assert plan["commands_to_submit"] == []
    command = plan["stages"][0]["commands"][0]
    assert command["blocked"] is True
    assert command["batch_report"]["blocked_reason"] == "pilot_validation_not_passed"


def test_stale_job_audit_falls_back_when_slurm_tools_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    report = module.audit_stale_jobs(
        campaign_root=tmp_path / "campaign",
        user="test-user",
        lookback_hours=48,
    )
    assert report["available"] is False
    assert report["status"] == "slurm_tools_unavailable"


def test_stale_job_audit_excludes_canceled_squeue_rows_from_active_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/slurm-tool")

    def fake_safe_slurm_run(command: list[str]) -> dict[str, object]:
        if command[0] == "squeue":
            return {
                "ok": True,
                "returncode": 0,
                "stdout": (
                    "101|CANCELLED|1:23|mesouq-emb-34um-dnn-causal-array|acn01\n"
                    "102|RUNNING|0:45|mesouq-emb-34um-dnn-causal-array|acn02\n"
                    "103|PENDING|2:30:00|mesouq-emb-34um-dnn-causal-array|Priority\n"
                ),
                "stderr": "",
                "command": command,
            }
        if command[0] == "sacct":
            return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "command": command}
        raise AssertionError(f"Unexpected Slurm command: {command}")

    monkeypatch.setattr(module, "_safe_slurm_run", fake_safe_slurm_run)

    report = module.audit_stale_jobs(
        campaign_root=tmp_path / "campaign",
        user="test-user",
        lookback_hours=48,
    )

    assert [item["job_id"] for item in report["active_jobs"]] == ["102", "103"]
    assert report["squeue_observed_job_count"] == 3
    assert report["non_active_squeue_job_count"] == 1
    assert [item["job_id"] for item in report["stale_pending_jobs"]] == ["103"]


def test_repair_source_index_advances_past_filtered_design_indices() -> None:
    repair = _load_repair_module()

    assert repair._replacement_source_index(
        stream_start=100,
        planned_count=3,
        prior_records=[
            {"source_index": 100},
            {"source_index": 102},
            {"source_index": 104},
        ],
        sequence=1,
    ) == 105
