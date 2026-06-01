from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    is_dnn_causal_low_corner_excluded,
)


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
            "mode": "pilot",
            "replica": 0,
            "cycle": 0,
            "candidate_count": 32,
            "execution_mode": "render-only",
            "array": "0-31%30",
            "batch_summary_path": str(campaign_root / "pilot" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable --array=0-31%30 --export=EXECUTION_MODE=render-only,...",
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
            "walltime": "00:12:00",
            "concurrent_jobs": 30,
            "retry_limit": 0,
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
    assert manifest["pilot_validation"]["required"] is True
    assert manifest["pilot_validation"]["stage_name"] == "pilot_verify"
    assert manifest["command_inventory"]["count"] == 12
    assert manifest["replicas"] == [1, 2]
    assert manifest["cycle_count"] == 2
    assert manifest["stage_order"] == [
        "prepare_design",
        "pilot_submit",
        "pilot_verify",
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
    assert len(manifest["parallel_arrays"]) == 12
    assert all(entry["execution_mode"] in {"render-only", "execute", "dry-run"} for entry in manifest["parallel_arrays"])
    assert {entry["array"] for entry in manifest["parallel_arrays"]} == {"0-99%30", "0-31%30"}
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert stages["pilot_submit"]["dependencies"] == ["prepare_design"]
    assert stages["pilot_verify"]["dependencies"] == ["pilot_submit"]
    assert stages["pilot_verify"]["expected_output_roots"] == [
        str(campaign_root / "pilot" / "validation" / "emb_34um_dnn_causal_validation_pilot_summary.json")
    ]
    verify_command = stages["pilot_verify"]["commands"][0]["command"]
    assert "analyze_emb_34um_dnn_causal_validation_pilot.py" in verify_command
    assert "--campaign-root" in verify_command
    assert "--output-root" in verify_command
    assert stages["unseen_test_submit"]["dependencies"] == ["pilot_verify"]
    assert stages["shared_initial_submit"]["dependencies"] == ["pilot_verify"]
    assert "pilot_submit" not in stages["final_ingest"]["dependencies"]

    assert stages["al-step-01-train-score-select"]["dependencies"] == ["shared_initial_submit"]
    assert stages["al-step-02-train-score-select"]["dependencies"] == ["al-step-01-submit"]
    assert stages["al-step-02-render-selected"]["dependencies"] == ["al-step-02-train-score-select"]
    assert stages["final_analyze"]["dependencies"] == ["final_ingest"]

    train_commands = [item["command"] for item in stages["al-step-01-train-score-select"]["commands"]]
    assert any("--stage-action al-build-inputs" in command for command in train_commands)
    assert any("run_emb_34um_dnn_train_score_select.py" in command for command in train_commands)
    assert any("--candidate-space d4" in command for command in train_commands)
    assert any("--acquisition-score-mode curve_error" in command for command in train_commands)
    assert any("--top-n 150" in command for command in train_commands)
    execute_commands = [command for command in train_commands if "sbatch --parsable" in command]
    if execute_commands:
        assert any("CANDIDATE_SPACE=" in command for command in execute_commands)
        assert any("ACQUISITION_SCORE_MODE=curve_error" in command for command in execute_commands)
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


def test_controller_candidate_pool_avoids_low_ka_low_kb_timeout_corner() -> None:
    module = _load_controller_module()

    pool, metadata = module._generate_candidate_pool(
        replica=1,
        step=1,
        pool_size=500,
        existing_points=[],
    )

    assert len(pool) == 500
    assert metadata["actual_size"] == 500
    assert metadata["candidate_pool_generator"] == "auto"
    assert metadata["acquisition_score_mode"] == "curve_error"
    assert not any(
        is_dnn_causal_low_corner_excluded(row["ka"], row["kb"], row["radp"], row["shell_th"])
        for row in pool
    )
    assert all("exclusion_policy" in row for row in pool)
    for row in pool:
        assert {"ka", "kb", "radp", "shell_th", "runtime_fingerprint"}.issubset(row)
        assert row["runtime_fingerprint"]["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE
        assert EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0] <= row["radp"] <= EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][1]
        assert EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0] <= row["shell_th"] <= EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][1]
        assert row["b1"] == EMB_34UM_DNN_CAUSAL_B1_VALUE
        assert row["b2"] == EMB_34UM_DNN_CAUSAL_B2_VALUE
        assert row["a3"] == EMB_34UM_DNN_CAUSAL_A3_VALUE
        assert row["a4"] == EMB_34UM_DNN_CAUSAL_A4_VALUE


def test_controller_candidate_pool_keeps_d4_rows_with_shared_ka_kb_but_distinct_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controller_module()

    ka_target = 6000.0
    kb_target = 6000.0
    radp_existing_target = 6.5
    shell_existing_target = 3.8e-9
    radp_new_target = 6.6
    shell_new_target = 3.9e-9

    def _fake_build_d4_candidate_pool(*, requested_size, seed, generator, existing_points):
        assert requested_size == 1
        assert seed == 43001
        assert generator == "auto"
        assert existing_points == [(ka_target, kb_target, radp_existing_target, shell_existing_target)]
        return (
            (
                {
                    "candidate_id": "source-pool-row",
                    "ka": ka_target,
                    "kb": kb_target,
                    "radp": radp_new_target,
                    "shell_th": shell_new_target,
                    "runtime_fingerprint": {"bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE},
                },
            ),
            {
                "generator_type": "stratified",
                "requested_size": 1,
                "actual_size": 1,
                "seed": seed,
                "bounds": {},
                "dimension_names": ["ka", "kb", "radp", "shell_th"],
            },
        )

    monkeypatch.setattr(module, "build_d4_candidate_pool", _fake_build_d4_candidate_pool)

    pool, metadata = module._generate_candidate_pool(
        replica=1,
        step=1,
        pool_size=1,
        existing_points=[(ka_target, kb_target, radp_existing_target, shell_existing_target)],
    )

    assert len(pool) == 1
    assert metadata["acquisition_score_mode"] == "curve_error"
    assert pool[0]["candidate_id"] == "rep001-step01-pool-000001"
    assert math.isclose(pool[0]["ka"], ka_target)
    assert math.isclose(pool[0]["kb"], kb_target)
    assert math.isclose(pool[0]["radp"], radp_new_target)
    assert math.isclose(pool[0]["shell_th"], shell_new_target)
    assert not math.isclose(pool[0]["radp"], radp_existing_target)
    assert not math.isclose(pool[0]["shell_th"], shell_existing_target)


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
    pilot_submit = stages["pilot_submit"]["commands"]
    submit_stage = stages["shared_initial_submit"]["commands"]
    assert pilot_submit
    assert stages["pilot_submit"]["expected_output_roots"][0].endswith("/pilot/emb_34um_batch_summary.json")
    assert submit_stage
    assert all("EXECUTION_MODE=execute" in item["command"] for item in submit_stage)
    assert not any("EXECUTION_MODE=render-only" in item["command"] for item in submit_stage)

    train_commands = [item["command"] for item in stages["al-step-01-train-score-select"]["commands"]]
    assert any("emb_34um_dnn_train_score_select.sbatch" in command for command in train_commands)
    assert any("sbatch --parsable" in command for command in train_commands)
    assert any("CANDIDATE_SPACE=d4" in command for command in train_commands)
    assert any("ACQUISITION_SCORE_MODE=curve_error" in command for command in train_commands)
    assert any("TOP_N=150" in command for command in train_commands)
    assert not any("--dry-run" in command for command in train_commands)

    final_ingest_command = stages["final_ingest"]["commands"][0]["command"]
    assert "emb_34um_dnn_causal_validation_metrics.sbatch" in final_ingest_command
    assert "RUN_ANALYZE=1" in final_ingest_command
    assert "ALLOW_BLOCKED=1" in final_ingest_command
    pilot_commands = module._dispatch_stage_commands(
        stage_action="pilot_submit",
        campaign_root=campaign_root,
        execution_mode="render-only",
    )
    assert pilot_commands
    assert all("EXECUTION_MODE=render-only" in command for command in pilot_commands)

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


def test_controller_honors_active_replicate_count_from_policy(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    entries: list[dict[str, object]] = [
        {
            "mode": "unseen_test",
            "replica": 0,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "unseen_test" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable unseen",
        }
    ]
    for replica in range(1, 6):
        entries.extend(
            [
                {
                    "mode": "shared_initial",
                    "replica": replica,
                    "cycle": 0,
                    "candidate_count": 100,
                    "execution_mode": "render-only",
                    "array": "0-99%30",
                    "batch_summary_path": str(campaign_root / f"replica-{replica:03d}" / "shared_initial" / "emb_34um_batch_summary.json"),
                    "command": f"sbatch --parsable shared_{replica:03d}",
                },
                {
                    "mode": "lhs-step-01",
                    "replica": replica,
                    "cycle": 1,
                    "candidate_count": 100,
                    "execution_mode": "render-only",
                    "array": "0-99%30",
                    "batch_summary_path": str(campaign_root / f"replica-{replica:03d}" / "lhs-step-01" / "emb_34um_batch_summary.json"),
                    "command": f"sbatch --parsable lhs_{replica:03d}",
                },
                {
                    "mode": "al-step-01",
                    "replica": replica,
                    "cycle": 1,
                    "candidate_count": 100,
                    "execution_mode": "render-only",
                    "array": "0-99%30",
                    "batch_summary_path": str(campaign_root / f"replica-{replica:03d}" / "al-step-01" / "emb_34um_batch_summary.json"),
                    "command": f"sbatch --parsable al_{replica:03d}",
                },
            ]
        )
    manifest_path = campaign_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "run_id_prefix": "emb-34um-dnn-causal-test",
                "policy": {
                    "active_replicate_count": 3,
                    "replicate_count": 3,
                    "replicates": [1, 2, 3, 4, 5],
                },
                "command_inventory": {
                    "count": len(entries),
                    "walltime": "00:25:00",
                    "concurrent_jobs": 30,
                    "retry_limit": 0,
                    "entries": entries,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
    )
    built = result["manifest"]
    stages = {stage["name"]: stage for stage in built["stages"]}
    assert built["replicas"] == [1, 2, 3]
    assert built["active_replicate_count"] == 3
    assert built["command_inventory"]["count"] == 10
    assert len(stages["al-step-01-render-selected"]["commands"]) == 3
    assert len(stages["al-step-01-submit"]["commands"]) == 3


def test_controller_selected_render_uses_manifest_scheduler_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    stage_root = campaign_root / "replica-001" / "al-step-01"
    train_root = stage_root / "train_score_select"
    selection_inputs = stage_root / "selection_inputs"
    train_root.mkdir(parents=True, exist_ok=True)
    selection_inputs.mkdir(parents=True, exist_ok=True)
    (train_root / "emb_34um_dnn_train_score_select_manifest.json").write_text(
        json.dumps(
            {
                "fit": {"force_grid": [5000.0 * index / 7.0 for index in range(8)]},
                "selected_points": [
                    {"candidate_id": "pool-0001", "ka": 6000.0, "kb": 6000.0},
                    {"candidate_id": "pool-0002", "ka": 10000.0, "kb": 5000.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (selection_inputs / "candidate_pool.json").write_text(
        json.dumps(
            [
                {
                    "candidate_id": "pool-0001",
                    "ka": 6000.0,
                    "kb": 6000.0,
                    "radp": 6.5,
                    "shell_th": 3.8e-9,
                    "runtime_fingerprint": {"radp": 6.5, "shell_th": 3.8e-9, "bpress": -91.0},
                },
                {
                    "candidate_id": "pool-0002",
                    "ka": 4000.0,
                    "kb": 5000.0,
                    "radp": 6.6,
                    "shell_th": 4.0e-9,
                    "runtime_fingerprint": {"radp": 6.6, "shell_th": 4.0e-9, "bpress": -91.0},
                }
            ],
        ),
        encoding="utf-8",
    )
    (campaign_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "run_id_prefix": "emb-34um-dnn-causal-test",
                "command_inventory": {
                    "walltime": "00:20:00",
                    "concurrent_jobs": 11,
                    "retry_limit": 2,
                    "entries": [
                        {
                            "mode": "al-step-01",
                            "replica": 1,
                            "cycle": 1,
                            "candidate_count": 1,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class _Prepare:
        @staticmethod
        def _build_candidates_for_batch(candidates, *, force_grid):
            captured["candidate_records"] = tuple(candidates)
            return tuple(candidates)

        @staticmethod
        def _render_batch(*, candidates, batch_root, run_id, batch_id, walltime, concurrent_jobs, retry_limit):
            captured["walltime"] = walltime
            captured["concurrent_jobs"] = concurrent_jobs
            captured["retry_limit"] = retry_limit
            return {
                "candidate_count": len(candidates),
                "rendered_candidate_manifests": [],
                "expected_output_roots": [],
            }

    monkeypatch.setattr(module, "_load_prepare_module", lambda: _Prepare())
    payload = module._render_selected_candidates_for_stage(
        campaign_root=campaign_root,
        replica=1,
        step=1,
        execution_mode="execute",
    )

    assert payload["selection_manifest_path"]
    assert captured["walltime"] == "00:20:00"
    assert captured["concurrent_jobs"] == 11
    assert captured["retry_limit"] == 2
    assert "candidate_records" in captured
    candidates = tuple(captured["candidate_records"])
    assert len(candidates) == 1
    assert candidates[0]["ka"] == 6000.0
    assert candidates[0]["kb"] == 6000.0
    assert candidates[0]["radp"] == 6.5
    assert candidates[0]["shell_th"] == 3.8e-9
    assert candidates[0]["runtime_fingerprint"] == {
        "radp": 6.5,
        "shell_th": 3.8e-9,
        "bpress": -91.0,
    }
    selection_manifest = json.loads((stage_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert selection_manifest["candidate_reserve"][0]["radp"] == 6.6
    assert selection_manifest["candidate_reserve"][0]["shell_th"] == 4.0e-9
    assert selection_manifest["candidate_reserve"][0]["runtime_fingerprint"]["radp"] == 6.6


def test_controller_prepare_command_forwards_escalation_replicate_counts(tmp_path: Path) -> None:
    module = _load_controller_module()
    campaign_root = tmp_path / "campaign"
    entries: list[dict[str, object]] = [
        {
            "mode": "unseen_test",
            "replica": 0,
            "cycle": 0,
            "candidate_count": 100,
            "execution_mode": "render-only",
            "array": "0-99%30",
            "batch_summary_path": str(campaign_root / "unseen_test" / "emb_34um_batch_summary.json"),
            "command": "sbatch --parsable unseen",
        }
    ]
    for replica in range(1, EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT + 1):
        entries.append(
            {
                "mode": "shared_initial",
                "replica": replica,
                "cycle": 0,
                "candidate_count": 100,
                "execution_mode": "render-only",
                "array": "0-99%30",
                "batch_summary_path": str(campaign_root / f"replica-{replica:03d}" / "shared_initial" / "emb_34um_batch_summary.json"),
                "command": f"sbatch --parsable shared_{replica:03d}",
            }
        )
    manifest_path = campaign_root / module.EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "timestamp": "20260522_130000",
                "run_id_prefix": "emb-34um-dnn-causal-test",
                "policy": {
                    "active_replicate_count": EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
                    "max_replicate_count": EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
                    "replicates": list(range(1, EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT + 1)),
                },
                "command_inventory": {
                    "count": len(entries),
                    "walltime": "00:12:00",
                    "concurrent_jobs": 30,
                    "retry_limit": 0,
                    "entries": entries,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = module.build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=manifest_path,
    )
    stages = {stage["name"]: stage for stage in result["manifest"]["stages"]}
    prepare_command = stages["prepare_design"]["commands"][0]["command"]

    assert "--active-replicate-count 5" in prepare_command
    assert "--max-replicate-count 5" in prepare_command
