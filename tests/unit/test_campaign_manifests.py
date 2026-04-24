import json
from pathlib import Path

from meso_uq.campaign_manifests import (
    MANDATORY_MAIN_FIGURES,
    MANDATORY_SUPPLEMENTARY_FIGURES,
    MANDATORY_TABLES,
    LANE_STAGE_ORDER,
    build_paper_release_manifest,
    build_required_asset_entries,
    build_job_manifest,
    build_lane_manifest,
    build_partition_policy_metadata,
    build_phase2_backend_policy,
    derive_asset_source_map,
    write_manifest,
)


def test_build_job_manifest_contains_contract_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("phase2_backend: native-cuda\n", encoding="utf-8")
    stdout_log = tmp_path / "phase2.stdout.log"
    stderr_log = tmp_path / "phase2.stderr.log"
    stdout_log.write_text("ok\n", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")

    payload = build_job_manifest(
        job_id="lane_phase2_01",
        slurm_job_id="12345",
        stage="phase2",
        lane="compression:full-model:production",
        status="passed",
        command=["python", "run_inference_stage.py", "--stage", "phase2"],
        cwd=Path.cwd(),
        start_utc="2026-04-23T12:00:00+00:00",
        end_utc="2026-04-23T12:05:00+00:00",
        duration_seconds=300.0,
        partition="gpu",
        time_limit="00:45:00",
        node_list="gpu-node-01",
        gpu_type="a100",
        gpu_count=1,
        cpu_count=8,
        mem_mb=64000,
        config_path=config_path,
        logs={"stdout": str(stdout_log), "stderr": str(stderr_log)},
        backend_metadata={
            "phase2_backend": "native-cuda",
            "phase2_backend_policy": build_phase2_backend_policy(
                stage="phase2", profile="production", phase2_backend="native-cuda"
            ),
            "partition_policy": build_partition_policy_metadata(
                is_gpu_job=True, partition="gpu", time_limit="00:45:00"
            ),
        },
        git_metadata={"git_commit": "deadbeef", "git_branch": "main", "dirty_worktree": False},
        input_files=[config_path],
        output_files=[stdout_log, stderr_log],
    )

    required_keys = {
        "job_id",
        "slurm_job_id",
        "stage",
        "lane",
        "status",
        "git_commit",
        "git_branch",
        "dirty_worktree",
        "command",
        "cwd",
        "start_utc",
        "end_utc",
        "duration_seconds",
        "partition",
        "time_limit",
        "node_list",
        "gpu_type",
        "gpu_count",
        "cpu_count",
        "mem_mb",
        "config_path",
        "config_sha256",
        "input_files",
        "output_files",
        "logs",
        "backend_metadata",
    }
    assert required_keys.issubset(set(payload.keys()))
    assert isinstance(payload["config_sha256"], str)
    assert len(payload["config_sha256"]) == 64
    assert payload["input_files"][0]["sha256"] is not None
    assert payload["output_files"][0]["sha256"] is not None


def test_policy_flags_for_phase2_backend_and_partition():
    backend_ok = build_phase2_backend_policy(
        stage="phase2", profile="production", phase2_backend="native-cuda"
    )
    backend_bad = build_phase2_backend_policy(
        stage="phase2", profile="production", phase2_backend="cpu-mpi"
    )
    partition_ok = build_partition_policy_metadata(
        is_gpu_job=True, partition="dev", time_limit="00:20:00"
    )
    partition_bad = build_partition_policy_metadata(
        is_gpu_job=True, partition="gpu", time_limit="00:20:00"
    )

    assert backend_ok["is_compliant"] is True
    assert backend_bad["is_compliant"] is False
    assert partition_ok["expected_partition"] == "dev"
    assert partition_ok["is_compliant"] is True
    assert partition_bad["is_compliant"] is False


def test_build_lane_manifest_rolls_up_stage_order_and_policy():
    steps = [
        {"name": "phase1", "returncode": 0},
        {"name": "phase2", "returncode": 0},
        {"name": "phase3b", "returncode": 0},
    ]
    violating_job = {
        "job_id": "job_phase2",
        "backend_metadata": {
            "phase2_backend_policy": {"is_compliant": False, "violation_reason": "bad backend"},
            "partition_policy": {"is_compliant": True},
        },
    }

    lane_manifest = build_lane_manifest(
        lane="compression:full-model:production",
        selection={
            "experiment": "compression",
            "model_family": "full-model",
            "profile": "production",
        },
        steps=steps,
        artifacts={"phase3b_map_manifest": "/tmp/map.json"},
        job_manifest_paths=["/tmp/jobs/job_phase1.json", "/tmp/jobs/job_phase2.json"],
        job_manifests=[violating_job],
    )

    status_by_stage = {entry["stage"]: entry["status"] for entry in lane_manifest["stage_statuses"]}
    assert tuple(status_by_stage.keys()) == LANE_STAGE_ORDER
    assert status_by_stage["phase1"] == "passed"
    assert status_by_stage["propagation_phase3b"] == "skipped"
    assert lane_manifest["policy"]["status"] == "fail"
    assert lane_manifest["policy"]["violations"]

    json.dumps(lane_manifest)


def test_paper_release_manifest_pass_with_complete_assets_and_clean_lane(tmp_path):
    lane_manifest = {
        "lane": "compression:full-model:production",
        "stage_statuses": [{"stage": stage, "status": "passed"} for stage in LANE_STAGE_ORDER],
        "artifacts": {"phase3b_map_manifest": str(tmp_path / "map.json")},
        "job_manifests": [str(tmp_path / "jobs" / "job1.json")],
        "policy": {"status": "pass", "violations": []},
    }
    (tmp_path / "map.json").write_text("{}", encoding="utf-8")
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "jobs" / "job1.json").write_text("{}", encoding="utf-8")
    lane_manifest_path = tmp_path / "lane.json"
    write_manifest(lane_manifest_path, lane_manifest)

    figures_main_root = tmp_path / "figures" / "main"
    figures_supp_root = tmp_path / "figures" / "supplementary"
    tables_root = tmp_path / "tables"
    figures_main_root.mkdir(parents=True, exist_ok=True)
    figures_supp_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    for name in MANDATORY_MAIN_FIGURES:
        (figures_main_root / name).write_text("main", encoding="utf-8")
    for name in MANDATORY_SUPPLEMENTARY_FIGURES:
        (figures_supp_root / name).write_text("supp", encoding="utf-8")
    for name in MANDATORY_TABLES:
        (tables_root / name).write_text("table", encoding="utf-8")

    source_map = {
        "figures/main/experimental_reference_curves.pdf": {
            "source_lane": "compression:full-model:production",
            "source_artifacts": ["runs/compression/full-model/production/phase3b"],
        }
    }
    main_entries = build_required_asset_entries(
        category="figures/main",
        root=figures_main_root,
        required_files=MANDATORY_MAIN_FIGURES,
        source_map=source_map,
    )
    supp_entries = build_required_asset_entries(
        category="figures/supplementary",
        root=figures_supp_root,
        required_files=MANDATORY_SUPPLEMENTARY_FIGURES,
        source_map=source_map,
    )
    table_entries = build_required_asset_entries(
        category="tables",
        root=tables_root,
        required_files=MANDATORY_TABLES,
        source_map=source_map,
    )

    manifest = build_paper_release_manifest(
        run_campaign_id="campaign-001",
        generated_at_utc="2026-04-23T10:00:00+00:00",
        lane_manifest_paths=[lane_manifest_path],
        figures_main_entries=main_entries,
        figures_supplementary_entries=supp_entries,
        table_entries=table_entries,
    )
    assert manifest["release_status"] == "PASS"
    assert manifest["hard_failures"] == []
    mapped = next(
        item
        for item in manifest["assets"]["figures_main"]
        if item["asset_id"] == "experimental_reference_curves.pdf"
    )
    assert mapped["source_lane"] == "compression:full-model:production"
    assert mapped["sha256"] is not None


def test_paper_release_manifest_failure_propagates_policy_and_missing_assets(tmp_path):
    lane_manifest = {
        "lane": "compression:full-model:production",
        "stage_statuses": [{"stage": "phase1", "status": "passed"}],
        "artifacts": {"phase3b_map_manifest": str(tmp_path / "missing-map.json")},
        "job_manifests": [str(tmp_path / "missing-job.json")],
        "policy": {"status": "fail", "violations": ["partition policy violation"]},
    }
    lane_manifest_path = tmp_path / "lane.json"
    write_manifest(lane_manifest_path, lane_manifest)

    main_entries = build_required_asset_entries(
        category="figures/main",
        root=tmp_path / "figures" / "main",
        required_files=MANDATORY_MAIN_FIGURES,
    )
    supp_entries = build_required_asset_entries(
        category="figures/supplementary",
        root=tmp_path / "figures" / "supplementary",
        required_files=MANDATORY_SUPPLEMENTARY_FIGURES,
    )
    table_entries = build_required_asset_entries(
        category="tables",
        root=tmp_path / "tables",
        required_files=MANDATORY_TABLES,
    )

    manifest = build_paper_release_manifest(
        run_campaign_id="campaign-002",
        generated_at_utc="2026-04-23T10:05:00+00:00",
        lane_manifest_paths=[lane_manifest_path],
        figures_main_entries=main_entries,
        figures_supplementary_entries=supp_entries,
        table_entries=table_entries,
    )
    assert manifest["release_status"] == "FAIL"
    assert any("policy violation" in failure for failure in manifest["hard_failures"])
    assert any("missing asset:" in failure for failure in manifest["hard_failures"])


def test_derive_asset_source_map_from_lane_metadata():
    lane_manifests = [
        {
            "lane": "compression:full-model:production",
            "selection": {"experiment": "compression", "model_family": "full-model", "profile": "production"},
            "artifacts": {
                "phase3b_map_manifest": "/runs/compression/full-model/production/map_phase3b/phase3b_map_manifest.json"
            },
        },
        {
            "lane": "indentation:reduced-model:production",
            "selection": {"experiment": "indentation", "model_family": "reduced-model", "profile": "production"},
            "artifacts": {
                "phase3b_map_manifest": "/runs/indentation/reduced-model/production/map_phase3b/phase3b_map_manifest.json"
            },
        },
    ]
    mapping = derive_asset_source_map(
        required_relative_paths=[
            "figures/main/map_confirmation_compression_2.1um.pdf",
            "figures/main/map_confirmation_reduced_representative.pdf",
            "tables/map_parameters_reduced_phase3b.csv",
        ],
        lane_manifests=lane_manifests,
    )
    assert mapping["figures/main/map_confirmation_compression_2.1um.pdf"]["source_lane"] == (
        "compression:full-model:production"
    )
    assert mapping["tables/map_parameters_reduced_phase3b.csv"]["source_lane"] == (
        "indentation:reduced-model:production"
    )
    derive_asset_source_map,
