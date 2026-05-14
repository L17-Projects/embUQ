from __future__ import annotations

from meso_uq.core import ArtifactClass, Platform
from meso_uq.orchestration import (
    LineageArtifact,
    LineageValidationEvidence,
    RunLineageRecord,
    RunStageRecord,
    RunStageStatus,
    config_digest,
    stable_run_id,
)
from meso_uq.orchestration.lineage import new_run_lineage


def test_run_lineage_serializes_resume_and_failure_state() -> None:
    config = {"agent_family": "emb", "modality": "compression", "seed": 7}
    lineage = new_run_lineage(
        prefix="active-learning",
        config=config,
        platform=Platform.KAROLINA,
        code_version="abc123",
    )

    failed = RunStageRecord(
        stage_id="simulate",
        status=RunStageStatus.FAILED,
        command=("python", "simulate.py"),
        error="simulator rejected candidate",
    )
    reused = RunStageRecord(
        stage_id="load-surrogate",
        status=RunStageStatus.REUSED,
        reused_from="surrogate-run-1",
    )
    lineage = lineage.with_stage(failed).with_stage(reused)

    payload = lineage.as_dict()
    restored = RunLineageRecord.from_dict(payload)

    assert restored.run_id == stable_run_id("active-learning", config)
    assert restored.config_digest == config_digest(config)
    assert restored.failed_stages()[0].stage_id == "simulate"
    assert restored.resumable_stage_ids() == ("simulate",)


def test_lineage_records_artifacts_and_gpu_validation_evidence() -> None:
    lineage = RunLineageRecord(
        run_id="validation-run",
        config_digest="deadbeef",
        platform=Platform.KAROLINA,
        code_version="278cf66",
    )
    artifact = LineageArtifact(
        artifact_id="posterior",
        artifact_class=ArtifactClass.POSTERIOR,
        path="artifacts/posteriors/summary.json",
        role="output",
    )
    evidence = LineageValidationEvidence(
        validation_id="gpu-matrix",
        matrix_report_path="/scratch/project/eu-26-17/eubrieucb/mesouq/runs/report.json",
        status="passed",
        platform=Platform.KAROLINA,
        slurm_job_id="4303824",
        hard_failures=(),
    )

    lineage = lineage.with_artifact(artifact).with_validation(evidence)
    payload = lineage.as_dict()

    assert payload["artifacts"][0]["artifact_class"] == "posterior"
    assert payload["validations"][0]["status"] == "passed"
    assert payload["validations"][0]["metadata"]["path_policy"] == "external"


def test_stable_run_id_is_deterministic_for_sorted_config_payload() -> None:
    left = stable_run_id("run", {"b": 2, "a": 1})
    right = stable_run_id("run", {"a": 1, "b": 2})

    assert left == right
    assert left.startswith("run-")
