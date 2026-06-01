from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
    EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME,
    build_emb_34um_dnn_causal_validation_ingestion_report,
    normalize_emb_34um_dnn_causal_validation_records,
    write_emb_34um_dnn_causal_validation_ingestion_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_f_delta(path: Path, *, ka: float, kb: float, force_grid: list[float], curve: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [0.0, ka, kb, 0.0, 0.0, 0.0, 0.0, 6.8] + curve + force_grid
    path.write_text(" ".join(str(item) for item in row) + "\n", encoding="utf-8")


def _candidate_manifest(
    *,
    campaign_root: Path,
    stage_root: Path,
    candidate_id: str,
    ka: float,
    kb: float,
    radp: float = 6.52,
    shell_th: float = 3.75e-9,
) -> Path:
    output_root = stage_root / "emb" / candidate_id
    manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
    parameters = {"ka": ka, "kb": kb, "radp": radp, "shell_th": shell_th}
    fingerprint = {"radp": radp, "shell_th": shell_th}
    _write_json(
        manifest_path,
        {
            "candidate_id": candidate_id,
            "output_root": str(output_root),
            "normalized_payload": {
                "candidate_id": candidate_id,
                "output_root": str(output_root),
                "parameters": parameters,
                "fingerprint": fingerprint,
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": candidate_id,
                    "output_root": str(output_root),
                    "parameters": parameters,
                    "fingerprint": fingerprint,
                    "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                }
            },
            "active_learning_metadata": {
                "campaign_root": str(campaign_root),
            },
        },
    )
    return manifest_path


def _stage_summary(stage_root: Path, manifests: list[Path]) -> Path:
    summary_path = stage_root / "emb_34um_batch_summary.json"
    _write_json(summary_path, {"rendered_candidate_manifests": [str(path) for path in manifests]})
    return summary_path


def _campaign_manifest(campaign_root: Path) -> Path:
    manifest_path = campaign_root / "emb_34um_dnn_causal_validation_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1",
            "campaign_root": str(campaign_root),
            "policy": {
                "ensemble_size": 10,
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            },
        },
    )
    return manifest_path


def test_dnn_ingestion_walks_campaign_and_collects_completed_rows(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)

    unseen_stage = campaign_root / "unseen_test"
    shared_stage = campaign_root / "replica-001" / "shared_initial"
    lhs_stage = campaign_root / "replica-001" / "lhs-step-01"
    al_stage = campaign_root / "replica-001" / "al-step-01"

    unseen_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=unseen_stage,
        candidate_id="unseen-001",
        ka=1.0e3,
        kb=2.0e3,
    )
    shared_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=shared_stage,
        candidate_id="shared-001",
        ka=2.0e3,
        kb=3.0e3,
    )
    lhs_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=lhs_stage,
        candidate_id="lhs-001",
        ka=3.0e3,
        kb=4.0e3,
    )
    al_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=al_stage,
        candidate_id="al-001",
        ka=4.0e3,
        kb=5.0e3,
    )

    _stage_summary(unseen_stage, [unseen_manifest])
    _stage_summary(shared_stage, [shared_manifest])
    _stage_summary(lhs_stage, [lhs_manifest])
    _stage_summary(al_stage, [al_manifest])

    unseen_output = unseen_manifest.parent
    _write_json(
        unseen_output / "emb_34um_result.json",
        {
            "candidate_id": "unseen-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 1) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(unseen_output / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})

    shared_output = shared_manifest.parent
    _write_f_delta(
        shared_output / "F_Delta.dat",
        ka=2.0e3,
        kb=3.0e3,
        force_grid=list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        curve=[float(index + 10) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
    )
    _write_json(shared_output / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 1})

    lhs_output = lhs_manifest.parent
    _write_json(
        lhs_output / "emb_34um_result.json",
        {
            "candidate_id": "lhs-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 20) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(
        lhs_output / "emb_34um_runtime_status.json",
        {"status": "completed", "retry_count": 2, "replacement": True, "replacement_for": "lhs-000"},
    )

    al_output = al_manifest.parent
    _write_json(
        al_output / "emb_34um_runtime_status.json",
        {"status": "failed", "retry_count": 1, "retry_limit": 3},
    )

    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
    )

    assert artifacts.manifest_path == campaign_root / "ingest" / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME
    assert artifacts.report_path == campaign_root / "ingest" / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME
    assert artifacts.summary_csv_path == campaign_root / "ingest" / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME
    assert artifacts.completed_rows_path == campaign_root / "ingest" / EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME
    assert artifacts.report["status_counts"] == {"completed": 3, "failed": 1, "missing": 0, "partial": 0}
    assert artifacts.report["passed"] is False
    assert any("branch=al" in blocker for blocker in artifacts.report["blockers"])

    completed_rows = artifacts.manifest["completed_rows"]
    assert len(completed_rows) == 3
    branches = {row["branch"] for row in completed_rows}
    assert branches == {"unseen_test", "shared_initial", "lhs"}
    lhs_row = next(row for row in completed_rows if row["branch"] == "lhs")
    assert lhs_row["replacement"] is True
    assert lhs_row["replacement_for"] == "lhs-000"
    assert lhs_row["cycle"] == 1
    assert lhs_row["candidate_space"] == "d4"
    assert lhs_row["parameters"]["radp"] == 6.52
    assert lhs_row["parameters"]["shell_th"] == 3.75e-9
    assert lhs_row["radp"] == 6.52
    assert lhs_row["shell_th"] == 3.75e-9
    assert len(lhs_row["force_grid"]) == len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)

    summary_payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION
    assert summary_payload["record_count"] == 4


def test_dnn_ingestion_includes_completed_replacement_batch_summary(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)

    stage_root = campaign_root / "replica-001" / "al-step-01"
    original_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=stage_root,
        candidate_id="al-001",
        ka=4.0e3,
        kb=5.0e3,
    )
    _stage_summary(stage_root, [original_manifest])
    _write_json(
        original_manifest.parent / "emb_34um_runtime_status.json",
        {"status": "failed", "retry_count": 1, "retry_limit": 3},
    )

    replacement_output_root = stage_root / "replacement" / "batch-001" / "emb" / "al-replacement-001"
    replacement_manifest_path = replacement_output_root / "dpd_sampling_candidate_manifest.json"
    _write_json(
        replacement_manifest_path,
        {
            "candidate_id": "al-replacement-001",
            "output_root": str(replacement_output_root),
            "normalized_payload": {
                "candidate_id": "al-replacement-001",
                "output_root": str(replacement_output_root),
                "parameters": {"ka": 6.0e3, "kb": 7.0e3, "radp": 6.61, "shell_th": 3.8e-9},
                "fingerprint": {"radp": 6.61, "shell_th": 3.8e-9},
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": "al-replacement-001",
                    "output_root": str(replacement_output_root),
                    "parameters": {"ka": 6.0e3, "kb": 7.0e3, "radp": 6.61, "shell_th": 3.8e-9},
                    "fingerprint": {"radp": 6.61, "shell_th": 3.8e-9},
                    "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                }
            },
            "active_learning_metadata": {
                "campaign_root": str(campaign_root),
                "replacement": True,
                "replacement_for": "al-001",
            },
        },
    )
    _write_json(
        replacement_output_root / "emb_34um_result.json",
        {
            "candidate_id": "al-replacement-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 30) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(
        replacement_output_root / "emb_34um_runtime_status.json",
        {
            "status": "completed",
            "retry_count": 0,
            "replacement": True,
            "replacement_for": "al-001",
        },
    )

    _write_json(
        stage_root
        / "replacement"
        / "batch-001"
        / EMB_34UM_DNN_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME,
        {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_replacement_batch_summary.v1",
            "status": "replacement_rendered",
            "campaign_root": str(campaign_root),
            "replica": 1,
            "cycle": 1,
            "stage": "al-step-01",
            "replacement_batch_index": 1,
            "replacement_records": [
                {
                    "failed_candidate_index": 0,
                    "failed_candidate_id": "al-001",
                    "failed_candidate_manifest_path": str(original_manifest),
                    "failed_output_root": str(original_manifest.parent),
                    "replacement_candidate_id": "al-replacement-001",
                    "replacement_candidate_manifest_path": str(replacement_manifest_path),
                    "replacement_output_root": str(replacement_output_root),
                    "replacement_batch_index": 1,
                    "replacement_sequence": 1,
                    "replacement_policy": "candidate_reserve",
                }
            ],
            "rendered_candidate_manifests": [str(replacement_manifest_path)],
            "expected_output_roots": [str(replacement_output_root)],
        },
    )

    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
    )

    replacement_rows = [
        row for row in artifacts.manifest["completed_rows"] if row["candidate_id"] == "al-replacement-001"
    ]
    assert len(replacement_rows) == 1
    assert replacement_rows[0]["replacement"] is True
    assert replacement_rows[0]["replacement_for"] == "al-001"
    assert replacement_rows[0]["stage"] == "al-step-01"
    assert replacement_rows[0]["replicate"] == 1
    assert replacement_rows[0]["cycle"] == 1
    assert artifacts.report["status_counts"]["completed"] == 1
    assert artifacts.manifest["record_count"] == 2


def test_dnn_ingestion_quarantines_timeout_with_zero_retry_limit(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)
    stage_root = campaign_root / "replica-001" / "lhs-step-01"
    candidate_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=stage_root,
        candidate_id="lhs-timeout-001",
        ka=4.0e3,
        kb=5.0e3,
    )
    _stage_summary(stage_root, [candidate_manifest])
    _write_json(
        candidate_manifest.parent / "emb_34um_runtime_status.json",
        {"status": "timeout", "retry_count": 0, "retry_limit": 0},
    )

    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
    )

    record = artifacts.manifest["records"][0]
    assert record["status"] == "failed"
    assert record["runtime_status"] == "failed"
    assert record["quarantined"] is True
    assert record["retry_count"] == 0
    assert record["retry_limit"] == 0


def test_dnn_ingestion_treats_textual_none_replacement_for_as_empty(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)

    stage_root = campaign_root / "replica-001" / "al-step-01"
    candidate_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=stage_root,
        candidate_id="al-001",
        ka=4.0e3,
        kb=5.0e3,
    )
    manifest_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    manifest_payload["active_learning_metadata"]["replacement_for"] = "None"
    _write_json(candidate_manifest, manifest_payload)
    _stage_summary(stage_root, [candidate_manifest])

    output_root = candidate_manifest.parent
    _write_json(
        output_root / "emb_34um_result.json",
        {
            "candidate_id": "al-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 30) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(
        output_root / "emb_34um_runtime_status.json",
        {"status": "completed", "retry_count": 0, "replacement_for": "None"},
    )

    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
    )

    completed_rows = artifacts.manifest["completed_rows"]
    assert len(completed_rows) == 1
    assert completed_rows[0]["replacement"] is False
    assert completed_rows[0]["replacement_for"] == ""


def test_dnn_ingestion_rejects_missing_d4_geometry(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)
    stage_root = campaign_root / "unseen_test"
    candidate_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=stage_root,
        candidate_id="unseen-001",
        ka=1.0e3,
        kb=2.0e3,
    )
    manifest_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    manifest_payload["normalized_payload"]["parameters"] = {"ka": 1.0e3, "kb": 2.0e3}
    manifest_payload["normalized_payload"]["fingerprint"] = {}
    manifest_payload["rendered_payload"]["request_payload"]["parameters"] = {"ka": 1.0e3, "kb": 2.0e3}
    manifest_payload["rendered_payload"]["request_payload"]["fingerprint"] = {}
    _write_json(candidate_manifest, manifest_payload)
    _stage_summary(stage_root, [candidate_manifest])

    _write_json(
        candidate_manifest.parent / "emb_34um_result.json",
        {
            "candidate_id": "unseen-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 1) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(candidate_manifest.parent / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})

    with pytest.raises(ValueError, match="missing D4 runtime geometry"):
        write_emb_34um_dnn_causal_validation_ingestion_artifacts(
            campaign_manifest_path=manifest_path,
            output_root=campaign_root / "ingest",
        )


def test_normalize_and_build_report_accept_campaign_root(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)
    stage_root = campaign_root / "unseen_test"
    candidate_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=stage_root,
        candidate_id="unseen-001",
        ka=1.0e3,
        kb=2.0e3,
    )
    _stage_summary(stage_root, [candidate_manifest])
    _write_json(
        candidate_manifest.parent / "emb_34um_result.json",
        {
            "candidate_id": "unseen-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 1) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(candidate_manifest.parent / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})

    rows = normalize_emb_34um_dnn_causal_validation_records(campaign_root)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "unseen-001"
    assert rows[0]["status"] == "completed"

    report = build_emb_34um_dnn_causal_validation_ingestion_report(manifest_path)
    assert report["status"] == "passed"
    assert report["completed_curve_count"] == 1


def test_ingestion_excludes_pilot_rows_from_completed_rows(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = _campaign_manifest(campaign_root)

    prod_stage = campaign_root / "replica-001" / "al-step-01"
    prod_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=prod_stage,
        candidate_id="al-prod-001",
        ka=4.1e3,
        kb=5.2e3,
    )
    _stage_summary(prod_stage, [prod_manifest])
    _write_json(
        prod_manifest.parent / "emb_34um_result.json",
        {
            "candidate_id": "al-prod-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 30) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(prod_manifest.parent / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})

    pilot_stage = campaign_root / "pilot"
    pilot_manifest = _candidate_manifest(
        campaign_root=campaign_root,
        stage_root=pilot_stage,
        candidate_id="pilot-001",
        ka=2.2e3,
        kb=3.3e3,
    )
    _stage_summary(pilot_stage, [pilot_manifest])
    _write_json(
        pilot_manifest.parent / "emb_34um_result.json",
        {
            "candidate_id": "pilot-001",
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "vertical_diameter": [float(index + 40) for index in range(len(EMB_34UM_DNN_CAUSAL_FORCE_GRID))],
        },
    )
    _write_json(pilot_manifest.parent / "emb_34um_runtime_status.json", {"status": "completed", "retry_count": 0})

    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
    )

    completed_rows = artifacts.manifest["completed_rows"]
    assert len(completed_rows) == 1
    assert completed_rows[0]["candidate_id"] == "al-prod-001"
    assert all(row["candidate_id"] != "pilot-001" for row in completed_rows)
    stream_counts = {entry["stream"]: entry for entry in artifacts.report["stream_counts"]}
    assert stream_counts["pilot"]["completed"] == 1
