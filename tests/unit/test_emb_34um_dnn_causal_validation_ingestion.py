from __future__ import annotations

import json
import sys
from pathlib import Path

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
) -> Path:
    output_root = stage_root / "emb" / candidate_id
    manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
    _write_json(
        manifest_path,
        {
            "candidate_id": candidate_id,
            "output_root": str(output_root),
            "normalized_payload": {
                "candidate_id": candidate_id,
                "output_root": str(output_root),
                "parameters": {"ka": ka, "kb": kb},
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": candidate_id,
                    "output_root": str(output_root),
                    "parameters": {"ka": ka, "kb": kb},
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
    assert len(lhs_row["force_grid"]) == len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)

    summary_payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION
    assert summary_payload["record_count"] == 4


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
