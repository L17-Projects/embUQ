from __future__ import annotations

import json

import pytest
from pathlib import Path

from meso_uq.active_learning import (
    Candidate,
)
from meso_uq.active_learning.contracts import candidate_hash
from meso_uq.active_learning.dpd_sampling_gate import (
    ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION,
    ACTIVE_LEARNING_DPD_DRY_RUN_GATE_MANIFEST_FILENAME,
    ACTIVE_LEARNING_DPD_DRY_RUN_GATE_REPORT_FILENAME,
    build_and_render_active_learning_dpd_sampling_gate,
)


_GV_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.1,
    "b2": 0.2,
    "a3": 0.3,
    "a4": 0.4,
    "mu_l": 0.5,
    "c": 0.6,
}


def _gv_candidate(candidate_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        parameters={
            "gv_launch": {
                "experiment": "stretching",
                "controls": {"tot_force": [500.0, 750.0]},
            }
        },
        metadata={"selection_batch": "al-dry-run"},
    )


def _emb_candidate(candidate_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        parameters={
            "family": "emb",
            "emb_launch": {
                "experiment": "stretching",
                "material_parameters": {"stiffness": 1.5},
            },
        },
        metadata={"selection_batch": "al-dry-run"},
    )


def _gv_defaults() -> dict[str, object]:
    return {
        "geometry": {"radGV": 2.0, "height": 14.28},
        "material_parameters": _GV_MATERIAL_PARAMETERS,
        "controls": {"bpress": -91.0},
    }


def _run_root(tmp_path: Path, run_id: str, iteration: int | str = "0000") -> Path:
    return tmp_path / "_runs" / "active_learning" / run_id / "iterations" / f"iter_{int(iteration):04d}"


def _png_has_payload(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_build_and_render_gv_dry_run_gate_enforces_no_submission_and_records_hashes(tmp_path: Path) -> None:
    root = _run_root(tmp_path, "run-dry-gv")
    result = build_and_render_active_learning_dpd_sampling_gate(
        [_gv_candidate("gv-gate-01"), _gv_candidate("gv-gate-02")],
        run_id="run-dry-gv",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        provenance_tags={"source_issue": "MES-22"},
        campaign_root=root,
        batch_id="batch-gate-gv",
        platforms=("karolina", "vega"),
        overwrite=True,
        upstream_validation_plot_paths=[tmp_path / "upstream-gv.png"],
    )

    assert result.batch_request.family == "gv"
    assert result.batch_request.platform == "karolina"
    assert result.batch_request.walltime == "00:30:00"
    assert result.render_result.validation_report.submission["submitted"] is False
    assert result.render_result.validation_report.submission["submission_commands"] == []
    assert result.manifest["schema_version"] == ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION
    assert result.report["schema_version"] == ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION
    assert result.report["status"] == "passed"
    assert result.manifest["status"] == "passed"
    assert result.report["candidate_count"] == 2

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    report_payload = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert manifest_payload["candidate_records"][0]["candidate_hash"] == candidate_hash(_gv_candidate("gv-gate-01"))
    assert report_payload["candidate_ids"] == ["gv-gate-01", "gv-gate-02"]
    assert report_payload["candidate_hashes"] == [
        candidate_hash(_gv_candidate("gv-gate-01")),
        candidate_hash(_gv_candidate("gv-gate-02")),
    ]
    assert manifest_payload["family"] == "gv"
    assert manifest_payload["platform"] == "karolina"
    assert manifest_payload["scheduler_boundary"]["mode"] == "render_only"
    assert manifest_payload["scheduler_boundary"]["submission"]["submitted"] is False
    assert "upstream_validation_plot_paths" in manifest_payload["validation_artifacts"]
    assert manifest_payload["validation_artifacts"]["upstream_validation_plot_paths"][0] == str(
        tmp_path / "upstream-gv.png"
    )
    assert _png_has_payload(Path(manifest_payload["validation_artifacts"]["dpd_plot_paths"][0]))
    assert result.plot_path is not None and result.plot_path.is_file()
    assert result.plot_sidecar_path is not None and result.plot_sidecar_path.is_file()
    assert result.manifest_path.name == ACTIVE_LEARNING_DPD_DRY_RUN_GATE_MANIFEST_FILENAME
    assert result.report_path.name == ACTIVE_LEARNING_DPD_DRY_RUN_GATE_REPORT_FILENAME


def test_build_and_render_emb_dry_run_gate_records_expected_hdf5_contracts(tmp_path: Path) -> None:
    root = _run_root(tmp_path, "run-dry-emb")
    candidates = [
        _emb_candidate("emb-gate-001"),
        _emb_candidate("emb-gate-002"),
    ]
    result = build_and_render_active_learning_dpd_sampling_gate(
        candidates,
        run_id="run-dry-emb",
        iteration=0,
        platform="vega",
        walltime="00:30:00",
        gpu_count=1,
        campaign_root=root,
        batch_id="batch-gate-emb",
    )

    assert result.batch_request.family == "emb"
    assert result.batch_request.platform == "vega"
    assert result.report["status"] == "passed"
    assert result.report["criteria"]["render_only_mode"] is True
    assert result.report["criteria"]["single_family"] is True
    assert all(
        item["family"] == "emb"
        for item in result.manifest["candidate_records"]
    )

    expected_ref_count = result.manifest["expected_hdf5_ref_count"]
    assert expected_ref_count == 2
    assert len(result.manifest["expected_hdf5_refs"]) == 2
    assert all("dataset_id" in item for item in result.manifest["expected_hdf5_refs"])
    assert {item["candidate_id"] for item in result.manifest["candidate_records"]} == {
        "emb-gate-001",
        "emb-gate-002",
    }


def test_gate_blocks_mixed_family_batch_before_submission(tmp_path: Path) -> None:
    root = _run_root(tmp_path, "run-dry-mixed")
    candidates = [
        _gv_candidate("gv-mixed"),
        _emb_candidate("emb-mixed"),
    ]

    with pytest.raises(ValueError, match="Mixed candidate families"):
        build_and_render_active_learning_dpd_sampling_gate(
            candidates,
            run_id="run-dry-mixed",
            iteration=0,
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            campaign_root=root,
            batch_id="batch-mixed",
        )


def test_gate_reports_failed_criteria_for_multi_experiment_batch(tmp_path: Path) -> None:
    root = _run_root(tmp_path, "run-dry-exp")
    candidates = [
        Candidate(
            candidate_id="gv-stretch",
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0]},
                }
            },
        ),
        Candidate(
            candidate_id="gv-buckle",
            parameters={
                "gv_launch": {
                    "experiment": "buckling",
                    "controls": {"buck": [0.5]},
                }
            },
        ),
    ]
    result = build_and_render_active_learning_dpd_sampling_gate(
        candidates,
        run_id="run-dry-exp",
        iteration=0,
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        campaign_root=root,
        batch_id="batch-exp",
        overwrite=True,
    )

    assert result.report["status"] == "failed"
    assert result.report["failure_reasons"] == ["multiple_experiments"]
    assert result.report["criteria"]["single_experiment"] is False
    assert result.report["passed"] is False
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
