from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_causal_validation_ingestion import (
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_MANIFEST_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME,
    assert_same_validation_set,
    build_emb_34um_causal_validation_ingestion_report,
    write_emb_34um_causal_validation_ingestion_artifacts,
)
from meso_uq.active_learning.emb_34um_causal_validation_design import (
    EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    build_emb_34um_causal_validation_campaign_manifest,
)


def _record(tmp_path: Path, cid: str, *, method: str, step: int, seed: int = 1, with_output: bool = True, provenance: str = "causal-validation-dpd") -> dict:
    out = tmp_path / cid
    out.mkdir(parents=True, exist_ok=True)
    if with_output:
        (out / "F_Delta.dat").write_text("ok\n", encoding="utf-8")
    return {
        "candidate_id": cid,
        "output_root": str(out),
        "vault_output_root": str(out / "vault"),
        "method": method,
        "seed": seed,
        "step": step,
        "ka": 1.0,
        "kb": 2.0,
        "provenance": provenance,
        "validation_set_id": f"validation-seed-{seed}",
        "hashes": {"input": "abc"},
    }


def _batch(step: int | None, records: list[dict]) -> dict:
    payload = {"records": records}
    if step is not None:
        payload["step"] = step
    return payload


def _manifest(tmp_path: Path, *, validation_id: str = "val-a", extra_reserve: bool = False, failed_al: bool = False) -> Path:
    al_primary = _record(tmp_path, "al-001", method="al", step=1, with_output=not failed_al)
    if failed_al:
        Path(al_primary["output_root"]).joinpath("runtime_status.json").write_text('{"status":"failed"}', encoding="utf-8")
    payload = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "policy": {"shared_size": 1, "validation_size": 1, "step_count": 1, "step_size": 1},
        "seeds": [
            {
                "seed": 1,
                "shared_initial": _batch(None, [_record(tmp_path, "shared-001", method="shared_initial", step=0)]),
                "validation": _batch(None, [{**_record(tmp_path, "val-001", method="validation", step=0), "validation_set_id": validation_id}]),
                "al_steps": [_batch(1, [al_primary])],
                "lhs_steps": [_batch(1, [_record(tmp_path, "lhs-001", method="lhs", step=1)])],
                "candidate_reserve": {
                    "al": {
                        "records": [
                            _record(tmp_path, "al-reserve-001", method="al", step=1),
                            _record(tmp_path, "al-reserve-002", method="al", step=1),
                        ]
                    }
                    if extra_reserve
                    else {"records": []}
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest_with_counts(
    tmp_path: Path,
    *,
    shared_size: int = 3,
    validation_size: int = 3,
    step_count: int = 5,
    step_size: int = 3,
    failed_al: bool = False,
    failed_al_step: int = 1,
    fail_candidate_id: str = "al-01-001",
) -> tuple[Path, str]:
    campaign_root = tmp_path / "campaign"

    def _step_candidate(method: str, step: int, index: int, *, with_output: bool = True) -> dict:
        candidate_id = f"{method}-{step:02d}-{index:03d}"
        out = tmp_path / candidate_id
        out.mkdir(parents=True, exist_ok=True)
        if with_output:
            (out / "F_Delta.dat").write_text("ok\n", encoding="utf-8")
        return {
            "candidate_id": candidate_id,
            "output_root": str(out),
            "vault_output_root": str(out / "vault"),
            "method": method,
            "seed": 1,
            "step": step,
            "ka": 1.0,
            "kb": 2.0,
            "provenance": "causal-validation-dpd",
            "validation_set_id": "validation-seed-1",
            "hashes": {"input": "abc"},
        }

    shared_records = [_step_candidate("shared_initial", 0, index=i) for i in range(1, shared_size + 1)]
    validation_records = [_step_candidate("validation", 0, index=i) for i in range(1, validation_size + 1)]

    al_steps = []
    for step in range(1, step_count + 1):
        records = []
        for index in range(1, step_size + 1):
            candidate_id = f"al-{step:02d}-{index:03d}"
            with_output = not (failed_al and step == failed_al_step and candidate_id == fail_candidate_id)
            record = _step_candidate("al", step, index, with_output=with_output)
            if not with_output and failed_al:
                Path(record["output_root"]).joinpath("runtime_status.json").write_text('{"status":"failed"}', encoding="utf-8")
            records.append(record)
        al_steps.append(_batch(step, records))

    lhs_steps = [_batch(step, [_step_candidate("lhs", step, index) for index in range(1, step_size + 1)]) for step in range(1, step_count + 1)]

    payload = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "campaign_root": str(campaign_root),
        "policy": {
            "shared_size": shared_size,
            "validation_size": validation_size,
            "step_count": step_count,
            "step_size": step_size,
        },
        "seeds": [
            {
                "seed": 1,
                "shared_initial": {"records": shared_records},
                "validation": {"records": validation_records},
                "al_steps": al_steps,
                "lhs_steps": lhs_steps,
                "candidate_reserve": {
                    "al": {"records": []},
                    "lhs": {"records": []},
                    "validation": {"records": []},
                    "shared_initial": {"records": []},
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest_counts.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, fail_candidate_id


def _design_manifest(tmp_path: Path, *, timestamp: str = "20260520_140000") -> tuple[Path, Path]:
    campaign_root = tmp_path / "scratch" / timestamp
    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp=timestamp,
        run_id_prefix="emb-34um-causal-validation-test",
        campaign_root=campaign_root,
        scratch_root=campaign_root,
        vault_root_timestamp=tmp_path / "vault" / timestamp,
        force_grid=(1.0, 2.0, 5.0, 8.0, 13.0),
        step_count=5,
        seed_count=3,
        include_coverage_plot_requirements=False,
    )
    manifest_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, campaign_root


def _write_replacement_artifacts(
    campaign_root: Path,
    *,
    replacement_candidate: str,
    failed_candidate: str,
    step: int,
    with_output: bool = True,
    batch_index: int = 1,
) -> None:
    replacement_root = (
        campaign_root
        / f"seed-001"
        / f"al-step-{step:02d}"
        / "replacement"
        / f"batch-{batch_index:03d}"
    )
    replacement_root.mkdir(parents=True, exist_ok=True)
    replacement_output_root = replacement_root / "emb" / replacement_candidate
    if with_output:
        replacement_output_root.mkdir(parents=True, exist_ok=True)
        (replacement_output_root / "F_Delta.dat").write_text("ok\n", encoding="utf-8")

    replacement_record = {
        "replacement_batch_index": batch_index,
        "replacement_sequence": 1,
        "failed_candidate_id": failed_candidate,
        "replacement_candidate_id": replacement_candidate,
        "replacement_candidate_hash": "sha256:placeholder",
        "candidate_pool_id": f"pool-{replacement_candidate}",
        "selection_order": 1,
        "selection_pool_rank": 1,
        "selection_seed": 1,
        "force_grid": [1.0, 2.0, 3.0],
        "ka": 10.5,
        "kb": 20.5,
        "output_root": str(replacement_output_root),
        "vault_output_root": str(replacement_output_root / "vault"),
    }

    batch_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_replacement_batch_summary.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "replacement_batch_index": batch_index,
        "replacement_record_count": 1,
        "failed_candidate_ids": [failed_candidate],
        "replacement_records": [replacement_record],
    }
    replacement_batch_summary_path = replacement_root / EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME
    replacement_batch_summary_path.write_text(json.dumps(batch_summary), encoding="utf-8")

    replacement_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_replacement_manifest.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "seed": 1,
        "step": step,
        "stage": f"al-step-{step:02d}",
        "selection_manifest_path": str(replacement_batch_summary_path),
        "replacement_records": [replacement_record],
    }
    replacement_manifest_path = replacement_root.parent / EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME
    replacement_manifest_path.write_text(json.dumps(replacement_manifest), encoding="utf-8")


def _write_lhs_replacement_artifacts(
    campaign_root: Path,
    *,
    replacement_candidate: str,
    failed_candidate: str,
    step: int,
    with_output: bool = True,
) -> None:
    replacement_root = (
        campaign_root
        / "seed-001"
        / f"lhs-step-{step:02d}"
        / "replacement"
        / "batch-001"
    )
    replacement_root.mkdir(parents=True, exist_ok=True)
    replacement_output_root = replacement_root / "emb" / replacement_candidate
    if with_output:
        replacement_output_root.mkdir(parents=True, exist_ok=True)
        (replacement_output_root / "F_Delta.dat").write_text("ok\n", encoding="utf-8")

    replacement_record = {
        "replacement_batch_index": 1,
        "replacement_sequence": 1,
        "failed_candidate_id": failed_candidate,
        "replacement_candidate_id": replacement_candidate,
        "replacement_candidate_hash": "sha256:lhs-placeholder",
        "force_grid": [1.0, 2.0, 3.0],
        "ka": 11.5,
        "kb": 21.5,
        "output_root": str(replacement_output_root),
        "vault_output_root": str(replacement_output_root / "vault"),
        "replacement_policy": "rerun-original",
    }

    batch_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_lhs_replacement_batch_summary.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "replacement_batch_index": 1,
        "replacement_record_count": 1,
        "failed_candidate_ids": [failed_candidate],
        "replacement_records": [replacement_record],
    }
    replacement_batch_summary_path = replacement_root / EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME
    replacement_batch_summary_path.write_text(json.dumps(batch_summary), encoding="utf-8")

    replacement_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_lhs_replacement_manifest.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "seed": 1,
        "step": step,
        "stage": f"lhs-step-{step:02d}",
        "replacement_policy": "rerun-original",
        "replacement_records": [replacement_record],
    }
    replacement_manifest_path = replacement_root.parent / EMB_34UM_CAUSAL_VALIDATION_LHS_REPLACEMENT_MANIFEST_FILENAME
    replacement_manifest_path.write_text(json.dumps(replacement_manifest), encoding="utf-8")


def test_classification_completed_failed_missing(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, failed_al=True)
    out = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["seeds"][0]["lhs_steps"][0]["records"][0]["output_root"])
    (out / "F_Delta.dat").unlink()
    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    statuses = {row["candidate_id"]: row["status"] for row in report_manifest["records"]}
    assert statuses["shared-001"] == "completed"
    assert statuses["al-001"] == "failed"
    assert statuses["lhs-001"] == "missing"
    assert report["status"] == "blocked"


def test_replacement_plan_is_deterministic(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, extra_reserve=True, failed_al=True)
    artifacts = write_emb_34um_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=tmp_path / "artifacts",
        write_replacement_plan=True,
    )
    assert artifacts.replacement_plan is not None
    records = artifacts.replacement_plan["records"]
    assert records
    assert records[0]["candidate_id"] == "al-reserve-001"
    assert records[0]["replacement_for"] == "al-001"
    assert records[0]["original_candidate_id"] == "al-001"
    assert records[0]["reason"] in {"failed", "missing"}


def test_fresh_only_rejects_legacy_records(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seeds"][0]["shared_initial"]["records"][0]["provenance"] = "legacy final-gate reused"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh-only"):
        build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path, fresh_only=True)


def test_shortfall_blockers_present(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["policy"]["shared_size"] = 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    assert report["status"] == "blocked"
    assert any("shortfall" in item for item in report["blockers"])


def test_shortfall_blockers_include_missing_al_rows(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seeds"][0]["al_steps"][0]["records"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    assert report_manifest["record_count"] == 3
    assert report["status"] == "blocked"
    assert any("policy=al completed=0 expected=1 shortfall=1" in item for item in report["blockers"])


def test_dynamic_al_records_from_rendered_batch_summary(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seeds"][0]["al_steps"][0]["records"] = []

    dynamic_root = tmp_path / "seed-001" / "al-step-01"
    dynamic_output_root = dynamic_root / "emb" / "al-dyn-001"
    dynamic_output_root.mkdir(parents=True, exist_ok=True)
    (dynamic_output_root / "F_Delta.dat").write_text("ok\n", encoding="utf-8")

    candidate_manifest_path = dynamic_root / "candidate-001.json"
    candidate_manifest_path.write_text(
        json.dumps(
            {
                "candidate_id": "al-dyn-001",
                "active_learning_metadata": {
                    "seed": 1,
                    "step": 1,
                    "selection_mode": "causal_fresh_only",
                    "fresh_only": True,
                },
                "rendered_payload": {
                    "request_payload": {
                        "candidate_id": "al-dyn-001",
                        "seed": 1,
                        "step": 1,
                        "output_root": str(dynamic_output_root),
                        "parameters": {"ka": 12.5, "kb": 34.5},
                        "selection_mode": "causal_fresh_only",
                        "fresh_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    batch_summary_path = dynamic_root / "emb_34um_batch_summary.json"
    batch_summary_path.write_text(
        json.dumps({"rendered_candidate_manifests": [str(candidate_manifest_path)]}),
        encoding="utf-8",
    )

    payload["seeds"][0]["al_steps"][0]["expected_batch_summary_path"] = str(batch_summary_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    dynamic_row = next(row for row in report_manifest["records"] if row["candidate_id"] == "al-dyn-001")
    assert dynamic_row["method"] == "al"
    assert dynamic_row["seed"] == 1
    assert dynamic_row["step"] == 1
    assert dynamic_row["output_root"] == str(dynamic_output_root)
    assert dynamic_row["ka"] == pytest.approx(12.5)
    assert dynamic_row["kb"] == pytest.approx(34.5)
    assert dynamic_row["status"] == "completed"
    assert report["status"] == "passed"


def test_dynamic_al_string_false_fresh_only_is_rejected(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seeds"][0]["al_steps"][0]["records"] = []

    dynamic_root = tmp_path / "seed-001" / "al-step-01"
    dynamic_output_root = dynamic_root / "emb" / "al-dyn-001"
    dynamic_output_root.mkdir(parents=True, exist_ok=True)
    (dynamic_output_root / "F_Delta.dat").write_text("ok\n", encoding="utf-8")

    candidate_manifest_path = dynamic_root / "candidate-001.json"
    candidate_manifest_path.write_text(
        json.dumps(
            {
                "candidate_id": "al-dyn-001",
                "active_learning_metadata": {
                    "seed": 1,
                    "step": 1,
                    "selection_mode": "manual",
                    "fresh_only": "false",
                },
                "rendered_payload": {
                    "request_payload": {
                        "candidate_id": "al-dyn-001",
                        "seed": 1,
                        "step": 1,
                        "output_root": str(dynamic_output_root),
                        "parameters": {"ka": 12.5, "kb": 34.5},
                        "selection_mode": "manual",
                        "fresh_only": "false",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    batch_summary_path = dynamic_root / "emb_34um_batch_summary.json"
    batch_summary_path.write_text(
        json.dumps({"rendered_candidate_manifests": [str(candidate_manifest_path)]}),
        encoding="utf-8",
    )

    payload["seeds"][0]["al_steps"][0]["expected_batch_summary_path"] = str(batch_summary_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh-only"):
        build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)


def test_replacement_outputs_are_included_when_completed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, failed_al=True)
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_replacement_artifacts(campaign_root=campaign_root, replacement_candidate="al-replacement-001", failed_candidate="al-001", step=1)

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    replacement_records = [row for row in report_manifest["records"] if row["candidate_id"] == "al-replacement-001"]
    assert len(replacement_records) == 1
    assert replacement_records[0]["replacement"] is True
    assert replacement_records[0]["replacement_for"] == "al-001"
    assert replacement_records[0]["status"] == "completed"
    assert report["status"] == "passed"


def test_replacement_string_false_fresh_only_is_rejected(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, failed_al=True)
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_replacement_artifacts(
        campaign_root=campaign_root,
        replacement_candidate="al-replacement-001",
        failed_candidate="al-001",
        step=1,
    )
    for path in (
        campaign_root
        / "seed-001"
        / "al-step-01"
        / "replacement"
        / EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_MANIFEST_FILENAME,
        campaign_root
        / "seed-001"
        / "al-step-01"
        / "replacement"
        / "batch-001"
        / EMB_34UM_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME,
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["replacement_records"][0]["metadata"] = {"selection_mode": "manual", "fresh_only": "false"}
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh-only"):
        build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)


def test_replacement_outputs_with_missing_success_file_do_not_complete_policy(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, failed_al=True)
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_replacement_artifacts(campaign_root=campaign_root, replacement_candidate="al-replacement-002", failed_candidate="al-001", step=1, with_output=False)

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    replacement_records = [row for row in report_manifest["records"] if row["candidate_id"] == "al-replacement-002"]
    assert len(replacement_records) == 1
    assert replacement_records[0]["status"] == "missing"


def test_lhs_replacement_outputs_are_included_when_completed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_lhs_replacement_artifacts(
        campaign_root=campaign_root,
        replacement_candidate="lhs-replacement-001",
        failed_candidate="lhs-001",
        step=1,
    )

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    replacement_records = [row for row in report_manifest["records"] if row["candidate_id"] == "lhs-replacement-001"]
    assert len(replacement_records) == 1
    assert replacement_records[0]["method"] == "lhs"
    assert replacement_records[0]["policy_group"] == "lhs"
    assert replacement_records[0]["stage"] == "lhs-step-01"
    assert replacement_records[0]["replacement"] is True
    assert replacement_records[0]["replacement_for"] == "lhs-001"
    assert replacement_records[0]["replacement_policy"] == "rerun-original"
    assert replacement_records[0]["status"] == "completed"
    assert report["status"] == "passed"


def test_lhs_method_with_embedded_stage_is_grouped_in_lhs_policy(tmp_path: Path) -> None:
    manifest_path, _ = _manifest_with_counts(
        tmp_path,
        shared_size=1,
        validation_size=1,
        step_count=1,
        step_size=2,
        failed_al=False,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    lhs_records = payload["seeds"][0]["lhs_steps"][0]["records"]
    for record in lhs_records:
        record["method"] = "lhs-step-01"
    failed_lhs = str(lhs_records[0]["candidate_id"])
    Path(lhs_records[0]["output_root"]).joinpath("F_Delta.dat").unlink()

    campaign_root = Path(payload["campaign_root"])
    _write_lhs_replacement_artifacts(
        campaign_root=campaign_root,
        replacement_candidate="lhs-replacement-001",
        failed_candidate=failed_lhs,
        step=1,
    )

    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    assert report["status"] == "passed"
    lhs_rows = [
        row
        for row in report_manifest["records"]
        if str(row.get("candidate_id", "")).startswith("lhs")
    ]
    assert all(row["policy_group"] == "lhs" for row in lhs_rows)
    assert all(row["stage"] == "lhs-step-01" for row in lhs_rows)

    lhs_stage_row = next(
        row
        for row in report["counts_by_seed_stage_policy"]
        if row["seed"] == 1 and row["policy_group"] == "lhs" and row["stage"] == "lhs-step-01"
    )
    assert lhs_stage_row["completed"] == 2


def test_stage_counts_are_evaluated_after_replacements(tmp_path: Path) -> None:
    manifest_path, failed_candidate = _manifest_with_counts(
        tmp_path,
        shared_size=5,
        validation_size=5,
        step_count=5,
        step_size=4,
        failed_al=True,
        failed_al_step=1,
        fail_candidate_id="al-01-001",
    )
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_replacement_artifacts(
        campaign_root=campaign_root,
        replacement_candidate="al-replacement-001",
        failed_candidate=failed_candidate,
        step=1,
        batch_index=1,
    )

    _, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)
    expected = {
        row["stage"]: row["expected"]
        for row in report["expected_counts_by_seed_stage_policy"]
        if row["seed"] == 1 and row["policy_group"] in {"shared_initial", "validation", "al", "lhs"}
    }
    assert expected["shared_initial"] == 5
    assert expected["validation"] == 5
    assert expected["al-step-01"] == 4
    assert expected["lhs-step-01"] == 4
    assert expected["al-step-05"] == 4
    assert expected["lhs-step-05"] == 4

    al_step_counts = {
        row["stage"]: row
        for row in report["counts_by_seed_stage_policy"]
        if row["seed"] == 1 and row["policy_group"] == "al" and row["stage"].endswith("-01")
    }
    assert al_step_counts["al-step-01"]["completed"] == 4
    assert al_step_counts["al-step-01"]["failed"] == 0
    assert al_step_counts["al-step-01"]["quarantined"] == 1

    assert report["status"] == "passed"


def test_replaced_failed_original_is_quarantined_and_not_replanned(tmp_path: Path) -> None:
    manifest_path, failed_candidate = _manifest_with_counts(
        tmp_path,
        shared_size=1,
        validation_size=1,
        step_count=1,
        step_size=2,
        failed_al=True,
        failed_al_step=1,
        fail_candidate_id="al-01-001",
    )
    campaign_root = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_root"])
    _write_replacement_artifacts(
        campaign_root=campaign_root,
        replacement_candidate="al-replacement-001",
        failed_candidate=failed_candidate,
        step=1,
    )

    report_manifest, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)

    failed_rows = [row for row in report_manifest["records"] if row["candidate_id"] == failed_candidate]
    assert any(row.get("status") == "quarantined" for row in failed_rows)

    replacement_rows = [row for row in report_manifest["records"] if row.get("candidate_id") == "al-replacement-001"]
    assert len(replacement_rows) == 1
    assert replacement_rows[0]["replacement"] is True
    assert replacement_rows[0]["replacement_for"] == failed_candidate

    plan_records = report["replacement_plan_preview"]["records"]
    assert all(row["replacement_for"] != failed_candidate for row in plan_records)

    assert report["status"] == "passed"


def test_same_validation_set_mismatch() -> None:
    with pytest.raises(ValueError, match="Validation set mismatch"):
        assert_same_validation_set([{"validation_set_id": "a"}, {"validation_set_id": "b"}])


def test_cli_smoke(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, extra_reserve=True, failed_al=True)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "ingest_emb_34um_causal_validation.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--campaign-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "cli-out"),
            "--replacement-manifest-output",
            str(tmp_path / "cli-out" / "plan.json"),
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ingestion_report=" in result.stdout
    assert "ingestion_manifest=" in result.stdout


def test_real_design_manifest_ingestion_blocks_on_missing_outputs(tmp_path: Path) -> None:
    manifest_path, campaign_root = _design_manifest(tmp_path, timestamp="20260520_140001")
    _, report = build_emb_34um_causal_validation_ingestion_report(campaign_manifest_path=manifest_path)

    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert not any("fresh-only" in item for item in report["blockers"])
    assert any("shortfall" in item for item in report["blockers"])

    artifacts = write_emb_34um_causal_validation_ingestion_artifacts(
        campaign_manifest_path=manifest_path,
        output_root=campaign_root / "ingest",
        write_replacement_plan=True,
    )
    assert artifacts.artifact_dir == campaign_root / "ingest"
    assert artifacts.manifest_path.name == EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME
    assert artifacts.report_path.name == EMB_34UM_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME
    assert artifacts.summary_csv_path.name == EMB_34UM_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME
    assert artifacts.replacement_plan_path is not None
