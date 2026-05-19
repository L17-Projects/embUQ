from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import ingest_active_learning_results, write_result_ingestion_artifacts


def _png_has_signature(path: Path) -> bool:
    return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_ingest_accepts_gv_and_emb_shapes_and_normalizes_records() -> None:
    gv_metadata = {
        "acquisition_score": 0.91,
        "selection_batch_id": "al-batch-001",
        "selection_timestamp": "2026-05-19T13:00:00Z",
    }
    ingestion = ingest_active_learning_results(
        [
            {
                "candidate_id": "gv-candidate-001",
                "family": "gv",
                "experiment": "stretching",
                "status": "completed",
                "expected_hdf5_datasets": {
                    "campaign": {
                        "family": "gv",
                        "dataset_id": "gv:stretching:campaign",
                        "hdf5_path": "_runs/gv/campaign/numerical_dataset.h5",
                        "manifest_path": "_runs/gv/campaign/manifest.json",
                    },
                    "runs": [
                        {
                            "family": "gv",
                            "dataset_id": "gv:stretching:run:0",
                            "hdf5_path": "_runs/gv/campaign/run_0000/numerical_dataset.h5",
                            "manifest_path": "_runs/gv/campaign/run_0000/manifest.json",
                        }
                    ],
                },
                "reduced_observables": {"peak_force": 10.5, "energy": 4.2},
                "active_learning_metadata": gv_metadata,
                "candidate_lineage": {
                    "candidate_id": "gv-candidate-001",
                    "active_learning_metadata": gv_metadata,
                    "iteration_id": "iter-0001",
                    "parent_iteration_id": "iter-0000",
                },
            },
            {
                "candidate_id": "emb-candidate-002",
                "family": "emb",
                "normalized_payload": {"experiment": "indentation"},
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": "emb:indentation:dataset",
                        "hdf5_path": "_runs/emb/emb_indentation_dataset.h5",
                        "manifest_path": "_runs/emb/emb_placeholder_manifest.json",
                    }
                ],
                "reduced_observables": {"stiffness": 2.1},
            },
        ],
        run_id="al-run",
        iteration=3,
    )

    assert ingestion.iteration == "iter_0003"
    assert len(ingestion.records) == 2
    assert [record.family for record in ingestion.records] == ["emb", "gv"]
    gv_record = next(item for item in ingestion.records if item.family == "gv")
    emb_record = next(item for item in ingestion.records if item.family == "emb")
    assert gv_record.experiment == "stretching"
    assert emb_record.experiment == "indentation"
    assert len(gv_record.raw_curve_refs) == 2
    assert len(emb_record.raw_curve_refs) == 1
    assert gv_record.active_learning_metadata == gv_metadata
    assert gv_record.candidate_lineage["iteration_id"] == "iter-0001"
    assert gv_record.candidate_lineage["active_learning_metadata"] == gv_metadata
    assert ingestion.status_counts["completed"] == 2


def test_ingest_preserves_dpd_lineage_canonical_fields() -> None:
    metadata = {
        "acquisition_score": 0.73,
        "selection_batch_id": "al-batch-009",
        "selection_rank": 2,
    }
    ingestion = ingest_active_learning_results(
        [
            {
                "candidate_id": "gv-lineage-001",
                "family": "gv",
                "normalized_payload": {"experiment": "stretching"},
                "active_learning_metadata": metadata,
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": "gv:stretching:lineage",
                        "hdf5_path": "_runs/gv/lineage.h5",
                        "manifest_path": "_runs/gv/lineage.json",
                    }
                ],
                "reduced_observables": {"peak_force": 5.0},
            }
        ],
        run_id="al-run",
        iteration=1,
    )

    record = ingestion.records[0]
    assert record.active_learning_metadata == metadata
    assert record.candidate_lineage == {
        "candidate_id": "gv-lineage-001",
        "active_learning_metadata": metadata,
    }

    manifest = ingestion.as_manifest()
    report = ingestion.as_report()
    assert manifest["candidate_lineage"] == [record.candidate_lineage]
    assert report["candidate_lineage"] == [record.candidate_lineage]
    assert manifest["records"][0]["active_learning_metadata"] == metadata
    assert "lineage" not in manifest["records"][0]


def test_ingest_marks_partial_and_reports_missing_refs() -> None:
    ingestion = ingest_active_learning_results(
        [
            {
                "candidate_id": "gv-partial-001",
                "family": "gv",
                "experiment": "buckling",
                "status": "completed",
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": "gv:buckling:dataset",
                        "hdf5_path": "_runs/gv/gv_buckling_dataset.h5",
                        "manifest_path": "",
                    }
                ],
            },
            {
                "candidate_id": "emb-missing-002",
                "family": "emb",
                "experiment": "indentation",
                "status": "missing",
                "expected_hdf5_datasets": [],
            },
        ],
        run_id="al-run",
        iteration="iter_0007",
    )

    partial = next(item for item in ingestion.records if item.candidate_id == "gv-partial-001")
    missing = next(item for item in ingestion.records if item.candidate_id == "emb-missing-002")
    assert partial.status == "partial"
    assert "completed_but_incomplete_payload" in partial.reason_codes
    assert "missing_manifest_path" in partial.reason_codes
    assert "missing_reduced_observables" in partial.reason_codes
    assert missing.status == "missing"
    assert "missing_expected_hdf5_refs" in missing.reason_codes

    report = ingestion.as_report()
    missing_ids = {entry["candidate_id"] for entry in report["missing_refs"]}
    assert missing_ids == {"gv-partial-001", "emb-missing-002"}
    assert report["status_counts"] == {
        "completed": 0,
        "failed": 0,
        "missing": 1,
        "partial": 1,
    }


@pytest.mark.parametrize(
    ("bad_ref", "expected_reason"),
    [
        (
            {
                "dataset_id": "gv:bad:missing-hdf5",
                "hdf5_path": "",
                "manifest_path": "_runs/gv/bad.json",
            },
            "missing_hdf5_path",
        ),
        (
            {
                "dataset_id": "gv:bad:missing-manifest",
                "hdf5_path": "_runs/gv/bad.h5",
                "manifest_path": "",
            },
            "missing_manifest_path",
        ),
        (
            {
                "dataset_id": "gv:bad:invalid-hdf5-suffix",
                "hdf5_path": "_runs/gv/bad.csv",
                "manifest_path": "_runs/gv/bad.json",
            },
            "invalid_hdf5_path_suffix",
        ),
        (
            {
                "dataset_id": "gv:bad:invalid-manifest-suffix",
                "hdf5_path": "_runs/gv/bad.h5",
                "manifest_path": "_runs/gv/bad.txt",
            },
            "invalid_manifest_path_suffix",
        ),
    ],
)
def test_completed_records_with_invalid_raw_curve_refs_are_not_counted_as_completed(
    bad_ref: dict[str, str],
    expected_reason: str,
) -> None:
    ingestion = ingest_active_learning_results(
        [
            {
                "candidate_id": "bad-ref",
                "family": "gv",
                "experiment": "stretching",
                "status": "completed",
                "expected_hdf5_datasets": [bad_ref],
                "reduced_observables": {"peak_force": 7.0},
            },
            {
                "candidate_id": "valid-ref",
                "family": "gv",
                "experiment": "stretching",
                "status": "completed",
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": "gv:valid",
                        "hdf5_path": "_runs/gv/valid.h5",
                        "manifest_path": "_runs/gv/valid.json",
                    }
                ],
                "reduced_observables": {"peak_force": 8.0},
            },
        ],
        run_id="al-run",
        iteration=8,
    )

    bad_record = next(item for item in ingestion.records if item.candidate_id == "bad-ref")
    valid_record = next(item for item in ingestion.records if item.candidate_id == "valid-ref")
    assert bad_record.raw_curve_refs
    assert bad_record.status == "partial"
    assert expected_reason in bad_record.reason_codes
    assert "completed_but_incomplete_payload" in bad_record.reason_codes
    assert valid_record.status == "completed"
    assert ingestion.status_counts["completed"] == 1
    assert ingestion.status_counts["partial"] == 1

    report = ingestion.as_report()
    assert report["status_counts"]["completed"] == 1
    missing_refs = {entry["candidate_id"]: entry for entry in report["missing_refs"]}
    assert set(missing_refs) == {"bad-ref"}
    assert missing_refs["bad-ref"]["status"] == "partial"
    assert expected_reason in missing_refs["bad-ref"]["reason_codes"]


def test_ingestion_hash_is_deterministic_across_manifest_order() -> None:
    manifests = [
        {
            "candidate_id": "b-candidate",
            "family": "gv",
            "experiment": "stretching",
            "expected_hdf5_datasets": [
                {
                    "dataset_id": "gv:b",
                    "hdf5_path": "_runs/gv/b.h5",
                    "manifest_path": "_runs/gv/b.json",
                }
            ],
            "reduced_observables": {"obs": 2.0},
        },
        {
            "candidate_id": "a-candidate",
            "family": "emb",
            "experiment": "indentation",
            "expected_hdf5_datasets": [
                {
                    "dataset_id": "emb:a",
                    "hdf5_path": "_runs/emb/a.h5",
                    "manifest_path": "_runs/emb/a.json",
                }
            ],
            "reduced_observables": {"obs": 1.0},
        },
    ]
    first = ingest_active_learning_results(manifests, run_id="hash-run", iteration=0)
    second = ingest_active_learning_results(list(reversed(manifests)), run_id="hash-run", iteration=0)
    assert first.ingestion_hash == second.ingestion_hash
    assert [record.candidate_id for record in first.records] == ["a-candidate", "b-candidate"]


def test_write_result_ingestion_artifacts_creates_manifest_report_plot_and_sidecar(tmp_path: Path) -> None:
    ingestion = ingest_active_learning_results(
        [
            {
                "candidate_id": "gv-001",
                "family": "gv",
                "experiment": "stretching",
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": "gv:stretching:001",
                        "hdf5_path": "_runs/gv/gv_001.h5",
                        "manifest_path": "_runs/gv/gv_001.json",
                    }
                ],
                "reduced_observables": {"peak_force": 7.5},
            }
        ],
        run_id="artifact-run",
        iteration=2,
    )

    artifacts = write_result_ingestion_artifacts(
        output_root=tmp_path / "_runs" / "active_learning",
        ingestion=ingestion,
        include_plot=False,
    )

    assert artifacts.manifest_path.is_file()
    assert artifacts.report_path.is_file()
    assert artifacts.plot_path.is_file()
    assert artifacts.plot_sidecar_path.is_file()
    assert artifacts.plot_path.stat().st_size > 0
    assert _png_has_signature(artifacts.plot_path)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(artifacts.plot_sidecar_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert report["family_counts"] == {"emb": 0, "gv": 1}
    assert report["candidate_lineage"] == [
        {
            "candidate_id": "gv-001",
            "active_learning_metadata": {},
        }
    ]
    assert sidecar == report


def test_result_ingestion_module_import_does_not_load_matplotlib() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('meso_uq.active_learning.result_ingestion'); "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.stdout.strip() == "False"
