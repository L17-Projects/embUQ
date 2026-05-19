from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    ACTIVE_LEARNING_REPORT_FILENAME,
    ACTIVE_LEARNING_REPORT_MARKDOWN_FILENAME,
    ACTIVE_LEARNING_CONCISE_REPORT_SCHEMA_VERSION,
    build_active_learning_concise_report,
    write_active_learning_concise_report,
)


def test_build_active_learning_concise_report_merges_dictionary_artifacts(tmp_path: Path) -> None:
    candidate_generation_report = {
        "candidate_ids": ["cand-001", "cand-002"],
        "candidate_hashes": {"cand-001": "hash-a", "cand-002": "hash-b"},
        "family_distribution": {"gv": 2},
        "experiment_distribution": {"stretching": 2},
    }
    acquisition_report = {
        "policy": "weighted_sum",
        "candidate_ids": ["cand-001", "cand-002", "cand-003"],
        "family_distribution": {"gv": 1, "emb": 2},
        "weights": {"uncertainty": 0.8, "diversity": 0.2},
        "scores": [{"candidate_id": "cand-001", "score": 0.9}],
        "scored_count": 2,
    }
    constraints_report = {
        "reason_code_counts": {"invalid_experiment": 1},
        "rejected_candidate_ids": ["bad-cand"],
        "per_candidate_reasons": {
            "bad-cand": [{"code": "invalid_experiment", "message": "bad family"}],
        },
        "valid_candidates": [
            {
                "candidate_id": "cand-001",
                "parameters": {},
            },
            {"candidate_id": "cand-002", "parameters": {}},
        ],
    }
    selected_batch_report = {
        "selected_candidates": ["cand-001"],
        "selected_scores": [0.91],
        "selected_candidate_hashes": ["hash-a"],
    }
    simulation_manifest = {
        "records": [
            {
                "candidate_id": "cand-001",
                "family": "gv",
                "experiment": "stretching",
                "run_status": "success",
            },
            {
                "candidate_id": "cand-004",
                "family": "emb",
                "experiment": "indentation",
                "run_status": "failed",
            },
        ]
    }
    quarantine_summary = {"failed_count": 1, "reason_distribution": {"time_out": 1}}
    ingestion_report = {
        "status_counts": {"completed": 2, "failed": 1},
        "family_counts": {"gv": 2, "emb": 1},
        "record_count": 3,
    }

    report = build_active_learning_concise_report(
        run_id="report-test-run",
        iteration=7,
        candidate_generation_report=candidate_generation_report,
        acquisition_report=acquisition_report,
        constraints_report=constraints_report,
        selected_batch_report=selected_batch_report,
        simulation_manifest=simulation_manifest,
        quarantine_summary=quarantine_summary,
        result_ingestion_report=ingestion_report,
        validation_plot_paths=(
            "/tmp/plots/one.png",
            {"plot": "/tmp/plots/two.png"},
            {"artifact": {"uri": "/tmp/plots/three.png"}},
        ),
        reproducibility_metadata={"seed": 42},
    )

    assert report["schema_version"] == ACTIVE_LEARNING_CONCISE_REPORT_SCHEMA_VERSION
    assert report["run_id"] == "report-test-run"
    assert report["iteration"] == "iter_0007"
    assert report["candidate_hashes"] == ["hash-a", "hash-b"]
    assert set(report["candidate_ids"]) >= {"cand-001", "cand-002", "cand-003", "cand-004", "bad-cand"}
    assert report["acquisition_terms"]["policy"] == "weighted_sum"
    assert report["acquisition_terms"]["weights"] == {"uncertainty": 0.8, "diversity": 0.2}
    assert report["selected_batch_summary"]["selected_count"] == 1
    assert report["selected_batch_summary"]["selected_candidates"] == ["cand-001"]
    assert report["constraint_reasons"]["rejected_candidate_ids"] == ["bad-cand"]
    assert report["constraint_reasons"]["reason_code_counts"]["invalid_experiment"] == 1
    assert report["simulation_summary"]["record_count"] == 2
    assert report["quarantine_summary"]["failed_count"] == 1
    assert report["result_ingestion_status"]["status_counts"]["completed"] == 2
    assert report["validation_plot_paths"] == ["/tmp/plots/one.png", "/tmp/plots/two.png", "/tmp/plots/three.png"]
    assert report["reproducibility_metadata"] == {"seed": 42}


def test_write_active_learning_concise_report_emits_json_and_markdown(tmp_path: Path) -> None:
    candidate_generation_report = tmp_path / "candidate_generation_report.json"
    acquisition_report = tmp_path / "acquisition_report.json"
    constraints_report = tmp_path / "constraints_report.json"
    result_ingestion_report = tmp_path / "result_ingestion_report.json"
    selected_batch_report = tmp_path / "selected_batch.json"
    simulation_manifest = tmp_path / "simulation_manifest.json"

    candidate_generation_report.write_text(
        json.dumps(
            {
                "candidate_ids": ["c1"],
                "candidate_hashes": {"c1": "h1"},
                "family_distribution": {"gv": 1},
                "experiment_distribution": {"stretching": 1},
            }
        ),
        encoding="utf-8",
    )
    acquisition_report.write_text(
        json.dumps(
            {
                "policy": "uncertainty",
                "candidate_ids": ["c1"],
                "scores": [{"candidate_id": "c1", "score": 0.9}],
                "candidate_count": 1,
                "scored_count": 1,
            }
        ),
        encoding="utf-8",
    )
    constraints_report.write_text(
        json.dumps({"reason_code_counts": {}, "rejected_candidate_ids": [], "per_candidate_reasons": {}}),
        encoding="utf-8",
    )
    selected_batch_report.write_text(
        json.dumps({"selected_candidates": ["c1"], "selected_scores": [0.9], "selected_candidate_hashes": ["h1"]}),
        encoding="utf-8",
    )
    simulation_manifest.write_text(
        json.dumps({"records": [{"candidate_id": "c1", "family": "gv", "experiment": "stretching", "run_status": "success"}]}),
        encoding="utf-8",
    )
    result_ingestion_report.write_text(
        json.dumps({"status_counts": {"completed": 1}, "family_counts": {"gv": 1}, "record_count": 1}),
        encoding="utf-8",
    )

    artifacts = write_active_learning_concise_report(
        output_root=tmp_path / "_runs" / "active_learning",
        run_id="report-output-run",
        iteration=2,
        candidate_generation_report=candidate_generation_report,
        acquisition_report=acquisition_report,
        constraints_report=constraints_report,
        selected_batch_report=selected_batch_report,
        simulation_manifest=simulation_manifest,
        result_ingestion_report=result_ingestion_report,
        validation_plot_paths=(
            tmp_path / "artifact_a.png",
            {"plot": tmp_path / "artifact_b.png"},
        ),
    )

    assert artifacts.json_path == (
        tmp_path
        / "_runs"
        / "active_learning"
        / "report-output-run"
        / "iterations"
        / "iter_0002"
        / ACTIVE_LEARNING_REPORT_FILENAME
    )
    assert artifacts.markdown_path == (
        tmp_path
        / "_runs"
        / "active_learning"
        / "report-output-run"
        / "iterations"
        / "iter_0002"
        / ACTIVE_LEARNING_REPORT_MARKDOWN_FILENAME
    )
    assert artifacts.json_path.is_file()
    assert artifacts.markdown_path.is_file()

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert payload["run_id"] == "report-output-run"
    assert payload["iteration"] == "iter_0002"
    assert payload["family_distribution"]["gv"] == 1
    assert "run_id: report-output-run" in markdown
    assert "- policy: uncertainty" in markdown
    assert "## Result Ingestion" in markdown


def test_importing_reports_module_does_not_import_matplotlib(tmp_path: Path) -> None:
    code = """
import sys
import meso_uq.active_learning.reports

blocked = ["matplotlib", "matplotlib.pyplot"]
loaded = [name for name in blocked if name in sys.modules]
assert loaded == [], loaded
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
