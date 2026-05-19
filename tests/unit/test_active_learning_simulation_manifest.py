from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import (
    SimulationBatchManifest,
    SimulationManifestRecord,
    SubmissionBoundaryState,
    write_failed_simulation_quarantine,
)


def _record(
    *,
    candidate_id: str,
    family: str,
    experiment: str,
    run_status: str = "success",
    error_message: str | None = None,
    stage: str = "simulation",
    retry_count: int = 0,
) -> SimulationManifestRecord:
    return SimulationManifestRecord(
        candidate_id=candidate_id,
        family=family,
        experiment=experiment,
        expected_dataset_refs=(f"/datasets/{candidate_id}/state.h5:/state",),
        scheduler_artifact_refs=(f"/scheduler/{candidate_id}/job.sbatch",),
        render_artifact_refs=(f"/render/{candidate_id}/render.json",),
        submission=SubmissionBoundaryState(
            submitted=run_status in {"running", "success", "failed"},
            state="submitted" if run_status in {"running", "success", "failed"} else "not_submitted",
            submission_commands=("sbatch job.sbatch",),
        ),
        stage=stage,
        run_status=run_status,
        error_message=error_message,
        retry_count=retry_count,
        provenance_hashes={
            "candidate_hash": f"cand-{candidate_id}",
            "payload_hash": f"payload-{candidate_id}",
        },
    )


def _png_has_signature(path: Path) -> bool:
    return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_simulation_manifest_hashes_are_deterministic() -> None:
    records = (
        _record(candidate_id="gv-0001", family="gv", experiment="stretching", run_status="success"),
        _record(
            candidate_id="emb-0001",
            family="emb",
            experiment="indentation",
            run_status="failed",
            error_message="solver_error",
            retry_count=1,
        ),
    )
    manifest = SimulationBatchManifest(run_id="run-001", iteration=2, records=records)
    restored = SimulationBatchManifest.from_dict(manifest.as_dict())

    assert manifest.stable_hash() == restored.stable_hash()
    assert records[0].stable_hash() == SimulationManifestRecord.from_dict(records[0].as_dict()).stable_hash()


def test_simulation_manifest_json_roundtrip() -> None:
    manifest = SimulationBatchManifest(
        run_id="run-roundtrip",
        iteration=0,
        records=(
            _record(candidate_id="gv-010", family="gv", experiment="buckling"),
            _record(candidate_id="emb-010", family="emb", experiment="stretching"),
        ),
        metadata={"source_issue": "MES-11"},
    )
    payload = manifest.as_dict()
    restored = SimulationBatchManifest.from_dict(payload)

    assert restored == manifest


def test_failed_result_quarantine_writes_manifest_png_and_sidecar(tmp_path: Path) -> None:
    manifest = SimulationBatchManifest(
        run_id="run-quarantine",
        iteration=3,
        records=(
            _record(candidate_id="gv-ok", family="gv", experiment="stretching", run_status="success"),
            _record(
                candidate_id="emb-fail",
                family="emb",
                experiment="indentation",
                run_status="failed",
                error_message="time_limit_exceeded",
                stage="scheduler",
                retry_count=2,
            ),
        ),
    )

    artifacts = write_failed_simulation_quarantine(output_root=tmp_path / "_runs" / "active_learning", manifest=manifest)

    assert artifacts is not None
    assert artifacts.manifest_path.is_file()
    assert artifacts.validation_sidecar_path.is_file()
    assert artifacts.validation_png_path.is_file()
    assert artifacts.validation_png_path.stat().st_size > 0
    assert _png_has_signature(artifacts.validation_png_path)

    quarantine_manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert quarantine_manifest["record_count"] == 1
    assert quarantine_manifest["records"][0]["candidate_id"] == "emb-fail"

    sidecar = json.loads(artifacts.validation_sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["failed_count"] == 1
    assert sidecar["family_distribution"] == {"emb": 1}
    assert sidecar["stage_distribution"] == {"scheduler": 1}
    assert sidecar["reason_distribution"] == {"time_limit_exceeded": 1}


def test_success_only_quarantine_creates_no_artifacts(tmp_path: Path) -> None:
    manifest = SimulationBatchManifest(
        run_id="run-success-only",
        iteration=1,
        records=(
            _record(candidate_id="gv-001", family="gv", experiment="stretching", run_status="success"),
            _record(candidate_id="emb-001", family="emb", experiment="indentation", run_status="success"),
        ),
    )
    output_root = tmp_path / "_runs" / "active_learning"

    artifacts = write_failed_simulation_quarantine(output_root=output_root, manifest=manifest)

    assert artifacts is None
    quarantine_dir = output_root / "run-success-only" / "iterations" / "iter_0001" / "quarantine"
    assert not quarantine_dir.exists()


def test_success_rerun_clears_stale_failed_quarantine(tmp_path: Path) -> None:
    output_root = tmp_path / "_runs" / "active_learning"
    failed_manifest = SimulationBatchManifest(
        run_id="run-rerun",
        iteration=2,
        records=(
            _record(
                candidate_id="gv-retry",
                family="gv",
                experiment="stretching",
                run_status="failed",
                error_message="retryable_solver_error",
                retry_count=1,
            ),
        ),
    )
    stale_artifacts = write_failed_simulation_quarantine(output_root=output_root, manifest=failed_manifest)
    assert stale_artifacts is not None
    assert stale_artifacts.quarantine_dir.is_dir()
    assert stale_artifacts.manifest_path.is_file()

    success_manifest = SimulationBatchManifest(
        run_id="run-rerun",
        iteration=2,
        records=(
            _record(
                candidate_id="gv-retry",
                family="gv",
                experiment="stretching",
                run_status="success",
                retry_count=2,
            ),
        ),
    )

    artifacts = write_failed_simulation_quarantine(output_root=output_root, manifest=success_manifest)

    assert artifacts is None
    assert not stale_artifacts.quarantine_dir.exists()


def test_importing_simulation_manifest_does_not_load_matplotlib() -> None:
    code = """
import sys
import meso_uq.active_learning.simulation_manifest
loaded = [name for name in ('matplotlib', 'matplotlib.pyplot') if name in sys.modules]
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

    assert result.returncode == 0, result.stderr
