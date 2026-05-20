from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_final_gate_ingestion import (  # noqa: E402
    EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME,
    build_emb_34um_final_gate_ingestion_report,
    write_emb_34um_final_gate_ingestion_artifacts,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_manifest(
    *,
    root: Path,
    candidate_id: str,
    gate: str,
    round_index: int | None,
    yt: float,
    kb: float,
    force_grid: list[float],
    ka: float = 1.0,
) -> Path:
    output_root = root / gate / "emb" / candidate_id
    manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
    metadata: dict[str, object] = {"campaign": "canary" if gate == "canary_gate" else "full"}
    if round_index is not None:
        metadata["round"] = round_index
    _write_json(
        manifest_path,
        {
            "candidate_id": candidate_id,
            "family": "emb",
            "output_root": str(output_root),
            "active_learning_metadata": metadata,
            "normalized_payload": {
                "candidate_id": candidate_id,
                "experiment": "indentation",
                "campaign_root": str(root / gate),
                "output_root": str(output_root),
                "parameters": {"Yt": yt, "ka": ka, "kb": kb, "b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0},
                "force_grid": force_grid,
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": candidate_id,
                    "output_root": str(output_root),
                    "force_grid": force_grid,
                }
            },
        },
    )
    return manifest_path


def _write_f_delta(
    path: Path,
    *,
    yt: float,
    ka: float,
    kb: float,
    outputs: list[float],
    forces: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [yt, ka, kb, 0.0, 0.0, 0.0, 0.0, 6.8] + outputs + forces
    path.write_text(" ".join(str(item) for item in row) + "\n", encoding="utf-8")


def _campaign_manifest(root: Path, candidate_paths: dict[str, list[Path]], *, retry_limit: int = 3) -> Path:
    campaign_path = root / "emb_34um_final_gate_campaign_manifest.json"
    _write_json(
        campaign_path,
        {
            "schema_version": "meso_uq.active_learning.emb_34um_final_gate.v1",
            "campaign_root": str(root),
            "expected_resources": {"retry_limit": retry_limit},
            "canary_gate": {
                "rendered_candidate_manifests": [str(item) for item in candidate_paths.get("canary_gate", [])],
            },
            "full_gate": {
                "rendered_candidate_manifests": [str(item) for item in candidate_paths.get("full_gate", [])],
            },
        },
    )
    return campaign_path


def test_emb_34um_final_gate_ingestion_accepts_completed_f_delta_rows(tmp_path: Path) -> None:
    force_grid = [0.0, 10.0]
    canary_force_grid = [0.0, 5.0, 10.0]
    candidates = {
        "canary_gate": [
            _candidate_manifest(
                root=tmp_path,
                candidate_id="emb-34um-final-gate-canary-001",
                gate="canary_gate",
                round_index=None,
                yt=10.0,
                kb=20.0,
                force_grid=canary_force_grid,
            )
        ],
        "full_gate": [
            _candidate_manifest(
                root=tmp_path,
                candidate_id=f"emb-34um-final-gate-full-r0{round_index}-c001",
                gate="full_gate",
                round_index=round_index,
                yt=100.0 + round_index,
                kb=200.0 + round_index,
                force_grid=force_grid,
            )
            for round_index in (1, 2, 3)
        ],
    }
    for path in candidates["canary_gate"] + candidates["full_gate"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload["normalized_payload"]["parameters"]
        forces = payload["normalized_payload"]["force_grid"]
        outputs = [float(index + 1) for index, _ in enumerate(forces)]
        _write_f_delta(
            Path(payload["normalized_payload"]["output_root"]) / "F_Delta.dat",
            yt=float(params["Yt"]),
            ka=float(params["ka"]),
            kb=float(params["kb"]),
            outputs=outputs,
            forces=[float(item) for item in forces],
        )

    campaign_path = _campaign_manifest(tmp_path, candidates)
    manifest, report = build_emb_34um_final_gate_ingestion_report(
        campaign_manifest_path=campaign_path,
        expected_full_count=3,
        expected_canary_count=1,
    )

    assert manifest["status_counts"] == {"completed": 4, "failed": 0, "missing": 0, "partial": 0}
    assert report["passed"] is True
    assert report["status"] == "passed"
    assert report["quarantine_count"] == 0
    assert all(record["f_delta_path"].endswith("F_Delta.dat") for record in report["records"])


def test_emb_34um_final_gate_ingestion_quarantines_failed_after_retry_limit(tmp_path: Path) -> None:
    candidate_path = _candidate_manifest(
        root=tmp_path,
        candidate_id="emb-34um-final-gate-full-r01-c001",
        gate="full_gate",
        round_index=1,
        yt=100.0,
        kb=200.0,
        force_grid=[0.0, 10.0],
    )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    status_path = Path(payload["normalized_payload"]["output_root"]) / "runtime_status.json"
    _write_json(status_path, {"status": "failed", "retry_count": 3, "error_message": "runtime failed"})
    campaign_path = _campaign_manifest(tmp_path, {"full_gate": [candidate_path]}, retry_limit=3)

    _, report = build_emb_34um_final_gate_ingestion_report(
        campaign_manifest_path=campaign_path,
        expected_full_count=1,
        expected_canary_count=0,
    )

    record = report["records"][0]
    assert report["passed"] is False
    assert report["status"] == "blocked"
    assert record["status"] == "failed"
    assert record["quarantined"] is True
    assert "runtime_failed" in record["reason_codes"]
    assert report["quarantine_count"] == 1


def test_ingestion_uses_adaptive_rounds_instead_of_unsubmitted_static_tail(tmp_path: Path) -> None:
    force_grid = [0.0, 10.0]
    static_paths = [
        _candidate_manifest(
            root=tmp_path,
            candidate_id=f"emb-34um-final-gate-full-r0{round_index}-c001",
            gate="full_gate",
            round_index=round_index,
            yt=100.0 + round_index,
            kb=200.0 + round_index,
            force_grid=force_grid,
        )
        for round_index in (1, 2, 3)
    ]
    adaptive_paths = []
    for round_index in (2, 3):
        path = _candidate_manifest(
            root=tmp_path,
            candidate_id=f"emb-34um-final-gate-adaptive-r0{round_index}-c001",
            gate=f"adaptive_round_0{round_index}",
            round_index=round_index,
            yt=300.0 + round_index,
            kb=400.0 + round_index,
            force_grid=force_grid,
        )
        adaptive_paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload["normalized_payload"]["parameters"]
        _write_f_delta(
            Path(payload["normalized_payload"]["output_root"]) / "F_Delta.dat",
            yt=float(params["Yt"]),
            ka=float(params["ka"]),
            kb=float(params["kb"]),
            outputs=[1.0, 2.0],
            forces=force_grid,
        )
        _write_json(
            tmp_path / f"adaptive_round_0{round_index}" / "emb_34um_batch_summary.json",
            {"rendered_candidate_manifests": [str(path)]},
        )

    first_payload = json.loads(static_paths[0].read_text(encoding="utf-8"))
    first_params = first_payload["normalized_payload"]["parameters"]
    _write_f_delta(
        Path(first_payload["normalized_payload"]["output_root"]) / "F_Delta.dat",
        yt=float(first_params["Yt"]),
        ka=float(first_params["ka"]),
        kb=float(first_params["kb"]),
        outputs=[1.0, 2.0],
        forces=force_grid,
    )
    campaign_path = _campaign_manifest(tmp_path, {"full_gate": static_paths})

    _, report = build_emb_34um_final_gate_ingestion_report(
        campaign_manifest_path=campaign_path,
        expected_full_count=3,
        expected_canary_count=0,
    )

    assert report["passed"] is True
    assert report["status_counts"] == {"completed": 3, "failed": 0, "missing": 0, "partial": 0}
    ids = {record["candidate_id"] for record in report["records"]}
    assert "emb-34um-final-gate-full-r02-c001" not in ids
    assert "emb-34um-final-gate-adaptive-r02-c001" in ids
    assert "emb-34um-final-gate-adaptive-r03-c001" in ids


def test_write_emb_34um_final_gate_ingestion_artifacts_reports_missing_outputs(tmp_path: Path) -> None:
    candidate_path = _candidate_manifest(
        root=tmp_path,
        candidate_id="emb-34um-final-gate-full-r01-c001",
        gate="full_gate",
        round_index=1,
        yt=100.0,
        kb=200.0,
        force_grid=[0.0, 10.0],
    )
    campaign_path = _campaign_manifest(tmp_path, {"full_gate": [candidate_path]})

    artifacts = write_emb_34um_final_gate_ingestion_artifacts(
        campaign_manifest_path=campaign_path,
        output_root=tmp_path / "report",
        expected_full_count=1,
        expected_canary_count=0,
        include_plot=False,
    )

    assert artifacts.report_path == tmp_path / "report" / EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME
    assert artifacts.report_path.is_file()
    assert artifacts.summary_csv_path.is_file()
    assert artifacts.plot_path.is_file()
    assert artifacts.quarantine_path.is_file()

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["status_counts"]["missing"] == 1
    assert payload["passed"] is False
    assert "No completed F_Delta.dat rows were ingested" in " ".join(payload["blockers"])
    assert payload["plot_paths"]["ingestion_validation"] == str(artifacts.plot_path)


def test_ingestion_does_not_read_cwd_when_candidate_output_root_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_root = tmp_path / "campaign"
    candidate_path = tmp_path / "candidate_manifest.json"
    _write_json(
        candidate_path,
        {
            "candidate_id": "emb-34um-final-gate-full-r01-c001",
            "family": "emb",
            "active_learning_metadata": {"campaign": "full", "round": 1},
            "normalized_payload": {
                "candidate_id": "emb-34um-final-gate-full-r01-c001",
                "experiment": "indentation",
                "campaign_root": str(campaign_root / "full_gate"),
                "parameters": {
                    "Yt": 100.0,
                    "ka": 1.0,
                    "kb": 200.0,
                    "b1": 0.0,
                    "b2": 0.0,
                    "a3": 0.0,
                    "a4": 0.0,
                },
                "force_grid": [0.0, 10.0],
            },
            "rendered_payload": {"request_payload": {"force_grid": [0.0, 10.0]}},
        },
    )
    _write_f_delta(
        tmp_path / "F_Delta.dat",
        yt=100.0,
        ka=1.0,
        kb=200.0,
        outputs=[1.0, 2.0],
        forces=[0.0, 10.0],
    )
    monkeypatch.chdir(tmp_path)
    campaign_path = _campaign_manifest(campaign_root, {"full_gate": [candidate_path]})

    manifest, report = build_emb_34um_final_gate_ingestion_report(
        campaign_manifest_path=campaign_path,
        expected_full_count=1,
        expected_canary_count=0,
    )

    record = report["records"][0]
    assert manifest["status_counts"]["missing"] == 1
    assert record["status"] == "missing"
    assert record["f_delta_path"] == ""
    assert str(tmp_path / "F_Delta.dat") not in record["expected_f_delta_paths"]
