from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
)


def _load_module() -> Any:
    script_path = (
        REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "analyze_emb_34um_dnn_causal_validation_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_emb_34um_dnn_causal_validation_pilot", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _runtime_fingerprint(*, radp: float, shell_th: float, bpress: float = EMB_34UM_DNN_CAUSAL_BPRESS_VALUE) -> dict[str, object]:
    lx = float(math.ceil(2.0 * radp + 6.0))
    lz = float(math.ceil(2.0 * radp + 10.0))
    return {
        "radp": radp,
        "shell_th": shell_th,
        "bpress": bpress,
        "Lx": lx,
        "Ly": lx,
        "Lz": lz,
        "L": lz,
    }


def _seed_candidate(
    *,
    stage_root: Path,
    index: int,
    result_kind: str = "completed",
    wrong_bpress: bool = False,
) -> Path:
    candidate_id = f"pilot-{index:03d}"
    output_root = stage_root / "emb" / candidate_id
    manifest_path = output_root / "dpd_sampling_candidate_manifest.json"
    radp = 6.45 + 0.01 * (index % 5)
    shell_th = 3.75e-9 + 0.025e-9 * (index % 7)
    bpress = -90.5 if wrong_bpress else EMB_34UM_DNN_CAUSAL_BPRESS_VALUE
    fingerprint = _runtime_fingerprint(radp=radp, shell_th=shell_th, bpress=bpress)
    parameters = {
        "ka": 1.0e3 + 75.0 * index,
        "kb": 2.0e3 + 90.0 * index,
        "radp": radp,
        "shell_th": shell_th,
        "bpress": bpress,
    }
    _write_json(
        manifest_path,
        {
            "candidate_id": candidate_id,
            "output_root": str(output_root),
            "normalized_payload": {
                "candidate_id": candidate_id,
                "output_root": str(output_root),
                "candidate_space": "d4",
                "parameters": parameters,
                "fingerprint": fingerprint,
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            },
            "rendered_payload": {
                "request_payload": {
                    "candidate_id": candidate_id,
                    "output_root": str(output_root),
                    "candidate_space": "d4",
                    "parameters": parameters,
                    "fingerprint": fingerprint,
                    "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                },
                "expected_hdf5_datasets": [
                    {
                        "dataset_id": f"{candidate_id}-dataset",
                        "hdf5_path": str(output_root / f"{candidate_id}.h5"),
                        "manifest_path": str(manifest_path),
                    }
                ],
            },
            "expected_hdf5_datasets": [
                {
                    "dataset_id": f"{candidate_id}-dataset",
                    "hdf5_path": str(output_root / f"{candidate_id}.h5"),
                    "manifest_path": str(manifest_path),
                }
            ],
        },
    )

    _write_json(
        output_root / "emb_34um_runtime_status.json",
        {
            "status": "completed" if result_kind in {"completed", "nonfinite"} else "running",
            "runtime_seconds": 100.0 + float(index),
        },
    )
    if result_kind == "completed":
        _write_json(
            output_root / "emb_34um_result.json",
            {
                "candidate_id": candidate_id,
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                "vertical_diameter": [1.0 + 0.01 * index + 0.1 * step for step, _ in enumerate(EMB_34UM_DNN_CAUSAL_FORCE_GRID)],
            },
        )
        (output_root / f"{candidate_id}.h5").write_bytes(b"hdf5")
    elif result_kind == "nonfinite":
        _write_json(
            output_root / "emb_34um_result.json",
            {
                "candidate_id": candidate_id,
                "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                "vertical_diameter": [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0],
            },
        )
        (output_root / f"{candidate_id}.h5").write_bytes(b"hdf5")
    return manifest_path


def test_pilot_analysis_writes_artifacts_and_reports_pass(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    stage_root = campaign_root / "pilot"
    manifests = [_seed_candidate(stage_root=stage_root, index=index) for index in range(EMB_34UM_DNN_CAUSAL_PILOT_SIZE)]
    _write_json(
        stage_root / "emb_34um_batch_summary.json",
        {
            "rendered_candidate_manifests": [str(path) for path in manifests],
            "expected_output_roots": [str(path.parent) for path in manifests],
        },
    )

    rc = module.main(["--campaign-root", str(campaign_root), "--no-plots"])
    assert rc == 0

    output_root = campaign_root / "pilot" / "validation"
    summary_path = output_root / module.SUMMARY_FILENAME
    rows_path = output_root / module.ROWS_FILENAME
    assert summary_path.is_file()
    assert rows_path.is_file()
    assert (output_root / module.RUNTIME_PNG_FILENAME).is_file()
    assert (output_root / module.RUNTIME_SIDECAR_FILENAME).is_file()
    assert (output_root / module.COVERAGE_PNG_FILENAME).is_file()
    assert (output_root / module.COVERAGE_SIDECAR_FILENAME).is_file()
    assert (output_root / module.FORCE_PNG_FILENAME).is_file()
    assert (output_root / module.FORCE_SIDECAR_FILENAME).is_file()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert payload["status_counts"]["total"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert payload["status_counts"]["completed"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert payload["missing_outputs"]["result_json_missing_count"] == 0
    assert payload["missing_outputs"]["hdf5_missing_count"] == 0
    assert payload["decision"]["passed"] is True
    assert payload["decision"]["blocked_reasons"] == []
    assert payload["promotion_readiness"]["passed"] is True
    assert payload["runtime_seconds_distribution_completed"]["count"] == EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    assert all(row["force_grid_count"] == 8 for row in payload["rows"])
    assert all(row["bounds_valid"] for row in payload["rows"])
    assert all(row["bpress"] == EMB_34UM_DNN_CAUSAL_BPRESS_VALUE for row in payload["rows"])


def test_pilot_analysis_reports_nonfinite_and_invalid_bpress_blockers(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    stage_root = campaign_root / "pilot"
    manifests = []
    for index in range(EMB_34UM_DNN_CAUSAL_PILOT_SIZE):
        if index == 0:
            manifests.append(_seed_candidate(stage_root=stage_root, index=index, result_kind="nonfinite"))
        elif index == 1:
            manifests.append(_seed_candidate(stage_root=stage_root, index=index, wrong_bpress=True))
        elif index == 2:
            manifests.append(_seed_candidate(stage_root=stage_root, index=index, result_kind="running"))
        else:
            manifests.append(_seed_candidate(stage_root=stage_root, index=index))
    batch_summary_path = stage_root / "emb_34um_batch_summary.json"
    _write_json(
        batch_summary_path,
        {
            "rendered_candidate_manifests": [str(path) for path in manifests],
            "expected_output_roots": [str(path.parent) for path in manifests],
        },
    )

    report = module.build_pilot_validation_report(batch_summary_path=batch_summary_path, generation_command="test")
    assert report["status_counts"]["nonfinite"] == 1
    assert report["status_counts"]["running"] == 1
    assert report["decision"]["passed"] is False
    assert report["promotion_readiness"]["passed"] is False
    blockers = "\n".join(report["decision"]["blocked_reasons"])
    assert "non-finite" in blockers
    assert "bpress" in blockers

    assert module.main(["--campaign-root", str(campaign_root), "--no-plots"]) == 1
    assert module.main(["--campaign-root", str(campaign_root), "--no-plots", "--allow-blocked"]) == 0


def test_pilot_analysis_can_enrich_timeout_from_slurm_state(tmp_path: Path) -> None:
    module = _load_module()
    campaign_root = tmp_path / "campaign"
    stage_root = campaign_root / "pilot"
    manifests = []
    for index in range(EMB_34UM_DNN_CAUSAL_PILOT_SIZE):
        result_kind = "running" if index == 0 else "completed"
        manifests.append(_seed_candidate(stage_root=stage_root, index=index, result_kind=result_kind))
    batch_summary_path = stage_root / "emb_34um_batch_summary.json"
    _write_json(
        batch_summary_path,
        {
            "rendered_candidate_manifests": [str(path) for path in manifests],
            "expected_output_roots": [str(path.parent) for path in manifests],
        },
    )

    report = module.build_pilot_validation_report(
        batch_summary_path=batch_summary_path,
        generation_command="test",
        slurm_states={0: {"job_id": "123_0", "state": "TIMEOUT", "elapsed": "00:30:23", "exit_code": "0:0"}},
    )

    first = report["rows"][0]
    assert first["status"] == "failed"
    assert first["runtime_status"] == "failed"
    assert first["slurm_state"] == "TIMEOUT"
    assert "slurm_timeout" in first["reasons"]
    assert report["status_counts"]["failed"] == 1
    assert report["decision"]["passed"] is False
    assert "failed at runtime" in "\n".join(report["decision"]["blocked_reasons"])


def test_sacct_parser_maps_only_array_parent_rows() -> None:
    module = _load_module()
    payload = """123_0|TIMEOUT|00:30:23|0:0
123_0.batch|CANCELLED|00:30:24|0:15
123_1|COMPLETED|00:02:54|0:0
123_[2-4]|PENDING|00:00:00|0:0
"""

    parsed = module._parse_sacct_rows(payload)

    assert parsed[0]["state"] == "TIMEOUT"
    assert parsed[0]["elapsed"] == "00:30:23"
    assert parsed[1]["state"] == "COMPLETED"
    assert 2 not in parsed
