from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "workflows"
    / "emb"
    / "uq_emb"
    / "compare_forward_canaries.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_forward_canaries", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(
    site: str,
    root: str,
    prediction: float = 1.25,
    *,
    interior_prediction: float = 1.5,
) -> dict:
    return {
        "schema_version": "mesouq.uq_emb.forward_canary.v1",
        "status": "passed",
        "site": site,
        "agent": "sonovue",
        "device": "cuda",
        "wall_seconds": 1.0 if site == "karolina" else 2.0,
        "config_path": f"{root}/config.yaml",
        "config_sha256": site * 8,
        "config_semantic_sha256": "b" * 64,
        "materialization_receipt": f"{root}/config.materialization.json",
        "materialization_receipt_sha256": site * 16,
        "accepted_source_verification": {
            "status": "PASS",
            "root": f"{root}/accepted",
            "manifest": f"{root}/accepted.json",
            "manifest_sha256": "1" * 64,
            "file_count": 1,
        },
        "dependency_verification": {
            "status": "PASS",
            "root": f"{root}/dependencies",
            "manifest": f"{root}/dependencies.json",
            "manifest_sha256": "2" * 64,
            "file_count": 200,
            "logical_size_bytes": 67044794,
        },
        "provenance": {
            "git_commit": "c" * 40,
            "git_status_clean": True,
            "hostname": f"{site}.example",
            "python_executable": f"{root}/python",
            "python_version": "3.11" if site == "karolina" else "3.10",
            "slurm_job_id": "1" if site == "karolina" else "2",
            "site_runtime_root": root,
        },
        "datasets": [
            {
                "dataset_name": "indentation_3.2um",
                "shape": [1, 3],
                "prediction_min": prediction,
                "prediction_max": 2.0,
                "predictions": [[prediction, interior_prediction, 2.0]],
                "standard_deviation_min": 0.1,
                "standard_deviation_max": 0.3,
                "standard_deviations": [[0.1, 0.2, 0.3]],
                "reference_points": [1.0, 2.0, 3.0],
                "reference_data": [4.0, 5.0, 6.0],
                "reference_input_sha256": "e" * 64,
                "parameter_batch": [[1.0, 2.0, 3.0, 4.0]],
                "parameter_batch_sha256": "f" * 64,
                "path": f"{root}/model.pkl",
                "sha256": "a" * 64,
                "data_file": f"{root}/data.dat",
            }
        ],
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_comparison_ignores_only_location_and_timing_fields(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    _write(karolina, _receipt("karolina", "/scratch"))
    _write(vega, _receipt("vega", "/ceph"))

    report = module.compare_receipts(karolina, vega, expected_agent="sonovue")

    assert report["status"] == "passed"
    assert len(report["scientific_payload_sha256"]) == 64
    assert report["config_semantic_sha256"] == "b" * 64
    assert report["git_commit"] == "c" * 40


def test_comparison_rejects_scientific_difference(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    _write(karolina, _receipt("karolina", "/scratch"))
    _write(vega, _receipt("vega", "/ceph", prediction=1.5))

    with pytest.raises(ValueError, match="Full forward-canary predictions differ"):
        module.compare_receipts(karolina, vega, expected_agent="sonovue")


def test_comparison_rejects_interior_difference_with_same_extrema(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    _write(karolina, _receipt("karolina", "/scratch", interior_prediction=1.5))
    _write(vega, _receipt("vega", "/ceph", interior_prediction=1.75))

    with pytest.raises(ValueError, match="Full forward-canary predictions differ"):
        module.compare_receipts(karolina, vega, expected_agent="sonovue")


def test_comparison_rejects_semantically_different_materialized_configs(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    karolina_payload = _receipt("karolina", "/scratch")
    vega_payload = _receipt("vega", "/ceph")
    vega_payload["config_semantic_sha256"] = "d" * 64
    _write(karolina, karolina_payload)
    _write(vega, vega_payload)

    with pytest.raises(ValueError, match="semantic config digest"):
        module.compare_receipts(karolina, vega, expected_agent="sonovue")


def test_comparison_rejects_different_git_commits(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    karolina_payload = _receipt("karolina", "/scratch")
    vega_payload = _receipt("vega", "/ceph")
    vega_payload["provenance"]["git_commit"] = "d" * 40
    _write(karolina, karolina_payload)
    _write(vega, vega_payload)

    with pytest.raises(ValueError, match="same Git commit"):
        module.compare_receipts(karolina, vega, expected_agent="sonovue")
