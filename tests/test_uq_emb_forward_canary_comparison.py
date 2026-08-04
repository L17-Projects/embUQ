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


def _receipt(site: str, root: str, prediction: float = 1.25) -> dict:
    return {
        "schema_version": "mesouq.uq_emb.forward_canary.v1",
        "status": "passed",
        "site": site,
        "agent": "sonovue",
        "device": "cuda",
        "wall_seconds": 1.0 if site == "karolina" else 2.0,
        "config_path": f"{root}/config.yaml",
        "config_sha256": site * 8,
        "datasets": [
            {
                "prediction_min": prediction,
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


def test_comparison_rejects_scientific_difference(tmp_path: Path) -> None:
    module = _load_module()
    karolina = tmp_path / "karolina.json"
    vega = tmp_path / "vega.json"
    _write(karolina, _receipt("karolina", "/scratch"))
    _write(vega, _receipt("vega", "/ceph", prediction=1.5))

    with pytest.raises(ValueError, match="payloads differ"):
        module.compare_receipts(karolina, vega, expected_agent="sonovue")
