from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_gv_dnn_smoke_workflow_generates_synthetic_reference_and_model(tmp_path: Path) -> None:
    module = _load_module(
        REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py",
        "gv_dnn_smoke_workflow_test",
    )
    output_root = tmp_path / "gv_dnn_smoke"

    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--experiment",
            "torsion",
            "--control",
            "theta=0.03",
            "--max-epoch",
            "4",
            "--width",
            "6",
            "--depth",
            "2",
            "--batch-size",
            "4",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output_root / "gv_dnn_surrogate_smoke_report.json").read_text(encoding="utf-8"))
    reference = json.loads((output_root / "gv_reference_manifest.json").read_text(encoding="utf-8"))
    with (output_root / "gv_surrogate_smoke_dataset.csv").open(encoding="utf-8", newline="") as handle:
        dataset_rows = list(csv.DictReader(handle))

    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "torsion"
    assert manifest["geometry"].startswith("gv_rad")
    assert manifest["controls"] == {"theta": 0.03}
    assert manifest["reference_kind"] == "synthetic"
    assert manifest["backend"] == "dnn"
    assert report["status"] in {"passed", "dry-run"}
    assert report["backend"] == "dnn"
    if report["status"] == "passed":
        assert report["reload_validation"]["val_rmse_phys"] >= 0.0
        assert Path(manifest["artifacts"]["model_path"]).is_file()
    else:
        assert report["execution_mode"] == "dry_run_manifest_only"
        assert report["training"]["missing_dependency"] == "torch"
    assert report["row_count"] == len(dataset_rows)
    assert reference["runtime_manifest"]["structure"] == "gv"
    assert reference["runtime_manifest"]["experiment"] == "torsion"
    assert reference["runtime_manifest"]["controls"] == {"theta": 0.03}
    assert Path(manifest["artifacts"]["training_report"]).is_file()
    assert "required_reference_manifest_fields" in report["upstream_contract"]


def test_gv_dnn_smoke_workflow_consumes_synthetic_reference_manifest(tmp_path: Path) -> None:
    module = _load_module(
        REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py",
        "gv_dnn_smoke_workflow_external_manifest_test",
    )
    output_root = tmp_path / "gv_dnn_smoke_external"
    reference_manifest_path = tmp_path / "synthetic_reference_manifest.json"
    reference_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "eigenmodes",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": {
                    "id": "gv_rad2_height14_28",
                    "parameters": {"radius": 2.0, "height": 14.28},
                    "label": "GV radius 2.0 height 14.28",
                    "source": "test",
                },
                "controls": {"bpress": -91.0},
                "reference_kind": "synthetic",
                "dataset_id": "gv:eigenmodes:gv_rad2_height14_28:bpress_-91",
                "notes": ["synthetic reference emitted by test seam"],
            }
        ),
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--output-root",
            str(output_root),
            "--reference-manifest",
            str(reference_manifest_path),
            "--max-epoch",
            "3",
            "--width",
            "5",
            "--depth",
            "2",
            "--batch-size",
            "4",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output_root / "gv_dnn_surrogate_smoke_report.json").read_text(encoding="utf-8"))
    normalized_reference = json.loads((output_root / "gv_reference_manifest.json").read_text(encoding="utf-8"))

    assert manifest["experiment"] == "eigenmodes"
    assert manifest["controls"] == {"bpress": -91.0}
    assert manifest["reference_kind"] == "synthetic"
    assert manifest["backend"] == "dnn"
    assert report["experiment"] == "eigenmodes"
    assert report["reference_kind"] == "synthetic"
    assert report["training"]["n_train"] > 0
    assert report["validation_row_count"] > 0
    assert normalized_reference["backend"] == "dnn"
    assert normalized_reference["upstream_contract"]["catalog_seam"]
    if report["status"] == "dry-run":
        assert report["training"]["missing_dependency"] == "torch"
