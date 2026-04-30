from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dry_run.py"
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gv_dnn_smoke_consumes_runtime_manifest_and_writes_reloadable_artifacts(tmp_path: Path) -> None:
    runtime_module = _load_module(RUNTIME_SCRIPT_PATH, "mesouq_test_gv_runtime_stage")
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_stage")

    runtime_root = tmp_path / "runtime"
    rc = runtime_module.main(
        [
            "--experiment",
            "shear_flow",
            "--include-experimental",
            "--output-root",
            str(runtime_root),
        ]
    )
    assert rc == 0

    runtime_manifest_path = runtime_root / "gv_runtime_dry_run_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    assert runtime_manifest["experiment"] == "shear_flow"
    assert runtime_manifest["controls"]["ptan"] == pytest.approx(0.4)

    smoke_root = tmp_path / "smoke"
    rc = smoke_module.main(
        [
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--output-root",
            str(smoke_root),
            "--seed",
            "11",
            "--num-curves",
            "4",
            "--points-per-curve",
            "5",
            "--max-epoch",
            "6",
            "--width",
            "6",
            "--depth",
            "2",
            "--batch-size",
            "4",
        ]
    )
    assert rc == 0

    smoke_manifest_path = smoke_root / "gv_dnn_surrogate_smoke_manifest.json"
    smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    assert smoke_manifest["workflow"] == "gv_dnn_surrogate_smoke"
    assert smoke_manifest["surrogate_family"] == "dnn"
    assert smoke_manifest["input_manifests"]["runtime_dry_run"] == str(runtime_manifest_path.resolve())
    assert smoke_manifest["runtime_stage"]["runtime_package"] == "mirheoOBMD"
    assert smoke_manifest["controls"] == runtime_manifest["controls"]
    assert smoke_manifest["geometry"] == runtime_manifest["geometry"]
    assert smoke_manifest["synthetic_fixture"]["num_rows"] == 20
    assert smoke_manifest["training"]["report"]["n_train"] == 15
    assert smoke_manifest["training"]["report"]["n_val"] == 5
    assert smoke_manifest["reload"]["status"] == "passed"
    assert smoke_manifest["reload"]["passed_val_replay_tol_1e_6"] is True

    dataset_csv = Path(smoke_manifest["synthetic_fixture"]["dataset_csv"])
    split_csv = Path(smoke_manifest["synthetic_fixture"]["split_manifest"])
    artifact_path = Path(smoke_manifest["artifacts"]["artifact_path"])
    report_path = Path(smoke_manifest["artifacts"]["training_report"])
    assert dataset_csv.is_file()
    assert split_csv.is_file()
    assert artifact_path.is_file()
    assert report_path.is_file()

    with dataset_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "radius",
        "height",
        "ptan",
        "afsi",
        "bpress",
        "shear_coord",
        "shear_response",
        "source_curve_id",
    ]
    assert len({row["source_curve_id"] for row in rows}) == 4
    shear_coords = [float(row["shear_coord"]) for row in rows]
    assert min(shear_coords) == pytest.approx(0.0)
    assert max(shear_coords) == pytest.approx(1.0)


def test_gv_dnn_smoke_rejects_runtime_manifest_without_geometry_parameters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_bad_manifest")
    runtime_manifest_path = tmp_path / "bad_runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "shear_flow",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": {"parameters": {"height": 14.28}},
                "controls": {"ptan": 0.4, "afsi": 0.0, "bpress": -91.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        smoke_module.main(["--runtime-manifest", str(runtime_manifest_path), "--output-root", str(tmp_path / "out")])
    assert "geometry_spec.parameters.radius" in capsys.readouterr().err
