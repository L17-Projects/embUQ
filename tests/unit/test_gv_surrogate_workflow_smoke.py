from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    if smoke_module.torch is None:
        assert smoke_manifest["execution_mode"] == "dry_run_manifest_only"
        assert smoke_manifest["reload"]["status"] == "not_run"
        assert smoke_manifest["artifacts"]["artifact_path"] is None
        assert smoke_manifest["training"]["report"]["status"] == "skipped_missing_dependency"
        return
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


def test_gv_dnn_smoke_helper_validation_branches(tmp_path: Path) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_validation_branches")
    structure = smoke_module.get_structure("gv")

    with pytest.raises(ValueError, match="Invalid control override"):
        smoke_module._parse_controls(["theta"], "torsion", structure)

    with pytest.raises(ValueError, match="Unsupported control"):
        smoke_module._parse_controls(["unknown=1.0"], "torsion", structure)

    geometry = smoke_module._resolve_geometry(
        SimpleNamespace(geometry_id="gv_rad2_height14_28", radius=None, height=None),
        structure,
    )
    assert geometry.id == "gv_rad2_height14_28"

    with pytest.raises(ValueError, match="requires both --radius and --height"):
        smoke_module._resolve_geometry(SimpleNamespace(geometry_id=None, radius=2.1, height=None), structure)

    custom_geometry = smoke_module._resolve_geometry(
        SimpleNamespace(geometry_id=None, radius=2.2, height=15.0),
        structure,
    )
    assert custom_geometry.parameters == {"radius": 2.2, "height": 15.0}

    with pytest.raises(ValueError, match="missing geometry_spec"):
        smoke_module._validate_geometry_parameters({"geometry_spec": None})

    with pytest.raises(ValueError, match="missing geometry_spec.parameters"):
        smoke_module._validate_geometry_parameters({"geometry_spec": {"parameters": None}})

    with pytest.raises(ValueError, match="at least 2"):
        smoke_module._axis_points(1)

    with pytest.raises(ValueError, match="at least 2"):
        smoke_module._curve_offsets(1)

    with pytest.raises(ValueError, match="Need at least 2 rows"):
        smoke_module._split_row_indices(1, val_fraction=0.5, seed=1)

    with pytest.raises(ValueError, match="val_fraction must be"):
        smoke_module._split_row_indices(4, val_fraction=1.0, seed=1)

    perm, n_val = smoke_module._split_row_indices(2, val_fraction=0.99, seed=None)
    assert sorted(perm.tolist()) == [0, 1]
    assert n_val == 1

    assert smoke_module._control_columns({"experiment": "not_registered", "controls": {"z": 1.0, "a": 2.0}}) == [
        "a",
        "z",
    ]

    args = SimpleNamespace(
        runtime_manifest=None,
        reference_manifest=None,
        experiment="torsion",
        structure="gv",
        geometry_id=None,
        radius=2.2,
        height=15.0,
        control=["theta=0.04"],
        output_root=str(tmp_path),
        include_experimental=False,
        reference_kind="dpd_generated",
        backend="dnn",
    )
    manifest = smoke_module._build_reference_manifest(args)
    assert manifest["geometry_spec"]["parameters"] == {"radius": 2.2, "height": 15.0}
    assert manifest["controls"] == {"theta": 0.04}
    assert manifest["reference_kind"] == "dpd_generated"

    with pytest.raises(ValueError, match="--experiment is required"):
        smoke_module._build_reference_manifest(
            SimpleNamespace(
                runtime_manifest=None,
                reference_manifest=None,
                experiment=None,
                structure="gv",
                geometry_id=None,
                radius=None,
                height=None,
                control=[],
                output_root=str(tmp_path),
                include_experimental=False,
                reference_kind="synthetic",
                backend="dnn",
            )
        )


def test_gv_dnn_smoke_manifest_loaders_report_invalid_inputs(tmp_path: Path) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_manifest_loader_errors")
    valid_geometry_spec = {
        "id": "gv_rad2_height14_28",
        "parameters": {"radius": 2.0, "height": 14.28},
    }

    runtime_missing = tmp_path / "runtime_missing.json"
    runtime_missing.write_text(json.dumps({"structure": "gv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        smoke_module._load_runtime_manifest(runtime_missing)

    runtime_wrong_structure = tmp_path / "runtime_wrong_structure.json"
    runtime_wrong_structure.write_text(
        json.dumps(
            {
                "structure": "emb",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": valid_geometry_spec,
                "controls": {"theta": 0.03},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="structure='gv'"):
        smoke_module._load_runtime_manifest(runtime_wrong_structure)

    runtime_bad_controls = tmp_path / "runtime_bad_controls.json"
    runtime_bad_controls.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": valid_geometry_spec,
                "controls": ["theta"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controls must be a mapping"):
        smoke_module._load_runtime_manifest(runtime_bad_controls)

    reference_missing = tmp_path / "reference_missing.json"
    reference_missing.write_text(json.dumps({"structure": "gv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        smoke_module._load_reference_manifest(reference_missing)

    reference_wrong_structure = tmp_path / "reference_wrong_structure.json"
    reference_wrong_structure.write_text(
        json.dumps(
            {
                "structure": "emb",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="structure='gv'"):
        smoke_module._load_reference_manifest(reference_wrong_structure)

    reference_bad_controls = tmp_path / "reference_bad_controls.json"
    reference_bad_controls.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": ["theta"],
                "reference_kind": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controls must be a mapping"):
        smoke_module._load_reference_manifest(reference_bad_controls)

    reference_bad_kind = tmp_path / "reference_bad_kind.json"
    reference_bad_kind.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "real",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        smoke_module._load_reference_manifest(reference_bad_kind)

    with pytest.raises(ValueError, match="Use either --runtime-manifest or --reference-manifest"):
        smoke_module._build_reference_manifest(
            SimpleNamespace(
                runtime_manifest=str(runtime_bad_controls),
                reference_manifest=str(reference_bad_kind),
                reference_kind="synthetic",
                backend="dnn",
            )
        )


def test_gv_dnn_smoke_dry_run_mode_when_torch_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_no_torch")
    monkeypatch.setattr(smoke_module, "torch", None)

    reference_manifest_path = tmp_path / "reference_manifest.json"
    reference_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "geometry_spec": {
                    "id": "gv_rad2_height14_28",
                    "parameters": {"radius": 2.0, "height": 14.28},
                },
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    rc = smoke_module.main(
        [
            "--reference-manifest",
            str(reference_manifest_path),
            "--output-root",
            str(output_root),
            "--num-curves",
            "2",
            "--points-per-curve",
            "2",
            "--val-fraction",
            "0.5",
        ]
    )
    assert rc == 0

    report = json.loads((output_root / "gv_dnn_surrogate_smoke_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry-run"
    assert report["execution_mode"] == "dry_run_manifest_only"
    assert report["training"]["n_train"] == 2
    assert report["training"]["n_val"] == 2
    assert manifest["reload"]["status"] == "not_run"
    assert manifest["artifacts"]["artifact_path"] is None

    with pytest.raises(RuntimeError, match="torch is required"):
        smoke_module._train_smoke_surrogate(
            [],
            input_cols=[],
            target_col="response",
            out_path=tmp_path / "model.pkl",
            width=2,
            depth=1,
            batch_size=1,
            max_epoch=1,
            seed=1,
            val_fraction=0.5,
            report_path=tmp_path / "report.json",
        )

    with pytest.raises(RuntimeError, match="torch is required"):
        smoke_module._predict_rows(tmp_path / "model.pkl", [], input_cols=[], target_col="response")


def test_gv_dnn_smoke_validation_helpers_cover_error_paths(tmp_path: Path) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_validation_helpers")
    structure = smoke_module.get_structure("gv")

    with pytest.raises(ValueError, match="Invalid control override"):
        smoke_module._parse_controls(["theta"], "torsion", structure)

    with pytest.raises(ValueError, match="Unsupported control"):
        smoke_module._parse_controls(["bpress=-91"], "torsion", structure)

    args = smoke_module.build_parser().parse_args(
        ["--output-root", str(tmp_path), "--experiment", "torsion", "--radius", "2.0"]
    )
    with pytest.raises(ValueError, match="Custom geometry requires both"):
        smoke_module._resolve_geometry(args, structure)

    bad_runtime = tmp_path / "bad_runtime.json"
    bad_runtime.write_text(
        json.dumps({"structure": "emb", "experiment": "torsion", "geometry": "gv_rad2_height14_28", "controls": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only supports structure='gv'"):
        smoke_module._load_runtime_manifest(bad_runtime)

    bad_reference = tmp_path / "bad_reference.json"
    bad_reference.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": [],
                "reference_kind": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controls must be a mapping"):
        smoke_module._load_reference_manifest(bad_reference)

    with pytest.raises(ValueError, match="Use either --runtime-manifest or --reference-manifest"):
        smoke_module._build_reference_manifest(
            smoke_module.build_parser().parse_args(
                [
                    "--output-root",
                    str(tmp_path),
                    "--runtime-manifest",
                    str(bad_runtime),
                    "--reference-manifest",
                    str(bad_reference),
                ]
            )
        )

    with pytest.raises(ValueError, match="--experiment is required"):
        smoke_module._build_reference_manifest(smoke_module.build_parser().parse_args(["--output-root", str(tmp_path)]))


def test_gv_dnn_smoke_rows_and_split_helpers_cover_edges(tmp_path: Path) -> None:
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_dnn_smoke_row_helpers")
    reference_manifest = {
        "structure": "gv",
        "experiment": "unknown_experiment",
        "geometry": "gv_rad2_height14_28",
        "geometry_spec": {"parameters": {"radius": 1.5, "height": 9.0}},
        "controls": {"zeta": 2.0},
        "reference_kind": "synthetic",
    }

    assert smoke_module._control_columns(reference_manifest) == ["zeta"]
    rows = smoke_module._build_smoke_rows(reference_manifest, num_curves=2, points_per_curve=2)
    assert len(rows) == 4
    assert rows[0]["radius"] == 1.5
    assert rows[0]["height"] == 9.0
    assert rows[0]["response"] > 0.0

    csv_path = tmp_path / "dataset.csv"
    smoke_module._write_dataset_csv(csv_path, rows, smoke_module._ordered_dataset_columns(reference_manifest))
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].endswith("response,source_curve_id")

    split_path = tmp_path / "split.csv"
    assert smoke_module._write_split_manifest_csv(split_path, n_rows=3, val_fraction=0.99, seed=None) == 2

    with pytest.raises(ValueError, match="at least 2 rows"):
        smoke_module._split_row_indices(1, val_fraction=0.5, seed=1)

    with pytest.raises(ValueError, match="val_fraction"):
        smoke_module._split_row_indices(3, val_fraction=1.0, seed=1)

    if smoke_module.torch is None:
        with pytest.raises(RuntimeError, match="torch is required"):
            smoke_module._train_smoke_surrogate(
                rows,
                input_cols=["ka", "radius", "height", "zeta", "observable_axis"],
                target_col="response",
                out_path=tmp_path / "model.pkl",
                width=2,
                depth=1,
                batch_size=2,
                max_epoch=1,
                seed=1,
                val_fraction=0.5,
                report_path=tmp_path / "report.json",
            )
        with pytest.raises(RuntimeError, match="torch is required"):
            smoke_module._predict_rows(
                tmp_path / "model.pkl",
                rows,
                input_cols=["ka", "radius", "height", "zeta", "observable_axis"],
                target_col="response",
            )
