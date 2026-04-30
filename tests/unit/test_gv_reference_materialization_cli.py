from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.runtime.catalog import load_runtime_descriptor

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "materialize_gv_reference.py"
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_no_local_absolute_paths(payload: object, forbidden_root: Path) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    assert str(forbidden_root) not in rendered
    assert "/ceph/hpc/home/" not in rendered


def test_materialize_gv_reference_direct_cli_writes_portable_artifacts(tmp_path: Path) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_direct")
    repo_root = tmp_path / "repo"

    rc = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--selection",
            "gv:torsion",
            "--control",
            "theta=0.03",
            "--synthetic-point-count",
            "4",
        ]
    )

    assert rc == 0
    reference_manifest_path = (
        repo_root
        / "_runs/gv/torsion/gv_rad2_height14_28/theta_0.03/synthetic/reference_manifest.json"
    )
    reference_dataset_path = reference_manifest_path.with_name("reference_dataset.npz")
    runtime_manifest_path = reference_manifest_path.with_name("gv_runtime_dry_run_manifest.json")
    manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))

    assert runtime_manifest_path.is_file()
    assert manifest["workflow"] == "gv_reference_materialization"
    assert manifest["status"] == "experimental_reference_materialized"
    assert manifest["artifacts"]["reference_manifest"].startswith("_runs/gv/torsion/")
    assert manifest["artifacts"]["reference_dataset"].startswith("_runs/gv/torsion/")
    assert manifest["artifacts"]["runtime_manifest"].startswith("_runs/gv/torsion/")
    assert manifest["input_manifests"]["runtime_dry_run"].startswith("_runs/gv/torsion/")
    _assert_no_local_absolute_paths(manifest, tmp_path)

    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    assert runtime_manifest["runtime_package"] == "reference_materialization_only"
    with np.load(reference_dataset_path) as dataset:
        assert dataset["points"].shape == (4,)
        assert dataset["runtime_manifest"].item().startswith("_runs/gv/torsion/")


def test_materialize_gv_reference_source_manifest_copies_portable_inputs(tmp_path: Path) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_source")
    repo_root = tmp_path / "repo"
    source_manifest_path = tmp_path / "source_reference.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
                "points": [0.0, 1.0],
                "values": [0.2, 0.4],
            }
        ),
        encoding="utf-8",
    )

    rc = module.main(["--repo-root", str(repo_root), "--reference-manifest", str(source_manifest_path)])

    assert rc == 0
    reference_manifest_path = (
        repo_root
        / "_runs/gv/torsion/gv_rad2_height14_28/theta_0.03/synthetic/reference_manifest.json"
    )
    manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    source_copy = repo_root / manifest["artifacts"]["source_reference_manifest"]
    runtime_manifest = repo_root / manifest["artifacts"]["runtime_manifest"]

    assert source_copy.is_file()
    assert runtime_manifest.is_file()
    assert manifest["input_manifests"]["source_reference"].startswith("_runs/gv/torsion/")
    assert manifest["input_manifests"]["runtime_dry_run"].startswith("_runs/gv/torsion/")
    _assert_no_local_absolute_paths(manifest, tmp_path)
    _assert_no_local_absolute_paths(json.loads(source_copy.read_text(encoding="utf-8")), tmp_path)


def test_materialize_gv_reference_dpd_cli_packs_source_data(tmp_path: Path) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_dpd")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=repo_root / "runtime").to_manifest()
    runtime_manifest_path = tmp_path / "runtime_manifest.json"
    runtime_manifest_path.write_text(json.dumps(runtime_manifest), encoding="utf-8")
    source_data_path = repo_root / "gv_simulation_files/eigenmodes/reference.dat"
    source_data_path.parent.mkdir(parents=True)
    source_data_path.write_text("0.0 1.0\n1.0 0.5\n", encoding="utf-8")

    rc = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--reference-kind",
            "dpd_generated",
            "--dpd-data",
            f"{runtime_manifest['dataset_id']}={source_data_path}",
        ]
    )

    assert rc == 0
    reference_manifest_path = (
        repo_root
        / "_runs/gv/eigenmodes/gv_rad2_height14_28/bpress_-91/dpd_generated/reference_manifest.json"
    )
    manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["source_data"] == "gv_simulation_files/eigenmodes/reference.dat"
    assert manifest["data_reference"]["path"] == "gv_simulation_files/eigenmodes/reference.dat"
    _assert_no_local_absolute_paths(manifest, tmp_path)

    with np.load(reference_manifest_path.with_name("reference_dataset.npz")) as dataset:
        assert dataset["source_data_path"].item() == "gv_simulation_files/eigenmodes/reference.dat"
        assert np.array_equal(dataset["data"], np.asarray([[0.0, 1.0], [1.0, 0.5]]))


def test_gv_dnn_smoke_consumes_materialized_reference_stage(tmp_path: Path) -> None:
    materialize_module = _load_module(
        MATERIALIZE_SCRIPT_PATH,
        "mesouq_test_gv_reference_materialize_for_smoke",
    )
    smoke_module = _load_module(SMOKE_SCRIPT_PATH, "mesouq_test_gv_smoke_materialized_reference")
    repo_root = tmp_path / "repo"
    assert materialize_module.main(
        [
            "--repo-root",
            str(repo_root),
            "--selection",
            "gv:torsion",
            "--control",
            "theta=0.03",
            "--synthetic-point-count",
            "3",
        ]
    ) == 0
    materialized_manifest = (
        repo_root
        / "_runs/gv/torsion/gv_rad2_height14_28/theta_0.03/synthetic/reference_manifest.json"
    )
    materialized_dataset = materialized_manifest.with_name("reference_dataset.npz")
    smoke_root = tmp_path / "smoke"

    rc = smoke_module.main(
        [
            "--reference-manifest",
            str(materialized_manifest),
            "--output-root",
            str(smoke_root),
            "--num-curves",
            "2",
            "--points-per-curve",
            "2",
            "--max-epoch",
            "1",
            "--width",
            "2",
            "--depth",
            "1",
            "--batch-size",
            "2",
            "--val-fraction",
            "0.5",
        ]
    )

    assert rc == 0
    smoke_manifest = json.loads((smoke_root / "gv_dnn_surrogate_smoke_manifest.json").read_text(encoding="utf-8"))
    assert smoke_manifest["input_manifests"]["reference_materialization"] == str(materialized_manifest.resolve())
    assert smoke_manifest["reference_stage"]["status"] == "available"
    assert smoke_manifest["reference_stage"]["reference_dataset"] == str(materialized_dataset.resolve())
    assert smoke_manifest["artifacts"]["reference_dataset"] == str(materialized_dataset.resolve())


def test_materialize_gv_reference_helper_validation_paths(tmp_path: Path) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_helpers")
    structure = module.get_structure("gv")
    runtime_path = tmp_path / "runtime.json"
    reference_path = tmp_path / "reference.json"

    runtime_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        module._load_json(runtime_path, label="runtime manifest")

    with pytest.raises(ValueError, match="Runtime manifest .* missing required fields"):
        module._normalize_runtime_manifest({"structure": "gv"}, path=runtime_path)

    with pytest.raises(ValueError, match="structure='gv'"):
        module._normalize_runtime_manifest(
            {"structure": "emb", "experiment": "torsion", "geometry": "gv_rad2_height14_28", "controls": {}},
            path=runtime_path,
        )

    with pytest.raises(ValueError, match="controls must be a mapping"):
        module._normalize_runtime_manifest(
            {"structure": "gv", "experiment": "torsion", "geometry": "gv_rad2_height14_28", "controls": []},
            path=runtime_path,
        )

    with pytest.raises(ValueError, match="Reference manifest .* missing required fields"):
        module._normalize_reference_manifest({"structure": "gv"}, path=reference_path)

    with pytest.raises(ValueError, match="structure='gv'"):
        module._normalize_reference_manifest(
            {
                "structure": "emb",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "synthetic",
            },
            path=reference_path,
        )

    with pytest.raises(ValueError, match="controls must be a mapping"):
        module._normalize_reference_manifest(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": [],
                "reference_kind": "synthetic",
            },
            path=reference_path,
        )

    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        module._normalize_reference_manifest(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
                "reference_kind": "real",
            },
            path=reference_path,
        )

    args = SimpleNamespace(experiment="torsion", selection=["gv:torsion", "gv:eigenmodes"])
    assert module._resolve_selected_experiments(args) == ["torsion", "eigenmodes"]
    with pytest.raises(ValueError, match="form gv:<experiment>"):
        module._resolve_selected_experiments(SimpleNamespace(experiment=None, selection=["torsion"]))
    with pytest.raises(ValueError, match="does not accept structure"):
        module._resolve_selected_experiments(SimpleNamespace(experiment=None, selection=["emb:compression"]))

    with pytest.raises(ValueError, match="Invalid control override"):
        module._parse_controls(["theta"], "torsion", structure)
    with pytest.raises(ValueError, match="Unsupported control"):
        module._parse_controls(["bpress=-91"], "torsion", structure)
    with pytest.raises(ValueError, match="Missing GV controls"):
        module._parse_controls(["tot_force=500"], "stretching", structure)
    with pytest.raises(ValueError, match="Invalid geometry"):
        module._parse_custom_geometry("2.0")

    geometries = module._resolve_geometries(
        SimpleNamespace(geometry_id=["gv_rad2_height14_28"], geometry=["2.5,12.0"]),
        structure,
    )
    assert [geometry.id for geometry in geometries] == ["gv_rad2_height14_28", "gv_rad2_5_height12"]

    with pytest.raises(ValueError, match="Expected DATASET_ID=PATH"):
        module._dpd_data_map(["missing_separator"])

    with pytest.raises(ValueError, match="Provide at least one"):
        module._build_requests(module.build_parser().parse_args(["--repo-root", str(tmp_path)]))
    with pytest.raises(ValueError, match="exactly one experiment"):
        module._build_requests(
            module.build_parser().parse_args(
                [
                    "--repo-root",
                    str(tmp_path),
                    "--selection",
                    "gv:torsion",
                    "--selection",
                    "gv:eigenmodes",
                    "--control",
                    "theta=0.03",
                ]
            )
        )


def test_materialize_gv_reference_path_and_payload_helpers(tmp_path: Path) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_payloads")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    inside = repo_root / "data" / "reference.dat"
    outside = tmp_path / "external.dat"
    inside.parent.mkdir()
    inside.write_text("1 2\n", encoding="utf-8")
    outside.write_text("not numeric\n", encoding="utf-8")

    assert module._portable_path(repo_root, inside) == "data/reference.dat"
    assert module._portable_path(repo_root, outside) == "external.dat"
    assert module._sanitize_manifest_paths({"p": inside, "items": (outside, "/plain/path")}, repo_root=repo_root) == {
        "p": "data/reference.dat",
        "items": ["external.dat", "path"],
    }
    assert module._resolve_existing_path(None, repo_root=repo_root) is None
    assert module._resolve_existing_path("data/reference.dat", repo_root=repo_root) == inside.resolve()
    assert module._write_source_reference_manifest_artifact(None, destination=repo_root / "unused.json", repo_root=repo_root) is None

    reference_manifest = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "geometry_spec": {"parameters": {"radius": 2.0, "height": 14.28}},
        "control_id": "theta_0.03",
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
        "reference_kind": "dpd_generated",
        "controls": {"theta": 0.03},
        "points": [0.0, 1.0],
        "values": [0.2, 0.4],
        "observable": "torsion_response",
        "observable_names": ["torsion_response"],
        "data_reference": {"path": "data/reference.dat", "format": "dat"},
    }
    payload = module._materialized_dataset_payload(
        reference_manifest,
        repo_root=repo_root,
        source_data_path=inside,
        source_reference_manifest_path=repo_root / "source_reference_manifest.json",
        runtime_manifest_path=repo_root / "runtime_manifest.json",
    )
    assert payload["data"].shape == (1, 2)
    assert payload["source_data_path"].item() == "data/reference.dat"
    assert payload["observable"].item() == "torsion_response"

    non_numeric_payload = module._materialized_dataset_payload(
        reference_manifest,
        repo_root=repo_root,
        source_data_path=outside,
        source_reference_manifest_path=None,
        runtime_manifest_path=None,
    )
    assert non_numeric_payload["data_parse_status"].item() == "not_numeric_text"

    missing_payload = module._materialized_dataset_payload(
        reference_manifest,
        repo_root=repo_root,
        source_data_path=repo_root / "missing.dat",
        source_reference_manifest_path=None,
        runtime_manifest_path=None,
    )
    assert missing_payload["data_parse_status"].item() == "missing_source_data"

    runtime_manifest = {"structure": "gv", "experiment": "torsion", "geometry": "gv_rad2_height14_28", "controls": {"theta": 0.03}}
    context = module.resolve_gv_reference_context(runtime_manifest=runtime_manifest)
    runtime_destination = repo_root / "runtime_manifest.json"
    request = module.MaterializationRequest(
        runtime_manifest=None,
        runtime_manifest_path=None,
        source_reference_manifest=None,
        source_reference_manifest_path=None,
        experiment="torsion",
        geometry=context.geometry,
        controls=context.controls,
        reference_kind="synthetic",
    )
    module._write_runtime_manifest_artifact(
        request=request,
        context=context,
        source_manifest={"runtime_manifest": runtime_manifest},
        destination=runtime_destination,
        repo_root=repo_root,
    )
    assert json.loads(runtime_destination.read_text(encoding="utf-8"))["experiment"] == "torsion"


def test_materialize_gv_reference_reports_actionable_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module(MATERIALIZE_SCRIPT_PATH, "mesouq_test_gv_reference_materialize_errors")
    repo_root = tmp_path / "repo"
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=repo_root / "runtime").to_manifest()
    runtime_manifest_path = tmp_path / "runtime_manifest.json"
    runtime_manifest_path.write_text(json.dumps(runtime_manifest), encoding="utf-8")

    with pytest.raises(SystemExit):
        module.main(
            [
                "--repo-root",
                str(repo_root),
                "--runtime-manifest",
                str(runtime_manifest_path),
                "--reference-kind",
                "dpd_generated",
            ]
        )
    assert "DPD materialization requires --dpd-data" in capsys.readouterr().err
