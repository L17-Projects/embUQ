from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.references import build_gv_dpd_generated_reference_manifest
from meso_uq.references import build_gv_dpd_generated_fixture
from meso_uq.references import dpd_generated_fixture_from_runtime_manifest
from meso_uq.references import load_gv_dpd_generated_reference
from meso_uq.references import materialize_gv_dpd_generated_reference
from meso_uq.references import materialize_gv_dpd_generated_reference_dataset
from meso_uq.structures.gv.runtime.catalog import load_runtime_descriptor


def test_dpd_generated_gv_reference_manifest_tracks_identity_and_paths(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=tmp_path / "runs").to_manifest()
    data_path = tmp_path / "references" / "eigenmodes_reference.dat"
    data_path.parent.mkdir()
    data_path.write_text("x y\n0.0 1.0\n")

    manifest = load_gv_dpd_generated_reference(data_path, runtime_manifest=runtime_manifest).to_manifest()

    assert manifest["reference_kind"] == "dpd_generated"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["controls"] == {"bpress": -91.0}
    assert manifest["provenance"]["runtime_provenance_root"].endswith("gv_simulation_files/eigenmodes/gv")
    assert manifest["outputs"]["output_root"] == str((tmp_path / "runs").resolve())
    assert manifest["data_reference"]["path"] == str(data_path.resolve())
    assert manifest["data_reference"]["format"] == "dat"


def test_dpd_generated_gv_reference_missing_data_raises_actionable_error(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("stretching").plan(output_root=tmp_path).to_manifest()
    missing_path = tmp_path / "missing" / "stretching_reference.dat"

    with pytest.raises(FileNotFoundError, match="Generate the DPD reference artifact first or pass a valid existing path"):
        load_gv_dpd_generated_reference(missing_path, runtime_manifest=runtime_manifest)

    with pytest.raises(FileNotFoundError, match=runtime_manifest["dataset_id"]):
        load_gv_dpd_generated_reference(missing_path, runtime_manifest=runtime_manifest)


def test_dpd_generated_reference_manifest_wrapper_accepts_explicit_axes(tmp_path: Path) -> None:
    data_path = tmp_path / "torsion_reference"
    data_path.write_text("theta response\n0.0 0.1\n", encoding="utf-8")

    manifest = build_gv_dpd_generated_reference_manifest(
        data_path,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
    )

    assert manifest["reference_kind"] == "dpd_generated"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert manifest["data_reference"]["format"] == "unknown"
    assert manifest["data_reference"]["exists"] is True


def test_dpd_generated_materialization_writes_canonical_npz_and_manifest(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=tmp_path / "runs").to_manifest()
    source_data_path = tmp_path / "references" / "eigenmodes_reference.dat"
    source_data_path.parent.mkdir()
    source_data_path.write_text("0.0 1.0\n1.0 0.5\n", encoding="utf-8")

    result = materialize_gv_dpd_generated_reference(
        source_data_path,
        runtime_manifest=runtime_manifest,
        output_root=tmp_path / "materialized",
    )

    assert result.dataset_path.exists()
    assert result.manifest_path.exists()
    manifest = result.to_manifest()
    assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
    assert manifest["structure"] == "gv"
    assert manifest["reference_kind"] == "dpd_generated"
    assert manifest["artifacts"]["source_data"] == str(source_data_path.resolve())
    with np.load(result.dataset_path) as dataset:
        assert dataset["source_data_path"].item() == str(source_data_path.resolve())
        assert dataset["dataset_id"].item() == runtime_manifest["dataset_id"]
        assert dataset["data"].shape == (2, 2)


def test_dpd_generated_materialization_keeps_non_numeric_sources_as_metadata(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("eigenmodes").plan(output_root=tmp_path / "runs").to_manifest()
    source_data_path = tmp_path / "references" / "eigenmodes_reference.dat"
    source_data_path.parent.mkdir()
    source_data_path.write_text("mode response\nfirst second\n", encoding="utf-8")

    result = materialize_gv_dpd_generated_reference(
        source_data_path,
        runtime_manifest=runtime_manifest,
        output_root=tmp_path / "materialized",
    )

    with np.load(result.dataset_path) as dataset:
        assert dataset["source_data_path"].item() == str(source_data_path.resolve())
        assert dataset["dataset_id"].item() == runtime_manifest["dataset_id"]
        assert dataset["data_parse_status"].item() == "not_numeric_text"
        assert "manifest_json" in dataset.files


def test_dpd_generated_fixture_dataset_materialization_filters_records(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("torsion").plan(output_root=tmp_path / "runtime").to_manifest()
    runtime_fixture = dpd_generated_fixture_from_runtime_manifest(
        runtime_manifest,
        sigma=0.02,
        d0=1.0,
        points=(0.0, 1.0),
        values=(0.1, 0.2),
        metadata={"batch": "runtime"},
    )
    explicit_fixture = build_gv_dpd_generated_fixture(
        experiment="eigenmodes",
        geometry="gv_rad2_height14_28",
        controls={"bpress": -91.0},
        sigma=0.03,
        points=(1.0, 2.0),
        values=(0.3, 0.4),
    )

    result = materialize_gv_dpd_generated_reference_dataset(
        [runtime_fixture, explicit_fixture],
        data_path=tmp_path / "dpd_refs.npz",
        experiments=["torsion"],
    )

    manifest = result.to_manifest()
    assert manifest["entry_count"] == 1
    assert manifest["entries"][0]["experiment"] == "torsion"
    assert manifest["entries"][0]["nuisance_parameters"] == {"sigma": 0.02, "d0": 1.0}
    with np.load(result.dataset_path) as dataset:
        assert dataset["experiments"].tolist() == ["torsion"]
        assert dataset["point_counts"].tolist() == [2]
