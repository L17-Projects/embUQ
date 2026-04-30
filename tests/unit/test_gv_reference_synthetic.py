from __future__ import annotations

import json
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.references import build_gv_synthetic_fixture
from meso_uq.references import build_gv_synthetic_reference_manifest
from meso_uq.references import generate_gv_synthetic_reference
from meso_uq.references import materialize_gv_synthetic_reference_dataset
from meso_uq.references import gv_common as gv_common_module
from meso_uq.references import synthetic as synthetic_module
from meso_uq.references.gv_common import resolve_gv_reference_context
from meso_uq.references.gv_common import materialize_reference_records
from meso_uq.structures.gv.parameters import GV_PARAMETER_CONTRACT
from meso_uq.structures.gv.runtime.catalog import load_runtime_descriptor


def _require_public_materializer(module: object, public_name: str):
    materializer = getattr(module, public_name, None)
    if materializer is None:
        pytest.xfail(
            f"Expected public API `{module.__name__}.{public_name}` to materialize GV synthetic references to .npz + JSON."
        )
    return materializer


def _materialized_paths(result: Any) -> tuple[Path, Path]:
    if isinstance(result, Mapping):
        dataset_value = result.get("dataset_path")
        manifest_value = result.get("manifest_path")
    else:
        dataset_value = getattr(result, "dataset_path", None)
        manifest_value = getattr(result, "manifest_path", None)
    if not dataset_value or not manifest_value:
        pytest.fail("Materialized GV synthetic reference must expose `dataset_path` and `manifest_path`.")
    return Path(str(dataset_value)), Path(str(manifest_value))


def test_synthetic_gv_reference_is_deterministic_for_seed(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("stretching").plan(output_root=tmp_path).to_manifest()

    reference_a = generate_gv_synthetic_reference(seed=17, runtime_manifest=runtime_manifest)
    reference_b = generate_gv_synthetic_reference(seed=17, runtime_manifest=runtime_manifest)
    reference_c = generate_gv_synthetic_reference(seed=18, runtime_manifest=runtime_manifest)

    assert reference_a.points == reference_b.points
    assert reference_a.values == reference_b.values
    assert reference_a.to_manifest()["values"] == reference_b.to_manifest()["values"]
    assert reference_a.values != reference_c.values


def test_synthetic_gv_reference_manifest_includes_contract_and_runtime_metadata(tmp_path: Path) -> None:
    runtime_manifest = load_runtime_descriptor("torsion").plan(output_root=tmp_path).to_manifest()

    manifest = generate_gv_synthetic_reference(seed=5, runtime_manifest=runtime_manifest).to_manifest()

    assert manifest["reference_kind"] == "synthetic"
    assert manifest["surrogate_backend"] == "dnn"
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "torsion"
    assert manifest["geometry"].startswith("gv_rad")
    assert manifest["controls"] == {"theta": 0.01}
    assert manifest["calibrated_parameter_names"] == list(GV_PARAMETER_CONTRACT.calibrated_names)
    assert manifest["noise_model"]["kind"] == "multiplicative"
    assert manifest["provenance"]["runtime_provenance_root"].endswith("gv_simulation_files/torsion/gv")
    assert manifest["outputs"]["output_root"] == str(tmp_path.resolve())
    assert manifest["outputs"]["work_dir"].endswith("/gv/torsion/gv_rad2_height14_28/theta_0_01_0_1")
    assert len(manifest["points"]) == 64
    assert len(manifest["values"]) == 64


def test_synthetic_reference_can_resolve_explicit_axes_without_manifest() -> None:
    reference = generate_gv_synthetic_reference(
        seed=2,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=3,
    )
    manifest = reference.to_manifest()

    assert manifest["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert manifest["control_id"] == "theta_0.03"
    assert manifest["points"] == [0.0, 0.5, 1.0]


def test_synthetic_reference_context_reports_missing_axes_and_controls() -> None:
    with pytest.raises(ValueError, match="require an experiment name"):
        resolve_gv_reference_context()

    with pytest.raises(ValueError, match="require a geometry id"):
        resolve_gv_reference_context(experiment="torsion")

    with pytest.raises(ValueError, match="require controls"):
        resolve_gv_reference_context(experiment="torsion", geometry="gv_rad2_height14_28")

    with pytest.raises(ValueError, match="Unknown GV controls"):
        resolve_gv_reference_context(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03, "tot_force": 1.0},
        )

    with pytest.raises(ValueError, match="Missing GV controls"):
        resolve_gv_reference_context(
            experiment="stretching",
            geometry="gv_rad2_height14_28",
            controls={"tot_force": 500.0},
        )

    with pytest.raises(ValueError, match="at least two sample points"):
        generate_gv_synthetic_reference(
            seed=1,
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            point_count=1,
        )


def test_synthetic_reference_context_rejects_non_gv_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    class NonGVStructure:
        name = "emb"

    monkeypatch.setattr(gv_common_module, "get_structure", lambda _name: NonGVStructure())

    with pytest.raises(ValueError, match="only support structure 'gv'"):
        gv_common_module.resolve_gv_reference_context(
            runtime_manifest={
                "structure": "emb",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": {"theta": 0.03},
            }
        )


def test_synthetic_reference_context_accepts_custom_geometry_specs() -> None:
    context = resolve_gv_reference_context(
        experiment="torsion",
        geometry={
            "id": "gv_custom",
            "label": "custom GV",
            "shape": "cylinder",
            "parameters": {"radius": 2.5, "height": 12.0},
            "source": "test",
        },
        controls={"theta": 0.03},
    )

    assert context.geometry.id == "gv_custom"
    assert context.geometry.parameters == {"radius": 2.5, "height": 12.0}

    with pytest.raises(ValueError, match="must include an id"):
        gv_common_module._geometry_spec_from_mapping({"parameters": {}})

    with pytest.raises(ValueError, match="parameters must be a mapping"):
        resolve_gv_reference_context(
            experiment="torsion",
            geometry={"id": "gv_custom", "parameters": []},
            controls={"theta": 0.03},
        )


def test_synthetic_reference_context_merges_manifest_and_explicit_controls() -> None:
    context = resolve_gv_reference_context(
        runtime_manifest={
            "structure": "gv",
            "experiment": "stretching",
            "geometry": "gv_rad2_height14_28",
            "controls": {"tot_force": 500.0},
        },
        controls={"bpress": -91.0},
    )

    assert context.controls == {"tot_force": 500.0, "bpress": -91.0}
    assert context.control_id == "bpress_-91__tot_force_500"
    assert context.dataset_id == "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_500"


def test_synthetic_fixture_supports_multiple_geometries_immediately() -> None:
    default_geometry = build_gv_synthetic_fixture(
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        sigma=0.02,
    ).to_manifest()
    alternate_geometry = build_gv_synthetic_fixture(
        experiment="torsion",
        geometry="gv_rad2_5_height12",
        controls={"theta": 0.03},
        sigma=0.02,
    ).to_manifest()

    assert default_geometry["geometry"] == "gv_rad2_height14_28"
    assert alternate_geometry["geometry"] == "gv_rad2_5_height12"
    assert default_geometry["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert alternate_geometry["dataset_id"] == "gv:torsion:gv_rad2_5_height12:theta_0.03"


def test_synthetic_materialization_contract_writes_deterministic_npz_and_manifest_when_available(tmp_path: Path) -> None:
    materialize = _require_public_materializer(synthetic_module, "materialize_gv_synthetic_reference")
    runtime_manifest = load_runtime_descriptor("stretching").plan(output_root=tmp_path / "runtime").to_manifest()

    first = materialize(seed=19, runtime_manifest=runtime_manifest, output_root=tmp_path / "first", point_count=5)
    second = materialize(seed=19, runtime_manifest=runtime_manifest, output_root=tmp_path / "second", point_count=5)

    dataset_path_a, manifest_path_a = _materialized_paths(first)
    dataset_path_b, manifest_path_b = _materialized_paths(second)

    assert dataset_path_a.suffix == ".npz"
    assert dataset_path_b.suffix == ".npz"
    assert manifest_path_a.suffix == ".json"
    assert manifest_path_b.suffix == ".json"

    manifest_a = json.loads(manifest_path_a.read_text(encoding="utf-8"))
    manifest_b = json.loads(manifest_path_b.read_text(encoding="utf-8"))
    with np.load(dataset_path_a) as dataset_a, np.load(dataset_path_b) as dataset_b:
        assert set(dataset_a.files) >= {"points", "values"}
        assert dataset_a.files == dataset_b.files
        assert np.array_equal(dataset_a["points"], dataset_b["points"])
        assert np.array_equal(dataset_a["values"], dataset_b["values"])
        assert dataset_a["points"].shape == (5,)
        assert dataset_a["values"].shape == (5,)

    for manifest in (manifest_a, manifest_b):
        assert manifest["dataset_id"] == runtime_manifest["dataset_id"]
        assert manifest["structure"] == "gv"
        assert manifest["experiment"] == "stretching"
        assert manifest["geometry"] == runtime_manifest["geometry"]
        assert manifest["controls"] == runtime_manifest["controls"]
        assert manifest["calibrated_parameter_names"] == list(GV_PARAMETER_CONTRACT.calibrated_names)
        assert manifest["noise_model"]["kind"] == "multiplicative"
        assert manifest["noise_model"]["parameter"] == "sigma"
        assert manifest["reference_kind"] == "synthetic"
        assert manifest["provenance"]["runtime_provenance_root"].endswith("gv_simulation_files/stretching/gv")
        assert manifest["provenance"]["source_files"] == list(runtime_manifest["source_files"])
        assert manifest["reference_source"]

    assert manifest_a["points"] == manifest_b["points"]
    assert manifest_a["values"] == manifest_b["values"]


def test_synthetic_reference_dataset_materialization_filters_and_records_entries(tmp_path: Path) -> None:
    torsion_fixture = build_gv_synthetic_fixture(
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        sigma=0.02,
        points=(0.0, 0.5, 1.0),
        values=(0.1, 0.2, 0.3),
    )
    stretching_fixture = build_gv_synthetic_fixture(
        experiment="stretching",
        geometry="gv_rad2_height14_28",
        controls={"tot_force": 500.0, "bpress": -91.0},
        sigma=0.03,
        points=(0.0, 1.0),
        values=(0.4, 0.6),
    )

    result = materialize_gv_synthetic_reference_dataset(
        [torsion_fixture, stretching_fixture],
        data_path=tmp_path / "materialized" / "synthetic_refs.npz",
        collection_id="gv:synthetic:test-collection",
        experiments=["stretching"],
    )

    assert result.dataset_path.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "gv:synthetic:test-collection"
    assert manifest["entry_count"] == 1
    assert manifest["selection"] == {
        "experiments": ["stretching"],
        "geometries": [],
        "dataset_ids": [],
    }
    assert manifest["entries"][0]["dataset_id"] == stretching_fixture.identity.dataset_id
    assert manifest["entries"][0]["controls"] == {"tot_force": 500.0, "bpress": -91.0}
    assert manifest["entries"][0]["nuisance_parameters"] == {"sigma": 0.03}

    with np.load(result.dataset_path) as dataset:
        assert dataset["dataset_ids"].tolist() == [stretching_fixture.identity.dataset_id]
        assert dataset["experiments"].tolist() == ["stretching"]
        assert dataset["geometries"].tolist() == ["gv_rad2_height14_28"]
        assert dataset["point_counts"].tolist() == [2]
        array_key = manifest["entries"][0]["array_key"]
        assert np.array_equal(dataset[f"{array_key}__points"], np.asarray([0.0, 1.0]))
        assert np.array_equal(dataset[f"{array_key}__values"], np.asarray([0.4, 0.6]))


def test_synthetic_reference_dataset_materialization_rejects_empty_and_mixed_collections(tmp_path: Path) -> None:
    synthetic_fixture = build_gv_synthetic_fixture(
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        sigma=0.02,
    )
    dpd_fixture = build_gv_synthetic_fixture(
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.04},
        sigma=0.02,
        reference_kind="dpd_generated",
    )

    with pytest.raises(ValueError, match="No GV reference records remain"):
        materialize_gv_synthetic_reference_dataset(
            [synthetic_fixture],
            data_path=tmp_path / "empty.npz",
            experiments=["stretching"],
        )

    with pytest.raises(ValueError, match="Mixed GV reference kinds"):
        materialize_reference_records(
            [synthetic_fixture.to_record(), dpd_fixture.to_record()],
            data_path=tmp_path / "mixed_kind.npz",
        )

    record = synthetic_fixture.to_record()
    with pytest.raises(ValueError, match="Mixed surrogate backends"):
        materialize_reference_records(
            [record, replace(record, surrogate_backend="other")],
            data_path=tmp_path / "mixed_backend.npz",
        )

    emb_context = replace(record.context, structure=SimpleNamespace(name="emb"))
    with pytest.raises(ValueError, match="Mixed structures"):
        materialize_reference_records(
            [record, replace(record, context=emb_context)],
            data_path=tmp_path / "mixed_structure.npz",
        )


def test_synthetic_gv_reference_rejects_bnn_backend() -> None:
    with pytest.raises(ValueError, match="only the 'dnn' surrogate backend"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            surrogate_backend="bnn",
        )


def test_synthetic_fixture_rejects_invalid_series_reference_kind_and_templates() -> None:
    with pytest.raises(ValueError, match="points must contain at least one value"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            points=[],
        )

    with pytest.raises(ValueError, match="No synthetic fixture template"):
        build_gv_synthetic_fixture(
            experiment="not_registered",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
        )

    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            reference_kind="real",
        )

    with pytest.raises(ValueError, match="same length"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            sigma=0.02,
            points=[0.0, 1.0],
            values=[0.0],
        )

    with pytest.raises(ValueError, match="separate from calibrated parameters"):
        build_gv_synthetic_fixture(
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03, "ka": 1.0},
            sigma=0.02,
        )


def test_synthetic_reference_wrappers_and_private_axis_helpers() -> None:
    manifest = build_gv_synthetic_reference_manifest(
        seed=9,
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        point_count=2,
    )
    assert manifest["points"] == [0.0, 1.0]
    assert manifest["values"]

    identity = synthetic_module._identity_from_axes(
        structure="gv",
        experiment="torsion",
        geometry="gv_rad2_height14_28",
        controls={"theta": 0.03},
        surrogate_backend="dnn",
        reference_kind="synthetic",
        dataset_id="explicit",
    )
    assert identity.dataset_id == "explicit"
    assert synthetic_module._control_id({}) == "default"
    assert synthetic_module._seeded_synthetic_value(3, "dataset", "not_registered", {}, 0.5) > 0.0

    runtime_fixture = synthetic_module.synthetic_fixture_from_runtime_manifest(
        {
            "structure": "gv",
            "experiment": "torsion",
            "geometry": "gv_rad2_height14_28",
            "controls": {"theta": 0.03},
            "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0.03",
            "source_files": ("gv_simulation_files/torsion/gv/run.py",),
            "runtime_package": "mirheoOBMD",
        },
        sigma=0.02,
        d0=1.0,
        metadata={"path": Path("metadata.json"), "items": ("a", "b")},
    )
    runtime_manifest = runtime_fixture.to_manifest()
    assert runtime_manifest["runtime_manifest"]["runtime_package"] == "mirheoOBMD"
    assert runtime_manifest["nuisance_parameters"] == {"sigma": 0.02, "d0": 1.0}
    assert runtime_fixture.to_record().metadata["items"] == ["a", "b"]

    with pytest.raises(ValueError, match="only the 'dnn' surrogate backend"):
        synthetic_module._identity_from_axes(
            structure="gv",
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            surrogate_backend="bnn",
            reference_kind="synthetic",
        )

    with pytest.raises(ValueError, match="Unsupported GV reference kind"):
        synthetic_module._identity_from_axes(
            structure="gv",
            experiment="torsion",
            geometry="gv_rad2_height14_28",
            controls={"theta": 0.03},
            surrogate_backend="dnn",
            reference_kind="real",
        )
