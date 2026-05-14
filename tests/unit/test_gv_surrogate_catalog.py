from __future__ import annotations

import pytest

from meso_uq.surrogate.catalogs import resolve_surrogate_catalog_entry
from meso_uq.surrogate.catalogs import (
    SurrogateDatasetIdentity,
    resolve_surrogate_catalog_entries,
)
from meso_uq.surrogate.gv_catalog import (
    _GV_EXPERIMENT_LANES,
    iter_gv_surrogate_catalog_entries,
    resolve_gv_surrogate_catalog_entries,
    resolve_gv_surrogate_catalog_entry,
)
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY
from meso_uq.structures.registry import GeometrySpec


def test_gv_catalog_entries_are_structure_aware_and_dnn_only() -> None:
    entries = iter_gv_surrogate_catalog_entries()
    assert {entry.identity.structure for entry in entries} == {"gv"}
    assert {entry.identity.surrogate_backend for entry in entries} == {"dnn"}
    assert {entry.identity.reference_kind for entry in entries} == {"synthetic", "dpd_generated"}


def test_catalog_identity_rejects_unsupported_axes() -> None:
    with pytest.raises(ValueError, match="Unsupported reference kind"):
        SurrogateDatasetIdentity(
            structure="gv",
            experiment="torsion",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="theta_0_01_0_1",
            reference_kind="mock",
        )

    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        SurrogateDatasetIdentity(
            structure="gv",
            experiment="torsion",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="theta_0_01_0_1",
            surrogate_backend="cnn",
        )


def test_gv_catalog_iter_subset_filters_by_experiment_and_geometry() -> None:
    entries = iter_gv_surrogate_catalog_entries(
        include_experimental=True,
        experiments=("shear_flow", "torsion"),
        geometries=(DEFAULT_GV_GEOMETRY.id,),
    )
    assert entries
    assert {entry.identity.experiment for entry in entries} == {"shear_flow", "torsion"}
    assert {entry.identity.geometry for entry in entries} == {DEFAULT_GV_GEOMETRY.id}

    resolved = resolve_gv_surrogate_catalog_entries(
        "/repo",
        include_experimental=True,
        experiments=("shear_flow",),
        geometries=(DEFAULT_GV_GEOMETRY.id,),
    )
    assert resolved
    assert {entry["experiment"] for entry in resolved} == {"shear_flow"}


def test_gv_catalog_lookup_requires_full_identity() -> None:
    with pytest.raises(
        ValueError,
        match="GV surrogate lookup requires structure, experiment, geometry, and controls",
    ):
        resolve_gv_surrogate_catalog_entry("/repo", experiment="stretching")


def test_gv_catalog_rejects_bnn_backend() -> None:
    with pytest.raises(
        ValueError,
        match="supports only the 'dnn' backend in this tranche",
    ):
        resolve_gv_surrogate_catalog_entry(
            "/repo",
            experiment="stretching",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="tot_force_500_50000__bpress_-91",
            surrogate_backend="bnn",
        )


def test_gv_catalog_multi_geometry_subset_lookup_is_geometry_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    alt_geometry = GeometrySpec(
        id="gv_custom_radius2_height18",
        label="Custom geometry",
        shape="disc",
        parameters={"radius": 2.0, "height": 18.0},
        source="tests",
    )
    from meso_uq.surrogate import gv_catalog

    monkeypatch.setattr(gv_catalog, "_catalog_geometries", lambda: (DEFAULT_GV_GEOMETRY, alt_geometry))

    entries = iter_gv_surrogate_catalog_entries(
        experiments=("eigenmodes",),
        geometries=(DEFAULT_GV_GEOMETRY.id, alt_geometry.id),
    )
    assert {entry.identity.geometry for entry in entries} == {DEFAULT_GV_GEOMETRY.id, alt_geometry.id}
    assert len(entries) == 4
    entry = resolve_gv_surrogate_catalog_entry(
        "/repo",
        experiment="eigenmodes",
        geometry=alt_geometry.id,
        controls="bpress_-91",
    )
    assert entry["geometry"] == alt_geometry.id
    assert entry["metadata"]["geometry"] == alt_geometry.id


def test_gv_catalog_rejects_calibrated_or_nuisance_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    mutated_lanes = (
        {
            **_GV_EXPERIMENT_LANES[0],
            "controls": "ka_1.0",
            "control_values": {"ka": 1.0},
        },
    )
    monkeypatch.setattr("meso_uq.surrogate.gv_catalog._GV_EXPERIMENT_LANES", mutated_lanes)
    with pytest.raises(ValueError, match="cannot collide with calibrated or nuisance parameters"):
        iter_gv_surrogate_catalog_entries()


def test_gv_catalog_metadata_exposes_feature_parameter_and_schema_contract() -> None:
    entry = iter_gv_surrogate_catalog_entries(experiments=("torsion",), geometries=(DEFAULT_GV_GEOMETRY.id,))[0]
    metadata = entry.metadata
    assert metadata["geometry"] == DEFAULT_GV_GEOMETRY.id
    assert metadata["controls"] == {"theta": {"start": 0.01, "stop": 0.1, "steps": 10}}
    assert metadata["backend"] == "dnn"
    assert metadata["reference_provenance"]["runtime_source_root"] == "gv/torsion/src"
    assert metadata["feature_order"]["axis"] == "observable_axis"
    assert metadata["feature_order"]["target"] == "response"
    assert metadata["parameter_order"]["calibrated"][:3] == ["ka", "kb", "mu"]
    assert metadata["observable_schema"]["identity"] == "meso_uq.structures.gv"
    assert metadata["observable_schema"]["repository_path"] == "src/meso_uq/structures/gv"


def test_gv_catalog_resolution_carries_structure_identity_and_paths() -> None:
    resolved = resolve_gv_surrogate_catalog_entry(
        "/repo",
        experiment="stretching",
        geometry=DEFAULT_GV_GEOMETRY.id,
        controls="tot_force_500_50000__bpress_-91",
        reference_kind="synthetic",
    )
    assert resolved["dataset_id"] == (
        f"gv:stretching:{DEFAULT_GV_GEOMETRY.id}:tot_force_500_50000__bpress_-91"
    )
    assert resolved["catalog_key"].endswith(":synthetic:dnn")
    assert resolved["metadata"]["supports_bnn"] is False
    assert resolved["metadata"]["paths"]["provenance_root"] == "gv/stretching/src"
    assert resolved["metadata"]["paths"]["source_root"] == "gv/stretching/src"
    assert resolved["metadata"]["paths"]["legacy_import_root"] == "gv_simulation_files/stretching/gv"
    assert resolved["surrogate_artifact"].endswith(
        "/_runs/gv/stretching/gv_rad2_height14_28/tot_force_500_50000__bpress_-91/synthetic/dnn/model.pt"
    )


def test_gv_catalog_paths_and_metadata_distinguish_source_and_legacy_import_roots() -> None:
    entries = resolve_gv_surrogate_catalog_entries("/repo")

    for entry in entries:
        metadata_paths = entry["metadata"]["paths"]
        experiment = str(entry["experiment"])

        assert metadata_paths["provenance_root"] == f"gv/{experiment}/src"
        assert metadata_paths["source_root"] == f"gv/{experiment}/src"
        assert "gv_simulation_files" not in metadata_paths["provenance_root"]
        assert "gv_simulation_files" not in metadata_paths["source_root"]
        assert metadata_paths["legacy_import_root"].startswith("gv_simulation_files/")
        if experiment == "shear_flow":
            assert metadata_paths["legacy_import_root"] == "gv_simulation_files/shear_flow"
        else:
            assert metadata_paths["legacy_import_root"] == f"gv_simulation_files/{experiment}/gv"


def test_catalog_list_filters_structure_backend_and_reference_kind() -> None:
    gv_entries = resolve_surrogate_catalog_entries("/repo", structure="gv")
    emb_bnn_entries = resolve_surrogate_catalog_entries("/repo", structure="emb", surrogate_backend="bnn")
    synthetic_gv_entries = resolve_gv_surrogate_catalog_entries("/repo", reference_kind="synthetic")
    gv_dnn_entries = resolve_surrogate_catalog_entries("/repo", structure="gv", surrogate_backend="dnn")

    assert gv_entries
    assert {entry["structure"] for entry in gv_entries} == {"gv"}
    assert emb_bnn_entries
    assert {entry["structure"] for entry in emb_bnn_entries} == {"emb"}
    assert {entry["surrogate_backend"] for entry in emb_bnn_entries} == {"bnn"}
    assert synthetic_gv_entries
    assert {entry["reference_kind"] for entry in synthetic_gv_entries} == {"synthetic"}
    assert gv_dnn_entries
    assert {entry["structure"] for entry in gv_dnn_entries} == {"gv"}
    assert {entry["surrogate_backend"] for entry in gv_dnn_entries} == {"dnn"}

    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        resolve_surrogate_catalog_entries("/repo", surrogate_backend="cnn")

    with pytest.raises(ValueError, match="Unsupported reference kind"):
        resolve_gv_surrogate_catalog_entries("/repo", reference_kind="mock")


def test_gv_catalog_experimental_and_reference_filters_are_explicit() -> None:
    production_entries = iter_gv_surrogate_catalog_entries()
    experimental_entries = iter_gv_surrogate_catalog_entries(include_experimental=True)
    dpd_entries = resolve_gv_surrogate_catalog_entries(
        "/repo",
        reference_kind="dpd_generated",
        include_experimental=True,
    )

    assert all(entry.identity.experiment != "shear_flow" for entry in production_entries)
    assert any(entry.identity.experiment == "shear_flow" for entry in experimental_entries)
    assert dpd_entries
    assert {entry["reference_kind"] for entry in dpd_entries} == {"dpd_generated"}
    assert any(entry["experiment"] == "shear_flow" for entry in dpd_entries)

    shear_entry = resolve_gv_surrogate_catalog_entry(
        "/repo",
        experiment="shear_flow",
        geometry=DEFAULT_GV_GEOMETRY.id,
        controls="ptan_0.4__afsi_0__bpress_-91",
        reference_kind="synthetic",
        include_experimental=True,
    )
    assert shear_entry["metadata"]["experimental"] is True
    assert shear_entry["metadata"]["requires_opt_in"] is True
    assert shear_entry["metadata"]["known_issues"][0]["id"] == "bouncer_collision_candidates_coarse"
    assert (
        shear_entry["metadata"]["known_issues"][0]["evidence"]
        == "gv/shear_flow/src/fixtures/bouncer_collision_candidates_coarse_excerpt.txt"
    )

    with pytest.raises(KeyError, match="shear_flow"):
        resolve_gv_surrogate_catalog_entry(
            "/repo",
            experiment="shear_flow",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="ptan_0.4__afsi_0__bpress_-91",
            reference_kind="synthetic",
        )

    with pytest.raises(ValueError, match="Unsupported reference kind"):
        resolve_gv_surrogate_catalog_entry(
            "/repo",
            experiment="torsion",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="theta_0.01_0.1",
            reference_kind="mock",
        )


def test_common_catalog_preserves_emb_lookup_behavior() -> None:
    resolved = resolve_surrogate_catalog_entry(
        "/repo",
        structure="emb",
        experiment="compression",
        geometry="diameter_2.1um",
        controls="default",
        surrogate_backend="bnn",
    )
    assert resolved["dataset_id"] == "emb:compression:diameter_2.1um:default"
    assert resolved["metadata"]["legacy_name"] == "compression_2.1um"
    assert resolved["surrogate_artifact"].endswith(
        "/emb/compression/surrogate/diameters/2.1um/trained/microbubble_force_BNN.pt"
    )


def test_common_catalog_routes_gv_errors() -> None:
    with pytest.raises(
        ValueError,
        match="GV surrogate lookup requires structure, experiment, geometry, and controls",
    ):
        resolve_surrogate_catalog_entry("/repo", structure="gv", experiment="stretching")


def test_catalog_missing_entries_fail_with_full_key() -> None:
    with pytest.raises(ValueError, match="requires a geometry selection"):
        resolve_surrogate_catalog_entry("/repo", structure="emb", experiment="compression")

    with pytest.raises(KeyError, match="emb:compression:diameter_9.9um:default"):
        resolve_surrogate_catalog_entry(
            "/repo",
            structure="emb",
            experiment="compression",
            geometry="diameter_9.9um",
        )

    with pytest.raises(KeyError, match="vesicle:compression:diameter_2.1um:default"):
        resolve_surrogate_catalog_entry(
            "/repo",
            structure="vesicle",
            experiment="compression",
            geometry="diameter_2.1um",
        )

    with pytest.raises(KeyError, match="No GV surrogate catalog entry registered"):
        resolve_gv_surrogate_catalog_entry(
            "/repo",
            experiment="torsion",
            geometry=DEFAULT_GV_GEOMETRY.id,
            controls="theta_missing",
        )
