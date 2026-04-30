from __future__ import annotations

import pytest

from meso_uq.surrogate.catalogs import resolve_surrogate_catalog_entry
from meso_uq.surrogate.catalogs import (
    SurrogateDatasetIdentity,
    resolve_surrogate_catalog_entries,
)
from meso_uq.surrogate.gv_catalog import (
    iter_gv_surrogate_catalog_entries,
    resolve_gv_surrogate_catalog_entries,
    resolve_gv_surrogate_catalog_entry,
)
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY


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
    assert resolved["metadata"]["paths"]["provenance_root"] == "gv_simulation_files/stretching/gv"
    assert resolved["surrogate_artifact"].endswith(
        "/_runs/gv/stretching/gv_rad2_height14_28/tot_force_500_50000__bpress_-91/synthetic/dnn/model.pt"
    )


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
        "/compression/surrogate/diameters/2.1um/trained/microbubble_force_BNN.pt"
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
