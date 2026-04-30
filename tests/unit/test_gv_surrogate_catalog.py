from __future__ import annotations

import pytest

from meso_uq.surrogate.catalogs import resolve_surrogate_catalog_entry
from meso_uq.surrogate.gv_catalog import (
    iter_gv_surrogate_catalog_entries,
    resolve_gv_surrogate_catalog_entry,
)
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY


def test_gv_catalog_entries_are_structure_aware_and_dnn_only() -> None:
    entries = iter_gv_surrogate_catalog_entries()
    assert {entry.identity.structure for entry in entries} == {"gv"}
    assert {entry.identity.surrogate_backend for entry in entries} == {"dnn"}
    assert {entry.identity.reference_kind for entry in entries} == {"synthetic", "dpd_generated"}


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
