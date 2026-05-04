from __future__ import annotations

import json

import pytest

from meso_uq.structures.gv.numerical_data import (
    GV_NUMERICAL_MANIFEST_SCHEMA_VERSION,
    GV_NUMERICAL_POSTPROCESSOR_VERSION,
    build_gv_numerical_dataset_manifest,
    campaign_dataset_dir,
    campaign_dataset_hdf5_path,
    campaign_dataset_manifest_path,
    validate_gv_numerical_manifest,
)


_GV_MANIFEST_BASE = {
    "campaign_id": "campaign-001",
    "geometry_radius": 2.0,
    "geometry_height": 14.28,
    "material_parameters": {
        "ka": 1.1,
        "kb": 1.2,
        "mu": 0.9,
        "b1": 0.1,
        "b2": 0.2,
        "a3": 0.3,
        "a4": 0.4,
        "mu_l": 0.5,
        "c": 0.6,
    },
    "controls": {
        "theta": 0.03,
    },
    "quality_flags": {
        "finite_observables": True,
        "finite_observable_ratio": 1.0,
        "canary_failures": [],
    },
    "raw_provenance": {
        "mirheo_log": "gv/torsion/logs/mirheo.log",
    },
}


def test_build_gv_numerical_dataset_manifest_is_json_serializable() -> None:
    manifest = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        **_GV_MANIFEST_BASE,
    )
    payload = manifest.to_manifest()

    assert payload["manifest_schema_version"] == GV_NUMERICAL_MANIFEST_SCHEMA_VERSION
    assert payload["postprocessor_version"] == GV_NUMERICAL_POSTPROCESSOR_VERSION
    assert payload["geometry"] == "gv_rad2_height14_28"
    assert payload["control_id"] == "theta_0.03"
    assert payload["dataset_id"] == "gv:torsion:gv_rad2_height14_28:theta_0.03"
    assert payload["quality_flags"]["finite_observables"] is True
    json.dumps(payload)


def test_validate_gv_numerical_manifest_rejects_mixed_structure_identifiers() -> None:
    manifest = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        **_GV_MANIFEST_BASE,
    ).to_manifest()
    validate_gv_numerical_manifest(manifest)

    bad_manifest = dict(manifest)
    bad_manifest["dataset_id"] = bad_manifest["dataset_id"].replace("gv:", "emb:", 1)
    bad_manifest["structure"] = "emb"
    with pytest.raises(ValueError, match="GV numerical data is only defined"):
        validate_gv_numerical_manifest(bad_manifest)


def test_validate_gv_numerical_manifest_detects_invalid_controls_and_material_schema() -> None:
    valid = dict(_GV_MANIFEST_BASE)

    invalid_controls = dict(valid)
    invalid_controls["controls"] = {"ka": 1.0}
    with pytest.raises(ValueError, match="Unknown GV controls"):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            **invalid_controls,
        )

    invalid_material = dict(valid)
    invalid_material["material_parameters"] = dict(_GV_MANIFEST_BASE["material_parameters"])
    invalid_material["material_parameters"].pop("c")
    with pytest.raises(ValueError, match="requires all calibrated"):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            **invalid_material,
        )

    invalid_flags = dict(valid)
    invalid_flags["quality_flags"] = {"finite_observable_ratio": 1.0}
    with pytest.raises(ValueError, match="must include 'finite_observables'"):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            **invalid_flags,
        )


def test_campaign_dataset_path_helpers() -> None:
    manifest = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        **_GV_MANIFEST_BASE,
    )
    dataset_dir = campaign_dataset_dir(campaign_id="campaign-001", dataset_id=manifest.dataset_id)
    manifest_path = campaign_dataset_manifest_path(
        campaign_id="campaign-001",
        dataset_id=manifest.dataset_id,
    )
    hdf5_path = campaign_dataset_hdf5_path(
        campaign_id="campaign-001",
        dataset_id=manifest.dataset_id,
    )

    assert dataset_dir.as_posix().startswith("_runs/gv/numerical_data/campaign-001/datasets/gv/torsion")
    assert manifest_path.parent == dataset_dir
    assert hdf5_path.name == "numerical_dataset.h5"
