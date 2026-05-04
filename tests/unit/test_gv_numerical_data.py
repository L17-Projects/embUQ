from __future__ import annotations

import json
from pathlib import Path

import pytest

from meso_uq.structures.gv import numerical_data
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


def test_build_gv_numerical_manifest_covers_jsonable_provenance_and_defaults() -> None:
    manifest = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        raw_provenance={"path": Path("logs/mirheo.log"), "items": (Path("a"), ["b"])},
        quality_flags=None,
        **{key: value for key, value in _GV_MANIFEST_BASE.items() if key not in {"raw_provenance", "quality_flags"}},
    )

    payload = manifest.to_manifest()
    assert payload["raw_provenance"] == {"path": "logs/mirheo.log", "items": ["a", ["b"]]}
    assert payload["quality_flags"]["finite_observables"] is True
    assert payload["quality_flags"]["finite_observable_ratio"] == 1.0
    assert payload["quality_flags"]["canary_failures"] == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"campaign_id": ""}, "campaign_id must be a non-empty"),
        ({"campaign_id": "bad/path"}, "path separators"),
        ({"campaign_id": "bad..path"}, "path traversal"),
        ({"structure": "emb"}, "only defined for structure 'gv'"),
        ({"experiment": ""}, "experiment must be a non-empty"),
        ({"experiment": "compression"}, "belongs to structure 'emb'"),
        ({"experiment": "not_real"}, "Unknown experiment"),
        ({"geometry_radius": "bad"}, "geometry_radius must be numeric"),
        ({"geometry_height": float("inf")}, "geometry_height must be finite"),
        ({"geometry_radius": 0.0}, "radius must be positive"),
        ({"geometry_height": -1.0}, "height must be positive"),
        ({"raw_provenance": "not-a-map"}, "raw_provenance must be a mapping"),
    ],
)
def test_build_gv_numerical_manifest_rejects_invalid_inputs(overrides, message) -> None:
    kwargs = {"structure": "gv", "experiment": "torsion", **_GV_MANIFEST_BASE, **overrides}

    with pytest.raises(ValueError, match=message):
        build_gv_numerical_dataset_manifest(**kwargs)


def test_build_gv_numerical_manifest_rejects_missing_geometry_axes(monkeypatch) -> None:
    class GeometryWithoutHeight:
        id = "gv_bad"
        parameters = {"radius": 2.0}

    monkeypatch.setattr(numerical_data, "build_geometry", lambda **_kwargs: GeometryWithoutHeight())

    with pytest.raises(ValueError, match="must include height"):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            **_GV_MANIFEST_BASE,
        )


@pytest.mark.parametrize(
    ("units", "message"),
    [
        ("not-a-map", "units must be a mapping"),
        ({"controls": []}, "units.controls must be a mapping"),
        ({"material_parameters": []}, "units.material_parameters must be a mapping"),
        ({"controls": {"unexpected": "1"}}, "unexpected control"),
        ({"controls": {"theta": 1}}, "control unit"),
        ({"material_parameters": {"unexpected": "1"}}, "unexpected material"),
        ({"material_parameters": {"ka": 1}}, "material unit"),
    ],
)
def test_build_gv_numerical_manifest_rejects_bad_units(units, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            units=units,
            **_GV_MANIFEST_BASE,
        )


@pytest.mark.parametrize(
    ("normalization", "message"),
    [
        ({"scales": []}, "normalization.scales must be a mapping"),
        ({"offsets": []}, "normalization.offsets must be a mapping"),
        ({"scales": {"unexpected": 1.0}}, "normalization.scales contains unexpected"),
        ({"offsets": {"unexpected": 0.0}}, "normalization.offsets contains unexpected"),
        ({"scales": {"ka": "bad"}}, "normalization.scales\\['ka'\\] must be numeric"),
    ],
)
def test_build_gv_numerical_manifest_rejects_bad_normalization(normalization, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            normalization=normalization,
            **_GV_MANIFEST_BASE,
        )


@pytest.mark.parametrize(
    ("quality_flags", "message"),
    [
        ([], "quality_flags must be a mapping"),
        ({"finite_observables": "yes"}, "must be boolean"),
        ({"finite_observables": True, "finite_observable_ratio": -0.1}, "between 0 and 1"),
        ({"finite_observables": True, "canary_failures": 3}, "must be a sequence"),
    ],
)
def test_build_gv_numerical_manifest_rejects_bad_quality_flags(quality_flags, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_gv_numerical_dataset_manifest(
            structure="gv",
            experiment="torsion",
            quality_flags=quality_flags,
            **{key: value for key, value in _GV_MANIFEST_BASE.items() if key != "quality_flags"},
        )


def test_build_gv_numerical_manifest_accepts_string_canary_failure() -> None:
    manifest = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        quality_flags={"finite_observables": False, "canary_failures": "gn58_cuda_busy"},
        **{key: value for key, value in _GV_MANIFEST_BASE.items() if key != "quality_flags"},
    )

    assert manifest.to_manifest()["quality_flags"]["canary_failures"] == ["gn58_cuda_busy"]


@pytest.mark.parametrize(
    ("dataset_id", "message"),
    [
        (3, "must be a string"),
        ("gv:torsion:missing", "exactly 4 components"),
        ("emb:torsion:gv_rad2_height14_28:theta_0.03", "structure mismatch"),
        ("gv:torsion::theta_0.03", "non-empty geometry"),
        ("gv:torsion:gv_rad2_height14_28:", "non-empty control"),
        ("gv:torsion:../outside:theta_0.03", "path separators"),
        ("gv:torsion:gv_rad2_height14_28:theta/0.03", "path separators"),
        ("gv:torsion:gv_rad2_height14_28:theta\\\\0.03", "path separators"),
    ],
)
def test_validate_gv_numerical_dataset_id_rejects_invalid_values(dataset_id, message) -> None:
    with pytest.raises(ValueError, match=message):
        numerical_data.validate_gv_numerical_dataset_id(dataset_id, structure="gv")


def test_validate_gv_numerical_manifest_rejects_payload_schema_errors() -> None:
    payload = build_gv_numerical_dataset_manifest(
        structure="gv",
        experiment="torsion",
        **_GV_MANIFEST_BASE,
    ).to_manifest()

    with pytest.raises(ValueError, match="must be a mapping"):
        validate_gv_numerical_manifest([])

    missing = dict(payload)
    missing.pop("dataset_id")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_gv_numerical_manifest(missing)

    bad_version_type = dict(payload, manifest_schema_version="1")
    with pytest.raises(ValueError, match="must be an integer"):
        validate_gv_numerical_manifest(bad_version_type)

    bad_version = dict(payload, manifest_schema_version=999)
    with pytest.raises(ValueError, match="schema version mismatch"):
        validate_gv_numerical_manifest(bad_version)

    bad_postprocessor = dict(payload, postprocessor_version="old")
    with pytest.raises(ValueError, match="postprocessor_version is unsupported"):
        validate_gv_numerical_manifest(bad_postprocessor)

    bad_geometry_type = dict(payload, geometry_parameters=[])
    with pytest.raises(ValueError, match="geometry_parameters must be a mapping"):
        validate_gv_numerical_manifest(bad_geometry_type)

    missing_radius = dict(payload, geometry_parameters={"height": 14.28})
    with pytest.raises(ValueError, match="must include 'radius'"):
        validate_gv_numerical_manifest(missing_radius)

    bad_geometry_token = dict(payload, geometry="gv_wrong")
    with pytest.raises(ValueError, match="does not match geometry_parameters"):
        validate_gv_numerical_manifest(bad_geometry_token)

    extra_geometry = dict(payload, geometry_parameters={**payload["geometry_parameters"], "width": 1.0})
    with pytest.raises(ValueError, match="exactly radius and height"):
        validate_gv_numerical_manifest(extra_geometry)

    bad_controls = dict(payload, controls=[])
    with pytest.raises(ValueError, match="controls must be a mapping"):
        validate_gv_numerical_manifest(bad_controls)

    bad_control_id = dict(payload, control_id="wrong")
    with pytest.raises(ValueError, match="control_id must match"):
        validate_gv_numerical_manifest(bad_control_id)

    bad_raw = dict(payload, raw_provenance=[])
    with pytest.raises(ValueError, match="raw_provenance must be a mapping"):
        validate_gv_numerical_manifest(bad_raw)


def test_campaign_dataset_root_rejects_bad_campaign_id() -> None:
    with pytest.raises(ValueError, match="path separators"):
        numerical_data.campaign_dataset_root(campaign_id="bad/path")
