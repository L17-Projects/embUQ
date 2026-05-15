from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.structures.gv.launch import (
    GVLaunchRequest,
    build_gv_launch_campaign_manifest,
    validate_gv_launch_request,
)


_VALID_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.1,
    "b2": 0.2,
    "a3": 0.3,
    "a4": 0.4,
    "mu_l": 0.5,
    "c": 0.6,
}


def test_validate_gv_launch_request_normalizes_required_fields() -> None:
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-001",
        walltime="01:30:00",
        gpu_count=2,
        provenance_tags={"linear_issue": "MES-177", "source": "manual"},
    )

    assert isinstance(request, GVLaunchRequest)
    assert request.experiment == "stretching"
    assert request.geometry_id == "gv_rad2_height14_28"
    assert request.fixed_controls == {"bpress": -91.0}
    assert request.sweep.axis == "tot_force"
    assert request.sweep.values == (500.0, 750.0)
    assert request.platform.value == "vega"
    assert request.output_root == Path("_runs/gv/launches/gv-campaign-001")
    assert request.campaign_id == "gv-campaign-001"
    assert request.walltime_seconds == 5400
    assert request.provenance_tags == {"linear_issue": "MES-177", "source": "manual"}
    assert request.to_manifest()["manifest_schema_version"] == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"platform": "workstation"}, "platform must be one of"),
        ({"output_root": "../bad-root"}, "path traversal"),
        ({"walltime": "not-a-time"}, "valid SLURM time limit"),
        ({"gpu_count": 0}, "gpu_count must be positive"),
        ({"provenance_tags": {}}, "must include at least one tag"),
        (
            {"material_parameters": {**_VALID_MATERIAL_PARAMETERS, "b1": 0.0}},
            "must be finite and > 0",
        ),
    ],
)
def test_validate_gv_launch_request_rejects_invalid_inputs(overrides, message: str) -> None:
    kwargs = {
        "experiment": "stretching",
        "material_parameters": _VALID_MATERIAL_PARAMETERS,
        "geometry": {"radGV": 2.0, "height": 14.28},
        "controls": {"tot_force": (500.0, 750.0), "bpress": -91.0},
        "platform": "vega",
        "output_root": "_runs/gv/launches/gv-campaign-001",
        "walltime": "01:30:00",
        "gpu_count": 2,
        "provenance_tags": {"linear_issue": "MES-177"},
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_gv_launch_request(**kwargs)


def test_validate_gv_launch_request_fails_before_creating_output_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_gv_launch_request(
            experiment="stretching",
            material_parameters={**_VALID_MATERIAL_PARAMETERS, "a4": 0.0},
            geometry={"radGV": 2.0, "height": 14.28},
            controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
            platform="vega",
            output_root="gv-campaign-001",
            walltime="01:30:00",
            gpu_count=2,
            provenance_tags={"linear_issue": "MES-180"},
        )

    assert not (tmp_path / "gv-campaign-001").exists()


def test_build_gv_launch_campaign_manifest_keeps_single_point_identity_easy() -> None:
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0,), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-single",
        walltime="01:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-179"},
    )

    manifest = build_gv_launch_campaign_manifest(request)

    assert manifest.artifact_scope == "control_point"
    assert manifest.output_id == "bpress_-91__tot_force_500"
    assert manifest.dataset_id == "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_500"
    assert len(manifest.runs) == 1
    assert manifest.runs[0].output_id == manifest.output_id
    assert manifest.runs[0].dataset_id == manifest.dataset_id
    assert manifest.runs[0].hdf5_path == manifest.hdf5_path
    assert manifest.hdf5_path.as_posix().startswith("_runs/gv/launches/gv-campaign-single/datasets/")


def test_build_gv_launch_campaign_manifest_uses_sweep_aware_identity_for_multi_point_campaigns() -> None:
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-sweep",
        walltime="01:30:00",
        gpu_count=2,
        provenance_tags={"linear_issue": "MES-179"},
    )

    manifest = build_gv_launch_campaign_manifest(request)

    assert manifest.artifact_scope == "sweep"
    assert manifest.output_id == "bpress_-91__sweep_tot_force__values_500__750"
    assert manifest.dataset_id == "gv:stretching:gv_rad2_height14_28:bpress_-91__sweep_tot_force__values_500__750"
    assert manifest.hdf5_path.as_posix().endswith(
        "_runs/gv/launches/gv-campaign-sweep/datasets/gv/stretching/gv_rad2_height14_28/bpress_-91__sweep_tot_force__values_500__750/numerical_dataset.h5"
    )
    assert [run.output_id for run in manifest.runs] == [
        "bpress_-91__tot_force_500",
        "bpress_-91__tot_force_750",
    ]
    assert [run.dataset_id for run in manifest.runs] == [
        "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_500",
        "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_750",
    ]
    assert manifest.hdf5_path != manifest.runs[0].hdf5_path
    assert manifest.to_manifest()["runs"][1]["hdf5_path"].endswith(
        "_runs/gv/launches/gv-campaign-sweep/datasets/gv/stretching/gv_rad2_height14_28/bpress_-91__tot_force_750/numerical_dataset.h5"
    )


def test_build_gv_launch_campaign_manifest_stops_at_identity_and_submission_free_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-pure",
        walltime="01:30:00",
        gpu_count=2,
        provenance_tags={"linear_issue": "MES-177"},
    )

    manifest = build_gv_launch_campaign_manifest(request).to_manifest()

    assert "scheduler" not in manifest
    assert "submission" not in manifest
    assert "slurm" not in manifest
    assert not (tmp_path / "_runs" / "gv" / "launches" / "gv-campaign-pure").exists()
