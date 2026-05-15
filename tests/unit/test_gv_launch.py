from __future__ import annotations

import json
from pathlib import Path

import pytest

from meso_uq.structures.gv.launch import (
    GVLaunchRequest,
    build_gv_launch_campaign_manifest,
    render_gv_launch_campaign,
    render_gv_launch_campaigns,
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
        (
            {"controls": {"tot_force": (500.0, 500.0), "bpress": -91.0}},
            "sweep values must produce unique formatted output ids",
        ),
        (
            {"controls": {"tot_force": (500.000000000001, 500.000000000002), "bpress": -91.0}},
            "sweep values must produce unique formatted output ids",
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


@pytest.mark.parametrize("output_root", ["src", "tests", "scripts", "gv", "gv_simulation_files", "src/generated"])
def test_validate_gv_launch_request_rejects_repo_source_output_roots(output_root: str) -> None:
    with pytest.raises(ValueError, match="output_root must not be"):
        validate_gv_launch_request(
            experiment="stretching",
            material_parameters=_VALID_MATERIAL_PARAMETERS,
            geometry={"radGV": 2.0, "height": 14.28},
            controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
            platform="vega",
            output_root=output_root,
            walltime="01:30:00",
            gpu_count=2,
            provenance_tags={"linear_issue": "MES-177"},
        )


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


def test_render_gv_launch_campaign_materializes_karolina_and_vega_without_submission(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-render",
        walltime="01:30:00",
        gpu_count=2,
        provenance_tags={"linear_issue": "MES-178", "source": "unit-test"},
    )

    rendered = render_gv_launch_campaign(request, platforms=("karolina", "vega"))

    assert rendered.campaign_dir == Path("_runs/gv/launches/gv-campaign-render")
    assert rendered.manifest_path.is_file()
    assert [script.platform.value for script in rendered.scheduler_scripts] == ["karolina", "vega"]
    payload = json.loads(rendered.manifest_path.read_text(encoding="utf-8"))
    assert payload["render_manifest_schema_version"] == 1
    assert payload["submission"] == {"submitted": False, "submission_commands": []}
    assert payload["platform_renderers"] == [
        {
            "platform": "karolina",
            "renderer": "gv_launch_renderer:karolina_slurm",
            "scheduler": "slurm",
        },
        {
            "platform": "vega",
            "renderer": "gv_launch_renderer:vega_slurm",
            "scheduler": "slurm",
        },
    ]
    assert payload["expected_hdf5_datasets"]["campaign"]["hdf5_path"].endswith(
        "_runs/gv/launches/gv-campaign-render/datasets/gv/stretching/gv_rad2_height14_28/bpress_-91__sweep_tot_force__values_500__750/numerical_dataset.h5"
    )
    assert len(payload["expected_hdf5_datasets"]["runs"]) == 2
    assert payload["generated_script_paths"] == [
        "_runs/gv/launches/gv-campaign-render/scripts/karolina/gv-campaign-render.sbatch",
        "_runs/gv/launches/gv-campaign-render/scripts/vega/gv-campaign-render.sbatch",
    ]

    karolina_script = Path(payload["generated_script_paths"][0]).read_text(encoding="utf-8")
    vega_script = Path(payload["generated_script_paths"][1]).read_text(encoding="utf-8")
    assert "#SBATCH --account=eu-26-17" in karolina_script
    assert "#SBATCH --partition=qgpu" in karolina_script
    assert "#SBATCH --gpus=2" in karolina_script
    assert 'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"' in karolina_script
    assert "scripts/platforms/karolina/run_gv_runtime.py" in karolina_script
    assert "#SBATCH --partition=gpu" in vega_script
    assert "#SBATCH --gres=gpu:2" in vega_script
    assert "module purge" in vega_script
    assert "_vega/gv_venv/env.sh" in vega_script
    assert "scripts/platforms/vega/run_gv_runtime.py" in vega_script
    assert 'export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"' in vega_script
    assert 'export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"' in vega_script
    for script in (karolina_script, vega_script):
        lowered = script.lower()
        assert "qsub" not in lowered
        assert " sbatch " not in lowered
        assert "--control bpress=-91.0 --control tot_force=500.0" in script
        assert "--control bpress=-91.0 --control tot_force=750.0" in script
        assert "--material ka=1.1" in script
        assert "--array=0-1" in script


def test_render_gv_launch_campaign_fails_on_existing_campaign_dir_unless_overwrite(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0,), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-repeat",
        walltime="01:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-178"},
    )
    first = render_gv_launch_campaign(request)
    first_payload = first.manifest_path.read_text(encoding="utf-8")
    first_script = first.scheduler_scripts[0].script_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="campaign directory already exists"):
        render_gv_launch_campaign(request)

    second = render_gv_launch_campaign(request, overwrite=True)
    assert second.manifest_path.read_text(encoding="utf-8") == first_payload
    assert second.scheduler_scripts[0].script_path.read_text(encoding="utf-8") == first_script


def test_render_gv_launch_campaigns_accepts_batch_of_selected_requests(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0,), "bpress": -91.0},
        platform="karolina",
        output_root="_runs/gv/launches/gv-campaign-batch-a",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-178"},
    )
    second = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (750.0,), "bpress": -91.0},
        platform="vega",
        output_root="_runs/gv/launches/gv-campaign-batch-b",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-181"},
    )

    rendered = render_gv_launch_campaigns((first, second))

    assert [item.request.campaign_id for item in rendered] == [
        "gv-campaign-batch-a",
        "gv-campaign-batch-b",
    ]
    assert rendered[0].scheduler_scripts[0].platform.value == "karolina"
    assert rendered[1].scheduler_scripts[0].platform.value == "vega"
    assert rendered[0].manifest_path.is_file()
    assert rendered[1].manifest_path.is_file()

    with pytest.raises(ValueError, match="duplicate campaign output roots"):
        render_gv_launch_campaigns((first, first), overwrite=True)


def test_render_gv_launch_campaign_rejects_generic_slurm_renderer(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    request = validate_gv_launch_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry={"radGV": 2.0, "height": 14.28},
        controls={"tot_force": (500.0,), "bpress": -91.0},
        platform="generic_slurm",
        output_root="_runs/gv/launches/gv-campaign-generic",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-181"},
    )

    with pytest.raises(ValueError, match="render platform must be one of: karolina, vega"):
        render_gv_launch_campaign(request)
