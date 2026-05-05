from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from meso_uq.structures.gv import sampling as sampling_module
from meso_uq.structures.gv.sampling import (
    GVMaterialGeometry,
    GVRuntimeOptions,
    GVSampleResult,
    GVSweep,
    validate_sample_gv_request,
    sample_gv,
    validate_explicit_controls,
    validate_gv_experiment,
    validate_geometry,
)
import pytest


_VALID_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 0.9,
    "mu": 0.7,
    "b1": 0.2,
    "b2": 0.3,
    "a3": 0.4,
    "a4": 0.5,
    "mu_l": 0.6,
    "c": 0.8,
}


def test_material_geometry_requires_positive_finite_shape_inputs() -> None:
    geometry = GVMaterialGeometry(radGV=2.0, height=14.28)
    assert geometry.radGV == 2.0
    assert geometry.height == 14.28

    with pytest.raises(ValueError, match="radGV must be positive"):
        GVMaterialGeometry(radGV=-1.0, height=14.28)

    with pytest.raises(ValueError, match="height must be finite"):
        GVMaterialGeometry(radGV=2.0, height=float("inf"))


def test_material_geometry_rejects_nonnumeric_radius() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        GVMaterialGeometry(radGV="bad", height=14.28)


def test_material_geometry_rejects_negative_height() -> None:
    with pytest.raises(ValueError, match="height must be positive"):
        GVMaterialGeometry(radGV=2.0, height=-1.0)


def test_material_geometry_output_root_default() -> None:
    geometry = GVMaterialGeometry(radGV=2.0, height=14.28)
    assert geometry.output_root_default == Path("_runs/gv/sampling")


def test_gv_sweep_validates_axis_and_finite_non_empty_values() -> None:
    sweep = GVSweep(axis="tot_force", values=(10.0, 20.0))
    assert sweep.axis == "tot_force"
    assert sweep.values == (10.0, 20.0)

    with pytest.raises(ValueError, match="cannot be empty"):
        GVSweep(axis="tot_force", values=())

    with pytest.raises(ValueError, match="finite"):
        GVSweep(axis="tot_force", values=(float("nan"),))


def test_validate_explicit_controls_requires_one_sweep_axis() -> None:
    fixed, sweep = validate_explicit_controls(
        experiment="stretching",
        controls={"tot_force": (500.0, 1000.0), "bpress": -91.0},
    )

    assert fixed == {"bpress": -91.0}
    assert sweep.axis == "tot_force"
    assert sweep.values == (500.0, 1000.0)

    with pytest.raises(ValueError, match="explicit sweep axis"):
        validate_explicit_controls(experiment="stretching", controls={"tot_force": -91.0})

    with pytest.raises(ValueError, match="not valid"):
        validate_explicit_controls(
            experiment="stretching",
            controls={"tot_force": (1.0, 2.0), "bad_control": 1.0},
        )


def test_validate_explicit_controls_rejects_non_mapping_input() -> None:
    with pytest.raises(ValueError, match="GV controls must be a mapping"):
        validate_explicit_controls(experiment="stretching", controls="tot_force")


def test_validate_explicit_controls_rejects_multiple_sweep_axes() -> None:
    with pytest.raises(ValueError, match="exactly one explicit sweep axis"):
        validate_explicit_controls(
            experiment="stretching",
            controls={"tot_force": (500.0, 750.0), "bpress": (100.0, 200.0)},
        )


def test_validate_explicit_controls_rejects_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="Unknown GV experiment"):
        validate_explicit_controls(experiment="invalid", controls={"tot_force": (1.0, 2.0)})


def test_validate_explicit_controls_rejects_non_numeric_control() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        validate_explicit_controls(experiment="stretching", controls={"tot_force": "bad"})


def test_validate_explicit_controls_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_explicit_controls(experiment="stretching", controls={"tot_force": (float("nan"),)})


def test_validate_gv_experiment_rejects_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="Unsupported GV experiment"):
        validate_gv_experiment("not_an_experiment")


def test_validate_geometry_rejects_non_mapping_or_missing_fields() -> None:
    with pytest.raises(ValueError, match="GV geometry must be a GVMaterialGeometry or mapping"):
        validate_geometry("not-mapping", radGV=None, height=None)


def test_validate_geometry_rejects_mixed_direct_and_explicit_fields() -> None:
    with pytest.raises(ValueError, match="either geometry directly or explicit"):
        validate_geometry(GVMaterialGeometry(2.0, 14.28), radGV=2.0)


def test_validate_geometry_accepts_mapping_aliases() -> None:
    geometry = validate_geometry({"geometry_height": 14.28}, radGV=2.0)
    assert geometry.radGV == 2.0
    assert geometry.height == 14.28

    with pytest.raises(ValueError, match="GV geometry radGV is required"):
        validate_geometry(None, height=14.28)


def test_validate_gv_experiment_rejects_shear_flow() -> None:
    with pytest.raises(ValueError, match="shear_flow"):
        validate_gv_experiment("shear_flow")


def test_validate_sample_gv_request_prefers_runtime_options_controls_when_missing() -> None:
    experiment, _, _, controls, sweep, runtime_options = validate_sample_gv_request(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
        controls=None,
        runtime_options=GVRuntimeOptions(controls={"tot_force": (500.0, 750.0), "bpress": -91.0}),
        radGV=None,
        height=None,
    )
    assert experiment == "stretching"
    assert controls == {"bpress": -91.0}
    assert sweep.axis == "tot_force"
    assert sweep.values == (500.0, 750.0)
    assert isinstance(runtime_options.output_root, str)


def test_sample_gv_dry_run_builds_runtime_plan_without_execution() -> None:
    result = sample_gv(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
        controls={"tot_force": (500.0,), "bpress": -91.0},
        runtime_options=GVRuntimeOptions(output_root="_runs/gv/test-sampling-types"),
        dry_run=True,
    )

    assert result.status == "planned"
    assert result.channels == {}
    assert len(result.runtime_manifests) == 1
    assert len(result.plan_manifests) == 1

    with pytest.raises(ValueError, match="not supported for sampling"):
        sample_gv(
            experiment="shear_flow",
            material_parameters=_VALID_MATERIAL_PARAMETERS,
            geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
            controls={"afsi": (0.01, 0.02), "ptan": 0.1, "bpress": -10.0},
            dry_run=True,
        )


def test_sample_gv_executes_each_sweep_value_and_merges_channels(monkeypatch, tmp_path) -> None:
    calls = []

    class _Runtime:
        def __init__(self, controls):
            self.controls = controls
            self.work_dir = str(tmp_path / f"work-{controls['tot_force']}")

        def to_manifest(self):
            return {
                "controls": dict(self.controls),
                "work_dir": self.work_dir,
                "commands": [
                    {
                        "argv": [
                            "python3",
                            "generate.py",
                            "-p",
                            "tot_force",
                            str(self.controls["tot_force"]),
                            str(self.controls["tot_force"]),
                            "1",
                            "-p",
                            "bpress",
                            "-91",
                            "-91",
                            "1",
                        ],
                        "cwd": self.work_dir,
                    },
                    {"argv": ["bash", "commands.txt"], "cwd": self.work_dir},
                ],
            }

    def _fake_plan_runtime(*_args, controls, **_kwargs):
        calls.append(float(controls["tot_force"]))
        return _Runtime(controls)

    def _fake_execute(plan, **_kwargs):
        return SimpleNamespace(executed_commands=("ok",), return_codes=(0,), plan=plan.to_manifest())

    def _fake_extract(*, controls, **_kwargs):
        value = float(controls["tot_force"])
        return {"tot_force": np.asarray([value]), "force": np.asarray([value * 2.0])}

    monkeypatch.setattr(sampling_module, "plan_runtime", _fake_plan_runtime)
    monkeypatch.setattr(sampling_module, "execute_sampling_plan", _fake_execute)
    monkeypatch.setattr(sampling_module, "extract_sampling_channels", _fake_extract)

    result = sample_gv(
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
        controls={"tot_force": (500.0, 750.0), "bpress": -91.0},
        runtime_options=GVRuntimeOptions(output_root=tmp_path / "runtime"),
        write_artifacts=False,
    )

    assert result.status == "completed"
    assert calls == [500.0, 750.0]
    assert result.channels["force"].tolist() == [1000.0, 1500.0]
    assert len(result.execution_manifests) == 2


def test_sample_gv_write_artifacts_records_manifest_and_paths(monkeypatch, tmp_path) -> None:
    manifest_calls = []

    class _Runtime:
        def __init__(self, controls):
            self.controls = controls
            self.work_dir = str(tmp_path / "runtime" / "work")

        def to_manifest(self):
            return {
                "controls": dict(self.controls),
                "work_dir": self.work_dir,
                "commands": [
                    {
                        "argv": [
                            "python3",
                            "generate.py",
                            "-p",
                            "tot_force",
                            str(self.controls["tot_force"]),
                            str(self.controls["tot_force"]),
                            "1",
                            "-p",
                            "bpress",
                            "-91",
                            "-91",
                            "1",
                        ],
                        "cwd": self.work_dir,
                    },
                    {"argv": ["bash", "commands.txt"], "cwd": self.work_dir},
                ],
            }

    def _fake_plan_runtime(*_args, controls, **_kwargs):
        return _Runtime(controls)

    def _fake_execute(plan, **_kwargs):
        return SimpleNamespace(executed_commands=("generate", "run"), return_codes=(0, 0), plan=plan.to_manifest())

    def _fake_extract(*, controls, **_kwargs):
        value = float(controls["tot_force"])
        return {"tot_force": np.asarray([value]), "force": np.asarray([value * 2.0])}

    def _fake_manifest(**kwargs):
        manifest_calls.append(kwargs)
        return {
            "campaign_id": kwargs["campaign_id"],
            "experiment": kwargs["experiment"],
            "dataset_id": "gv:stretching:gv_rad2_height14_28:bpress_-91__tot_force_500",
        }

    def _fake_write(payload):
        assert payload["channels"]["force"].tolist() == [1000.0]
        return SimpleNamespace(
            manifest=dict(payload["manifest"]),
            manifest_path=tmp_path / "manifest.json",
            hdf5_path=tmp_path / "numerical_dataset.h5",
        )

    monkeypatch.setattr(sampling_module, "plan_runtime", _fake_plan_runtime)
    monkeypatch.setattr(sampling_module, "execute_sampling_plan", _fake_execute)
    monkeypatch.setattr(sampling_module, "extract_sampling_channels", _fake_extract)
    monkeypatch.setattr(sampling_module, "build_gv_numerical_dataset_manifest", _fake_manifest)
    monkeypatch.setattr(sampling_module, "write_sampling_artifacts", _fake_write)

    result = sample_gv(
        campaign_id="campaign-write-artifacts",
        experiment="stretching",
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
        controls={"tot_force": (500.0,), "bpress": -91.0},
        runtime_options=GVRuntimeOptions(output_root=tmp_path / "runtime"),
    )

    assert result.status == "completed"
    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.hdf5_path == tmp_path / "numerical_dataset.h5"
    assert result.manifest["campaign_id"] == "campaign-write-artifacts"
    assert manifest_calls[0]["controls"] == {"bpress": -91.0, "tot_force": 500.0}
    assert manifest_calls[0]["raw_provenance"]["sweep"] == {"axis": "tot_force", "values": [500.0]}
    assert manifest_calls[0]["quality_flags"]["finite_observables"] is True


def test_sample_result_manifest_includes_status_and_payload() -> None:
    geometry = GVMaterialGeometry(radGV=2.0, height=14.28)
    result = GVSampleResult(
        experiment="stretching",
        geometry=geometry,
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        runtime_options=GVRuntimeOptions(controls={"bpress": -91.0}),
        sweep=GVSweep(axis="tot_force", values=(500.0, 700.0)),
    )
    manifest = result.as_manifest()

    assert manifest["experiment"] == "stretching"
    assert manifest["geometry"]["radGV"] == 2.0
    assert manifest["sweep"]["axis"] == "tot_force"


def test_validate_gv_runtime_options_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="GV runtime timeout_seconds must be an integer"):
        GVRuntimeOptions(timeout_seconds=1.5)


def test_gv_runtime_options_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GVRuntimeOptions(timeout_seconds=0)


def test_gv_sweep_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        GVSweep(axis="tot_force", values=("bad",))


def test_gv_sweep_rejects_blank_axis() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        GVSweep(axis="   ", values=(1.0,))


def test_sample_result_manifest_includes_runtime_seconds_and_status() -> None:
    result = GVSampleResult(
        experiment="stretching",
        geometry=GVMaterialGeometry(radGV=2.0, height=14.28),
        material_parameters=_VALID_MATERIAL_PARAMETERS,
        runtime_options=GVRuntimeOptions(controls={"bpress": -91.0}),
        sweep=GVSweep(axis="tot_force", values=(500.0, 700.0)),
        runtime_seconds=3.5,
        status="completed",
    )
    manifest = result.as_manifest()
    assert manifest["status"] == "completed"
    assert manifest["runtime_seconds"] == 3.5
