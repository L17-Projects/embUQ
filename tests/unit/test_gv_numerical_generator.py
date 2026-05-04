from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.generator import generate_gv_numerical_data
from meso_uq.structures.gv import generator as generator_module
from meso_uq.structures.gv.geometries import DEFAULT_GV_GEOMETRY
from meso_uq.structures.gv.postprocessing import common


_MATERIAL_PARAMETERS = {
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


def _install_fake_h5py(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    recorded = SimpleNamespace(file=None)

    class FakeH5File:
        def __init__(self, path, *_args, **_kwargs):
            self.path = Path(path)
            self.attrs: dict[str, object] = {}
            self.datasets: dict[str, object] = {}
            recorded.file = self

        def __enter__(self):
            return self

        def __exit__(self, *_) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("h5 placeholder", encoding="utf-8")

        def create_dataset(self, name: str, data, *_args, **_kwargs) -> None:
            self.datasets[name] = list(data)

    class FakeH5Module:
        File = FakeH5File

    monkeypatch.setattr(common, "_import_h5py", lambda: FakeH5Module)
    return recorded


def _fixture_for_experiment(experiment: str) -> dict[str, object]:
    if experiment == "stretching":
        return {"channels": {"tot_force": [500.0, 600.0], "force": [1.0, 2.0], "displacement": [3.0, 4.0]}}
    if experiment == "buckling":
        return {"channels": {"buck": [0.0, 0.1], "force": [1.0, 2.0], "displacement": [0.1, 0.2]}}
    if experiment == "torsion":
        return {
            "channels": {
                "torsion_coord": [0.0, 0.1],
                "torsion_response": [1.0, 2.0],
                "force": [3.0, 4.0],
            }
        }
    if experiment == "eigenmodes":
        return {
            "channels": {
                "eigenvalue": [0.2, 0.3],
                "frequency": [1.0, 2.0],
            }
        }
    raise ValueError(f"No fixture available for experiment: {experiment}")


def test_generate_gv_numerical_data_stages_runtime_and_returns_expected_manifest_intent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = generate_gv_numerical_data(
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        campaign_id="campaign-runtime",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"tot_force": 750.0},
    )

    assert result.experiment == "stretching"
    assert result.geometry_id == DEFAULT_GV_GEOMETRY.id
    assert result.controls["tot_force"] == 750.0
    assert result.postprocess_result is None
    assert result.runtime_dry_run.geometry == result.geometry_id
    assert result.runtime_dry_run.output_root.startswith(str((tmp_path / "_runs" / "gv" / "runtime").resolve()))
    assert str(result.expected_dataset_id).startswith("gv:stretching:gv_rad2_height14_28:")
    assert result.expected_manifest_path.name == "numerical_dataset_manifest.json"
    assert result.expected_hdf5_path.name == "numerical_dataset.h5"
    assert "_runs/gv/numerical_data/campaign-runtime" in str(result.expected_manifest_path)
    assert not result.expected_hdf5_path.exists()
    assert result.raw_provenance["structure"] == "gv"
    assert result.expected_manifest["quality_flags"]["finite_observables"] is False
    assert result.expected_manifest["quality_flags"]["canary_failures"] == ["postprocessing_not_run"]


def test_generate_gv_numerical_data_accepts_radgv_and_height_aliases_with_explicit_controls(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = generate_gv_numerical_data(
        experiment="buckling",
        radGV=2.0,
        height=14.28,
        campaign_id="campaign-alias",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"buck": 0.0},
    )

    assert result.experiment == "buckling"
    assert result.geometry_id == "gv_rad2_height14_28"
    assert result.controls["buck"] == 0.0


def test_generate_gv_numerical_data_optional_postprocessing_uses_fixture_like_and_writes_hdf5(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    recorded = _install_fake_h5py(monkeypatch)

    result = generate_gv_numerical_data(
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        campaign_id="campaign-postprocess",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"tot_force": 750.0},
        fixture_like=_fixture_for_experiment("stretching"),
    )

    assert result.postprocess_result is not None
    assert recorded.file is not None
    assert result.expected_hdf5_path.as_posix().startswith("_runs/gv/numerical_data/campaign-postprocess/")
    assert "tot_force" in recorded.file.datasets
    assert result.postprocess_result.hdf5_path == result.expected_hdf5_path
    assert result.postprocess_result.manifest["dataset_id"] == result.expected_dataset_id
    assert result.postprocess_result.manifest_path == result.expected_manifest_path


def test_generate_gv_numerical_data_can_scan_multiple_experiments_and_geometries(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = generate_gv_numerical_data(
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        campaign_id="campaign-matrix",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"tot_force": 750.0},
    )
    second = generate_gv_numerical_data(
        experiment="buckling",
        geometry_radius=2.5,
        geometry_height=15.0,
        campaign_id="campaign-matrix",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"buck": 0.25},
    )

    assert first.experiment == "stretching"
    assert second.experiment == "buckling"
    assert first.geometry_id != second.geometry_id
    assert first.expected_dataset_id != second.expected_dataset_id
    assert first.expected_manifest_path.as_posix() != second.expected_manifest_path.as_posix()
    assert first.expected_hdf5_path.as_posix() != second.expected_hdf5_path.as_posix()


@pytest.mark.parametrize("experiment", ["shear_flow", "compression"])
def test_generate_gv_numerical_data_rejects_unsupported_experiments(experiment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unsupported GV experiment|not yet supported"):
        generate_gv_numerical_data(
            experiment=experiment,
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-bad",
            material_parameters=_MATERIAL_PARAMETERS,
        )


def test_generate_gv_numerical_data_rejects_invalid_material_controls_and_structure_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    incomplete_material = dict(_MATERIAL_PARAMETERS)
    incomplete_material.pop("mu_l")
    with pytest.raises(ValueError, match="Missing required GV material parameters"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-invalid-material",
            material_parameters=incomplete_material,
        )

    with pytest.raises(ValueError, match="Unknown controls for GV runtime dry-run"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-invalid-controls",
            material_parameters=_MATERIAL_PARAMETERS,
            controls={"ka": 1.1},
        )

    with pytest.raises(ValueError, match="control 'tot_force' must be finite"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-invalid-control-value",
            material_parameters=_MATERIAL_PARAMETERS,
            controls={"tot_force": float("inf")},
        )

    with pytest.raises(ValueError, match="controls are required"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-missing-controls",
            material_parameters=_MATERIAL_PARAMETERS,
        )

    with pytest.raises(ValueError, match="at least one explicit control"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-empty-controls",
            material_parameters=_MATERIAL_PARAMETERS,
            controls={},
        )

    with pytest.raises(ValueError, match="path separators"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign/bad",
            material_parameters=_MATERIAL_PARAMETERS,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"geometry_height": 14.28}, "Both geometry size components are required"),
        ({"geometry_radius": float("inf"), "geometry_height": 14.28}, "must be finite values"),
        ({"geometry_radius": 0.0, "geometry_height": 14.28}, "geometry_radius must be positive"),
        ({"geometry_radius": 2.0, "geometry_height": 0.0}, "geometry_height must be positive"),
        ({"geometry_radius": 2.0, "geometry_height": 14.28, "controls": []}, "controls must be a mapping"),
    ],
)
def test_generate_gv_numerical_data_rejects_bad_geometry_and_controls(
    kwargs, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base_kwargs = {
        "experiment": "stretching",
        "campaign_id": "campaign-invalid-shape",
        "material_parameters": _MATERIAL_PARAMETERS,
        "controls": {"tot_force": 750.0},
    }
    if "controls" in kwargs:
        base_kwargs.pop("controls")

    with pytest.raises(ValueError, match=message):
        generate_gv_numerical_data(**base_kwargs, **kwargs)


def test_generate_gv_numerical_data_preserves_explicit_quality_flags_for_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = generate_gv_numerical_data(
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        campaign_id="campaign-quality-flags",
        material_parameters=_MATERIAL_PARAMETERS,
        controls={"tot_force": 750.0},
        quality_flags={"finite_observables": True, "finite_observable_ratio": 0.5, "canary_failures": ["gpu_busy"]},
    )

    assert result.expected_manifest["quality_flags"] == {
        "finite_observables": True,
        "finite_observable_ratio": 0.5,
        "canary_failures": ["gpu_busy"],
    }


def test_generate_gv_numerical_data_rejects_non_mapping_fixture_like_before_postprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-fixture-type",
            material_parameters=_MATERIAL_PARAMETERS,
            controls={"tot_force": 750.0},
            fixture_like=[],
        )


def test_generate_gv_numerical_data_rejects_missing_postprocessor_for_fixture_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(generator_module.POSTPROCESSORS, "gv:stretching", raising=False)

    with pytest.raises(ValueError, match="No postprocessor is registered"):
        generate_gv_numerical_data(
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            campaign_id="campaign-missing-postprocessor",
            material_parameters=_MATERIAL_PARAMETERS,
            controls={"tot_force": 750.0},
            fixture_like=_fixture_for_experiment("stretching"),
        )
