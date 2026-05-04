from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.postprocessing import common
from meso_uq.structures.gv.postprocessing import torsion
from meso_uq.structures.gv.numerical_data import build_gv_numerical_dataset_manifest
from meso_uq.structures.gv.numerical_data import campaign_dataset_manifest_path


_BASE_MATERIAL_PARAMETERS = {
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


def _install_fake_h5py(monkeypatch: pytest.MonkeyPatch):
    recorded = SimpleNamespace(file=None)

    class FakeH5File:
        def __init__(self, path, *_args, **_kwargs):
            self.path = Path(path)
            self.attrs: dict[str, object] = {}
            self.datasets: dict[str, np.ndarray] = {}
            recorded.file = self

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("h5 placeholder", encoding="utf-8")

        def create_dataset(self, name: str, data, *_args, **_kwargs) -> None:
            self.datasets[name] = np.asarray(data)

    class FakeH5:
        File = FakeH5File

    monkeypatch.setattr(common, "_import_h5py", lambda: FakeH5)
    return recorded


def test_parse_torsion_fixture_channels_includes_aliases_and_candidate_channels() -> None:
    fixture = {
        "channels": {
            "theta": [0.0, 0.25, 0.5],
            "response": [1.0, 0.8, 0.6],
            "torsion_torque": [0.5, 0.6, 0.7],
            "cap_angle": [0.0, 0.05, 0.11],
            "stored_elastic_response_proxy": [0.2, 0.3, 0.4],
            "anchor_motion": [0.1, 0.2, 0.3],
            "tot_force": [10.0, 11.0, 12.0],
            "notes": [1, 2, 3],
        },
    }

    channels = torsion.parse_torsion_fixture_channels(fixture)

    assert np.array_equal(channels["torsion_coord"], np.array([0.0, 0.25, 0.5]))
    assert np.array_equal(channels["torsion_response"], np.array([1.0, 0.8, 0.6]))
    assert np.array_equal(channels["torsion_torque_proxy"], np.array([0.5, 0.6, 0.7]))
    assert np.array_equal(channels["cap_angular_displacement"], np.array([0.0, 0.05, 0.11]))
    assert np.array_equal(
        channels["stored_elastic_response"],
        np.array([0.2, 0.3, 0.4]),
    )
    assert np.array_equal(channels["cap_motion"], np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(channels["force"], np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(channels["notes"], np.array([1.0, 2.0, 3.0]))


def test_parse_torsion_fixture_channels_rejects_missing_required_channel() -> None:
    fixture = {"channels": {"theta": [0.0, 0.25]}}

    with pytest.raises(ValueError, match="Torsion parser requires at least channels"):
        torsion.parse_torsion_fixture_channels(fixture)


def test_torsion_helpers_cover_empty_names_observables_and_filtered_channels() -> None:
    assert torsion._canonicalize_channel_name("!!!") == ""

    channels = torsion.parse_torsion_fixture_channels(
        {
            "observables": {
                "theta": [0.0, 0.25],
                "response": [1.0, 0.8],
                "cap_angle": [0.0, 0.05],
                1: [99.0, 98.0],
            }
        },
        include_optional_channels=False,
    )

    assert set(channels) == {"torsion_coord", "torsion_response"}


def test_parse_torsion_fixture_channels_accepts_root_channel_mapping() -> None:
    channels = torsion.parse_torsion_fixture_channels(
        {
            "theta": [0.0, 0.25],
            "response": [1.0, 0.8],
            "cap_angle": [0.0, 0.05],
        }
    )

    assert set(channels) == {"torsion_coord", "torsion_response", "cap_angular_displacement"}


def test_parse_torsion_fixture_channels_rejects_non_mapping_fixture() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        torsion.parse_torsion_fixture_channels(["bad-payload"])


def test_parse_torsion_fixture_channels_rejects_non_mapping_observables() -> None:
    with pytest.raises(ValueError, match="No numeric channel mapping found"):
        torsion.parse_torsion_fixture_channels({"observables": ["bad-payload"]})


def test_parse_torsion_fixture_channels_rejects_non_finite_observables() -> None:
    fixture = {
        "channels": {
            "theta": [0.0, 0.25],
            "response": [1.0, np.nan],
        },
    }

    with pytest.raises(ValueError, match="non-finite"):
        torsion.parse_torsion_fixture_channels(fixture)


def test_quality_flags_for_torsion_channels_track_non_finite_values() -> None:
    flags = torsion._quality_flags_for_channels(
        {
            "torsion_coord": np.array([0.0, 0.25]),
            "torsion_response": np.array([1.0, np.nan]),
        }
    )

    assert flags["finite_observables"] is False
    assert flags["finite_observable_ratio"] == pytest.approx(0.75)
    assert flags["canary_failures"] == ("torsion_response:1 non-finite",)


def test_process_torsion_numerical_dataset_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = _install_fake_h5py(monkeypatch)

    fixture = {
        "channels": {
            "theta": [0.0, 0.25],
            "response": [1.0, 0.8],
            "torsion_torque": [0.5, 0.6],
            "cap_angle": [0.0, 0.05],
            "stored_elastic_response_proxy": [0.2, 0.3],
            "anchor_motion": [0.1, 0.2],
            "tot_force": [10.0, 11.0],
            "notes": [1.0, 2.0],
        },
    }

    result = torsion.process_torsion_numerical_dataset(
        campaign_id="campaign-torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": 0.03},
        raw_provenance={"source": "fixture-like"},
        fixture_like=fixture,
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )

    assert result.manifest["experiment"] == "torsion"
    assert result.manifest["dataset_id"].startswith("gv:torsion:")
    assert result.manifest["controls"] == {"theta": 0.03}
    assert "theta" in result.manifest["controls"]
    assert result.manifest_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.hdf5_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.hdf5_path.exists()
    assert result.manifest_path.exists()
    assert state.file is not None
    assert set(state.file.datasets) == {
        "torsion_coord",
        "torsion_response",
        "torsion_torque_proxy",
        "cap_angular_displacement",
        "stored_elastic_response",
        "cap_motion",
        "force",
        "notes",
    }


def test_process_torsion_numerical_dataset_rejects_material_control_collision() -> None:
    material_with_theta = dict(_BASE_MATERIAL_PARAMETERS, theta=0.03)

    with pytest.raises(ValueError, match="unexpected material parameters"):
        torsion.process_torsion_numerical_dataset(
            campaign_id="campaign-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=material_with_theta,
            controls={"theta": 0.03},
            raw_provenance={"source": "fixture-like"},
            fixture_like={"channels": {"theta": [0.0], "response": [1.0], "force": [2.0]}},
            quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
        )


def test_process_torsion_numerical_dataset_rejects_non_finite_channels_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torsion,
        "parse_torsion_fixture_channels",
        lambda *_, **__: {
            "torsion_coord": np.array([0.0, 0.25]),
            "torsion_response": np.array([1.0, np.nan]),
        },
    )

    with pytest.raises(ValueError, match="contain non-finite observables"):
        torsion.process_torsion_numerical_dataset(
            campaign_id="campaign-torsion-nonfinite",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"theta": 0.03},
            raw_provenance={"source": "fixture-like"},
            fixture_like={"channels": {"theta": [0.0], "response": [1.0]}},
        )


def test_process_torsion_numerical_dataset_requires_h5py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        common,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("missing h5py")),
    )

    fixture = {"channels": {"theta": [0.0], "response": [1.0], "force": [2.0]}}

    with pytest.raises(RuntimeError, match="`h5py` is required"):
        torsion.process_torsion_numerical_dataset(
            campaign_id="campaign-torsion",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"theta": 0.03},
            raw_provenance={"source": "fixture-like"},
            fixture_like=fixture,
            quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
        )

    manifest = build_gv_numerical_dataset_manifest(
        campaign_id="campaign-torsion",
        structure="gv",
        experiment="torsion",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"theta": 0.03},
        raw_provenance={"source": "fixture-like"},
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )
    assert not campaign_dataset_manifest_path(
        campaign_id="campaign-torsion",
        dataset_id=manifest.dataset_id,
    ).exists()
