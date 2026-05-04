from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.postprocessing import buckling
from meso_uq.structures.gv.postprocessing import common
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


def test_parse_buckling_fixture_channels_includes_aliases_and_trajectory_candidates() -> None:
    fixture = {
        "channels": {
            "buck": [0.0, 0.25, 0.5],
            "bpress": [-91.0, -91.5, -92.0],
            "response": [1.0, 0.9, 0.8],
            "pressure response": [1.0, 0.95, 0.9],
            "shape amplitude": [0.0, 0.2, 0.4],
            "mesh": {"sim0": [1.0, 2.0, 3.0]},
            "object": {"stats": {"mean_area": [10.0, 11.0, 12.0]}},
            "extra_metric": [2.0, 3.0, 4.0],
        },
    }

    channels = buckling.parse_buckling_fixture_channels(fixture)

    assert np.array_equal(channels["buck"], np.array([0.0, 0.25, 0.5]))
    assert np.array_equal(channels["bpress"], np.array([-91.0, -91.5, -92.0]))
    assert np.array_equal(channels["buckling_response"], np.array([1.0, 0.9, 0.8]))
    assert np.array_equal(channels["pressure_response"], np.array([1.0, 0.95, 0.9]))
    assert np.array_equal(channels["shape_amplitude"], np.array([0.0, 0.2, 0.4]))
    assert np.array_equal(channels["mesh_sim0"], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(channels["object_stats_mean_area"], np.array([10.0, 11.0, 12.0]))
    assert "extra_metric" in channels


def test_parse_buckling_fixture_channels_filters_optional_channels() -> None:
    fixture = {
        "channels": {
            "buck": [0.0, 0.25],
            "bpress": [-91.0, -91.5],
            "force response": [1.0, 0.9],
            "mesh": {"sim0": [1.0, 2.0]},
            "anchor": {"path": [3.0, 4.0]},
            "notes": ["a", "b"],
        },
    }

    channels = buckling.parse_buckling_fixture_channels(fixture, include_optional_channels=False)

    assert "buck" in channels
    assert "bpress" in channels
    assert "force_response" in channels
    assert "mesh_sim0" in channels
    assert "anchor_path" in channels
    assert "notes" not in channels


def test_parse_buckling_fixture_channels_preserves_slash_style_input() -> None:
    fixture = {
        "channels": {
            "buck": [0.0, 0.5],
            "bpress": [-91.0, -91.5],
            "buckling_response": [1.0, 0.5],
            "mesh/sim0": [1.0, 2.0],
            "object/stats": [3.0, 4.0],
        },
    }

    channels = buckling.parse_buckling_fixture_channels(fixture)

    assert np.array_equal(channels["mesh_sim0"], np.array([1.0, 2.0]))
    assert np.array_equal(channels["object_stats"], np.array([3.0, 4.0]))


def test_parse_buckling_fixture_channels_rejects_missing_response_channel() -> None:
    fixture = {
        "channels": {
            "buck": [0.0, 0.25],
            "bpress": [-91.0, -91.5],
            "notes": [1, 2],
        },
    }

    with pytest.raises(ValueError, match="Buckling parser requires at least one response channel"):
        buckling.parse_buckling_fixture_channels(fixture)


def test_buckling_helpers_handle_empty_names_and_skip_non_string_keys() -> None:
    assert buckling._canonical_channel_name("!!!") == ""

    channels = buckling.parse_buckling_fixture_channels(
        {
            "channels": {
                1: [99.0],
                "shape": [0.1, 0.2],
            }
        }
    )

    assert set(channels) == {"shape_amplitude"}


def test_parse_buckling_fixture_channels_accepts_root_channel_mapping() -> None:
    channels = buckling.parse_buckling_fixture_channels(
        {
            "buck": [0.0, 0.25],
            "response": [1.0, 0.8],
            "unregistered_optional": [5.0, 6.0],
        }
    )

    assert set(channels) == {"buck", "buckling_response", "unregistered_optional"}


def test_parse_buckling_fixture_channels_rejects_non_mapping_fixture() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        buckling.parse_buckling_fixture_channels(["not", "a", "mapping"])


def test_parse_buckling_fixture_channels_rejects_non_mapping_observables() -> None:
    with pytest.raises(ValueError, match="No numeric channel mapping found"):
        buckling.parse_buckling_fixture_channels({"observables": ["bad-payload"]})


def test_parse_buckling_fixture_channels_rejects_empty_candidate_mapping() -> None:
    with pytest.raises(ValueError, match="received no numeric channels"):
        buckling.parse_buckling_fixture_channels({"observables": {1: [1.0], 2: [2.0]}})


def test_parse_buckling_fixture_channels_rejects_non_finite_observables() -> None:
    fixture = {
        "channels": {
            "buck": [0.0, 0.25],
            "bpress": [-91.0, -91.5],
            "deformation": [0.0, np.nan],
        },
    }

    with pytest.raises(ValueError, match="non-finite"):
        buckling.parse_buckling_fixture_channels(fixture)


def test_process_buckling_numerical_dataset_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = _install_fake_h5py(monkeypatch)

    fixture = {
        "channels": {
            "buck": [0.0, 0.5],
            "bpress": [-91.0, -92.0],
            "buckling_response": [1.0, 0.5],
            "force_response": [8.0, 12.0],
            "mesh": {"sim0": [10.0, 20.0]},
            "ply": {"sim0": [5.0, 6.0]},
            "anchor": {"count": [4.0, 4.0]},
            "object": {"stats": {"mean": [1.0, 2.0]}},
            "notes": [1.0, 2.0],
        },
    }

    result = buckling.process_buckling_numerical_dataset(
        campaign_id="campaign-buckling",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"buck": 0.0, "bpress": -91.0},
        raw_provenance={"source": "fixture-like"},
        fixture_like=fixture,
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )

    assert result.manifest["experiment"] == "buckling"
    assert result.manifest["dataset_id"].startswith("gv:buckling:")
    assert result.hdf5_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.manifest_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.hdf5_path.exists()
    assert result.manifest_path.exists()

    assert set(state.file.datasets) == {
        "buck",
        "bpress",
        "buckling_response",
        "force_response",
        "mesh_sim0",
        "ply_sim0",
        "anchor_count",
        "object_stats_mean",
        "notes",
    }
    assert "manifest_json" in state.file.attrs


def test_process_buckling_numerical_dataset_requires_h5py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        common,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("missing h5py")),
    )

    fixture = {
        "channels": {
            "buck": [0.0],
            "bpress": [-91.0],
            "buckling_response": [1.0],
            "shape_amplitude": [0.3],
        },
    }

    with pytest.raises(RuntimeError, match="`h5py` is required"):
        buckling.process_buckling_numerical_dataset(
            campaign_id="campaign-buckling",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"buck": 0.0, "bpress": -91.0},
            raw_provenance={"source": "fixture-like"},
            fixture_like=fixture,
            quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
        )

    manifest = build_gv_numerical_dataset_manifest(
        campaign_id="campaign-buckling",
        structure="gv",
        experiment="buckling",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"buck": 0.0, "bpress": -91.0},
        raw_provenance={"source": "fixture-like"},
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )
    manifest_path = campaign_dataset_manifest_path(
        campaign_id="campaign-buckling",
        dataset_id=manifest.dataset_id,
    )
    assert not manifest_path.exists()
