from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.postprocessing import stretching
from meso_uq.structures.gv.postprocessing import common


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


def _fake_h5py(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(file=None)

    class FakeH5File:
        def __init__(self, path, *_args, **_kwargs):
            self.path = Path(path)
            self.attrs: dict[str, object] = {}
            self.datasets: dict[str, object] = {}
            state.file = self

        def __enter__(self):
            return self

        def __exit__(self, *_) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("h5 placeholder", encoding="utf-8")

        def create_dataset(self, name: str, data, *_args, **_kwargs) -> None:
            self.datasets[name] = np.asarray(data)

    class FakeH5:
        File = FakeH5File

    def _import_fake() -> object:
        return FakeH5

    monkeypatch.setattr(common, "_import_h5py", _import_fake)
    return state


def test_parse_stretching_fixture_channels_includes_required_and_optional() -> None:
    parsed = {
        "channels": {
            "tot_force": [500, 1000],
            "force": [5.0, 6.0],
            "displacement": [0.1, 0.3],
            "extra_metric": [1.0, 2.0],
        }
    }

    channels = stretching.parse_stretching_fixture_channels(parsed)

    assert "tot_force" in channels
    assert "force" in channels
    assert "displacement" in channels
    assert "extra_metric" in channels


def test_parse_stretching_fixture_channels_rejects_missing_required_channels() -> None:
    parsed = {
        "channels": {
            "tot_force": [500, 1000],
            "force": [5.0, 6.0],
        }
    }

    with pytest.raises(ValueError, match="Stretching parser requires at least channels"):
        stretching.parse_stretching_fixture_channels(parsed)


def test_process_stretching_numerical_dataset_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = _fake_h5py(monkeypatch)

    fixture = {
        "channels": {
            "tot_force": [500.0, 1000.0],
            "force": [10.0, 20.0],
            "displacement": [0.1, 0.2],
            "extra_metric": [3.0, 4.0],
        },
    }

    result = stretching.process_stretching_numerical_dataset(
        campaign_id="campaign-stretch",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"tot_force": 500.0, "bpress": -91.0},
        raw_provenance={"source": "fixture-like"},
        fixture_like=fixture,
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )

    assert result.manifest["experiment"] == "stretching"
    assert result.manifest_path.exists()
    assert result.hdf5_path.exists()
    assert result.manifest["dataset_id"].startswith("gv:stretching:")
    assert result.hdf5_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.manifest_path.as_posix().startswith("_runs/gv/numerical_data")
    assert state.file is not None
    assert set(state.file.datasets) == {"tot_force", "force", "displacement", "extra_metric"}
    assert "manifest_json" in state.file.attrs

    # Ensure optional channels are preserved.
    assert "tot_force" in fixture["channels"]
