from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.postprocessing import common
from meso_uq.structures.gv.sampling import artifacts

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

_CONTROLS = {"tot_force": 600.0, "bpress": -91.0}


def _install_fake_h5py(monkeypatch: pytest.MonkeyPatch):
    store: dict[Path, dict[str, np.ndarray]] = {}

    class _FakeHandle:
        def __init__(self, path: Path, mode: str):
            self.path = Path(path)
            self.mode = mode
            self.attrs: dict[str, object] = {}
            if "r" in mode:
                if self.path not in store:
                    raise FileNotFoundError(self.path)
                self._datasets = {name: np.array(values) for name, values in store[self.path].items()}
            else:
                self._datasets: dict[str, np.ndarray] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            if "w" in self.mode:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text("h5 placeholder", encoding="utf-8")
                store[self.path] = {name: np.array(values) for name, values in self._datasets.items()}

        def create_dataset(self, name: str, data, *_args, **_kwargs) -> None:
            self._datasets[name] = np.asarray(data)

        def keys(self):
            return self._datasets.keys()

        def __getitem__(self, key: str):
            return self._datasets[key]

    class _FakeH5:
        File = _FakeHandle

    monkeypatch.setattr(common, "_import_h5py", lambda: _FakeH5)
    monkeypatch.setattr(artifacts, "_import_h5py", lambda: _FakeH5)
    return store


def test_write_sampling_artifacts_and_verify_payloads_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = _install_fake_h5py(monkeypatch)

    result = artifacts.sample_gv(
        campaign_id="campaign-sampling-artifacts",
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_MATERIAL_PARAMETERS,
        controls=_CONTROLS,
        channels={
            "tot_force": [500.0, 700.0],
            "force": [1.0, 2.0],
            "displacement": [0.0, 0.1],
        },
    )

    assert result.manifest_path is not None
    assert result.hdf5_path is not None
    assert result.hdf5_path.as_posix().startswith("_runs/")
    assert result.manifest_path.as_posix().startswith("_runs/")
    assert set(store[result.hdf5_path].keys()) == {"tot_force", "force", "displacement"}
    artifacts.assert_sampling_artifacts_match(result)
    assert result.hdf5_path.exists()
    assert result.manifest_path.exists()


def test_verify_sampling_artifacts_reports_channel_differences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _install_fake_h5py(monkeypatch)

    result = artifacts.sample_gv(
        campaign_id="campaign-sampling-artifacts-mismatch",
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_MATERIAL_PARAMETERS,
        controls=_CONTROLS,
        channels={"tot_force": [500.0], "force": [1.0], "displacement": [0.0]},
    )

    with pytest.raises(ValueError, match="differs between result payload and HDF5"):
        artifacts.verify_sampling_artifacts_match(
            {
                "manifest": result.manifest,
                "channels": {
                    "tot_force": np.array([500.0]),
                    "force": np.array([9.0]),
                    "displacement": np.array([0.0]),
                },
            }
        )


def test_sample_gv_dry_run_does_not_emit_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = artifacts.sample_gv(
        campaign_id="campaign-sampling-artifacts-dry-run",
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_MATERIAL_PARAMETERS,
        controls=_CONTROLS,
        channels={"tot_force": [500.0], "force": [1.0], "displacement": [0.0]},
        dry_run=True,
    )

    assert result.manifest is not None
    assert result.manifest_path is None
    assert result.hdf5_path is None
