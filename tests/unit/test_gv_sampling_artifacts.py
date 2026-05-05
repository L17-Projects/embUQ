from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_coerce_numeric_and_channel_branches_cover_validation_errors() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        artifacts._coerce_float("not-a-number", name="controls[x]")

    with pytest.raises(ValueError, match="must be finite"):
        artifacts._coerce_float(float("nan"), name="controls[x]")

    with pytest.raises(ValueError, match="controls are required"):
        artifacts._coerce_controls(None, fixture_like=None)

    with pytest.raises(ValueError, match="fixture_like must provide controls under 'controls'"):
        artifacts._coerce_controls(None, fixture_like={"controls": [1, 2]})

    assert artifacts._coerce_controls(None, fixture_like={"controls": {"tot_force": "600", "bpress": -91}}) == {
        "tot_force": 600.0,
        "bpress": -91.0,
    }

    with pytest.raises(ValueError, match="At least one control value is required"):
        artifacts._coerce_controls({}, fixture_like=None)

    with pytest.raises(ValueError, match="Either --fixture-path or --channels must be provided"):
        artifacts._coerce_channels(None, fixture_like=None, experiment="stretching")

    channels = artifacts._coerce_channels(
        {"tot_force": [5.0], "force": [1.0], "displacement": [0.0]},
        fixture_like=None,
        experiment="stretching",
    )
    assert set(channels.keys()) == {"tot_force", "force", "displacement"}

    parsed_stretching = artifacts._coerce_channels(
        None,
        fixture_like={
            "tot_force": [5.0],
            "force": [1.0],
            "displacement": [0.0],
        },
        experiment="stretching",
    )
    assert set(parsed_stretching.keys()) == {"tot_force", "force", "displacement"}

    parsed_buckling = artifacts._coerce_channels(
        None,
        fixture_like={
            "buckling_response": [1.0],
            "force_response": [1.0],
            "pressure_response": [1.0],
            "shape_amplitude": [1.0],
            "deformation_amplitude": [1.0],
        },
        experiment="buckling",
    )
    assert set(parsed_buckling.keys()) == {
        "buckling_response",
        "force_response",
        "pressure_response",
        "shape_amplitude",
        "deformation_amplitude",
    }

    with pytest.raises(ValueError, match="No fixture lane parser is available"):
        artifacts._coerce_channels(
            None,
            fixture_like={"tot_force": [1.0]},
            experiment="mystery",
        )


def test_manifest_payload_helpers_cover_object_paths_and_errors():
    with pytest.raises(ValueError, match="manifest\\.to_manifest\\(\\) must return a mapping"):
        artifacts._as_manifest_payload(SimpleNamespace(to_manifest=lambda: [1, 2, 3]))

    with pytest.raises(ValueError, match="Manifest must be a mapping or expose a \\.to_manifest\\(\\) method"):
        artifacts._as_manifest_payload(1.23)


def test_extract_sampling_payload_supports_object_payloads_and_optional_paths():
    class _Result:
        def __init__(self) -> None:
            self.manifest = SimpleNamespace(
                to_manifest=lambda: {
                    "campaign_id": "campaign-artifacts",
                    "experiment": "stretching",
                    "dataset_id": "gv:stretching:gv_rad2_height14_28:tot_force_500__bpress_-91",
                }
            )
            self.channels = {"tot_force": [500.0], "force": [1.0], "displacement": [0.0]}
            self.manifest_path = "existing-manifest.json"
            self.hdf5_path = "existing-dataset.h5"

    payload = _Result()
    manifest, channels, manifest_path, hdf5_path = artifacts._extract_sampling_payload(payload)

    assert manifest["campaign_id"] == "campaign-artifacts"
    assert set(channels.keys()) == {"tot_force", "force", "displacement"}
    assert isinstance(manifest_path, Path)
    assert isinstance(hdf5_path, Path)

    with pytest.raises(ValueError, match="Sampling payload object must provide manifest and channels"):
        artifacts._extract_sampling_payload(SimpleNamespace(manifest={"campaign_id": "x"}))


def test_extract_sampling_payload_rejects_mapping_without_manifest_or_channels():
    with pytest.raises(ValueError, match="Sampling payload must provide manifest and channels"):
        artifacts._extract_sampling_payload({"manifest": {"campaign_id": "x"}})


def test_optional_path_and_verify_with_explicit_hdf5_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert artifacts._optional_path(None) is None
    assert artifacts._optional_path("relative/path.h5") == Path("relative/path.h5")
    monkeypatch.chdir(tmp_path)
    store = _install_fake_h5py(monkeypatch)

    result = artifacts.sample_gv(
        campaign_id="campaign-sampling-artifacts-explicit",
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

    explicit = tmp_path / "explicit.h5"
    store[explicit] = dict(store[result.hdf5_path])
    assert artifacts.verify_sampling_artifacts_match(
        {
            "manifest": result.manifest,
            "channels": result.channels,
        },
        hdf5_path=explicit,
    )


def test_verify_sampling_artifacts_strictness_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    store = _install_fake_h5py(monkeypatch)

    result = artifacts.sample_gv(
        campaign_id="campaign-sampling-artifacts-mismatch-checks",
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_MATERIAL_PARAMETERS,
        controls=_CONTROLS,
        channels={
            "tot_force": [500.0],
            "force": [1.0],
            "displacement": [0.0],
        },
    )
    base_store = {key: value.copy() for key, value in store[result.hdf5_path].items()}

    with pytest.raises(ValueError, match="extra in HDF5"):
        store[result.hdf5_path] = {key: value.copy() for key, value in base_store.items()}
        store[result.hdf5_path]["extra"] = np.array([9.0])
        artifacts.verify_sampling_artifacts_match(result, strict=True)

    store[result.hdf5_path] = {key: value.copy() for key, value in base_store.items()}
    with pytest.raises(ValueError, match="missing in HDF5.*extra in HDF5"):
        store[result.hdf5_path]["extra"] = np.array([9.0])
        del store[result.hdf5_path]["displacement"]
        artifacts.verify_sampling_artifacts_match(result, strict=True)

    store[result.hdf5_path] = {key: value.copy() for key, value in base_store.items()}
    assert artifacts.verify_sampling_artifacts_match(result, strict=False)

    store[result.hdf5_path] = {key: value.copy() for key, value in base_store.items()}
    del store[result.hdf5_path]["force"]
    with pytest.raises(ValueError, match="missing in HDF5"):
        artifacts.verify_sampling_artifacts_match(result, strict=True)

    store[result.hdf5_path] = {key: value.copy() for key, value in base_store.items()}
    del store[result.hdf5_path]["force"]
    with pytest.raises(ValueError, match="Missing channels in HDF5"):
        artifacts.verify_sampling_artifacts_match(result, strict=False)


def test_assert_sampling_artifacts_match_propagates_assertion_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    def _always_false(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(artifacts, "verify_sampling_artifacts_match", _always_false)

    with pytest.raises(AssertionError, match="Sampling artifact verification failed"):
        artifacts.assert_sampling_artifacts_match(
            {
                "manifest": {
                    "campaign_id": "x",
                    "experiment": "stretching",
                    "dataset_id": "x",
                },
                "channels": {"tot_force": np.array([1.0])},
            }
        )


def test_sample_gv_with_write_artifacts_disabled_bypasses_writer(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.chdir(tmp_path)

        def _fail_if_called(*_args, **_kwargs) -> None:
            raise AssertionError("Writer should not run when write_artifacts=False")

        monkeypatch.setattr(artifacts, "write_sampling_artifacts", _fail_if_called)

        result = artifacts.sample_gv(
            campaign_id="campaign-sampling-artifacts-write-disabled",
            experiment="stretching",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_MATERIAL_PARAMETERS,
            controls=_CONTROLS,
            channels={"tot_force": [500.0], "force": [1.0], "displacement": [0.0]},
            write_artifacts=False,
        )

        assert result.manifest_path is None
        assert result.hdf5_path is None
    finally:
        monkeypatch.undo()
