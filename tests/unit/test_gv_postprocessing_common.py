from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.numerical_data import build_gv_numerical_dataset_manifest
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


def _base_manifest() -> dict:
    return build_gv_numerical_dataset_manifest(
        campaign_id="campaign-postprocess-common",
        structure="gv",
        experiment="stretching",
        geometry_radius=2.0,
        geometry_height=14.28,
        controls={"tot_force": 500.0, "bpress": -91.0},
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        raw_provenance={"mirheo_log": "gv/stretching/logs/mirheo.log"},
    ).to_manifest()


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

        def __exit__(self, *_) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("h5 placeholder", encoding="utf-8")

        def create_dataset(self, name: str, data, *_args, **_kwargs) -> None:
            self.datasets[name] = np.asarray(data)

    class FakeH5:
        File = FakeH5File

    monkeypatch.setattr(common, "_import_h5py", lambda: FakeH5)
    return recorded


def test_validate_numeric_channels_rejects_non_finite_arrays() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        common.validate_numeric_channels({"force": [1.0, np.nan]})


def test_write_numerical_dataset_artifacts_writes_manifest_and_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _base_manifest()

    monkeypatch.chdir(tmp_path)
    recorded = _install_fake_h5py(monkeypatch)

    result = common.write_numerical_dataset_artifacts(
        manifest=manifest,
        channels={
            "tot_force": [500.0, 1000.0],
            "force": [10.0, 20.0],
            "displacement": [0.1, 0.2],
            "extra_metric": [3.0, 4.0],
        },
    )

    assert result.hdf5_path.name == "numerical_dataset.h5"
    assert result.manifest_path.name == "numerical_dataset_manifest.json"
    assert result.hdf5_path.as_posix().startswith("_runs/gv/numerical_data")
    assert result.hdf5_path.exists()
    assert result.manifest_path.exists()

    datasets = set(recorded.file.datasets)
    assert "tot_force" in datasets
    assert "force" in datasets
    assert "displacement" in datasets

    assert "manifest_json" in recorded.file.attrs
    assert str(recorded.file.attrs["dataset_id"]) == manifest["dataset_id"]
    assert str(recorded.file.attrs["postprocessor_version"]) == manifest["postprocessor_version"]

    manifest_written = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_written["dataset_id"] == manifest["dataset_id"]


def test_write_numerical_dataset_artifacts_requires_h5py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        common,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("missing h5py")),
    )
    manifest = _base_manifest()

    with pytest.raises(RuntimeError, match="`h5py` is required"):
        common.write_numerical_dataset_artifacts(
            manifest=manifest,
            channels={"tot_force": [1.0], "force": [2.0], "displacement": [3.0]},
        )

    manifest_path = (
        Path("_runs/gv/numerical_data")
        / manifest["campaign_id"]
        / "datasets"
        / manifest["structure"]
        / manifest["experiment"]
        / manifest["geometry"]
        / manifest["control_id"]
        / "numerical_dataset_manifest.json"
    )
    assert not manifest_path.exists()
