from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.postprocessing import common
from meso_uq.structures.gv.postprocessing import eigenmodes


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


def _fake_h5py(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
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
            self.datasets[name] = np.asarray(data)

    class FakeH5:
        File = FakeH5File

    monkeypatch.setattr(common, "_import_h5py", lambda: FakeH5)
    return recorded


def test_parse_eigenmodes_fixture_channels_supports_aliases_and_nested() -> None:
    fixture = {
        "channels": {
            "eigvalue": [1.0, 2.0, 3.0],
            "eigvector": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        },
    }

    channels = eigenmodes.parse_eigenmodes_fixture_channels(fixture)

    assert channels["eigenvalues"].shape == (3,)
    assert channels["eigenvectors"].shape == (3, 2)


def test_parse_eigenmodes_fixture_channels_with_eigenmode_spectrum_tuple() -> None:
    fixture = {
        "eigenmode_spectrum": (
            [1.0, 2.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    channels = eigenmodes.parse_eigenmodes_fixture_channels(fixture)

    assert "eigenvalues" in channels
    assert "eigenvectors" in channels


def test_parse_eigenmodes_fixture_channels_accepts_single_value_spectrum_tuple() -> None:
    channels = eigenmodes.parse_eigenmodes_fixture_channels({"eigenmode_spectrum": ([1.0, 2.0],)})

    assert set(channels) == {"eigenvalues"}


def test_normalize_eigenmodes_fixture_mapping_supports_nested_mapping_and_skips_non_string_keys() -> None:
    normalized = eigenmodes._normalize_fixture_mapping(
        {
            "observables": {
                "eigenmode_spectrum": {
                    "eigvalue": [1.0, 2.0],
                    "eigvector": [[1.0], [0.0]],
                    3: "ignored",
                },
                4: "ignored",
            }
        }
    )

    assert normalized["eigvalue"] == [1.0, 2.0]
    assert normalized["eigvector"] == [[1.0], [0.0]]
    assert 4 not in normalized


def test_normalize_eigenmodes_fixture_mapping_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="No mapping payload found"):
        eigenmodes._normalize_fixture_mapping({"channels": ["bad-payload"]})


def test_parse_eigenmodes_fixture_channels_rejects_missing_required_channels() -> None:
    fixture = {"observables": {"eigenvectors": [[1.0, 0.0], [0.0, 1.0]]}}

    with pytest.raises(ValueError, match="eigenvalues"):
        eigenmodes.parse_eigenmodes_fixture_channels(fixture)


def test_parse_eigenmodes_fixture_channels_can_drop_optional_when_disabled() -> None:
    fixture = {
        "channels": {
            "eigenvalues": [1.0, 2.0, 3.0],
            "eigenvectors": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        },
    }

    channels = eigenmodes.parse_eigenmodes_fixture_channels(
        fixture,
        include_optional_channels=False,
    )

    assert "eigenvalues" in channels
    assert "eigenvectors" not in channels


@pytest.mark.parametrize(
    ("eigenvectors", "match"),
    [
        (np.zeros((2, 2, 1)), "1D or 2D"),
        ([1.0, 0.0], "match eigenvalue count"),
        ([[1.0, 0.0], [0.0, 1.0]], "first axis must match"),
    ],
)
def test_parse_eigenmodes_fixture_channels_rejects_invalid_eigenvector_shapes(
    eigenvectors: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        eigenmodes.parse_eigenmodes_fixture_channels(
            {
                "channels": {
                    "eigenvalues": [1.0, 2.0, 3.0],
                    "eigenvectors": eigenvectors,
                }
            }
        )


def test_resolve_quality_flags_without_requested_flags_tracks_non_finite_values() -> None:
    flags = eigenmodes._resolve_quality_flags(
        {
            "eigenvalues": np.array([1.0, np.nan]),
            "eigenvectors": np.array([[1.0], [0.0]]),
        }
    )

    assert flags["finite_observables"] is False
    assert flags["finite_observable_ratio"] == pytest.approx(0.75)
    assert flags["canary_failures"] == ("eigenvalues:1 non-finite",)


def test_parse_eigenmodes_fixture_channels_rejects_non_mapping_fixture() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        eigenmodes.parse_eigenmodes_fixture_channels(["bad-payload"])


def test_process_eigenmodes_numerical_dataset_writes_hdf5_and_merges_analysis_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recorded = _fake_h5py(monkeypatch)

    fixture = {
        "channels": {
            "eigenvalues": [1.0, 2.0, 3.0],
            "eigenvectors": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        },
        "analysis_provenance": {
            "method": "covariance",
            "mode": "trim_svd",
        },
    }

    result = eigenmodes.process_eigenmodes_numerical_dataset(
        campaign_id="campaign-eigenmodes",
        geometry_radius=2.0,
        geometry_height=14.28,
        material_parameters=_BASE_MATERIAL_PARAMETERS,
        controls={"bpress": -91.0},
        raw_provenance={"source": "fixture-like"},
        fixture_like=fixture,
        quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
    )

    assert result.manifest["experiment"] == "eigenmodes"
    assert result.manifest["raw_provenance"]["analysis"] == fixture["analysis_provenance"]
    assert result.manifest["quality_flags"]["finite_observables"] is True
    assert result.hdf5_path.exists()
    assert result.manifest_path.exists()
    assert result.hdf5_path.as_posix().startswith("_runs/gv/numerical_data")
    assert "eigenvalues" in recorded.file.datasets
    assert "eigenvectors" in recorded.file.datasets


def test_process_eigenmodes_numerical_dataset_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        eigenmodes.process_eigenmodes_numerical_dataset(
            campaign_id="campaign-eigenmodes-nonfinite",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"bpress": -91.0},
            raw_provenance={"source": "fixture-like"},
            fixture_like={
                "channels": {
                    "eigenvalues": [1.0, float("nan"), 3.0],
                    "eigenvectors": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                }
            },
        )


def test_process_eigenmodes_numerical_dataset_requires_h5py(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        common,
        "_import_h5py",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("missing h5py")),
    )

    fixture = {"eigenvalues": [1.0, 2.0, 3.0]}

    with pytest.raises(RuntimeError, match="`h5py` is required"):
        eigenmodes.process_eigenmodes_numerical_dataset(
            campaign_id="campaign-eigenmodes-no-h5py",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"bpress": -91.0},
            raw_provenance={"source": "fixture-like"},
            fixture_like=fixture,
            quality_flags={"finite_observables": True, "finite_observable_ratio": 1.0, "canary_failures": ()},
        )


def test_process_eigenmodes_numerical_dataset_rejects_non_mapping_raw_provenance() -> None:
    with pytest.raises(ValueError, match="raw_provenance must be a mapping"):
        eigenmodes.process_eigenmodes_numerical_dataset(
            campaign_id="campaign-eigenmodes-bad-provenance",
            geometry_radius=2.0,
            geometry_height=14.28,
            material_parameters=_BASE_MATERIAL_PARAMETERS,
            controls={"bpress": -91.0},
            raw_provenance=["bad-provenance"],
            fixture_like={"channels": {"eigenvalues": [1.0, 2.0]}},
        )
