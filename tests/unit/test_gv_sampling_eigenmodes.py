from __future__ import annotations

import numpy as np
import pytest

from meso_uq.structures.gv.sampling import eigenmodes


def test_extract_eigenmodes_lane_uses_default_mode_count_of_30() -> None:
    fixture = {
        "controls": {"pressure": -91.0},
        "channels": {
            "eigenfrequencies": np.arange(40, dtype=float) + 0.5,
            "eigenvectors": np.arange(120, dtype=float).reshape(40, 3),
        },
    }

    lane = eigenmodes.extract_eigenmodes_lane(fixture)

    assert lane["axis"] == "mode_index"
    assert lane["controls"] == {"bpress": -91.0}
    assert lane["channels"]["mode_index"].shape == (30,)
    assert lane["channels"]["eigenfrequencies"].shape == (30,)
    assert lane["channels"]["eigenvectors"].shape == (30, 3)
    assert np.array_equal(lane["channels"]["mode_index"], np.arange(30))


def test_parse_eigenmodes_lane_channels_selects_requested_mode_count_and_aliases() -> None:
    channels = eigenmodes.parse_eigenmodes_lane_channels(
        {
            "eigenmode_spectrum": {
                "eigvalue": [1.0, 2.0, 3.0, 4.0],
                "eigfreq": [10.0, 20.0, 30.0, 40.0],
                "eigvector": [[1.0], [2.0], [3.0], [4.0]],
            }
        },
        mode_count=2,
    )

    assert np.array_equal(channels["mode_index"], np.array([0, 1]))
    assert np.array_equal(channels["eigenvalues"], np.array([1.0, 2.0]))
    assert np.array_equal(channels["eigenfrequencies"], np.array([10.0, 20.0]))
    assert np.array_equal(channels["eigenvectors"], np.array([[1.0], [2.0]]))


def test_parse_eigenmodes_lane_channels_rejects_missing_spectra_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="requires eigenvalues or eigenfrequencies"):
        eigenmodes.parse_eigenmodes_lane_channels({"channels": {"eigenvectors": [[1.0], [2.0]]}})

    with pytest.raises(ValueError, match="non-finite"):
        eigenmodes.parse_eigenmodes_lane_channels({"channels": {"eigenvalues": [1.0, np.nan]}})


def test_parse_eigenmodes_lane_channels_rejects_mismatched_lengths_and_short_eigenvectors() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        eigenmodes.parse_eigenmodes_lane_channels(
            {"channels": {"eigenvalues": [1.0, 2.0], "eigenfrequencies": [10.0]}},
        )

    with pytest.raises(ValueError, match="at least as many modes as selected"):
        eigenmodes.parse_eigenmodes_lane_channels(
            {
                "channels": {
                    "eigenvalues": [1.0, 2.0, 3.0],
                    "eigenvectors": [[1.0], [2.0]],
                }
            },
            mode_count=3,
        )


def test_parse_eigenmodes_lane_channels_can_drop_optional_eigenvectors() -> None:
    channels = eigenmodes.parse_eigenmodes_lane_channels(
        {
            "channels": {
                "eigenvalues": [1.0, 2.0, 3.0],
                "eigenvectors": [[1.0], [2.0], [3.0]],
            }
        },
        include_eigenvectors=False,
    )

    assert "eigenvectors" not in channels


def test_parse_eigenmodes_lane_channels_from_observables_with_aliases() -> None:
    channels = eigenmodes.parse_eigenmodes_lane_channels(
        {
            "observables": {
                "eigvalues": [1.0, 2.0],
                "eigfreq": [10.0, 20.0],
                "eigvector": [[1.0], [2.0]],
            }
        },
        mode_count=2,
    )
    assert np.array_equal(channels["eigenvalues"], np.array([1.0, 2.0]))
    assert np.array_equal(channels["eigenfrequencies"], np.array([10.0, 20.0]))
    assert np.array_equal(channels["eigenvectors"], np.array([[1.0], [2.0]]))


def test_parse_eigenmodes_lane_controls_uses_pressure_alias() -> None:
    controls = eigenmodes.parse_eigenmodes_lane_controls(
        {"controls": {"pressure": -91.0, "notes": "ignore-me"}},
        controls=None,
    )
    assert controls == {"bpress": -91.0}


def test_parse_eigenmodes_lane_channels_rejects_invalid_eigenvector_ndim() -> None:
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        eigenmodes.parse_eigenmodes_lane_channels(
            {
                "channels": {
                    "eigenvalues": [1.0, 2.0],
                    "eigenvectors": [[[1.0], [2.0]], [[3.0], [4.0]]],
                }
            }
        )


def test_parse_eigenmodes_sampling_lane_wrapper() -> None:
    payload = eigenmodes.parse_eigenmodes_sampling_lane(
        {"channels": {"eigenvalues": [1.0], "eigenfrequencies": [2.0]}}
    )
    assert payload["axis"] == "mode_index"


def test_parse_eigenmodes_lane_channels_supports_eigenmode_spectrum_tuple() -> None:
    channels = eigenmodes.parse_eigenmodes_lane_channels(
        {"eigenmode_spectrum": ([1.0, 4.0], [2.0, 3.0], [10.0, 20.0])},
        mode_count=2,
    )
    assert np.array_equal(channels["eigenvalues"], np.array([1.0, 4.0]))
    assert np.array_equal(channels["eigenvectors"], np.array([2.0, 3.0]))
    assert np.array_equal(channels["eigenfrequencies"], np.array([10.0, 20.0]))


def test_parse_eigenmodes_lane_channels_requires_mode_count_positive() -> None:
    with pytest.raises(ValueError, match="mode_count must be a positive integer"):
        eigenmodes.parse_eigenmodes_lane_channels({"channels": {"eigenvalues": [1.0, 2.0]}}, mode_count=0)


def test_parse_eigenmodes_lane_channels_rejects_non_1d_eigenvalues() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        eigenmodes.parse_eigenmodes_lane_channels(
            {"channels": {"eigenvalues": [[1.0, 2.0], [3.0, 4.0]], "eigenfrequencies": [1.0, 2.0]}}
        )


def test_parse_eigenmodes_lane_channels_rejects_non_numeric_payload() -> None:
    with pytest.raises(ValueError, match="numeric data"):
        eigenmodes._coerce_numeric_array(["bad"], name="eigenvalues")


def test_parse_eigenmodes_lane_channels_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="must contain data"):
        eigenmodes._coerce_numeric_array([], name="eigenvalues")


def test_parse_eigenmodes_lane_controls_rejects_non_mapping_controls_input() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        eigenmodes.parse_eigenmodes_lane_controls({"controls": "bad"})


def test_parse_eigenmodes_lane_controls_rejects_non_scalar_control_value() -> None:
    with pytest.raises(ValueError, match="must be a scalar value"):
        eigenmodes.parse_eigenmodes_lane_controls({"controls": {"pressure": [1.0, 2.0]}})
