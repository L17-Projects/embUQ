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
