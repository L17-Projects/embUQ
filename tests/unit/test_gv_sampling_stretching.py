from __future__ import annotations

import numpy as np
import pytest

from meso_uq.structures.gv.sampling import stretching


def test_extract_stretching_lane_returns_controls_and_candidate_channels() -> None:
    fixture = {
        "controls": {"tot_force": 600.0, "bpress": -91.0},
        "channels": {
            "tot_force": [500.0, 600.0],
            "force": [3.0, 4.0],
            "displacement": [0.1, 0.2],
            "radius": [2.0, 2.04, 2.09],
            "mesh": {
                "sim0": [1.0, 2.0, 3.0],
            },
        },
    }

    lane = stretching.extract_stretching_lane(fixture)

    assert lane["axis"] == "tot_force"
    assert lane["controls"] == {"tot_force": 600.0, "bpress": -91.0}
    channels = lane["channels"]
    assert np.array_equal(channels["tot_force"], np.array([500.0, 600.0]))
    assert np.array_equal(channels["force"], np.array([3.0, 4.0]))
    assert np.array_equal(channels["displacement"], np.array([0.1, 0.2]))
    assert np.array_equal(channels["radius"], np.array([2.0, 2.04, 2.09]))
    assert np.allclose(channels["radius_change"], np.array([0.0, 0.04, 0.09]))
    assert np.array_equal(channels["mesh_sim0"], np.array([1.0, 2.0, 3.0]))


def test_extract_stretching_lane_rejects_missing_controls_and_non_mapping_controls() -> None:
    with pytest.raises(ValueError, match="requires controls"):
        stretching.extract_stretching_lane(
            {"channels": {"tot_force": [1.0], "force": [2.0], "displacement": [0.1]}},
        )

    with pytest.raises(ValueError, match="controls must be a mapping"):
        stretching.extract_stretching_lane(
            {"controls": [("tot_force", 500.0)], "channels": {"tot_force": [1.0], "force": [2.0], "displacement": [0.1]}},
        )


def test_parse_stretching_lane_channels_rejects_missing_required_channel_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="requires at least channels"):
        stretching.parse_stretching_lane_channels(
            {"channels": {"tot_force": [1.0], "force": [2.0]}},
        )

    with pytest.raises(ValueError, match="non-finite"):
        stretching.parse_stretching_lane_channels(
            {"channels": {"tot_force": [1.0], "force": [np.nan], "displacement": [0.1]}},
        )

    with pytest.raises(ValueError, match="Channel 'displacement' must contain data"):
        stretching.parse_stretching_lane_channels(
            {"channels": {"tot_force": [1.0], "force": [2.0], "displacement": []}},
        )


def test_parse_stretching_lane_channels_uses_control_like_aliases() -> None:
    controls = stretching.parse_stretching_lane_controls(
        {
            "controls": {"total_force": 500.0, "pressure": -91.0},
            "channels": {"tot_force": [1.0, 2.0], "force": [1.0, 2.0], "displacement": [0.1, 0.2]},
        },
    )

    assert controls == {"tot_force": 500.0, "bpress": -91.0}
