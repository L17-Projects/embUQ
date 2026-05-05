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


def test_parse_stretching_lane_controls_rejects_non_scalar_control() -> None:
    with pytest.raises(ValueError, match="must be a scalar value"):
        stretching.parse_stretching_lane_controls(
            {"channels": {"force": [1.0], "displacement": [0.1], "tot_force": [1.0, 2.0]}},
            controls={"tot_force": [1.0, 2.0], "bpress": -91.0},
        )


def test_parse_stretching_lane_controls_rejects_non_numeric_control() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        stretching.parse_stretching_lane_controls(
            {"controls": {"tot_force": "bad", "bpress": -91.0}, "channels": {"force": [1.0], "displacement": [0.1], "tot_force": [1.0]}}
        )


def test_parse_stretching_lane_channels_accepts_fallback_fixture_payload() -> None:
    channels = stretching.parse_stretching_lane_channels(
        {
            "tot_force": [1.0, 2.0],
            "force": [2.0, 3.0],
            "displacement": [0.1, 0.2],
        }
    )

    assert np.array_equal(channels["force"], np.array([2.0, 3.0]))
    assert np.array_equal(channels["displacement"], np.array([0.1, 0.2]))


def test_parse_stretching_lane_channels_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="No numeric channel mapping"):
        stretching.parse_stretching_lane_channels({"channels": "bad", "controls": {"tot_force": 500.0, "bpress": -91.0}})


def test_parse_stretching_lane_controls_rejects_non_mapping_input() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        stretching.parse_stretching_lane_controls(123)


def test_extract_stretching_sampling_lane_uses_metadata_controls() -> None:
    lane = stretching.parse_stretching_sampling_lane(
        {
            "metadata": {"controls": {"total_force": 600.0, "pressure": -91.0}},
            "observables": {"force": [1.0, 2.0], "displacement": [0.1, 0.2], "tot_force": [1.0, 2.0]},
        },
    )

    assert lane["axis"] == "tot_force"
    assert lane["controls"] == {"tot_force": 600.0, "bpress": -91.0}
    assert set(lane["channels"]).issuperset({"force", "displacement"})


def test_extract_stretching_lane_rejects_non_mapping_metadata_controls() -> None:
    with pytest.raises(ValueError, match="Stretching fixture metadata.controls must be a mapping"):
        stretching.parse_stretching_lane_controls({"metadata": {"controls": [("tot_force", 500.0)]}})


def test_extract_stretching_lane_uses_explicit_controls_override() -> None:
    lane = stretching.extract_stretching_lane(
        {
            "controls": {"tot_force": 1.0, "bpress": -90.0},
            "channels": {"tot_force": [1.0], "force": [2.0], "displacement": [0.1]},
        },
        controls={"tot_force": 2.0, "bpress": -91.0},
    )

    assert lane["controls"] == {"tot_force": 2.0, "bpress": -91.0}
