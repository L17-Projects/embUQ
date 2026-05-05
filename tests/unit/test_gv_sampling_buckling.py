from __future__ import annotations

import numpy as np
import pytest

from meso_uq.structures.gv.sampling import buckling


def test_extract_buckling_lane_returns_controls_and_key_channels() -> None:
    fixture = {
        "controls": {"buck": 0.5, "bpress": -91.0},
        "channels": {
            "buck": [0.0, 0.5],
            "bpress": [-91.0, -92.0],
            "response": [1.0, 0.8],
            "force_response": [4.0, 3.7],
            "relative_volume": [1.0, 0.93],
            "shape": [0.1, 0.2],
            "deformation": [0.2, 0.25],
            "object": {"stats": {"mean": [1.0, 1.2]}},
        },
    }

    lane = buckling.extract_buckling_lane(fixture)

    assert lane["axis"] == "buck"
    assert lane["controls"] == {"buck": 0.5, "bpress": -91.0}

    channels = lane["channels"]
    assert np.array_equal(channels["buck"], np.array([0.0, 0.5]))
    assert np.array_equal(channels["buckling_response"], np.array([1.0, 0.8]))
    assert np.array_equal(channels["force_response"], np.array([4.0, 3.7]))
    assert np.array_equal(channels["relative_volume"], np.array([1.0, 0.93]))
    assert np.array_equal(channels["shape_amplitude"], np.array([0.1, 0.2]))
    assert np.array_equal(channels["deformation_amplitude"], np.array([0.2, 0.25]))
    assert np.array_equal(channels["object_stats_mean"], np.array([1.0, 1.2]))


def test_extract_buckling_lane_rejects_missing_controls() -> None:
    with pytest.raises(ValueError, match="requires controls"):
        buckling.extract_buckling_lane(
            {"channels": {"buck": [0.0], "response": [1.0], "force": [2.0]}},
        )


def test_extract_buckling_lane_controls_aliases_and_implicit_pressure() -> None:
    controls = buckling.parse_buckling_lane_controls(
        {
            "controls": {"buckling": 0.3, "pressure": -91.0},
            "channels": {
                "buck": [0.0],
                "response": [1.0],
            },
        },
    )

    assert controls == {"buck": 0.3, "bpress": -91.0}


def test_parse_buckling_lane_controls_rejects_non_scalar_control() -> None:
    with pytest.raises(ValueError, match="must be a scalar value"):
        buckling.parse_buckling_lane_controls({"channels": {"buck": [0.0], "response": [1.0], "buck": [0.0]}, "controls": {"buck": [0.1, 0.2]}})


def test_parse_buckling_lane_channels_accepts_fallback_fixture_payload() -> None:
    channels = buckling.parse_buckling_lane_channels(
        {
            "buck": [0.0, 0.5],
            "buckling_response": [1.0, 0.8],
        }
    )
    assert np.array_equal(channels["buckling_response"], np.array([1.0, 0.8]))
    assert np.array_equal(channels["buck"], np.array([0.0, 0.5]))


def test_parse_buckling_lane_channels_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="No numeric channel mapping found"):
        buckling.parse_buckling_lane_channels({"channels": []})


def test_parse_buckling_lane_controls_rejects_non_mapping_input() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        buckling.parse_buckling_lane_controls(123)


def test_parse_buckling_lane_channels_rejects_missing_response_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="requires at least one response channel"):
        buckling.parse_buckling_lane_channels(
            {"channels": {"buck": [0.0], "relative_volume": [1.0, 0.93]}},
        )

    with pytest.raises(ValueError, match="non-finite"):
        buckling.parse_buckling_lane_channels(
            {"channels": {"buck": [0.0], "buckling_response": [1.0, np.nan]}},
        )

    with pytest.raises(ValueError, match="Channel 'buckling_response' must contain data"):
        buckling.parse_buckling_lane_channels(
            {"channels": {"buck": [0.0], "buckling_response": []}},
        )


def test_parse_buckling_lane_channels_accepts_observables_payload() -> None:
    channels = buckling.parse_buckling_lane_channels(
        {
            "observables": {
                "buck": [0.0, 0.5],
                "response": [1.0, 0.8],
                "force_response": [4.0, 3.7],
            }
        }
    )

    assert np.array_equal(channels["buck"], np.array([0.0, 0.5]))
    assert np.array_equal(channels["buckling_response"], np.array([1.0, 0.8]))


def test_parse_buckling_lane_controls_requires_valid_mapping() -> None:
    with pytest.raises(ValueError, match="Buckling fixture controls must be a mapping"):
        buckling.parse_buckling_lane_controls({"controls": [("buck", 0.5)]})


def test_parse_buckling_lane_controls_uses_explicit_controls_argument() -> None:
    controls = buckling.parse_buckling_lane_controls(
        {"channels": {"buck": [0.1], "response": [1.0]}},
        controls={"buckling": 0.5, "pressure": -91.0},
    )
    assert controls == {"buck": 0.5, "bpress": -91.0}


def test_parse_buckling_sampling_lane_wrapper() -> None:
    result = buckling.parse_buckling_sampling_lane(
        {
            "controls": {"buckling": 0.3, "pressure": -91.0},
            "channels": {"buck": [0.0], "buckling_response": [1.0]},
        },
    )
    assert result["axis"] == "buck"
    assert result["controls"] == {"buck": 0.3, "bpress": -91.0}
