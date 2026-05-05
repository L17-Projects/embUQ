from __future__ import annotations

import numpy as np
import pytest

from meso_uq.structures.gv.sampling import torsion


def test_extract_torsion_lane_builds_canonical_gamma_and_sigma_phi_r() -> None:
    fixture = {
        "controls": {"theta": 0.2},
        "geometry": {"radius": 2.0, "H0_cyl": 10.0},
        "channels": {
            "theta": [0.0, 0.1, 0.2],
            "anchor_min_forces": [[[1.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            "anchor_max_forces": [[[0.0, 1.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]],
        },
    }

    lane = torsion.extract_torsion_lane(fixture)

    assert lane["axis"] == "gamma"
    assert lane["controls"] == {"theta": 0.2}
    assert lane["geometry"] == {"radius": 2.0, "height": 10.0}
    assert np.allclose(lane["channels"]["gamma"], np.array([0.0, 0.02, 0.04]))

    reduced_force = np.array([
        6.0,
        6.0,
        4.0 * np.sqrt(2.0),
    ])
    expected_sigma = reduced_force / (2.0 * np.pi * 2.0 * 10.0)
    assert np.allclose(lane["channels"]["sigma_phi_r"], expected_sigma)


def test_parse_torsion_lane_channels_accepts_direct_force_source_array() -> None:
    channels = torsion.parse_torsion_lane_channels(
        {
            "geometry": {"r0": 2.0, "height": 8.0},
            "channels": {
                "theta": [0.2, 0.4],
                "constrained_vertex_forces": [10.0, 12.0],
            },
        }
    )

    assert np.allclose(channels["gamma"], np.array([0.05, 0.1]))
    assert np.allclose(channels["sigma_phi_r"], np.array([10.0, 12.0]) / (32.0 * np.pi))


def test_parse_torsion_lane_channels_rejects_missing_force_source() -> None:
    with pytest.raises(ValueError, match="requires a constrained-vertex force source"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {"theta": [0.1, 0.2]},
            }
        )


def test_parse_torsion_lane_channels_rejects_bad_geometry_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 0.0, "height": 10.0},
                "channels": {"theta": [0.1], "constrained_vertex_forces": [1.0]},
            }
        )

    with pytest.raises(ValueError, match="non-finite"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {"theta": [0.1, np.nan], "constrained_vertex_forces": [1.0, 2.0]},
            }
        )

    with pytest.raises(ValueError, match="must contain data"):
        torsion.compute_torsion_sigma_phi_r([], radius=2.0, height=10.0)


def test_parse_torsion_lane_controls_requires_theta() -> None:
    with pytest.raises(ValueError, match="requires controls"):
        torsion.parse_torsion_lane_controls({"controls": {"bpress": -91.0}})
