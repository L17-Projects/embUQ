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


def test_parse_torsion_lane_channels_uses_metadata_geometry_and_anchor_pair() -> None:
    channels = torsion.parse_torsion_lane_channels(
        {
            "metadata": {
                "geometry": {"r0_cyl": 2.0, "h0_cyl": 10.0},
                "controls": {"theta": 0.4},
            },
            "channels": {
                "theta": [0.2, 0.4],
                "anchor_min_forces": [[[1.0, 0.0], [2.0, 0.0]], [[1.0, 1.0], [2.0, 0.0]]],
                "anchor_max_forces": [[[0.0, 1.0], [0.0, 2.0]], [[0.0, 0.5], [1.0, 0.5]]],
            },
        }
    )

    expected_gamma = np.array([0.04, 0.08])
    expected_sigma = (
        np.linalg.norm(np.asarray([[[1.0, 0.0], [2.0, 0.0]], [[1.0, 1.0], [2.0, 0.0]]]), axis=2).sum(axis=1)
        + np.linalg.norm(np.asarray([[[0.0, 1.0], [0.0, 2.0]], [[0.0, 0.5], [1.0, 0.5]]]), axis=2).sum(axis=1)
    ) / (2.0 * np.pi * 2.0 * 10.0)
    assert np.allclose(channels["gamma"], expected_gamma)
    assert np.allclose(channels["sigma_phi_r"], expected_sigma)


def test_parse_torsion_lane_channels_rejects_missing_force_source() -> None:
    with pytest.raises(ValueError, match="requires a constrained-vertex force source"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {"theta": [0.1, 0.2]},
            }
        )


def test_parse_torsion_lane_channels_accepts_canonical_gamma_directly() -> None:
    channels = torsion.parse_torsion_lane_channels(
        {
            "geometry": {"radius": 2.0, "height": 10.0},
            "channels": {
                "gamma": [0.0, 0.05],
                "constrained_vertex_forces": [4.0, 8.0],
            },
        }
    )
    assert np.allclose(channels["gamma"], np.array([0.0, 0.05]))
    assert np.allclose(channels["sigma_phi_r"], np.array([4.0, 8.0]) / (20.0 * np.pi * 2.0))


def test_parse_torsion_lane_channels_accepts_fully_canonical_payload() -> None:
    channels = torsion.parse_torsion_lane_channels(
        {
            "geometry": {"radius": 2.0, "height": 10.0},
            "channels": {
                "gamma": [0.0, 0.05],
                "sigma_phi_r": [0.2, 0.4],
            },
        }
    )

    assert np.allclose(channels["gamma"], [0.0, 0.05])
    assert np.allclose(channels["sigma_phi_r"], [0.2, 0.4])


def test_parse_torsion_lane_channels_rejects_anchor_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must share the same shape"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {
                    "theta": [0.1, 0.2],
                    "anchor_min_forces": [[1.0, 2.0], [1.0, 2.0]],
                    "anchor_max_forces": [[1.0, 2.0, 3.0]],
                },
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


def test_parse_torsion_lane_channels_rejects_nonnumeric_channel_payload() -> None:
    with pytest.raises(ValueError, match="numeric data"):
        torsion._coerce_numeric_array("bad", name="theta")


def test_parse_torsion_lane_controls_rejects_non_scalar_theta() -> None:
    with pytest.raises(ValueError, match="must have exactly one value"):
        torsion.parse_torsion_lane_controls({"controls": {"theta": [1.0, 2.0]}})


def test_parse_torsion_lane_controls_rejects_non_mapping_controls_override() -> None:
    with pytest.raises(ValueError, match="fixture controls must be a mapping"):
        torsion.parse_torsion_lane_controls({"channels": {"theta": [1.0]}, "controls": []})


def test_parse_torsion_lane_channels_rejects_non_mapping_channel_payload() -> None:
    with pytest.raises(ValueError, match="No numeric channel mapping found"):
        torsion.parse_torsion_lane_channels(
            {"geometry": {"radius": 2.0, "height": 10.0}, "channels": "not-a-mapping"}
        )


def test_parse_torsion_lane_geometry_rejects_invalid_geometry_in_metadata() -> None:
    with pytest.raises(ValueError, match="metadata.geometry must be a mapping"):
        torsion.parse_torsion_lane_channels(
            {
                "metadata": {"geometry": "bad"},
                "channels": {"theta": [0.1], "constrained_vertex_forces": [1.0]},
            }
        )


def test_parse_torsion_lane_channels_accepts_observables_payload_scalar_force() -> None:
    channels = torsion.parse_torsion_lane_channels(
        {
            "geometry": {"radius": 2.0, "height": 10.0},
            "controls": {"theta": 0.4},
            "observables": {
                "theta": [0.4],
                "constrained_vertex_forces": 1.0,
            },
        }
    )

    assert channels["gamma"].shape == (1,)
    assert channels["gamma"].tolist() == [0.4 * 2.0 / 10.0]
    assert channels["sigma_phi_r"].tolist() == [1.0 / (2.0 * np.pi * 2.0 * 10.0)]


def test_torsion_paper_anchor_helpers_cover_flattened_and_error_paths() -> None:
    flattened = torsion._coerce_anchor_force_timeseries(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        name="anchor_min_forces",
        particle_count=2,
    )
    assert flattened.shape == (1, 2, 3)
    assert torsion._has_force_source({"anchor_min_forces": [1.0]}) is True
    assert torsion._has_force_source({"nested": {"other": [1.0]}}) is False

    with pytest.raises(ValueError, match="numeric data"):
        torsion._coerce_anchor_force_timeseries("bad", name="anchor_min_forces", particle_count=1)
    with pytest.raises(ValueError, match="must contain data"):
        torsion._coerce_anchor_force_timeseries([], name="anchor_min_forces", particle_count=1)
    with pytest.raises(ValueError, match="at least one timestep"):
        torsion._paper_torque_statistics(np.empty((0, 1, 3)), positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        torsion._paper_torque_statistics(
            np.array([[[np.nan, 0.0, 0.0]]]),
            positions=np.ones((1, 3)),
        )


def test_compute_torsion_gamma_rejects_nonfinite_radius_or_height() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        torsion.compute_torsion_gamma([0.1], radius=float("nan"), height=1.0)


def test_parse_torsion_lane_channels_rejects_unsupported_force_ndim() -> None:
    with pytest.raises(ValueError, match="1D sample array"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {
                    "theta": [0.1, 0.2],
                    "constrained_vertex_forces": [[[[1.0]]]],
                },
            }
        )


def test_parse_torsion_lane_controls_requires_theta() -> None:
    with pytest.raises(ValueError, match="requires controls"):
        torsion.parse_torsion_lane_controls({"controls": {"bpress": -91.0}})


def test_torsion_wrapper_accepts_direct_fixture_payload_and_metadata_controls() -> None:
    lane = torsion.parse_torsion_sampling_lane(
        {
            "radius": 2.0,
            "radGV": 99.0,
            "height": [8.0],
            "metadata": {"controls": {"angle": np.asarray([0.25])}},
            "twist_angle": [0.2, 0.4],
            "anchor_force": [[3.0, 4.0], [0.0, 5.0]],
            7: "ignored",
        }
    )

    assert lane["controls"] == {"theta": 0.25}
    assert lane["geometry"] == {"radius": 2.0, "height": 8.0}
    assert np.allclose(lane["channels"]["gamma"], [0.05, 0.1])
    assert np.allclose(lane["channels"]["sigma_phi_r"], np.asarray([5.0, 5.0]) / (2.0 * np.pi * 2.0 * 8.0))


def test_torsion_parsers_accept_explicit_controls_and_geometry_overrides() -> None:
    assert torsion.parse_torsion_lane_controls({}, controls={"theta": np.asarray([0.2])}) == {"theta": 0.2}
    assert torsion.parse_torsion_lane_geometry({}, geometry={"R0": [2.0], "H0": [10.0]}) == {
        "radius": 2.0,
        "height": 10.0,
    }


def test_torsion_parsers_reject_bad_fixture_shapes_and_missing_channels() -> None:
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        torsion.parse_torsion_lane_geometry([])
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        torsion.parse_torsion_lane_channels([])
    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        torsion.parse_torsion_lane_controls([])
    with pytest.raises(ValueError, match="metadata.controls must be a mapping"):
        torsion.parse_torsion_lane_controls({"metadata": {"controls": []}})
    with pytest.raises(ValueError, match="fixture geometry must be a mapping"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": [],
                "channels": {"theta": [0.1], "constrained_vertex_forces": [1.0]},
            }
        )
    with pytest.raises(ValueError, match="requires geometry"):
        torsion.parse_torsion_lane_channels(
            {
                "channels": {"theta": [0.1], "constrained_vertex_forces": [1.0]},
            }
        )
    with pytest.raises(ValueError, match="requires a theta"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {"constrained_vertex_forces": [1.0]},
            }
        )
    with pytest.raises(ValueError, match="matching shapes"):
        torsion.parse_torsion_lane_channels(
            {
                "geometry": {"radius": 2.0, "height": 10.0},
                "channels": {"gamma": [0.1, 0.2], "constrained_vertex_forces": [1.0]},
            }
        )


def test_torsion_geometry_helpers_reject_nonpositive_stress_and_strain_geometry() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        torsion.compute_torsion_gamma([0.1], radius=2.0, height=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        torsion.compute_torsion_sigma_phi_r([1.0], radius=float("inf"), height=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        torsion.compute_torsion_sigma_phi_r([1.0], radius=2.0, height=-1.0)
    with pytest.raises(ValueError, match="finite radius and anchor dz"):
        torsion.compute_torsion_paper_gamma([0.1], radius=float("inf"), dz=1.0)
    with pytest.raises(ValueError, match="positive radius and anchor dz"):
        torsion.compute_torsion_paper_gamma([0.1], radius=2.0, dz=0.0)


def test_torsion_paper_anchor_helpers_cover_flattened_forces_and_shape_errors() -> None:
    mesh_vertices = np.array(
        [
            [0.0, 0.0, -6.0],
            [1.0, 0.0, -5.5],
            [0.0, 0.0, 5.5],
            [1.0, 0.0, 6.0],
        ],
        dtype=float,
    )
    flat_bottom_forces = np.tile(np.array([[0.0, 1.0, 0.0, 0.0, 2.0, 0.0]], dtype=float), (4, 1))
    flat_top_forces = np.tile(np.array([[0.0, 1.5, 0.0, 0.0, 3.0, 0.0]], dtype=float), (4, 1))

    channels = torsion.reconstruct_torsion_paper_channels(
        {
            "geometry": {"radius": 2.0, "height": 14.28},
            "controls": {"theta": 0.1},
            "channels": {
                "mesh_vertices": mesh_vertices,
                "anchor_min_forces": flat_bottom_forces,
                "anchor_max_forces": flat_top_forces,
            },
        }
    )

    assert channels["gamma"].shape == (1,)
    assert channels["sigma_phi_r"].shape == (1,)
    assert channels["sigma_std"].shape == (1,)

    with pytest.raises(ValueError, match="fixture_like must be a mapping"):
        torsion.reconstruct_torsion_anchor_geometry([])
    with pytest.raises(ValueError, match="shape \\(n, 3\\)"):
        torsion.reconstruct_torsion_anchor_geometry(
            {
                "geometry": {"radius": 2.0, "height": 14.28},
                "channels": {"mesh_vertices": [[0.0, 1.0]]},
            }
        )
    with pytest.raises(ValueError, match="flattened shape"):
        torsion.reconstruct_torsion_paper_channels(
            {
                "geometry": {"radius": 2.0, "height": 14.28},
                "controls": {"theta": 0.1},
                "channels": {
                    "mesh_vertices": mesh_vertices,
                    "anchor_min_forces": np.ones((4, 5), dtype=float),
                    "anchor_max_forces": flat_top_forces,
                },
            }
        )


def test_parse_torsion_lane_controls_ignores_non_string_control_keys() -> None:
    assert torsion.parse_torsion_lane_controls({"controls": {7: 1.0, "theta": 0.3}}) == {"theta": 0.3}
