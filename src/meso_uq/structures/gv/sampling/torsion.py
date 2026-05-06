from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

_TORSION_AXIS = "gamma"
_REQUIRED_CONTROLS = ("theta",)
_FORCE_TO_STRESS_AREA_SCALE = 2.0 * np.pi
_PAPER_ANCHOR_FRACTION = 0.35
_PAPER_STEADY_STATE_FRACTION = 0.25
_PAPER_GAMMA_RADIUS_FACTOR = 2.0

_CHANNEL_ALIASES = {
    "theta": "theta",
    "twist_angle": "theta",
    "torsion_coord": "theta",
    "rotation": "theta",
    "angle": "theta",
    "gamma": "gamma",
    "shear_strain": "gamma",
    "sigma_phi_r": "sigma_phi_r",
    "sigma_std": "sigma_std",
    "mesh_vertices": "mesh_vertices",
    "vertices": "mesh_vertices",
    "vertex_positions": "mesh_vertices",
    "positions": "mesh_vertices",
    "constrained_vertex_force": "constrained_vertex_forces",
    "constrained_vertex_forces": "constrained_vertex_forces",
    "constrained_forces": "constrained_vertex_forces",
    "constrained_force": "constrained_vertex_forces",
    "anchor_force": "constrained_vertex_forces",
    "anchor_forces": "constrained_vertex_forces",
    "anchor_reaction_force": "constrained_vertex_forces",
    "anchor_reaction_forces": "constrained_vertex_forces",
    "anchor_min_force": "constrained_vertex_forces_min",
    "anchor_min_forces": "constrained_vertex_forces_min",
    "bottom_anchor_force": "constrained_vertex_forces_min",
    "bottom_anchor_forces": "constrained_vertex_forces_min",
    "anchor_max_force": "constrained_vertex_forces_max",
    "anchor_max_forces": "constrained_vertex_forces_max",
    "top_anchor_force": "constrained_vertex_forces_max",
    "top_anchor_forces": "constrained_vertex_forces_max",
}
_GEOMETRY_ALIASES = {
    "radius": "radius",
    "radgv": "radius",
    "r0": "radius",
    "r0_cyl": "radius",
    "height": "height",
    "h0": "height",
    "h0_cyl": "height",
    "height_cyl": "height",
}


def _canonical_name(raw_name: object) -> str:
    normalized = re.sub(r"[^\w]+", "_", str(raw_name).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _coerce_numeric_array(payload: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Torsion channel {name!r} must be numeric data.") from exc
    if array.size == 0:
        raise ValueError(f"Torsion channel {name!r} must contain data.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Torsion channel {name!r} contains non-finite values.")
    return array


def _normalize_scalar(value: object, *, name: str) -> float:
    array = _coerce_numeric_array(value, name=name)
    if array.shape == ():
        scalar = float(array.item())
    elif array.size == 1:
        scalar = float(array.reshape(-1)[0])
    else:
        raise ValueError(f"Torsion scalar {name!r} must have exactly one value.")
    return scalar


def _extract_channel_payload(fixture_like: Mapping[str, Any]) -> Mapping[str, Any]:
    if "channels" in fixture_like:
        channel_like = fixture_like["channels"]
    elif "observables" in fixture_like:
        channel_like = fixture_like["observables"]
    else:
        channel_like = fixture_like
    if not isinstance(channel_like, Mapping):
        raise ValueError("No numeric channel mapping found in torsion fixture.")
    return channel_like


def _extract_control_payload(
    fixture_like: Mapping[str, Any],
    controls: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if controls is not None:
        return controls
    if "controls" in fixture_like:
        payload = fixture_like["controls"]
        if not isinstance(payload, Mapping):
            raise ValueError("Torsion fixture controls must be a mapping.")
        return payload
    metadata = fixture_like.get("metadata")
    if isinstance(metadata, Mapping) and "controls" in metadata:
        payload = metadata["controls"]
        if not isinstance(payload, Mapping):
            raise ValueError("Torsion fixture metadata.controls must be a mapping.")
        return payload
    return {}


def _extract_geometry_payload(
    fixture_like: Mapping[str, Any],
    geometry: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if geometry is not None:
        return geometry
    if "geometry" in fixture_like:
        payload = fixture_like["geometry"]
        if not isinstance(payload, Mapping):
            raise ValueError("Torsion fixture geometry must be a mapping.")
        return payload
    metadata = fixture_like.get("metadata")
    if isinstance(metadata, Mapping) and "geometry" in metadata:
        payload = metadata["geometry"]
        if not isinstance(payload, Mapping):
            raise ValueError("Torsion fixture metadata.geometry must be a mapping.")
        return payload
    return fixture_like


def parse_torsion_lane_geometry(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract radius/height geometry scalars used by torsion canonicalization."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    payload = _extract_geometry_payload(fixture_like, geometry)
    normalized: dict[str, float] = {}
    for raw_name, value in payload.items():
        if not isinstance(raw_name, str):
            continue
        name = _GEOMETRY_ALIASES.get(_canonical_name(raw_name))
        if name is None or name in normalized:
            continue
        normalized[name] = _normalize_scalar(value, name=name)

    if "radius" not in normalized or "height" not in normalized:
        raise ValueError("Torsion lane requires geometry with radius and height/H0_cyl.")
    if normalized["radius"] <= 0.0 or normalized["height"] <= 0.0:
        raise ValueError("Torsion geometry radius and height/H0_cyl must be positive.")
    return normalized


def compute_torsion_gamma(
    theta: object,
    *,
    radius: float,
    height: float,
) -> np.ndarray:
    """Compute canonical torsion strain gamma = R0 * theta / H0_cyl."""

    if not np.isfinite(radius) or not np.isfinite(height):
        raise ValueError("Torsion geometry radius and height/H0_cyl must be finite.")
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("Torsion geometry radius and height/H0_cyl must be positive.")
    theta_array = _coerce_numeric_array(theta, name="theta")
    return (float(radius) * theta_array) / float(height)


def compute_torsion_paper_gamma(
    theta: object,
    *,
    radius: float,
    dz: float,
) -> np.ndarray:
    """Compute paper-replay torsion strain gamma = 2 * theta * radGV / dz."""

    if not np.isfinite(radius) or not np.isfinite(dz):
        raise ValueError("Torsion paper replay requires finite radius and anchor dz.")
    if radius <= 0.0 or dz <= 0.0:
        raise ValueError("Torsion paper replay requires positive radius and anchor dz.")
    theta_array = _coerce_numeric_array(theta, name="theta")
    return (_PAPER_GAMMA_RADIUS_FACTOR * float(radius) * theta_array) / float(dz)


def _reduce_force_payload(payload: object, *, name: str) -> np.ndarray:
    array = _coerce_numeric_array(payload, name=name)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return np.abs(array)
    if array.ndim == 2:
        return np.linalg.norm(array, axis=1)
    if array.ndim == 3:
        return np.linalg.norm(array, axis=2).sum(axis=1)
    raise ValueError(
        f"Torsion force source {name!r} must be a 1D sample array, 2D vector array, or 3D per-vertex vector array."
    )


def compute_torsion_sigma_phi_r(
    constrained_vertex_forces: object,
    *,
    radius: float,
    height: float,
) -> np.ndarray:
    """Convert constrained-vertex force magnitudes to sigma_phi_r.

    Fixture convention: the force source must represent the constrained-vertex
    tangential reaction force per sample, either as total magnitudes (`shape=(n,)`),
    per-sample vectors (`shape=(n,2|3)`), or per-sample per-vertex vectors
    (`shape=(n,m,2|3)`). The canonical stress uses the cylindrical side area
    `2*pi*R0*H0_cyl`.
    """

    if not np.isfinite(radius) or not np.isfinite(height):
        raise ValueError("Torsion geometry radius and height/H0_cyl must be finite.")
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("Torsion geometry radius and height/H0_cyl must be positive.")
    reduced_force = _reduce_force_payload(
        constrained_vertex_forces,
        name="constrained_vertex_forces",
    )
    return reduced_force / (_FORCE_TO_STRESS_AREA_SCALE * float(radius) * float(height))


def _extract_canonical_channels(channel_payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    canonical: dict[str, np.ndarray] = {}
    for raw_name, value in _flatten_mapping(channel_payload).items():
        name = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        if name in ("theta", "gamma", "sigma_phi_r", "sigma_std") and name not in canonical:
            canonical[name] = _coerce_numeric_array(value, name=name)
    return canonical


def _extract_mesh_vertices(
    fixture_like: Mapping[str, Any],
    channel_payload: Mapping[str, Any],
) -> np.ndarray | None:
    for payload in (channel_payload, fixture_like):
        flattened = _flatten_mapping(payload)
        for raw_name, value in flattened.items():
            name = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
            if name != "mesh_vertices":
                continue
            vertices = _coerce_numeric_array(value, name="mesh_vertices")
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("Torsion paper replay mesh_vertices must have shape (n, 3).")
            return vertices
    return None


def _looks_like_paper_raw_payload(
    fixture_like: Mapping[str, Any],
    channel_payload: Mapping[str, Any],
) -> bool:
    return _extract_mesh_vertices(fixture_like, channel_payload) is not None


def _find_anchor_indices(vertices: np.ndarray, *, height: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = _PAPER_ANCHOR_FRACTION * float(height)
    top_indices = np.flatnonzero(vertices[:, 2] > threshold)
    bottom_indices = np.flatnonzero(vertices[:, 2] < -threshold)
    if top_indices.size == 0 or bottom_indices.size == 0:
        raise ValueError(
            "Torsion paper replay could not reconstruct anchor regions from mesh_vertices and height."
        )
    return top_indices, bottom_indices


def _rotate_points_around_z(vertices: np.ndarray, indices: np.ndarray, theta: float) -> np.ndarray:
    points = vertices[np.asarray(indices, dtype=int), :].copy()
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    points[:, :2] = points[:, :2] @ rotation.T
    return points


def reconstruct_torsion_anchor_geometry(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct paper-replay anchor regions and dz from mesh vertices."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")
    resolved_geometry = parse_torsion_lane_geometry(fixture_like, geometry=geometry)
    channel_payload = _extract_channel_payload(fixture_like)
    vertices = _extract_mesh_vertices(fixture_like, channel_payload)
    if vertices is None:
        raise ValueError("Torsion paper replay requires mesh_vertices to reconstruct anchor geometry.")
    top_indices, bottom_indices = _find_anchor_indices(vertices, height=resolved_geometry["height"])
    dz = float(np.min(vertices[top_indices, 2]) - np.max(vertices[bottom_indices, 2]))
    if not np.isfinite(dz) or dz <= 0.0:
        raise ValueError("Torsion paper replay requires positive anchor dz from mesh_vertices.")
    return {
        "mesh_vertices": vertices,
        "top_indices": top_indices,
        "bottom_indices": bottom_indices,
        "dz": dz,
    }


def _coerce_anchor_force_timeseries(
    payload: object,
    *,
    name: str,
    particle_count: int,
) -> np.ndarray:
    try:
        array = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Torsion paper replay {name!r} must be numeric data.") from exc
    if array.size == 0:
        raise ValueError(f"Torsion paper replay {name!r} must contain data.")
    if array.ndim == 3 and array.shape[1:] == (particle_count, 3):
        return array
    if array.ndim == 2 and array.shape[1] == particle_count * 3:
        return array.reshape(array.shape[0], particle_count, 3)
    raise ValueError(
        f"Torsion paper replay {name!r} must have shape (timesteps, particles, 3) "
        "or flattened shape (timesteps, particles*3)."
    )


def _paper_torque_statistics(forces: np.ndarray, *, positions: np.ndarray) -> tuple[float, float]:
    torque_per_timestep = np.cross(positions[None, :, :], forces, axis=2).sum(axis=1)
    start = int((1.0 - _PAPER_STEADY_STATE_FRACTION) * torque_per_timestep.shape[0])
    steady_state = torque_per_timestep[start:]
    if steady_state.shape[0] == 0:
        raise ValueError("Torsion paper replay requires at least one timestep for anchor torque averaging.")
    if not np.all(np.isfinite(steady_state)):
        raise ValueError("Torsion paper replay steady-state anchor torque contains non-finite values.")
    tau_z = steady_state[:, 2]
    return float(np.mean(tau_z)), float(np.std(tau_z))


def reconstruct_torsion_paper_channels(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct canonical torsion channels from paper-style anchor-force payloads."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    resolved_geometry = parse_torsion_lane_geometry(fixture_like, geometry=geometry)
    resolved_controls = parse_torsion_lane_controls(fixture_like, controls=controls)
    anchor_geometry = reconstruct_torsion_anchor_geometry(fixture_like, geometry=geometry)
    channel_payload = _extract_channel_payload(fixture_like)
    flattened = _flatten_mapping(channel_payload)

    force_sources: dict[str, Any] = {}
    for raw_name, value in flattened.items():
        name = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        if name in ("constrained_vertex_forces_min", "constrained_vertex_forces_max"):
            force_sources[name] = value

    missing = [name for name in ("constrained_vertex_forces_min", "constrained_vertex_forces_max") if name not in force_sources]
    if missing:
        raise ValueError(
            "Torsion paper replay requires both anchor_min/anchor_max force payloads for raw reconstruction."
        )

    theta = resolved_controls["theta"]
    top_positions = _rotate_points_around_z(
        anchor_geometry["mesh_vertices"],
        anchor_geometry["top_indices"],
        theta,
    )
    bottom_positions = _rotate_points_around_z(
        anchor_geometry["mesh_vertices"],
        anchor_geometry["bottom_indices"],
        -theta,
    )
    bottom_forces = _coerce_anchor_force_timeseries(
        force_sources["constrained_vertex_forces_min"],
        name="anchor_min_forces",
        particle_count=bottom_positions.shape[0],
    )
    top_forces = _coerce_anchor_force_timeseries(
        force_sources["constrained_vertex_forces_max"],
        name="anchor_max_forces",
        particle_count=top_positions.shape[0],
    )

    bottom_tau_z, bottom_std = _paper_torque_statistics(bottom_forces, positions=bottom_positions)
    top_tau_z, top_std = _paper_torque_statistics(top_forces, positions=top_positions)
    area = _FORCE_TO_STRESS_AREA_SCALE * float(resolved_geometry["radius"]) ** 2
    sigma_phi_r = 0.5 * (abs(bottom_tau_z / area) + abs(top_tau_z / area))
    sigma_std = 0.5 * (abs(bottom_std / area) + abs(top_std / area))

    return {
        "gamma": compute_torsion_paper_gamma(
            np.array([theta], dtype=float),
            radius=resolved_geometry["radius"],
            dz=float(anchor_geometry["dz"]),
        ),
        "sigma_phi_r": np.array([sigma_phi_r], dtype=float),
        "sigma_std": np.array([sigma_std], dtype=float),
    }


def _flatten_mapping(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_name, value in payload.items():
        if not isinstance(raw_name, str):
            continue
        name = _canonical_name(raw_name)
        full_name = f"{prefix}_{name}" if prefix else name
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=full_name))
        else:
            flattened[full_name] = value
    return flattened


def _resolve_force_source(channel_payload: Mapping[str, Any]) -> np.ndarray:
    remapped: dict[str, Any] = {}
    for raw_name, value in _flatten_mapping(channel_payload).items():
        canonical = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        remapped.setdefault(canonical, value)

    if "constrained_vertex_forces" in remapped:
        return _reduce_force_payload(
            remapped["constrained_vertex_forces"],
            name="constrained_vertex_forces",
        )

    force_terms = []
    for name in ("constrained_vertex_forces_min", "constrained_vertex_forces_max"):
        if name in remapped:
            force_terms.append(_reduce_force_payload(remapped[name], name=name))
    if force_terms:
        baseline = force_terms[0]
        for term in force_terms[1:]:
            if term.shape != baseline.shape:
                raise ValueError("Torsion constrained-vertex force sources must share the same shape.")
        return np.sum(np.stack(force_terms, axis=0), axis=0)

    raise ValueError(
        "Torsion lane requires a constrained-vertex force source via `constrained_vertex_forces` "
        "or paired `anchor_min_forces`/`anchor_max_forces` style keys."
    )


def parse_torsion_lane_channels(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Extract canonical torsion channels `gamma` and `sigma_phi_r`."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    resolved_geometry = parse_torsion_lane_geometry(fixture_like, geometry=geometry)
    channel_payload = _extract_channel_payload(fixture_like)
    paper_raw_payload = _looks_like_paper_raw_payload(fixture_like, channel_payload)
    canonical_channels = _extract_canonical_channels(channel_payload)
    theta = canonical_channels.get("theta")
    gamma = canonical_channels.get("gamma")
    sigma_phi_r = canonical_channels.get("sigma_phi_r")
    sigma_std = canonical_channels.get("sigma_std")

    if gamma is None:
        if theta is None:
            if not paper_raw_payload and _has_force_source(channel_payload):
                raise ValueError("Torsion lane requires a theta channel or control to compute gamma.")
            try:
                reconstructed = reconstruct_torsion_paper_channels(
                    fixture_like,
                    geometry=geometry,
                    controls=controls,
                )
            except ValueError as exc:
                if paper_raw_payload:
                    raise
                raise ValueError(
                    "Torsion lane requires canonical gamma/sigma_phi_r or reconstructable paper raw anchors."
                ) from exc
            return reconstructed
        try:
            anchor_geometry = reconstruct_torsion_anchor_geometry(fixture_like, geometry=geometry)
        except ValueError:
            gamma = compute_torsion_gamma(
                theta,
                radius=resolved_geometry["radius"],
                height=resolved_geometry["height"],
            )
        else:
            gamma = compute_torsion_paper_gamma(
                theta,
                radius=resolved_geometry["radius"],
                dz=float(anchor_geometry["dz"]),
            )

    if sigma_phi_r is None:
        try:
            reconstructed = reconstruct_torsion_paper_channels(
                fixture_like,
                geometry=geometry,
                controls=controls,
            )
        except ValueError:
            if paper_raw_payload:
                raise
            constrained_force = _resolve_force_source(channel_payload)
            sigma_phi_r = compute_torsion_sigma_phi_r(
                constrained_force,
                radius=resolved_geometry["radius"],
                height=resolved_geometry["height"],
            )
        else:
            sigma_phi_r = reconstructed["sigma_phi_r"]
            sigma_std = reconstructed.get("sigma_std")

    if gamma.shape != sigma_phi_r.shape:
        raise ValueError("Torsion canonical gamma and sigma_phi_r must have matching shapes.")

    channels = {
        "gamma": gamma,
        "sigma_phi_r": sigma_phi_r,
    }
    if sigma_std is not None:
        if sigma_std.shape != sigma_phi_r.shape:
            raise ValueError("Torsion canonical sigma_std and sigma_phi_r must have matching shapes.")
        channels["sigma_std"] = sigma_std
    return channels


def _has_force_source(channel_payload: Mapping[str, Any]) -> bool:
    for raw_name in _flatten_mapping(channel_payload):
        name = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        if name in (
            "constrained_vertex_forces",
            "constrained_vertex_forces_min",
            "constrained_vertex_forces_max",
        ):
            return True
    return False


def parse_torsion_lane_controls(
    fixture_like: Mapping[str, Any],
    *,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract and validate torsion controls."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")
    payload = _extract_control_payload(fixture_like, controls)
    normalized: dict[str, float] = {}
    for raw_name, value in payload.items():
        if not isinstance(raw_name, str):
            continue
        canonical = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        if canonical == "theta":
            normalized["theta"] = _normalize_scalar(value, name="theta")
    missing = [name for name in _REQUIRED_CONTROLS if name not in normalized]
    if missing:
        raise ValueError("Torsion lane requires controls: " + ", ".join(_REQUIRED_CONTROLS) + ".")
    return normalized


def extract_torsion_lane(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a canonical GV torsion sampling lane from fixture-like input."""

    return {
        "axis": _TORSION_AXIS,
        "geometry": parse_torsion_lane_geometry(fixture_like, geometry=geometry),
        "controls": parse_torsion_lane_controls(fixture_like, controls=controls),
        "channels": parse_torsion_lane_channels(
            fixture_like,
            geometry=geometry,
            controls=controls,
        ),
    }


def parse_torsion_sampling_lane(
    fixture_like: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for torsion lane parsing."""

    return extract_torsion_lane(
        fixture_like,
        geometry=geometry,
        controls=controls,
    )


__all__ = [
    "_FORCE_TO_STRESS_AREA_SCALE",
    "_PAPER_ANCHOR_FRACTION",
    "_PAPER_GAMMA_RADIUS_FACTOR",
    "_PAPER_STEADY_STATE_FRACTION",
    "_REQUIRED_CONTROLS",
    "_TORSION_AXIS",
    "compute_torsion_gamma",
    "compute_torsion_paper_gamma",
    "compute_torsion_sigma_phi_r",
    "extract_torsion_lane",
    "parse_torsion_lane_channels",
    "parse_torsion_lane_controls",
    "parse_torsion_lane_geometry",
    "parse_torsion_sampling_lane",
    "reconstruct_torsion_anchor_geometry",
    "reconstruct_torsion_paper_channels",
]
