from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

_TORSION_AXIS = "gamma"
_REQUIRED_CONTROLS = ("theta",)
_FORCE_TO_STRESS_AREA_SCALE = 2.0 * np.pi

_CHANNEL_ALIASES = {
    "theta": "theta",
    "twist_angle": "theta",
    "torsion_coord": "theta",
    "rotation": "theta",
    "angle": "theta",
    "gamma": "gamma",
    "shear_strain": "gamma",
    "sigma_phi_r": "sigma_phi_r",
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
    "anchor_max_force": "constrained_vertex_forces_max",
    "anchor_max_forces": "constrained_vertex_forces_max",
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
) -> dict[str, np.ndarray]:
    """Extract canonical torsion channels `gamma` and `sigma_phi_r`."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    resolved_geometry = parse_torsion_lane_geometry(fixture_like, geometry=geometry)
    channel_payload = _extract_channel_payload(fixture_like)
    flattened = _flatten_mapping(channel_payload)

    theta = None
    gamma = None
    sigma_phi_r = None
    for raw_name, value in flattened.items():
        canonical = _CHANNEL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
        if canonical == "theta" and theta is None:
            theta = _coerce_numeric_array(value, name="theta")
        elif canonical == "gamma" and gamma is None:
            gamma = _coerce_numeric_array(value, name="gamma")
        elif canonical == "sigma_phi_r" and sigma_phi_r is None:
            sigma_phi_r = _coerce_numeric_array(value, name="sigma_phi_r")

    if gamma is None:
        if theta is None:
            raise ValueError("Torsion lane requires a theta/torsion_coord channel or canonical gamma.")
        gamma = compute_torsion_gamma(
            theta,
            radius=resolved_geometry["radius"],
            height=resolved_geometry["height"],
        )

    if sigma_phi_r is None:
        constrained_force = _resolve_force_source(channel_payload)
        sigma_phi_r = compute_torsion_sigma_phi_r(
            constrained_force,
            radius=resolved_geometry["radius"],
            height=resolved_geometry["height"],
        )

    if gamma.shape != sigma_phi_r.shape:
        raise ValueError("Torsion canonical gamma and sigma_phi_r must have matching shapes.")

    return {
        "gamma": gamma,
        "sigma_phi_r": sigma_phi_r,
    }


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
        "channels": parse_torsion_lane_channels(fixture_like, geometry=geometry),
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
    "_REQUIRED_CONTROLS",
    "_TORSION_AXIS",
    "compute_torsion_gamma",
    "compute_torsion_sigma_phi_r",
    "extract_torsion_lane",
    "parse_torsion_lane_channels",
    "parse_torsion_lane_controls",
    "parse_torsion_lane_geometry",
    "parse_torsion_sampling_lane",
]
