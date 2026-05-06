from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

_BUCKLING_AXIS = "buck"
_REQUIRED_CONTROLS = ("buck", "bpress")

_CHANNEL_ALIASES = {
    "pressure": "bpress",
    "background_pressure": "bpress",
    "b_pressure": "bpress",
    "volumetric_strain": "relative_volume",
    "volume_strain": "relative_volume",
    "relative_deformation": "relative_deformation",
    "shape": "shape_amplitude",
    "deformation": "deformation_amplitude",
    "buckling": "buckling_response",
}

_CONTROL_ALIASES = {
    "buckling": "buck",
    "pressure": "bpress",
    "background_pressure": "bpress",
    "b_pressure": "bpress",
}


def _canonical_name(raw_name: object) -> str:
    normalized = re.sub(r"[^\w]+", "_", str(raw_name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _normalize_scalar_control(value: object, *, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        scalar = float(array.item())
    elif array.size == 1:
        scalar = float(array.reshape(-1)[0])
    else:
        raise ValueError(f"Buckling control {name!r} must be a scalar value.")
    if not np.isfinite(scalar):
        raise ValueError(f"Buckling control {name!r} must be finite.")
    return scalar


def _extract_controls(controls_like: Mapping[str, Any] | None = None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_name, value in (controls_like or {}).items():
        if not isinstance(raw_name, str):
            continue
        name = _canonical_name(raw_name)
        name = _CONTROL_ALIASES.get(name, _CHANNEL_ALIASES.get(name, name))
        normalized[name] = _normalize_scalar_control(value, name=name)

    missing = [name for name in _REQUIRED_CONTROLS if name not in normalized]
    if missing:
        raise ValueError(
            "Buckling lane requires controls: " + ", ".join(_REQUIRED_CONTROLS) + "."
        )
    return normalized


def _extract_control_payload(
    fixture_like: Mapping[str, Any],
    controls: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if controls is not None:
        return dict(controls)

    if "controls" in fixture_like:
        if not isinstance(fixture_like["controls"], Mapping):
            raise ValueError("Buckling fixture controls must be a mapping.")
        return dict(fixture_like["controls"])

    return {}


def _extract_channel_payload(fixture_like: Mapping[str, Any]) -> dict[str, Any]:
    if "channels" in fixture_like:
        channel_like = fixture_like["channels"]
    elif "observables" in fixture_like:
        channel_like = fixture_like["observables"]
    else:
        channel_like = fixture_like

    if not isinstance(channel_like, Mapping):
        raise ValueError("No numeric channel mapping found in buckling fixture.")
    return dict(channel_like)


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


def parse_buckling_lane_channels(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool = True,
) -> dict[str, np.ndarray]:
    """Extract and validate buckling lane channels from fixture-like payloads."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    from ..postprocessing import buckling as postprocessing

    channel_payload = _flatten_mapping(_extract_channel_payload(fixture_like))
    remapped: dict[str, Any] = {}
    for raw_name, values in channel_payload.items():
        if not isinstance(raw_name, str):
            continue
        canonical = _canonical_name(raw_name)
        canonical = _CHANNEL_ALIASES.get(canonical, canonical)
        remapped.setdefault(canonical, values)

    return postprocessing.parse_buckling_fixture_channels(
        {"channels": remapped},
        include_optional_channels=include_optional_channels,
    )


def parse_buckling_lane_controls(
    fixture_like: Mapping[str, Any],
    *,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract and validate buckling lane controls."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")
    return _extract_controls(_extract_control_payload(fixture_like, controls=controls))


def extract_buckling_lane(
    fixture_like: Mapping[str, Any],
    *,
    controls: Mapping[str, Any] | None = None,
    include_optional_channels: bool = True,
) -> dict[str, Any]:
    """Extract buckling lane controls and channels from fixture-like input."""

    return {
        "axis": _BUCKLING_AXIS,
        "controls": parse_buckling_lane_controls(
            fixture_like,
            controls=controls,
        ),
        "channels": parse_buckling_lane_channels(
            fixture_like,
            include_optional_channels=include_optional_channels,
        ),
    }


def parse_buckling_sampling_lane(
    fixture_like: Mapping[str, Any],
    *,
    controls: Mapping[str, Any] | None = None,
    include_optional_channels: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for lane parsing."""

    return extract_buckling_lane(
        fixture_like,
        controls=controls,
        include_optional_channels=include_optional_channels,
    )


__all__ = [
    "_BUCKLING_AXIS",
    "_REQUIRED_CONTROLS",
    "extract_buckling_lane",
    "parse_buckling_lane_channels",
    "parse_buckling_lane_controls",
    "parse_buckling_sampling_lane",
]
