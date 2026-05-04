from __future__ import annotations

import re
from typing import Any, Mapping

from ..numerical_data import build_gv_numerical_dataset_manifest
from .common import (
    GVNumericalPostprocessResult,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)


_POSTPROCESSOR_EXPERIMENT = "buckling"

_REQUIRED_RESPONSE_CHANNELS = (
    "buckling_response",
    "force_response",
    "pressure_response",
    "shape_amplitude",
    "deformation_amplitude",
)

_KNOWN_CHANNELS = (
    "buck",
    "bpress",
    *_REQUIRED_RESPONSE_CHANNELS,
)

_TRAJECTORY_CHANNEL_PREFIXES = ("mesh", "ply", "anchor", "object", "object_stats", "stats")

_CHANNEL_NAME_ALIASES = {
    "response": "buckling_response",
    "buckling": "buckling_response",
    "buckling_response": "buckling_response",
    "force_response": "force_response",
    "force": "force_response",
    "pressure_response": "pressure_response",
    "pressure": "pressure_response",
    "shape": "shape_amplitude",
    "shape_amp": "shape_amplitude",
    "shapeamplitude": "shape_amplitude",
    "deformation": "deformation_amplitude",
    "deformation_amp": "deformation_amplitude",
    "deformationamplitude": "deformation_amplitude",
}


def _canonical_channel_name(name: str) -> str:
    canonical = re.sub(r"[^\w]+", "_", str(name).strip().lower())
    canonical = re.sub(r"_+", "_", canonical).strip("_")
    if not canonical:
        return ""
    return _CHANNEL_NAME_ALIASES.get(canonical, canonical)


def _extract_channel_mapping(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    channels: dict[str, Any] = {}

    for raw_name, raw_value in payload.items():
        if not isinstance(raw_name, str):
            continue
        candidate_name = raw_name if not prefix else f"{prefix}_{raw_name}"
        name = _canonical_channel_name(candidate_name)

        if isinstance(raw_value, Mapping):
            nested = _extract_channel_mapping(raw_value, prefix=name)
            channels.update(nested)
            continue

        channels[name] = raw_value

    return channels


def _filter_channel_candidates(
    channels: Mapping[str, Any],
    *,
    include_optional_channels: bool,
) -> dict[str, Any]:
    if include_optional_channels:
        return dict(channels)

    included: dict[str, Any] = {}
    for name, value in channels.items():
        if name in _KNOWN_CHANNELS:
            included[name] = value
            continue
        if any(name == prefix or name.startswith(prefix + "_") for prefix in _TRAJECTORY_CHANNEL_PREFIXES):
            included[name] = value

    return included


def _required_response_channels_present(channels: Mapping[str, Any]) -> bool:
    return any(name in channels for name in _REQUIRED_RESPONSE_CHANNELS)


def parse_buckling_fixture_channels(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool = True,
) -> dict[str, Any]:
    """Parse fixture-like payloads into canonical buckling numeric channels.

    Accepted fixture layouts:
    - channel mapping in the payload root,
    - mapping under ``channels``,
    - mapping under ``observables``.
    """

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    if "channels" in fixture_like:
        channel_like = fixture_like.get("channels")
    elif "observables" in fixture_like:
        channel_like = fixture_like.get("observables")
    else:
        channel_like = fixture_like

    if not isinstance(channel_like, Mapping):
        raise ValueError("No numeric channel mapping found in fixture-like payload.")

    raw_channels = _extract_channel_mapping(channel_like)
    candidate_channels = _filter_channel_candidates(
        raw_channels,
        include_optional_channels=include_optional_channels,
    )

    if not candidate_channels:
        raise ValueError("Buckling parser received no numeric channels.")

    if not _required_response_channels_present(candidate_channels):
        raise ValueError(
            "Buckling parser requires at least one response channel: "
            + ", ".join(_REQUIRED_RESPONSE_CHANNELS)
            + "."
        )

    return validate_numeric_channels(candidate_channels)


def _build_buckling_manifest(
    *,
    campaign_id: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, Any],
    controls: Mapping[str, Any],
    raw_provenance: Mapping[str, Any],
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
):
    return build_gv_numerical_dataset_manifest(
        campaign_id=campaign_id,
        structure="gv",
        experiment=_POSTPROCESSOR_EXPERIMENT,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=raw_provenance,
        quality_flags=quality_flags,
        units=units,
        normalization=normalization,
    )


def process_buckling_numerical_dataset(
    *,
    campaign_id: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, Any],
    controls: Mapping[str, Any],
    raw_provenance: Mapping[str, Any],
    fixture_like: Mapping[str, Any],
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
    include_optional_channels: bool = True,
) -> GVNumericalPostprocessResult:
    """Build and write a canonical buckling numerical postprocessing artifact."""

    channels = parse_buckling_fixture_channels(
        fixture_like,
        include_optional_channels=include_optional_channels,
    )

    manifest = _build_buckling_manifest(
        campaign_id=campaign_id,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=raw_provenance,
        quality_flags=quality_flags,
        units=units,
        normalization=normalization,
    )

    return write_numerical_dataset_artifacts(manifest=manifest, channels=channels)


__all__ = [
    "_POSTPROCESSOR_EXPERIMENT",
    "_KNOWN_CHANNELS",
    "_REQUIRED_RESPONSE_CHANNELS",
    "parse_buckling_fixture_channels",
    "process_buckling_numerical_dataset",
]
