from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from .common import (
    GVNumericalPostprocessResult,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)
from ..numerical_data import build_gv_numerical_dataset_manifest


_POSTPROCESSOR_EXPERIMENT = "torsion"
_TORSION_REQUIRED_CHANNELS = ("torsion_coord", "torsion_response")
_TORSION_ALIAS_MAP: dict[str, str] = {
    # axis-like / twist coordinate channels
    "torsion_coord": "torsion_coord",
    "theta": "torsion_coord",
    "twist_angle": "torsion_coord",
    "rotation": "torsion_coord",
    "angle": "torsion_coord",

    # torsional response channels
    "torsion_response": "torsion_response",
    "response": "torsion_response",
    "axis_response": "torsion_response",
    "twist_response": "torsion_response",

    # torque-twist proxy candidates
    "torsion_torque": "torsion_torque_proxy",
    "torque": "torsion_torque_proxy",
    "twist_torque": "torsion_torque_proxy",
    "torque_torque_proxy": "torsion_torque_proxy",
    "torsional_torque": "torsion_torque_proxy",

    # cap angular displacement candidates
    "cap_angular_displacement": "cap_angular_displacement",
    "cap_rotation": "cap_angular_displacement",
    "cap_angle": "cap_angular_displacement",
    "twist_angle_cap": "cap_angular_displacement",

    # stored elastic response proxy
    "stored_elastic_response": "stored_elastic_response",
    "stored_elastic_response_proxy": "stored_elastic_response",
    "elastic_response_proxy": "stored_elastic_response",
    "elastic_response": "stored_elastic_response",

    # anchor/cap motion
    "cap_motion": "cap_motion",
    "anchor_motion": "cap_motion",
    "anchor_displacement": "cap_motion",
    "cap_displacement": "cap_motion",

    # force channels
    "force": "force",
    "tot_force": "force",
    "total_force": "force",
}


def _canonicalize_channel_name(raw_name: str) -> str:
    normalized = re.sub(r"[^\w]+", "_", str(raw_name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return ""
    return _TORSION_ALIAS_MAP.get(normalized, normalized)


def _extract_channel_payload(fixture_like: Mapping[str, Any]) -> Mapping[str, Any]:
    if "channels" in fixture_like:
        channel_payload = fixture_like["channels"]
    elif "observables" in fixture_like:
        channel_payload = fixture_like["observables"]
    else:
        channel_payload = fixture_like

    if not isinstance(channel_payload, Mapping):
        raise ValueError("No numeric channel mapping found in fixture-like payload.")
    return channel_payload


def _quality_flags_for_channels(
    channels: Mapping[str, Any],
    requested_quality_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total_count = 0
    finite_count = 0
    canary_failures: list[str] = []

    for name, values in channels.items():
        finite_mask = np.isfinite(values)
        total_count += int(finite_mask.size)
        finite_count += int(np.count_nonzero(finite_mask))
        if not np.all(finite_mask):
            missing = int(finite_mask.size - np.count_nonzero(finite_mask))
            canary_failures.append(f"{name}:{missing} non-finite")

    finite_ratio = finite_count / total_count if total_count else 0.0
    resolved_flags: dict[str, Any] = {
        "finite_observables": bool(finite_ratio == 1.0),
        "finite_observable_ratio": float(finite_ratio),
        "canary_failures": tuple(canary_failures),
    }

    if requested_quality_flags is None:
        return resolved_flags

    merged = dict(requested_quality_flags)
    merged["finite_observables"] = bool(resolved_flags["finite_observables"])
    merged["finite_observable_ratio"] = float(finite_ratio)
    merged_failures = list(requested_quality_flags.get("canary_failures", ()))
    merged_failures.extend(canary_failures)
    merged["canary_failures"] = tuple(str(item) for item in merged_failures)
    return merged


def parse_torsion_fixture_channels(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool = True,
) -> dict[str, np.ndarray]:
    """Parse fixture-like torsion channels into canonical numeric channel arrays."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    channel_payload = _extract_channel_payload(fixture_like)
    channels: dict[str, Any] = {}

    for raw_name, values in channel_payload.items():
        if not isinstance(raw_name, str):
            continue
        canonical_name = _canonicalize_channel_name(raw_name)
        if not include_optional_channels and canonical_name not in _TORSION_REQUIRED_CHANNELS:
            continue
        channels[canonical_name] = values

    missing = [name for name in _TORSION_REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(
            "Torsion parser requires at least channels: "
            + ", ".join(_TORSION_REQUIRED_CHANNELS)
        )

    return validate_numeric_channels(channels)


def build_torsion_numerical_manifest(
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
) -> object:
    """Build a canonical GV torsion numerical manifest."""

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


def process_torsion_numerical_dataset(
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
    strict_finite_observables: bool = True,
) -> GVNumericalPostprocessResult:
    """Build and write a canonical GV torsion numerical dataset artifact."""

    channels = parse_torsion_fixture_channels(
        fixture_like,
        include_optional_channels=include_optional_channels,
    )

    resolved_quality_flags = _quality_flags_for_channels(channels, requested_quality_flags=quality_flags)
    if strict_finite_observables and not bool(resolved_quality_flags["finite_observables"]):
        raise ValueError("Torsion fixture channels contain non-finite observables.")

    manifest = build_torsion_numerical_manifest(
        campaign_id=campaign_id,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=raw_provenance,
        quality_flags=resolved_quality_flags,
        units=units,
        normalization=normalization,
    )

    return write_numerical_dataset_artifacts(manifest=manifest, channels=channels)


__all__ = [
    "_POSTPROCESSOR_EXPERIMENT",
    "_TORSION_REQUIRED_CHANNELS",
    "parse_torsion_fixture_channels",
    "process_torsion_numerical_dataset",
    "build_torsion_numerical_manifest",
]
