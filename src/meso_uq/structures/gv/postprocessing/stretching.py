from __future__ import annotations

from typing import Any, Mapping

from .common import (
    GVNumericalPostprocessResult,
    build_gv_numerical_manifest,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)


_POSTPROCESSOR_EXPERIMENT = "stretching"
_STRETCHING_REQUIRED_CHANNELS = ("tot_force", "force", "displacement")


def parse_stretching_fixture_channels(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool = True,
) -> dict[str, Any]:
    """Parse fixture-like postprocessing payload into candidate numeric channels.

    The parser supports several fixture layouts used by lightweight unit tests:
    - top-level channel names as the payload itself;
    - mapping inside the `channels` field;
    - mapping inside the `observables` field.
    """

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    if "channels" in fixture_like:
        channel_candidate = fixture_like.get("channels")
    elif "observables" in fixture_like:
        channel_candidate = fixture_like.get("observables")
    else:
        channel_candidate = fixture_like

    if not isinstance(channel_candidate, Mapping):
        raise ValueError("No numeric channel mapping found in fixture-like payload.")

    channel_payload: dict[str, Any] = {}
    for name, value in channel_candidate.items():
        if not isinstance(name, str):
            continue
        if not include_optional_channels and name not in _STRETCHING_REQUIRED_CHANNELS:
            continue
        channel_payload[name] = value

    missing = [name for name in _STRETCHING_REQUIRED_CHANNELS if name not in channel_payload]
    if missing:
        raise ValueError(
            "Stretching parser requires at least channels: "
            + ", ".join(_STRETCHING_REQUIRED_CHANNELS)
            + "."
        )

    return validate_numeric_channels(channel_payload)


def process_stretching_numerical_dataset(
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
    """Build and write a canonical stretching numerical postprocessing artifact."""

    channels = parse_stretching_fixture_channels(
        fixture_like,
        include_optional_channels=include_optional_channels,
    )

    manifest = build_gv_numerical_manifest(
        campaign_id=campaign_id,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=raw_provenance,
        experiment="stretching",
        quality_flags=quality_flags,
        units=units,
        normalization=normalization,
    )

    # Build under the shared postprocessing experiment name for now.
    return write_numerical_dataset_artifacts(manifest=manifest, channels=channels)


__all__ = [
    "_POSTPROCESSOR_EXPERIMENT",
    "_STRETCHING_REQUIRED_CHANNELS",
    "parse_stretching_fixture_channels",
    "process_stretching_numerical_dataset",
]
