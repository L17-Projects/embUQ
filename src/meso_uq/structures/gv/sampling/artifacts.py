"""Helpers for GV sampling artifact persistence and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from meso_uq.structures.gv.numerical_data import (
    build_gv_numerical_dataset_manifest,
    campaign_dataset_hdf5_path,
)
from meso_uq.structures.gv.postprocessing.common import (
    _import_h5py,
    _manifest_to_payload,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)
from meso_uq.structures.gv.sampling.buckling import parse_buckling_lane_channels
from meso_uq.structures.gv.sampling.stretching import parse_stretching_lane_channels

__all__ = [
    "GVSamplingArtifactsResult",
    "assert_sampling_artifacts_match",
    "sample_gv",
    "verify_sampling_artifacts_match",
    "write_sampling_artifacts",
]


def _coerce_float(value: object, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _coerce_controls(
    controls: Mapping[str, Any] | None,
    fixture_like: Mapping[str, Any] | None,
) -> dict[str, float]:
    if controls is None:
        if fixture_like is None:
            raise ValueError("controls are required for GV sampling.")
        fixture_controls = fixture_like.get("controls")
        if not isinstance(fixture_controls, Mapping):
            raise ValueError("fixture_like must provide controls under 'controls'.")
        controls_like = fixture_controls
    else:
        controls_like = controls

    normalized: dict[str, float] = {}
    for name, value in controls_like.items():
        normalized[str(name)] = _coerce_float(value, name=f"controls[{name}]")
    if not normalized:
        raise ValueError("At least one control value is required.")
    return normalized


def _coerce_channels(
    channels: Mapping[str, Any] | None,
    fixture_like: Mapping[str, Any] | None,
    experiment: str,
) -> dict[str, np.ndarray]:
    if channels is not None:
        return validate_numeric_channels(channels)

    if fixture_like is None:
        raise ValueError("Either --fixture-path or --channels must be provided.")

    if experiment == "stretching":
        return parse_stretching_lane_channels(fixture_like)
    if experiment == "buckling":
        return parse_buckling_lane_channels(fixture_like)
    raise ValueError(f"No fixture lane parser is available for {experiment!r}.")


def _coerce_payload_channels(channels: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return validate_numeric_channels(channels)


def _as_manifest_payload(raw_manifest: Any) -> dict[str, Any]:
    if isinstance(raw_manifest, Mapping):
        return dict(raw_manifest)
    if hasattr(raw_manifest, "to_manifest"):
        manifest_payload = raw_manifest.to_manifest()
        if not isinstance(manifest_payload, Mapping):
            raise ValueError("manifest.to_manifest() must return a mapping.")
        return dict(manifest_payload)
    raise ValueError("Manifest must be a mapping or expose a .to_manifest() method.")


def _extract_sampling_payload(
    result: Mapping[str, Any] | Any,
    *,
    channels: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Path | None, Path | None]:
    if isinstance(result, Mapping):
        manifest_like = result.get("manifest")
        channels_like = channels if channels is not None else result.get("channels")
        if manifest_like is None or channels_like is None:
            raise ValueError("Sampling payload must provide manifest and channels.")
        manifest = _as_manifest_payload(manifest_like)
        return manifest, _coerce_payload_channels(channels_like), _optional_path(result.get("manifest_path")), _optional_path(
            result.get("hdf5_path")
        )

    manifest_like = getattr(result, "manifest", None)
    channels_like = channels if channels is not None else getattr(result, "channels", None)
    if manifest_like is None or channels_like is None:
        raise ValueError("Sampling payload object must provide manifest and channels.")

    return (
        _as_manifest_payload(manifest_like),
        _coerce_payload_channels(channels_like),
        _optional_path(getattr(result, "manifest_path", None)),
        _optional_path(getattr(result, "hdf5_path", None)),
    )


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def write_sampling_artifacts(
    result: Mapping[str, Any] | Any,
    *,
    channels: Mapping[str, Any] | None = None,
) -> "GVSamplingArtifactsResult":
    manifest_payload, channels_payload, _, _ = _extract_sampling_payload(
        result,
        channels=channels,
    )
    validated_manifest = _manifest_to_payload(manifest_payload)
    write_result = write_numerical_dataset_artifacts(manifest=validated_manifest, channels=channels_payload)

    return GVSamplingArtifactsResult(
        manifest=validated_manifest,
        channels=channels_payload,
        manifest_path=write_result.manifest_path,
        hdf5_path=write_result.hdf5_path,
        campaign_id=str(validated_manifest["campaign_id"]),
        experiment=str(validated_manifest["experiment"]),
    )


def verify_sampling_artifacts_match(
    result: Mapping[str, Any] | Any,
    *,
    channels: Mapping[str, Any] | None = None,
    strict: bool = True,
    hdf5_path: Path | None = None,
) -> bool:
    manifest_payload, expected_channels, _, payload_hdf5 = _extract_sampling_payload(
        result,
        channels=channels,
    )
    dataset_id = str(manifest_payload["dataset_id"])
    campaign_id = str(manifest_payload["campaign_id"])
    target_hdf5 = hdf5_path or payload_hdf5 or campaign_dataset_hdf5_path(
        campaign_id=campaign_id,
        dataset_id=dataset_id,
    )

    with _import_h5py().File(target_hdf5, "r") as hdf5:
        available = {name for name in hdf5.keys()}
        expected = set(expected_channels)
        if strict and available != expected:
            missing = sorted(expected - available)
            extra = sorted(available - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing in HDF5: {missing}")
            if extra:
                details.append(f"extra in HDF5: {extra}")
            raise ValueError("; ".join(details))
        missing = expected - available
        if missing:
            raise ValueError(f"Missing channels in HDF5: {missing}")
        for name, expected_values in expected_channels.items():
            observed = np.asarray(hdf5[name][()])
            if not np.array_equal(np.asarray(expected_values), observed):
                raise ValueError(f"Channel {name!r} differs between result payload and HDF5.")

    return True


def assert_sampling_artifacts_match(
    result: Mapping[str, Any] | Any,
    *,
    channels: Mapping[str, Any] | None = None,
    strict: bool = True,
    hdf5_path: Path | None = None,
) -> None:
    if not verify_sampling_artifacts_match(
        result,
        channels=channels,
        strict=strict,
        hdf5_path=hdf5_path,
    ):
        raise AssertionError("Sampling artifact verification failed.")


@dataclass(frozen=True)
class GVSamplingArtifactsResult:
    """Result payload after sampling and optional artifact persistence."""

    manifest: dict[str, Any]
    channels: dict[str, np.ndarray]
    manifest_path: Path | None
    hdf5_path: Path | None
    campaign_id: str
    experiment: str


def sample_gv(
    *,
    campaign_id: str,
    experiment: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, Any],
    controls: Mapping[str, Any] | None = None,
    channels: Mapping[str, Any] | None = None,
    fixture_like: Mapping[str, Any] | None = None,
    raw_provenance: Mapping[str, Any] | None = None,
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
    dry_run: bool = False,
    write_artifacts: bool = True,
) -> GVSamplingArtifactsResult:
    """Create a GV sampling in-memory result and optionally persist HDF5 + manifests."""

    normalized_controls = _coerce_controls(controls, fixture_like=fixture_like)
    normalized_channels = _coerce_channels(channels, fixture_like=fixture_like, experiment=experiment)
    manifest = build_gv_numerical_dataset_manifest(
        campaign_id=campaign_id,
        structure="gv",
        experiment=experiment,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=normalized_controls,
        raw_provenance=dict({"workflow": "gv_sampling", **dict(raw_provenance or {})}),
        quality_flags=quality_flags,
        units=units,
        normalization=normalization,
    )

    if dry_run or not write_artifacts:
        return GVSamplingArtifactsResult(
            manifest=manifest.to_manifest(),
            channels=normalized_channels,
            manifest_path=None,
            hdf5_path=None,
            campaign_id=campaign_id,
            experiment=experiment,
        )

    written = write_sampling_artifacts(
        {
            "manifest": manifest,
            "channels": normalized_channels,
        }
    )
    return GVSamplingArtifactsResult(
        manifest=written.manifest,
        channels=written.channels,
        manifest_path=written.manifest_path,
        hdf5_path=written.hdf5_path,
        campaign_id=written.campaign_id,
        experiment=written.experiment,
    )
