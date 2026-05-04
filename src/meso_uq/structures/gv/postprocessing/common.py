from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np

from ..numerical_data import (
    GVNumericalDatasetManifest,
    build_gv_numerical_dataset_manifest,
    campaign_dataset_hdf5_path,
    campaign_dataset_manifest_path,
    validate_gv_numerical_manifest,
)


@dataclass(frozen=True)
class GVNumericalPostprocessResult:
    """Result payload returned by numerical-data postprocessors."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    hdf5_path: Path


def _import_h5py():
    """Import `h5py` lazily.

    The import is isolated so tests can monkeypatch this helper when optional
    dependencies are absent.
    """

    try:
        import h5py

        return h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-dependent branch
        raise RuntimeError(
            "`h5py` is required to write numerical GV datasets. Install h5py and retry."
        ) from exc


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, (np.number, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _manifest_to_payload(manifest: GVNumericalDatasetManifest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, GVNumericalDatasetManifest):
        payload = manifest.to_manifest()
    elif isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        raise ValueError("manifest must be a GVNumericalDatasetManifest or mapping payload.")

    validate_gv_numerical_manifest(payload)
    return payload


def _coerce_numeric_array(payload: Any, *, channel_name: str) -> np.ndarray:
    try:
        array = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Channel {channel_name!r} must be numeric data.") from exc

    if array.size == 0:
        raise ValueError(f"Channel {channel_name!r} must contain data.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"Channel {channel_name!r} contains non-finite values.")

    return array


def validate_numeric_channels(channels: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Normalize a channel mapping and enforce finite numeric arrays."""

    if not isinstance(channels, Mapping):
        raise ValueError("channels must be a mapping from channel name to numeric array.")

    normalized: dict[str, np.ndarray] = {}
    for name, payload in channels.items():
        if not isinstance(name, str):
            raise ValueError("Channel names must be strings.")
        normalized[name] = _coerce_numeric_array(payload, channel_name=name)
    return normalized


def _write_manifest_json(payload: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_hdf5_manifest_attrs(h5_handle: Any, manifest_payload: Mapping[str, Any]) -> None:
    """Attach manifest metadata on the HDF5 root attrs where feasible."""

    manifest_payload_json = json.dumps(_as_jsonable(manifest_payload), sort_keys=True)
    h5_handle.attrs["manifest_json"] = manifest_payload_json

    for key, value in manifest_payload.items():
        try:
            if isinstance(value, str):
                h5_handle.attrs[str(key)] = str(value)
            elif isinstance(value, bool):
                h5_handle.attrs[str(key)] = bool(value)
            elif isinstance(value, int):
                h5_handle.attrs[str(key)] = int(value)
            elif isinstance(value, float):
                h5_handle.attrs[str(key)] = float(value)
            elif value is None:
                h5_handle.attrs[str(key)] = "null"
            else:
                # Store structured metadata as a JSON attribute.
                h5_handle.attrs[f"{key}_json"] = json.dumps(_as_jsonable(value), sort_keys=True)
        except Exception:
            # HDF5 attrs are optional; skip values that cannot be serialized.
            continue


def write_numerical_dataset_artifacts(
    *,
    manifest: GVNumericalDatasetManifest | Mapping[str, Any],
    channels: Mapping[str, Any],
) -> GVNumericalPostprocessResult:
    """Write a numerical GV dataset (HDF5 + manifest JSON)."""

    manifest_payload = _manifest_to_payload(manifest)
    dataset_id = str(manifest_payload["dataset_id"])
    campaign_id = str(manifest_payload["campaign_id"])

    hdf5_path = campaign_dataset_hdf5_path(campaign_id=campaign_id, dataset_id=dataset_id)
    manifest_path = campaign_dataset_manifest_path(campaign_id=campaign_id, dataset_id=dataset_id)

    if not hdf5_path.as_posix().startswith("_runs/"):
        raise ValueError("GV numerical datasets must be written under the _runs tree.")

    normalized_channels = validate_numeric_channels(channels)
    if not normalized_channels:
        raise ValueError("At least one numeric channel is required.")

    try:
        h5py = _import_h5py()
    except ImportError as exc:
        raise RuntimeError(
            "`h5py` is required to write numerical GV datasets. Install h5py and retry."
        ) from exc
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as hdf5:
        _write_hdf5_manifest_attrs(hdf5, manifest_payload)
        for channel_name, values in normalized_channels.items():
            hdf5.create_dataset(channel_name, data=values)

    # Write manifest only after successful HDF5 creation to avoid partial artifacts
    # when I/O or optional dependencies fail.
    _write_manifest_json(manifest_payload, manifest_path)

    return GVNumericalPostprocessResult(
        manifest=dict(manifest_payload),
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
    )


def build_gv_numerical_manifest(
    *,
    campaign_id: str,
    experiment: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, Any],
    controls: Mapping[str, Any],
    raw_provenance: Mapping[str, Any],
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
) -> GVNumericalDatasetManifest:
    """Build and validate a canonical GV numerical manifest for postprocessing."""

    return build_gv_numerical_dataset_manifest(
        campaign_id=campaign_id,
        structure="gv",
        experiment=experiment,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=raw_provenance,
        quality_flags=quality_flags,
        units=units,
        normalization=normalization,
    )


__all__ = [
    "GVNumericalPostprocessResult",
    "build_gv_numerical_manifest",
    "_import_h5py",
    "_coerce_numeric_array",
    "_write_hdf5_manifest_attrs",
    "validate_numeric_channels",
    "write_numerical_dataset_artifacts",
]
