from __future__ import annotations

from typing import Any, Mapping

from ..numerical_data import build_gv_numerical_dataset_manifest
from .common import (
    GVNumericalPostprocessResult,
    validate_numeric_channels,
    write_numerical_dataset_artifacts,
)

import numpy as np

_POSTPROCESSOR_EXPERIMENT = "eigenmodes"
_EIGENMODE_REQUIRED_CHANNELS = ("eigenvalues",)
_EIGENVALUE_KEYS = ("eigenvalues", "eigenvalue", "eigvalues", "eigvalue")
_EIGENVECTOR_KEYS = ("eigenvectors", "eigenvector", "eigvectors", "eigvector")
_OPTIONAL_CHANNELS = ("eigenvectors",)


def _normalize_fixture_mapping(fixture_like: Mapping[str, Any]) -> dict[str, Any]:
    if "channels" in fixture_like:
        candidate = fixture_like["channels"]
    elif "observables" in fixture_like:
        candidate = fixture_like["observables"]
    else:
        candidate = fixture_like

    if not isinstance(candidate, Mapping):
        raise ValueError("No mapping payload found in fixture input.")

    normalized = dict(candidate)

    # Support common nested output shapes for spectral fixtures.
    nested = candidate.get("eigenmode_spectrum")
    if isinstance(nested, Mapping):
        normalized.update({str(name): value for name, value in nested.items() if isinstance(name, str)})
    elif isinstance(nested, tuple | list) and len(nested) >= 1:
        normalized.setdefault("eigenvalues", nested[0])
        if len(nested) >= 2:
            normalized.setdefault("eigenvectors", nested[1])

    flattened: dict[str, Any] = {}
    for name, value in normalized.items():
        if not isinstance(name, str):
            continue
        flattened[str(name)] = value
    return flattened


def _coerce_channel_mapping(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool,
) -> dict[str, np.ndarray]:
    channel_payload = _normalize_fixture_mapping(fixture_like)

    canonical_channels: dict[str, Any] = {}
    lower_lookup = {str(name).lower(): str(name) for name in channel_payload}

    for alias in _EIGENVALUE_KEYS:
        source_key = lower_lookup.get(alias)
        if source_key is not None and "eigenvalues" not in canonical_channels:
            canonical_channels["eigenvalues"] = channel_payload[source_key]
            break

    if include_optional_channels:
        for alias in _EIGENVECTOR_KEYS:
            source_key = lower_lookup.get(alias)
            if source_key is not None and "eigenvectors" not in canonical_channels:
                canonical_channels["eigenvectors"] = channel_payload[source_key]
                break

    missing_required = [name for name in _EIGENMODE_REQUIRED_CHANNELS if name not in canonical_channels]
    if missing_required:
        raise ValueError(
            "Eigenmodes parser requires at least the following channels: "
            + ", ".join(missing_required)
            + "."
        )

    normalized_arrays = validate_numeric_channels(canonical_channels)

    # Optional shape checks for eigenvectors.
    if "eigenvectors" in normalized_arrays:
        eigenvalues = normalized_arrays["eigenvalues"]
        eigenvectors = normalized_arrays["eigenvectors"]
        if eigenvectors.ndim > 2:
            raise ValueError("eigenvectors channel must be 1D or 2D.")
        if eigenvectors.ndim == 1 and eigenvectors.size != eigenvalues.size:
            raise ValueError("1D eigenvectors must match eigenvalue count.")
        if eigenvectors.ndim == 2 and eigenvalues.size != eigenvectors.shape[0]:
            raise ValueError("eigenvectors first axis must match eigenvalue count.")

    return normalized_arrays


def _collect_analysis_provenance(fixture_like: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("analysis_provenance", "analysis", "analysis_metadata"):
        value = fixture_like.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _resolve_quality_flags(
    channels: Mapping[str, np.ndarray],
    *,
    requested_quality_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total_count = 0
    finite_count = 0
    canary_failures: list[str] = []
    for name, values in channels.items():
        finite = np.isfinite(values)
        total_count += int(finite.size)
        finite_count += int(finite.sum())
        if not bool(np.all(finite)):
            missing = int((~finite).sum())
            canary_failures.append(f"{name}:{missing} non-finite")

    finite_ratio = finite_count / total_count if total_count else 0.0
    computed_flags = {
        "finite_observables": finite_ratio == 1.0,
        "finite_observable_ratio": finite_ratio,
        "canary_failures": tuple(canary_failures),
    }

    if requested_quality_flags is None:
        return computed_flags

    merged = dict(requested_quality_flags)
    merged["finite_observable_ratio"] = float(finite_ratio)
    merged["finite_observables"] = bool(computed_flags["finite_observables"])
    merged_failures = list(requested_quality_flags.get("canary_failures", ()))
    merged_failures.extend(canary_failures)
    merged["canary_failures"] = tuple(str(item) for item in merged_failures)
    return merged


def parse_eigenmodes_fixture_channels(
    fixture_like: Mapping[str, Any],
    *,
    include_optional_channels: bool = True,
) -> dict[str, np.ndarray]:
    """Parse fixture-like eigenmode output into numeric channels.

    The parser accepts common fixture layouts where payload is either a top-level
    mapping, keyed under `channels`, keyed under `observables`, or nested under
    `eigenmode_spectrum`.
    """
    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    return _coerce_channel_mapping(
        fixture_like,
        include_optional_channels=include_optional_channels,
    )


def process_eigenmodes_numerical_dataset(
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
    """Build and write a canonical gv:eigenmodes numerical postprocessor artifact."""
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("raw_provenance must be a mapping.")

    channels = parse_eigenmodes_fixture_channels(
        fixture_like,
        include_optional_channels=include_optional_channels,
    )
    resolved_quality_flags = _resolve_quality_flags(channels, requested_quality_flags=quality_flags)

    resolved_provenance = dict(raw_provenance)
    provenance_analysis = _collect_analysis_provenance(fixture_like)
    if provenance_analysis is not None:
        resolved_provenance["analysis"] = provenance_analysis

    manifest = build_gv_numerical_dataset_manifest(
        campaign_id=campaign_id,
        structure="gv",
        experiment=_POSTPROCESSOR_EXPERIMENT,
        geometry_radius=geometry_radius,
        geometry_height=geometry_height,
        material_parameters=material_parameters,
        controls=controls,
        raw_provenance=resolved_provenance,
        quality_flags=resolved_quality_flags,
        units=units,
        normalization=normalization,
    )

    return write_numerical_dataset_artifacts(
        manifest=manifest,
        channels=channels,
    )


__all__ = [
    "_POSTPROCESSOR_EXPERIMENT",
    "_OPTIONAL_CHANNELS",
    "_EIGENMODE_REQUIRED_CHANNELS",
    "parse_eigenmodes_fixture_channels",
    "process_eigenmodes_numerical_dataset",
]
