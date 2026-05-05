from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

_DEFAULT_EIGENMODE_COUNT = 30
_EIGENMODES_AXIS = "mode_index"
_VALUE_KEYS = ("eigenvalues", "eigenvalue", "eigvalues", "eigvalue")
_FREQUENCY_KEYS = (
    "eigenfrequencies",
    "eigenfrequency",
    "frequencies",
    "frequency",
    "eigfreq",
    "eigfreqs",
)
_VECTOR_KEYS = ("eigenvectors", "eigenvector", "eigvectors", "eigvector")
_CONTROL_ALIASES = {
    "pressure": "bpress",
    "background_pressure": "bpress",
    "b_pressure": "bpress",
}


def _canonical_name(raw_name: object) -> str:
    normalized = re.sub(r"[^\w]+", "_", str(raw_name).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _coerce_numeric_array(payload: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Eigenmodes channel {name!r} must be numeric data.") from exc
    if array.size == 0:
        raise ValueError(f"Eigenmodes channel {name!r} must contain data.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Eigenmodes channel {name!r} contains non-finite values.")
    return array


def _normalize_fixture_mapping(fixture_like: Mapping[str, Any]) -> dict[str, Any]:
    if "channels" in fixture_like:
        payload = fixture_like["channels"]
    elif "observables" in fixture_like:
        payload = fixture_like["observables"]
    else:
        payload = fixture_like
    if not isinstance(payload, Mapping):
        raise ValueError("No mapping payload found in eigenmodes fixture.")

    normalized = dict(payload)
    nested = payload.get("eigenmode_spectrum")
    if isinstance(nested, Mapping):
        normalized.update({str(name): value for name, value in nested.items() if isinstance(name, str)})
    elif isinstance(nested, tuple | list) and len(nested) >= 1:
        normalized.setdefault("eigenvalues", nested[0])
        if len(nested) >= 2:
            normalized.setdefault("eigenvectors", nested[1])
        if len(nested) >= 3:
            normalized.setdefault("eigenfrequencies", nested[2])

    return {str(name): value for name, value in normalized.items() if isinstance(name, str)}


def _select_spectrum(
    payload: Mapping[str, Any],
    aliases: tuple[str, ...],
    *,
    canonical_name: str,
) -> np.ndarray | None:
    lowered = {_canonical_name(name): value for name, value in payload.items()}
    for alias in aliases:
        match = lowered.get(alias)
        if match is not None:
            array = _coerce_numeric_array(match, name=canonical_name)
            if array.ndim != 1:
                raise ValueError(f"Eigenmodes channel {canonical_name!r} must be 1D.")
            return array
    return None


def _select_eigenvectors(payload: Mapping[str, Any]) -> np.ndarray | None:
    lowered = {_canonical_name(name): value for name, value in payload.items()}
    for alias in _VECTOR_KEYS:
        match = lowered.get(alias)
        if match is not None:
            array = _coerce_numeric_array(match, name="eigenvectors")
            if array.ndim not in (1, 2):
                raise ValueError("Eigenmodes channel 'eigenvectors' must be 1D or 2D.")
            return array
    return None


def _slice_count(
    requested_mode_count: int | None,
    *,
    available_count: int,
) -> int:
    if requested_mode_count is None:
        mode_count = _DEFAULT_EIGENMODE_COUNT
    else:
        mode_count = int(requested_mode_count)
    if mode_count <= 0:
        raise ValueError("Eigenmodes mode_count must be a positive integer.")
    return min(mode_count, available_count)


def parse_eigenmodes_lane_channels(
    fixture_like: Mapping[str, Any],
    *,
    mode_count: int = _DEFAULT_EIGENMODE_COUNT,
    include_eigenvectors: bool = True,
) -> dict[str, np.ndarray]:
    """Extract the first `mode_count` eigenmodes from fixture-like payloads."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    payload = _normalize_fixture_mapping(fixture_like)
    eigenvalues = _select_spectrum(payload, _VALUE_KEYS, canonical_name="eigenvalues")
    eigenfrequencies = _select_spectrum(
        payload,
        _FREQUENCY_KEYS,
        canonical_name="eigenfrequencies",
    )
    if eigenvalues is None and eigenfrequencies is None:
        raise ValueError("Eigenmodes lane requires eigenvalues or eigenfrequencies.")

    available_count = (
        eigenvalues.size
        if eigenvalues is not None
        else int(eigenfrequencies.size)
    )
    if eigenvalues is not None and eigenfrequencies is not None and eigenvalues.size != eigenfrequencies.size:
        raise ValueError("Eigenmodes eigenvalues and eigenfrequencies must have matching lengths.")

    selected_count = _slice_count(mode_count, available_count=available_count)
    channels: dict[str, np.ndarray] = {
        "mode_index": np.arange(selected_count, dtype=int),
    }
    if eigenvalues is not None:
        channels["eigenvalues"] = eigenvalues[:selected_count]
    if eigenfrequencies is not None:
        channels["eigenfrequencies"] = eigenfrequencies[:selected_count]

    if include_eigenvectors:
        eigenvectors = _select_eigenvectors(payload)
        if eigenvectors is not None:
            if eigenvectors.shape[0] < selected_count:
                raise ValueError("Eigenmodes eigenvectors must provide at least as many modes as selected.")
            if eigenvectors.ndim == 1:
                channels["eigenvectors"] = eigenvectors[:selected_count]
            else:
                channels["eigenvectors"] = eigenvectors[:selected_count, ...]

    return channels


def parse_eigenmodes_lane_controls(
    fixture_like: Mapping[str, Any],
    *,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract optional eigenmodes controls from fixture-like input."""

    if not isinstance(fixture_like, Mapping):
        raise ValueError("fixture_like must be a mapping.")

    if controls is None:
        payload = fixture_like.get("controls", {})
        if payload and not isinstance(payload, Mapping):
            raise ValueError("Eigenmodes fixture controls must be a mapping.")
    else:
        payload = controls

    normalized: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for raw_name, value in payload.items():
            if not isinstance(raw_name, str):
                continue
            name = _CONTROL_ALIASES.get(_canonical_name(raw_name), _canonical_name(raw_name))
            if name != "bpress":
                continue
            array = _coerce_numeric_array(value, name=name)
            if array.shape == ():
                normalized[name] = float(array.item())
            elif array.size == 1:
                normalized[name] = float(array.reshape(-1)[0])
            else:
                raise ValueError("Eigenmodes control 'bpress' must be a scalar value.")
    return normalized


def extract_eigenmodes_lane(
    fixture_like: Mapping[str, Any],
    *,
    mode_count: int = _DEFAULT_EIGENMODE_COUNT,
    include_eigenvectors: bool = True,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a canonical GV eigenmodes lane from fixture-like input."""

    return {
        "axis": _EIGENMODES_AXIS,
        "controls": parse_eigenmodes_lane_controls(fixture_like, controls=controls),
        "channels": parse_eigenmodes_lane_channels(
            fixture_like,
            mode_count=mode_count,
            include_eigenvectors=include_eigenvectors,
        ),
    }


def parse_eigenmodes_sampling_lane(
    fixture_like: Mapping[str, Any],
    *,
    mode_count: int = _DEFAULT_EIGENMODE_COUNT,
    include_eigenvectors: bool = True,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for eigenmodes lane parsing."""

    return extract_eigenmodes_lane(
        fixture_like,
        mode_count=mode_count,
        include_eigenvectors=include_eigenvectors,
        controls=controls,
    )


__all__ = [
    "_DEFAULT_EIGENMODE_COUNT",
    "_EIGENMODES_AXIS",
    "extract_eigenmodes_lane",
    "parse_eigenmodes_lane_channels",
    "parse_eigenmodes_lane_controls",
    "parse_eigenmodes_sampling_lane",
]
