from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from meso_uq.mirheo.baseline import validate_training_baseline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EMB_RUNTIME_DEFAULTS_PATH = _REPO_ROOT / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"
_EMB_TRAINING_DATA_PATH = _REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "data" / "samples_all.dat"

EMB_34UM_DPD_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_34um_request.v1"
EMB_34UM_DIMENSION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "Yt": (1e5, 1e9),
    "kb": (400.0, 70000.0),
}
EMB_34UM_PARAMETER_NAMES = ("Yt", "kb")
EMB_34UM_RETRY_LIMIT = 3
EMB_34UM_CANARY_FORCE_POINT_COUNT = 3
EMB_34UM_PLATFORM_DEFAULTS: Mapping[str, Any] = {
    "platform": "karolina",
    "walltime": "00:30:00",
    "gpu_count": 1,
}
EMB_34UM_RUNTIME_FINGERPRINT: Mapping[str, Any] = {
    "radp": 6.80,
    "L": 25,
    "fscale": 0.0074,
    "shell_th": 5.0e-9,
    "numsteps": 5000,
    "numsteps_eq": 10000,
}


@dataclass(frozen=True)
class Emb34umRequestArtifacts:
    """Manifest-ready request metadata for one 3.4um EMB force-sweep candidate."""

    candidate_id: str
    experiment: str
    output_root: Path
    campaign_root: Path
    force_grid: tuple[float, ...]
    force_grid_is_canary: bool
    parameters: Mapping[str, float]
    runtime_fingerprint: Mapping[str, Any]
    expected_hdf5_dataset_id: str
    expected_hdf5_path: Path
    expected_request_manifest_path: Path

    def normalized_request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
            "request_type": "emb_34um_full_force_sweep",
            "candidate_id": self.candidate_id,
            "experiment": self.experiment,
            "platform": EMB_34UM_PLATFORM_DEFAULTS["platform"],
            "output_root": str(self.output_root),
            "campaign_root": str(self.campaign_root),
            "retry_limit": EMB_34UM_RETRY_LIMIT,
            "platform_defaults": dict(EMB_34UM_PLATFORM_DEFAULTS),
            "parameters": dict(self.parameters),
            "force_grid": list(self.force_grid),
            "force_grid_count": len(self.force_grid),
            "fingerprint": dict(self.runtime_fingerprint),
            "expected_output_paths": {
                "hdf5": str(self.expected_hdf5_path),
                "request_manifest": str(self.expected_request_manifest_path),
            },
        }
        if self.force_grid_is_canary:
            point_indices = _canonical_canary_indices(len(self.force_grid), EMB_34UM_CANARY_FORCE_POINT_COUNT)
            payload["canary"] = {
                "enabled": True,
                "point_count": EMB_34UM_CANARY_FORCE_POINT_COUNT,
                "point_indices": point_indices,
                "point_values": [self.force_grid[index] for index in point_indices],
            }
        return payload


@dataclass(frozen=True)
class Emb34umForceGrid:
    values: tuple[float, ...]
    canary: bool


class _ValidationError(ValueError):
    pass


def _repo_root() -> Path:
    return _REPO_ROOT


def _coerce_candidate_id(candidate_id: str, *, field_name: str) -> str:
    normalized = str(candidate_id).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_parameter(value: Any, *, name: str, candidate_id: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Candidate {candidate_id!r} has non-numeric {name!r}: {value!r}.") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"Candidate {candidate_id!r} has non-finite {name!r} value {numeric!r}.")
    return float(numeric)


def _validate_emb_34um_parameters(payload: Mapping[str, Any], *, candidate_id: str) -> tuple[float, float]:
    if "Yt" not in payload or "kb" not in payload:
        raise ValueError(
            f"Candidate {candidate_id!r} is missing required EMB 3.4um dimensions; expected 'Yt' and 'kb'."
        )

    yt = _coerce_parameter(payload["Yt"], name="Yt", candidate_id=candidate_id)
    kb = _coerce_parameter(payload["kb"], name="kb", candidate_id=candidate_id)

    yt_min, yt_max = EMB_34UM_DIMENSION_BOUNDS["Yt"]
    kb_min, kb_max = EMB_34UM_DIMENSION_BOUNDS["kb"]
    if not (yt_min <= yt <= yt_max):
        raise ValueError(f"Candidate {candidate_id!r} has Yt={yt} outside bounds [{yt_min}, {yt_max}].")
    if not (kb_min <= kb <= kb_max):
        raise ValueError(f"Candidate {candidate_id!r} has kb={kb} outside bounds [{kb_min}, {kb_max}].")

    return yt, kb


def _coerce_sequence(values: Sequence[float], *, candidate_id: str, field_name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must be a non-empty sequence for candidate {candidate_id!r}.")
    normalized = tuple(_coerce_parameter(value, name=field_name, candidate_id=candidate_id) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} contains duplicate values.")
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} must be non-negative.")
    if normalized != tuple(sorted(normalized)):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} must be sorted ascending and deterministic.")
    return normalized


def _coerce_experiment(payload: Mapping[str, Any], *, candidate_id: str) -> str:
    experiment = payload.get("experiment", "indentation")
    return _coerce_nonempty_text(experiment, field_name=f"candidate {candidate_id} experiment")


def _coerce_canary(payload: Mapping[str, Any], *, candidate_id: str) -> bool:
    canary = payload.get("canary", False)
    if isinstance(canary, str):
        return canary.strip().lower() in {"1", "true", "yes", "on"}
    return bool(canary)


def _load_default_parameters(path: Path | None = None) -> dict[str, Any]:
    source = path or _EMB_RUNTIME_DEFAULTS_PATH
    if not source.exists():
        raise FileNotFoundError(f"Expected EMB parameter default file at {source}")

    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("Invalid EMB parameter default payload; expected a mapping in parameters-default.emb.yaml.")
    return dict(payload)


def _load_emb_34um_force_grid_line(payload: str, *, candidate_id: str) -> tuple[float, ...]:
    tokens = payload.split()
    if not tokens or tokens[0].startswith("#"):
        return tuple()
    try:
        values = [float(value) for value in tokens]
    except ValueError as exc:
        raise _ValidationError(f"Could not parse EMB training row for {candidate_id!r}.") from exc
    remaining = len(values) - 8
    if remaining <= 0 or remaining % 2 != 0:
        raise _ValidationError("Invalid training data row shape for wide EMB table format.")
    m = remaining // 2
    force_grid = values[8 + m : 8 + (2 * m)]
    if len(force_grid) != m:
        raise _ValidationError("Unable to parse force-grid half from wide EMB table row.")
    return _coerce_sequence(force_grid, candidate_id=candidate_id, field_name="force_grid")


def _canonical_canary_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0:
        return tuple()
    if count <= 1:
        return (0,)
    if length <= count:
        return tuple(range(length))
    if count == 2:
        return (0, length - 1)
    if count == 3:
        return (0, (length - 1) // 2, length - 1)
    indices = []
    for i in range(count):
        index = int(round((length - 1) * i / (count - 1)))
        bounded_index = max(0, min(index, length - 1))
        if bounded_index not in indices:
            indices.append(bounded_index)
    i = 0
    while len(indices) < count and i < length:
        if i not in indices:
            indices.append(i)
        i += 1
    return tuple(indices[:count])


def load_emb_34um_force_grid(*, data_path: str | Path | None = None) -> tuple[float, ...]:
    """Load and validate the canonical 3.4um training force grid.

    The source data is the wide-format indentation table containing curve displacements/forces.
    """

    source = Path(data_path) if data_path is not None else _EMB_TRAINING_DATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"Could not locate 3.4um training table at {source!s}")

    force_grid: tuple[float, ...] | None = None
    for line in source.read_text(encoding="utf-8").splitlines():
        parsed = _load_emb_34um_force_grid_line(line.strip(), candidate_id="3.4um training source")
        if not parsed:
            continue
        if force_grid is None:
            force_grid = parsed
            continue
        if parsed != force_grid:
            raise ValueError("Inconsistent force grids detected in 3.4um training table rows; expected a fixed grid.")

    if force_grid is None:
        raise ValueError(f"No valid force-grid row found in {source!s}")
    return force_grid


def derive_emb_34um_force_grid(
    *,
    canary: bool = False,
    data_path: str | Path | None = None,
) -> tuple[float, ...]:
    grid = load_emb_34um_force_grid(data_path=data_path)
    if not canary:
        return grid
    indices = _canonical_canary_indices(len(grid), EMB_34UM_CANARY_FORCE_POINT_COUNT)
    return tuple(float(grid[index]) for index in indices)


def is_emb_34um_full_request_payload(payload: Mapping[str, Any]) -> bool:
    return all(key in payload for key in EMB_34UM_PARAMETER_NAMES)


def _resolve_runtime_fingerprint(
    *,
    candidate_id: str,
    defaults_path: Path | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _load_default_parameters(path=defaults_path)
    base: dict[str, float | int | str] = {
        "fscale": defaults.get("fscale", EMB_34UM_RUNTIME_FINGERPRINT["fscale"]),
        "shell_th": defaults.get("shell_th", EMB_34UM_RUNTIME_FINGERPRINT["shell_th"]),
        "numsteps": defaults.get("numsteps", EMB_34UM_RUNTIME_FINGERPRINT["numsteps"]),
        "numsteps_eq": defaults.get("numsteps_eq", EMB_34UM_RUNTIME_FINGERPRINT["numsteps_eq"]),
        "radp": EMB_34UM_RUNTIME_FINGERPRINT["radp"],
        "L": EMB_34UM_RUNTIME_FINGERPRINT["L"],
    }

    runtime_overrides = payload.get("runtime_fingerprint", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(runtime_overrides, Mapping):
        raise ValueError(f"Candidate {candidate_id!r} runtime_fingerprint must be a mapping.")

    overrides = {}
    for key in EMB_34UM_RUNTIME_FINGERPRINT:
        if key in runtime_overrides:
            overrides[key] = runtime_overrides[key]
        elif isinstance(payload, Mapping) and key in payload:
            overrides[key] = payload[key]
    for key in EMB_34UM_RUNTIME_FINGERPRINT:
        if key in overrides:
            base[key] = overrides[key]

    runtime_payload = {
        "radp": float(base["radp"]),
        "L": float(base["L"]),
        "fscale": float(base["fscale"]),
        "shell_th": float(base["shell_th"]),
        "numsteps": int(float(base["numsteps"])),
        "numsteps_eq": int(float(base["numsteps_eq"])),
    }

    validate_training_baseline(
        {
            "fscale": runtime_payload["fscale"],
            "shell_th": runtime_payload["shell_th"],
            "numsteps": runtime_payload["numsteps"],
            "numsteps_eq": runtime_payload["numsteps_eq"],
        },
        source_label="emb_34um_runtime",
    )

    if runtime_payload["radp"] != EMB_34UM_RUNTIME_FINGERPRINT["radp"]:
        raise ValueError("Expected radp override mismatch in EMB fingerprint metadata.")
    if runtime_payload["L"] != EMB_34UM_RUNTIME_FINGERPRINT["L"]:
        raise ValueError("Expected L override mismatch in EMB fingerprint metadata.")

    return runtime_payload


def _build_parameter_payload(
    *,
    yt: float,
    kb: float,
    defaults: Mapping[str, Any],
) -> dict[str, float]:
    payload = {
        "Yt": yt,
        "kb": kb,
        "b1": float(defaults.get("b1", 0.0)),
        "b2": float(defaults.get("b2", 0.0)),
        "a3": float(defaults.get("a3", 0.0)),
        "a4": float(defaults.get("a4", 0.0)),
    }
    return payload


def build_emb_34um_request(
    candidate_id: str,
    payload: Mapping[str, Any],
    campaign_root: str | Path,
    *,
    data_path: str | Path | None = None,
    force_grid: Sequence[float] | None = None,
) -> Emb34umRequestArtifacts:
    candidate_id = _coerce_candidate_id(candidate_id, field_name="candidate_id")
    base_root = Path(campaign_root)
    experiment = _coerce_experiment(payload, candidate_id=candidate_id)
    yt, kb = _validate_emb_34um_parameters(payload, candidate_id=candidate_id)

    defaults = _load_default_parameters()
    payload_force_grid = payload.get("force_grid")
    if force_grid is not None:
        requested_grid = _coerce_sequence(force_grid, candidate_id=candidate_id, field_name="force_grid")
    elif payload_force_grid is not None:
        requested_grid = _coerce_sequence(payload_force_grid, candidate_id=candidate_id, field_name="force_grid")
    else:
        requested_grid = derive_emb_34um_force_grid(
            canary=_coerce_canary(payload, candidate_id=candidate_id),
            data_path=data_path,
        )
    runtime_fingerprint = _resolve_runtime_fingerprint(
        candidate_id=candidate_id,
        payload=payload,
    )

    output_root = base_root / "emb" / candidate_id
    dataset_id = f"emb_34um_{candidate_id}_{experiment}"
    expected_hdf5_path = output_root / f"{dataset_id}.h5"
    expected_request_manifest_path = output_root / "emb_34um_request_manifest.json"

    return Emb34umRequestArtifacts(
        candidate_id=candidate_id,
        experiment=experiment,
        output_root=output_root,
        campaign_root=base_root,
        force_grid=tuple(requested_grid),
        force_grid_is_canary=_coerce_canary(payload, candidate_id=candidate_id),
        parameters=_build_parameter_payload(yt=yt, kb=kb, defaults=defaults),
        runtime_fingerprint=runtime_fingerprint,
        expected_hdf5_dataset_id=dataset_id,
        expected_hdf5_path=expected_hdf5_path,
        expected_request_manifest_path=expected_request_manifest_path,
    )


__all__ = [
    "EMB_34UM_CANARY_FORCE_POINT_COUNT",
    "EMB_34UM_DPD_SCHEMA_VERSION",
    "EMB_34UM_DIMENSION_BOUNDS",
    "EMB_34UM_PARAMETER_NAMES",
    "EMB_34UM_PLATFORM_DEFAULTS",
    "EMB_34UM_RETRY_LIMIT",
    "EMB_34UM_RUNTIME_FINGERPRINT",
    "Emb34umForceGrid",
    "Emb34umRequestArtifacts",
    "build_emb_34um_request",
    "derive_emb_34um_force_grid",
    "is_emb_34um_full_request_payload",
    "load_emb_34um_force_grid",
]
