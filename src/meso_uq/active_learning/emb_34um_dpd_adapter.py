from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
)
from meso_uq.mirheo.baseline import validate_training_baseline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EMB_RUNTIME_DEFAULTS_PATH = _REPO_ROOT / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"
_EMB_TRAINING_DATA_PATH = _REPO_ROOT / "emb" / "indentation" / "surrogate" / "diameters" / "3.4um" / "data" / "samples_all.dat"

EMB_34UM_DPD_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_34um_request.v1"
EMB_34UM_DIMENSION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "ka": (1e2, 6e5),
    "kb": (400.0, 70000.0),
    "radp": (1.0, 100.0),
    "shell_th": (1.0e-12, 1.0e-6),
}
EMB_34UM_PARAMETER_NAMES = ("ka", "kb")
EMB_34UM_RETRY_LIMIT = EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT
EMB_34UM_CANARY_FORCE_POINT_COUNT = 3
EMB_34UM_FORCE_GRID = tuple(float(value) for value in EMB_34UM_DNN_CAUSAL_FORCE_GRID)
EMB_34UM_PLATFORM_DEFAULTS: Mapping[str, Any] = {
    "platform": "karolina",
    "walltime": "00:30:00",
    "gpu_count": 1,
}
_DEFAULT_RUNTIME_RADP = 6.80
_DEFAULT_RUNTIME_SHELL_TH = 5.0e-9
_expected_fscale = 0.0074
_expected_numsteps = 5000
_expected_numsteps_eq = 10000


def _coerce_float(value: Any, *, name: str, candidate_id: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Candidate {candidate_id!r} has non-numeric {name!r}: {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Candidate {candidate_id!r} has non-finite {name!r} value {numeric!r}.")
    return float(numeric)


def _coerce_bool(value: Any, *, name: str, candidate_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Candidate {candidate_id!r} has non-boolean {name!r}: {value!r}.")


def _coerce_bounded_float(
    value: Any,
    *,
    name: str,
    candidate_id: str,
    low: float,
    high: float,
) -> float:
    numeric = _coerce_float(value=value, name=name, candidate_id=candidate_id)
    if not (low <= numeric <= high):
        raise ValueError(
            f"Candidate {candidate_id!r} has {name}={numeric} outside bounds [{low}, {high}]."
        )
    return numeric


def _compute_box_dimensions(radp: float, *, explicit_cubic: bool = False) -> tuple[float, float, float]:
    if explicit_cubic:
        box = float(math.ceil(2.0 * radp + 10.0))
        return (box, box, box)
    lx = float(math.ceil(2.0 * radp + 6.0))
    return (lx, lx, float(math.ceil(2.0 * radp + 10.0)))


def _coerce_force_grid(values: Sequence[float], *, candidate_id: str) -> tuple[float, ...]:
    normalized = tuple(_coerce_float(value=value, name="force_grid item", candidate_id=candidate_id) for value in values)
    if not normalized:
        raise ValueError(f"Candidate {candidate_id!r} force_grid must be non-empty.")
    if normalized != tuple(sorted(normalized)):
        raise ValueError(f"force_grid for candidate {candidate_id!r} must be sorted ascending and deterministic.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"force_grid for candidate {candidate_id!r} must not contain duplicate values.")
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"force_grid for candidate {candidate_id!r} must be non-negative.")
    return normalized


EMB_34UM_RUNTIME_FINGERPRINT: Mapping[str, Any] = {
    "radp": _DEFAULT_RUNTIME_RADP,
    "shell_th": _DEFAULT_RUNTIME_SHELL_TH,
    "fscale": _expected_fscale,
    "numsteps": _expected_numsteps,
    "numsteps_eq": _expected_numsteps_eq,
    "Lx": _compute_box_dimensions(_DEFAULT_RUNTIME_RADP)[0],
    "Ly": _compute_box_dimensions(_DEFAULT_RUNTIME_RADP)[1],
    "Lz": _compute_box_dimensions(_DEFAULT_RUNTIME_RADP)[2],
    "L": _compute_box_dimensions(_DEFAULT_RUNTIME_RADP)[2],
    "direct_stiffness_override": True,
    "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
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


def _legacy_yt_to_ka(yt: float, *, defaults: Mapping[str, Any], candidate_id: str) -> float:
    # Compatibility-only path: old campaign controls emit Yt for legacy active-learning workflows.
    required = ("ul", "kbol", "t0", "shell_th", "nu", "fscale")
    for key in required:
        if key not in defaults:
            raise ValueError(f"Candidate {candidate_id!r} defaults are missing {key!r} for Yt-to-ka compatibility conversion.")
    try:
        ul = float(defaults["ul"])
        kbol = float(defaults["kbol"])
        t0 = float(defaults["t0"])
        shell_th = float(defaults["shell_th"])
        nu = float(defaults["nu"])
        fscale = float(defaults["fscale"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Candidate {candidate_id!r} has invalid default constants for compatibility conversion.") from exc

    if any(math.isclose(value, 0.0) or not math.isfinite(value) for value in (ul, kbol, t0, shell_th, fscale, nu, 1.0 - nu)):
        raise ValueError(f"Candidate {candidate_id!r} has invalid default constants for compatibility conversion.")
    if nu == 1.0:
        raise ValueError(f"Candidate {candidate_id!r} has invalid Poisson ratio (nu == 1.0) for compatibility conversion.")

    ue = kbol * t0
    ka_per_yt = fscale * shell_th / (2.0 * (1.0 - nu)) / (ue / ul**2)
    return yt * ka_per_yt


def _validate_emb_34um_parameters(
    payload: Mapping[str, Any], *, candidate_id: str, defaults: Mapping[str, Any]
) -> tuple[float, float]:
    has_legacy_yt = "Yt" in payload and "ka" not in payload
    if not ("kb" in payload and ("ka" in payload or has_legacy_yt)):
        raise ValueError(
            f"Candidate {candidate_id!r} is missing required EMB 3.4um dimensions; expected 'ka' and 'kb'."
        )

    if "ka" in payload:
        ka = _coerce_parameter(payload["ka"], name="ka", candidate_id=candidate_id)
    else:
        yt = _coerce_parameter(payload["Yt"], name="Yt", candidate_id=candidate_id)
        ka = _legacy_yt_to_ka(yt, defaults=defaults, candidate_id=candidate_id)
    kb = _coerce_parameter(payload["kb"], name="kb", candidate_id=candidate_id)

    ka_min, ka_max = EMB_34UM_DIMENSION_BOUNDS["ka"]
    kb_min, kb_max = EMB_34UM_DIMENSION_BOUNDS["kb"]
    if not (ka_min <= ka <= ka_max):
        raise ValueError(f"Candidate {candidate_id!r} has ka={ka} outside bounds [{ka_min}, {ka_max}].")
    if not (kb_min <= kb <= kb_max):
        raise ValueError(f"Candidate {candidate_id!r} has kb={kb} outside bounds [{kb_min}, {kb_max}].")

    return ka, kb


def _coerce_sequence(values: Sequence[float], *, candidate_id: str, field_name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must be a non-empty sequence for candidate {candidate_id!r}.")
    normalized = tuple(_coerce_float(value=value, name=field_name, candidate_id=candidate_id) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} contains duplicate values.")
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} must be non-negative.")
    if normalized != tuple(sorted(normalized)):
        raise ValueError(f"{field_name} for candidate {candidate_id!r} must be sorted ascending and deterministic.")
    return normalized


def _has_explicit_d4_geometry(payload: Mapping[str, Any]) -> bool:
    if {"radp", "shell_th"} <= payload.keys():
        return True
    runtime_fingerprint = payload.get("runtime_fingerprint")
    if isinstance(runtime_fingerprint, Mapping) and {"radp", "shell_th"} <= runtime_fingerprint.keys():
        return True
    return False


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
        if data_path is None:
            return EMB_34UM_FORCE_GRID
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
    grid = EMB_34UM_FORCE_GRID if data_path is None else load_emb_34um_force_grid(data_path=data_path)
    if not canary:
        return grid
    indices = _canonical_canary_indices(len(grid), EMB_34UM_CANARY_FORCE_POINT_COUNT)
    return tuple(float(grid[index]) for index in indices)


def is_emb_34um_full_request_payload(payload: Mapping[str, Any]) -> bool:
    if "kb" not in payload:
        return False
    if "Yt" in payload and "ka" not in payload:
        return True
    return "ka" in payload and _has_explicit_d4_geometry(payload)


def _is_legacy_d2_payload(payload: Mapping[str, Any], *, runtime_source: Mapping[str, Any]) -> bool:
    return "Yt" in payload and "ka" not in payload


def _resolve_runtime_fingerprint(
    *,
    candidate_id: str,
    defaults_path: Path | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _load_default_parameters(path=defaults_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Candidate {candidate_id!r} payload must be a mapping.")

    runtime_source = dict(payload.get("runtime_fingerprint", {}) if payload.get("runtime_fingerprint", {}) is not None else {})
    if not isinstance(runtime_source, Mapping):
        raise ValueError(f"Candidate {candidate_id!r} runtime_fingerprint must be a mapping.")
    for key in (
        "radp",
        "shell_th",
        "fscale",
        "numsteps",
        "numsteps_eq",
        "L",
        "Lx",
        "Ly",
        "Lz",
        "direct_stiffness_override",
        "bpress",
    ):
        if key in payload and key not in runtime_source:
            runtime_source[key] = payload[key]

    legacy_d2 = _is_legacy_d2_payload(payload, runtime_source=runtime_source)
    if not legacy_d2:
        missing = [key for key in ("radp", "shell_th") if key not in runtime_source]
        if missing:
            raise ValueError(
                f"Candidate {candidate_id!r} is missing D4 runtime parameters {missing}; provide both 'radp' and 'shell_th' "
                "unless using explicit legacy Yt/kb input."
            )

    radp = _coerce_bounded_float(
        runtime_source.get("radp", _DEFAULT_RUNTIME_RADP),
        name="radp",
        candidate_id=candidate_id,
        low=EMB_34UM_DIMENSION_BOUNDS["radp"][0],
        high=EMB_34UM_DIMENSION_BOUNDS["radp"][1],
    )
    shell_th = _coerce_bounded_float(
        runtime_source.get("shell_th", _DEFAULT_RUNTIME_SHELL_TH),
        name="shell_th",
        candidate_id=candidate_id,
        low=EMB_34UM_DIMENSION_BOUNDS["shell_th"][0],
        high=EMB_34UM_DIMENSION_BOUNDS["shell_th"][1],
    )
    fscale = _coerce_float(
        value=runtime_source.get("fscale", defaults.get("fscale", EMB_34UM_RUNTIME_FINGERPRINT["fscale"])),
        name="fscale",
        candidate_id=candidate_id,
    )
    numsteps = int(
        _coerce_float(
            value=runtime_source.get("numsteps", defaults.get("numsteps", EMB_34UM_RUNTIME_FINGERPRINT["numsteps"])),
            name="numsteps",
            candidate_id=candidate_id,
        )
    )
    numsteps_eq = int(
        _coerce_float(
            value=runtime_source.get(
                "numsteps_eq", defaults.get("numsteps_eq", EMB_34UM_RUNTIME_FINGERPRINT["numsteps_eq"])
            ),
            name="numsteps_eq",
            candidate_id=candidate_id,
        )
    )
    direct_stiffness_override = _coerce_bool(
        value=runtime_source.get(
            "direct_stiffness_override", EMB_34UM_RUNTIME_FINGERPRINT["direct_stiffness_override"]
        ),
        name="direct_stiffness_override",
        candidate_id=candidate_id,
    )
    bpress = _coerce_float(
        value=runtime_source.get("bpress", EMB_34UM_RUNTIME_FINGERPRINT["bpress"]),
        name="bpress",
        candidate_id=candidate_id,
    )

    if not math.isclose(fscale, EMB_34UM_RUNTIME_FINGERPRINT["fscale"]):
        raise ValueError(
            f"Candidate {candidate_id!r} runtime_fingerprint.fscale mismatch; "
            f"expected {EMB_34UM_RUNTIME_FINGERPRINT['fscale']}."
        )
    if numsteps != EMB_34UM_RUNTIME_FINGERPRINT["numsteps"]:
        raise ValueError(
            f"Candidate {candidate_id!r} runtime_fingerprint.numsteps mismatch; "
            f"expected {EMB_34UM_RUNTIME_FINGERPRINT['numsteps']}."
        )
    if numsteps_eq != EMB_34UM_RUNTIME_FINGERPRINT["numsteps_eq"]:
        raise ValueError(
            f"Candidate {candidate_id!r} runtime_fingerprint.numsteps_eq mismatch; "
            f"expected {EMB_34UM_RUNTIME_FINGERPRINT['numsteps_eq']}."
        )
    if direct_stiffness_override != EMB_34UM_RUNTIME_FINGERPRINT["direct_stiffness_override"]:
        raise ValueError("runtime_fingerprint.direct_stiffness_override must be true.")
    if not math.isclose(bpress, EMB_34UM_RUNTIME_FINGERPRINT["bpress"]):
        raise ValueError(
            f"Candidate {candidate_id!r} runtime_fingerprint.bpress mismatch; "
            f"expected {EMB_34UM_RUNTIME_FINGERPRINT['bpress']}."
        )

    explicit_cubic = "L" in runtime_source
    if explicit_cubic:
        _coerce_float(value=runtime_source["L"], name="L", candidate_id=candidate_id)
        Lx, Ly, Lz = _compute_box_dimensions(radp, explicit_cubic=True)
    else:
        Lx, Ly, Lz = _compute_box_dimensions(radp, explicit_cubic=False)

    runtime_payload = {
        "radp": float(radp),
        "shell_th": float(shell_th),
        "fscale": float(fscale),
        "numsteps": numsteps,
        "numsteps_eq": numsteps_eq,
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "direct_stiffness_override": True,
        "bpress": float(bpress),
    }

    # Preserve the legacy cubic-extent key as a compatibility artifact while
    # moving runtime payloads to explicit box dimensions.
    runtime_payload["L"] = float(Lz)

    validate_training_baseline(
        {
            "fscale": runtime_payload["fscale"],
            "shell_th": defaults.get("shell_th", EMB_34UM_RUNTIME_FINGERPRINT["shell_th"]),
            "numsteps": runtime_payload["numsteps"],
            "numsteps_eq": runtime_payload["numsteps_eq"],
        },
        source_label="emb_34um_runtime",
    )
    return runtime_payload


def _build_parameter_payload(
    *,
    ka: float,
    kb: float,
    radp: float,
    shell_th: float,
    bpress: float,
) -> dict[str, float]:
    payload = {
        "ka": ka,
        "kb": kb,
        "radp": radp,
        "shell_th": shell_th,
        "b1": 0.0,
        "b2": 0.0,
        "a3": 0.0,
        "a4": 0.0,
        "bpress": bpress,
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
    defaults = _load_default_parameters()
    ka, kb = _validate_emb_34um_parameters(payload, candidate_id=candidate_id, defaults=defaults)

    payload_force_grid = payload.get("force_grid")
    if force_grid is not None:
        requested_grid = _coerce_force_grid(force_grid, candidate_id=candidate_id)
    elif payload_force_grid is not None:
        requested_grid = _coerce_force_grid(payload_force_grid, candidate_id=candidate_id)
    else:
        requested_grid = derive_emb_34um_force_grid(canary=False, data_path=data_path)
        if _coerce_canary(payload, candidate_id=candidate_id):
            requested_grid = derive_emb_34um_force_grid(canary=True, data_path=data_path)
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
        parameters=_build_parameter_payload(
            ka=ka,
            kb=kb,
            radp=float(runtime_fingerprint["radp"]),
            shell_th=float(runtime_fingerprint["shell_th"]),
            bpress=float(runtime_fingerprint["bpress"]),
        ),
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
