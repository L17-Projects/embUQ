#!/usr/bin/env python3
"""Run one EMB 3.4um final-gate active-learning candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMB_34UM_REQUEST_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_34um_request.v1"
EMB_34UM_DIAMETER_UM = 3.4
DEFAULT_DT_SCALE_FACTOR = 0.5
EMB_34UM_FORCE_GRID = tuple((5000.0 * index / 7.0) for index in range(8))
TRANSIENT_OUTPUT_NAMES = (
    "out_hierarchical",
    "F_Delta.dat",
    "emb_34um_result.json",
    "emb_34um_runtime_status.json",
)


def _compute_box_dimensions(radp: float, *, explicit_cubic: bool = False) -> tuple[float, float, float]:
    if explicit_cubic:
        box = float(math.ceil(2.0 * radp + 10.0))
        return (box, box, box)
    lx = float(math.ceil(2.0 * radp + 6.0))
    lz = float(math.ceil(2.0 * radp + 10.0))
    return (lx, lx, lz)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


def _configure_python_path(repo_root: Path) -> None:
    entries = [
        repo_root,
        repo_root / "src",
        repo_root / "emb" / "indentation",
        repo_root / "emb" / "indentation" / "evalkit",
    ]
    for entry in reversed(entries):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


@dataclass(frozen=True)
class Emb34umCandidateRequest:
    candidate_manifest_path: Path
    candidate_id: str
    output_root: Path
    expected_hdf5_path: Path
    expected_request_manifest_path: Path
    parameters: Mapping[str, float]
    force_grid: tuple[float, ...]
    retry_limit: int
    fingerprint: Mapping[str, Any]
    request_payload: Mapping[str, Any]
    compatibility_yt: float | None = None

    @property
    def parameter_vector(self) -> tuple[float, ...]:
        return (
            float(self.parameters["ka"]),
            float(self.parameters["kb"]),
            float(self.parameters.get("b1", 0.0)),
            float(self.parameters.get("b2", 0.0)),
            float(self.parameters.get("a3", 0.0)),
            float(self.parameters.get("a4", 0.0)),
        )

    @property
    def compute_parameter_vector(self) -> tuple[float, ...]:
        yt = self.compatibility_yt
        if yt is None:
            defaults = _load_emb_default_parameters()
            defaults.update(self.fingerprint)
            runtime_params = {
                "fscale": defaults.get("fscale", _DEFAULT_EMB_FINGERPRINT["fscale"]),
                "shell_th": defaults.get("shell_th", _DEFAULT_EMB_FINGERPRINT["shell_th"]),
                "ul": defaults.get("ul"),
                "kbol": defaults.get("kbol"),
                "t0": defaults.get("t0"),
                "nu": defaults.get("nu"),
            }
            yt = _ka_to_yt(float(self.parameters["ka"]), defaults=runtime_params)
        return (
            float(yt),
            float(self.parameters["kb"]),
            float(self.parameters.get("b1", 0.0)),
            float(self.parameters.get("b2", 0.0)),
            float(self.parameters.get("a3", 0.0)),
            float(self.parameters.get("a4", 0.0)),
            0.0,
            0.0,
        )


_EMB_DEFAULTS_PATH = (
    _repo_root()
    / "emb"
    / "indentation"
    / "src"
    / "parameters-default.emb.yaml"
)
_DEFAULT_EMB_FINGERPRINT = {
    "fscale": 0.0074,
    "radp": 6.60,
    "shell_th": 3.75e-9,
}


def _load_emb_default_parameters() -> dict[str, Any]:
    import yaml

    if not _EMB_DEFAULTS_PATH.exists():
        raise FileNotFoundError(f"Missing EMB defaults at {_EMB_DEFAULTS_PATH}")
    with _EMB_DEFAULTS_PATH.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("Invalid EMB defaults payload; expected mapping.")
    return dict(payload)


def _coerce_ka_yt_scale(*, defaults: Mapping[str, Any], candidate_id: str) -> float:
    required = ("ul", "kbol", "t0", "shell_th", "nu", "fscale")
    for key in required:
        if key not in defaults:
            raise ValueError(f"Candidate {candidate_id!r} is missing {key!r} for ka<->Yt conversion.")

    ul = float(defaults["ul"])
    kbol = float(defaults["kbol"])
    t0 = float(defaults["t0"])
    shell_th = float(defaults["shell_th"])
    nu = float(defaults["nu"])
    fscale = float(defaults["fscale"])
    if not all(math.isfinite(value) for value in (ul, kbol, t0, shell_th, nu, fscale)):
        raise ValueError(f"Candidate {candidate_id!r} has invalid material constants for ka<->Yt conversion.")
    if math.isclose(ul, 0.0) or math.isclose(kbol, 0.0) or math.isclose(t0, 0.0) or math.isclose(shell_th, 0.0):
        raise ValueError(f"Candidate {candidate_id!r} has invalid material constants for ka<->Yt conversion.")
    if nu == 1.0:
        raise ValueError(f"Candidate {candidate_id!r} has invalid Poisson ratio for ka<->Yt conversion.")

    return fscale * shell_th / (2.0 * (1.0 - nu)) / ((kbol * t0) / ul**2)


def _legacy_yt_to_ka(yt: float, *, defaults: Mapping[str, Any], candidate_id: str) -> float:
    return float(yt) * _coerce_ka_yt_scale(defaults=defaults, candidate_id=candidate_id)


def _ka_to_yt(ka: float, *, defaults: Mapping[str, Any], candidate_id: str | None = None) -> float:
    scale = _coerce_ka_yt_scale(defaults=defaults, candidate_id=candidate_id or "candidate")
    return float(ka) / scale


def _coerce_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite; got {numeric!r}.")
    return numeric


def _coerce_force_grid(values: Any, *, candidate_id: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Candidate {candidate_id!r} force_grid must be a sequence.")
    grid = tuple(_coerce_float(value, field_name="force_grid item") for value in values)
    if not grid:
        raise ValueError(f"Candidate {candidate_id!r} force_grid must not be empty.")
    if len(grid) != len(EMB_34UM_FORCE_GRID):
        raise ValueError(
            f"Candidate {candidate_id!r} force_grid must contain {len(EMB_34UM_FORCE_GRID)} points "
            f"from 0 to 5000, got {len(grid)}."
        )
    if grid != tuple(sorted(grid)):
        raise ValueError(f"Candidate {candidate_id!r} force_grid must be sorted ascending and deterministic.")
    for actual, expected in zip(grid, EMB_34UM_FORCE_GRID):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                f"Candidate {candidate_id!r} force_grid must be exactly eight evenly spaced points from 0 to 5000."
            )
    if any(value < 0.0 for value in grid):
        raise ValueError(f"Candidate {candidate_id!r} force_grid must be non-negative.")
    return grid


def _request_runtime_fingerprint(request: Emb34umCandidateRequest) -> dict[str, Any]:
    runtime_fingerprint = dict(request.fingerprint)
    request_parameters = request.request_payload.get("parameters")
    if isinstance(request_parameters, Mapping):
        for key in ("radp", "shell_th", "bpress", "L", "Lx", "Ly", "Lz"):
            if key not in runtime_fingerprint and key in request_parameters:
                runtime_fingerprint[key] = request_parameters[key]

    if "radp" not in runtime_fingerprint and "radp" in request.parameters:
        runtime_fingerprint["radp"] = request.parameters["radp"]
    if "shell_th" not in runtime_fingerprint and "shell_th" in request.parameters:
        runtime_fingerprint["shell_th"] = request.parameters["shell_th"]
    if "bpress" not in runtime_fingerprint and "bpress" in request.parameters:
        runtime_fingerprint["bpress"] = request.parameters["bpress"]

    runtime_fingerprint.setdefault("fscale", _DEFAULT_EMB_FINGERPRINT["fscale"])
    runtime_fingerprint.setdefault("shell_th", _DEFAULT_EMB_FINGERPRINT["shell_th"])
    runtime_fingerprint.setdefault("numsteps", 5000)
    runtime_fingerprint.setdefault("numsteps_eq", 10000)
    runtime_fingerprint.setdefault("direct_stiffness_override", True)
    runtime_fingerprint.setdefault("bpress", -91.0)

    if "radp" in runtime_fingerprint and not {"Lx", "Ly", "Lz"} <= runtime_fingerprint.keys():
        explicit_cubic = "L" in runtime_fingerprint
        target_box = _compute_box_dimensions(float(runtime_fingerprint["radp"]), explicit_cubic=explicit_cubic)
        runtime_fingerprint.setdefault("Lx", target_box[0])
        runtime_fingerprint.setdefault("Ly", target_box[1])
        runtime_fingerprint.setdefault("Lz", target_box[2])
        runtime_fingerprint.setdefault("L", target_box[2])

    return runtime_fingerprint


def _coerce_parameters(values: Any, *, candidate_id: str) -> tuple[dict[str, float], float | None]:
    payload = _coerce_mapping(values, field_name=f"candidate {candidate_id} parameters")
    if "ka" in payload:
        missing = [name for name in ("ka", "kb") if name not in payload]
        if missing:
            raise ValueError(f"Candidate {candidate_id!r} is missing parameters: {missing}.")
        return (
            {
                "ka": _coerce_float(payload["ka"], field_name="parameters.ka"),
                "kb": _coerce_float(payload["kb"], field_name="parameters.kb"),
                "b1": 0.0,
                "b2": 0.0,
                "a3": 0.0,
                "a4": 0.0,
            },
            None,
        )
    if "Yt" in payload:
        missing = [name for name in ("kb",) if name not in payload]
        if missing:
            raise ValueError(f"Candidate {candidate_id!r} is missing parameters: {missing}.")
        yt = _coerce_float(payload["Yt"], field_name="parameters.Yt")
        ka = _legacy_yt_to_ka(
            yt,
            defaults={**_load_emb_default_parameters(), **payload},
            candidate_id=candidate_id,
        )
        return (
            {
                "ka": ka,
                "kb": _coerce_float(payload["kb"], field_name="parameters.kb"),
                "b1": 0.0,
                "b2": 0.0,
                "a3": 0.0,
                "a4": 0.0,
            },
            yt,
        )

    missing = [name for name in ("ka", "kb") if name not in payload]
    if missing:
        raise ValueError(f"Candidate {candidate_id!r} is missing parameters: {missing}.")

    # Defensive fallback; this branch should only occur when legacy names were not provided.
    return (
        {
            "ka": _coerce_float(payload["ka"], field_name="parameters.ka"),
            "kb": _coerce_float(payload["kb"], field_name="parameters.kb"),
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
        },
        None,
    )


def _validate_direct_ka_kb_bounds(*, parameters: Mapping[str, float], candidate_id: str) -> None:
    from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_BOUNDS

    for key in ("ka", "kb"):
        lower, upper = EMB_34UM_DNN_CAUSAL_BOUNDS[key]
        value = _coerce_float(parameters[key], field_name=f"parameters.{key}")
        if not (float(lower) <= value <= float(upper)):
            raise ValueError(
                f"Candidate {candidate_id!r} parameters.{key}={value} "
                f"is outside bounds [{lower}, {upper}]."
            )


def load_candidate_request(candidate_manifest_path: str | Path) -> Emb34umCandidateRequest:
    path = Path(candidate_manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rendered_payload = _coerce_mapping(manifest.get("rendered_payload"), field_name="rendered_payload")
    request_payload = _coerce_mapping(rendered_payload.get("request_payload"), field_name="request_payload")

    schema_version = request_payload.get("schema_version")
    if schema_version != EMB_34UM_REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported request_payload.schema_version={schema_version!r}; "
            f"expected {EMB_34UM_REQUEST_SCHEMA_VERSION!r}."
        )
    if request_payload.get("request_type") != "emb_34um_full_force_sweep":
        raise ValueError("request_payload.request_type must be 'emb_34um_full_force_sweep'.")
    if request_payload.get("experiment") != "indentation":
        raise ValueError("Only EMB 3.4um indentation requests are supported.")

    candidate_id = str(request_payload.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("request_payload.candidate_id must be non-empty.")

    expected_paths = _coerce_mapping(
        request_payload.get("expected_output_paths"), field_name="expected_output_paths"
    )
    output_root_text = str(request_payload.get("output_root", "")).strip()
    if not output_root_text:
        raise ValueError("request_payload.output_root must be non-empty.")
    hdf5_text = str(expected_paths.get("hdf5", "")).strip()
    request_manifest_text = str(expected_paths.get("request_manifest", "")).strip()
    if not hdf5_text or not request_manifest_text:
        raise ValueError("expected_output_paths must include non-empty hdf5 and request_manifest paths.")

    retry_limit = int(_coerce_float(request_payload.get("retry_limit", 0), field_name="retry_limit"))
    if retry_limit < 0:
        raise ValueError("request_payload.retry_limit must be non-negative.")

    request_parameters = request_payload.get("parameters")
    parameters, compatibility_yt = _coerce_parameters(
        request_parameters,
        candidate_id=candidate_id,
    )
    runtime_source: dict[str, Any] = {}
    if compatibility_yt is None:
        _validate_direct_ka_kb_bounds(parameters=parameters, candidate_id=candidate_id)
    if compatibility_yt is None and isinstance(request_parameters, Mapping):
        fingerprint_source = request_payload.get("fingerprint", {})
        if isinstance(fingerprint_source, Mapping):
            runtime_source.update(fingerprint_source)
        runtime_source.update(request_parameters)
        missing_runtime_fields = [key for key in ("radp", "shell_th") if key not in runtime_source]
        if missing_runtime_fields:
            raise ValueError(
                f"Candidate {candidate_id!r} is missing D4 runtime parameters {missing_runtime_fields}; "
                "provide both 'radp' and 'shell_th' unless using explicit legacy/D2 Yt compatibility."
            )
        from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import EMB_34UM_DNN_CAUSAL_BOUNDS

        for key in ("radp", "shell_th"):
            lower, upper = EMB_34UM_DNN_CAUSAL_BOUNDS[key]
            value = _coerce_float(runtime_source[key], field_name=f"runtime_fingerprint.{key}")
            if not (float(lower) <= value <= float(upper)):
                raise ValueError(
                    f"Candidate {candidate_id!r} runtime_fingerprint.{key}={value} "
                    f"is outside bounds [{lower}, {upper}]."
                )
    from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import is_dnn_causal_low_corner_excluded

    if is_dnn_causal_low_corner_excluded(
        parameters["ka"],
        parameters["kb"],
        runtime_source.get("radp"),
        runtime_source.get("shell_th"),
    ):
        raise ValueError(
            f"Candidate {candidate_id!r} is in the EMB 3.4um runtime-risk timeout region "
            "and must not be submitted."
        )

    return Emb34umCandidateRequest(
        candidate_manifest_path=path,
        candidate_id=candidate_id,
        output_root=Path(output_root_text),
        expected_hdf5_path=Path(hdf5_text),
        expected_request_manifest_path=Path(request_manifest_text),
        parameters=parameters,
        force_grid=_coerce_force_grid(request_payload.get("force_grid"), candidate_id=candidate_id),
        retry_limit=retry_limit,
        fingerprint=dict(_coerce_mapping(request_payload.get("fingerprint", {}), field_name="fingerprint")),
        compatibility_yt=compatibility_yt,
        request_payload=dict(request_payload),
    )


def select_candidate_manifest(batch_summary_path: str | Path, task_index: int) -> Path:
    if task_index < 0:
        raise ValueError("task_index must be non-negative.")
    summary_path = Path(batch_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("rendered_candidate_manifests")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{summary_path} does not contain rendered_candidate_manifests.")
    if task_index >= len(paths):
        raise IndexError(
            f"SLURM array task index {task_index} is outside rendered candidate range 0-{len(paths) - 1}."
        )
    return Path(str(paths[task_index]))


def build_srun_command(
    *,
    runner_script: str | Path,
    candidate_manifest: str | Path,
    retry_attempt: int,
    python_executable: str = "python",
    dry_run: bool = False,
    ntasks: int = 2,
) -> list[str]:
    if retry_attempt < 0:
        raise ValueError("retry_attempt must be non-negative.")
    if ntasks != 2:
        raise ValueError("EMB 3.4um Mirheo execution requires exactly 2 MPI ranks.")
    command = [
        "srun",
        "--ntasks=2",
        python_executable,
        str(runner_script),
        "--candidate-manifest",
        str(candidate_manifest),
        "--retry-attempt",
        str(retry_attempt),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def execution_plan(request: Emb34umCandidateRequest, *, retry_attempt: int) -> dict[str, Any]:
    return {
        "candidate_id": request.candidate_id,
        "candidate_manifest": str(request.candidate_manifest_path),
        "output_root": str(request.output_root),
        "expected_hdf5_path": str(request.expected_hdf5_path),
        "expected_request_manifest_path": str(request.expected_request_manifest_path),
        "force_grid_count": len(request.force_grid),
        "retry_attempt": retry_attempt,
        "retry_limit": request.retry_limit,
        "parameter_vector": list(request.parameter_vector),
    }


def _write_status_manifest(
    request: Emb34umCandidateRequest,
    *,
    status: str,
    retry_attempt: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    request.output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": EMB_34UM_REQUEST_SCHEMA_VERSION,
        "status": status,
        "candidate_id": request.candidate_id,
        "candidate_manifest": str(request.candidate_manifest_path),
        "retry_attempt": retry_attempt,
        "retry_count": retry_attempt,
        "retry_limit": request.retry_limit,
        "request_payload": dict(request.request_payload),
    }
    if extra:
        payload.update(dict(extra))
    request.expected_request_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (request.expected_request_manifest_path, request.output_root / "emb_34um_runtime_status.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _clean_transient_candidate_outputs(request: Emb34umCandidateRequest) -> None:
    request.output_root.mkdir(parents=True, exist_ok=True)
    output_root = request.output_root.resolve()
    targets = [request.output_root / name for name in TRANSIENT_OUTPUT_NAMES]
    if _is_relative_to(request.expected_hdf5_path, output_root):
        targets.append(request.expected_hdf5_path)

    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not _is_relative_to(target, output_root):
            continue
        _remove_path(target)


def _setup_indentation_init_dir(
    request: Emb34umCandidateRequest,
    *,
    repo_root: Path,
    retry_attempt: int,
    dt_scale_factor: float,
    rank: int,
    comm: Any,
) -> Path:
    import yaml

    from meso_uq.mirheo.baseline import validate_training_baseline

    source_dir = repo_root / "emb" / "indentation" / "src"
    defaults_path = source_dir / "parameters-default.emb.yaml"
    init_dir = request.output_root / f"_init_indentation_{EMB_34UM_DIAMETER_UM:.1f}um_attempt{retry_attempt}"

    if rank == 0:
        if init_dir.exists():
            shutil.rmtree(init_dir)
        shutil.copytree(source_dir, init_dir)

        with defaults_path.open("rb") as stream:
            emb_defaults = yaml.load(stream, Loader=yaml.CLoader)
        validate_training_baseline(emb_defaults, str(defaults_path))

        fingerprint = _request_runtime_fingerprint(request)
        dt_multiplier = dt_scale_factor**retry_attempt
        numsteps_multiplier = 1.0 / dt_multiplier
        target_radp = float(fingerprint.get("radp", _DEFAULT_EMB_FINGERPRINT["radp"]))
        target_numsteps = int(float(fingerprint.get("numsteps", emb_defaults.get("numsteps", 5000))))
        target_numsteps_eq = int(float(fingerprint.get("numsteps_eq", emb_defaults.get("numsteps_eq", 10000))))
        target_shell_th = float(fingerprint.get("shell_th", emb_defaults.get("shell_th", 5.0e-9)))
        direct_stiffness_override = bool(fingerprint.get("direct_stiffness_override", True))
        target_bpress = float(fingerprint.get("bpress", emb_defaults.get("bpress", 0.0)))

        if {"Lx", "Ly", "Lz"} <= fingerprint.keys():
            target_box = (
                float(fingerprint.get("Lx", 25.0)),
                float(fingerprint.get("Ly", 25.0)),
                float(fingerprint.get("Lz", 25.0)),
            )
        else:
            explicit_cubic = "L" in fingerprint
            target_box = _compute_box_dimensions(target_radp, explicit_cubic=explicit_cubic)

        parameter_dir = init_dir / "parameter"
        for parameter_file in parameter_dir.glob("parameters-default*.yaml"):
            with parameter_file.open("rb") as stream:
                params = yaml.load(stream, Loader=yaml.CLoader) or {}
            validate_training_baseline(params, str(parameter_file))

            params["Lx"], params["Ly"], params["Lz"] = target_box
            params["radp"] = target_radp
            params["fscale"] = float(fingerprint.get("fscale", params.get("fscale", 0.0074)))
            params["shell_th"] = target_shell_th
            params["direct_stiffness_override"] = direct_stiffness_override
            params["bpress"] = target_bpress
            if "dt" in params:
                params["dt"] = float(params["dt"]) * dt_multiplier
            if "dt_eq" in params:
                params["dt_eq"] = float(params["dt_eq"]) * dt_multiplier
            params["numsteps"] = max(1, int(round(target_numsteps * numsteps_multiplier)))
            params["numsteps_eq"] = max(1, int(round(target_numsteps_eq * numsteps_multiplier)))

            with parameter_file.open("w", encoding="utf-8") as stream:
                yaml.dump(params, stream, Dumper=yaml.CDumper, default_flow_style=False, sort_keys=False)

    comm.Barrier()
    return init_dir


def _write_outputs(
    request: Emb34umCandidateRequest,
    sample: Mapping[str, Any],
    *,
    retry_attempt: int,
    runtime_seconds: float | None = None,
) -> None:
    import h5py
    import numpy as np

    request.output_root.mkdir(parents=True, exist_ok=True)
    request.expected_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    forces = np.asarray(request.force_grid, dtype=float)
    diameters = np.asarray(sample["Reference Evaluations"], dtype=float)
    std = np.asarray(sample.get("Standard Deviation", []), dtype=float)
    params = np.asarray(request.parameter_vector, dtype=float)
    runtime_fingerprint = _request_runtime_fingerprint(request)
    radp = float(runtime_fingerprint.get("radp", _DEFAULT_EMB_FINGERPRINT["radp"]))
    shell_th = float(runtime_fingerprint.get("shell_th", _DEFAULT_EMB_FINGERPRINT["shell_th"]))
    bpress = float(runtime_fingerprint.get("bpress", -91.0))
    if {"Lx", "Ly", "Lz"} <= runtime_fingerprint.keys():
        box_dimensions = {
            "Lx": float(runtime_fingerprint["Lx"]),
            "Ly": float(runtime_fingerprint["Ly"]),
            "Lz": float(runtime_fingerprint["Lz"]),
        }
    else:
        target_box = _compute_box_dimensions(radp, explicit_cubic="L" in runtime_fingerprint)
        box_dimensions = {"Lx": target_box[0], "Ly": target_box[1], "Lz": target_box[2]}
        runtime_fingerprint.setdefault("Lx", target_box[0])
        runtime_fingerprint.setdefault("Ly", target_box[1])
        runtime_fingerprint.setdefault("Lz", target_box[2])
        runtime_fingerprint.setdefault("L", target_box[2])

    result_payload = {
        "candidate_id": request.candidate_id,
        "diameter_um": EMB_34UM_DIAMETER_UM,
        "parameter_names": ["ka", "kb", "b1", "b2", "a3", "a4"],
        "parameters": params.tolist(),
        "force_grid": forces.tolist(),
        "vertical_diameter": diameters.tolist(),
        "standard_deviation": std.tolist(),
        "retry_attempt": retry_attempt,
        "evaluation_mode": "mirheo_emb_34um_final_gate",
        "runtime_fingerprint": runtime_fingerprint,
        "radp": radp,
        "shell_th": shell_th,
        "bpress": bpress,
        "box_dimensions": box_dimensions,
    }
    if runtime_seconds is not None:
        result_payload["runtime_seconds"] = float(runtime_seconds)
    (request.output_root / "emb_34um_result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    with h5py.File(request.expected_hdf5_path, "w") as h5:
        h5.attrs["schema_version"] = EMB_34UM_REQUEST_SCHEMA_VERSION
        h5.attrs["candidate_id"] = request.candidate_id
        h5.attrs["diameter_um"] = EMB_34UM_DIAMETER_UM
        h5.attrs["evaluation_mode"] = "mirheo_emb_34um_final_gate"
        h5.attrs["radp"] = radp
        h5.attrs["shell_th"] = shell_th
        h5.attrs["bpress"] = bpress
        h5.attrs["Lx"] = box_dimensions["Lx"]
        h5.attrs["Ly"] = box_dimensions["Ly"]
        h5.attrs["Lz"] = box_dimensions["Lz"]
        h5.create_dataset("parameter_vector", data=params)
        h5.create_dataset("force_grid", data=forces)
        h5.create_dataset("vertical_diameter", data=diameters)
        h5.create_dataset("standard_deviation", data=std)

    generated_f_delta = request.output_root / "out_hierarchical" / "F_Delta.dat"
    if generated_f_delta.is_file():
        shutil.copy2(generated_f_delta, request.output_root / "F_Delta.dat")


def run_candidate_once(
    request: Emb34umCandidateRequest,
    *,
    repo_root: Path,
    retry_attempt: int,
    dt_scale_factor: float = DEFAULT_DT_SCALE_FACTOR,
) -> None:
    from mpi4py import MPI

    _configure_python_path(repo_root)
    from emb.indentation.evalkit.posterior_indentation import compute_indentation

    runtime_start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 2:
        if rank == 0:
            raise RuntimeError(f"EMB 3.4um Mirheo execution requires exactly 2 MPI ranks, got {size}.")
        return

    if rank == 0:
        _clean_transient_candidate_outputs(request)
        _write_status_manifest(request, status="running", retry_attempt=retry_attempt)
        request.output_root.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    init_dir = _setup_indentation_init_dir(
        request,
        repo_root=repo_root,
        retry_attempt=retry_attempt,
        dt_scale_factor=dt_scale_factor,
        rank=rank,
        comm=comm,
    )

    previous_cwd = Path.cwd()
    os.chdir(request.output_root)
    try:
        sample: dict[str, Any] = {
            "Parameters": list(request.compute_parameter_vector),
            "Sample Id": request.candidate_id,
        }
        compute_indentation(
            sample,
            list(request.force_grid),
            diameter_um=EMB_34UM_DIAMETER_UM,
            init_indentation_path=str(init_dir),
        )
    finally:
        os.chdir(previous_cwd)

    if rank == 0:
        runtime_end = time.time()
        runtime_seconds = max(0.0, runtime_end - runtime_start)
        _write_outputs(request, sample, retry_attempt=retry_attempt, runtime_seconds=runtime_seconds)
        _write_status_manifest(
            request,
            status="completed",
            retry_attempt=retry_attempt,
            extra={
                "result_json": str(request.output_root / "emb_34um_result.json"),
                "hdf5": str(request.expected_hdf5_path),
                "f_delta": str(request.output_root / "F_Delta.dat"),
                "runtime_started_epoch": runtime_start,
                "runtime_finished_epoch": runtime_end,
                "runtime_seconds": runtime_seconds,
            },
        )
        if init_dir.exists():
            shutil.rmtree(init_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--retry-attempt", type=int, default=0)
    parser.add_argument("--dt-scale-factor", type=float, default=DEFAULT_DT_SCALE_FACTOR)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark-failed", action="store_true")
    parser.add_argument("--error-message", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    _configure_python_path(repo_root)

    if args.retry_attempt < 0:
        parser.error("--retry-attempt must be non-negative.")
    request = load_candidate_request(args.candidate_manifest)
    if args.retry_attempt > request.retry_limit:
        raise ValueError(
            f"retry_attempt={args.retry_attempt} exceeds request retry_limit={request.retry_limit}."
        )

    if args.dry_run:
        print(json.dumps(execution_plan(request, retry_attempt=args.retry_attempt), indent=2, sort_keys=True))
        return 0
    if args.mark_failed:
        _write_status_manifest(
            request,
            status="failed",
            retry_attempt=args.retry_attempt,
            extra={"error_message": args.error_message or "candidate execution failed"},
        )
        return 0

    run_candidate_once(
        request,
        repo_root=repo_root,
        retry_attempt=args.retry_attempt,
        dt_scale_factor=args.dt_scale_factor,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
