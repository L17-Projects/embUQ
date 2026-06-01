#!/usr/bin/env python3
from __future__ import annotations

"""Build pilot-only validation artifacts for the EMB 3.4um DNN causal campaign."""

import argparse
import csv
import json
import math
import shlex
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
    validate_dnn_causal_force_grid,
)


BATCH_SUMMARY_FILENAME = "emb_34um_batch_summary.json"
SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_pilot_summary.json"
ROWS_FILENAME = "emb_34um_dnn_causal_validation_pilot_rows.csv"
RUNTIME_PNG_FILENAME = "emb_34um_dnn_causal_validation_pilot_runtime_status.png"
RUNTIME_SIDECAR_FILENAME = f"{RUNTIME_PNG_FILENAME}.json"
COVERAGE_PNG_FILENAME = "emb_34um_dnn_causal_validation_pilot_d4_parameter_coverage.png"
COVERAGE_SIDECAR_FILENAME = f"{COVERAGE_PNG_FILENAME}.json"
FORCE_PNG_FILENAME = "emb_34um_dnn_causal_validation_pilot_force_curve_overlay.png"
FORCE_SIDECAR_FILENAME = f"{FORCE_PNG_FILENAME}.json"
SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_causal_validation_pilot.v1"
DEFAULT_STATUS_FILENAMES = (
    "emb_34um_runtime_status.json",
    "runtime_status.json",
    "result_status.json",
)
_SLURM_FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
_SLURM_RUNNING_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RUNNING",
    "SUSPENDED",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path!s}.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _first_nonempty(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _coerce_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    return parsed


def _coerce_int(value: object, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _extract_ref_mappings(payload: object) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        source = {str(key): value for key, value in payload.items()}
        if {"dataset_id", "hdf5_path", "manifest_path"}.issubset(source):
            return [source]
        refs: list[dict[str, Any]] = []
        if "campaign" in source:
            refs.extend(_extract_ref_mappings(source["campaign"]))
        if "runs" in source:
            refs.extend(_extract_ref_mappings(source["runs"]))
        if refs:
            return refs
        for value in source.values():
            refs.extend(_extract_ref_mappings(value))
        return refs
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        refs: list[dict[str, Any]] = []
        for item in payload:
            refs.extend(_extract_ref_mappings(item))
        return refs
    return []


def _compute_box_dimensions(radp: float, *, explicit_cubic: bool = False) -> tuple[float, float, float]:
    if explicit_cubic:
        box = float(math.ceil(2.0 * radp + 10.0))
        return (box, box, box)
    lx = float(math.ceil(2.0 * radp + 6.0))
    return (lx, lx, float(math.ceil(2.0 * radp + 10.0)))


def _runtime_payload(output_root: Path) -> dict[str, Any]:
    for filename in DEFAULT_STATUS_FILENAMES:
        candidate = output_root / filename
        if candidate.is_file():
            payload = _read_json(candidate)
            payload["status_path"] = str(candidate)
            return payload
    return {}


def _classify_runtime_status(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if status in {"failed", "failure", "error", "cancelled", "timeout"}:
        return "failed"
    if status in {"running", "submitted", "queued", "pending", "rendered", "in_progress", "partial"}:
        return "running"
    if status:
        return "running"
    return "missing"


def _runtime_seconds(payload: Mapping[str, Any]) -> float | None:
    value = _first_nonempty(payload.get("runtime_seconds"), payload.get("elapsed_seconds"), payload.get("duration_seconds"))
    if value is None:
        return None
    parsed = _coerce_float(value, label="runtime_seconds")
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _hdf5_refs(candidate_manifest: Mapping[str, Any], rendered_payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    refs_payload = candidate_manifest.get("expected_hdf5_datasets")
    if refs_payload is None:
        refs_payload = rendered_payload.get("expected_hdf5_datasets")
    if refs_payload is None:
        refs_payload = candidate_manifest.get("expected_hdf5_refs")
    if refs_payload is None:
        refs_payload = rendered_payload.get("expected_hdf5_refs")
    return tuple(_extract_ref_mappings(refs_payload))


def _resolve_hdf5_path(ref: Mapping[str, Any], *, output_root: Path) -> Path | None:
    path_text = str(ref.get("hdf5_path", "")).strip()
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return output_root / path.name


def _result_curve(output_root: Path) -> tuple[Path | None, tuple[float, ...], tuple[float, ...], bool]:
    result_path = output_root / "emb_34um_result.json"
    if not result_path.is_file():
        return None, tuple(), tuple(), False
    payload = _read_json(result_path)
    force_raw = payload.get("force_grid")
    curve_raw = payload.get("vertical_diameter", payload.get("output_curve", payload.get("reference_curve")))
    if force_raw is None or curve_raw is None:
        return result_path, tuple(), tuple(), False
    try:
        force_grid = tuple(_coerce_float(item, label="result.force_grid") for item in _coerce_sequence(force_raw, label="result.force_grid"))
        curve = tuple(_coerce_float(item, label="result.vertical_diameter") for item in _coerce_sequence(curve_raw, label="result.vertical_diameter"))
    except ValueError:
        return result_path, tuple(), tuple(), False
    finite = len(force_grid) == len(curve) and all(math.isfinite(value) for value in (*force_grid, *curve))
    return result_path, force_grid, curve, finite


def _candidate_row(candidate_manifest_path: Path, expected_output_root: Path | None, *, array_task_index: int) -> dict[str, Any]:
    candidate_manifest = _read_json(candidate_manifest_path)
    rendered_payload = _coerce_mapping(candidate_manifest.get("rendered_payload", {}), label="rendered_payload")
    request_payload = _coerce_mapping(rendered_payload.get("request_payload", {}), label="request_payload")
    normalized_payload = _coerce_mapping(candidate_manifest.get("normalized_payload", {}), label="normalized_payload")
    parameters = _coerce_mapping(
        _first_nonempty(normalized_payload.get("parameters"), request_payload.get("parameters"), {}) or {},
        label="parameters",
    )
    fingerprint = _coerce_mapping(
        _first_nonempty(normalized_payload.get("fingerprint"), request_payload.get("fingerprint"), {}) or {},
        label="fingerprint",
    )
    candidate_id = str(_first_nonempty(candidate_manifest.get("candidate_id"), normalized_payload.get("candidate_id"), request_payload.get("candidate_id")) or "").strip()
    if not candidate_id:
        raise ValueError(f"{candidate_manifest_path!s} is missing candidate_id.")
    output_root_value = _first_nonempty(
        normalized_payload.get("output_root"),
        request_payload.get("output_root"),
        candidate_manifest.get("output_root"),
        str(expected_output_root) if expected_output_root is not None else None,
    )
    if output_root_value is None:
        raise ValueError(f"{candidate_id} is missing output_root.")
    output_root = Path(str(output_root_value))

    ka = _coerce_float(parameters.get("ka"), label=f"{candidate_id}.ka")
    kb = _coerce_float(parameters.get("kb"), label=f"{candidate_id}.kb")
    radp = _coerce_float(_first_nonempty(parameters.get("radp"), fingerprint.get("radp")), label=f"{candidate_id}.radp")
    shell_th = _coerce_float(_first_nonempty(parameters.get("shell_th"), fingerprint.get("shell_th")), label=f"{candidate_id}.shell_th")
    bpress = _coerce_float(_first_nonempty(parameters.get("bpress"), fingerprint.get("bpress"), EMB_34UM_DNN_CAUSAL_BPRESS_VALUE), label=f"{candidate_id}.bpress")

    lx_value = _first_nonempty(parameters.get("Lx"), fingerprint.get("Lx"))
    ly_value = _first_nonempty(parameters.get("Ly"), fingerprint.get("Ly"))
    lz_value = _first_nonempty(parameters.get("Lz"), fingerprint.get("Lz"), parameters.get("L"), fingerprint.get("L"))
    if lx_value is None or ly_value is None or lz_value is None:
        box_dimensions = _compute_box_dimensions(radp, explicit_cubic=_first_nonempty(parameters.get("L"), fingerprint.get("L")) is not None)
    else:
        box_dimensions = (
            _coerce_float(lx_value, label=f"{candidate_id}.Lx"),
            _coerce_float(ly_value, label=f"{candidate_id}.Ly"),
            _coerce_float(lz_value, label=f"{candidate_id}.Lz"),
        )

    runtime = _runtime_payload(output_root)
    runtime_status = _classify_runtime_status(runtime)
    runtime_seconds = _runtime_seconds(runtime)
    result_path, force_grid, force_curve, finite_curve = _result_curve(output_root)

    force_grid_count = len(force_grid)
    force_grid_valid = False
    if force_grid:
        try:
            validate_dnn_causal_force_grid(force_grid)
            force_grid_valid = True
        except ValueError:
            force_grid_valid = False

    bounds_valid = all(
        EMB_34UM_DNN_CAUSAL_BOUNDS[name][0] <= value <= EMB_34UM_DNN_CAUSAL_BOUNDS[name][1]
        for name, value in {
            "ka": ka,
            "kb": kb,
            "radp": radp,
            "shell_th": shell_th,
        }.items()
    )
    bpress_valid = math.isclose(bpress, EMB_34UM_DNN_CAUSAL_BPRESS_VALUE, rel_tol=0.0, abs_tol=1e-12)

    hdf5_refs = _hdf5_refs(candidate_manifest, rendered_payload)
    hdf5_records: list[dict[str, Any]] = []
    for ref in hdf5_refs:
        resolved = _resolve_hdf5_path(ref, output_root=output_root)
        hdf5_records.append(
            {
                "dataset_id": str(ref.get("dataset_id", "")).strip(),
                "hdf5_path": "" if resolved is None else str(resolved),
                "present": bool(resolved is not None and resolved.is_file()),
            }
        )
    hdf5_present = bool(hdf5_records) and all(item["present"] for item in hdf5_records)

    status = "missing"
    reasons: list[str] = []
    if result_path is not None and force_curve and not finite_curve:
        status = "nonfinite"
        reasons.append("nonfinite_force_curve")
    elif result_path is not None and force_curve:
        status = "completed"
    elif runtime_status == "failed":
        status = "failed"
        reasons.append("runtime_failed")
    elif runtime_status == "running":
        status = "running"
        reasons.append("runtime_incomplete")
    else:
        status = "missing"
        reasons.append("missing_result_json")

    if result_path is None:
        reasons.append("missing_result_json")
    if not hdf5_records:
        reasons.append("missing_hdf5_contract")
    elif not hdf5_present:
        reasons.append("missing_hdf5_output")
    if result_path is not None and not force_grid_valid:
        reasons.append("invalid_force_grid")
    if result_path is not None and force_curve and len(force_grid) != len(force_curve):
        reasons.append("mismatched_force_curve_length")
    if not bounds_valid:
        reasons.append("out_of_bounds_d4_parameters")
    if not bpress_valid:
        reasons.append("invalid_bpress")

    return {
        "candidate_id": candidate_id,
        "array_task_index": int(array_task_index),
        "candidate_manifest_path": str(candidate_manifest_path),
        "output_root": str(output_root),
        "runtime_status_path": str(runtime.get("status_path", "")),
        "result_json_path": "" if result_path is None else str(result_path),
        "status": status,
        "runtime_status": runtime_status,
        "runtime_seconds": runtime_seconds,
        "ka": ka,
        "kb": kb,
        "radp": radp,
        "shell_th": shell_th,
        "bpress": bpress,
        "box_x": box_dimensions[0],
        "box_y": box_dimensions[1],
        "box_z": box_dimensions[2],
        "force_grid_count": force_grid_count,
        "force_grid": list(force_grid),
        "force_curve": list(force_curve),
        "finite_force_curve": bool(force_curve and finite_curve),
        "force_grid_valid": force_grid_valid,
        "bounds_valid": bounds_valid,
        "bpress_valid": bpress_valid,
        "result_json_present": result_path is not None,
        "hdf5_refs": hdf5_records,
        "hdf5_present": hdf5_present,
        "missing_hdf5_count": sum(0 if item["present"] else 1 for item in hdf5_records),
        "reasons": sorted(set(reasons)),
    }


def _load_rows(batch_summary_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(batch_summary_path)
    manifests = [
        Path(str(item))
        for item in _coerce_sequence(payload.get("rendered_candidate_manifests", ()), label="rendered_candidate_manifests")
    ]
    output_roots_raw = payload.get("expected_output_roots", ())
    output_roots = [
        Path(str(item))
        for item in _coerce_sequence(output_roots_raw, label="expected_output_roots")
    ] if output_roots_raw not in (None, "") else []

    rows: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(manifests):
        expected_output_root = output_roots[index] if index < len(output_roots) else None
        rows.append(_candidate_row(manifest_path, expected_output_root, array_task_index=index))
    return rows


def _normalize_slurm_state(state: object) -> str:
    text = str(state or "").strip().upper()
    return text.split()[0] if text else ""


def _parse_sacct_rows(text: str) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("JobID|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, state, elapsed, exit_code = parts[:4]
        if "." in job_id:
            continue
        if "_" not in job_id:
            continue
        task_text = job_id.rsplit("_", 1)[-1]
        if not task_text.isdigit():
            continue
        states[int(task_text)] = {
            "job_id": job_id,
            "state": _normalize_slurm_state(state),
            "elapsed": elapsed,
            "exit_code": exit_code,
        }
    return states


def _load_slurm_states(slurm_job_id: str | None) -> dict[int, dict[str, Any]]:
    if not slurm_job_id:
        return {}
    command = [
        "sacct",
        "-j",
        str(slurm_job_id),
        "--format=JobID,State,Elapsed,ExitCode",
        "-P",
        "--noheader",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    return _parse_sacct_rows(result.stdout)


def _apply_slurm_states(rows: list[dict[str, Any]], slurm_states: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not slurm_states:
        return rows
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        task_index = int(updated.get("array_task_index", -1))
        slurm = dict(slurm_states.get(task_index, {}))
        slurm_state = _normalize_slurm_state(slurm.get("state"))
        updated["slurm_job_id"] = str(slurm.get("job_id", ""))
        updated["slurm_state"] = slurm_state
        updated["slurm_elapsed"] = str(slurm.get("elapsed", ""))
        updated["slurm_exit_code"] = str(slurm.get("exit_code", ""))
        reasons = list(updated.get("reasons", ()))
        if slurm_state in _SLURM_FAILED_STATES and updated.get("status") != "completed":
            updated["status"] = "failed"
            updated["runtime_status"] = "failed"
            reasons.append(f"slurm_{slurm_state.lower()}")
        elif slurm_state in _SLURM_RUNNING_STATES and updated.get("status") == "missing":
            updated["status"] = "running"
            updated["runtime_status"] = "running"
            reasons.append(f"slurm_{slurm_state.lower()}")
        elif slurm_state == "COMPLETED" and updated.get("status") == "missing":
            reasons.append("slurm_completed_without_result_json")
        updated["reasons"] = sorted(set(reasons))
        normalized_rows.append(updated)
    return normalized_rows


def _runtime_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runtimes = [
        float(row["runtime_seconds"])
        for row in rows
        if row.get("status") == "completed" and row.get("runtime_seconds") is not None
    ]
    if not runtimes:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(runtimes),
        "min": min(runtimes),
        "median": median(runtimes),
        "mean": mean(runtimes),
        "max": max(runtimes),
    }


def _readiness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    total = len(rows)

    if total != EMB_34UM_DNN_CAUSAL_PILOT_SIZE:
        blockers.append(f"expected {EMB_34UM_DNN_CAUSAL_PILOT_SIZE} pilot candidates, found {total}")
    if len(completed_rows) != EMB_34UM_DNN_CAUSAL_PILOT_SIZE:
        blockers.append(f"expected {EMB_34UM_DNN_CAUSAL_PILOT_SIZE} completed candidates, found {len(completed_rows)}")
    if any(not row.get("result_json_present") for row in rows):
        blockers.append("missing result JSON outputs")
    if any(not row.get("hdf5_present") for row in rows):
        blockers.append("missing expected HDF5 outputs")
    if any(int(row.get("force_grid_count", 0)) != len(EMB_34UM_DNN_CAUSAL_FORCE_GRID) for row in rows):
        blockers.append("force grid count is not the required 8 points for every candidate")
    if any(not row.get("force_grid_valid") for row in rows if row.get("result_json_present")):
        blockers.append("one or more result JSON force grids do not match the required 8-point campaign grid")
    if any(row.get("status") == "nonfinite" for row in rows):
        blockers.append("one or more force curves are non-finite")
    if any(row.get("status") == "failed" for row in rows):
        blockers.append("one or more pilot candidates failed at runtime")
    if any(not row.get("finite_force_curve") for row in completed_rows):
        blockers.append("one or more completed force curves are non-finite")
    if any(not row.get("bounds_valid") for row in rows):
        blockers.append("one or more D4 parameter sets are outside protocol bounds")
    if any(not row.get("bpress_valid") for row in rows):
        blockers.append(f"bpress must be fixed at {EMB_34UM_DNN_CAUSAL_BPRESS_VALUE}")

    return {"passed": not blockers, "blockers": blockers}


def build_pilot_validation_report(
    *,
    batch_summary_path: Path,
    generation_command: str | None = None,
    slurm_states: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = _apply_slurm_states(_load_rows(batch_summary_path), slurm_states or {})
    status_counts = {
        "total": len(rows),
        "completed": sum(1 for row in rows if row["status"] == "completed"),
        "running": sum(1 for row in rows if row["status"] == "running"),
        "failed": sum(1 for row in rows if row["status"] == "failed"),
        "missing": sum(1 for row in rows if row["status"] == "missing"),
        "nonfinite": sum(1 for row in rows if row["status"] == "nonfinite"),
    }
    readiness = _readiness(rows)
    missing_outputs = {
        "result_json_missing_count": sum(1 for row in rows if not row["result_json_present"]),
        "hdf5_missing_count": sum(int(row["missing_hdf5_count"]) for row in rows),
        "candidate_ids_without_result_json": [row["candidate_id"] for row in rows if not row["result_json_present"]],
        "candidate_ids_with_missing_hdf5": [row["candidate_id"] for row in rows if not row["hdf5_present"]],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_root": str(batch_summary_path.parent.parent),
        "batch_summary_path": str(batch_summary_path),
        "pilot_stage_root": str(batch_summary_path.parent),
        "generation_command": generation_command or "",
        "slurm_state_enrichment": {
            "enabled": bool(slurm_states),
            "matched_task_count": len(slurm_states or {}),
        },
        "expected_pilot_size": EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
        "required_force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        "required_force_grid_count": len(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        "required_bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
        "d4_bounds": {name: list(bounds) for name, bounds in EMB_34UM_DNN_CAUSAL_BOUNDS.items()},
        "status_counts": status_counts,
        "runtime_seconds_distribution_completed": _runtime_distribution(rows),
        "missing_outputs": missing_outputs,
        "decision": {
            "passed": bool(readiness["passed"]),
            "blocked_reasons": list(readiness["blockers"]),
        },
        "promotion_readiness": readiness,
        "rows": rows,
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "array_task_index",
        "status",
        "runtime_status",
        "slurm_state",
        "slurm_elapsed",
        "slurm_exit_code",
        "slurm_job_id",
        "runtime_seconds",
        "ka",
        "kb",
        "radp",
        "shell_th",
        "bpress",
        "box_x",
        "box_y",
        "box_z",
        "force_grid_count",
        "finite_force_curve",
        "force_grid_valid",
        "bounds_valid",
        "bpress_valid",
        "result_json_present",
        "hdf5_present",
        "missing_hdf5_count",
        "candidate_manifest_path",
        "output_root",
        "runtime_status_path",
        "result_json_path",
        "reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in fieldnames}
            payload["reasons"] = ";".join(str(item) for item in row.get("reasons", ()))
            writer.writerow(payload)


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 16
    height = 16
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((80, 120, 180, 255) if (x + y) % 2 else (230, 235, 240, 255))
        rows.append(bytes(row))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + _png_chunk(b"IEND", b"")


def _write_runtime_plot(path: Path, report: Mapping[str, Any], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    counts = report["status_counts"]
    runtime_payload = report["runtime_seconds_distribution_completed"]
    runtime_values = [
        float(row["runtime_seconds"])
        for row in report["rows"]
        if row.get("status") == "completed" and row.get("runtime_seconds") is not None
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["completed", "running", "failed", "missing", "nonfinite"]
    values = [counts[label] for label in labels]
    axes[0].bar(labels, values, color=["#4c956c", "#f4a259", "#bc4b51", "#8d99ae", "#6c757d"])
    axes[0].set_ylabel("Candidates")
    axes[0].set_title("Pilot runtime and status counts")
    axes[0].tick_params(axis="x", rotation=20)

    if runtime_values:
        axes[1].hist(runtime_values, bins=min(10, max(3, len(runtime_values) // 2)), color="#457b9d")
    axes[1].set_title(
        "Completed runtime seconds"
        if runtime_payload["count"]
        else "Completed runtime seconds unavailable"
    )
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Completed candidates")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_coverage_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    completed = [row for row in rows if row.get("status") == "completed"]
    other = [row for row in rows if row.get("status") != "completed"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        ("ka", "kb", True, True),
        ("ka", "radp", True, False),
        ("kb", "shell_th", True, True),
        ("radp", "shell_th", False, True),
    ]
    for axis, (x_key, y_key, x_log, y_log) in zip(axes.flat, specs):
        if completed:
            axis.scatter([row[x_key] for row in completed], [row[y_key] for row in completed], color="#4c956c", label="completed")
        if other:
            axis.scatter([row[x_key] for row in other], [row[y_key] for row in other], color="#bc4b51", label="not ready")
        if x_log:
            axis.set_xscale("log")
        if y_log:
            axis.set_yscale("log")
        axis.set_xlabel(x_key)
        axis.set_ylabel(y_key)
        axis.set_title(f"{x_key} vs {y_key}")
        axis.axvspan(*EMB_34UM_DNN_CAUSAL_BOUNDS[x_key], alpha=0.08, color="#457b9d")
        axis.axhspan(*EMB_34UM_DNN_CAUSAL_BOUNDS[y_key], alpha=0.08, color="#457b9d")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_force_overlay_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    completed = [row for row in rows if row.get("status") == "completed" and row.get("finite_force_curve")]
    fig, axis = plt.subplots(figsize=(8, 5))
    if completed:
        for row in completed:
            axis.plot(row["force_grid"], row["force_curve"], alpha=0.35, linewidth=1.0)
        pointwise_mean = [
            mean(float(row["force_curve"][index]) for row in completed)
            for index in range(len(completed[0]["force_grid"]))
        ]
        axis.plot(completed[0]["force_grid"], pointwise_mean, color="#bc4b51", linewidth=2.2, label="mean")
        axis.legend()
    axis.set_xlabel("Force")
    axis.set_ylabel("Vertical diameter")
    axis.set_title("Pilot force-curve overlay")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _coverage_sidecar(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parameters = ("ka", "kb", "radp", "shell_th")
    coverage = {
        name: {
            "min": min(float(row[name]) for row in rows) if rows else None,
            "max": max(float(row[name]) for row in rows) if rows else None,
            "bounds": list(EMB_34UM_DNN_CAUSAL_BOUNDS[name]),
            "out_of_bounds_candidate_ids": [
                row["candidate_id"]
                for row in rows
                if not (EMB_34UM_DNN_CAUSAL_BOUNDS[name][0] <= float(row[name]) <= EMB_34UM_DNN_CAUSAL_BOUNDS[name][1])
            ],
        }
        for name in parameters
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_modes": [
            "log10(ka)-vs-log10(kb)",
            "log10(ka)-vs-radp",
            "log10(kb)-vs-log10(shell_th)",
            "radp-vs-log10(shell_th)",
        ],
        "candidate_count": len(rows),
        "coverage": coverage,
    }


def _runtime_sidecar(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status_counts": dict(report["status_counts"]),
        "runtime_seconds_distribution_completed": dict(report["runtime_seconds_distribution_completed"]),
        "promotion_readiness": dict(report["promotion_readiness"]),
    }


def _force_sidecar(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("finite_force_curve")]
    aggregate_curve = []
    if completed:
        aggregate_curve = [
            mean(float(row["force_curve"][index]) for row in completed)
            for index in range(len(completed[0]["force_curve"]))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_curve_count": len(completed),
        "candidate_ids": [row["candidate_id"] for row in completed],
        "force_grid": list(completed[0]["force_grid"]) if completed else [],
        "mean_force_curve": aggregate_curve,
    }


def write_pilot_validation_artifacts(
    *,
    batch_summary_path: Path,
    output_root: Path,
    include_plot: bool = True,
    generation_command: str | None = None,
    slurm_states: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report = build_pilot_validation_report(
        batch_summary_path=batch_summary_path,
        generation_command=generation_command,
        slurm_states=slurm_states,
    )
    rows = list(report["rows"])

    summary_path = output_root / SUMMARY_FILENAME
    rows_path = output_root / ROWS_FILENAME
    runtime_png_path = output_root / RUNTIME_PNG_FILENAME
    runtime_sidecar_path = output_root / RUNTIME_SIDECAR_FILENAME
    coverage_png_path = output_root / COVERAGE_PNG_FILENAME
    coverage_sidecar_path = output_root / COVERAGE_SIDECAR_FILENAME
    force_png_path = output_root / FORCE_PNG_FILENAME
    force_sidecar_path = output_root / FORCE_SIDECAR_FILENAME

    _write_json(summary_path, report)
    _write_rows_csv(rows_path, rows)
    _write_runtime_plot(runtime_png_path, report, include_plot=include_plot)
    _write_coverage_plot(coverage_png_path, rows, include_plot=include_plot)
    _write_force_overlay_plot(force_png_path, rows, include_plot=include_plot)
    _write_json(runtime_sidecar_path, _runtime_sidecar(report))
    _write_json(coverage_sidecar_path, _coverage_sidecar(rows))
    _write_json(force_sidecar_path, _force_sidecar(rows))

    return {
        "summary_path": summary_path,
        "rows_path": rows_path,
        "runtime_png_path": runtime_png_path,
        "runtime_sidecar_path": runtime_sidecar_path,
        "coverage_png_path": coverage_png_path,
        "coverage_sidecar_path": coverage_sidecar_path,
        "force_png_path": force_png_path,
        "force_sidecar_path": force_sidecar_path,
        "report": report,
    }


def _resolve_batch_summary_path(*, campaign_root: str | None, batch_summary: str | None) -> Path:
    if batch_summary:
        return Path(batch_summary)
    if campaign_root:
        return Path(campaign_root) / "pilot" / BATCH_SUMMARY_FILENAME
    raise ValueError("Either --campaign-root or --batch-summary must be provided.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", default=None, help="Campaign root containing pilot/emb_34um_batch_summary.json.")
    parser.add_argument("--batch-summary", default=None, help="Explicit path to pilot/emb_34um_batch_summary.json.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Validation artifact output directory. Defaults to <campaign-root>/pilot/validation.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write fallback PNGs without matplotlib.")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit successfully after writing artifacts even when pilot promotion readiness fails.",
    )
    parser.add_argument(
        "--slurm-job-id",
        default=None,
        help="Optional Slurm array job ID used to enrich rows with TIMEOUT/CANCELLED/FAILED task states.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    batch_summary_path = _resolve_batch_summary_path(campaign_root=args.campaign_root, batch_summary=args.batch_summary)
    campaign_root = batch_summary_path.parent.parent
    output_root = Path(args.output_root) if args.output_root else campaign_root / "pilot" / "validation"
    command = " ".join(shlex.quote(item) for item in [Path(__file__).name, *raw_args])
    slurm_states = _load_slurm_states(args.slurm_job_id)
    artifacts = write_pilot_validation_artifacts(
        batch_summary_path=batch_summary_path,
        output_root=output_root,
        include_plot=not args.no_plots,
        generation_command=command,
        slurm_states=slurm_states,
    )
    print(artifacts["summary_path"])
    print(artifacts["rows_path"])
    print(f"decision_passed={artifacts['report']['decision']['passed']}")
    if artifacts["report"]["decision"]["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
