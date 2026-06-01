from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (
    EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_completed_rows.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME = (
    "emb_34um_dnn_causal_validation_ingestion_manifest.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME = (
    "emb_34um_dnn_causal_validation_ingestion_report.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME = (
    "emb_34um_dnn_causal_validation_ingestion_summary.csv"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME = (
    "emb_34um_dnn_causal_validation_completed_rows.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME = (
    "emb_34um_dnn_causal_validation_replacement_batch_summary.json"
)

DEFAULT_STATUS_FILENAMES = (
    "emb_34um_runtime_status.json",
    "runtime_status.json",
    "result_status.json",
)
DEFAULT_SUCCESS_FILENAMES = ("emb_34um_result.json", "F_Delta.dat")

_FLOAT_TOL = 1e-6
_STAGE_PATTERN = re.compile(
    r"^replica-(?P<replicate>\d{3})/(?P<stage>shared_initial|al-step-\d{2}|lhs-step-\d{2})$"
)
_CYCLE_PATTERN = re.compile(r"-(?P<cycle>\d{2})$")
_PRODUCTION_BRANCHES = frozenset({"unseen_test", "shared_initial", "lhs", "al"})


@dataclass(frozen=True)
class Emb34umDnnCausalValidationIngestionArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    summary_csv_path: Path
    completed_rows_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _coerce_path(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty path.")
    return Path(text)


def _coerce_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _coerce_int(value: object, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _is_empty_marker(value: object) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _first_nonempty(*values: object) -> object | None:
    for value in values:
        if not _is_empty_marker(value):
            return value
    return None


def _coerce_candidate_space(value: object, *, label: str, default: str = "d4") -> str:
    text = str(default if value in (None, "") else value).strip().lower()
    if text not in {"d2", "d4"}:
        raise ValueError(f"{label} must be 'd2' or 'd4'.")
    return text


def _runtime_status_from_output_root(
    output_root: Path,
    *,
    status_filenames: Sequence[str],
) -> dict[str, Any]:
    for name in status_filenames:
        path = output_root / str(name)
        if path.is_file():
            payload = _read_json(path)
            payload["status_path"] = str(path)
            return payload
    return {}


def _runtime_status_label(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if status in {"failed", "failure", "error", "cancelled", "timeout"}:
        return "failed"
    if status:
        return "partial"
    return "missing"


def _retry_count(payload: Mapping[str, Any]) -> int:
    for key in ("retry_count", "retries", "attempts", "retry_attempt"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _retry_limit(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("retry_limit")
    if value in (None, ""):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _parse_f_delta_rows(path: Path, *, expected_force_count: int) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    expected_width = 8 + 2 * expected_force_count
    for row_index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values = [float(item) for item in line.split()]
        except ValueError:
            continue
        if len(values) != expected_width:
            continue
        rows.append(
            {
                "row_index": row_index,
                "parameters": values[:8],
                "curve": values[8 : 8 + expected_force_count],
                "force_grid": values[8 + expected_force_count :],
            }
        )
    return tuple(rows)


def _close_series(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL) for a, b in zip(left, right))


def _read_curve_from_result_json(
    output_root: Path,
    *,
    expected_force_grid: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...], Path] | None:
    result_path = output_root / "emb_34um_result.json"
    if not result_path.is_file():
        return None
    payload = _read_json(result_path)
    raw_force = payload.get("force_grid")
    raw_curve = payload.get("vertical_diameter", payload.get("output_curve", payload.get("reference_curve")))
    if raw_force is None or raw_curve is None:
        return None
    force_grid = tuple(_coerce_float(item, label="result.force_grid") for item in _coerce_sequence(raw_force, label="result.force_grid"))
    curve = tuple(_coerce_float(item, label="result.vertical_diameter") for item in _coerce_sequence(raw_curve, label="result.vertical_diameter"))
    if len(force_grid) != len(curve):
        return None
    if expected_force_grid is not None and not _close_series(force_grid, expected_force_grid):
        return None
    return force_grid, curve, result_path


def _read_curve_from_f_delta(
    output_root: Path,
    *,
    ka: float,
    kb: float,
    expected_force_grid: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...], Path, int] | None:
    f_delta_path = output_root / "F_Delta.dat"
    rows = _parse_f_delta_rows(f_delta_path, expected_force_count=len(expected_force_grid))
    for row in rows:
        params = row["parameters"]
        if not math.isclose(float(params[1]), ka, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL):
            continue
        if not math.isclose(float(params[2]), kb, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL):
            continue
        force_grid = tuple(float(item) for item in row["force_grid"])
        if not _close_series(force_grid, expected_force_grid):
            continue
        curve = tuple(float(item) for item in row["curve"])
        return force_grid, curve, f_delta_path, int(row["row_index"])
    return None


def _candidate_stage_context(
    *,
    campaign_root: Path,
    summary_path: Path,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    relative = summary_path.parent.relative_to(campaign_root).as_posix()
    if relative == "unseen_test":
        return {"stage": "unseen_test", "branch": "unseen_test", "replicate": 0, "cycle": 0}

    match = _STAGE_PATTERN.match(relative)
    if match:
        replicate = int(match.group("replicate"))
        stage = str(match.group("stage"))
        if stage == "shared_initial":
            return {"stage": stage, "branch": "shared_initial", "replicate": replicate, "cycle": 0}
        cycle_match = _CYCLE_PATTERN.search(stage)
        cycle = int(cycle_match.group("cycle")) if cycle_match else 0
        branch = "lhs" if stage.startswith("lhs-step-") else "al"
        return {"stage": stage, "branch": branch, "replicate": replicate, "cycle": cycle}

    if relative == "pilot" or relative.startswith("pilot/"):
        mode = str(summary_payload.get("mode", summary_payload.get("stage", ""))).strip().lower()
        replicate_value = _first_nonempty(summary_payload.get("replica"), summary_payload.get("seed"), 0)
        cycle_value = _first_nonempty(summary_payload.get("cycle"), 0)
        try:
            replicate = int(replicate_value)
        except (TypeError, ValueError):
            replicate = 0
        try:
            cycle = int(cycle_value)
        except (TypeError, ValueError):
            cycle = 0
        cycle_match = _CYCLE_PATTERN.search(mode)
        if cycle_match:
            cycle = int(cycle_match.group("cycle"))
        return {"stage": mode or relative, "branch": "pilot", "replicate": replicate, "cycle": cycle}

    mode = str(summary_payload.get("mode", summary_payload.get("stage", ""))).strip().lower()
    replicate_value = _first_nonempty(summary_payload.get("replica"), summary_payload.get("seed"), 0)
    cycle_value = _first_nonempty(summary_payload.get("cycle"), 0)
    try:
        replicate = int(replicate_value)
    except (TypeError, ValueError):
        replicate = 0
    try:
        cycle = int(cycle_value)
    except (TypeError, ValueError):
        cycle = 0
    cycle_match = _CYCLE_PATTERN.search(mode)
    if cycle_match:
        cycle = int(cycle_match.group("cycle"))
    if mode.startswith("lhs-step-"):
        branch = "lhs"
    elif mode.startswith("al-step-"):
        branch = "al"
    elif mode == "shared_initial":
        branch = "shared_initial"
    elif mode == "unseen_test":
        branch = "unseen_test"
    else:
        branch = "unknown"
    return {"stage": mode or relative, "branch": branch, "replicate": replicate, "cycle": cycle}


def _campaign_stream_for_summary(*, campaign_root: Path, summary_path: Path) -> str:
    try:
        relative_parts = summary_path.parent.relative_to(campaign_root).parts
    except ValueError:
        return "production"
    if relative_parts and str(relative_parts[0]).strip().lower() == "pilot":
        return "pilot"
    return "production"


def _candidate_id(candidate: Mapping[str, Any], request_payload: Mapping[str, Any]) -> str:
    value = _first_nonempty(candidate.get("candidate_id"), request_payload.get("candidate_id"))
    text = str(value or "").strip()
    if not text:
        raise ValueError("candidate manifest is missing candidate_id.")
    return text


def _extract_candidate_record(
    *,
    campaign_root: Path,
    campaign_stream: str,
    summary_path: Path,
    summary_payload: Mapping[str, Any],
    candidate_manifest_path: Path,
    status_filenames: Sequence[str],
) -> dict[str, Any]:
    candidate_manifest = _read_json(candidate_manifest_path)
    rendered_payload = _coerce_mapping(candidate_manifest.get("rendered_payload", {}), label="rendered_payload")
    request_payload = _coerce_mapping(rendered_payload.get("request_payload", {}), label="request_payload")
    normalized_payload = _coerce_mapping(candidate_manifest.get("normalized_payload", {}), label="normalized_payload")
    metadata = _coerce_mapping(candidate_manifest.get("active_learning_metadata", {}), label="active_learning_metadata")

    context = _candidate_stage_context(
        campaign_root=campaign_root,
        summary_path=summary_path,
        summary_payload=summary_payload,
    )
    stage = str(context["stage"])
    branch = str(context["branch"])
    replicate = int(context["replicate"])
    cycle = int(context["cycle"])

    candidate_id = _candidate_id(candidate_manifest, request_payload)
    output_root = _coerce_path(
        _first_nonempty(
            normalized_payload.get("output_root"),
            request_payload.get("output_root"),
            candidate_manifest.get("output_root"),
            metadata.get("output_root"),
        ),
        label=f"{candidate_id}.output_root",
    )
    parameters = _coerce_mapping(
        _first_nonempty(normalized_payload.get("parameters"), request_payload.get("parameters"), {}) or {},
        label=f"{candidate_id}.parameters",
    )
    ka = _coerce_float(parameters.get("ka"), label=f"{candidate_id}.ka")
    kb = _coerce_float(parameters.get("kb"), label=f"{candidate_id}.kb")
    runtime_fingerprint = _coerce_mapping(
        _first_nonempty(normalized_payload.get("fingerprint"), request_payload.get("fingerprint"), {}) or {},
        label=f"{candidate_id}.fingerprint",
    )
    candidate_space = _coerce_candidate_space(
        _first_nonempty(
            normalized_payload.get("candidate_space"),
            request_payload.get("candidate_space"),
            candidate_manifest.get("candidate_space"),
            metadata.get("candidate_space"),
        ),
        label=f"{candidate_id}.candidate_space",
    )
    radp_value = _first_nonempty(parameters.get("radp"), runtime_fingerprint.get("radp"))
    shell_th_value = _first_nonempty(parameters.get("shell_th"), runtime_fingerprint.get("shell_th"))
    radp = _coerce_float(radp_value, label=f"{candidate_id}.radp") if radp_value is not None else None
    shell_th = _coerce_float(shell_th_value, label=f"{candidate_id}.shell_th") if shell_th_value is not None else None
    if (radp is None) != (shell_th is None):
        raise ValueError(f"{candidate_id} must provide both radp and shell_th together.")
    if candidate_space == "d4" and (radp is None or shell_th is None):
        raise ValueError(
            f"{candidate_id} is missing D4 runtime geometry; provide both radp and shell_th "
            "or mark the record explicitly as candidate_space='d2'."
        )

    force_grid_raw = _first_nonempty(
        normalized_payload.get("force_grid"),
        request_payload.get("force_grid"),
        EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    )
    force_grid = tuple(
        _coerce_float(item, label=f"{candidate_id}.force_grid")
        for item in _coerce_sequence(force_grid_raw, label=f"{candidate_id}.force_grid")
    )

    runtime = _runtime_status_from_output_root(output_root, status_filenames=status_filenames)
    runtime_label = _runtime_status_label(runtime)
    retry_count = _retry_count(runtime)
    retry_limit = _retry_limit(runtime)
    runtime_seconds = _coerce_float(
        _first_nonempty(runtime.get("runtime_seconds"), 0.0),
        label=f"{candidate_id}.runtime_seconds",
    )

    result_curve = _read_curve_from_result_json(output_root, expected_force_grid=force_grid)
    f_delta_curve: tuple[tuple[float, ...], tuple[float, ...], Path, int] | None = None
    if result_curve is None:
        f_delta_curve = _read_curve_from_f_delta(
            output_root,
            ka=ka,
            kb=kb,
            expected_force_grid=force_grid,
        )

    reason_codes: list[str] = []
    curve_source_path = ""
    curve_source_kind = ""
    f_delta_row_index: int | None = None
    curve: tuple[float, ...] = tuple()
    curve_force_grid = force_grid

    if result_curve is not None:
        curve_force_grid, curve, curve_path = result_curve
        curve_source_path = str(curve_path)
        curve_source_kind = "result_json"
    elif f_delta_curve is not None:
        curve_force_grid, curve, curve_path, row_index = f_delta_curve
        curve_source_path = str(curve_path)
        curve_source_kind = "f_delta_dat"
        f_delta_row_index = row_index
    else:
        if runtime_label == "failed":
            reason_codes.append("runtime_failed")
        elif runtime_label == "partial":
            reason_codes.append("runtime_partial")
        elif runtime_label == "completed":
            reason_codes.append("runtime_completed_without_curve")
        else:
            reason_codes.append("missing_curve_outputs")

    if curve:
        status = "completed"
    elif runtime_label in {"failed", "partial"}:
        status = runtime_label
    else:
        status = "missing"

    replacement = _as_bool(
        _first_nonempty(
            candidate_manifest.get("replacement"),
            metadata.get("replacement"),
            runtime.get("replacement"),
        ),
        default=False,
    )
    replacement_for_value = _first_nonempty(
        candidate_manifest.get("replacement_for"),
        metadata.get("replacement_for"),
        runtime.get("replacement_for"),
    )
    replacement_for = "" if replacement_for_value is None else str(replacement_for_value).strip()
    if not replacement_for:
        failed_candidate_value = _first_nonempty(
            candidate_manifest.get("failed_candidate_id"),
            metadata.get("failed_candidate_id"),
            runtime.get("failed_candidate_id"),
        )
        replacement_for = "" if failed_candidate_value is None else str(failed_candidate_value).strip()
    if replacement_for:
        replacement = True

    quarantined = bool(
        status in {"failed", "partial", "missing"}
        and retry_limit is not None
        and retry_count >= retry_limit
    )

    if status in {"failed", "partial", "missing"} and retry_limit is not None and retry_count < retry_limit:
        reason_codes.append("retry_window_open")
    if not reason_codes and status != "completed":
        reason_codes.append("unknown_ingestion_failure")

    record = {
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "batch_summary_path": str(summary_path),
        "campaign_stream": str(campaign_stream),
        "output_root": str(output_root),
        "stage": stage,
        "branch": branch,
        "replicate": replicate,
        "cycle": cycle,
        "candidate_space": candidate_space,
        "ka": ka,
        "kb": kb,
        "force_grid": list(curve_force_grid),
        "force_curve": list(curve),
        "reference_curve": list(curve),
        "status": status,
        "runtime_status": runtime_label,
        "runtime_status_path": str(runtime.get("status_path", "")),
        "runtime_seconds": runtime_seconds,
        "retry_count": retry_count,
        "retry_limit": retry_limit,
        "replacement": replacement,
        "replacement_for": replacement_for,
        "quarantined": quarantined,
        "reason_codes": sorted(set(reason_codes)),
        "curve_source": {
            "kind": curve_source_kind,
            "path": curve_source_path,
            "f_delta_row_index": f_delta_row_index,
        },
    }
    if radp is not None:
        record["radp"] = radp
    if shell_th is not None:
        record["shell_th"] = shell_th
    return record


def _is_production_completed_record(record: Mapping[str, Any]) -> bool:
    if str(record.get("status", "")).strip().lower() != "completed":
        return False
    if str(record.get("campaign_stream", "production")).strip().lower() != "production":
        return False
    return str(record.get("branch", "")).strip().lower() in _PRODUCTION_BRANCHES


def _completed_row(record: Mapping[str, Any]) -> dict[str, Any]:
    candidate_space = _coerce_candidate_space(record.get("candidate_space"), label="record.candidate_space")
    ka = _coerce_float(record.get("ka"), label="record.ka")
    kb = _coerce_float(record.get("kb"), label="record.kb")
    radp = _coerce_float(record.get("radp"), label="record.radp") if record.get("radp") is not None else None
    shell_th = (
        _coerce_float(record.get("shell_th"), label="record.shell_th")
        if record.get("shell_th") is not None
        else None
    )
    if (radp is None) != (shell_th is None):
        raise ValueError("completed record must provide both radp and shell_th together.")
    if candidate_space == "d4" and (radp is None or shell_th is None):
        raise ValueError(
            "completed record is missing D4 runtime geometry; provide both radp and shell_th "
            "or mark the record explicitly as candidate_space='d2'."
        )
    force_grid = tuple(
        _coerce_float(item, label="record.force_grid")
        for item in _coerce_sequence(record.get("force_grid", ()), label="record.force_grid")
    )
    curve = tuple(
        _coerce_float(item, label="record.force_curve")
        for item in _coerce_sequence(record.get("force_curve", ()), label="record.force_curve")
    )
    if len(force_grid) != len(curve):
        raise ValueError("completed record force_grid and force_curve lengths must match.")
    validate_dnn_causal_force_grid(force_grid)
    validate_dnn_causal_ensemble_size(EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE)
    completed = {
        "curve_id": str(record.get("candidate_id", "")),
        "candidate_id": str(record.get("candidate_id", "")),
        "candidate_path": str(record.get("candidate_path", "")),
        "candidate_manifest_path": str(record.get("candidate_manifest_path", "")),
        "output_root": str(record.get("output_root", "")),
        "branch": str(record.get("branch", "")),
        "stage": str(record.get("stage", "")),
        "replicate": _coerce_int(record.get("replicate", 0), label="record.replicate"),
        "cycle": _coerce_int(record.get("cycle", 0), label="record.cycle"),
        "candidate_space": candidate_space,
        "ka": ka,
        "kb": kb,
        "parameters": {"ka": ka, "kb": kb},
        "force_grid": list(force_grid),
        "force_curve": list(curve),
        "reference_curve": list(curve),
        "runtime_seconds": _coerce_float(record.get("runtime_seconds", 0.0), label="record.runtime_seconds"),
        "retry_count": _coerce_int(record.get("retry_count", 0), label="record.retry_count"),
        "replacement": _as_bool(record.get("replacement"), default=False),
        "replacement_for": str(record.get("replacement_for", "")),
        "quarantine": _as_bool(record.get("quarantined"), default=False),
        "status": "completed",
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_SCHEMA_VERSION,
        "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    }
    if radp is not None:
        completed["radp"] = radp
        completed["parameters"]["radp"] = radp
    if shell_th is not None:
        completed["shell_th"] = shell_th
        completed["parameters"]["shell_th"] = shell_th
    return completed


def _summary_paths(campaign_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for filename in (
        EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
        EMB_34UM_DNN_CAUSAL_VALIDATION_REPLACEMENT_BATCH_SUMMARY_FILENAME,
    ):
        paths.extend(path for path in sorted(campaign_root.glob(f"**/{filename}")) if path.is_file())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(paths):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _candidate_manifest_paths(summary_payload: Mapping[str, Any]) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[str] = set()
    if "rendered_candidate_manifests" in summary_payload:
        rendered_paths = _coerce_sequence(summary_payload.get("rendered_candidate_manifests"), label="rendered_candidate_manifests")
        for item in rendered_paths:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            paths.append(Path(text))

    if "replacement_records" in summary_payload:
        replacement_records = _coerce_sequence(summary_payload.get("replacement_records"), label="replacement_records")
        for item in replacement_records:
            if not isinstance(item, Mapping):
                continue
            text = str(
                _first_nonempty(
                    item.get("replacement_candidate_manifest_path"),
                    item.get("candidate_manifest_path"),
                    item.get("manifest_path"),
                    "",
                )
            ).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            paths.append(Path(text))

    return tuple(paths)


def _resolve_campaign_manifest_path(
    *,
    campaign_manifest_path: str | Path | None,
    campaign_root: str | Path | None,
) -> Path:
    if campaign_manifest_path is not None:
        return _coerce_path(campaign_manifest_path, label="campaign_manifest_path")
    if campaign_root is None:
        raise ValueError("Either campaign_manifest_path or campaign_root must be provided.")
    return _coerce_path(campaign_root, label="campaign_root") / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME


def _build_ingestion_payloads(
    *,
    campaign_manifest_path: str | Path | None,
    campaign_root: str | Path | None,
    status_filenames: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _resolve_campaign_manifest_path(
        campaign_manifest_path=campaign_manifest_path,
        campaign_root=campaign_root,
    )
    campaign_manifest = _read_json(manifest_path)
    resolved_root = _coerce_path(
        _first_nonempty(campaign_manifest.get("campaign_root"), campaign_root, manifest_path.parent),
        label="campaign_root",
    )

    summary_paths = _summary_paths(resolved_root)
    records: list[dict[str, Any]] = []
    seen_candidate_manifest_paths: set[str] = set()
    for summary_path in summary_paths:
        summary_payload = _read_json(summary_path)
        campaign_stream = _campaign_stream_for_summary(
            campaign_root=resolved_root,
            summary_path=summary_path,
        )
        manifest_paths = _candidate_manifest_paths(summary_payload)
        for candidate_manifest_path in manifest_paths:
            candidate_manifest_key = str(candidate_manifest_path)
            if candidate_manifest_key in seen_candidate_manifest_paths:
                continue
            seen_candidate_manifest_paths.add(candidate_manifest_key)
            if not candidate_manifest_path.is_file():
                records.append(
                    {
                        "candidate_id": candidate_manifest_path.stem,
                        "candidate_path": str(candidate_manifest_path),
                        "candidate_manifest_path": str(candidate_manifest_path),
                        "batch_summary_path": str(summary_path),
                        "campaign_stream": campaign_stream,
                        "output_root": "",
                        "stage": _candidate_stage_context(
                            campaign_root=resolved_root,
                            summary_path=summary_path,
                            summary_payload=summary_payload,
                        )["stage"],
                        "branch": _candidate_stage_context(
                            campaign_root=resolved_root,
                            summary_path=summary_path,
                            summary_payload=summary_payload,
                        )["branch"],
                        "replicate": _candidate_stage_context(
                            campaign_root=resolved_root,
                            summary_path=summary_path,
                            summary_payload=summary_payload,
                        )["replicate"],
                        "cycle": _candidate_stage_context(
                            campaign_root=resolved_root,
                            summary_path=summary_path,
                            summary_payload=summary_payload,
                        )["cycle"],
                        "ka": 0.0,
                        "kb": 0.0,
                        "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
                        "force_curve": [],
                        "reference_curve": [],
                        "status": "missing",
                        "runtime_status": "missing",
                        "runtime_status_path": "",
                        "runtime_seconds": 0.0,
                        "retry_count": 0,
                        "retry_limit": None,
                        "replacement": False,
                        "replacement_for": "",
                        "quarantined": False,
                        "reason_codes": ["missing_candidate_manifest"],
                        "curve_source": {"kind": "", "path": "", "f_delta_row_index": None},
                    }
                )
                continue
            records.append(
                _extract_candidate_record(
                    campaign_root=resolved_root,
                    campaign_stream=campaign_stream,
                    summary_path=summary_path,
                    summary_payload=summary_payload,
                    candidate_manifest_path=candidate_manifest_path,
                    status_filenames=status_filenames,
                )
            )

    completed_rows = tuple(_completed_row(record) for record in records if _is_production_completed_record(record))

    status_counts = {
        "completed": sum(1 for row in records if row.get("status") == "completed"),
        "failed": sum(1 for row in records if row.get("status") == "failed"),
        "partial": sum(1 for row in records if row.get("status") == "partial"),
        "missing": sum(1 for row in records if row.get("status") == "missing"),
    }

    stage_counts: dict[str, dict[str, int]] = {}
    for row in records:
        stage = str(row.get("stage", ""))
        bucket = stage_counts.setdefault(stage, {"completed": 0, "failed": 0, "partial": 0, "missing": 0, "total": 0})
        status = str(row.get("status", ""))
        if status in bucket:
            bucket[status] += 1
        bucket["total"] += 1

    branch_counts: dict[str, dict[str, int]] = {}
    for row in records:
        branch = str(row.get("branch", ""))
        bucket = branch_counts.setdefault(branch, {"completed": 0, "failed": 0, "partial": 0, "missing": 0, "total": 0})
        status = str(row.get("status", ""))
        if status in bucket:
            bucket[status] += 1
        bucket["total"] += 1

    stream_counts: dict[str, dict[str, int]] = {}
    for row in records:
        stream = str(row.get("campaign_stream", "production")).strip().lower() or "production"
        bucket = stream_counts.setdefault(
            stream,
            {"completed": 0, "failed": 0, "partial": 0, "missing": 0, "total": 0},
        )
        status = str(row.get("status", ""))
        if status in bucket:
            bucket[status] += 1
        bucket["total"] += 1

    blockers: list[str] = []
    if not summary_paths:
        blockers.append("No batch summaries were found under the campaign root.")
    if not records:
        blockers.append("No candidate manifests were discovered from batch summaries.")
    if not completed_rows:
        blockers.append("No completed DNN causal result curves were ingested.")
    for required_branch in ("unseen_test", "shared_initial", "lhs", "al"):
        counts = branch_counts.get(required_branch)
        if counts and counts["total"] > 0 and counts["completed"] == 0:
            blockers.append(f"branch={required_branch} has discovered candidates but zero completed curves.")
    unknown_production_completed = [
        row for row in records
        if str(row.get("campaign_stream", "production")).strip().lower() == "production"
        and str(row.get("status", "")).strip().lower() == "completed"
        and str(row.get("branch", "")).strip().lower() not in _PRODUCTION_BRANCHES
    ]
    if unknown_production_completed:
        blockers.append(
            "Production completed rows include unsupported branches; check stage naming and campaign manifests."
        )

    ingestion_manifest = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "completed_rows_schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_SCHEMA_VERSION,
        "campaign_manifest_path": str(manifest_path),
        "campaign_root": str(resolved_root),
        "batch_summary_count": len(summary_paths),
        "record_count": len(records),
        "completed_curve_count": len(completed_rows),
        "records": records,
        "completed_rows": list(completed_rows),
        "metadata": {
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "force_grid": list(EMB_34UM_DNN_CAUSAL_FORCE_GRID),
            "success_filenames": list(DEFAULT_SUCCESS_FILENAMES),
            "status_filenames": list(status_filenames),
        },
    }
    report = {
        **ingestion_manifest,
        "status_counts": status_counts,
        "stage_counts": [
            {"stage": stage, **counts}
            for stage, counts in sorted(stage_counts.items())
        ],
        "branch_counts": [
            {"branch": branch, **counts}
            for branch, counts in sorted(branch_counts.items())
        ],
        "stream_counts": [
            {"stream": stream, **counts}
            for stream, counts in sorted(stream_counts.items())
        ],
        "blockers": blockers,
        "status": "blocked" if blockers else "passed",
        "passed": not blockers,
    }
    return ingestion_manifest, report


def normalize_emb_34um_dnn_causal_validation_records(source: object) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            payload, _ = _build_ingestion_payloads(campaign_manifest_path=None, campaign_root=path, status_filenames=DEFAULT_STATUS_FILENAMES)
            return list(payload["completed_rows"])
        payload = _read_json(path)
        if "completed_rows" in payload and isinstance(payload["completed_rows"], Sequence):
            return [dict(item) for item in payload["completed_rows"] if isinstance(item, Mapping)]
        if "records" in payload and isinstance(payload["records"], Sequence):
            return [dict(item) for item in payload["records"] if isinstance(item, Mapping)]
        if payload.get("schema_version") == "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1":
            ingestion_manifest, _ = _build_ingestion_payloads(
                campaign_manifest_path=path,
                campaign_root=None,
                status_filenames=DEFAULT_STATUS_FILENAMES,
            )
            return list(ingestion_manifest["completed_rows"])
        raise ValueError(f"Unsupported payload format for {path!s}.")
    if isinstance(source, Mapping):
        if "completed_rows" in source:
            return [dict(item) for item in _coerce_sequence(source["completed_rows"], label="completed_rows") if isinstance(item, Mapping)]
        if "records" in source:
            return [dict(item) for item in _coerce_sequence(source["records"], label="records") if isinstance(item, Mapping)]
        raise ValueError("Mapping payload must include completed_rows or records.")
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return [dict(item) for item in source if isinstance(item, Mapping)]
    raise ValueError("records payload must be a sequence, mapping, or campaign path.")


def build_emb_34um_dnn_causal_validation_ingestion_report(
    source: object,
) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            _, report = _build_ingestion_payloads(
                campaign_manifest_path=None,
                campaign_root=path,
                status_filenames=DEFAULT_STATUS_FILENAMES,
            )
            return report
        payload = _read_json(path)
        if payload.get("schema_version") == "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1":
            _, report = _build_ingestion_payloads(
                campaign_manifest_path=path,
                campaign_root=None,
                status_filenames=DEFAULT_STATUS_FILENAMES,
            )
            return report
        if "status" in payload and "blockers" in payload:
            return dict(payload)
    records = normalize_emb_34um_dnn_causal_validation_records(source)
    completed = [row for row in records if str(row.get("status", "completed")) == "completed"]
    blockers: list[str] = []
    if not completed:
        blockers.append("No completed DNN causal result curves were provided.")
    return {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
        "record_count": len(records),
        "completed_curve_count": len(completed),
        "status": "blocked" if blockers else "passed",
        "passed": not blockers,
        "blockers": blockers,
    }


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "candidate_id",
        "stage",
        "branch",
        "replicate",
        "cycle",
        "status",
        "replacement",
        "retry_count",
        "reason_codes",
        "output_root",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id"),
                    "stage": row.get("stage"),
                    "branch": row.get("branch"),
                    "replicate": row.get("replicate"),
                    "cycle": row.get("cycle"),
                    "status": row.get("status"),
                    "replacement": bool(row.get("replacement", False)),
                    "retry_count": row.get("retry_count", 0),
                    "reason_codes": ";".join(row.get("reason_codes", ())),
                    "output_root": row.get("output_root", ""),
                }
            )


def write_emb_34um_dnn_causal_validation_ingestion_artifacts(
    *,
    campaign_manifest_path: str | Path | None = None,
    campaign_root: str | Path | None = None,
    output_root: str | Path | None = None,
    status_filenames: Sequence[str] = DEFAULT_STATUS_FILENAMES,
) -> Emb34umDnnCausalValidationIngestionArtifacts:
    ingestion_manifest, report = _build_ingestion_payloads(
        campaign_manifest_path=campaign_manifest_path,
        campaign_root=campaign_root,
        status_filenames=status_filenames,
    )
    resolved_campaign_root = _coerce_path(ingestion_manifest["campaign_root"], label="campaign_root")
    artifact_dir = Path(output_root) if output_root is not None else resolved_campaign_root / "ingest"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME
    report_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME
    completed_rows_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME

    _write_json(manifest_path, ingestion_manifest)
    _write_json(report_path, report)
    _write_summary_csv(summary_csv_path, ingestion_manifest["records"])
    _write_json(
        completed_rows_path,
        {
            "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_SCHEMA_VERSION,
            "campaign_manifest_path": ingestion_manifest["campaign_manifest_path"],
            "campaign_root": ingestion_manifest["campaign_root"],
            "record_count": len(ingestion_manifest["completed_rows"]),
            "metadata": ingestion_manifest["metadata"],
            "records": ingestion_manifest["completed_rows"],
        },
    )

    return Emb34umDnnCausalValidationIngestionArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        completed_rows_path=completed_rows_path,
        manifest=ingestion_manifest,
        report=report,
    )


__all__ = [
    "EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_SCHEMA_VERSION",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_REPORT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_INGESTION_SUMMARY_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME",
    "DEFAULT_STATUS_FILENAMES",
    "DEFAULT_SUCCESS_FILENAMES",
    "Emb34umDnnCausalValidationIngestionArtifacts",
    "normalize_emb_34um_dnn_causal_validation_records",
    "build_emb_34um_dnn_causal_validation_ingestion_report",
    "write_emb_34um_dnn_causal_validation_ingestion_artifacts",
]
