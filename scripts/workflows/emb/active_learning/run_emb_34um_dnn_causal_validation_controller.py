#!/usr/bin/env python3
"""Build and optionally dispatch EMB 3.4um DNN causal-validation controller stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _script_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_acquisition import (  # noqa: E402
    EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE,
    build_d4_candidate_pool,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_AL_REPLACEMENT_RESERVE_SIZE,
    EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_report import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (  # noqa: E402
    write_emb_34um_dnn_causal_validation_ingestion_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME,
    write_emb_34um_dnn_causal_validation_metric_artifacts,
)


CONTROLLER_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_causal_validation_controller.v2"
CONTROLLER_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_controller_manifest.json"
AL_TRAIN_SCORE_SELECT_CANDIDATE_SPACE = "d4"
AL_TRAIN_SCORE_SELECT_ACQUISITION_SCORE_MODE = "curve_error"
DEFAULT_EXECUTION_MODE = "render-only"
SUPPORTED_EXECUTION_MODES = frozenset({"render-only", "dry-run", "execute"})
INGEST_ROWS_FILENAME = EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME
INGEST_REPORT_FILENAME = EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME
PILOT_VALIDATION_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_pilot_summary.json"
CANDIDATE_POOL_GENERATION_FILENAME = "candidate_pool_generation.json"

_STEP_PATTERN = re.compile(r"^(?:al|lhs)-step-(\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    normalized: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"{path} items must be JSON objects.")
        normalized.append(dict(item))
    return normalized


def _candidate_space_midpoint(key: str) -> float:
    low, high = EMB_34UM_DNN_CAUSAL_BOUNDS[key]
    return (float(low) + float(high)) / 2.0


def _load_prepare_module() -> Any:
    script_path = _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_dnn_causal_validation.py"
    spec = importlib.util.spec_from_file_location("_prepare_emb_34um_dnn_causal_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prepare module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_exec() -> str:
    return shlex.quote(sys.executable)


def _quoted(value: str | Path | int | float | bool | None) -> str:
    return shlex.quote(str(value))


def _coerce_execution_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(f"Unsupported execution mode {value!r}; expected one of {sorted(SUPPORTED_EXECUTION_MODES)}.")
    return mode


def _coerce_stage_entries(entries: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(entries, list):
        raise ValueError("command_inventory.entries must be a list.")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("command_inventory.entries items must be mappings.")
        normalized.append(dict(entry))
    return tuple(normalized)


def _replace_execution_mode(command: str, *, execution_mode: str) -> str:
    if "EXECUTION_MODE=render-only" in command:
        return command.replace("EXECUTION_MODE=render-only", f"EXECUTION_MODE={execution_mode}")
    if "EXECUTION_MODE=execute" in command:
        return command.replace("EXECUTION_MODE=execute", f"EXECUTION_MODE={execution_mode}")
    if "EXECUTION_MODE=dry-run" in command:
        return command.replace("EXECUTION_MODE=dry-run", f"EXECUTION_MODE={execution_mode}")
    return f"EXECUTION_MODE={execution_mode} {command}"


def _entry_step(mode: str) -> int | None:
    match = _STEP_PATTERN.fullmatch(mode)
    if match is None:
        return None
    return int(match.group(1))


def _entry_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(entry.get("mode", ""))
    return {
        "mode": mode,
        "replica": int(entry.get("replica", 0)),
        "cycle": int(entry.get("cycle", 0)),
        "step": _entry_step(mode) or int(entry.get("cycle", 0)),
        "candidate_count": int(entry.get("candidate_count", 0)),
        "array": str(entry.get("array", "")),
        "execution_mode": str(entry.get("execution_mode", DEFAULT_EXECUTION_MODE)),
        "batch_summary_path": str(entry.get("batch_summary_path", "")),
        "command": str(entry.get("command", "")),
    }


def _stage_root(campaign_root: Path, *, replica: int, step: int) -> Path:
    return campaign_root / f"replica-{replica:03d}" / f"al-step-{step:02d}"


def _selection_inputs_root(campaign_root: Path, *, replica: int, step: int) -> Path:
    return _stage_root(campaign_root, replica=replica, step=step) / "selection_inputs"


def _selection_training_root(campaign_root: Path, *, replica: int, step: int) -> Path:
    return _stage_root(campaign_root, replica=replica, step=step) / "train_score_select"


def _selection_training_manifest_path(campaign_root: Path, *, replica: int, step: int) -> Path:
    return _selection_training_root(campaign_root, replica=replica, step=step) / "emb_34um_dnn_train_score_select_manifest.json"


def _selection_training_report_path(campaign_root: Path, *, replica: int, step: int) -> Path:
    return _selection_training_root(campaign_root, replica=replica, step=step) / "emb_34um_dnn_train_score_select_report.json"


def _selection_timing_report_path(campaign_root: Path, *, replica: int, step: int) -> Path:
    return _selection_training_root(campaign_root, replica=replica, step=step) / "emb_34um_dnn_train_score_select_timing_canary.json"


def _build_prepare_stage_command(
    *,
    campaign_root: Path,
    design_manifest: Mapping[str, Any],
    execution_mode: str,
    command_inventory: Mapping[str, Any],
    cycle_count: int,
    active_replicate_count: int,
) -> str:
    prepare_script = _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_dnn_causal_validation.py"
    timestamp = str(design_manifest.get("timestamp", campaign_root.name))
    scratch_root = Path(str(design_manifest.get("scratch_root", campaign_root.parent)))
    run_id_prefix = str(design_manifest.get("run_id_prefix", "emb-34um-dnn-causal-validation"))
    walltime = str(command_inventory.get("walltime", EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT))
    concurrent_jobs = int(command_inventory.get("concurrent_jobs", 30))
    retry_limit = int(command_inventory.get("retry_limit", EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT))
    policy = design_manifest.get("policy", {})
    protocol = design_manifest.get("protocol", {})
    policy_map = policy if isinstance(policy, Mapping) else {}
    protocol_map = protocol if isinstance(protocol, Mapping) else {}
    max_replicate_count = _coerce_positive_int(
        policy_map.get("max_replicate_count", protocol_map.get("max_replicate_count", active_replicate_count)),
        label="max_replicate_count",
    )
    force_grid = design_manifest.get("force_grid_source")
    vault_root_timestamp = str(design_manifest.get("vault_root_timestamp", "")).strip()
    vault_root = Path(vault_root_timestamp).parent if vault_root_timestamp else campaign_root.parent / "vault"

    parts = [
        f"EXECUTION_MODE={_quoted(execution_mode)}",
        _python_exec(),
        _quoted(prepare_script),
        "--timestamp",
        _quoted(timestamp),
        "--scratch-root",
        _quoted(scratch_root),
        "--vault-root",
        _quoted(vault_root),
        "--walltime",
        _quoted(walltime),
        "--concurrent-jobs",
        str(concurrent_jobs),
        "--retry-limit",
        str(retry_limit),
        "--run-id-prefix",
        _quoted(run_id_prefix),
        "--cycle-count",
        str(cycle_count),
        "--active-replicate-count",
        str(active_replicate_count),
        "--max-replicate-count",
        str(max_replicate_count),
        "--include-coverage-plot-requirements",
    ]
    if isinstance(force_grid, str) and force_grid and force_grid != "protocol_default":
        parts.extend(["--force-grid", _quoted(force_grid)])
    return " ".join(parts)


def _build_stage_submit_commands(
    entries: list[dict[str, Any]],
    *,
    execution_mode: str,
) -> list[str]:
    return [_replace_execution_mode(str(entry["command"]), execution_mode=execution_mode) for entry in entries]


def _build_al_inputs_command(
    *,
    campaign_root: Path,
    replica: int,
    step: int,
    execution_mode: str,
) -> str:
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_exec(),
            _quoted(Path(__file__).resolve()),
            "--stage-action",
            "al-build-inputs",
            "--campaign-root",
            _quoted(campaign_root),
            "--replica",
            str(replica),
            "--cycle",
            str(step),
            "--execution-mode",
            _quoted(execution_mode),
        ]
    )


def _build_al_train_score_select_command(
    *,
    campaign_root: Path,
    replica: int,
    step: int,
    candidate_count: int,
    execution_mode: str,
) -> str:
    selection_count = int(candidate_count) + int(EMB_34UM_DNN_CAUSAL_AL_REPLACEMENT_RESERVE_SIZE)
    input_root = _selection_inputs_root(campaign_root, replica=replica, step=step)
    completed_rows = input_root / "completed_rows.json"
    candidate_pool = input_root / "candidate_pool.json"
    output_root = _selection_training_root(campaign_root, replica=replica, step=step)
    if execution_mode == "execute":
        wrapper = _REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_dnn_train_score_select.sbatch"
        return " ".join(
            [
                "sbatch --parsable",
                "--export=ALL,"
                f"REPO_ROOT={_quoted(_REPO_ROOT)},"
                f"COMPLETED_ROWS={_quoted(completed_rows)},"
                f"CANDIDATE_POOL={_quoted(candidate_pool)},"
                f"OUTPUT_ROOT={_quoted(output_root)},"
                "DEVICE=cuda,"
                "EPOCHS=200,"
                "BATCH_SIZE=128,"
                "LEARNING_RATE=0.001,"
                "VALIDATION_FRACTION=0.1,"
                f"TOP_N={selection_count},"
                f"CANDIDATE_SPACE={AL_TRAIN_SCORE_SELECT_CANDIDATE_SPACE},"
                f"ACQUISITION_SCORE_MODE={AL_TRAIN_SCORE_SELECT_ACQUISITION_SCORE_MODE},"
                "TIMING_CANARY=1",
                _quoted(wrapper),
            ]
        )

    script = _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "run_emb_34um_dnn_train_score_select.py"
    parts = [
        f"EXECUTION_MODE={_quoted(execution_mode)}",
        _python_exec(),
        _quoted(script),
        "--completed-rows",
        _quoted(completed_rows),
        "--candidate-pool",
        _quoted(candidate_pool),
        "--output-root",
        _quoted(output_root),
        "--top-n",
        str(selection_count),
        "--candidate-space",
        AL_TRAIN_SCORE_SELECT_CANDIDATE_SPACE,
        "--acquisition-score-mode",
        AL_TRAIN_SCORE_SELECT_ACQUISITION_SCORE_MODE,
        "--timing-canary",
    ]
    if execution_mode in {"render-only", "dry-run"}:
        parts.append("--dry-run")
    return " ".join(parts)


def _build_al_render_selected_command(
    *,
    campaign_root: Path,
    replica: int,
    step: int,
    execution_mode: str,
) -> str:
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_exec(),
            _quoted(Path(__file__).resolve()),
            "--stage-action",
            "al-render-selected",
            "--campaign-root",
            _quoted(campaign_root),
            "--replica",
            str(replica),
            "--cycle",
            str(step),
            "--execution-mode",
            _quoted(execution_mode),
        ]
    )


def _build_final_ingest_command(*, campaign_root: Path, execution_mode: str) -> str:
    if execution_mode == "execute":
        wrapper = _REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_dnn_causal_validation_metrics.sbatch"
        return " ".join(
            [
                "sbatch --parsable",
                "--export=ALL,"
                f"REPO_ROOT={_quoted(_REPO_ROOT)},"
                f"CAMPAIGN_ROOT={_quoted(campaign_root)},"
                f"OUTPUT_ROOT={_quoted(campaign_root / 'ingest')},"
                f"ANALYZE_OUTPUT_ROOT={_quoted(campaign_root / 'analyze')},"
                "DEVICE=cuda,"
                "DRY_RUN=0,"
                "RUN_ANALYZE=1,"
                "ALLOW_BLOCKED=1",
                _quoted(wrapper),
            ]
        )
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_exec(),
            _quoted(Path(__file__).resolve()),
            "--stage-action",
            "final-ingest",
            "--campaign-root",
            _quoted(campaign_root),
            "--execution-mode",
            _quoted(execution_mode),
        ]
    )


def _build_final_analyze_command(*, campaign_root: Path, execution_mode: str) -> str:
    script = _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "analyze_emb_34um_dnn_causal_validation.py"
    rows_path = campaign_root / "ingest" / INGEST_ROWS_FILENAME
    output_root = campaign_root / "analyze"
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_exec(),
            _quoted(script),
            "--rows",
            _quoted(rows_path),
            "--output-root",
            _quoted(output_root),
            "--allow-blocked",
        ]
    )


def _pilot_validation_summary_path(*, campaign_root: Path, design_manifest: Mapping[str, Any]) -> Path:
    configured = design_manifest.get("pilot_validation_summary_path")
    if isinstance(configured, str) and configured.strip():
        configured_path = Path(configured.strip())
        if configured_path.is_absolute():
            return configured_path
        return (campaign_root / configured_path).resolve()
    return campaign_root / "pilot" / "validation" / PILOT_VALIDATION_SUMMARY_FILENAME


def _build_pilot_verify_command(
    *,
    campaign_root: Path,
    summary_path: Path,
    execution_mode: str,
) -> str:
    script = _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "analyze_emb_34um_dnn_causal_validation_pilot.py"
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_exec(),
            _quoted(script),
            "--campaign-root",
            _quoted(campaign_root),
            "--output-root",
            _quoted(summary_path.parent),
        ]
    )


def _build_stage(
    *,
    name: str,
    description: str,
    command_type: str,
    commands: list[str],
    dependencies: list[str],
    expected_output_roots: list[str],
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "command_type": command_type,
        "dependencies": list(dependencies),
        "commands": [{"command": command} for command in commands],
        "expected_output_roots": list(expected_output_roots),
        "status": "blocked" if blockers else "planned",
    }
    if blockers:
        payload["blockers"] = list(blockers)
    return payload


def _load_result_row(output_root: Path) -> dict[str, Any]:
    result_path = output_root / "emb_34um_result.json"
    payload = _load_json(result_path)
    names = payload.get("parameter_names", [])
    values = payload.get("parameters", [])
    if not isinstance(names, list) or not isinstance(values, list):
        raise ValueError(f"Malformed result payload at {result_path}.")
    params = {str(name): float(value) for name, value in zip(names, values)}
    force_grid = tuple(float(item) for item in payload.get("force_grid", []))
    target_curve = tuple(float(item) for item in payload.get("vertical_diameter", []))
    if "ka" not in params or "kb" not in params:
        raise ValueError(f"{result_path} does not expose ka/kb.")
    if len(force_grid) != len(target_curve) or not force_grid:
        raise ValueError(f"{result_path} force grid and target curve must be non-empty with equal length.")
    parameters: dict[str, float] = {
        "ka": float(params["ka"]),
        "kb": float(params["kb"]),
    }
    if "radp" in params:
        parameters["radp"] = float(params["radp"])
    elif "radp" in payload:
        parameters["radp"] = float(payload["radp"])
    if "shell_th" in params:
        parameters["shell_th"] = float(params["shell_th"])
    elif "shell_th" in payload:
        parameters["shell_th"] = float(payload["shell_th"])

    return {
        "curve_id": str(payload.get("candidate_id") or output_root.name),
        "parameters": parameters,
        "force_grid": list(force_grid),
        "target_curve": list(target_curve),
    }


def _stage_batch_summary_roots(summary_path: Path) -> tuple[Path, ...]:
    payload = _load_json(summary_path)
    roots = payload.get("expected_output_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{summary_path} does not contain expected_output_roots.")
    return tuple(Path(str(item)) for item in roots)


def _existing_points(completed_rows: list[dict[str, Any]]) -> list[tuple[float, ...]]:
    points: list[tuple[float, ...]] = []
    for row in completed_rows:
        parameters = row.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        ka = float(parameters.get("ka", 0.0))
        kb = float(parameters.get("kb", 0.0))
        if ka > 0.0 and kb > 0.0:
            if "radp" in parameters and "shell_th" in parameters:
                points.append((ka, kb, float(parameters["radp"]), float(parameters["shell_th"])))
            else:
                points.append((ka, kb))
    return points


def _candidate_pool_key(point: tuple[float, ...]) -> tuple[float, ...]:
    if len(point) >= 4:
        return (
            round(float(point[0]), 12),
            round(float(point[1]), 12),
            round(float(point[2]), 12),
            round(float(point[3]), 12),
        )
    if len(point) >= 2:
        return (
            round(float(point[0]), 12),
            round(float(point[1]), 12),
        )
    raise ValueError("Candidate points must contain at least ka and kb.")


def _generate_candidate_pool(
    *,
    replica: int,
    step: int,
    pool_size: int,
    existing_points: list[tuple[float, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = replica * 10_000 + 33_000 + step
    pool, metadata = build_d4_candidate_pool(
        requested_size=pool_size,
        seed=seed,
        generator="auto",
        existing_points=existing_points,
    )
    tagged_pool = []
    for index, row in enumerate(pool, start=1):
        payload = dict(row)
        payload["candidate_id"] = f"rep{replica:03d}-step{step:02d}-pool-{index:06d}"
        tagged_pool.append(payload)
    metadata = {
        "candidate_pool_generator": "auto",
        "acquisition_score_mode": AL_TRAIN_SCORE_SELECT_ACQUISITION_SCORE_MODE,
        **dict(metadata),
    }
    return tagged_pool, metadata


def _write_al_selection_inputs(*, campaign_root: Path, replica: int, step: int) -> dict[str, str]:
    manifest = _load_json(campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME)
    prior_summary_paths = [campaign_root / f"replica-{replica:03d}" / "shared_initial" / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME]
    for prior in range(1, step):
        prior_summary_paths.append(
            campaign_root
            / f"replica-{replica:03d}"
            / f"al-step-{prior:02d}"
            / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
        )

    completed_rows: list[dict[str, Any]] = []
    for summary_path in prior_summary_paths:
        for output_root in _stage_batch_summary_roots(summary_path):
            completed_rows.append(_load_result_row(output_root))

    target_candidate_count = 100
    for entry in _coerce_stage_entries(manifest.get("command_inventory", {}).get("entries", [])):
        mode = str(entry.get("mode", ""))
        if mode == f"al-step-{step:02d}" and int(entry.get("replica", 0)) == replica:
            target_candidate_count = int(entry.get("candidate_count", 100))
            break

    inputs_root = _selection_inputs_root(campaign_root, replica=replica, step=step)
    inputs_root.mkdir(parents=True, exist_ok=True)
    completed_path = inputs_root / "completed_rows.json"
    candidate_pool_path = inputs_root / "candidate_pool.json"
    candidate_pool_generation_path = inputs_root / CANDIDATE_POOL_GENERATION_FILENAME
    completed_path.write_text(json.dumps(completed_rows, indent=2, sort_keys=True), encoding="utf-8")

    pool_size = EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE
    existing_points = _existing_points(completed_rows)
    candidate_pool, candidate_pool_metadata = _generate_candidate_pool(
        replica=replica,
        step=step,
        pool_size=pool_size,
        existing_points=existing_points,
    )
    candidate_pool_path.write_text(json.dumps(candidate_pool, indent=2, sort_keys=True), encoding="utf-8")
    candidate_pool_generation_path.write_text(
        json.dumps(candidate_pool_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "completed_rows_path": str(completed_path),
        "candidate_pool_path": str(candidate_pool_path),
        "candidate_pool_generation_path": str(candidate_pool_generation_path),
        "candidate_count": str(target_candidate_count),
        "candidate_pool_count": str(pool_size),
    }


def _candidate_pool_for_stage(*, campaign_root: Path, replica: int, step: int) -> dict[str, dict[str, Any]]:
    candidate_pool = _load_json_list(_selection_inputs_root(campaign_root, replica=replica, step=step) / "candidate_pool.json")
    pool_by_id: dict[str, dict[str, Any]] = {}
    for record in candidate_pool:
        candidate_id = str(record.get("candidate_id", "")).strip()
        if candidate_id:
            pool_by_id[candidate_id] = record
    return pool_by_id


def _selected_row_d4_fields(*, pool_row: Mapping[str, Any] | None) -> tuple[float, float, dict[str, Any]]:
    if pool_row is None:
        raise ValueError("Selected AL candidate is missing its source candidate_pool row; D4 geometry cannot be inferred.")
    missing = [key for key in ("radp", "shell_th") if key not in pool_row]
    if missing:
        raise ValueError(f"Selected AL candidate pool row is missing D4 fields: {missing}.")
    radp = float(pool_row["radp"])
    shell_th = float(pool_row["shell_th"])
    runtime_fingerprint = dict(pool_row.get("runtime_fingerprint", {}))
    runtime_fingerprint.setdefault("radp", radp)
    runtime_fingerprint.setdefault("shell_th", shell_th)
    runtime_fingerprint.setdefault("bpress", EMB_34UM_DNN_CAUSAL_BPRESS_VALUE)
    return radp, shell_th, runtime_fingerprint


def _render_selected_candidates_for_stage(
    *,
    campaign_root: Path,
    replica: int,
    step: int,
    execution_mode: str,
) -> dict[str, Any]:
    train_manifest_path = _selection_training_manifest_path(campaign_root, replica=replica, step=step)
    train_manifest = _load_json(train_manifest_path)
    selected_points = train_manifest.get("selected_points", [])
    if not isinstance(selected_points, list) or not selected_points:
        raise ValueError(f"train_score_select manifest has no selected_points: {train_manifest_path}")

    design_manifest = _load_json(campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME)
    run_id_prefix = str(design_manifest.get("run_id_prefix", "emb-34um-dnn-causal-validation"))
    stage_root = _stage_root(campaign_root, replica=replica, step=step)
    target_count = 100
    for entry in _coerce_stage_entries(design_manifest.get("command_inventory", {}).get("entries", [])):
        if str(entry.get("mode", "")) == f"al-step-{step:02d}" and int(entry.get("replica", 0)) == replica:
            target_count = int(entry.get("candidate_count", 100))
            break

    force_grid = tuple(float(item) for item in train_manifest.get("fit", {}).get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID))
    candidate_rows = selected_points[:target_count]
    if not candidate_rows:
        raise ValueError(f"No selected candidates available for replica={replica} step={step}.")
    candidate_pool = _candidate_pool_for_stage(campaign_root=campaign_root, replica=replica, step=step)

    candidate_records: list[dict[str, Any]] = []
    for order, row in enumerate(candidate_rows, start=1):
        ka = float(row["ka"])
        kb = float(row["kb"])
        candidate_pool_id = str(row.get("candidate_pool_id", ""))
        candidate_pool_candidate_id = str(row.get("candidate_id", ""))
        pool_row = candidate_pool.get(candidate_pool_id) if candidate_pool_id else None
        if pool_row is None and candidate_pool_candidate_id:
            pool_row = candidate_pool.get(candidate_pool_candidate_id)
        radp, shell_th, runtime_fingerprint = _selected_row_d4_fields(pool_row=pool_row)
        candidate_id = f"{run_id_prefix}-rep{replica:03d}-al-step{step:02d}-c{order:03d}"
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "family": "emb",
                "experiment": "indentation",
                "ka": ka,
                "kb": kb,
                "radp": radp,
                "shell_th": shell_th,
                "selection_seed": replica * 10_000 + step,
                "selection_mode": "dnn_causal_fresh_only",
                "selection_source": "dnn_ensemble_disagreement_diversity",
                "selection_status": "rendered",
                "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
                "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
                "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
                "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
                "runtime_fingerprint": runtime_fingerprint,
                "selection_payload": {
                    "candidate_pool_id": str(row.get("candidate_pool_id", row.get("candidate_id", ""))),
                    "acquisition_score": float(row.get("acquisition_score", 0.0)),
                    "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                    "diversity_term": float(row.get("diversity_term", 0.0)),
                },
            }
        )

    reserve_rows = selected_points[target_count:]
    reserve_payload: list[dict[str, Any]] = []
    for row in reserve_rows:
        if not isinstance(row, Mapping) or "ka" not in row or "kb" not in row:
            continue
        candidate_pool_id = str(row.get("candidate_pool_id", row.get("candidate_id", "")))
        pool_row = candidate_pool.get(candidate_pool_id) if candidate_pool_id else None
        radp, shell_th, runtime_fingerprint = _selected_row_d4_fields(pool_row=pool_row)
        reserve_payload.append(
            {
                "candidate_pool_id": candidate_pool_id,
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "radp": radp,
                "shell_th": shell_th,
                "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                "runtime_fingerprint": runtime_fingerprint,
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                "diversity_term": float(row.get("diversity_term", 0.0)),
            }
        )

    selection_manifest_path = stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME
    selection_summary_path = stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME
    runtime_batch_summary_path = stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
    walltime, concurrent_jobs, retry_limit = _resolve_stage_runtime_controls(design_manifest)

    if execution_mode == "execute":
        prepare = _load_prepare_module()
        candidates = prepare._build_candidates_for_batch(tuple(candidate_records), force_grid=force_grid)
        render_result = prepare._render_batch(
            candidates=candidates,
            batch_root=stage_root,
            run_id=f"{run_id_prefix}-rep{replica:03d}-al-step{step:02d}",
            batch_id=f"rep{replica:03d}-al-step-{step:02d}",
            walltime=walltime,
            concurrent_jobs=concurrent_jobs,
            retry_limit=retry_limit,
        )
    else:
        runtime_batch_summary_path.parent.mkdir(parents=True, exist_ok=True)
        render_result = {
            "batch_root": str(stage_root),
            "candidate_count": len(candidate_records),
            "rendered_candidate_manifests": [],
            "expected_output_roots": [],
            "submission_expected": {
                "render_only": execution_mode != "execute",
                "submission_commands_empty": True,
                "submitted": False,
            },
        }
        runtime_batch_summary_path.write_text(json.dumps(render_result, indent=2, sort_keys=True), encoding="utf-8")

    selection_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_selection_manifest.v1",
        "status": "selection_rendered" if execution_mode == "execute" else "selection_planned",
        "replica": replica,
        "cycle": step,
        "candidate_count": len(candidate_records),
        "candidate_records": candidate_records,
        "candidate_reserve": reserve_payload,
        "candidate_reserve_count": len(reserve_payload),
        "selected_from_pool_count": len(selected_points),
        "force_grid": list(force_grid),
        "train_score_select_manifest_path": str(train_manifest_path),
        "train_score_select_report_path": str(_selection_training_report_path(campaign_root, replica=replica, step=step)),
        "batch_summary_path": str(runtime_batch_summary_path),
        "scheduler_controls": {
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
        },
    }
    selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest_path.write_text(json.dumps(selection_manifest, indent=2, sort_keys=True), encoding="utf-8")

    selection_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_selection_batch_summary.v1",
        "status": selection_manifest["status"],
        "replica": replica,
        "cycle": step,
        "candidate_count": len(candidate_records),
        "selected_count": len(candidate_records),
        "candidate_reserve_count": len(reserve_payload),
        "candidate_pool_count": len(selected_points),
        "batch_summary_path": str(runtime_batch_summary_path),
        "selection_manifest_path": str(selection_manifest_path),
    }
    selection_summary_path.write_text(json.dumps(selection_summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "selection_manifest_path": str(selection_manifest_path),
        "selection_summary_path": str(selection_summary_path),
        "batch_summary_path": str(runtime_batch_summary_path),
    }


def _run_final_ingest(*, campaign_root: Path, dry_run: bool) -> dict[str, str]:
    ingest_root = campaign_root / "ingest"
    ingest_root.mkdir(parents=True, exist_ok=True)
    ingestion_artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_root=campaign_root,
        output_root=ingest_root,
    )
    metrics_artifacts = write_emb_34um_dnn_causal_validation_metric_artifacts(
        completed_rows_source=ingestion_artifacts.manifest["completed_rows"],
        output_root=ingest_root,
        rows_filename=INGEST_ROWS_FILENAME,
        report_filename=INGEST_REPORT_FILENAME,
        dry_run=dry_run,
        ensemble_size=EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    )
    return {
        "rows_path": str(metrics_artifacts.rows_path),
        "report_path": str(metrics_artifacts.report_path),
        "completed_rows_path": str(ingestion_artifacts.completed_rows_path),
        "ingestion_manifest_path": str(ingestion_artifacts.manifest_path),
    }


def _infer_replicas(entries: tuple[dict[str, Any], ...]) -> list[int]:
    replicas = sorted({int(entry["replica"]) for entry in entries if int(entry["replica"]) > 0})
    if not replicas:
        replicas = [1]
    return replicas


def _coerce_positive_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 1:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _resolve_active_replicas(
    *,
    design_manifest: Mapping[str, Any],
    entries: tuple[dict[str, Any], ...],
) -> list[int]:
    policy = design_manifest.get("policy", {})
    protocol = design_manifest.get("protocol", {})
    policy_map = policy if isinstance(policy, Mapping) else {}
    protocol_map = protocol if isinstance(protocol, Mapping) else {}
    replicas_from_entries = _infer_replicas(entries)

    raw_count = (
        policy_map.get("active_replicate_count")
        if "active_replicate_count" in policy_map
        else protocol_map.get("active_replicate_count")
    )
    if raw_count is None:
        raw_count = policy_map.get("replicate_count")
    if raw_count is None:
        raw_count = protocol_map.get("replicate_count")
    if raw_count is None:
        return replicas_from_entries

    active_replicate_count = _coerce_positive_int(raw_count, label="active_replicate_count")
    explicit_replicas = policy_map.get("replicates")
    if isinstance(explicit_replicas, list):
        declared_replicas = sorted({int(item) for item in explicit_replicas if int(item) > 0})
        available_replicas = [replica for replica in declared_replicas if replica in set(replicas_from_entries)]
        if not available_replicas:
            available_replicas = declared_replicas
    else:
        available_replicas = replicas_from_entries
    if not available_replicas:
        available_replicas = replicas_from_entries
    active_replicas = available_replicas[:active_replicate_count]
    if not active_replicas:
        raise ValueError("No active replicas available from campaign manifest.")
    return active_replicas


def _filter_entries_for_active_replicas(
    entries: tuple[dict[str, Any], ...],
    *,
    active_replicas: list[int],
) -> tuple[dict[str, Any], ...]:
    active_set = {int(replica) for replica in active_replicas}
    return tuple(
        entry
        for entry in entries
        if int(entry.get("replica", 0)) == 0 or int(entry.get("replica", 0)) in active_set
    )


def _resolve_stage_runtime_controls(design_manifest: Mapping[str, Any]) -> tuple[str, int, int]:
    command_inventory = design_manifest.get("command_inventory", {})
    policy = design_manifest.get("policy", {})
    protocol = design_manifest.get("protocol", {})
    command_inventory_map = command_inventory if isinstance(command_inventory, Mapping) else {}
    policy_map = policy if isinstance(policy, Mapping) else {}
    protocol_map = protocol if isinstance(protocol, Mapping) else {}
    walltime = str(
        command_inventory_map.get(
            "walltime",
            policy_map.get(
                "dpd_walltime_target",
                protocol_map.get("dpd_walltime_target", EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT),
            ),
        )
    )
    concurrent_jobs = _coerce_positive_int(command_inventory_map.get("concurrent_jobs", 30), label="concurrent_jobs")
    retry_limit = _coerce_nonnegative_int(
        command_inventory_map.get("retry_limit", EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT),
        label="retry_limit",
    )
    return walltime, concurrent_jobs, retry_limit


def _infer_cycle_count(entries: tuple[dict[str, Any], ...]) -> int:
    cycles = [int(_entry_step(str(entry["mode"])) or 0) for entry in entries if str(entry["mode"]).startswith(("al-step-", "lhs-step-"))]
    return max(cycles) if cycles else 1


def _group_entries(entries: tuple[dict[str, Any], ...]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["mode"]), []).append(entry)
    for mode in grouped:
        grouped[mode].sort(key=lambda item: (int(item.get("replica", 0)), int(item.get("cycle", 0))))
    return grouped


def build_emb_34um_dnn_causal_validation_controller(
    *,
    campaign_root: Path,
    design_manifest_path: Path | None = None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    execution_mode = _coerce_execution_mode(execution_mode)
    design_manifest_path = Path(design_manifest_path or campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME)
    design_manifest = _load_json(design_manifest_path)
    command_inventory = design_manifest.get("command_inventory", {})
    if not isinstance(command_inventory, Mapping):
        raise ValueError("command_inventory must be a mapping.")

    raw_entries = _coerce_stage_entries(command_inventory.get("entries", []))
    entries = tuple(_entry_record(entry) for entry in raw_entries)
    active_replicas = _resolve_active_replicas(design_manifest=design_manifest, entries=entries)
    entries = _filter_entries_for_active_replicas(entries, active_replicas=active_replicas)
    entries_by_mode = _group_entries(entries)
    replicas = sorted(active_replicas)
    cycle_count = _infer_cycle_count(entries)
    pilot_summary_path = _pilot_validation_summary_path(campaign_root=campaign_root, design_manifest=design_manifest)

    stage_order = ["prepare_design", "pilot_submit", "pilot_verify", "unseen_test_submit", "shared_initial_submit"]
    for step in range(1, cycle_count + 1):
        stage_order.extend(
            [
                f"lhs-step-{step:02d}-submit",
                f"al-step-{step:02d}-train-score-select",
                f"al-step-{step:02d}-render-selected",
                f"al-step-{step:02d}-submit",
            ]
        )
    stage_order.extend(["final_ingest", "final_analyze"])

    prepare_command = _build_prepare_stage_command(
        campaign_root=campaign_root,
        design_manifest=design_manifest,
        execution_mode=execution_mode,
        command_inventory=command_inventory,
        cycle_count=cycle_count,
        active_replicate_count=len(replicas),
    )
    stages: list[dict[str, Any]] = []
    stages.append(
        _build_stage(
            name="prepare_design",
            description="Prepare/render campaign protocol and command inventory.",
            command_type="prepare",
            commands=[prepare_command],
            dependencies=[],
            expected_output_roots=[
                str(campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME),
                str(campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME),
            ],
        )
    )

    pilot_submit = _build_stage_submit_commands(entries_by_mode.get("pilot", []), execution_mode=execution_mode)
    stages.append(
        _build_stage(
            name="pilot_submit",
            description="Submit pilot scheduler arrays.",
            command_type="submission",
            commands=pilot_submit,
            dependencies=["prepare_design"],
            expected_output_roots=[entry["batch_summary_path"] for entry in entries_by_mode.get("pilot", [])],
        )
    )

    stages.append(
        _build_stage(
            name="pilot_verify",
            description="Analyze pilot outputs and emit the pilot validation decision summary.",
            command_type="verification",
            commands=[
                _build_pilot_verify_command(
                    campaign_root=campaign_root,
                    summary_path=pilot_summary_path,
                    execution_mode=execution_mode,
                )
            ],
            dependencies=["pilot_submit"],
            expected_output_roots=[str(pilot_summary_path)],
        )
    )

    unseen_submit = _build_stage_submit_commands(entries_by_mode.get("unseen_test", []), execution_mode=execution_mode)
    stages.append(
        _build_stage(
            name="unseen_test_submit",
            description="Submit unseen test scheduler arrays.",
            command_type="submission",
            commands=unseen_submit,
            dependencies=["pilot_verify"],
            expected_output_roots=[entry["batch_summary_path"] for entry in entries_by_mode.get("unseen_test", [])],
        )
    )

    shared_submit = _build_stage_submit_commands(entries_by_mode.get("shared_initial", []), execution_mode=execution_mode)
    stages.append(
        _build_stage(
            name="shared_initial_submit",
            description="Submit shared initial scheduler arrays.",
            command_type="submission",
            commands=shared_submit,
            dependencies=["pilot_verify"],
            expected_output_roots=[entry["batch_summary_path"] for entry in entries_by_mode.get("shared_initial", [])],
        )
    )

    for step in range(1, cycle_count + 1):
        lhs_mode = f"lhs-step-{step:02d}"
        al_mode = f"al-step-{step:02d}"
        lhs_stage_name = f"lhs-step-{step:02d}-submit"
        al_train_stage_name = f"al-step-{step:02d}-train-score-select"
        al_render_stage_name = f"al-step-{step:02d}-render-selected"
        al_submit_stage_name = f"al-step-{step:02d}-submit"

        lhs_deps = ["shared_initial_submit"] if step == 1 else [f"lhs-step-{step - 1:02d}-submit"]
        stages.append(
            _build_stage(
                name=lhs_stage_name,
                description=f"Submit LHS scheduler arrays for cycle {step}.",
                command_type="submission",
                commands=_build_stage_submit_commands(entries_by_mode.get(lhs_mode, []), execution_mode=execution_mode),
                dependencies=lhs_deps,
                expected_output_roots=[entry["batch_summary_path"] for entry in entries_by_mode.get(lhs_mode, [])],
            )
        )

        al_train_commands: list[str] = []
        expected_train_outputs: list[str] = []
        al_entries = entries_by_mode.get(al_mode, [])
        for entry in al_entries:
            replica = int(entry["replica"])
            candidate_count = int(entry["candidate_count"])
            al_train_commands.append(
                _build_al_inputs_command(
                    campaign_root=campaign_root,
                    replica=replica,
                    step=step,
                    execution_mode=execution_mode,
                )
            )
            al_train_commands.append(
                _build_al_train_score_select_command(
                    campaign_root=campaign_root,
                    replica=replica,
                    step=step,
                    candidate_count=candidate_count,
                    execution_mode=execution_mode,
                )
            )
            expected_train_outputs.extend(
                [
                    str(_selection_training_manifest_path(campaign_root, replica=replica, step=step)),
                    str(_selection_training_report_path(campaign_root, replica=replica, step=step)),
                    str(_selection_timing_report_path(campaign_root, replica=replica, step=step)),
                ]
            )

        train_deps = ["shared_initial_submit"] if step == 1 else [f"al-step-{step - 1:02d}-submit"]
        stages.append(
            _build_stage(
                name=al_train_stage_name,
                description=f"Build completed rows and run DNN train/score/select for cycle {step}.",
                command_type="train_score_select",
                commands=al_train_commands,
                dependencies=train_deps,
                expected_output_roots=expected_train_outputs,
            )
        )

        render_commands: list[str] = []
        render_outputs: list[str] = []
        for replica in replicas:
            render_commands.append(
                _build_al_render_selected_command(
                    campaign_root=campaign_root,
                    replica=replica,
                    step=step,
                    execution_mode=execution_mode,
                )
            )
            stage_root = _stage_root(campaign_root, replica=replica, step=step)
            render_outputs.extend(
                [
                    str(stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME),
                    str(stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME),
                    str(stage_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME),
                ]
            )
        stages.append(
            _build_stage(
                name=al_render_stage_name,
                description=f"Render selected AL candidates for cycle {step}.",
                command_type="render_selected",
                commands=render_commands,
                dependencies=[al_train_stage_name],
                expected_output_roots=render_outputs,
            )
        )

        stages.append(
            _build_stage(
                name=al_submit_stage_name,
                description=f"Submit rendered AL scheduler arrays for cycle {step}.",
                command_type="submission",
                commands=_build_stage_submit_commands(al_entries, execution_mode=execution_mode),
                dependencies=[al_render_stage_name],
                expected_output_roots=[entry["batch_summary_path"] for entry in al_entries],
            )
        )

    ingest_dependencies = ["unseen_test_submit", "shared_initial_submit"]
    ingest_dependencies.extend([f"lhs-step-{step:02d}-submit" for step in range(1, cycle_count + 1)])
    ingest_dependencies.extend([f"al-step-{step:02d}-submit" for step in range(1, cycle_count + 1)])
    stages.append(
        _build_stage(
            name="final_ingest",
            description="Build final DNN causal ingestion rows and audit report.",
            command_type="ingest",
            commands=[_build_final_ingest_command(campaign_root=campaign_root, execution_mode=execution_mode)],
            dependencies=ingest_dependencies,
            expected_output_roots=[
                str(campaign_root / "ingest" / INGEST_ROWS_FILENAME),
                str(campaign_root / "ingest" / INGEST_REPORT_FILENAME),
            ],
        )
    )

    stages.append(
        _build_stage(
            name="final_analyze",
            description="Run final DNN causal AL-vs-LHS analysis.",
            command_type="analysis",
            commands=[_build_final_analyze_command(campaign_root=campaign_root, execution_mode=execution_mode)],
            dependencies=["final_ingest"],
            expected_output_roots=[
                str(campaign_root / "analyze" / EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME),
                str(campaign_root / "analyze" / EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME),
            ],
        )
    )

    controller_manifest = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "render_only": execution_mode == "render-only",
        "dry_run": execution_mode == "dry-run",
        "campaign_root": str(campaign_root),
        "design_manifest_path": str(design_manifest_path),
        "design_command_inventory_path": str(campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME),
        "design_schema_version": str(design_manifest.get("schema_version", "")),
        "stage_order": stage_order,
        "stages": stages,
        "replicas": replicas,
        "active_replicate_count": len(replicas),
        "cycle_count": cycle_count,
        "command_inventory": {
            "count": len(entries),
            "walltime": str(command_inventory.get("walltime", "")),
            "concurrent_jobs": int(command_inventory.get("concurrent_jobs", 0)),
            "retry_limit": int(command_inventory.get("retry_limit", 0)),
            "entries": [dict(entry) for entry in entries],
            "stages": [{"name": stage["name"], "commands": [item["command"] for item in stage["commands"]]} for stage in stages],
        },
        "parallel_arrays": [dict(entry) for entry in entries],
        "pilot_validation": {
            "required": True,
            "stage_name": "pilot_verify",
            "summary_path": str(pilot_summary_path),
        },
        "submission": {
            "submitted": False,
            "submission_commands": [],
            "submit_allowed_only_in_execute_mode": True,
        },
    }

    manifest_path = campaign_root / CONTROLLER_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(controller_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest_path": manifest_path, "manifest": controller_manifest}


def _load_controller_manifest(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / CONTROLLER_MANIFEST_FILENAME
    payload = _load_json(path)
    if not isinstance(payload.get("stages"), list):
        raise ValueError(f"Controller manifest is malformed: {path}")
    return payload


def _dispatch_stage_commands(*, stage_action: str, campaign_root: Path, execution_mode: str) -> list[str]:
    if stage_action == "al-build-inputs":
        raise ValueError("al-build-inputs requires --replica and --cycle and is dispatched directly.")
    if stage_action == "al-render-selected":
        raise ValueError("al-render-selected requires --replica and --cycle and is dispatched directly.")
    if stage_action == "final-ingest":
        return [_build_final_ingest_command(campaign_root=campaign_root, execution_mode=execution_mode)]

    manifest = _load_controller_manifest(campaign_root)
    for stage in manifest["stages"]:
        if not isinstance(stage, Mapping):
            continue
        if str(stage.get("name", "")) != stage_action:
            continue
        commands = [str(item.get("command", "")) for item in stage.get("commands", []) if isinstance(item, Mapping)]
        return [_replace_execution_mode(command, execution_mode=execution_mode) for command in commands]
    raise ValueError(f"Unknown stage-action {stage_action!r}.")


def _dispatch_internal_stage(
    *,
    stage_action: str,
    campaign_root: Path,
    replica: int,
    cycle: int,
    execution_mode: str,
) -> dict[str, Any]:
    if stage_action == "al-build-inputs":
        return _write_al_selection_inputs(campaign_root=campaign_root, replica=replica, step=cycle)
    if stage_action == "al-render-selected":
        return _render_selected_candidates_for_stage(
            campaign_root=campaign_root,
            replica=replica,
            step=cycle,
            execution_mode=execution_mode,
        )
    raise ValueError(f"Unsupported internal stage-action {stage_action!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--design-manifest")
    parser.add_argument("--execution-mode", default=os.environ.get("EXECUTION_MODE", DEFAULT_EXECUTION_MODE))
    parser.add_argument("--stage-action", default=None)
    parser.add_argument("--submit", action="store_true", help="Run generated commands (execute mode only).")
    parser.add_argument("--replica", type=int, default=0, help="Internal stage argument.")
    parser.add_argument("--cycle", type=int, default=0, help="Internal stage argument.")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = build_parser().parse_args(argv)
    campaign_root = Path(args.campaign_root)
    execution_mode = _coerce_execution_mode(str(args.execution_mode))

    if args.stage_action:
        if args.stage_action in {"al-build-inputs", "al-render-selected"}:
            if args.replica < 1 or args.cycle < 1:
                raise ValueError(f"{args.stage_action} requires --replica >= 1 and --cycle >= 1.")
            payload = _dispatch_internal_stage(
                stage_action=args.stage_action,
                campaign_root=campaign_root,
                replica=int(args.replica),
                cycle=int(args.cycle),
                execution_mode=execution_mode,
            )
            print(json.dumps(payload, sort_keys=True))
            return 0
        if args.stage_action == "final-ingest":
            if execution_mode == "execute":
                command = _build_final_ingest_command(campaign_root=campaign_root, execution_mode=execution_mode)
                if args.submit:
                    subprocess.run(command, shell=True, check=True)
                print(command)
            else:
                payload = _run_final_ingest(
                    campaign_root=campaign_root,
                    dry_run=True,
                )
                print(json.dumps(payload, sort_keys=True))
            return 0

        commands = _dispatch_stage_commands(
            stage_action=str(args.stage_action),
            campaign_root=campaign_root,
            execution_mode=execution_mode,
        )
        if args.submit:
            if execution_mode != "execute":
                raise ValueError("--submit requires --execution-mode execute.")
            for command in commands:
                subprocess.run(command, shell=True, check=True)
        for command in commands:
            print(command)
        return 0

    result = build_emb_34um_dnn_causal_validation_controller(
        campaign_root=campaign_root,
        design_manifest_path=Path(args.design_manifest) if args.design_manifest else None,
        execution_mode=execution_mode,
    )
    if args.submit:
        if execution_mode != "execute":
            raise ValueError("--submit requires --execution-mode execute.")
        stage_commands = [
            command["command"]
            for stage in result["manifest"]["stages"]
            for command in stage.get("commands", [])
            if isinstance(command, Mapping) and command.get("command")
        ]
        for command in stage_commands:
            subprocess.run(command, shell=True, check=True)
        result["manifest"]["submission"]["submitted"] = True
        result["manifest"]["submission"]["submission_commands"] = stage_commands
        manifest_path = Path(result["manifest_path"])
        manifest_path.write_text(json.dumps(result["manifest"], indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
