#!/usr/bin/env python3
"""Build and dispatch command inventory for EMB 3.4um causal validation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _repo_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meso_uq.active_learning.emb_34um_causal_validation_design import (  # noqa: E402
    EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
    EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED,
    EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION as _DESIGN_SCHEMA_VERSION,
    EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
)

from meso_uq.active_learning import emb_34um_causal_validation_design as _causal_design  # noqa: E402
from meso_uq.active_learning.emb_34um_dpd_adapter import EMB_34UM_RUNTIME_FINGERPRINT  # noqa: E402


CONTROLLER_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_causal_validation_controller.v3"
CONTROLLER_MANIFEST_FILE = "emb_34um_causal_validation_controller_manifest.json"

DEFAULT_SCRATCH_ROOT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_causal_validation"
)
DEFAULT_VAULT_ROOT = Path(
    "/home/it4i-bbenvegnen/knowledge/vault/07 Sessions/UQ_DPD/assets/active_learning_emb_34um_causal_validation"
)
DEFAULT_FORCE_GRID_DATA = Path("emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat")
DEFAULT_WALLTIME = "01:00:00"
DEFAULT_CONCURRENT_JOBS = 30
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RUN_ID_PREFIX = "emb-34um-causal-validation"
RANDOM_SEED_COUNT_ISSUES = tuple(f"MES-{index}" for index in range(219, 229))

BATCH_SUMMARY_FILENAME = EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
BATCH_MANIFEST_FILENAME = "dpd_sampling_batch_manifest.json"
SELECTION_MANIFEST_FILENAME = EMB_34UM_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME
SELECTION_BATCH_SUMMARY_FILENAME = EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME
REPLACEMENT_MANIFEST_FILENAME = "emb_34um_causal_validation_replacement_manifest.json"
REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_causal_validation_replacement_batch_summary.json"
LHS_REPLACEMENT_MANIFEST_FILENAME = "emb_34um_causal_validation_lhs_replacement_manifest.json"
LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_causal_validation_lhs_replacement_batch_summary.json"
INGEST_MANIFEST_FILENAME = "emb_34um_causal_validation_ingest_manifest.json"
INGEST_REPORT_FILENAME = "emb_34um_causal_validation_ingest_report.json"
INGEST_SUMMARY_FILENAME = "emb_34um_causal_validation_ingestion_summary.csv"
INGEST_REPLACEMENT_PLAN_FILENAME = "emb_34um_causal_validation_replacement_plan.json"
ANALYSIS_REPORT_FILENAME = "emb_34um_causal_validation_report.json"
ANALYSIS_SUMMARY_FILENAME = "emb_34um_causal_validation_summary.csv"
ANALYSIS_OUTPUT_FILENAMES = (
    ANALYSIS_REPORT_FILENAME,
    ANALYSIS_SUMMARY_FILENAME,
    "emb_34um_causal_validation_learning_curves.png",
    "emb_34um_causal_validation_learning_curves.png.json",
    "emb_34um_causal_validation_final_paired_delta_ci.png",
    "emb_34um_causal_validation_final_paired_delta_ci.png.json",
    "emb_34um_causal_validation_step_delta_ci.png",
    "emb_34um_causal_validation_step_delta_ci.png.json",
    "emb_34um_causal_validation_added_samples_by_step.png",
    "emb_34um_causal_validation_added_samples_by_step.png.json",
    "emb_34um_causal_validation_runtime_replacement_diagnostics.png",
    "emb_34um_causal_validation_runtime_replacement_diagnostics.png.json",
    "emb_34um_causal_validation_acquisition_selection_diagnostics.png",
    "emb_34um_causal_validation_acquisition_selection_diagnostics.png.json",
)
ESCALATION_PLAN_FILENAME = "emb_34um_causal_validation_escalation_plan.json"

CANONICAL_STAGES = (
    "render_protocol",
    "shared_initial",
    "validation",
    *[
        item
        for step in range(1, EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS + 1)
        for item in (
            f"al-step-{step:02d}-select-render",
            f"al-step-{step:02d}-submit",
            f"lhs-step-{step:02d}-submit",
        )
    ],
    "ingest",
    "analyze",
    "escalation",
)


def _load_prepare_module():
    prepare_script = (
        _REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "prepare_emb_34um_causal_validation.py"
    )
    spec = importlib.util.spec_from_file_location("_prepare_emb_34um_causal_validation", prepare_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {prepare_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prepare = _load_prepare_module()


_BASE_ACTION_ALIASES = {
    "render": "render_protocol",
    "render-protocol": "render_protocol",
    "render_protocol": "render_protocol",
    "shared": "shared_initial",
    "shared-initial": "shared_initial",
    "shared_initial": "shared_initial",
    "val": "validation",
    "validation": "validation",
    "ingest": "ingest",
    "an": "analyze",
    "analyze": "analyze",
    "escalate": "escalation",
    "escalation": "escalation",
}

_STEP_ACTION_RE = re.compile(r"^(?P<family>al|lhs)-(?:step-)?(?P<step>\d+)(?:-(?P<suffix>submit|select-render))?$")


def _python_executable() -> str:
    return shlex.quote(sys.executable)


def _quoted(value: str | Path | int | float | bool | None) -> str:
    return shlex.quote(str(value))


def _coerce_positive_int(value: int, *, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _coerce_nonnegative_int(value: int, *, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _coerce_seed_count(seed_count: int) -> int:
    seed_count = _coerce_positive_int(seed_count, label="seed_count")
    if seed_count < len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS):
        raise ValueError(
            "seed_count must include all required primary seeds 1, 2, and 3. "
            f"Minimum is {len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS)}."
        )
    if seed_count > EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS:
        raise ValueError(
            f"seed_count must be in [3, {EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS}], got {seed_count}."
        )
    return seed_count


def _coerce_walltime(value: str) -> str:
    if value.count(":") != 2 or not value:
        raise ValueError("walltime must be in HH:MM:SS format.")
    return value


def _coerce_stage_action(action: str, *, step_count: int | None = None) -> str:
    if not isinstance(action, str):
        raise ValueError("stage-action must be a non-empty string.")
    normalized = action.strip().lower().replace("_", "-")
    canonical = _BASE_ACTION_ALIASES.get(normalized)
    if canonical is None:
        match = _STEP_ACTION_RE.fullmatch(normalized)
        if match is None:
            raise ValueError(f"Unknown stage-action {action!r}.")
        family = match.group("family")
        step = _coerce_positive_int(int(match.group("step")), label="stage-action step")
        if step_count is not None and step > step_count:
            raise ValueError(f"stage-action step {step} exceeds configured step-count {step_count}.")
        suffix = match.group("suffix")
        if family == "lhs" and suffix == "select-render":
            raise ValueError(f"Unknown stage-action {action!r}.")
        if family == "al" and suffix == "select-render":
            canonical = _al_select_stage_name(step)
        else:
            canonical = _al_submit_stage_name(step) if family == "al" else _lhs_submit_stage_name(step)
    return canonical


def _al_select_stage_name(step: int) -> str:
    return f"al-step-{step:02d}-select-render"


def _al_submit_stage_name(step: int) -> str:
    return f"al-step-{step:02d}-submit"


def _lhs_submit_stage_name(step: int) -> str:
    return f"lhs-step-{step:02d}-submit"


def _is_al_select_stage(stage_name: str) -> bool:
    return stage_name.startswith("al-step-") and stage_name.endswith("-select-render")


def _is_al_submit_stage(stage_name: str) -> bool:
    return stage_name.startswith("al-step-") and stage_name.endswith("-submit")


def _is_lhs_submit_stage(stage_name: str) -> bool:
    return stage_name.startswith("lhs-step-") and stage_name.endswith("-submit")


def _extract_step_from_stage(stage_name: str) -> int:
    tokens = stage_name.split("-")
    if len(tokens) < 3:
        raise ValueError(f"Malformed stage name: {stage_name!r}")
    try:
        return int(tokens[2])
    except ValueError as exc:
        raise ValueError(f"Malformed stage name: {stage_name!r}") from exc


def _stage_to_mode(stage_name: str) -> str:
    if _is_al_submit_stage(stage_name):
        return f"al-step-{_extract_step_from_stage(stage_name):02d}"
    if _is_lhs_submit_stage(stage_name):
        return f"lhs-step-{_extract_step_from_stage(stage_name):02d}"
    return stage_name


def _stage_batch_stages(step_count: int) -> tuple[str, ...]:
    return tuple(
        [
            "shared_initial",
            "validation",
            *[
                item
                for step in range(1, step_count + 1)
                for item in (f"al-step-{step:02d}", f"lhs-step-{step:02d}")
            ],
        ]
    )


def _stage_order(step_count: int) -> list[str]:
    if step_count < EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS:
        raise ValueError(
            f"step_count must be at least {EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS}, got {step_count}."
        )
    order = ["render_protocol", "shared_initial", "validation"]
    for step in range(1, step_count + 1):
        order.extend(
            [
                _al_select_stage_name(step),
                _al_submit_stage_name(step),
                _lhs_submit_stage_name(step),
            ]
        )
    order.extend(["ingest", "analyze", "escalation"])
    return order


def _seed_roots(campaign_root: Path, seed_count: int, step_count: int) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    for mode in _stage_batch_stages(step_count):
        roots: list[Path] = []
        for seed in range(1, seed_count + 1):
            roots.append(campaign_root / f"seed-{seed:03d}" / mode)
        outputs[mode] = roots
    return outputs


def _selection_roots(campaign_root: Path, seed_count: int, step_count: int) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    for step in range(1, step_count + 1):
        stage_name = _al_select_stage_name(step)
        roots: list[Path] = []
        for seed in range(1, seed_count + 1):
            roots.append(campaign_root / f"seed-{seed:03d}" / f"al-step-{step:02d}")
        outputs[stage_name] = roots
    return outputs


def _expected_output_roots(*, campaign_root: Path, seed_count: int, step_count: int) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {
        "render_protocol": [
            str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME),
            str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME),
            str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME),
            str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME),
        ],
        "ingest": [
            str(campaign_root / "ingest" / INGEST_MANIFEST_FILENAME),
            str(campaign_root / "ingest" / INGEST_REPORT_FILENAME),
            str(campaign_root / "ingest" / INGEST_SUMMARY_FILENAME),
        ],
        "analyze": [
            str(campaign_root / "analyze" / filename)
            for filename in ANALYSIS_OUTPUT_FILENAMES
        ],
        "escalation": [str(campaign_root / "escalation" / ESCALATION_PLAN_FILENAME)],
    }
    for mode, mode_roots in _seed_roots(campaign_root, seed_count=seed_count, step_count=step_count).items():
        stage_key = mode
        if mode.startswith("al-step-"):
            stage_key = _al_submit_stage_name(int(mode.split("-")[2]))
        elif mode.startswith("lhs-step-"):
            stage_key = _lhs_submit_stage_name(int(mode.split("-")[2]))
        manifests: list[str] = []
        for batch_root in mode_roots:
            manifests.append(str(batch_root / BATCH_MANIFEST_FILENAME))
            manifests.append(str(batch_root / BATCH_SUMMARY_FILENAME))
        outputs[stage_key] = manifests
    for stage_name, mode_roots in _selection_roots(campaign_root, seed_count=seed_count, step_count=step_count).items():
        manifests: list[str] = []
        for batch_root in mode_roots:
            manifests.append(str(batch_root / SELECTION_MANIFEST_FILENAME))
            manifests.append(str(batch_root / SELECTION_BATCH_SUMMARY_FILENAME))
        outputs[stage_name] = manifests
    return outputs


def _build_design_manifest(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    seed_count: int,
    step_count: int,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    include_plot_requirements: bool,
) -> dict[str, Any]:
    result = _prepare.prepare_emb_34um_causal_validation(
        timestamp=timestamp,
        scratch_root=Path(scratch_root),
        vault_root=Path(vault_root),
        force_grid_path=force_grid_path,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        run_id_prefix=run_id_prefix,
        step_count=step_count,
        seed_count=seed_count,
        include_plot_requirements=include_plot_requirements,
    )
    return result["manifest"]


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_mapping(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(payload)


def _runtime_fingerprint_with_defaults(source: Mapping[str, Any] | None) -> dict[str, Any]:
    return {**dict(EMB_34UM_RUNTIME_FINGERPRINT), **dict(source or {})}


def _coerce_candidate_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reserve candidate missing numeric {label!r} field") from exc


def _coerce_candidate_id(value: Any, *, label: str) -> str:
    candidate_id = str(value).strip()
    if not candidate_id:
        raise ValueError(f"{label} must be a non-empty candidate id.")
    return candidate_id


def _coerce_replacement_output_root(value: Any, *, label: str) -> Path:
    path = Path(str(value).strip())
    if not str(path):
        raise ValueError(f"{label} must be a non-empty output_root.")
    return path


def _coerce_reserve_order(payload: Mapping[str, Any]) -> tuple[int, int, int, str, int]:
    selection_order = payload.get("selection_order", 10**9)
    selection_pool_rank = payload.get("selection_pool_rank", 10**9)
    order = payload.get("order", 10**9)
    candidate_id = str(payload.get("candidate_id", ""))
    fallback_index = payload.get("fallback_index", 10**9)
    try:
        return (
            int(selection_order),
            int(selection_pool_rank),
            int(order),
            candidate_id,
            int(fallback_index),
        )
    except (TypeError, ValueError):
        return (10**9, 10**9, 10**9, candidate_id, 10**9)


def _coerce_candidate_record_list(payload: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"selection manifest {label} must be a list.")
    return tuple(_coerce_mapping(item, label=f"{label}[{index}]") for index, item in enumerate(payload))


def _read_replacement_manifest(*, stage_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = stage_root / REPLACEMENT_MANIFEST_FILENAME
    payload = _read_json_dict(manifest_path)
    if payload is None:
        return (
            {
                "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_replacement_manifest.v1",
                "campaign_root": str(stage_root.parent.parent),
                "status": "replacement_not_rendered",
                "seed": 0,
                "step": 0,
                "stage": "",
                "replacement_records": [],
                "replacement_batches": [],
            },
            [],
            [],
        )

    manifest_payload = _coerce_mapping(payload, label="replacement manifest")
    records_payload = manifest_payload.get("replacement_records", [])
    batches_payload = manifest_payload.get("replacement_batches", [])
    records = (
        tuple(_coerce_mapping(item, label=f"replacement_records[{index}]") for index, item in enumerate(records_payload))
        if records_payload
        else ()
    )
    batches = (
        tuple(_coerce_mapping(item, label=f"replacement_batches[{index}]") for index, item in enumerate(batches_payload))
        if batches_payload
        else ()
    )
    return manifest_payload, list(records), list(batches)


def _read_replacement_batch_summary(*, summary_path: Path) -> list[dict[str, Any]]:
    payload = _read_json_dict(summary_path)
    if payload is None:
        raise ValueError(f"Replacement batch summary is missing or invalid: {summary_path}")
    payload = _coerce_mapping(payload, label="replacement batch summary")
    status = str(payload.get("status", "")).strip()
    if status != "replacement_rendered":
        raise ValueError(f"Replacement batch summary has non-complete status: {summary_path} -> {status!r}.")
    records_payload = payload.get("replacement_records", [])
    if not isinstance(records_payload, list):
        raise ValueError(f"Replacement batch summary missing replacement_records: {summary_path}")
    return [
        _coerce_mapping(item, label="replacement batch record")
        for item in records_payload
    ]


def _load_replacement_records_for_stage(*, stage_root: Path) -> tuple[dict[str, Any], ...]:
    _, manifest_records, manifest_batches = _read_replacement_manifest(stage_root=stage_root)
    if manifest_batches:
        records: list[dict[str, Any]] = []
        for raw_batch in manifest_batches:
            if not isinstance(raw_batch, Mapping):
                raise ValueError(f"Malformed replacement batch metadata in {stage_root / REPLACEMENT_MANIFEST_FILENAME}")
            summary_path = Path(str(raw_batch.get("replacement_batch_summary_path", "")).strip())
            if not summary_path.is_file():
                raise ValueError(f"Replacement batch summary is missing for stage {stage_root}: {summary_path}")
            records.extend(_read_replacement_batch_summary(summary_path=summary_path))
        return tuple(records)

    if not manifest_records:
        return ()
    return tuple(manifest_records)


def _parse_command_inventory(command_inventory: dict[str, Any], *, step_count: int) -> dict[str, list[dict[str, Any]]]:
    entries = command_inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("command_inventory must contain a list.")
    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in _stage_batch_stages(step_count)}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("command_inventory entry must be a mapping.")
        mode = str(entry.get("mode", "")).strip()
        if mode in grouped:
            grouped[mode].append(entry)
    return grouped


def _replace_execution_mode(command: str, *, execution_mode: str) -> str:
    if "EXECUTION_MODE=render-only" in command:
        command = command.replace("EXECUTION_MODE=render-only", f"EXECUTION_MODE={execution_mode}")
    elif "EXECUTION_MODE=execute" in command:
        command = command.replace("EXECUTION_MODE=execute", f"EXECUTION_MODE={execution_mode}")
    else:
        command = f"EXECUTION_MODE={execution_mode} {command}"
    return command


def _append_batch_candidate_count(command: str, candidate_count: int) -> str:
    if "--batch-candidate-count" in command:
        return command
    return f"{command} --batch-candidate-count {int(candidate_count)}"


def _replace_sbatch_walltime(command: str, *, walltime: str | None) -> str:
    if walltime is None:
        return command
    desired = _coerce_walltime(str(walltime))
    if "--time=" in command:
        return re.sub(r"--time=\S+", f"--time={desired}", command, count=1)
    return command.replace("sbatch --parsable", f"sbatch --parsable --time={desired}", 1)


def _build_stage_command(entry: dict[str, Any], *, execution_mode: str, walltime: str | None = None) -> str:
    candidate_count = int(entry["candidate_count"])
    base = _replace_execution_mode(str(entry["command"]), execution_mode=execution_mode)
    base = _replace_sbatch_walltime(base, walltime=walltime)
    return _append_batch_candidate_count(base, candidate_count=candidate_count)


def _build_batch_commands(
    *,
    grouped_entries: dict[str, list[dict[str, Any]]],
    step_count: int,
    execution_mode: str,
) -> dict[str, list[str]]:
    commands_by_stage: dict[str, list[str]] = {}
    for mode in _stage_batch_stages(step_count):
        stage_name = mode
        if mode.startswith("al-step-"):
            stage_name = _al_submit_stage_name(int(mode.split("-")[2]))
        elif mode.startswith("lhs-step-"):
            stage_name = _lhs_submit_stage_name(int(mode.split("-")[2]))
        commands_by_stage[stage_name] = [
            _build_stage_command(entry, execution_mode=execution_mode) for entry in grouped_entries.get(mode, [])
        ]
    return commands_by_stage


def _build_render_protocol_command(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    seed_count: int,
    step_count: int,
    execution_mode: str,
) -> str:
    prepare_script = (
        _REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "prepare_emb_34um_causal_validation.py"
    )
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_executable(),
            _quoted(prepare_script),
            "--timestamp",
            _quoted(timestamp),
            "--scratch-root",
            _quoted(scratch_root),
            "--vault-root",
            _quoted(vault_root),
            "--force-grid",
            _quoted(force_grid_path),
            "--walltime",
            _quoted(walltime),
            "--concurrent-jobs",
            str(concurrent_jobs),
            "--retry-limit",
            str(retry_limit),
            "--run-id-prefix",
            _quoted(run_id_prefix),
            "--step-count",
            str(step_count),
            "--seed-count",
            str(seed_count),
            "--include-coverage-plot-requirements",
        ]
    )


def _build_ingest_command(*, campaign_root: Path, execution_mode: str) -> str:
    campaign_manifest_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME
    output_root = campaign_root / "ingest"
    ingest_script = (
        _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "ingest_emb_34um_causal_validation.py"
    )
    replacement_manifest_path = output_root / INGEST_REPLACEMENT_PLAN_FILENAME
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_executable(),
            _quoted(ingest_script),
            "--campaign-manifest",
            _quoted(campaign_manifest_path),
            "--output-root",
            _quoted(output_root),
            "--replacement-manifest-output",
            _quoted(replacement_manifest_path),
        ]
    )


def _build_analyze_command(*, campaign_root: Path, execution_mode: str) -> str:
    analyze_script = (
        _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "analyze_emb_34um_causal_validation.py"
    )
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_executable(),
            _quoted(analyze_script),
            "--ingestion-manifest",
            _quoted(campaign_root / "ingest" / INGEST_MANIFEST_FILENAME),
            "--output-root",
            _quoted(campaign_root / "analyze"),
        ]
    )


def _build_escalation_command(*, campaign_root: Path, execution_mode: str) -> str:
    output_path = campaign_root / "escalation" / ESCALATION_PLAN_FILENAME
    payload = json.dumps(
        {
            "status": "escalation_not_implemented_in_controller",
            "target": "escalate_from_shortfall_seed_count",
            "campaign_root": str(campaign_root),
        },
        sort_keys=True,
    )
    script = (
        "from pathlib import Path; import json; "
        f"root = Path({_quoted(str(campaign_root / 'escalation'))}); root.mkdir(parents=True, exist_ok=True); "
        f"Path({_quoted(str(output_path))}).write_text(json.dumps({payload}));"
    )
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_executable(),
            "-c",
            _quoted(script),
        ]
    )


def _is_clean_ingestion_audit(*, campaign_root: Path) -> bool:
    audit_path = campaign_root / "ingest" / INGEST_REPORT_FILENAME
    payload = _read_json_dict(audit_path)
    if payload is None:
        return False
    status = str(payload.get("status", "")).lower()
    if status in {"clean", "passed", "pass", "ok", "success"}:
        return True
    if isinstance(payload.get("passed"), bool):
        return bool(payload.get("passed"))
    audit = payload.get("audit")
    if isinstance(audit, dict):
        audit_status = str(audit.get("status", "")).lower()
        if audit_status in {"clean", "passed", "pass", "ok", "success"}:
            return True
    shortfall_count = payload.get("shortfall_count")
    try:
        return int(shortfall_count) <= 0
    except (TypeError, ValueError):
        return False


def _require_clean_ingestion_audit(*, campaign_root: Path, stage: str) -> None:
    if not _is_clean_ingestion_audit(campaign_root=campaign_root):
        raise ValueError(
            f"Downstream {stage} requires a clean ingestion audit at "
            f"{campaign_root / 'ingest' / INGEST_REPORT_FILENAME}."
        )


def _manifest_stage(
    *,
    name: str,
    description: str,
    command_type: str,
    commands: list[str],
    expected_output_roots: list[str],
    acceptance_criteria: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "name": name,
        "description": description,
        "command_type": command_type,
        "commands": [{"command": item} for item in commands],
        "expected_output_roots": expected_output_roots,
        "acceptance_criteria": acceptance_criteria or {},
        "status": "blocked" if blockers else "planned",
    }
    if blockers:
        stage["blockers"] = blockers
    return stage


def _build_al_select_render_command(*, campaign_root: Path, step: int, execution_mode: str) -> str:
    script_path = Path(__file__).resolve()
    return " ".join(
        [
            f"EXECUTION_MODE={_quoted(execution_mode)}",
            _python_executable(),
            _quoted(script_path),
            "--stage-action",
            _quoted(_al_select_stage_name(step)),
            "--campaign-root",
            _quoted(campaign_root),
            "--execute",
            "--run-commands",
        ]
    )


def _load_design_manifest(*, campaign_root: Path) -> dict[str, Any]:
    manifest_payload = _read_json_dict(campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME)
    if manifest_payload is None:
        raise ValueError(
            f"Campaign manifest not found: {campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME}"
        )
    if not isinstance(manifest_payload.get("seeds"), list):
        raise ValueError("Malformed campaign manifest: missing seed payloads.")
    return manifest_payload


def _stage_root_for_seed_step(*, campaign_root: Path, seed: int, step: int) -> Path:
    return campaign_root / f"seed-{seed:03d}" / f"al-step-{step:02d}"


def _batch_summary_output_roots(path: Path) -> tuple[Path, ...]:
    payload = _read_json_dict(path)
    if payload is None:
        raise FileNotFoundError(f"Batch summary is missing or invalid: {path}")
    roots = payload.get("expected_output_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{path} does not contain expected_output_roots.")
    return tuple(Path(str(item)) for item in roots)


def _completed_curve_record(*, output_root: Path, strategy: str, step: int, order: int) -> dict[str, Any]:
    result_path = output_root / "emb_34um_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Completed curve result is missing: {result_path}")
    result = _read_json_dict(result_path)
    if result is None:
        raise ValueError(f"Completed curve result is not a JSON object: {result_path}")
    names = tuple(str(item) for item in result.get("parameter_names", ()))
    values = tuple(float(item) for item in result.get("parameters", ()))
    params = dict(zip(names, values))
    if "ka" not in params or "kb" not in params:
        raise ValueError(f"{result_path} does not expose ka/kb parameter names.")
    force_grid = tuple(float(item) for item in result.get("force_grid", ()))
    reference_curve = tuple(float(item) for item in result.get("vertical_diameter", ()))
    if not force_grid or not reference_curve:
        raise ValueError(f"{result_path} is missing force_grid or vertical_diameter.")
    if len(force_grid) != len(reference_curve):
        raise ValueError(f"{result_path} has mismatched force/curve lengths.")
    return {
        "curve_id": str(result.get("candidate_id") or output_root.name),
        "candidate_id": str(result.get("candidate_id") or output_root.name),
        "strategy": strategy,
        "round": step,
        "order": order,
        "ka": float(params["ka"]),
        "kb": float(params["kb"]),
        "force_grid": force_grid,
        "reference_curve": reference_curve,
    }


def _completed_records_from_batch_summary(
    *,
    summary_path: Path,
    strategy: str,
    step: int,
    include_replacement_records: bool = False,
) -> tuple[dict[str, Any], ...]:
    roots = _batch_summary_output_roots(summary_path)
    if not include_replacement_records:
        return tuple(
            _completed_curve_record(output_root=root, strategy=strategy, step=step, order=order)
            for order, root in enumerate(roots, start=1)
        )

    stage_root = summary_path.parent
    replacement_records = _load_replacement_records_for_stage(stage_root=stage_root)
    expected_replacements: dict[str, dict[str, Any]] = {}
    replacement_keyed_records: dict[tuple[int, int], dict[str, Any]] = {}

    for index, replacement_record in enumerate(replacement_records):
        replacement_candidate_id = _coerce_candidate_id(
            replacement_record.get("replacement_candidate_id", ""),
            label=f"replacement_records[{index}].replacement_candidate_id",
        )
        failed_candidate_id = _coerce_candidate_id(
            replacement_record.get("failed_candidate_id", ""),
            label=f"replacement_records[{index}].failed_candidate_id",
        )
        if failed_candidate_id in expected_replacements:
            raise ValueError(
                f"Replacement manifest for {stage_root} has duplicate entries for failed_candidate_id={failed_candidate_id!r}."
            )
        expected_replacements[failed_candidate_id] = replacement_record

        replacement_key = (
            int(replacement_record.get("replacement_batch_index", 10**9)),
            int(replacement_record.get("replacement_sequence", 10**9)),
        )
        replacement_keyed_records[replacement_key] = replacement_record

    root_names = tuple(root.name for root in roots)
    completed_records: list[dict[str, Any]] = []
    next_order = 1
    for output_root in roots:
        candidate_id = output_root.name
        if candidate_id in expected_replacements:
            continue
        try:
            completed_records.append(
                _completed_curve_record(
                    output_root=output_root,
                    strategy=strategy,
                    step=step,
                    order=next_order,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Missing prior curve result for candidate {candidate_id} in {summary_path}; "
                f"no replacement record was found for this candidate."
            ) from exc
        next_order += 1

    for replacement_key in sorted(replacement_keyed_records):
        replacement_record = replacement_keyed_records[replacement_key]
        failed_candidate_id = str(replacement_record.get("failed_candidate_id", "")).strip()
        if failed_candidate_id not in root_names:
            raise ValueError(
                f"Replacement record references unknown failed_candidate_id={failed_candidate_id!r} for {summary_path}."
            )

        replacement_output_root = _coerce_replacement_output_root(
            replacement_record.get("output_root", ""),
            label=f"replacement_records for {failed_candidate_id}",
        )
        result_path = replacement_output_root / "emb_34um_result.json"
        if not result_path.is_file():
            raise ValueError(
                f"Replacement result is missing for candidate {failed_candidate_id!r} in {summary_path}: {result_path}."
            )

        replacement_record_payload = _coerce_mapping(
            replacement_record,
            label=f"replacement_records for failed_candidate_id={failed_candidate_id}",
        )
        replacement_row = _completed_curve_record(
            output_root=replacement_output_root,
            strategy=strategy,
            step=step,
            order=next_order,
        )
        replacement_row.update(
            {
                "is_replacement": True,
                "replacement_for": failed_candidate_id,
                "replacement_candidate_id": replacement_record_payload["replacement_candidate_id"],
                "replacement_batch_index": int(replacement_record_payload.get("replacement_batch_index", 0)),
                "replacement_sequence": int(replacement_record_payload.get("replacement_sequence", 0)),
            }
        )
        completed_records.append(replacement_row)
        next_order += 1
    return tuple(completed_records)


def _force_grid_from_manifest(manifest: Mapping[str, Any]) -> tuple[float, ...]:
    for seed_payload in manifest.get("seeds", ()):
        if not isinstance(seed_payload, Mapping):
            continue
        for batch_name in ("shared_initial", "validation"):
            batch = seed_payload.get(batch_name)
            if not isinstance(batch, Mapping):
                continue
            records = batch.get("candidate_records")
            if not isinstance(records, list) or not records:
                continue
            force_grid = records[0].get("force_grid")
            if isinstance(force_grid, list) and force_grid:
                return tuple(float(item) for item in force_grid)
    raise ValueError("Campaign manifest does not expose a force grid in static candidate records.")


def _selection_candidates_from_scored_rows(
    *,
    manifest: Mapping[str, Any],
    placeholder: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
    campaign_root: Path,
    seed: int,
    step: int,
    force_grid: tuple[float, ...],
) -> tuple[tuple[Any, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    from meso_uq.active_learning import Candidate
    from meso_uq.active_learning.contracts import candidate_hash

    run_id_prefix = str(manifest["run_id_prefix"])
    stage_root = _stage_root_for_seed_step(campaign_root=campaign_root, seed=seed, step=step)
    vault_root_timestamp = Path(str(manifest["vault_root_timestamp"]))
    selection_seed = int(placeholder.get("selection_seed", 0))
    acquisition_count = int(placeholder.get("acquisition_count", 80))
    exploration_count = int(placeholder.get("exploration_count", 20))
    target_count = acquisition_count + exploration_count

    ordered = [
        dict(row) for row in scored_rows
    ]
    ordered.sort(
        key=lambda item: (
            -float(item.get("acquisition_score", 0.0)),
            -float(item.get("ensemble_disagreement", 0.0)),
            float(item.get("ka")),
            float(item.get("kb")),
        )
    )

    selected_points: list[tuple[float, float]] = []
    selected_rows: list[dict[str, Any]] = []
    for _ in range(min(acquisition_count, len(ordered))):
        item = ordered.pop(0)
        selected_rows.append(item)
        point_ka = float(item["ka"])
        point_kb = float(item["kb"])
        selected_points.append((math.log10(point_ka), math.log10(point_kb)))

    while len(selected_rows) < target_count and ordered:
        best_index = -1
        best_score = (-math.inf, -math.inf)
        for index, item in enumerate(ordered):
            point = (math.log10(float(item["ka"])), math.log10(float(item["kb"])))
            diversity = min(
                math.dist(point, selected_point) for selected_point in selected_points
            )
            acquisition = float(item.get("acquisition_score", 0.0))
            score = (diversity, acquisition)
            if score > best_score:
                best_score = score
                best_index = index

        item = ordered.pop(best_index)
        selected_rows.append(item)
        point_ka = float(item["ka"])
        point_kb = float(item["kb"])
        selected_points.append((math.log10(point_ka), math.log10(point_kb)))

    reserve_rows = ordered

    candidates: list[Candidate] = []
    records: list[dict[str, Any]] = []
    for order, row in enumerate(selected_rows, start=1):
        source = "ensemble_disagreement_diversity" if order <= acquisition_count else "exploration"
        candidate_id = f"{run_id_prefix}-seed{seed:03d}-al-step{step:02d}-c{order:03d}"
        output_root = stage_root / "emb" / candidate_id
        vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)
        candidate = Candidate(
            candidate_id=candidate_id,
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "force_grid": list(force_grid),
                "runtime_fingerprint": _runtime_fingerprint_with_defaults(manifest.get("runtime_fingerprint")),
            },
            metadata={
                "seed": seed,
                "step": step,
                "method": f"al-step-{step:02d}",
                "source": source,
                "selection_source": source,
                "selection_mode": "causal_fresh_only",
                "fresh_only": True,
                "causal_validation_mode": True,
                "selection_seed": selection_seed,
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                "output_root": str(output_root),
                "vault_output_root": str(vault_output_root),
            },
        )
        candidates.append(candidate)
        score = float(row.get("acquisition_score", 0.0))
        disagreement = float(row.get("ensemble_disagreement", 0.0))
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_pool_id": str(row.get("candidate_id", candidate_id)),
                "candidate_hash": candidate_hash(candidate),
                "family": "emb",
                "experiment": "indentation",
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "force_grid": list(force_grid),
                "selection_seed": selection_seed,
                "selection_mode": "causal_fresh_only",
                "fresh_only": True,
                "step": step,
                "method": "al",
                "stage": f"al-step-{step:02d}",
                "seed": seed,
                "order": order,
                "source": source,
                "selection_source": source,
                "acquisition_score": score,
                "ensemble_disagreement": disagreement,
                "selection_score": (score + disagreement) / 2.0,
                "selection_rank": order,
                "output_root": str(output_root),
                "vault_output_root": str(vault_output_root),
            }
        )

    reserve_records: list[dict[str, Any]] = []
    for order, row in enumerate(reserve_rows, start=1):
        row_candidate_id = str(row.get("candidate_id", f"reserve-{seed:03d}-{step:02d}-{order:03d}"))
        reserve_output_root = stage_root / "reserve" / row_candidate_id
        reserve_vault_output_root = vault_root_timestamp / reserve_output_root.relative_to(campaign_root)
        reserve_candidate = Candidate(
            candidate_id=row_candidate_id,
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "force_grid": list(force_grid),
                "runtime_fingerprint": _runtime_fingerprint_with_defaults(manifest.get("runtime_fingerprint")),
            },
            metadata={
                "seed": seed,
                "step": step,
                "method": f"al-step-{step:02d}",
                "source": "reserve",
                "selection_source": "reserve",
                "selection_mode": "causal_fresh_only",
                "selection_order": order,
                "selection_pool_rank": acquisition_count + order,
                "fresh_only": True,
                "causal_validation_mode": True,
                "selection_seed": selection_seed,
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                "output_root": str(reserve_output_root),
                "vault_output_root": str(reserve_vault_output_root),
            },
        )
        score = float(row.get("acquisition_score", 0.0))
        disagreement = float(row.get("ensemble_disagreement", 0.0))
        reserve_records.append(
            {
                "candidate_id": row_candidate_id,
                "candidate_pool_id": str(row.get("candidate_id", row_candidate_id)),
                "candidate_hash": candidate_hash(reserve_candidate),
                "family": "emb",
                "experiment": "indentation",
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "force_grid": list(force_grid),
                "selection_seed": selection_seed,
                "selection_mode": "causal_fresh_only",
                "fresh_only": True,
                "step": step,
                "method": "al",
                "stage": f"al-step-{step:02d}",
                "seed": seed,
                "source": "reserve",
                "selection_source": "reserve",
                "selection_order": order,
                "order": order,
                "acquisition_score": score,
                "ensemble_disagreement": disagreement,
                "selection_score": (score + disagreement) / 2.0,
                "output_root": str(reserve_output_root),
                "vault_output_root": str(reserve_vault_output_root),
            }
        )

    return tuple(candidates), records, reserve_records


def _normalize_failed_candidate_ids(failed_candidate_ids: Sequence[str]) -> tuple[str, ...]:
    ids: list[str] = []
    for raw_id in failed_candidate_ids:
        candidate_id = str(raw_id).strip()
        if not candidate_id:
            continue
        if candidate_id in ids:
            continue
        ids.append(candidate_id)
    if not ids:
        raise ValueError("At least one failed or replaced candidate id is required.")
    return tuple(ids)


def _coerce_lhs_replacement_mode(mode: str) -> str:
    canonical = str(mode).strip().lower().replace("_", "-")
    if canonical not in {"rerun-original", "next-lhs"}:
        raise ValueError("mode must be either 'rerun-original' or 'next-lhs'.")
    return canonical


def _lhs_stage_root(*, campaign_root: Path, seed: int, step: int) -> Path:
    return campaign_root / f"seed-{seed:03d}" / f"lhs-step-{step:02d}"


def _lhs_seed_payload_from_manifest(campaign_manifest: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    seeds = campaign_manifest.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("Malformed campaign manifest: missing seeds for lhs replacement lookup.")
    for payload in seeds:
        if not isinstance(payload, Mapping):
            continue
        try:
            payload_seed = int(payload.get("seed", 0))
        except (TypeError, ValueError):
            continue
        if payload_seed == seed:
            return _coerce_mapping(payload, label=f"seed {seed}")
    raise ValueError(f"Campaign manifest does not include seed {seed}.")


def _lhs_step_payload(
    seed_payload: Mapping[str, Any], *, step: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    lhs_steps = seed_payload.get("lhs_steps")
    if not isinstance(lhs_steps, list):
        raise ValueError(
            f"Malformed seed payload for seed {seed_payload.get('seed')}: missing lhs_steps."
        )
    try:
        payload = lhs_steps[step - 1]
    except IndexError as exc:
        raise ValueError(
            f"Malformed seed payload for seed {seed_payload.get('seed')} missing lhs step {step}."
        ) from exc
    step_payload = _coerce_mapping(payload, label=f"seed {seed_payload.get('seed')} lhs step {step}")
    candidate_records = _coerce_candidate_record_list(
        step_payload.get("candidate_records", []),
        label=f"seed {seed_payload.get('seed')} lhs step {step} candidate_records",
    )
    force_grid_payload = step_payload.get("force_grid")
    if not force_grid_payload:
        force_grid_payload = []
        for record in candidate_records:
            raw_force_grid = record.get("force_grid")
            if isinstance(raw_force_grid, list) and raw_force_grid:
                force_grid_payload = raw_force_grid
                break
    return step_payload, list(candidate_records), list(force_grid_payload) if force_grid_payload else []


def _read_lhs_replacement_manifest(
    *, stage_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = stage_root / LHS_REPLACEMENT_MANIFEST_FILENAME
    payload = _read_json_dict(manifest_path)
    if payload is None:
        return (
            {
                "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_lhs_replacement_manifest.v1",
                "campaign_root": str(stage_root.parent.parent),
                "seed": 0,
                "step": 0,
                "stage": "",
                "status": "replacement_not_rendered",
                "replacement_records": [],
                "replacement_batches": [],
            },
            [],
            [],
        )

    manifest_payload = _coerce_mapping(payload, label="lhs replacement manifest")
    records_payload = manifest_payload.get("replacement_records", [])
    batches_payload = manifest_payload.get("replacement_batches", [])
    records = (
        tuple(_coerce_mapping(item, label=f"replacement_records[{index}]") for index, item in enumerate(records_payload))
        if records_payload
        else ()
    )
    batches = (
        tuple(_coerce_mapping(item, label=f"replacement_batches[{index}]") for index, item in enumerate(batches_payload))
        if batches_payload
        else ()
    )
    return manifest_payload, list(records), list(batches)


def _lhs_replacement_point(*, seed: int, step: int, replacement_index: int, mode: str) -> tuple[float, float]:
    design_index = _causal_design._selection_seed(seed=seed, method="lhs", step=step) + replacement_index
    try:
        point = _causal_design._sample_points(start_index=design_index, count=1)[0]
        return float(point[0]), float(point[1])
    except Exception:
        digest = hashlib.sha256(f"{seed}:{step}:{replacement_index}:{mode}".encode("utf-8")).digest()
        ka_fraction = int.from_bytes(digest[:8], "big") / (2**64)
        kb_fraction = int.from_bytes(digest[8:16], "big") / (2**64)
        return (
            float(_causal_design._from_unit_to_log_space(ka_fraction, _causal_design.EMB_34UM_CAUSAL_VALIDATION_BOUNDS["ka"])),
            float(_causal_design._from_unit_to_log_space(kb_fraction, _causal_design.EMB_34UM_CAUSAL_VALIDATION_BOUNDS["kb"])),
        )


def _build_lhs_replacement_candidates_from_mode(
    *,
    campaign_root: Path,
    seed: int,
    step: int,
    failed_candidate_ids: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    from meso_uq.active_learning import Candidate
    from meso_uq.active_learning.contracts import candidate_hash

    failed_candidate_ids = _normalize_failed_candidate_ids(failed_candidate_ids)
    mode = _coerce_lhs_replacement_mode(mode)
    step = _coerce_positive_int(step, label="step")

    design_manifest = _load_design_manifest(campaign_root=campaign_root)
    design_seed_payload = _lhs_seed_payload_from_manifest(design_manifest, seed=seed)
    _, lhs_candidate_records, fallback_force_grid = _lhs_step_payload(design_seed_payload, step=step)
    run_id_prefix = str(design_manifest.get("run_id_prefix", DEFAULT_RUN_ID_PREFIX))
    vault_root_timestamp = Path(str(design_manifest.get("vault_root_timestamp", campaign_root)))
    selection_manifest_path = _lhs_stage_root(campaign_root=campaign_root, seed=seed, step=step) / SELECTION_MANIFEST_FILENAME
    stage_root = _lhs_stage_root(campaign_root=campaign_root, seed=seed, step=step)

    candidate_lookup = {str(row.get("candidate_id", "")).strip(): row for row in lhs_candidate_records}
    if not candidate_lookup and mode == "rerun-original":
        raise ValueError(f"Seed {seed} step {step} does not expose lhs candidate_records for rerun-original mode.")

    replacement_root = stage_root / "replacement"
    replacement_manifest_payload, prior_records, prior_batches = _read_lhs_replacement_manifest(stage_root=stage_root)
    prior_failed = {
        str(item.get("failed_candidate_id", "")).strip(): item
        for item in prior_records
        if str(item.get("failed_candidate_id", "")).strip()
    }
    used_replacement_ids = {str(item.get("replacement_candidate_id", "")).strip() for item in prior_records}
    used_original_ids = {str(item.get("candidate_id", "")).strip() for item in lhs_candidate_records}

    batch_index = len(prior_batches) + 1
    batch_root = replacement_root / f"batch-{batch_index:03d}"
    batch_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{run_id_prefix}-seed{seed:03d}-lhs-step{step:02d}"
    run_command_payload = _load_design_command_inventory_payload(campaign_root=campaign_root)
    _, walltime, concurrent_jobs, retry_limit = run_command_payload
    replacement_records_payload: list[dict[str, Any]] = []
    candidates: list[Candidate] = []

    for replacement_sequence, failed_candidate_id in enumerate(failed_candidate_ids, start=1):
        failed_candidate_id = str(failed_candidate_id).strip()
        if failed_candidate_id in prior_failed and mode != "next-lhs":
            raise ValueError(f"Failed candidate already has replacement record: {failed_candidate_id!r}.")

        if mode == "rerun-original":
            source_record = candidate_lookup.get(failed_candidate_id)
            if source_record is None:
                raise ValueError(f"Candidate {failed_candidate_id!r} is not in seed {seed}, lhs step {step}.")
            replacement_candidate_id = failed_candidate_id
            ka = _coerce_candidate_float(source_record.get("ka"), "ka")
            kb = _coerce_candidate_float(source_record.get("kb"), "kb")
            force_grid_payload = source_record.get("force_grid", fallback_force_grid)
        else:
            replacement_order = len(lhs_candidate_records) + len(prior_records) + replacement_sequence
            replacement_candidate_id = f"{run_id}-c{replacement_order:03d}"
            if replacement_candidate_id in used_replacement_ids or replacement_candidate_id in used_original_ids:
                raise ValueError(f"Could not allocate deterministic replacement_candidate_id={replacement_candidate_id!r}.")
            ka, kb = _lhs_replacement_point(
                seed=seed,
                step=step,
                replacement_index=replacement_order,
                mode=mode,
            )
            force_grid_payload = fallback_force_grid

        force_grid = tuple(_coerce_candidate_float(item, "force_grid") for item in force_grid_payload)
        output_root = batch_root / "emb" / replacement_candidate_id
        vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)

        metadata: dict[str, Any] = {
            "seed": seed,
            "step": step,
            "method": f"lhs-step-{step:02d}",
            "source": "replacement",
            "selection_source": "reserve" if mode == "next-lhs" else "rerun-original",
            "selection_mode": "causal_fresh_only",
            "fresh_only": True,
            "causal_validation_mode": True,
            "replacement_for": failed_candidate_id,
            "replacement_batch_index": batch_index,
            "replacement_sequence": replacement_sequence,
            "replacement_policy": mode,
            "output_root": str(output_root),
            "vault_output_root": str(vault_output_root),
            "selection_order": replacement_sequence,
            "selection_pool_rank": replacement_sequence,
            "selection_rank": replacement_sequence,
        }
        candidate = Candidate(
            candidate_id=replacement_candidate_id,
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": ka,
                "kb": kb,
                "force_grid": list(force_grid),
                "runtime_fingerprint": _runtime_fingerprint_with_defaults(design_manifest.get("runtime_fingerprint")),
            },
            metadata=metadata,
        )

        candidates.append(candidate)
        replacement_records_payload.append(
            {
                "seed": seed,
                "step": step,
                "failed_candidate_id": failed_candidate_id,
                "replacement_candidate_id": replacement_candidate_id,
                "ka": ka,
                "kb": kb,
                "output_root": str(output_root),
                "vault_output_root": str(vault_output_root),
                "replacement_policy": mode,
                "replacement_candidate_hash": candidate_hash(candidate),
            }
        )

    batch_payload = _prepare._render_batch(
        candidates=tuple(candidates),
        batch_root=batch_root,
        run_id=f"{run_id}-replacement-{batch_index:03d}",
        batch_id=f"lhs-step{step:02d}-replacement-{batch_index:03d}",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )

    batch_summary_path = batch_root / LHS_REPLACEMENT_BATCH_SUMMARY_FILENAME
    batch_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_lhs_replacement_batch_summary.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "seed": seed,
        "step": step,
        "stage": f"lhs-step-{step:02d}",
        "selection_manifest_path": str(selection_manifest_path),
        "replacement_batch_root": str(batch_root),
        "batch_summary_path": str(batch_summary_path),
        "replacement_batch_index": batch_index,
        "replacement_record_count": len(replacement_records_payload),
        "replacement_policy": mode,
        "replacement_records": replacement_records_payload,
        "batch_payload": batch_payload,
    }
    batch_summary_path.write_text(json.dumps(batch_summary, sort_keys=True, indent=2), encoding="utf-8")

    prior_records.extend(replacement_records_payload)
    prior_batches.append(
        {
            "replacement_batch_index": batch_index,
            "replacement_batch_summary_path": str(batch_summary_path),
            "failed_candidate_ids": list(failed_candidate_ids),
            "replacement_record_count": len(replacement_records_payload),
        }
    )

    replacement_manifest_payload["schema_version"] = "meso_uq.active_learning.emb_34um_causal_validation_lhs_replacement_manifest.v1"
    replacement_manifest_payload["status"] = "replacement_rendered"
    replacement_manifest_payload["campaign_root"] = str(campaign_root)
    replacement_manifest_payload["seed"] = seed
    replacement_manifest_payload["step"] = step
    replacement_manifest_payload["stage"] = f"lhs-step-{step:02d}"
    replacement_manifest_payload["selection_manifest_path"] = str(selection_manifest_path)
    replacement_manifest_payload["replacement_policy"] = mode
    replacement_manifest_payload["replacement_records"] = prior_records
    replacement_manifest_payload["replacement_batches"] = prior_batches
    replacement_manifest_payload["replacement_count"] = len(prior_records)

    replacement_manifest_path = stage_root / LHS_REPLACEMENT_MANIFEST_FILENAME
    replacement_manifest_path.write_text(
        json.dumps(replacement_manifest_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    return {
        "replacement_manifest_path": str(replacement_manifest_path),
        "replacement_batch_manifest_payload": batch_summary,
        "replacement_batch_summary_path": str(batch_summary_path),
        "replacement_records": replacement_records_payload,
    }


def _load_design_command_inventory_payload(*, campaign_root: Path) -> tuple[str, str, int, int]:
    manifest = _load_design_manifest(campaign_root=campaign_root)
    command_inventory = _coerce_mapping(manifest.get("command_inventory", {}), label="command_inventory")
    run_id_prefix = str(manifest.get("run_id_prefix", "emb-34um-causal-validation"))
    walltime = str(command_inventory.get("walltime", DEFAULT_WALLTIME))
    concurrent_jobs = int(command_inventory.get("concurrent_jobs", DEFAULT_CONCURRENT_JOBS))
    retry_limit = int(command_inventory.get("retry_limit", DEFAULT_RETRY_LIMIT))
    return run_id_prefix, walltime, concurrent_jobs, retry_limit


def _build_replacement_candidates_from_reserve(*, campaign_root: Path, seed: int, step: int, failed_candidate_ids: Sequence[str]) -> dict[str, Any]:
    from meso_uq.active_learning import Candidate
    from meso_uq.active_learning.contracts import candidate_hash

    failed_candidate_ids = _normalize_failed_candidate_ids(failed_candidate_ids)
    stage_root = _stage_root_for_seed_step(campaign_root=campaign_root, seed=seed, step=step)
    selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
    stage_selection_payload = _read_json_dict(selection_manifest_path)
    if stage_selection_payload is None:
        raise ValueError(f"Selection manifest not found for seed={seed} step={step}: {selection_manifest_path}")
    stage_selection_payload = _coerce_mapping(stage_selection_payload, label="selection manifest")

    reserve_records = _coerce_candidate_record_list(
        stage_selection_payload.get("candidate_reserve", []),
        label="candidate_reserve",
    )
    if not reserve_records:
        raise ValueError(f"No reserve candidates are available for seed={seed} step={step} replacement.")

    replacement_manifest_payload, prior_records, prior_batches = _read_replacement_manifest(stage_root=stage_root)
    used_reserve_ids = {
        str(item.get("replacement_candidate_id", "")).strip()
        for item in prior_records
        if str(item.get("replacement_candidate_id", "")).strip()
    }

    ordered_reserve_records = sorted(
        ((row, index) for index, row in enumerate(reserve_records)),
        key=lambda item: (*_coerce_reserve_order(item[0]), item[1]),
    )
    candidate_pool = [row for row, _ in ordered_reserve_records if str(row.get("candidate_id", "")).strip() not in used_reserve_ids]

    required_count = len(failed_candidate_ids)
    if len(candidate_pool) < required_count:
        raise ValueError(
            f"No unused reserve candidates remain for seed={seed} step={step}: "
            f"requested {required_count}, available {len(candidate_pool)}."
        )

    selected_rows = candidate_pool[:required_count]
    run_id_prefix, walltime, concurrent_jobs, retry_limit = _load_design_command_inventory_payload(campaign_root=campaign_root)
    design_manifest = _load_design_manifest(campaign_root=campaign_root)
    vault_root_timestamp = Path(str(design_manifest.get("vault_root_timestamp", campaign_root)))
    runtime_fingerprint = _runtime_fingerprint_with_defaults(design_manifest.get("runtime_fingerprint"))

    replacement_root = stage_root / "replacement"
    batch_index = len(prior_batches) + 1
    batch_root = replacement_root / f"batch-{batch_index:03d}"
    batch_root.mkdir(parents=True, exist_ok=True)

    replacement_batch_records: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    for replacement_index, (failed_candidate_id, reserve_row) in enumerate(zip(failed_candidate_ids, selected_rows), start=1):
        replacement_candidate_id = str(reserve_row.get("candidate_id", "")).strip()
        if not replacement_candidate_id:
            raise ValueError(f"candidate_reserve[{replacement_index - 1}] is missing candidate_id.")
        if replacement_candidate_id in used_reserve_ids:
            raise ValueError(f"Internal reserve selection collision for candidate_id={replacement_candidate_id!r}.")

        ka = _coerce_candidate_float(reserve_row.get("ka"), "ka")
        kb = _coerce_candidate_float(reserve_row.get("kb"), "kb")
        force_grid_payload = reserve_row.get("force_grid")
        if not isinstance(force_grid_payload, list) or not force_grid_payload:
            raise ValueError(f"candidate_reserve[{replacement_candidate_id}] is missing force_grid")
        force_grid = tuple(_coerce_candidate_float(item, "force_grid") for item in force_grid_payload)

        replacement_output_root = batch_root / "emb" / replacement_candidate_id
        vault_output_root = vault_root_timestamp / replacement_output_root.relative_to(campaign_root)
        replacement_candidates_metadata: dict[str, Any] = {
            "seed": seed,
            "step": step,
            "method": f"al-step-{step:02d}",
            "source": "replacement",
            "selection_source": "reserve",
            "selection_mode": "causal_fresh_only",
            "fresh_only": True,
            "causal_validation_mode": True,
            "selection_seed": int(reserve_row.get("selection_seed", step * 100 + seed)),
            "selection_order": int(reserve_row.get("selection_order", replacement_index)),
            "selection_pool_rank": int(reserve_row.get("selection_pool_rank", replacement_index)),
            "replacement_for": failed_candidate_id,
            "replacement_sequence": replacement_index,
            "replacement_batch_index": batch_index,
            "candidate_pool_id": str(reserve_row.get("candidate_pool_id", replacement_candidate_id)),
            "acquisition_score": float(reserve_row.get("acquisition_score", 0.0)),
            "ensemble_disagreement": float(reserve_row.get("ensemble_disagreement", 0.0)),
            "output_root": str(replacement_output_root),
            "vault_output_root": str(vault_output_root),
        }
        replacement_candidate = Candidate(
            candidate_id=replacement_candidate_id,
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": ka,
                "kb": kb,
                "force_grid": list(force_grid),
                "runtime_fingerprint": dict(runtime_fingerprint),
            },
            metadata=replacement_candidates_metadata,
        )
        candidates.append(replacement_candidate)

        replacement_batch_records.append(
            {
                "replacement_batch_index": batch_index,
                "replacement_sequence": replacement_index,
                "failed_candidate_id": failed_candidate_id,
                "replacement_candidate_id": replacement_candidate_id,
                "replacement_candidate_hash": candidate_hash(replacement_candidate),
                "candidate_pool_id": str(reserve_row.get("candidate_pool_id", replacement_candidate_id)),
                "selection_order": int(reserve_row.get("selection_order", replacement_index)),
                "selection_pool_rank": int(reserve_row.get("selection_pool_rank", replacement_index)),
                "selection_rank": int(reserve_row.get("order", replacement_index)),
                "force_grid": list(force_grid),
                "ka": ka,
                "kb": kb,
                "output_root": str(replacement_output_root),
                "vault_output_root": str(vault_output_root),
            }
        )

    batch_payload = _prepare._render_batch(
        candidates=tuple(candidates),
        batch_root=batch_root,
        run_id=f"{run_id_prefix}-seed{seed:03d}-al-step{step:02d}-replacement-{batch_index:03d}",
        batch_id=f"al-step-{step:02d}-replacement-{batch_index:03d}",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )

    batch_summary_path = batch_root / REPLACEMENT_BATCH_SUMMARY_FILENAME
    replacement_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_replacement_batch_summary.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "seed": seed,
        "step": step,
        "stage": f"al-step-{step:02d}",
        "selection_manifest_path": str(selection_manifest_path),
        "replacement_batch_root": str(batch_root),
        "batch_summary_path": str(batch_summary_path),
        "replacement_batch_index": batch_index,
        "replacement_record_count": len(replacement_batch_records),
        "failed_candidate_ids": list(failed_candidate_ids),
        "replacement_records": replacement_batch_records,
        "batch_payload": batch_payload,
    }
    batch_summary_path.write_text(json.dumps(replacement_summary, sort_keys=True, indent=2), encoding="utf-8")

    replacement_batch_manifest = {
        "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_replacement_manifest.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "seed": seed,
        "step": step,
        "stage": f"al-step-{step:02d}",
        "selection_manifest_path": str(selection_manifest_path),
        "batch_summary_path": str(batch_summary_path),
        "batch_root": str(batch_root),
        "replacement_records": replacement_batch_records,
        "batch_payload": batch_payload,
    }
    batch_manifest_path = batch_root / REPLACEMENT_MANIFEST_FILENAME
    batch_manifest_path.write_text(json.dumps(replacement_batch_manifest, sort_keys=True, indent=2), encoding="utf-8")

    prior_records.extend(replacement_batch_records)
    prior_batches.append(
        {
            "replacement_batch_index": batch_index,
            "replacement_batch_manifest_path": str(batch_manifest_path),
            "replacement_batch_summary_path": str(batch_summary_path),
            "failed_candidate_ids": list(failed_candidate_ids),
            "replacement_record_count": len(replacement_batch_records),
        }
    )
    replacement_manifest_payload["schema_version"] = "meso_uq.active_learning.emb_34um_causal_validation_replacement_manifest.v1"
    replacement_manifest_payload["status"] = "replacement_rendered"
    replacement_manifest_payload["campaign_root"] = str(campaign_root)
    replacement_manifest_payload["seed"] = seed
    replacement_manifest_payload["step"] = step
    replacement_manifest_payload["stage"] = f"al-step-{step:02d}"
    replacement_manifest_payload["selection_manifest_path"] = str(selection_manifest_path)
    replacement_manifest_payload["replacement_records"] = prior_records
    replacement_manifest_payload["replacement_batches"] = prior_batches
    replacement_manifest_payload["replacement_count"] = len(prior_records)

    replacement_manifest_path = stage_root / REPLACEMENT_MANIFEST_FILENAME
    replacement_manifest_path.write_text(
        json.dumps(replacement_manifest_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    return {
        "replacement_manifest_path": str(replacement_manifest_path),
        "replacement_batch_manifest_path": str(batch_manifest_path),
        "replacement_batch_summary_path": str(batch_summary_path),
        "replacement_records": replacement_batch_records,
    }


def _selection_summary_paths(*, campaign_root: Path, seed_count: int, step: int) -> list[Path]:
    return [
        _stage_root_for_seed_step(campaign_root=campaign_root, seed=seed, step=step) / SELECTION_BATCH_SUMMARY_FILENAME
        for seed in range(1, seed_count + 1)
    ]


def _missing_selection_summaries(*, campaign_root: Path, seed_count: int, step: int) -> list[Path]:
    return [path for path in _selection_summary_paths(campaign_root=campaign_root, seed_count=seed_count, step=step) if not path.is_file()]


def _batch_summary_paths(*, campaign_root: Path, seed_count: int, step: int) -> list[Path]:
    return [
        _stage_root_for_seed_step(campaign_root=campaign_root, seed=seed, step=step) / BATCH_SUMMARY_FILENAME
        for seed in range(1, seed_count + 1)
    ]


def _missing_batch_summaries(*, campaign_root: Path, seed_count: int, step: int) -> list[Path]:
    return [path for path in _batch_summary_paths(campaign_root=campaign_root, seed_count=seed_count, step=step) if not path.is_file()]


def _seed_count_from_design_manifest(design_manifest: Mapping[str, Any]) -> int:
    seeds = design_manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Malformed campaign manifest: missing seed payloads.")
    return len(seeds)


def _build_al_submit_fallback_commands(
    *,
    campaign_root: Path,
    step: int,
    seed_count: int,
    execution_mode: str,
) -> list[str]:
    design_manifest = _load_design_manifest(campaign_root=campaign_root)
    entries = []
    for entry in design_manifest.get("command_inventory", {}).get("entries", []):
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("mode", "")) == f"al-step-{step:02d}":
            entries.append(entry)

    ordered_entries: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for entry in entries:
        try:
            seed = int(entry.get("seed"))
        except (TypeError, ValueError):
            seed = 0
        if seed <= 0 or seed > seed_count or seed in seen:
            continue
        seen.add(seed)
        ordered_entries.append((seed, entry))

    missing = [seed for seed in range(1, seed_count + 1) if seed not in seen]
    if missing:
        raise ValueError(
            f"{_al_submit_stage_name(step)} requires command inventory entries for all seeds before dispatch; "
            f"missing seeds: {', '.join(str(seed) for seed in sorted(missing))}."
        )

    ordered_entries.sort(key=lambda item: item[0])
    command_inventory = _coerce_mapping(design_manifest.get("command_inventory", {}), label="command_inventory")
    walltime = str(command_inventory.get("walltime", DEFAULT_WALLTIME))
    commands = [
        _build_stage_command(entry, execution_mode=execution_mode, walltime=walltime)
        for _, entry in ordered_entries
    ]
    return commands


def _seed_adaptive_placeholder(seed_payload: dict[str, Any], *, step: int) -> dict[str, Any]:
    al_steps = seed_payload.get("al_steps")
    if not isinstance(al_steps, list) or len(al_steps) < step:
        raise ValueError(f"Malformed campaign manifest: missing al_steps placeholder for step {step}.")
    payload = al_steps[step - 1]
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed campaign manifest: al_steps[{step - 1}] must be a mapping.")
    return payload


def _render_adaptive_selection(*, campaign_root: Path, step: int) -> list[str]:
    manifest = _load_design_manifest(campaign_root=campaign_root)
    missing_by_seed: dict[int, list[str]] = {}
    completed_curve_errors_by_seed: dict[int, str] = {}
    written: list[str] = []
    force_grid = _force_grid_from_manifest(manifest)

    from meso_uq.active_learning.emb_34um_final_gate_design import build_emb_34um_final_gate_design_round
    from meso_uq.active_learning.emb_34um_final_gate_surrogate import (
        EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
        score_emb_34um_candidate_pool,
        train_emb_34um_surrogate_ensemble,
    )

    for seed_payload in manifest["seeds"]:
        seed = int(seed_payload["seed"])
        placeholder = _seed_adaptive_placeholder(seed_payload, step=step)
        prereq_paths = [Path(path) for path in placeholder.get("selection_prerequisites", [])]
        missing = [str(path) for path in prereq_paths if not path.is_file()]
        if missing:
            missing_by_seed[seed] = missing
            continue

        stage_root = _stage_root_for_seed_step(campaign_root=campaign_root, seed=seed, step=step)
        stage_root.mkdir(parents=True, exist_ok=True)
        selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
        selection_summary_path = stage_root / SELECTION_BATCH_SUMMARY_FILENAME
        batch_summary_path = stage_root / BATCH_SUMMARY_FILENAME

        prior_records: list[dict[str, Any]] = []
        shared_summary_path = campaign_root / f"seed-{seed:03d}" / "shared_initial" / BATCH_SUMMARY_FILENAME
        try:
            prior_records.extend(
                _completed_records_from_batch_summary(
                    summary_path=shared_summary_path,
                    strategy="shared_initial",
                    step=0,
                    include_replacement_records=False,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            completed_curve_errors_by_seed[seed] = str(exc)
            continue
        for previous_step in range(1, step):
            previous_summary = _stage_root_for_seed_step(
                campaign_root=campaign_root,
                seed=seed,
                step=previous_step,
            ) / BATCH_SUMMARY_FILENAME
            try:
                prior_records.extend(
                    _completed_records_from_batch_summary(
                        summary_path=previous_summary,
                        strategy="al",
                        step=previous_step,
                        include_replacement_records=True,
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                completed_curve_errors_by_seed[seed] = str(exc)
                break
        if seed in completed_curve_errors_by_seed:
            continue
        if len(prior_records) < EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE + (step - 1) * EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE:
            completed_curve_errors_by_seed[seed] = (
                f"Adaptive select-render for seed={seed} step={step} requires completed shared/prior AL curves; "
                f"found {len(prior_records)} completed records."
            )
            continue

        report = train_emb_34um_surrogate_ensemble(
            records=prior_records,
            validation_records=prior_records,
            seeds=EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
            architecture_names=EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        )
        fit = report["fit"]
        existing_points = tuple((float(item["ka"]), float(item["kb"])) for item in prior_records)
        pool_size = int(placeholder.get("candidate_pool_size", 500))
        acquisition_count = int(placeholder.get("acquisition_count", 80))
        exploration_count = int(placeholder.get("exploration_count", 20))
        target_count = int(placeholder.get("candidate_count", EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE))
        if pool_size != 500:
            raise ValueError(
                f"Adaptive step {step} must use candidate_pool_size=500, got {pool_size}."
            )
        if target_count != EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE:
            raise ValueError(
                f"Adaptive step {step} must select {EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE} candidates, got {target_count}."
            )
        if acquisition_count + exploration_count != target_count:
            raise ValueError(
                f"Adaptive step {step} has acquisition+exploration != target count: "
                f"{acquisition_count}+{exploration_count}!={target_count}."
            )
        if pool_size < target_count:
            raise ValueError(f"Adaptive candidate pool size {pool_size} is smaller than target {target_count}.")
        design = build_emb_34um_final_gate_design_round(
            run_id=str(manifest["run_id_prefix"]),
            round_index=max(2, step + 1),
            seed=int(placeholder.get("selection_seed", seed * 10_000 + step)),
            existing_points=existing_points,
            candidate_pool_size=pool_size,
            candidate_prefix=f"causal-adaptive-step{step:02d}",
            use_log_space=str(manifest.get("sampling", {}).get("parameter_space", "log10")) == "log10",
        )
        pool_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "ka": float(candidate.parameters["ka"]),
                "kb": float(candidate.parameters["kb"]),
            }
            for candidate in design.round_pool
        ]
        scored = score_emb_34um_candidate_pool(
            fit,
            candidate_records=pool_rows,
            force_grid=force_grid,
            existing_points=existing_points,
        )
        candidate_pool_count = len(pool_rows)

        candidates, candidate_records, candidate_reserve = _selection_candidates_from_scored_rows(
            manifest=manifest,
            placeholder=placeholder,
            scored_rows=scored,
            campaign_root=campaign_root,
            seed=seed,
            step=step,
            force_grid=force_grid,
        )
        if len(candidate_records) != target_count:
            raise RuntimeError(
                f"Adaptive selection returned {len(candidate_records)} candidates, expected {target_count}."
            )
        selected_count = len(candidate_records)
        batch_payload = _prepare._render_batch(
            candidates=candidates,
            batch_root=stage_root,
            run_id=f"{manifest['run_id_prefix']}-seed{seed:03d}-al-step{step:02d}",
            batch_id=f"al-step-{step:02d}",
            walltime=str(manifest["command_inventory"].get("walltime", DEFAULT_WALLTIME)),
            concurrent_jobs=int(manifest["command_inventory"].get("concurrent_jobs", DEFAULT_CONCURRENT_JOBS)),
            retry_limit=int(manifest["command_inventory"].get("retry_limit", DEFAULT_RETRY_LIMIT)),
        )

        selection_payload = {
            "schema_version": "meso_uq.active_learning.emb_34um_causal_validation_selection_manifest.v1",
            "status": "selection_rendered",
            "blockers": [],
            "campaign_root": str(campaign_root),
            "seed": seed,
            "step": step,
            "stage": f"al-step-{step:02d}",
            "selection_manifest_path": str(selection_manifest_path),
            "selection_batch_summary_path": str(selection_summary_path),
            "expected_batch_summary_path": str(batch_summary_path),
            "candidate_count": target_count,
            "candidate_records": candidate_records,
            "candidate_reserve": candidate_reserve,
            "selection_policy": {
                "mode": "adaptive_select_render",
                "acquisition_policy": "dnn_ensemble_disagreement_plus_diversity",
                "exploration_policy": "deterministic_pool_exploration",
                "acquisition_count": acquisition_count,
                "exploration_count": exploration_count,
                "pool_size": pool_size,
                "render_only": False,
            },
            "training_count": len(prior_records),
            "candidate_pool_count": candidate_pool_count,
            "selected_count": selected_count,
            "candidate_reserve_count": len(candidate_reserve),
            "training_record_count": len(prior_records),
            "model_selection": {key: value for key, value in report["model_selection"].items() if key != "fit"},
            "batch_summary": batch_payload,
            "prerequisites": [str(path) for path in prereq_paths],
        }
        selection_manifest_path.write_text(json.dumps(selection_payload, sort_keys=True, indent=2), encoding="utf-8")
        selection_summary = {
            "status": "selection_rendered",
            "seed": seed,
            "step": step,
            "selection_manifest_path": str(selection_manifest_path),
            "candidate_count": selection_payload["candidate_count"],
            "training_count": selection_payload["training_count"],
            "training_record_count": selection_payload["training_record_count"],
            "candidate_pool_count": selection_payload["candidate_pool_count"],
            "selected_count": selection_payload["selected_count"],
            "candidate_reserve_count": len(candidate_reserve),
            "acquisition_policy": selection_payload["selection_policy"]["acquisition_policy"],
            "exploration_policy": selection_payload["selection_policy"]["exploration_policy"],
            "acquisition_count": selection_payload["selection_policy"]["acquisition_count"],
            "exploration_count": selection_payload["selection_policy"]["exploration_count"],
            "model_selection": selection_payload["model_selection"],
            "blockers": selection_payload["blockers"],
            "batch_summary_path": str(batch_summary_path),
            "rendered_candidate_manifest_count": len(batch_payload.get("rendered_candidate_manifests", [])),
        }
        selection_summary_path.write_text(json.dumps(selection_summary, sort_keys=True, indent=2), encoding="utf-8")
        written.append(str(selection_summary_path))

    if missing_by_seed:
        details = "; ".join(
            f"seed={seed}: {', '.join(paths)}"
            for seed, paths in sorted(missing_by_seed.items())
        )
        raise ValueError(f"Adaptive select-render prerequisites missing for step {step}: {details}")
    if completed_curve_errors_by_seed:
        details = "; ".join(
            f"seed={seed}: {message}"
            for seed, message in sorted(completed_curve_errors_by_seed.items())
        )
        raise ValueError(
            f"Adaptive select-render requires completed prior curves for step {step}: {details}"
        )
    return written


def build_emb_34um_causal_validation_controller(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    seed_count: int,
    step_count: int,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    execute: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not timestamp:
        raise ValueError("timestamp is required.")
    if not run_id_prefix.strip():
        raise ValueError("run_id_prefix is required.")

    seed_count = _coerce_seed_count(seed_count)
    step_count = _coerce_positive_int(step_count, label="step_count")
    if step_count < EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS:
        raise ValueError(f"step_count must be at least {EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS}, got {step_count}.")
    walltime = _coerce_walltime(walltime)
    concurrent_jobs = _coerce_positive_int(concurrent_jobs, label="concurrent_jobs")
    retry_limit = _coerce_nonnegative_int(retry_limit, label="retry_limit")

    campaign_root = Path(scratch_root) / timestamp
    execution_mode = "execute" if execute else "render-only"
    stage_order = _stage_order(step_count=step_count)

    design_manifest = _build_design_manifest(
        timestamp=timestamp,
        scratch_root=Path(scratch_root),
        vault_root=Path(vault_root),
        force_grid_path=Path(force_grid_path),
        seed_count=seed_count,
        step_count=step_count,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        run_id_prefix=run_id_prefix,
        include_plot_requirements=not dry_run,
    )

    grouped_entries = _parse_command_inventory(
        design_manifest["command_inventory"],
        step_count=step_count,
    )
    batch_commands = _build_batch_commands(
        grouped_entries=grouped_entries,
        step_count=step_count,
        execution_mode=execution_mode,
    )
    stage_outputs = _expected_output_roots(
        campaign_root=campaign_root,
        seed_count=seed_count,
        step_count=step_count,
    )

    stages: list[dict[str, Any]] = []
    stages.append(
        _manifest_stage(
            name="render_protocol",
            description="Render campaign manifest and batch command inventory.",
            command_type="render",
            commands=[
                _build_render_protocol_command(
                    timestamp=timestamp,
                    scratch_root=Path(scratch_root),
                    vault_root=Path(vault_root),
                    force_grid_path=Path(force_grid_path),
                    walltime=walltime,
                    concurrent_jobs=concurrent_jobs,
                    retry_limit=retry_limit,
                    run_id_prefix=run_id_prefix,
                    seed_count=seed_count,
                    step_count=step_count,
                    execution_mode=execution_mode,
                )
            ],
            expected_output_roots=stage_outputs["render_protocol"],
            acceptance_criteria={
                "command_count": 1,
                "seed_count": seed_count,
                "command_inventory_entries": design_manifest["command_inventory"].get("count"),
            },
        )
    )

    for mode in ("shared_initial", "validation"):
        candidate_targets = [int(entry.get("candidate_count", 0)) for entry in grouped_entries.get(mode, [])]
        acceptance = {
            "commands_expected": len(candidate_targets) if candidate_targets else seed_count,
            "candidate_count_expected": EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
            if mode == "shared_initial"
            else EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
            "command_execution_mode": execution_mode,
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
        }
        if candidate_targets:
            acceptance["candidate_count_by_seed"] = {"min": min(candidate_targets), "max": max(candidate_targets)}
        stages.append(
            _manifest_stage(
                name=mode,
                description=f"Submit scheduler-ready batch jobs for {mode}.",
                command_type="submission",
                commands=batch_commands[mode],
                expected_output_roots=stage_outputs[mode],
                acceptance_criteria=acceptance,
            )
        )

    for step in range(1, step_count + 1):
        al_mode = f"al-step-{step:02d}"
        al_submit_stage = _al_submit_stage_name(step)
        lhs_submit_stage = _lhs_submit_stage_name(step)
        select_stage = _al_select_stage_name(step)
        selection_missing = _missing_selection_summaries(campaign_root=campaign_root, seed_count=seed_count, step=step)

        stages.append(
            _manifest_stage(
                name=select_stage,
                description=(
                    f"Train/score/select adaptive AL candidates and render batch manifests for step {step:02d}."
                ),
                command_type="select_render",
                commands=[
                    _build_al_select_render_command(
                        campaign_root=campaign_root,
                        step=step,
                        execution_mode=execution_mode,
                    )
                ],
                expected_output_roots=stage_outputs[select_stage],
                acceptance_criteria={
                    "selection_count_expected": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
                    "candidate_pool_count_expected": 500,
                    "acquisition_count_expected": 80,
                    "exploration_count_expected": 20,
                    "requires_completed_prerequisites": True,
                },
            )
        )

        al_candidate_targets = [int(entry.get("candidate_count", 0)) for entry in grouped_entries.get(al_mode, [])]
        al_blockers = None
        al_commands = batch_commands[al_submit_stage]
        if selection_missing:
            al_blockers = ["selection_output_batch_summary_required_before_submit"]
            al_commands = []
        stages.append(
            _manifest_stage(
                name=al_submit_stage,
                description=f"Submit scheduler-ready adaptive AL jobs for step {step:02d}.",
                command_type="submission",
                commands=al_commands,
                expected_output_roots=stage_outputs[al_submit_stage],
                acceptance_criteria={
                    "commands_expected": len(al_candidate_targets) if al_candidate_targets else seed_count,
                    "candidate_count_expected": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
                    "command_execution_mode": execution_mode,
                    "selection_batch_summary_required": True,
                    "selection_batch_summary_paths": [str(path) for path in _selection_summary_paths(campaign_root=campaign_root, seed_count=seed_count, step=step)],
                },
                blockers=al_blockers,
            )
        )

        lhs_mode = f"lhs-step-{step:02d}"
        lhs_candidate_targets = [int(entry.get("candidate_count", 0)) for entry in grouped_entries.get(lhs_mode, [])]
        stages.append(
            _manifest_stage(
                name=lhs_submit_stage,
                description=f"Submit scheduler-ready LHS comparator jobs for step {step:02d}.",
                command_type="submission",
                commands=batch_commands[lhs_submit_stage],
                expected_output_roots=stage_outputs[lhs_submit_stage],
                acceptance_criteria={
                    "commands_expected": len(lhs_candidate_targets) if lhs_candidate_targets else seed_count,
                    "candidate_count_expected": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
                    "command_execution_mode": execution_mode,
                },
            )
        )

    stages.append(
        _manifest_stage(
            name="ingest",
            description="Build campaign-level ingestion artifacts.",
            command_type="ingest",
            commands=[_build_ingest_command(campaign_root=campaign_root, execution_mode=execution_mode)],
            expected_output_roots=stage_outputs["ingest"],
            acceptance_criteria={
                "command_type": "result_ingest",
                "expected_outputs": 3,
                "ingest_root": str(campaign_root / "ingest"),
            },
        )
    )

    if _is_clean_ingestion_audit(campaign_root=campaign_root):
        analyze_blockers = None
        analyze_commands = [_build_analyze_command(campaign_root=campaign_root, execution_mode=execution_mode)]
    else:
        analyze_blockers = ["ingestion_audit_required_for_analysis"]
        analyze_commands = []

    stages.append(
        _manifest_stage(
            name="analyze",
            description="Run campaign-level statistical analysis.",
            command_type="analysis",
            commands=analyze_commands,
            expected_output_roots=stage_outputs["analyze"],
            acceptance_criteria={"requires_clean_ingestion": True},
            blockers=analyze_blockers,
        )
    )

    if _is_clean_ingestion_audit(campaign_root=campaign_root):
        if seed_count == EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS:
            escalation_blockers = ["escalation_not_required_when_seed_count_is_max"]
            escalation_commands: list[str] = []
        else:
            escalation_blockers = None
            escalation_commands = [_build_escalation_command(campaign_root=campaign_root, execution_mode=execution_mode)]
    else:
        escalation_blockers = ["ingestion_audit_required_for_escalation"]
        escalation_commands = []

    stages.append(
        _manifest_stage(
            name="escalation",
            description="Build escalation plan when requested seed count is insufficient.",
            command_type="escalation",
            commands=escalation_commands,
            expected_output_roots=stage_outputs["escalation"],
            acceptance_criteria={"target_seed_count": EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS},
            blockers=escalation_blockers,
        )
    )

    per_seed_target_curves = (
        EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
        + EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
        + step_count * EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE * 2
    )

    controller_manifest = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "schema_version_design": _DESIGN_SCHEMA_VERSION,
        "timestamp": timestamp,
        "execution_mode": execution_mode,
        "stage_order": stage_order,
        "campaign_root": str(campaign_root),
        "scratch_root": str(scratch_root),
        "vault_root": str(vault_root),
        "force_grid_path": str(force_grid_path),
        "run_id_prefix": run_id_prefix,
        "linear_traceability": {
            "project": "Active Learning Causal Validation",
            "engine": "Active Learning Engine",
            "issues": list(RANDOM_SEED_COUNT_ISSUES),
        },
        "stages": stages,
        "target_curve_counts": {
            "requested_seed_count": seed_count,
            "per_seed": {
                "shared_initial": EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE,
                "validation": EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
                "al_step": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
                "lhs_step": EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
                "per_seed_total": int(per_seed_target_curves),
            },
            "requested_total_curves": seed_count * int(per_seed_target_curves),
            "three_seed_target_curves": 3 * int(per_seed_target_curves),
            "five_seed_target_curves": 5 * int(per_seed_target_curves),
        },
        "karolina_resources": {
            "site": "karolina",
            "partition": "qgpu",
            "account": "eu-26-17",
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
            "submitter_script": str(
                _REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_active_learning_array.sbatch"
            ),
            "submits": "sbatch --parsable",
        },
        "replacement_policy": {
            "mode": "metadata_only",
            "al_queue": {"max_replacements": 0, "enabled": True, "mode": "metadata_only"},
            "lhs_queue": {"max_replacements": 0, "enabled": True, "mode": "metadata_only"},
            "escalation_queue": {
                "enabled": seed_count < EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
                "target_seed_count": EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
            },
        },
        "gates": {
            "shortfall": {
                "enabled": True,
                "stop_when_shortfall": True,
                "trigger": "candidate_count_below_contract",
            },
            "escalation": {
                "enabled": seed_count < EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
                "target_seed_count": EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
            },
        },
        "expected_output_roots": stage_outputs,
        "design_manifest_path": str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME),
        "design_command_inventory_path": str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME),
        "command_inventory": {
            "stages": [
                {
                    "name": stage["name"],
                    "status": stage["status"],
                    "command_type": stage["command_type"],
                    "commands": [item["command"] for item in stage["commands"]],
                    "expected_output_roots": stage["expected_output_roots"],
                    "blockers": stage.get("blockers", []),
                }
                for stage in stages
            ]
        },
    }

    manifest_path = campaign_root / CONTROLLER_MANIFEST_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(controller_manifest, sort_keys=True, indent=2), encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "manifest": controller_manifest,
        "design_manifest": design_manifest,
        "stages": stages,
    }


def _run_commands(stages: list[dict[str, Any]], *, run_commands: bool, execute: bool) -> None:
    if not (run_commands and execute):
        return
    for stage in stages:
        for entry in stage["commands"]:
            subprocess.run(entry["command"], shell=True, check=True)


def _stage_command_inventory_from_manifest(manifest: dict[str, Any], stage_name: str) -> list[str]:
    for item in manifest.get("command_inventory", {}).get("stages", []):
        if str(item.get("name", "")).strip() == stage_name:
            return [str(command) for command in item.get("commands", [])]
    return []


def _load_controller_manifest(*, campaign_root: Path) -> dict[str, Any]:
    manifest_payload = _read_json_dict(campaign_root / CONTROLLER_MANIFEST_FILE)
    if manifest_payload is None:
        raise ValueError(f"Controller manifest not found: {campaign_root / CONTROLLER_MANIFEST_FILE}")
    if not isinstance(manifest_payload.get("stage_order"), list):
        raise ValueError("Malformed controller manifest: missing stage order.")
    return manifest_payload


def _execution_requested(args: argparse.Namespace) -> bool:
    return bool(args.execute) or str(args.execution_mode).strip().lower() == "execute"


def _build_dispatcher(*, action: str, campaign_root: Path, args: argparse.Namespace) -> list[str]:
    canonical = _coerce_stage_action(action)
    execution_requested = _execution_requested(args)
    execution_mode = "execute" if execution_requested else "render-only"

    if canonical != "render_protocol" and not execution_requested:
        raise ValueError(
            "--execute is required for non-render protocol stages. "
            "Use --execute or provide --execution-mode execute."
        )

    if canonical == "render_protocol":
        return [
            _build_render_protocol_command(
                timestamp=campaign_root.name,
                scratch_root=Path(args.scratch_root),
                vault_root=Path(args.vault_root),
                force_grid_path=Path(args.force_grid),
                walltime=args.walltime,
                concurrent_jobs=args.concurrent_jobs,
                retry_limit=args.retry_limit,
                run_id_prefix=args.run_id_prefix,
                seed_count=_coerce_seed_count(args.seed_count),
                step_count=_coerce_positive_int(args.step_count, label="step_count"),
                execution_mode=execution_mode,
            )
        ]

    if _is_al_select_stage(canonical):
        step = _extract_step_from_stage(canonical)
        _render_adaptive_selection(campaign_root=campaign_root, step=step)
        return []

    if canonical in {"analyze", "escalation"}:
        _require_clean_ingestion_audit(campaign_root=campaign_root, stage=canonical)

    manifest = _load_controller_manifest(campaign_root=campaign_root)
    commands = _stage_command_inventory_from_manifest(manifest=manifest, stage_name=canonical)

    if _is_al_submit_stage(canonical):
        step = _extract_step_from_stage(canonical)
        design_manifest = _load_design_manifest(campaign_root=campaign_root)
        seed_count = _seed_count_from_design_manifest(design_manifest)
        missing = _missing_batch_summaries(campaign_root=campaign_root, seed_count=seed_count, step=step)
        if missing:
            details = ", ".join(str(path) for path in missing)
            raise ValueError(
                f"{canonical} dispatch requires rendered batch summaries for all seeds. Missing: {details}"
            )
        if not commands:
            commands = _build_al_submit_fallback_commands(
                campaign_root=campaign_root,
                step=step,
                seed_count=seed_count,
                execution_mode=execution_mode,
            )

    if not commands:
        raise ValueError(f"Stage {canonical!r} is blocked or has no registered commands.")

    # Ensure dispatch reflects explicit execution intent.
    return [_replace_execution_mode(command, execution_mode=execution_mode) for command in commands]


def _dispatch_stage_action(*, action: str, campaign_root: Path, args: argparse.Namespace) -> list[str]:
    canonical = _coerce_stage_action(action, step_count=_coerce_positive_int(args.step_count, label="step_count"))
    return _build_dispatcher(action=canonical, campaign_root=campaign_root, args=args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="", help="Campaign timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT))
    parser.add_argument("--vault-root", default=str(DEFAULT_VAULT_ROOT))
    parser.add_argument("--force-grid", default=str(DEFAULT_FORCE_GRID_DATA))
    parser.add_argument("--seed-count", type=int, default=len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS))
    parser.add_argument("--step-count", type=int, default=EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument("--concurrent-jobs", type=int, default=DEFAULT_CONCURRENT_JOBS)
    parser.add_argument("--retry-limit", type=int, default=DEFAULT_RETRY_LIMIT)
    parser.add_argument("--run-id-prefix", default=DEFAULT_RUN_ID_PREFIX)
    parser.add_argument(
        "--stage-action",
        default="",
        help="Execute one canonical stage action instead of preparing the full controller.",
    )
    parser.add_argument("--campaign-root", default="")
    parser.add_argument("--execution-mode", default="render-only")
    parser.add_argument("--run-commands", action="store_true", help="Run staged commands.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", help="Allow execute-mode dispatch and running commands.")
    return parser


def _main_build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.timestamp:
        raise ValueError("--timestamp is required unless --stage-action is used.")
    return build_emb_34um_causal_validation_controller(
        timestamp=args.timestamp,
        scratch_root=Path(args.scratch_root),
        vault_root=Path(args.vault_root),
        force_grid_path=Path(args.force_grid),
        seed_count=args.seed_count,
        step_count=args.step_count,
        walltime=args.walltime,
        concurrent_jobs=args.concurrent_jobs,
        retry_limit=args.retry_limit,
        run_id_prefix=args.run_id_prefix,
        execute=bool(args.execute),
        dry_run=bool(args.dry_run),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stage_action:
        campaign_root_text = str(args.campaign_root or "").strip()
        if not campaign_root_text:
            raise ValueError("--campaign-root is required for --stage-action.")
        campaign_root = Path(campaign_root_text)
        commands = _dispatch_stage_action(
            action=args.stage_action,
            campaign_root=campaign_root,
            args=args,
        )
        _run_commands(
            [{"commands": [{"command": command} for command in commands]}],
            run_commands=args.run_commands,
            execute=_execution_requested(args),
        )
        return 0

    result = _main_build_plan(args)
    _run_commands(result["stages"], run_commands=args.run_commands, execute=args.execute)
    print(f"manifest_path={result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
