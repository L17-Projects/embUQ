#!/usr/bin/env python3
"""Render and register AL reserve replacements for failed DNN causal candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.active_learning import Candidate  # noqa: E402
from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (  # noqa: E402
    _sample_points_with_source_indices_4d as _design_sample_points_with_source_indices_4d,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    is_dnn_causal_low_corner_excluded,
)
from meso_uq.active_learning.emb_34um_dnn_acquisition import select_greedy_diversity  # noqa: E402


BATCH_SUMMARY_FILENAME = "emb_34um_batch_summary.json"
DESIGN_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_manifest.json"
SELECTION_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_selection_manifest.json"
SELECTION_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_selection_batch_summary.json"
REPLACEMENT_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_replacement_manifest.json"
REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_replacement_batch_summary.json"


def _extract_float_field(source: Mapping[str, Any], field: str) -> float | None:
    if field not in source:
        return None
    try:
        return float(source[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field!r} in source record is not numeric: {source[field]!r}.") from exc


def _compute_box_dimensions(radp: float, *, explicit_cubic: bool = False) -> tuple[float, float, float]:
    if explicit_cubic:
        box = float(math.ceil(2.0 * radp + 10.0))
        return (box, box, box)
    lx = float(math.ceil(2.0 * radp + 6.0))
    return (lx, lx, float(math.ceil(2.0 * radp + 10.0)))


def _d4_runtime_fields(
    primary: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    runtime_fingerprint: dict[str, Any] = {}
    for source in (fallback, primary):
        if source is None:
            continue
        source_runtime = source.get("runtime_fingerprint", {})
        if isinstance(source_runtime, Mapping):
            runtime_fingerprint.update(source_runtime)
        for key in ("radp", "shell_th", "fscale", "numsteps", "numsteps_eq", "L", "Lx", "Ly", "Lz", "direct_stiffness_override", "bpress"):
            if key in source and key not in runtime_fingerprint:
                runtime_fingerprint[key] = source[key]

    radp = _extract_float_field(primary, "radp")
    shell_th = _extract_float_field(primary, "shell_th")
    bpress = _extract_float_field(primary, "bpress")
    if radp is None:
        radp = _extract_float_field(runtime_fingerprint, "radp")
    if shell_th is None:
        shell_th = _extract_float_field(runtime_fingerprint, "shell_th")
    if bpress is None:
        bpress = _extract_float_field(runtime_fingerprint, "bpress")
    if fallback is not None:
        if radp is None:
            radp = _extract_float_field(fallback, "radp")
        if shell_th is None:
            shell_th = _extract_float_field(fallback, "shell_th")
        if bpress is None:
            bpress = _extract_float_field(fallback, "bpress")

    if radp is None or shell_th is None:
        raise ValueError("Replacement candidates require D4 radp and shell_th fields.")
    if bpress is None:
        bpress = EMB_34UM_DNN_CAUSAL_BPRESS_VALUE

    runtime_fingerprint["radp"] = float(radp)
    runtime_fingerprint["shell_th"] = float(shell_th)
    runtime_fingerprint["bpress"] = float(bpress)

    if not {"Lx", "Ly", "Lz"} <= runtime_fingerprint.keys():
        explicit_cubic = "L" in runtime_fingerprint
        box = _compute_box_dimensions(radp, explicit_cubic=explicit_cubic)
        runtime_fingerprint.setdefault("Lx", box[0])
        runtime_fingerprint.setdefault("Ly", box[1])
        runtime_fingerprint.setdefault("Lz", box[2])
        runtime_fingerprint.setdefault("L", box[2])
    elif "L" not in runtime_fingerprint:
        runtime_fingerprint["L"] = float(runtime_fingerprint["Lz"])

    d4_parameters = {
        "radp": float(radp),
        "shell_th": float(shell_th),
        "bpress": float(bpress),
    }
    return d4_parameters, dict(runtime_fingerprint)


def _load_prepare_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "prepare_emb_34um_dnn_causal_validation.py"
    spec = importlib.util.spec_from_file_location("_prepare_emb_34um_dnn_causal_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prepare module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return list(value)


def _as_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _candidate_source_id(row: Mapping[str, Any]) -> str:
    selection_payload = row.get("selection_payload", {})
    sources = (selection_payload, row) if isinstance(selection_payload, Mapping) else (row,)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("candidate_pool_id", "candidate_id"):
            token = str(source.get(key, "")).strip()
            if token:
                return token
    if "candidate_index" in row:
        return str(row["candidate_index"])
    return ""


def _derive_al_reserve_from_train_manifest(
    *,
    selection_manifest: Mapping[str, Any],
    active_candidate_records: Sequence[Mapping[str, Any]],
    active_reserve_records: Sequence[Mapping[str, Any]],
    prior_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_manifest_path = Path(str(selection_manifest.get("train_score_select_manifest_path", "")))
    if not train_manifest_path.is_file():
        return []
    train_manifest = _load_json_object(train_manifest_path)
    scored_rows = [
        _as_mapping(item, label="candidate_scores item")
        for item in _as_sequence(train_manifest.get("candidate_scores", ()), label="candidate_scores")
    ]
    if not scored_rows:
        return []

    used_ids = {
        token
        for token in (_candidate_source_id(item) for item in active_candidate_records)
        if token
    }
    used_ids.update(
        token
        for token in (_candidate_source_id(item) for item in prior_records)
        if token
    )
    used_ids.update(
        token
        for token in (_candidate_source_id(item) for item in active_reserve_records)
        if token
    )

    existing_points_payload = train_manifest.get("existing_points", {})
    existing_points = ()
    diversity_weight = 0.25
    if isinstance(existing_points_payload, Mapping):
        raw_points = existing_points_payload.get("points", ())
        if isinstance(raw_points, Sequence) and not isinstance(raw_points, (str, bytes, bytearray)):
            existing_points = tuple(raw_points)
        try:
            diversity_weight = float(existing_points_payload.get("diversity_weight", diversity_weight))
        except (TypeError, ValueError):
            diversity_weight = 0.25

    selected_order = select_greedy_diversity(
        scored_rows,
        count=len(scored_rows),
        existing_points=existing_points,
        candidate_space=str(train_manifest.get("candidate_space", "d4")),
        diversity_weight=diversity_weight,
    )

    reserve: list[dict[str, Any]] = []
    for row in selected_order:
        source_id = _candidate_source_id(row)
        if not source_id or source_id in used_ids:
            continue
        d4_parameters, runtime_fingerprint = _d4_runtime_fields(row)
        reserve.append(
            {
                "candidate_pool_id": source_id,
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                **d4_parameters,
                "runtime_fingerprint": runtime_fingerprint,
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                "diversity_term": float(row.get("diversity_term", 0.0)),
                "candidate_index": row.get("candidate_index"),
                "reserve_source": "train_score_select_candidate_scores",
            }
        )
        used_ids.add(source_id)
    return reserve


def _stage_root(campaign_root: Path, *, replica: int, cycle: int) -> Path:
    return campaign_root / f"replica-{replica:03d}" / f"al-step-{cycle:02d}"


def _resolve_stage_name(*, stage: str | None, cycle: int | None) -> tuple[str, int | None]:
    if stage is None:
        if cycle is None:
            raise ValueError("Either --stage or --cycle must be provided.")
        return f"al-step-{cycle:02d}", cycle

    normalized = str(stage).strip()
    if not normalized:
        raise ValueError("--stage must be non-empty.")
    if normalized in {"shared_initial", "unseen_test"}:
        return normalized, None
    if re.fullmatch(r"(?:al|lhs)-step-\d{2}", normalized):
        parsed_cycle = int(normalized.split("-")[-1])
        if cycle is not None and int(cycle) != parsed_cycle:
            raise ValueError(f"--cycle={cycle} does not match --stage={normalized!r}.")
        return normalized, parsed_cycle
    raise ValueError(
        "--stage must be one of shared_initial, unseen_test, al-step-XX, or lhs-step-XX."
    )


def _design_seed_payload(design_manifest: Mapping[str, Any], *, replica: int) -> dict[str, Any]:
    seeds = _as_sequence(design_manifest.get("seeds", ()), label="seeds")
    for index, seed_payload in enumerate(seeds):
        candidate = _as_mapping(seed_payload, label=f"seeds[{index}]")
        if int(candidate.get("replica", -1)) == int(replica):
            return candidate
    raise ValueError(f"Design manifest does not contain replica {replica:03d}.")


def _non_al_stage_stream(
    *,
    design_manifest: Mapping[str, Any],
    stage_name: str,
    replica: int,
    cycle: int | None,
) -> tuple[list[dict[str, Any]], int, str, str]:
    if stage_name == "shared_initial":
        seed_payload = _design_seed_payload(design_manifest, replica=replica)
        records = [
            _as_mapping(item, label="shared_initial candidate_record")
            for item in _as_sequence(seed_payload.get("shared_initial", {}).get("candidate_records", ()), label="shared_initial candidate_records")
        ]
        return records, (replica - 1) * 20_000, "shared_initial", "fresh_dpd"

    if stage_name == "unseen_test":
        unseen = _as_mapping(design_manifest.get("unseen_test", {}), label="unseen_test")
        records = [
            _as_mapping(item, label="unseen_test candidate_record")
            for item in _as_sequence(unseen.get("candidate_records", ()), label="unseen_test candidate_records")
        ]
        return records, 0, "unseen_test", EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE

    if stage_name.startswith("lhs-step-"):
        if cycle is None:
            raise ValueError("Cycle is required when repairing lhs-step stages.")
        seed_payload = _design_seed_payload(design_manifest, replica=replica)
        shared_records = [
            _as_mapping(item, label="shared_initial candidate_record")
            for item in _as_sequence(seed_payload.get("shared_initial", {}).get("candidate_records", ()), label="shared_initial candidate_records")
        ]
        lhs_steps = _as_sequence(seed_payload.get("lhs_steps", ()), label="lhs_steps")
        if cycle < 1 or cycle > len(lhs_steps):
            raise ValueError(f"lhs-step-{cycle:02d} is not present in the design manifest.")
        lhs_payload = _as_mapping(lhs_steps[cycle - 1], label=f"lhs_steps[{cycle - 1}]")
        records = [
            _as_mapping(item, label="lhs-step candidate_record")
            for item in _as_sequence(lhs_payload.get("candidate_records", ()), label="lhs-step candidate_records")
        ]
        stream_start = (replica - 1) * 20_000 + len(shared_records) + (cycle - 1) * len(records)
        return records, stream_start, f"lhs-step-{cycle:02d}", EMB_34UM_DNN_CAUSAL_LHS_SOURCE

    raise ValueError(f"Unsupported non-AL stage {stage_name!r}.")


def _replacement_source_index(
    *,
    stream_start: int,
    planned_count: int,
    prior_records: Sequence[Mapping[str, Any]],
    sequence: int,
) -> int:
    used_indices = {
        int(item["source_index"])
        for item in prior_records
        if isinstance(item, Mapping) and "source_index" in item
    }
    if used_indices:
        next_index = max(stream_start + planned_count, max(used_indices) + 1)
    else:
        next_index = stream_start + planned_count

    remaining = int(sequence) - 1
    while True:
        if next_index not in used_indices:
            if remaining == 0:
                return next_index
            remaining -= 1
        next_index += 1


def _render_replacement_batch(
    *,
    candidates: tuple[Candidate, ...],
    batch_root: Path,
    run_id: str,
    batch_id: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
) -> dict[str, Any]:
    prepare = _load_prepare_module()
    return prepare._render_batch(
        candidates=candidates,
        batch_root=batch_root,
        run_id=run_id,
        batch_id=batch_id,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )


def _normalize_failed_indices(
    *,
    failed_candidate_indices: Sequence[int],
    failed_candidate_ids: Sequence[str],
    candidate_records: Sequence[Mapping[str, Any]],
) -> list[int]:
    candidate_lookup = {
        str(record.get("candidate_id", "")).strip(): index
        for index, record in enumerate(candidate_records)
    }
    resolved = {int(index) for index in failed_candidate_indices}
    for candidate_id in failed_candidate_ids:
        lookup = candidate_lookup.get(str(candidate_id).strip())
        if lookup is None:
            raise ValueError(f"Candidate {candidate_id!r} is not present in the active AL candidate_records.")
        resolved.add(lookup)
    if not resolved:
        raise ValueError("At least one --candidate-index or --failed-candidate-id must be provided.")
    ordered = sorted(resolved)
    if ordered[0] < 0 or ordered[-1] >= len(candidate_records):
        raise ValueError(f"Candidate indices must be within [0, {len(candidate_records) - 1}].")
    return ordered


def _sampler_for_stage(stage_name: str) -> str:
    if stage_name.startswith("al-step-"):
        return "AL"
    return "LHS"


def _replacement_attempt_count(prior_records: Sequence[Mapping[str, Any]], *, failed_candidate_id: str) -> int:
    attempts = 1
    for item in prior_records:
        if str(item.get("failed_candidate_id", "")).strip() == failed_candidate_id:
            attempts += 1
    return attempts


def render_al_stage_replacements(
    *,
    campaign_root: Path,
    replica: int,
    cycle: int | None,
    stage: str | None = None,
    failed_candidate_indices: Sequence[int] = (),
    failed_candidate_ids: Sequence[str] = (),
    failure_reason: str = "runtime_failure_or_quarantine",
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    stage_name, resolved_cycle = _resolve_stage_name(stage=stage, cycle=cycle)
    if stage_name == "unseen_test":
        stage_root = campaign_root / stage_name
    elif stage_name == "shared_initial":
        stage_root = campaign_root / f"replica-{replica:03d}" / stage_name
    elif stage_name.startswith(("lhs-step-", "al-step-")):
        if resolved_cycle is None:
            raise ValueError(f"Cycle is required for stage {stage_name!r}.")
        stage_root = campaign_root / f"replica-{replica:03d}" / stage_name
    else:
        raise ValueError(f"Unsupported stage {stage_name!r}.")
    if not stage_root.is_dir():
        raise ValueError(f"Stage root does not exist: {stage_root}")

    design_manifest = _load_json_object(campaign_root / DESIGN_MANIFEST_FILENAME)
    batch_summary_path = stage_root / BATCH_SUMMARY_FILENAME
    batch_summary = _load_json_object(batch_summary_path)

    replacement_manifest_path = stage_root / REPLACEMENT_MANIFEST_FILENAME
    replacement_manifest = (
        _load_json_object(replacement_manifest_path)
        if replacement_manifest_path.is_file()
        else {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_replacement_manifest.v1",
            "replacement_records": [],
            "replacement_batches": [],
        }
    )
    prior_records = [_as_mapping(item, label="replacement record") for item in _as_sequence(replacement_manifest.get("replacement_records", ()), label="replacement_records")]
    prior_batches = [_as_mapping(item, label="replacement batch") for item in _as_sequence(replacement_manifest.get("replacement_batches", ()), label="replacement_batches")]

    run_id_prefix = str(design_manifest.get("run_id_prefix", "emb-34um-dnn-causal-validation"))
    command_inventory = _as_mapping(design_manifest.get("command_inventory", {}), label="command_inventory")
    walltime = str(command_inventory.get("walltime", EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT))
    concurrent_jobs = int(command_inventory.get("concurrent_jobs", 30))
    retry_limit = int(command_inventory.get("retry_limit", EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT))

    batch_index = len(prior_batches) + 1
    replacement_root = stage_root / "replacement"
    batch_root = replacement_root / f"batch-{batch_index:03d}"
    batch_root.mkdir(parents=True, exist_ok=True)

    replacement_records_payload: list[dict[str, Any]] = []
    replacement_candidate_records: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    sampler = _sampler_for_stage(stage_name)

    if stage_name.startswith("al-step-"):
        selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
        selection_summary_path = stage_root / SELECTION_SUMMARY_FILENAME
        selection_manifest = _load_json_object(selection_manifest_path)
        selection_summary = _load_json_object(selection_summary_path)
        active_candidate_records = [
            _as_mapping(item, label="candidate_records item")
            for item in _as_sequence(selection_manifest.get("candidate_records", ()), label="candidate_records")
        ]
        active_reserve = [
            _as_mapping(item, label="candidate_reserve item")
            for item in _as_sequence(selection_manifest.get("candidate_reserve", ()), label="candidate_reserve")
        ]
        failed_indices = _normalize_failed_indices(
            failed_candidate_indices=failed_candidate_indices,
            failed_candidate_ids=failed_candidate_ids,
            candidate_records=active_candidate_records,
        )
        if len(active_reserve) < len(failed_indices):
            active_reserve.extend(
                _derive_al_reserve_from_train_manifest(
                    selection_manifest=selection_manifest,
                    active_candidate_records=active_candidate_records,
                    active_reserve_records=active_reserve,
                    prior_records=prior_records,
                )
            )
        if len(active_reserve) < len(failed_indices):
            raise ValueError(
                f"Not enough reserve candidates remain for replica={replica} cycle={resolved_cycle}: "
                f"requested {len(failed_indices)}, available {len(active_reserve)}."
            )

        force_grid = tuple(float(item) for item in selection_manifest.get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID))
        selected_reserve = active_reserve[: len(failed_indices)]
        for sequence, (failed_index, reserve_row) in enumerate(zip(failed_indices, selected_reserve), start=1):
            failed_record = active_candidate_records[failed_index]
            failed_candidate_id = str(failed_record.get("candidate_id", "")).strip()
            attempt_count = _replacement_attempt_count(prior_records, failed_candidate_id=failed_candidate_id)
            replacement_candidate_id = (
                f"{run_id_prefix}-rep{replica:03d}-al-step{resolved_cycle:02d}-r{batch_index:02d}-{sequence:03d}"
            )
            ka = float(reserve_row["ka"])
            kb = float(reserve_row["kb"])
            d4_parameters, runtime_fingerprint = _d4_runtime_fields(reserve_row, fallback=failed_record)
            selection_payload = {
                "candidate_pool_id": str(reserve_row.get("candidate_pool_id", replacement_candidate_id)),
                "acquisition_score": float(reserve_row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(reserve_row.get("ensemble_disagreement", 0.0)),
                "diversity_term": float(reserve_row.get("diversity_term", 0.0)),
                "replacement_for": failed_candidate_id,
                "failed_candidate_index": failed_index,
                "replacement_batch_index": batch_index,
                "replacement_sequence": sequence,
                "replacement_policy": "candidate_reserve",
                "sampler": "AL",
                "replica": replica,
                "cycle": resolved_cycle,
                "original_candidate_id": failed_candidate_id,
                "replacement_candidate_id": replacement_candidate_id,
                "attempt_count": attempt_count,
                "reason": str(failure_reason),
                "skipped_by_feasibility_filter": bool(
                    is_dnn_causal_low_corner_excluded(
                        ka,
                        kb,
                        d4_parameters.get("radp"),
                        d4_parameters.get("shell_th"),
                    )
                ),
            }
            candidate = Candidate(
                candidate_id=replacement_candidate_id,
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": ka,
                    "kb": kb,
                    "force_grid": list(force_grid),
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                },
                metadata={
                    "selection_seed": replica * 10_000 + resolved_cycle,
                    "selection_mode": "dnn_causal_fresh_only",
                    "selection_source": "dnn_ensemble_disagreement_diversity",
                    "selection_status": "replacement_rendered",
                    "selection_payload": selection_payload,
                    "replacement": True,
                    "replacement_for": failed_candidate_id,
                    "failed_candidate_id": failed_candidate_id,
                    "source": "replacement",
                    "candidate_pool_id": selection_payload["candidate_pool_id"],
                    "acquisition_score": selection_payload["acquisition_score"],
                    "ensemble_disagreement": selection_payload["ensemble_disagreement"],
                    "diversity_term": selection_payload["diversity_term"],
                    "replacement_policy": "candidate_reserve",
                    "runtime_fingerprint": runtime_fingerprint,
                },
            )
            candidates.append(candidate)
            replacement_candidate_records.append(
                {
                    "candidate_id": replacement_candidate_id,
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": ka,
                    "kb": kb,
                    "selection_seed": replica * 10_000 + resolved_cycle,
                    "selection_mode": "dnn_causal_fresh_only",
                    "selection_source": "dnn_ensemble_disagreement_diversity",
                    "selection_status": "replacement_rendered",
                    "replacement": True,
                    "replacement_for": failed_candidate_id,
                    "replacement_policy": "candidate_reserve",
                    "selection_payload": selection_payload,
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                }
            )
            replacement_records_payload.append(
                {
                    "sampler": "AL",
                    "replica": replica,
                    "cycle": resolved_cycle,
                    "stage": stage_name,
                    "reason": str(failure_reason),
                    "attempt_count": attempt_count,
                    "original_candidate_id": failed_candidate_id,
                    "replacement_candidate_id": replacement_candidate_id,
                    "skipped_by_feasibility_filter": bool(
                        is_dnn_causal_low_corner_excluded(
                            ka,
                            kb,
                            d4_parameters.get("radp"),
                            d4_parameters.get("shell_th"),
                        )
                    ),
                    "failed_candidate_index": failed_index,
                    "failed_candidate_id": failed_candidate_id,
                    "failed_candidate_manifest_path": str(batch_summary["rendered_candidate_manifests"][failed_index]),
                    "failed_output_root": str(batch_summary["expected_output_roots"][failed_index]),
                    "candidate_pool_id": selection_payload["candidate_pool_id"],
                    "acquisition_score": selection_payload["acquisition_score"],
                    "ensemble_disagreement": selection_payload["ensemble_disagreement"],
                    "diversity_term": selection_payload["diversity_term"],
                    "replacement_batch_index": batch_index,
                    "replacement_sequence": sequence,
                    "replacement_policy": "candidate_reserve",
                    "sampler": "AL",
                    "replica": replica,
                    "cycle": resolved_cycle,
                    "original_candidate_id": failed_candidate_id,
                    "replacement_candidate_id": replacement_candidate_id,
                    "attempt_count": attempt_count,
                    "reason": str(failure_reason),
                    "skipped_by_feasibility_filter": bool(
                        is_dnn_causal_low_corner_excluded(
                            ka,
                            kb,
                            d4_parameters.get("radp"),
                            d4_parameters.get("shell_th"),
                        )
                    ),
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                }
            )
    else:
        active_candidate_records, stream_start, source_stage, selection_source = _non_al_stage_stream(
            design_manifest=design_manifest,
            stage_name=stage_name,
            replica=replica,
            cycle=resolved_cycle,
        )
        failed_indices = _normalize_failed_indices(
            failed_candidate_indices=failed_candidate_indices,
            failed_candidate_ids=failed_candidate_ids,
            candidate_records=active_candidate_records,
        )
        if len(active_candidate_records) != len(batch_summary.get("expected_output_roots", [])):
            raise ValueError(f"Stage {stage_name!r} has mismatched candidate and output-root counts.")
        force_grid = tuple(float(item) for item in design_manifest.get("policy", {}).get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID))
        for sequence, failed_index in enumerate(failed_indices, start=1):
            failed_record = active_candidate_records[failed_index]
            failed_candidate_id = str(failed_record.get("candidate_id", "")).strip()
            attempt_count = _replacement_attempt_count(prior_records, failed_candidate_id=failed_candidate_id)
            requested_source_index = _replacement_source_index(
                stream_start=stream_start,
                planned_count=len(active_candidate_records),
                prior_records=[*active_candidate_records, *prior_records, *replacement_records_payload],
                sequence=1,
            )
            source_index, ka, kb, radp, shell_th = _design_sample_points_with_source_indices_4d(
                start_index=requested_source_index,
                count=1,
            )[0]
            d4_parameters, runtime_fingerprint = _d4_runtime_fields(
                {
                    "radp": radp,
                    "shell_th": shell_th,
                    "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                },
                fallback=None,
            )
            skipped_source_index_count = max(0, int(source_index) - int(requested_source_index))
            skipped_by_feasibility_filter = skipped_source_index_count > 0
            replacement_candidate_id = (
                f"{run_id_prefix}-rep{replica:03d}-{stage_name.replace('_', '-')}-src{source_index:06d}"
            )
            selection_payload = {
                "original_candidate_id": failed_candidate_id,
                "replacement_for": failed_candidate_id,
                "replacement_batch_index": batch_index,
                "replacement_sequence": sequence,
                "replacement_policy": "deterministic_next_unused_fresh_dpd",
                "source_stage": source_stage,
                "source_index": source_index,
                "requested_source_index": requested_source_index,
                "selection_source": selection_source,
                "selection_mode": "dnn_causal_fresh_only",
                "candidate_pool_id": f"{source_stage}-source-{source_index:06d}",
                "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
                "sampler": sampler,
                "replica": replica,
                "cycle": resolved_cycle if resolved_cycle is not None else 0,
                "replacement_candidate_id": replacement_candidate_id,
                "attempt_count": attempt_count,
                "reason": str(failure_reason),
                "skipped_by_feasibility_filter": skipped_by_feasibility_filter,
                "skipped_source_index_count": skipped_source_index_count,
            }
            candidate = Candidate(
                candidate_id=replacement_candidate_id,
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": ka,
                    "kb": kb,
                    "force_grid": list(force_grid),
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                },
                metadata={
                    "selection_seed": source_index,
                    "selection_mode": "dnn_causal_fresh_only",
                    "selection_source": selection_source,
                    "selection_status": "replacement_rendered",
                    "selection_payload": selection_payload,
                    "replacement": True,
                    "replacement_for": failed_candidate_id,
                    "original_candidate_id": failed_candidate_id,
                    "source": "replacement",
                    "source_index": source_index,
                    "source_stage": source_stage,
                    "candidate_pool_id": selection_payload["candidate_pool_id"],
                    "replacement_policy": "deterministic_next_unused_fresh_dpd",
                    "sampler": sampler,
                    "replica": replica,
                    "cycle": resolved_cycle if resolved_cycle is not None else 0,
                    "replacement_candidate_id": replacement_candidate_id,
                    "attempt_count": attempt_count,
                    "reason": str(failure_reason),
                    "skipped_by_feasibility_filter": skipped_by_feasibility_filter,
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                    "skipped_source_index_count": skipped_source_index_count,
                },
            )
            candidates.append(candidate)
            replacement_candidate_records.append(
                {
                    "candidate_id": replacement_candidate_id,
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": ka,
                    "kb": kb,
                    "selection_seed": source_index,
                    "selection_mode": "dnn_causal_fresh_only",
                    "selection_source": selection_source,
                    "selection_status": "replacement_rendered",
                    "replacement": True,
                    "replacement_for": failed_candidate_id,
                    "original_candidate_id": failed_candidate_id,
                    "source_index": source_index,
                    "requested_source_index": requested_source_index,
                    "source_stage": source_stage,
                    "replacement_policy": "deterministic_next_unused_fresh_dpd",
                    "selection_payload": selection_payload,
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                }
            )
            replacement_records_payload.append(
                {
                    "sampler": sampler,
                    "replica": replica,
                    "cycle": resolved_cycle if resolved_cycle is not None else 0,
                    "stage": stage_name,
                    "reason": str(failure_reason),
                    "attempt_count": attempt_count,
                    "original_candidate_id": failed_candidate_id,
                    "replacement_candidate_id": replacement_candidate_id,
                    "skipped_by_feasibility_filter": skipped_by_feasibility_filter,
                    "skipped_source_index_count": skipped_source_index_count,
                    "failed_candidate_index": failed_index,
                    "failed_candidate_id": failed_candidate_id,
                    "failed_candidate_manifest_path": str(batch_summary["rendered_candidate_manifests"][failed_index]),
                    "failed_output_root": str(batch_summary["expected_output_roots"][failed_index]),
                    "source_stage": source_stage,
                    "source_index": source_index,
                    "requested_source_index": requested_source_index,
                    "replacement_policy": "deterministic_next_unused_fresh_dpd",
                    "replacement_batch_index": batch_index,
                    "replacement_sequence": sequence,
                    **d4_parameters,
                    "runtime_fingerprint": runtime_fingerprint,
                }
            )

    batch_payload = _render_replacement_batch(
        candidates=tuple(candidates),
        batch_root=batch_root,
        run_id=(
            f"{run_id_prefix}-rep{replica:03d}-al-step{resolved_cycle:02d}-replacement-{batch_index:03d}"
            if stage_name.startswith("al-step-")
            else f"{run_id_prefix}-rep{replica:03d}-{stage_name.replace('_', '-')}-replacement-{batch_index:03d}"
        ),
        batch_id=(
            f"rep{replica:03d}-al-step-{resolved_cycle:02d}-replacement-{batch_index:03d}"
            if stage_name.startswith("al-step-")
            else f"rep{replica:03d}-{stage_name.replace('_', '-')}-replacement-{batch_index:03d}"
        ),
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )
    batch_manifest = _load_json_object(Path(str(batch_payload["manifest_path"])))
    submission = _as_mapping(batch_manifest.get("scheduler_boundary", {}).get("submission", {}), label="scheduler_boundary.submission")
    submission_commands = [str(item) for item in _as_sequence(submission.get("submission_commands", ()), label="submission_commands") if str(item).strip()]
    replacement_manifest_paths = [str(item) for item in batch_payload.get("rendered_candidate_manifests", [])]
    replacement_output_roots = [str(item) for item in batch_payload.get("expected_output_roots", [])]
    if len(replacement_manifest_paths) != len(replacement_records_payload):
        raise ValueError("Replacement render returned an unexpected number of candidate manifests.")

    selection_manifest_path = stage_root / SELECTION_MANIFEST_FILENAME
    selection_summary_path = stage_root / SELECTION_SUMMARY_FILENAME
    if stage_name.startswith("al-step-"):
        original_candidate_records = selection_manifest.get("original_candidate_records")
        if not isinstance(original_candidate_records, list):
            selection_manifest["original_candidate_records"] = list(active_candidate_records)
        original_candidate_reserve = selection_manifest.get("original_candidate_reserve")
        if not isinstance(original_candidate_reserve, list):
            selection_manifest["original_candidate_reserve"] = list(active_reserve)
        original_rendered_candidate_manifests = batch_summary.get("original_rendered_candidate_manifests")
        if not isinstance(original_rendered_candidate_manifests, list):
            batch_summary["original_rendered_candidate_manifests"] = list(batch_summary.get("rendered_candidate_manifests", []))
        original_expected_output_roots = batch_summary.get("original_expected_output_roots")
        if not isinstance(original_expected_output_roots, list):
            batch_summary["original_expected_output_roots"] = list(batch_summary.get("expected_output_roots", []))
    else:
        selection_manifest = {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_selection_manifest.v1",
            "status": "replacement_rendered",
            "stage": stage_name,
            "replica": replica,
            "cycle": resolved_cycle if resolved_cycle is not None else 0,
            "candidate_count": len(active_candidate_records),
            "candidate_records": list(active_candidate_records),
            "candidate_reserve": [],
            "candidate_reserve_count": 0,
            "force_grid": list(force_grid),
            "design_manifest_path": str(campaign_root / DESIGN_MANIFEST_FILENAME),
            "batch_summary_path": str(batch_summary_path),
            "replacement_records": prior_records + replacement_records_payload,
            "replacement_count": len(prior_records) + len(replacement_records_payload),
            "replacement_policy": "deterministic_next_unused_fresh_dpd",
            "original_candidate_records": list(active_candidate_records),
            "original_expected_output_roots": list(batch_summary.get("expected_output_roots", [])),
        }
        selection_summary = {
            "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_selection_batch_summary.v1",
            "status": "replacement_rendered",
            "stage": stage_name,
            "replica": replica,
            "cycle": resolved_cycle if resolved_cycle is not None else 0,
            "candidate_count": len(active_candidate_records),
            "candidate_reserve_count": 0,
            "replacement_count": len(prior_records) + len(replacement_records_payload),
            "batch_summary_path": str(batch_summary_path),
            "selection_manifest_path": str(selection_manifest_path),
            "replacement_manifest_path": str(replacement_manifest_path),
        }
        batch_summary["original_rendered_candidate_manifests"] = list(batch_summary.get("rendered_candidate_manifests", []))
        batch_summary["original_expected_output_roots"] = list(batch_summary.get("expected_output_roots", []))

    for failed_index, replacement_record, replacement_candidate_record, manifest_path, output_root in zip(
        failed_indices,
        replacement_records_payload,
        replacement_candidate_records,
        replacement_manifest_paths,
        replacement_output_roots,
    ):
        batch_summary["rendered_candidate_manifests"][failed_index] = manifest_path
        batch_summary["expected_output_roots"][failed_index] = output_root
        replacement_record["replacement_candidate_manifest_path"] = manifest_path
        replacement_record["replacement_output_root"] = output_root
        if stage_name.startswith("al-step-"):
            active_candidate_records[failed_index] = replacement_candidate_record

    if stage_name.startswith("al-step-"):
        selection_manifest["candidate_records"] = active_candidate_records
        selection_manifest["candidate_reserve"] = active_reserve[len(failed_indices) :]
        selection_manifest["candidate_reserve_count"] = len(selection_manifest["candidate_reserve"])
        selection_manifest["replacement_records"] = prior_records + replacement_records_payload
        selection_manifest["replacement_count"] = len(selection_manifest["replacement_records"])
        selection_manifest["status"] = "replacement_rendered"
        selection_summary["candidate_reserve_count"] = len(selection_manifest["candidate_reserve"])
        selection_summary["replacement_count"] = int(selection_manifest["replacement_count"])
        selection_summary["replacement_manifest_path"] = str(replacement_manifest_path)
        selection_manifest["original_candidate_records"] = selection_manifest.get("original_candidate_records", list(active_candidate_records))
        selection_manifest["original_candidate_reserve"] = selection_manifest.get("original_candidate_reserve", list(active_reserve))
    else:
        selection_manifest["replacement_manifest_path"] = str(replacement_manifest_path)

    replacement_batch_summary_path = batch_root / REPLACEMENT_BATCH_SUMMARY_FILENAME
    failed_candidate_ids_payload = [str(item["failed_candidate_id"]) for item in replacement_records_payload]
    replacement_batch_summary_path = batch_root / REPLACEMENT_BATCH_SUMMARY_FILENAME
    replacement_batch_summary = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_replacement_batch_summary.v1",
        "status": "replacement_rendered",
        "campaign_root": str(campaign_root),
        "sampler": sampler,
        "replica": replica,
        "cycle": resolved_cycle if resolved_cycle is not None else 0,
        "stage": stage_name,
        "replacement_batch_index": batch_index,
        "replacement_batch_root": str(batch_root),
        "selection_manifest_path": str(selection_manifest_path),
        "replacement_submission_commands": submission_commands,
        "failed_candidate_indices": list(failed_indices),
        "failed_candidate_ids": failed_candidate_ids_payload,
        "replacement_records": replacement_records_payload,
        "quarantine_records": replacement_records_payload,
        "expected_output_roots": replacement_output_roots,
        "rendered_candidate_manifests": replacement_manifest_paths,
        "batch_payload": batch_payload,
        "reason": str(failure_reason),
        "replacement_policy": (
            "candidate_reserve" if stage_name.startswith("al-step-") else "deterministic_next_unused_fresh_dpd"
        ),
    }
    _write_json(replacement_batch_summary_path, replacement_batch_summary)

    replacement_batch_entry = {
        "replacement_batch_index": batch_index,
        "replacement_batch_summary_path": str(replacement_batch_summary_path),
        "sampler": sampler,
        "replica": replica,
        "cycle": resolved_cycle if resolved_cycle is not None else 0,
        "stage": stage_name,
        "reason": str(failure_reason),
        "failed_candidate_indices": list(failed_indices),
        "failed_candidate_ids": failed_candidate_ids_payload,
        "replacement_records": replacement_records_payload,
        "replacement_submission_commands": submission_commands,
        "expected_output_roots": list(replacement_output_roots),
        "replacement_policy": (
            "candidate_reserve" if stage_name.startswith("al-step-") else "deterministic_next_unused_fresh_dpd"
        ),
    }
    batch_summary["replacement_manifest_path"] = str(replacement_manifest_path)
    batch_summary["replacement_records"] = prior_records + replacement_records_payload
    batch_summary["replacement_batches"] = prior_batches + [replacement_batch_entry]
    batch_summary["replacement_count"] = len(batch_summary["replacement_records"])

    replacement_manifest.update(
        {
            "status": "replacement_rendered",
            "campaign_root": str(campaign_root),
            "sampler": sampler,
            "replica": replica,
            "cycle": resolved_cycle if resolved_cycle is not None else 0,
            "stage": stage_name,
            "selection_manifest_path": str(selection_manifest_path),
            "replacement_records": prior_records + replacement_records_payload,
            "quarantine_records": prior_records + replacement_records_payload,
            "replacement_batches": prior_batches + [replacement_batch_entry],
            "replacement_count": len(prior_records) + len(replacement_records_payload),
            "reason": str(failure_reason),
            "replacement_policy": (
                "candidate_reserve" if stage_name.startswith("al-step-") else "deterministic_next_unused_fresh_dpd"
            ),
        }
    )

    _write_json(selection_manifest_path, selection_manifest)
    if stage_name.startswith("al-step-"):
        _write_json(selection_summary_path, selection_summary)
    else:
        _write_json(selection_summary_path, selection_summary)
    _write_json(batch_summary_path, batch_summary)
    _write_json(replacement_manifest_path, replacement_manifest)

    return {
        "campaign_root": str(campaign_root),
        "stage_root": str(stage_root),
        "sampler": sampler,
        "replica": replica,
        "cycle": resolved_cycle if resolved_cycle is not None else 0,
        "stage": stage_name,
        "reason": str(failure_reason),
        "selection_manifest_path": str(selection_manifest_path),
        "batch_summary_path": str(batch_summary_path),
        "replacement_manifest_path": str(replacement_manifest_path),
        "replacement_batch_summary_path": str(replacement_batch_summary_path),
        "replacement_records": replacement_records_payload,
        "replacement_submission_commands": submission_commands,
        "replacement_policy": (
            "candidate_reserve" if stage_name.startswith("al-step-") else "deterministic_next_unused_fresh_dpd"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--replica", required=True, type=int)
    parser.add_argument("--cycle", type=int, default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--candidate-index", action="append", type=int, default=[])
    parser.add_argument("--failed-candidate-id", action="append", default=[])
    parser.add_argument("--reason", default="runtime_failure_or_quarantine")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = render_al_stage_replacements(
        campaign_root=Path(args.campaign_root),
        replica=int(args.replica),
        cycle=int(args.cycle) if args.cycle is not None else None,
        stage=str(args.stage) if args.stage is not None else None,
        failed_candidate_indices=tuple(args.candidate_index),
        failed_candidate_ids=tuple(str(item) for item in args.failed_candidate_id),
        failure_reason=str(args.reason),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
