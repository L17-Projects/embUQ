from __future__ import annotations

"""Reporting utilities for EMB 3.4um paired causal AL-vs-LHS validation."""

import csv
import json
import math
import random
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.emb_34um_causal_validation_ingestion import (
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION,
)
from meso_uq.active_learning.emb_34um_causal_validation_design import (
    EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
    EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE,
    EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
)
from meso_uq.active_learning.emb_34um_final_gate_surrogate import (
    EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
    EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
    train_emb_34um_surrogate_ensemble,
)


EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_causal_validation_report.v1"
EMB_34UM_CAUSAL_VALIDATION_REPORT_FILENAME = "emb_34um_causal_validation_report.json"
EMB_34UM_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME = "emb_34um_causal_validation_summary.csv"
EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_FILENAME = "emb_34um_causal_validation_learning_curves.png"
EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_FILENAME = "emb_34um_causal_validation_final_paired_delta_ci.png"
EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME = "emb_34um_causal_validation_step_delta_ci.png"
EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME = "emb_34um_causal_validation_added_samples_by_step.png"
EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME = "emb_34um_causal_validation_runtime_replacement_diagnostics.png"
EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_FILENAME = "emb_34um_causal_validation_acquisition_selection_diagnostics.png"
EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_learning_curves.png.json"
EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_final_paired_delta_ci.png.json"
EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_step_delta_ci.png.json"
EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_added_samples_by_step.png.json"
EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_runtime_replacement_diagnostics.png.json"
EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_SIDECAR_FILENAME = "emb_34um_causal_validation_acquisition_selection_diagnostics.png.json"

_DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
_DEFAULT_CI_SEED = 2_026_202_405
_DEFAULT_CONFIDENCE = 0.95
_CAUSAL_REQUIRED_STEP_COUNT = EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS
_CAUSAL_REQUIRED_STEP_SIZE = EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE
_CAUSAL_REQUIRED_SHARED_SIZE = EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
_CAUSAL_REQUIRED_VALIDATION_SIZE = EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE

_PLOT_DPI = 120
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}

_STRATEGY_ALIASES = ("strategy", "cohort", "kind", "mode", "source", "gate")
_SEED_ALIASES = ("seed", "selection_seed", "random_seed")
_STEP_ALIASES = ("step", "round", "iteration", "round_index", "iteration_index", "al_step", "lhs_step")
_VALIDATION_SET_ALIASES = (
    "validation_set_id",
    "validation_set",
    "validation_id",
    "validation_case",
    "holdout_set_id",
    "holdout_set",
    "sample_set_id",
    "set_id",
)
_METRIC_ALIASES = (
    "median_curve_rel_l2_pct",
    "held_out_median_curve_rel_l2_pct",
    "holdout_median_curve_rel_l2_pct",
    "curve_rel_l2_pct",
    "force_curve_rel_l2_pct",
    "rel_l2_pct",
)
_ID_ALIASES = ("candidate_id", "curve_id", "sample_id", "id")
_KA_ALIASES = ("ka", "Yt", "yt", "log10_ka", "ka_log10")
_KB_ALIASES = ("kb", "log10_kb", "kb_log10")
_RUNTIME_ALIASES = ("runtime_seconds", "runtime", "wall_time", "seconds", "elapsed_seconds", "elapsed_time", "duration")
_RUNTIME_JOIN_KEY_ALIASES = (
    "runtime_join_key",
    "join_key",
    "selected_candidate_id",
    "candidate_id",
    "sample_candidate_id",
)
_REPLACEMENT_ALIASES = ("replaced", "replacement", "is_replaced", "retry", "requeued", "superseded")
_FAILED_ALIASES = ("failed", "is_failed", "error", "invalid")
_QUARANTINE_ALIASES = ("quarantined", "quarantine", "is_quarantined")
_SOURCE_ALIASES = ("sample_source", "source", "selection_source", "selection_reason")
_STATUS_ALIASES = ("status", "state")
_METHOD_ALIASES = ("method", "stage", "policy_group")
_OUTPUT_ROOT_ALIASES = ("output_root", "output_dir")
_REPLACEMENT_FOR_ALIASES = ("replacement_for", "failed_candidate_id", "replaced_candidate_id", "original_candidate_id")
_CURVE_ALIASES = ("reference_curve", "output_curve", "vertical_diameter")
_FORCE_GRID_ALIASES = ("force_grid", "forces", "force_axis")

_AL_STRATEGY_TOKENS = {"al", "active", "active_learning", "full", "full_gate", "fullgate"}
_LHS_STRATEGY_TOKENS = {"lhs", "left", "baseline"}


@dataclass(frozen=True)
class Emb34umCausalValidationArtifacts:
    artifact_dir: Path
    report_path: Path
    summary_csv_path: Path
    plot_paths: dict[str, str]
    plot_sidecar_paths: dict[str, str]
    report: dict[str, Any]
    seed_step_rows: tuple[dict[str, Any], ...]
    step_rows: tuple[dict[str, Any], ...]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _coerce_rows(
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    *,
    record_keys: tuple[str, ...] = ("rows", "records", "source_rows"),
) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, Path)):
        path = Path(rows)
        if not path.exists():
            raise ValueError(f"rows source does not exist: {path!s}")
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                return tuple(dict(item) for item in csv.DictReader(handle))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return _coerce_rows(loaded, record_keys=record_keys)
    if isinstance(rows, Mapping):
        for key in record_keys:
            value = rows.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        raise ValueError(f"rows payload must contain one of: {', '.join(record_keys)}")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("rows must be a sequence, mapping, or JSON/CSV path.")
    return tuple(dict(item) for item in rows if isinstance(item, Mapping))


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path!s} must contain a JSON object.")
    return dict(payload)


def _coerce_numeric_sequence(value: object, *, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes, bytearray)):
            raise ValueError(f"{label} must decode to a sequence.")
        value = parsed
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    result: list[float] = []
    for item in value:
        number = _coerce_float(item, label=label)
        if number is None:
            raise ValueError(f"{label} entries must be numeric.")
        result.append(float(number))
    if not result:
        raise ValueError(f"{label} must be non-empty.")
    return tuple(result)


def _normalize_status(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"completed", "complete", "success", "succeeded", "ok"}:
        return "completed"
    if text in {"failed", "failure", "error", "missing", "cancelled", "timeout"}:
        return "failed"
    return text or "unknown"


def _method_from_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "shared" in text:
        return "shared_initial"
    if "validation" in text or "holdout" in text:
        return "validation"
    if "lhs" in text or "baseline" in text:
        return "lhs"
    if "al" in text or "active" in text:
        return "al"
    return text


def _step_from_method(method_text: str) -> int | None:
    normalized = method_text.replace("_", "-")
    if "step-" not in normalized:
        return None
    suffix = normalized.rsplit("step-", 1)[-1]
    token = "".join(char for char in suffix if char.isdigit())
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _extract_output_curve(result_payload: Mapping[str, Any], *, label: str) -> tuple[float, ...]:
    for alias in _CURVE_ALIASES:
        if alias in result_payload and result_payload.get(alias) not in (None, ""):
            return _coerce_numeric_sequence(result_payload[alias], label=f"{label}.{alias}")
    raise ValueError(f"{label} is missing reference curve payload.")


def _extract_force_grid_payload(result_payload: Mapping[str, Any], *, label: str) -> tuple[float, ...]:
    for alias in _FORCE_GRID_ALIASES:
        if alias in result_payload and result_payload.get(alias) not in (None, ""):
            return _coerce_numeric_sequence(result_payload[alias], label=f"{label}.{alias}")
    raise ValueError(f"{label} is missing force grid payload.")


def _curve_row_from_result_payload(
    payload: Mapping[str, Any],
    *,
    fallback_candidate_id: str,
    label: str,
) -> dict[str, Any]:
    names = payload.get("parameter_names")
    values = payload.get("parameters")
    params: dict[str, float] = {}
    if isinstance(names, Sequence) and isinstance(values, Sequence) and not isinstance(names, (str, bytes, bytearray)):
        for name, value in zip(names, values):
            params[str(name)] = float(_coerce_float(value, label=f"{label}.parameters") or 0.0)
    if "ka" not in params and payload.get("ka") not in (None, ""):
        params["ka"] = float(_coerce_float(payload.get("ka"), label=f"{label}.ka") or 0.0)
    if "kb" not in params and payload.get("kb") not in (None, ""):
        params["kb"] = float(_coerce_float(payload.get("kb"), label=f"{label}.kb") or 0.0)
    if "ka" not in params or "kb" not in params:
        raise ValueError(f"{label} is missing ka/kb.")

    force_grid = _extract_force_grid_payload(payload, label=label)
    reference_curve = _extract_output_curve(payload, label=label)
    if len(force_grid) != len(reference_curve):
        raise ValueError(f"{label} force-grid and curve lengths differ.")
    return {
        "curve_id": str(payload.get("candidate_id") or fallback_candidate_id),
        "candidate_id": str(payload.get("candidate_id") or fallback_candidate_id),
        "ka": float(params["ka"]),
        "kb": float(params["kb"]),
        "force_grid": force_grid,
        "reference_curve": reference_curve,
    }


def _curve_row_from_ingestion_record(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _first_text(row, aliases=_ID_ALIASES) or "unknown_candidate"
    force_grid_raw = next((row.get(alias) for alias in _FORCE_GRID_ALIASES if row.get(alias) not in (None, "")), None)
    reference_curve_raw = next((row.get(alias) for alias in _CURVE_ALIASES if row.get(alias) not in (None, "")), None)
    if force_grid_raw is not None and reference_curve_raw is not None:
        force_grid = _coerce_numeric_sequence(force_grid_raw, label=f"{candidate_id}.force_grid")
        reference_curve = _coerce_numeric_sequence(reference_curve_raw, label=f"{candidate_id}.reference_curve")
        ka = _coerce_float(_first_text(row, aliases=_KA_ALIASES), label=f"{candidate_id}.ka")
        kb = _coerce_float(_first_text(row, aliases=_KB_ALIASES), label=f"{candidate_id}.kb")
        if ka is None or kb is None:
            raise ValueError(f"{candidate_id} inline curve row is missing ka/kb.")
        if len(force_grid) != len(reference_curve):
            raise ValueError(f"{candidate_id} inline force-grid and curve lengths differ.")
        return {
            "curve_id": candidate_id,
            "candidate_id": candidate_id,
            "ka": float(ka),
            "kb": float(kb),
            "force_grid": force_grid,
            "reference_curve": reference_curve,
        }

    output_root_text = _first_text(row, aliases=_OUTPUT_ROOT_ALIASES)
    if output_root_text is None:
        raise ValueError(f"{candidate_id} has no output_root.")
    result_path = Path(output_root_text) / "emb_34um_result.json"
    payload = _read_json_object(result_path)
    return _curve_row_from_result_payload(payload, fallback_candidate_id=candidate_id, label=str(result_path))


def _policy_int(policy: Mapping[str, Any], key: str, *, default: int) -> int:
    value = policy.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy.{key} must be an integer.") from exc
    if parsed < 1:
        raise ValueError(f"policy.{key} must be positive.")
    return parsed


def _causal_policy_from_ingestion_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    policy_payload = payload.get("policy")
    if not isinstance(policy_payload, Mapping):
        campaign_manifest_path = str(payload.get("campaign_manifest_path") or "").strip()
        if campaign_manifest_path:
            try:
                campaign_manifest = _read_json_object(campaign_manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                campaign_manifest = {}
            policy_payload = campaign_manifest.get("policy")
    policy = policy_payload if isinstance(policy_payload, Mapping) else {}
    return {
        "step_count": _policy_int(policy, "step_count", default=_CAUSAL_REQUIRED_STEP_COUNT),
        "step_size": _policy_int(policy, "step_size", default=_CAUSAL_REQUIRED_STEP_SIZE),
        "shared_size": _policy_int(policy, "shared_size", default=_CAUSAL_REQUIRED_SHARED_SIZE),
        "validation_size": _policy_int(policy, "validation_size", default=_CAUSAL_REQUIRED_VALIDATION_SIZE),
    }


def _assess_ingestion_seed_pairing(
    seed_payload: Mapping[str, Any],
    seed: int,
    *,
    policy: Mapping[str, int],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    shared_rows = list(seed_payload.get("shared_initial", ()))
    validation_rows = list(seed_payload.get("validation", ()))
    al_steps = {int(step): list(rows) for step, rows in seed_payload.get("al", {}).items()}
    lhs_steps = {int(step): list(rows) for step, rows in seed_payload.get("lhs", {}).items()}

    required_step_count = _policy_int(policy, "step_count", default=_CAUSAL_REQUIRED_STEP_COUNT)
    required_step_size = _policy_int(policy, "step_size", default=_CAUSAL_REQUIRED_STEP_SIZE)
    required_shared_size = _policy_int(policy, "shared_size", default=_CAUSAL_REQUIRED_SHARED_SIZE)
    required_validation_size = _policy_int(policy, "validation_size", default=_CAUSAL_REQUIRED_VALIDATION_SIZE)

    shared_count = len(shared_rows)
    validation_count = len(validation_rows)
    al_step_counts = {step: len(rows) for step, rows in al_steps.items()}
    lhs_step_counts = {step: len(rows) for step, rows in lhs_steps.items()}
    al_total = sum(al_step_counts.values())
    lhs_total = sum(lhs_step_counts.values())

    expected_steps = tuple(range(1, required_step_count + 1))
    expected_per_strategy_total = required_step_count * required_step_size
    is_full_scale_candidate = (
        shared_count >= required_shared_size
        and validation_count >= required_validation_size
        and al_total >= expected_per_strategy_total
        and lhs_total >= expected_per_strategy_total
    )

    violations: list[str] = []
    if is_full_scale_candidate:
        if shared_count != required_shared_size:
            violations.append(f"shared_initial_count_{shared_count}_expected_{required_shared_size}")
        if validation_count != required_validation_size:
            violations.append(f"validation_count_{validation_count}_expected_{required_validation_size}")
        if al_total != expected_per_strategy_total:
            violations.append(f"al_total_count_{al_total}_expected_{expected_per_strategy_total}")
        if lhs_total != expected_per_strategy_total:
            violations.append(f"lhs_total_count_{lhs_total}_expected_{expected_per_strategy_total}")

        observed_al_steps = tuple(sorted(al_steps))
        observed_lhs_steps = tuple(sorted(lhs_steps))
        for expected_step in expected_steps:
            if al_step_counts.get(expected_step, 0) != required_step_size:
                violations.append(
                    f"seed_{seed}_al_step_{expected_step}_count_{al_step_counts.get(expected_step, 0)}_expected_{required_step_size}"
                )
            if lhs_step_counts.get(expected_step, 0) != required_step_size:
                violations.append(
                    f"seed_{seed}_lhs_step_{expected_step}_count_{lhs_step_counts.get(expected_step, 0)}_expected_{required_step_size}"
                )

        for step in observed_al_steps:
            if step not in expected_steps:
                violations.append(f"seed_{seed}_al_extra_step_{step}")
        for step in observed_lhs_steps:
            if step not in expected_steps:
                violations.append(f"seed_{seed}_lhs_extra_step_{step}")
        for step in expected_steps:
            if step not in observed_al_steps:
                violations.append(f"seed_{seed}_al_missing_step_{step}")
            if step not in observed_lhs_steps:
                violations.append(f"seed_{seed}_lhs_missing_step_{step}")

    return (
        {
            "seed": seed,
            "is_full_scale_candidate": bool(is_full_scale_candidate),
            "shared_initial_count": int(shared_count),
            "validation_count": int(validation_count),
            "al_total_count": int(al_total),
            "lhs_total_count": int(lhs_total),
            "al_step_counts": {str(step): int(count) for step, count in sorted(al_step_counts.items())},
            "lhs_step_counts": {str(step): int(count) for step, count in sorted(lhs_step_counts.items())},
            "pairing_violations": list(violations),
            "pairing_violation_count": int(len(violations)),
        },
        tuple(violations),
    )


def _train_validation_metric(
    *,
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model = train_emb_34um_surrogate_ensemble(
        records=training_rows,
        validation_records=validation_rows,
        seeds=EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
        architecture_names=EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
    )
    fit = model.get("fit")
    fit_runtime = float(getattr(fit, "runtime_seconds", 0.0)) if fit is not None else 0.0
    return {
        "median_curve_rel_l2_pct": float(model["median_curve_rel_l2_pct"]),
        "mean_curve_rel_l2_pct": float(model["mean_curve_rel_l2_pct"]),
        "max_curve_rel_l2_pct": float(model["max_curve_rel_l2_pct"]),
        "runtime_seconds": fit_runtime,
    }


def _curve_records_from_ingestion_payload(
    payload: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    records = _coerce_rows(payload, record_keys=("records", "rows"))
    policy = _causal_policy_from_ingestion_payload(payload)
    selection_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    pairing_profiles: list[dict[str, Any]] = []

    grouped: dict[int, dict[str, Any]] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in records:
        seed = _optional_int(_first_text(row, aliases=_SEED_ALIASES))
        if seed is None:
            skip("missing_seed")
            continue
        method_value = _first_text(row, aliases=_METHOD_ALIASES) or _first_text(row, aliases=("method", "stage")) or ""
        method = _method_from_text(method_value)
        step = _optional_int(_first_text(row, aliases=_STEP_ALIASES))
        if step is None:
            step = _step_from_method(str(method_value))
        if step is None:
            step = 0

        status = _normalize_status(_first_text(row, aliases=_STATUS_ALIASES) or "")
        candidate_id = _first_text(row, aliases=_ID_ALIASES) or f"seed{seed:03d}-{method}-{step:02d}"
        sample_source = _first_text(row, aliases=_SOURCE_ALIASES) or "candidate"
        replacement = bool(_optional_bool(_first_text(row, aliases=_REPLACEMENT_ALIASES)) or False)
        replacement_for = _first_text(row, aliases=_REPLACEMENT_FOR_ALIASES) or ""
        failed = bool(_optional_bool(_first_text(row, aliases=_FAILED_ALIASES)) or status == "failed")
        quarantined = bool(_optional_bool(_first_text(row, aliases=_QUARANTINE_ALIASES)) or False)

        try:
            ka = _coerce_float(_first_text(row, aliases=_KA_ALIASES), label=f"{candidate_id}.ka", required=False)
            kb = _coerce_float(_first_text(row, aliases=_KB_ALIASES), label=f"{candidate_id}.kb", required=False)
            curve_payload = _curve_row_from_ingestion_record(row) if status == "completed" else None
        except Exception:
            skip("invalid_curve_payload")
            curve_payload = None
            ka = _coerce_float(_first_text(row, aliases=_KA_ALIASES), label=f"{candidate_id}.ka", required=False)
            kb = _coerce_float(_first_text(row, aliases=_KB_ALIASES), label=f"{candidate_id}.kb", required=False)

        selection_rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "method": method,
                "strategy": "al" if method == "al" else ("lhs" if method == "lhs" else method),
                "candidate_id": candidate_id,
                "replacement_for": replacement_for,
                "ka": ka,
                "kb": kb,
                "status": status,
                "sample_source": sample_source,
                "replacement": replacement,
                "failed": failed,
                "quarantined": quarantined,
                "runtime_seconds": _coerce_float(_first_text(row, aliases=_RUNTIME_ALIASES), label="runtime_seconds", required=False),
            }
        )

        if curve_payload is None:
            continue
        bucket = grouped.setdefault(int(seed), {"shared_initial": [], "validation": [], "al": {}, "lhs": {}})
        curve_row = {
            **curve_payload,
            "seed": int(seed),
            "step": int(step),
            "method": method,
            "sample_source": sample_source,
            "replacement": bool(replacement),
            "replacement_for": replacement_for,
        }
        if method == "shared_initial":
            bucket["shared_initial"].append(curve_row)
        elif method == "validation":
            bucket["validation"].append(curve_row)
        elif method in {"al", "lhs"}:
            bucket[method].setdefault(int(step), []).append(curve_row)
        else:
            skip("unsupported_method")

    metric_rows: list[dict[str, Any]] = []
    seed_step_diagnostics: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        seed_payload = grouped[seed]
        profile, pairing_violations = _assess_ingestion_seed_pairing(seed_payload, seed, policy=policy)
        pairing_profiles.append(profile)
        if profile["is_full_scale_candidate"] and pairing_violations:
            skipped["seed_pairing_coverage_violation"] = skipped.get("seed_pairing_coverage_violation", 0) + 1
            continue

        shared_rows = list(seed_payload["shared_initial"])
        validation_rows = list(seed_payload["validation"])
        if not shared_rows:
            skip("missing_shared_initial")
            continue
        if not validation_rows:
            skip("missing_validation")
            continue

        al_steps = {int(step): list(rows) for step, rows in seed_payload["al"].items()}
        lhs_steps = {int(step): list(rows) for step, rows in seed_payload["lhs"].items()}
        paired_steps = sorted(set(al_steps).intersection(lhs_steps))
        if not paired_steps:
            skip("missing_paired_steps")
            continue

        for step in paired_steps:
            al_prefix: list[dict[str, Any]] = list(shared_rows)
            lhs_prefix: list[dict[str, Any]] = list(shared_rows)
            for prefix_step in sorted(s for s in al_steps if s <= step):
                al_prefix.extend(al_steps[prefix_step])
            for prefix_step in sorted(s for s in lhs_steps if s <= step):
                lhs_prefix.extend(lhs_steps[prefix_step])

            al_metric = _train_validation_metric(training_rows=al_prefix, validation_rows=validation_rows)
            lhs_metric = _train_validation_metric(training_rows=lhs_prefix, validation_rows=validation_rows)

            seed_step_diagnostics.append(
                {
                    "seed": seed,
                    "step": step,
                    "shared_curve_count": len(shared_rows),
                    "validation_curve_count": len(validation_rows),
                    "al_prefix_curve_count": len(al_prefix),
                    "lhs_prefix_curve_count": len(lhs_prefix),
                }
            )
            metric_rows.append(
                {
                    "strategy": "al",
                    "seed": seed,
                    "step": step,
                    "validation_set_id": f"seed-{seed:03d}-holdout",
                    "candidate_id": f"seed-{seed:03d}-step-{step:02d}-al",
                    "median_curve_rel_l2_pct": float(al_metric["median_curve_rel_l2_pct"]),
                    "runtime_seconds": float(al_metric["runtime_seconds"]),
                }
            )
            metric_rows.append(
                {
                    "strategy": "lhs",
                    "seed": seed,
                    "step": step,
                    "validation_set_id": f"seed-{seed:03d}-holdout",
                    "candidate_id": f"seed-{seed:03d}-step-{step:02d}-lhs",
                    "median_curve_rel_l2_pct": float(lhs_metric["median_curve_rel_l2_pct"]),
                    "runtime_seconds": float(lhs_metric["runtime_seconds"]),
                }
            )

    metadata = {
        "ingestion_manifest_schema_version": str(payload.get("schema_version", "")),
        "source_record_count": len(records),
        "policy": dict(policy),
        "selection_rows": selection_rows,
        "pairing_profiles": tuple(pairing_profiles),
        "seed_step_diagnostics": seed_step_diagnostics,
        "ingestion_skipped_reasons": skipped,
    }
    return tuple(metric_rows), tuple(selection_rows), metadata


def _coerce_text(value: object, *, label: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _first_text(row: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = row.get(alias)
        if value is None or value == "":
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object, *, label: str, required: bool = True) -> float | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{label} must be provided.")
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(value_float):
        raise ValueError(f"{label} must be finite.")
    return value_float


def _optional_bool(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


def _tokenize_strategy(value: object) -> tuple[str, ...]:
    text = _coerce_text(value, label="strategy")
    return tuple(token for token in text.replace("-", " ").replace("_", " ").split() if token)


def _coerce_strategy(row: Mapping[str, Any]) -> str:
    value = _first_text(row, aliases=_STRATEGY_ALIASES)
    if value is not None:
        tokens = {token.lower() for token in _tokenize_strategy(value)}
        if tokens.intersection(_LHS_STRATEGY_TOKENS):
            return "lhs"
        if tokens.intersection(_AL_STRATEGY_TOKENS):
            return "al"

    identifier = _first_text(row, aliases=_ID_ALIASES) or ""
    tokens = {token.lower() for token in _tokenize_strategy(identifier)}
    if tokens.intersection(_LHS_STRATEGY_TOKENS):
        return "lhs"
    if tokens.intersection(_AL_STRATEGY_TOKENS):
        return "al"
    raise ValueError("missing_strategy")


def _coerce_record(row: Mapping[str, Any], *, source_index: int) -> dict[str, Any]:
    strategy = _coerce_strategy(row)
    seed = _optional_int(_first_text(row, aliases=_SEED_ALIASES))
    if seed is None:
        raise ValueError("missing_seed")
    step = _optional_int(_first_text(row, aliases=_STEP_ALIASES))
    if step is None:
        raise ValueError("missing_step")
    validation_set_id = _first_text(row, aliases=_VALIDATION_SET_ALIASES)
    if validation_set_id is None:
        raise ValueError("missing_validation_set_id")

    metric = None
    for alias in _METRIC_ALIASES:
        if row.get(alias) not in (None, ""):
            metric = _coerce_float(row.get(alias), label=alias)
            break
    if metric is None:
        raise ValueError("missing_metric")

    ka = _coerce_float(_first_text(row, aliases=_KA_ALIASES), label="ka", required=False)
    kb = _coerce_float(_first_text(row, aliases=_KB_ALIASES), label="kb", required=False)
    runtime = _coerce_float(_first_text(row, aliases=_RUNTIME_ALIASES), label="runtime_seconds", required=False)
    replacement = _optional_bool(_first_text(row, aliases=_REPLACEMENT_ALIASES))
    source_id = _first_text(row, aliases=_ID_ALIASES) or f"row_{source_index:06d}"

    return {
        "strategy": strategy,
        "seed": int(seed),
        "step": int(step),
        "validation_set_id": str(validation_set_id),
        "source_id": source_id,
        "metric": float(metric),
        "ka": ka,
        "kb": kb,
        "runtime_seconds": runtime,
        "replacement": bool(replacement) if replacement is not None else False,
    }


def _runtime_join_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for alias in _RUNTIME_JOIN_KEY_ALIASES:
        key = _first_text(row, aliases=(alias,))
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _build_runtime_lookup(
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[tuple[int, int, str], float], dict[str, float]]:
    if runtime_rows is None:
        return {}, {}, {}

    rows = _coerce_rows(runtime_rows, record_keys=("runtime_rows", "rows", "records"))
    by_candidate: dict[str, float] = {}
    by_seed_step_strategy: dict[tuple[int, int, str], float] = {}
    by_join_key: dict[str, float] = {}

    for row in rows:
        runtime_value = _coerce_float(_first_text(row, aliases=_RUNTIME_ALIASES), label="runtime_seconds", required=False)
        if runtime_value is None:
            continue

        source_key = _first_text(row, aliases=_ID_ALIASES)
        if source_key is not None:
            by_candidate[source_key] = runtime_value

        strategy = ""
        try:
            strategy = _coerce_strategy(row)
        except Exception:
            strategy = ""
        seed = _optional_int(_first_text(row, aliases=_SEED_ALIASES))
        step = _optional_int(_first_text(row, aliases=_STEP_ALIASES))
        if seed is not None and step is not None and strategy:
            by_seed_step_strategy[(int(seed), int(step), strategy)] = runtime_value
        for key in _runtime_join_keys(row):
            by_join_key[key] = runtime_value
    return by_candidate, by_seed_step_strategy, by_join_key


def _attach_runtime(
    row: Mapping[str, Any],
    *,
    by_candidate: Mapping[str, float],
    by_seed_step_strategy: Mapping[tuple[int, int, str], float],
    by_join_key: Mapping[str, float],
) -> float | None:
    if row.get("runtime_seconds") is not None:
        return float(row["runtime_seconds"])

    source_id = str(row.get("source_id") or "")
    if source_id and source_id in by_candidate:
        return float(by_candidate[source_id])

    lookup_key = (int(row["seed"]), int(row["step"]), str(row["strategy"]))
    if lookup_key in by_seed_step_strategy:
        return float(by_seed_step_strategy[lookup_key])

    for key in _runtime_join_keys(row):
        if key in by_join_key:
            return float(by_join_key[key])
    return None


def _pair_rows_for_seed_steps(
    rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, int], int]:
    by_candidate, by_seed_step_strategy, by_join_key = _build_runtime_lookup(runtime_rows)

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    parsed_rows: list[dict[str, Any]] = []
    skipped_reasons: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        try:
            parsed = _coerce_record(row, source_index=index)
        except ValueError as exc:
            reason = str(exc)
            if reason.startswith("missing_"):
                reason = reason.removeprefix("missing_")
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        parsed["runtime_seconds"] = _attach_runtime(
            parsed,
            by_candidate=by_candidate,
            by_seed_step_strategy=by_seed_step_strategy,
            by_join_key=by_join_key,
        )
        parsed_rows.append(parsed)

        key = (parsed["seed"], parsed["step"], parsed["validation_set_id"])
        grouped.setdefault(key, []).append(parsed)

    seed_set_rows: list[dict[str, Any]] = []
    for (seed, step, validation_set_id), items in sorted(grouped.items()):
        al_values = [float(item["metric"]) for item in items if item["strategy"] == "al"]
        lhs_values = [float(item["metric"]) for item in items if item["strategy"] == "lhs"]
        if not al_values or not lhs_values:
            skipped_reasons["unpaired_validation_set_rows"] = skipped_reasons.get("unpaired_validation_set_rows", 0) + 1
            continue

        pair_median_al = statistics.median(al_values)
        pair_median_lhs = statistics.median(lhs_values)
        pair_delta = float(pair_median_al - pair_median_lhs)

        row_runtimes = [float(item["runtime_seconds"]) for item in items if item.get("runtime_seconds") is not None]
        seed_set_rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "validation_set_id": str(validation_set_id),
                "al_median_curve_rel_l2_pct": float(pair_median_al),
                "lhs_median_curve_rel_l2_pct": float(pair_median_lhs),
                "paired_seed_delta": float(pair_delta),
                "validation_row_count": int(len(items)),
                "runtime_seconds_count": int(len(row_runtimes)),
                "runtime_seconds_median": statistics.median(row_runtimes) if row_runtimes else None,
                "replacement_count": int(sum(int(bool(item.get("replacement", False))) for item in items)),
                "ka": next((float(item["ka"]) for item in items if item.get("ka") is not None), None),
                "kb": next((float(item["kb"]) for item in items if item.get("kb") is not None), None),
            }
        )

    by_seed_step: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in seed_set_rows:
        by_seed_step.setdefault((row["seed"], row["step"]), []).append(row)

    seed_step_rows: list[dict[str, Any]] = []
    for (seed, step), rows_for_seed_step in sorted(by_seed_step.items()):
        seed_delta_list = [float(item["paired_seed_delta"]) for item in rows_for_seed_step]
        seed_step_rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "matched_validation_set_count": int(len(rows_for_seed_step)),
                "al_median_curve_rel_l2_pct": statistics.median(
                    float(item["al_median_curve_rel_l2_pct"]) for item in rows_for_seed_step
                ),
                "lhs_median_curve_rel_l2_pct": statistics.median(
                    float(item["lhs_median_curve_rel_l2_pct"]) for item in rows_for_seed_step
                ),
                "paired_seed_delta": float(statistics.median(seed_delta_list)),
                "pairing_deltas": [float(item["paired_seed_delta"]) for item in rows_for_seed_step],
                "runtime_seconds_count": int(sum(int(item["runtime_seconds_count"]) for item in rows_for_seed_step)),
                "runtime_seconds_median": (
                    statistics.median(float(item["runtime_seconds_median"]) for item in rows_for_seed_step if item["runtime_seconds_median"] is not None)
                    if any(item["runtime_seconds_median"] is not None for item in rows_for_seed_step)
                    else None
                ),
                "replacement_count": int(sum(int(item.get("replacement_count", 0) or 0) for item in rows_for_seed_step)),
            }
        )

    seed_count = len({row["seed"] for row in seed_step_rows})
    return tuple(seed_step_rows), tuple(parsed_rows), skipped_reasons, seed_count


def _merge_selection_diagnostics_into_seed_steps(
    seed_step_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not seed_step_rows or not selection_rows:
        return tuple(dict(row) for row in seed_step_rows)

    diagnostics_by_seed_step: dict[tuple[int, int], dict[str, int]] = {}
    for row in selection_rows:
        seed = _optional_int(row.get("seed"))
        step = _optional_int(row.get("step"))
        if seed is None or step is None or step <= 0:
            continue
        bucket = diagnostics_by_seed_step.setdefault(
            (int(seed), int(step)),
            {"selection_replacement_count": 0, "selection_failed_count": 0, "selection_quarantine_count": 0},
        )
        if bool(row.get("replacement")):
            bucket["selection_replacement_count"] += 1
        if bool(row.get("failed")):
            bucket["selection_failed_count"] += 1
        if bool(row.get("quarantined")):
            bucket["selection_quarantine_count"] += 1

    merged_rows: list[dict[str, Any]] = []
    for row in seed_step_rows:
        merged = dict(row)
        key = (int(merged["seed"]), int(merged["step"]))
        diagnostics = diagnostics_by_seed_step.get(key, {})
        for field in ("selection_replacement_count", "selection_failed_count", "selection_quarantine_count"):
            merged[field] = int(diagnostics.get(field, 0))
        merged["metric_replacement_count"] = int(merged.get("replacement_count", 0) or 0)
        merged["replacement_count"] = int(merged["metric_replacement_count"]) + int(merged["selection_replacement_count"])
        merged_rows.append(merged)
    return tuple(merged_rows)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0.0, 1.0].")
    values = sorted(float(item) for item in values)
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * p
    left = int(math.floor(index))
    right = int(math.ceil(index))
    if left == right:
        return float(values[left])
    alpha = index - left
    return float(values[left] * (1.0 - alpha) + values[right] * alpha)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    random_seed: int = _DEFAULT_CI_SEED,
) -> dict[str, float]:
    if not values:
        raise ValueError("values must not be empty.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    base_values = [float(item) for item in values]
    n = len(base_values)
    if n == 1:
        value = base_values[0]
        return {
            "point_estimate": value,
            "ci_lower": value,
            "ci_upper": value,
            "n_resamples": int(n_resamples),
            "sample_size": 1,
            "confidence": float(confidence),
            "seed": int(random_seed),
        }

    alpha = 1.0 - confidence
    rng = random.Random(random_seed)
    bootstrap_means: list[float] = []
    for _ in range(int(n_resamples)):
        sample = [base_values[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)
    bootstrap_means.sort()
    return {
        "point_estimate": float(sum(base_values) / n),
        "ci_lower": _percentile(bootstrap_means, alpha / 2.0),
        "ci_upper": _percentile(bootstrap_means, 1.0 - alpha / 2.0),
        "n_resamples": int(n_resamples),
        "sample_size": int(n),
        "confidence": float(confidence),
        "seed": int(random_seed),
    }


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + type_code
        + data
        + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)
    )


def _fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 16
    height = 16
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows: list[bytes] = []
    for row_index in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((200, 220, 240) if (row_index + x) % 2 else (12, 58, 80))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _plot_learning_curves(path: Path, seed_step_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not seed_step_rows:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    steps: set[int] = set()
    per_seed: dict[int, list[tuple[int, float, float]]] = {}
    step_to_al_values: dict[int, list[float]] = {}
    step_to_lhs_values: dict[int, list[float]] = {}

    for row in seed_step_rows:
        seed = int(row["seed"])
        step = int(row["step"])
        steps.add(step)
        per_seed.setdefault(seed, []).append(
            (
                step,
                float(row["al_median_curve_rel_l2_pct"]),
                float(row["lhs_median_curve_rel_l2_pct"]),
            )
        )
        step_to_al_values.setdefault(step, []).append(float(row["al_median_curve_rel_l2_pct"]))
        step_to_lhs_values.setdefault(step, []).append(float(row["lhs_median_curve_rel_l2_pct"]))

    ordered_steps = sorted(steps)
    al_median = [statistics.median(step_to_al_values[step]) for step in ordered_steps]
    lhs_median = [statistics.median(step_to_lhs_values[step]) for step in ordered_steps]
    al_ci_lower = [
        _percentile(step_to_al_values[step], 0.025) if len(step_to_al_values[step]) >= 2 else float(step_to_al_values[step][0])
        for step in ordered_steps
    ]
    al_ci_upper = [
        _percentile(step_to_al_values[step], 0.975) if len(step_to_al_values[step]) >= 2 else float(step_to_al_values[step][0])
        for step in ordered_steps
    ]
    lhs_ci_lower = [
        _percentile(step_to_lhs_values[step], 0.025) if len(step_to_lhs_values[step]) >= 2 else float(step_to_lhs_values[step][0])
        for step in ordered_steps
    ]
    lhs_ci_upper = [
        _percentile(step_to_lhs_values[step], 0.975) if len(step_to_lhs_values[step]) >= 2 else float(step_to_lhs_values[step][0])
        for step in ordered_steps
    ]

    fig, axis = plt.subplots(1, 1, figsize=(10, 4.5))
    for seed in sorted(per_seed):
        seed_points = sorted(per_seed[seed], key=lambda item: item[0])
        axis.plot(
            [item[0] for item in seed_points],
            [item[1] for item in seed_points],
            color="#2b7a78",
            alpha=0.25,
            linewidth=1.2,
            label="AL trace" if seed == sorted(per_seed)[0] else None,
        )
        axis.plot(
            [item[0] for item in seed_points],
            [item[2] for item in seed_points],
            color="#ae2012",
            alpha=0.25,
            linewidth=1.2,
            label="LHS trace" if seed == sorted(per_seed)[0] else None,
        )

    axis.fill_between(ordered_steps, al_ci_lower, al_ci_upper, color="#2b7a78", alpha=0.13, label="AL median 95% seed ribbon")
    axis.fill_between(ordered_steps, lhs_ci_lower, lhs_ci_upper, color="#ae2012", alpha=0.13, label="LHS median 95% seed ribbon")
    axis.plot(ordered_steps, al_median, marker="o", color="#2b7a78", label="AL median")
    axis.plot(ordered_steps, lhs_median, marker="s", color="#ae2012", label="LHS median")
    axis.set_title("EMB 3.4um AL-vs-LHS median held-out force-curve relative L2")
    axis.set_xlabel("step")
    axis.set_ylabel("median held-out force-curve relative L2")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_final_delta_ci(path: Path, final_step: Mapping[str, Any], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not final_step:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    point = float(final_step.get("paired_delta", float("nan")))
    lower = float(final_step.get("paired_delta_ci_lower", point))
    upper = float(final_step.get("paired_delta_ci_upper", point))
    step = int(final_step.get("step", 0))

    fig, axis = plt.subplots(1, 1, figsize=(6.5, 4))
    axis.bar([0], [point], color="#5a8f7b")
    axis.errorbar([0], [point], yerr=[[point - lower], [upper - point]], fmt="none", color="#2f4f4f", capsize=10)
    axis.axhline(0.0, color="#444", linewidth=1.0, linestyle="--")
    axis.set_xticks([0], [f"step {step}"])
    axis.set_ylabel("paired_delta (AL - LHS)")
    axis.set_title("Final paired delta CI")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_step_delta_ci(path: Path, step_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not step_rows:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(step_rows, key=lambda row: int(row["step"]))
    steps = [int(item["step"]) for item in ordered]
    medians = [float(item["paired_delta"]) for item in ordered]
    lower = [float(item["paired_delta_ci_lower"]) for item in ordered]
    upper = [float(item["paired_delta_ci_upper"]) for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(9, 4.5))
    axis.plot(steps, medians, marker="o", color="#38618c", label="seed-wise paired delta")
    axis.fill_between(steps, lower, upper, color="#38618c", alpha=0.2, label="95% seed bootstrap CI")
    axis.axhline(0.0, color="#444", linestyle="--", linewidth=1.0)
    axis.set_title("Per-step paired delta CI")
    axis.set_xlabel("step")
    axis.set_ylabel("paired_delta (AL - LHS)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_added_samples_by_step(
    path: Path,
    *,
    include_plot: bool,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return

    strategy_points: dict[str, list[tuple[float, float, int]]] = {"al": [], "lhs": []}
    for row in rows:
        strategy = str(row.get("strategy") or "").strip().lower()
        if strategy not in strategy_points:
            continue
        status = _normalize_status(row.get("status"))
        if status != "completed":
            continue
        step = _optional_int(row.get("step"))
        if step is None or step <= 0:
            continue
        ka = row.get("ka")
        kb = row.get("kb")
        if ka is None or kb is None:
            continue
        try:
            strategy_points[strategy].append((float(ka), float(kb), int(step)))
        except (TypeError, ValueError):
            continue

    if not strategy_points["al"] and not strategy_points["lhs"]:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, strategy, title in (
        (axes[0], "al", "AL added samples by step"),
        (axes[1], "lhs", "LHS added samples by step"),
    ):
        points = strategy_points[strategy]
        if points:
            scatter = axis.scatter(
                [item[0] for item in points],
                [item[1] for item in points],
                c=[item[2] for item in points],
                cmap="viridis",
                alpha=0.8,
                s=18,
            )
            fig.colorbar(scatter, ax=axis, label="step")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("ka")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("kb")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_runtime_diagnostics(
    path: Path,
    step_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not step_rows:
        path.write_bytes(_fallback_png())
        return

    runtime_rows = [row for row in step_rows if int(float(row.get("step_runtime_count", 0) or 0)) > 0 or int(row.get("step_replacement_count", 0) or 0) > 0]
    if not runtime_rows:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(runtime_rows, key=lambda row: int(row["step"]))
    steps = [int(item["step"]) for item in ordered]
    median_runtime = [float(item.get("step_runtime_median") or 0.0) for item in ordered]
    replacement_count = [int(item.get("step_replacement_count") or 0) for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(9, 4))
    axis.plot(steps, median_runtime, marker="o", label="median runtime (s)")
    axis.plot(steps, replacement_count, marker="s", label="replacement count")
    axis.set_title("Runtime and replacement diagnostics")
    axis.set_xlabel("step")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_acquisition_selection_diagnostics(
    path: Path,
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not selection_rows:
        path.write_bytes(_fallback_png())
        return

    by_step: dict[int, dict[str, int]] = {}
    for row in selection_rows:
        strategy = str(row.get("strategy") or "").strip().lower()
        if strategy != "al":
            continue
        step = _optional_int(row.get("step"))
        if step is None or step <= 0:
            continue
        status = _normalize_status(row.get("status"))
        source = str(row.get("sample_source") or "candidate").strip().lower()
        bucket = by_step.setdefault(step, {"acquisition": 0, "exploration": 0, "candidate": 0, "failed": 0, "quarantine": 0})
        if status == "failed":
            bucket["failed"] += 1
        if bool(row.get("quarantined")):
            bucket["quarantine"] += 1
        if "acquis" in source or "disagree" in source or "ensemble" in source:
            bucket["acquisition"] += 1
        elif "explor" in source:
            bucket["exploration"] += 1
        else:
            bucket["candidate"] += 1

    if not by_step:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered_steps = sorted(by_step)
    acquisition = [by_step[step]["acquisition"] for step in ordered_steps]
    exploration = [by_step[step]["exploration"] for step in ordered_steps]
    candidate = [by_step[step]["candidate"] for step in ordered_steps]
    failed = [by_step[step]["failed"] for step in ordered_steps]
    quarantined = [by_step[step]["quarantine"] for step in ordered_steps]

    fig, axis = plt.subplots(1, 1, figsize=(9, 4.2))
    axis.plot(ordered_steps, acquisition, marker="o", label="acquisition-selected")
    axis.plot(ordered_steps, exploration, marker="s", label="exploration-selected")
    axis.plot(ordered_steps, candidate, marker="^", label="other selected")
    axis.plot(ordered_steps, failed, marker="x", label="failed")
    axis.plot(ordered_steps, quarantined, marker="d", label="quarantine")
    axis.set_title("AL acquisition/selection diagnostics")
    axis.set_xlabel("step")
    axis.set_ylabel("curve count")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "step",
        "seed_count",
        "al_median_curve_rel_l2_pct",
        "lhs_median_curve_rel_l2_pct",
        "paired_delta",
        "paired_delta_ci_lower",
        "paired_delta_ci_upper",
        "paired_seed_count",
        "paired_validation_set_count",
        "step_runtime_count",
        "step_runtime_median",
        "step_replacement_count",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _build_step_rows(
    seed_step_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_args: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in seed_step_rows:
        by_step.setdefault(int(row["step"]), []).append(dict(row))

    summary_rows: list[dict[str, Any]] = []
    for step, rows in sorted(by_step.items()):
        seed_deltas = [float(item["paired_seed_delta"]) for item in rows]
        ci = _bootstrap_ci(seed_deltas, **bootstrap_args)

        summary_rows.append(
            {
                "step": int(step),
                "seed_count": len(rows),
                "paired_seed_count": len(rows),
                "al_median_curve_rel_l2_pct": statistics.median(
                    float(item["al_median_curve_rel_l2_pct"]) for item in rows
                ),
                "lhs_median_curve_rel_l2_pct": statistics.median(
                    float(item["lhs_median_curve_rel_l2_pct"]) for item in rows
                ),
                "paired_delta": float(ci["point_estimate"]),
                "paired_delta_ci_lower": float(ci["ci_lower"]),
                "paired_delta_ci_upper": float(ci["ci_upper"]),
                "paired_validation_set_count": int(sum(int(item["matched_validation_set_count"]) for item in rows)),
                "step_runtime_count": int(sum(int(item.get("runtime_seconds_count", 0) or 0) for item in rows)),
                "step_runtime_median": (
                    statistics.median(float(item["runtime_seconds_median"]) for item in rows if item.get("runtime_seconds_median") is not None)
                    if any(item.get("runtime_seconds_median") is not None for item in rows)
                    else None
                ),
                "step_replacement_count": int(sum(int(item.get("replacement_count", 0) or 0) for item in rows)),
                "step_metric_replacement_count": int(sum(int(item.get("metric_replacement_count", 0) or 0) for item in rows)),
                "step_selection_replacement_count": int(sum(int(item.get("selection_replacement_count", 0) or 0) for item in rows)),
                "step_selection_failed_count": int(sum(int(item.get("selection_failed_count", 0) or 0) for item in rows)),
                "step_selection_quarantine_count": int(sum(int(item.get("selection_quarantine_count", 0) or 0) for item in rows)),
            }
        )
    return tuple(summary_rows)


def _coerce_plot_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: object,
    runtime_source: object | None,
) -> list[dict[str, Any]]:
    source_inputs: list[dict[str, Any]] = []
    if isinstance(source, (str, Path)):
        source_inputs.append({"name": "rows", "type": "file", "path": str(source)})
    elif isinstance(source, Mapping):
        source_rows = _coerce_rows(source, record_keys=("rows", "records", "source_rows"))
        source_inputs.append({"name": "rows", "type": "sequence", "count": len(source_rows)})
    else:
        source_inputs.append({"name": "rows", "type": "sequence", "count": len(rows)})

    if runtime_source is not None:
        if isinstance(runtime_source, (str, Path)):
            source_inputs.append({"name": "runtime_rows", "type": "file", "path": str(runtime_source)})
        elif isinstance(runtime_source, Mapping):
            runtime_rows = _coerce_rows(runtime_source, record_keys=("runtime_rows", "rows", "records"))
            source_inputs.append({"name": "runtime_rows", "type": "sequence", "count": len(runtime_rows)})
        else:
            source_inputs.append({"name": "runtime_rows", "type": "sequence", "count": len(_coerce_rows(runtime_source))})
    return source_inputs


def _emit_plot_sidecar(
    path: Path,
    plot_path: Path,
    *,
    source_inputs: Sequence[Mapping[str, Any]],
    generation_command: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> None:
    _write_json(
        path,
        {
            "plot_path": str(plot_path),
            "source_inputs": list(source_inputs),
            "generation_command": generation_command,
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_confidence": float(bootstrap_confidence),
        },
    )


def _rows_payload_mapping(rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(rows, Mapping):
        return dict(rows)
    if isinstance(rows, (str, Path)):
        path = Path(rows)
        if path.suffix.lower() == ".csv" or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def build_emb_34um_causal_validation_report(
    *,
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None = None,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = _DEFAULT_CI_SEED,
    confidence: float = _DEFAULT_CONFIDENCE,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows_payload = _rows_payload_mapping(rows)
    merged_metadata = dict(metadata or {})
    selection_rows: tuple[dict[str, Any], ...] = tuple()
    is_ingestion_payload = bool(
        rows_payload and str(rows_payload.get("schema_version")) == EMB_34UM_CAUSAL_VALIDATION_INGESTION_SCHEMA_VERSION
    )
    ingestion_policy = _causal_policy_from_ingestion_payload(rows_payload) if is_ingestion_payload and rows_payload else {}
    minimum_step_requirement = int(ingestion_policy.get("step_count", 1)) if is_ingestion_payload else 1

    if is_ingestion_payload and rows_payload:
        source_rows, selection_rows, ingest_metadata = _curve_records_from_ingestion_payload(rows_payload)
        merged_metadata["ingestion"] = {key: value for key, value in ingest_metadata.items() if key != "selection_rows"}
    else:
        source_rows = _coerce_rows(rows, record_keys=("rows", "records", "source_rows"))

    seed_step_rows, parsed_rows, skipped_reasons, seed_count = _pair_rows_for_seed_steps(source_rows, runtime_rows=runtime_rows)
    seed_step_rows = _merge_selection_diagnostics_into_seed_steps(seed_step_rows, selection_rows)
    if is_ingestion_payload:
        for reason, count in merged_metadata.get("ingestion", {}).get("ingestion_skipped_reasons", {}).items():
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + int(count)

    if not source_rows:
        blocked_reasons = {key: value for key, value in skipped_reasons.items() if value}
        if minimum_step_requirement > 0:
            blocked_reasons["minimum_step_count"] = max(blocked_reasons.get("minimum_step_count", 0), minimum_step_requirement)
        return (
            {
                "schema_version": EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
                "status": "blocked",
                "decision": {"status": "blocked", "passed": False, "reason": "no_rows"},
                "source_row_count": 0,
                "usable_pair_row_count": 0,
                "seed_count": 0,
                "step_count": 0,
                "primary_metric": "median_curve_rel_l2_pct",
                "seed_step_rows": tuple(),
                "step_rows": tuple(),
                "final_step": None,
                "blocked_reasons": blocked_reasons,
                "metadata": merged_metadata,
            },
            tuple(),
        )

    step_rows = _build_step_rows(
        seed_step_rows,
        bootstrap_args={"n_resamples": int(bootstrap_resamples), "confidence": float(confidence), "random_seed": int(bootstrap_seed)},
    )

    if not step_rows:
        return (
            {
                "schema_version": EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
                "status": "blocked",
                "decision": {"status": "blocked", "passed": False, "reason": "no_paired_steps"},
                "source_row_count": len(source_rows),
                "usable_pair_row_count": len(seed_step_rows),
                "seed_count": int(seed_count),
                "step_count": 0,
                "primary_metric": "median_curve_rel_l2_pct",
                "seed_step_rows": tuple(),
                "step_rows": tuple(),
                "final_step": None,
                "blocked_reasons": {key: value for key, value in skipped_reasons.items() if value},
                "metadata": merged_metadata,
            },
            tuple(),
        )

    final_step = dict(step_rows[-1])
    final_ci_lower = float(final_step["paired_delta_ci_lower"])
    final_ci_upper = float(final_step["paired_delta_ci_upper"])
    final_seed_count = int(final_step["seed_count"])

    if final_ci_upper < 0.0:
        status = "passed"
        passed = True
    elif final_seed_count == 3 and final_ci_lower <= 0.0 <= final_ci_upper:
        status = "inconclusive"
        passed = False
    else:
        status = "failed"
        passed = False

    if len(step_rows) < minimum_step_requirement:
        status = "blocked"
        passed = False
        skipped_reasons["minimum_step_count"] = max(minimum_step_requirement - len(step_rows), 0)

    report = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": status,
        "decision": {
            "status": status,
            "passed": bool(passed),
            "final_seed_count": final_seed_count,
            "final_step": int(final_step["step"]),
            "paired_delta_ci": {
                "lower": float(final_step["paired_delta_ci_lower"]),
                "upper": float(final_step["paired_delta_ci_upper"]),
                "point_estimate": float(final_step["paired_delta"]),
            },
            "criteria": {
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_resamples": int(bootstrap_resamples),
                "bootstrap_confidence": float(confidence),
                "minimum_step_count": int(minimum_step_requirement),
            },
        },
        "source_row_count": len(source_rows),
        "usable_pair_row_count": len(seed_step_rows),
        "seed_count": int(seed_count),
        "step_count": len(step_rows),
        "primary_metric": "median_curve_rel_l2_pct",
        "blocked_reasons": {key: value for key, value in skipped_reasons.items() if value},
        "seed_step_rows": tuple(dict(item) for item in seed_step_rows),
        "parsed_rows": tuple(dict(item) for item in parsed_rows),
        "selection_rows": tuple(dict(item) for item in selection_rows),
        "step_rows": tuple(dict(item) for item in step_rows),
        "final_step": final_step,
        "coverage_coordinates_present": any(
            item.get("ka") is not None and item.get("kb") is not None for item in selection_rows
        ) or any(
            item.get("ka") is not None and item.get("kb") is not None for item in parsed_rows
        ),
        "runtime_or_replacement_present": any(
            float(item.get("runtime_seconds_count", 0) or 0) > 0
            or int(item.get("replacement_count", 0) or 0) > 0
            for item in seed_step_rows
        ) or any(
            bool(item.get("replacement")) or bool(item.get("failed")) or bool(item.get("quarantined"))
            for item in selection_rows
        ),
        "metadata": merged_metadata,
    }

    return report, tuple(dict(item) for item in step_rows)


def write_emb_34um_causal_validation_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    output_root: str | Path,
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None = None,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = _DEFAULT_CI_SEED,
    confidence: float = _DEFAULT_CONFIDENCE,
    include_plot: bool = True,
    generation_command: str | None = None,
) -> Emb34umCausalValidationArtifacts:
    artifact_dir = Path(output_root)
    report_path = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_REPORT_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME
    learning_curves_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_FILENAME
    final_delta_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_FILENAME
    step_delta_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME
    coverage_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME
    runtime_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME
    acquisition_plot = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_FILENAME

    learning_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_SIDECAR_FILENAME
    final_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_SIDECAR_FILENAME
    step_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME
    coverage_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME
    runtime_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME
    acquisition_sidecar = artifact_dir / EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_SIDECAR_FILENAME

    report, step_rows = build_emb_34um_causal_validation_report(
        rows=rows,
        runtime_rows=runtime_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    source_rows = _coerce_rows(rows, record_keys=("rows", "records", "source_rows"))
    source_inputs = _coerce_plot_inputs(source_rows, source=rows, runtime_source=runtime_rows)
    generation = generation_command or f"write_emb_34um_causal_validation_artifacts(output_root={artifact_dir!s})"

    _write_csv(summary_csv_path, report.get("step_rows", ()))
    _plot_learning_curves(learning_curves_plot, report.get("seed_step_rows", ()), include_plot=include_plot)
    _plot_final_delta_ci(final_delta_plot, report.get("final_step") or {}, include_plot=include_plot)
    _plot_step_delta_ci(step_delta_plot, report.get("step_rows", ()), include_plot=include_plot)
    _plot_added_samples_by_step(
        coverage_plot,
        include_plot=include_plot and bool(report.get("coverage_coordinates_present", False)),
        rows=report.get("selection_rows", ()),
    )
    _plot_runtime_diagnostics(
        runtime_plot,
        report.get("step_rows", ()),
        include_plot=include_plot and bool(report.get("runtime_or_replacement_present", False)),
    )
    _plot_acquisition_selection_diagnostics(
        acquisition_plot,
        report.get("selection_rows", ()),
        include_plot=include_plot and bool(report.get("selection_rows", ())),
    )

    _emit_plot_sidecar(
        learning_sidecar,
        learning_curves_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )
    _emit_plot_sidecar(
        final_sidecar,
        final_delta_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )
    _emit_plot_sidecar(
        step_sidecar,
        step_delta_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )
    _emit_plot_sidecar(
        coverage_sidecar,
        coverage_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )
    _emit_plot_sidecar(
        runtime_sidecar,
        runtime_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )
    _emit_plot_sidecar(
        acquisition_sidecar,
        acquisition_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
    )

    plot_paths = {
        "learning_curves": str(learning_curves_plot),
        "final_delta_ci": str(final_delta_plot),
        "per_step_delta_ci": str(step_delta_plot),
        "added_samples_by_step": str(coverage_plot),
        "runtime_diagnostics": str(runtime_plot),
        "acquisition_selection_diagnostics": str(acquisition_plot),
    }
    plot_sidecar_paths = {
        "learning_curves": str(learning_sidecar),
        "final_delta_ci": str(final_sidecar),
        "per_step_delta_ci": str(step_sidecar),
        "added_samples_by_step": str(coverage_sidecar),
        "runtime_diagnostics": str(runtime_sidecar),
        "acquisition_selection_diagnostics": str(acquisition_sidecar),
    }

    report.update(
        {
            "artifact_dir": str(artifact_dir),
            "summary_csv_path": str(summary_csv_path),
            "plot_paths": plot_paths,
            "plot_sidecar_paths": plot_sidecar_paths,
        }
    )
    _write_json(report_path, report)

    return Emb34umCausalValidationArtifacts(
        artifact_dir=artifact_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        plot_paths=plot_paths,
        plot_sidecar_paths=plot_sidecar_paths,
        report=report,
        seed_step_rows=tuple(dict(item) for item in report.get("seed_step_rows", ())),
        step_rows=tuple(item for item in step_rows),
    )


__all__ = [
    "EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_ACQUISITION_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_FINAL_DELTA_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_LEARNING_CURVES_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_REPORT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION",
    "EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME",
    "Emb34umCausalValidationArtifacts",
    "build_emb_34um_causal_validation_report",
    "write_emb_34um_causal_validation_artifacts",
]
