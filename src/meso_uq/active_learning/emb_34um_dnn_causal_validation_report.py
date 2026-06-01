from __future__ import annotations

"""Reporting utilities for EMB 3.4um DNN causal AL-vs-LHS validation."""

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

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
    EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    validate_dnn_causal_cycle_count,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_report.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME = "emb_34um_dnn_causal_validation_report.json"
EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME = "emb_34um_dnn_causal_validation_summary.csv"
EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_FILENAME = "emb_34um_dnn_causal_validation_al_vs_lhs_curves.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_FILENAME = "emb_34um_dnn_causal_validation_replicate_curves.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME = "emb_34um_dnn_causal_validation_step_delta_ci.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_FILENAME = (
    "emb_34um_dnn_causal_validation_relative_improvement.png"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME = "emb_34um_dnn_causal_validation_runtime_replacement_diagnostics.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_FILENAME = (
    "emb_34um_dnn_causal_validation_parameter_coverage.png"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_FILENAME = (
    "emb_34um_dnn_causal_validation_selection_summary.png"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_JSON_FILENAME = (
    "emb_34um_dnn_causal_validation_artifact_sidecar.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_CSV_FILENAME = (
    "emb_34um_dnn_causal_validation_artifact_sidecar.csv"
)

EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_al_vs_lhs_curves.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_replicate_curves.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_step_delta_ci.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_relative_improvement.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_runtime_replacement_diagnostics.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_parameter_coverage.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_selection_summary.png.json"
)

_DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
_DEFAULT_CI_SEED = 2_026_202_405
_DEFAULT_CONFIDENCE = 0.95
_MIN_FINAL_RELATIVE_IMPROVEMENT = EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT
_PLOT_DPI = 120

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_BRANCH_ALIASES = ("branch", "strategy", "method")
_REPLICATE_ALIASES = ("replicate", "seed", "replica")
_CYCLE_ALIASES = ("cycle", "step", "round", "iteration")
_METRIC_ALIASES = ("relative_l2", "relative_l2_median", "curve_rel_l2", "median_curve_rel_l2", "rel_l2")
_METRIC_MEAN_ALIASES = ("relative_l2_mean", "unseen_force_curve_relative_l2_mean")
_METRIC_MEDIAN_ALIASES = ("relative_l2_median", "relative_l2", "unseen_force_curve_relative_l2_median")
_METRIC_STD_ALIASES = ("relative_l2_std", "unseen_force_curve_relative_l2_std")
_TRAIN_SECONDS_ALIASES = ("train_seconds", "train_time_seconds", "train_time", "train_runtime_seconds")
_SCORE_SECONDS_ALIASES = ("score_seconds", "score_time_seconds", "score_time", "score_runtime_seconds")
_DPD_RUNTIME_MEAN_ALIASES = ("dpd_runtime_seconds_mean", "runtime_seconds_mean")
_DPD_RUNTIME_MEDIAN_ALIASES = ("dpd_runtime_seconds_median", "runtime_seconds_median")
_DPD_RUNTIME_STD_ALIASES = ("dpd_runtime_seconds_std", "runtime_seconds_std")
_REPLACEMENT_ALIASES = ("replacement", "replaced", "is_replaced", "retry", "requeued", "superseded")
_REPLACEMENT_COUNT_ALIASES = ("replacement_count", "replaced_count", "retry_count")
_QUARANTINE_ALIASES = ("quarantine", "quarantined", "is_quarantined", "quarantined_flag")
_QUARANTINE_COUNT_ALIASES = ("quarantine_count", "quarantined_count")
_INITIAL_METRIC_ALIASES = ("initial_relative_l2_median", "initial_relative_l2", "initial_curve_rel_l2")
_INITIAL_TRAIN_SECONDS_ALIASES = ("initial_train_seconds",)
_INITIAL_SCORE_SECONDS_ALIASES = ("initial_score_seconds",)
_FINITE_TEST_COUNT_ALIASES = ("finite_test_count", "unseen_curve_count", "test_curve_count")
_TRAIN_CURVE_COUNT_ALIASES = ("train_curve_count", "train_count", "train_set_count")
_ADDED_COUNT_ALIASES = ("added_candidate_count", "candidate_count", "selection_candidate_count")
_ADDED_CANDIDATE_IDS_ALIASES = ("added_candidate_ids", "candidate_ids", "selected_candidate_ids")
_ADDED_KA_ALIASES = ("added_ka", "ka_values")
_ADDED_KB_ALIASES = ("added_kb", "kb_values")
_ADDED_RADP_ALIASES = ("added_radp", "radp_values")
_ADDED_SHELL_TH_ALIASES = ("added_shell_th", "shell_th_values")
_ADDED_SELECTION_SOURCES_ALIASES = (
    "added_selection_sources",
    "selection_sources",
)
_ADDED_CANDIDATE_POOL_IDS_ALIASES = (
    "added_candidate_pool_ids",
    "candidate_pool_ids",
)
_ADDED_SELECTION_STATUS_ALIASES = (
    "added_selection_status",
    "selection_statuses",
)
_ADDED_SELECTION_PAYLOADS_ALIASES = (
    "added_selection_payloads",
    "selection_payloads",
)
_PAIRED_DELTA_INTERPRETATION = (
    "paired_delta is defined as AL median relative L2 minus LHS median relative L2; "
    "negative values favor AL"
)

_PARAMETER_COVERAGE_PROJECTION_SPECS = (
    ("log10(ka)-vs-log10(kb)", "ka", "kb"),
    ("log10(ka)-vs-log10(radp)", "ka", "radp"),
    ("log10(ka)-vs-log10(shell_th)", "ka", "shell_th"),
    ("log10(kb)-vs-log10(radp)", "kb", "radp"),
    ("log10(kb)-vs-log10(shell_th)", "kb", "shell_th"),
    ("log10(radp)-vs-log10(shell_th)", "radp", "shell_th"),
)

_AL_BRANCH_TOKENS = {"al", "active", "active_learning"}
_LHS_BRANCH_TOKENS = {"lhs", "baseline"}


@dataclass(frozen=True)
class Emb34umDnnCausalValidationArtifacts:
    artifact_dir: Path
    report_path: Path
    summary_csv_path: Path
    plot_paths: dict[str, str]
    plot_sidecar_paths: dict[str, str]
    artifact_sidecar_json_path: Path
    artifact_sidecar_csv_path: Path
    report: dict[str, Any]
    summary_rows: tuple[dict[str, Any], ...]
    paired_cycle_rows: tuple[dict[str, Any], ...]


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _coerce_rows(payload, record_keys=record_keys)
    if isinstance(rows, Mapping):
        for key in record_keys:
            value = rows.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        raise ValueError(f"rows payload must contain one of: {', '.join(record_keys)}")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("rows must be a sequence, mapping, or JSON/CSV path.")
    return tuple(dict(item) for item in rows if isinstance(item, Mapping))


def _coerce_float(value: object, *, label: str, required: bool = True) -> float | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{label} must be provided.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


def _optional_sequence_of_text(value: object, *, label: str, required: bool = False) -> tuple[str, ...] | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{label} must be provided.")
        return None
    if isinstance(value, str):
        raise ValueError(f"{label} must be a sequence.")
    if not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence.")
    return tuple("" if item is None else str(item) for item in value)


def _optional_sequence_of_float(value: object, *, label: str) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        raise ValueError(f"{label} must be a sequence.")
    if not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence.")
    parsed: list[float] = []
    for item in value:
        parsed_value = _coerce_float(item, label=label, required=False)
        if parsed_value is None:
            raise ValueError(f"{label} must contain finite numbers.")
        parsed.append(float(parsed_value))
    return tuple(parsed)


def _optional_sequence_of_mapping(value: object, *, label: str) -> tuple[dict[str, Any], ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        raise ValueError(f"{label} must be a sequence.")
    if not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence.")
    parsed: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            parsed.append(dict(item))
        else:
            raise ValueError(f"{label} must be mappings.")
    return tuple(parsed)


def _first_text(row: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = row.get(alias)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any | None:
    for alias in aliases:
        value = row.get(alias)
        if value in (None, ""):
            continue
        return value
    return None


def _normalize_branch(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    tokens = {token for token in text.replace("_", " ").split() if token}
    if text in _AL_BRANCH_TOKENS or tokens.intersection(_AL_BRANCH_TOKENS):
        return "al"
    if text in _LHS_BRANCH_TOKENS or tokens.intersection(_LHS_BRANCH_TOKENS):
        return "lhs"
    raise ValueError("branch must be al or lhs.")


def _read_metadata(
    metadata: Mapping[str, Any] | None,
    rows_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if rows_payload is not None:
        payload_metadata = rows_payload.get("metadata")
        if isinstance(payload_metadata, Mapping):
            merged.update(dict(payload_metadata))
    if metadata is not None:
        merged.update(dict(metadata))
    ensemble_size = merged.get("ensemble_size")
    if ensemble_size is None:
        raise ValueError("metadata must include ensemble_size=10.")
    try:
        parsed = int(ensemble_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata.ensemble_size must be 10.") from exc
    if parsed != 10:
        raise ValueError("metadata.ensemble_size must be 10.")
    merged["ensemble_size"] = 10
    return merged


def _first_float(row: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> float:
    for alias in aliases:
        if row.get(alias) in (None, ""):
            continue
        value = _coerce_float(row.get(alias), label=label, required=False)
        if value is not None:
            return float(value)
    raise ValueError(f"{label} must be provided.")


def _coerce_record(row: Mapping[str, Any], *, source_index: int) -> dict[str, Any]:
    replicate = _optional_int(_first_value(row, aliases=_REPLICATE_ALIASES))
    if replicate is None:
        raise ValueError("missing_replicate")
    branch = _normalize_branch(_first_text(row, aliases=_BRANCH_ALIASES))
    cycle = _optional_int(_first_value(row, aliases=_CYCLE_ALIASES))
    if cycle is None:
        raise ValueError("missing_cycle")
    metric = _first_float(row, _METRIC_ALIASES, label="relative_l2")
    metric_mean = _coerce_float(
        _first_value(row, aliases=_METRIC_MEAN_ALIASES),
        label="relative_l2_mean",
        required=False,
    )
    metric_median = _coerce_float(
        _first_value(row, aliases=_METRIC_MEDIAN_ALIASES),
        label="relative_l2_median",
        required=False,
    )
    metric_std = _coerce_float(
        _first_value(row, aliases=_METRIC_STD_ALIASES),
        label="relative_l2_std",
        required=False,
    )
    train_seconds = _coerce_float(_first_text(row, aliases=_TRAIN_SECONDS_ALIASES), label="train_seconds", required=False)
    score_seconds = _coerce_float(_first_text(row, aliases=_SCORE_SECONDS_ALIASES), label="score_seconds", required=False)
    dpd_runtime_mean = _coerce_float(
        _first_value(row, aliases=_DPD_RUNTIME_MEAN_ALIASES),
        label="dpd_runtime_seconds_mean",
        required=False,
    )
    dpd_runtime_median = _coerce_float(
        _first_value(row, aliases=_DPD_RUNTIME_MEDIAN_ALIASES),
        label="dpd_runtime_seconds_median",
        required=False,
    )
    dpd_runtime_std = _coerce_float(
        _first_value(row, aliases=_DPD_RUNTIME_STD_ALIASES),
        label="dpd_runtime_seconds_std",
        required=False,
    )
    initial_relative_l2 = _coerce_float(
        _first_text(row, aliases=_INITIAL_METRIC_ALIASES),
        label="initial_relative_l2",
        required=False,
    )
    initial_train_seconds = _coerce_float(
        _first_text(row, aliases=_INITIAL_TRAIN_SECONDS_ALIASES),
        label="initial_train_seconds",
        required=False,
    )
    initial_score_seconds = _coerce_float(
        _first_text(row, aliases=_INITIAL_SCORE_SECONDS_ALIASES),
        label="initial_score_seconds",
        required=False,
    )
    finite_test_count = _optional_int(_first_text(row, aliases=_FINITE_TEST_COUNT_ALIASES))
    train_curve_count = _optional_int(_first_text(row, aliases=_TRAIN_CURVE_COUNT_ALIASES))
    replacement = _optional_bool(_first_value(row, aliases=_REPLACEMENT_ALIASES))
    replacement_count = _optional_int(_first_value(row, aliases=_REPLACEMENT_COUNT_ALIASES))
    quarantine = _optional_bool(_first_value(row, aliases=_QUARANTINE_ALIASES))
    quarantine_count = _optional_int(_first_value(row, aliases=_QUARANTINE_COUNT_ALIASES))
    added_candidate_ids = _optional_sequence_of_text(
        _first_value(row, aliases=_ADDED_CANDIDATE_IDS_ALIASES),
        label="added_candidate_ids",
    )
    added_ka = _optional_sequence_of_float(
        _first_value(row, aliases=_ADDED_KA_ALIASES),
        label="added_ka",
    )
    added_kb = _optional_sequence_of_float(
        _first_value(row, aliases=_ADDED_KB_ALIASES),
        label="added_kb",
    )
    added_radp = _optional_sequence_of_float(
        _first_value(row, aliases=_ADDED_RADP_ALIASES),
        label="added_radp",
    )
    added_shell_th = _optional_sequence_of_float(
        _first_value(row, aliases=_ADDED_SHELL_TH_ALIASES),
        label="added_shell_th",
    )
    added_selection_sources = _optional_sequence_of_text(
        _first_value(row, aliases=_ADDED_SELECTION_SOURCES_ALIASES),
        label="added_selection_sources",
    )
    added_candidate_pool_ids = _optional_sequence_of_text(
        _first_value(row, aliases=_ADDED_CANDIDATE_POOL_IDS_ALIASES),
        label="added_candidate_pool_ids",
    )
    added_selection_status = _optional_sequence_of_text(
        _first_value(row, aliases=_ADDED_SELECTION_STATUS_ALIASES),
        label="added_selection_status",
    )
    added_selection_payloads = _optional_sequence_of_mapping(
        _first_value(row, aliases=_ADDED_SELECTION_PAYLOADS_ALIASES),
        label="added_selection_payloads",
    )

    normalized_replacement_count = (
        int(replacement_count) if replacement_count is not None else (1 if bool(replacement) else 0)
    )
    normalized_quarantine_count = (
        int(quarantine_count) if quarantine_count is not None else (1 if bool(quarantine) else 0)
    )
    provided_added_count = _optional_int(_first_value(row, aliases=_ADDED_COUNT_ALIASES))
    observed_added_count = len(added_candidate_ids) if added_candidate_ids is not None else 0
    if provided_added_count is None:
        inferred_added_count = observed_added_count
    else:
        if provided_added_count < 0:
            raise ValueError("added_candidate_count must be non-negative.")
        if observed_added_count and provided_added_count != observed_added_count:
            raise ValueError("added_candidate_count does not match added_candidate_ids length.")
        inferred_added_count = int(provided_added_count)

    return {
        "replicate": int(replicate),
        "branch": branch,
        "cycle": int(cycle),
        "relative_l2": float(metric),
        "relative_l2_mean": float(metric_mean) if metric_mean is not None else float(metric),
        "relative_l2_median": float(metric_median) if metric_median is not None else float(metric),
        "relative_l2_std": float(metric_std) if metric_std is not None else None,
        "train_seconds": float(train_seconds) if train_seconds is not None else None,
        "score_seconds": float(score_seconds) if score_seconds is not None else None,
        "dpd_runtime_seconds_mean": float(dpd_runtime_mean) if dpd_runtime_mean is not None else None,
        "dpd_runtime_seconds_median": float(dpd_runtime_median) if dpd_runtime_median is not None else None,
        "dpd_runtime_seconds_std": float(dpd_runtime_std) if dpd_runtime_std is not None else None,
        "initial_relative_l2": float(initial_relative_l2) if initial_relative_l2 is not None else None,
        "initial_train_seconds": float(initial_train_seconds) if initial_train_seconds is not None else None,
        "initial_score_seconds": float(initial_score_seconds) if initial_score_seconds is not None else None,
        "finite_test_count": int(finite_test_count) if finite_test_count is not None else None,
        "train_curve_count": int(train_curve_count) if train_curve_count is not None else None,
        "replacement": normalized_replacement_count > 0,
        "replacement_count": normalized_replacement_count,
        "quarantine": normalized_quarantine_count > 0,
        "quarantine_count": normalized_quarantine_count,
        "added_candidate_count": int(inferred_added_count),
        "added_candidate_ids": list(added_candidate_ids) if added_candidate_ids is not None else [],
        "added_ka": list(added_ka) if added_ka is not None else [],
        "added_kb": list(added_kb) if added_kb is not None else [],
        "added_radp": list(added_radp) if added_radp is not None else [],
        "added_shell_th": list(added_shell_th) if added_shell_th is not None else [],
        "added_selection_sources": list(added_selection_sources) if added_selection_sources is not None else [],
        "added_candidate_pool_ids": list(added_candidate_pool_ids) if added_candidate_pool_ids is not None else [],
        "added_selection_status": list(added_selection_status) if added_selection_status is not None else [],
        "added_selection_payloads": list(added_selection_payloads) if added_selection_payloads is not None else [],
        "source_index": int(source_index),
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean([float(value) for value in values]))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median([float(value) for value in values]))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * p
    left = int(math.floor(index))
    right = int(math.ceil(index))
    if left == right:
        return float(ordered[left])
    alpha = index - left
    return float(ordered[left] * (1.0 - alpha) + ordered[right] * alpha)


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

    base_values = [float(value) for value in values]
    if len(base_values) == 1:
        value = base_values[0]
        return {
            "point_estimate": value,
            "ci_lower": value,
            "ci_upper": value,
            "sample_size": 1,
            "n_resamples": int(n_resamples),
            "confidence": float(confidence),
            "seed": int(random_seed),
        }

    rng = random.Random(random_seed)
    sample_size = len(base_values)
    resampled_means: list[float] = []
    for _ in range(int(n_resamples)):
        sample = [base_values[rng.randrange(sample_size)] for _ in range(sample_size)]
        resampled_means.append(sum(sample) / sample_size)
    resampled_means.sort()
    alpha = 1.0 - confidence
    return {
        "point_estimate": float(sum(base_values) / sample_size),
        "ci_lower": _percentile(resampled_means, alpha / 2.0),
        "ci_upper": _percentile(resampled_means, 1.0 - alpha / 2.0),
        "sample_size": int(sample_size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "seed": int(random_seed),
    }


def _trapz_normalized(cycles: Sequence[int], values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(zip(cycles, values), key=lambda item: item[0])
    xs = [float(item[0]) for item in ordered]
    ys = [float(item[1]) for item in ordered]
    span = xs[-1] - xs[0]
    if span <= 0.0:
        return float(statistics.mean(ys))
    area = 0.0
    for left, right in zip(range(len(xs) - 1), range(1, len(xs))):
        width = xs[right] - xs[left]
        area += width * (ys[left] + ys[right]) / 2.0
    return float(area / span)


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
        for col_index in range(width):
            if (row_index + col_index) % 2:
                row.extend((38, 73, 112))
            else:
                row.extend((215, 227, 240))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _summarize_by_branch_cycle(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["replicate"]), str(row["branch"]), int(row["cycle"]))
        grouped.setdefault(key, []).append(dict(row))

    summary_rows: list[dict[str, Any]] = []
    for (replicate, branch, cycle), items in sorted(grouped.items()):
        relative_l2_values = [float(item["relative_l2_median"]) for item in items]
        relative_l2_mean_values = [float(item["relative_l2_mean"]) for item in items]
        relative_l2_std_values = [
            float(item["relative_l2_std"]) for item in items if item.get("relative_l2_std") is not None
        ]
        train_seconds = [float(item["train_seconds"]) for item in items if item.get("train_seconds") is not None]
        score_seconds = [float(item["score_seconds"]) for item in items if item.get("score_seconds") is not None]
        dpd_runtime_mean = [
            float(item["dpd_runtime_seconds_mean"])
            for item in items
            if item.get("dpd_runtime_seconds_mean") is not None
        ]
        dpd_runtime_median = [
            float(item["dpd_runtime_seconds_median"])
            for item in items
            if item.get("dpd_runtime_seconds_median") is not None
        ]
        dpd_runtime_std = [
            float(item["dpd_runtime_seconds_std"])
            for item in items
            if item.get("dpd_runtime_seconds_std") is not None
        ]
        initial_values = [float(item["initial_relative_l2"]) for item in items if item.get("initial_relative_l2") is not None]
        initial_train_seconds = [
            float(item["initial_train_seconds"]) for item in items if item.get("initial_train_seconds") is not None
        ]
        initial_score_seconds = [
            float(item["initial_score_seconds"]) for item in items if item.get("initial_score_seconds") is not None
        ]
        finite_test_counts = [int(item["finite_test_count"]) for item in items if item.get("finite_test_count") is not None]
        summary_rows.append(
            {
                "replicate": int(replicate),
                "branch": str(branch),
                "cycle": int(cycle),
                "relative_l2_mean": _mean(relative_l2_mean_values),
                "relative_l2_median": _median(relative_l2_values),
                "relative_l2_std": _mean(relative_l2_std_values) if relative_l2_std_values else None,
                "relative_l2_min": float(min(relative_l2_values)),
                "relative_l2_max": float(max(relative_l2_values)),
                "row_count": int(len(items)),
                "train_seconds_total": float(sum(train_seconds)) if train_seconds else 0.0,
                "train_seconds_mean": _mean(train_seconds) if train_seconds else None,
                "score_seconds_total": float(sum(score_seconds)) if score_seconds else 0.0,
                "score_seconds_mean": _mean(score_seconds) if score_seconds else None,
                "dpd_runtime_seconds_mean": _mean(dpd_runtime_mean) if dpd_runtime_mean else None,
                "dpd_runtime_seconds_median": _median(dpd_runtime_median) if dpd_runtime_median else None,
                "dpd_runtime_seconds_std": _mean(dpd_runtime_std) if dpd_runtime_std else None,
                "initial_relative_l2_mean": _mean(initial_values) if initial_values else None,
                "initial_train_seconds_mean": _mean(initial_train_seconds) if initial_train_seconds else None,
                "initial_score_seconds_mean": _mean(initial_score_seconds) if initial_score_seconds else None,
                "finite_test_count": int(max(finite_test_counts)) if finite_test_counts else None,
                "replacement_count": int(sum(int(item.get("replacement_count", 0)) for item in items)),
                "quarantine_count": int(sum(int(item.get("quarantine_count", 0)) for item in items)),
            }
        )
    return tuple(summary_rows)


def _pair_replicate_cycle_rows(summary_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, dict[int, dict[str, Mapping[str, Any]]]] = {}
    for row in summary_rows:
        replicate = int(row["replicate"])
        cycle = int(row["cycle"])
        grouped.setdefault(replicate, {}).setdefault(cycle, {})[str(row["branch"])] = dict(row)

    paired_rows: list[dict[str, Any]] = []
    for replicate, rows_by_cycle in sorted(grouped.items()):
        replicate_rows: list[dict[str, Any]] = []
        for cycle, branches in sorted(rows_by_cycle.items()):
            al_row = branches.get("al")
            lhs_row = branches.get("lhs")
            if al_row is None or lhs_row is None:
                continue
            lhs_mean = float(lhs_row["relative_l2_mean"])
            al_mean = float(al_row["relative_l2_mean"])
            lhs_median = float(lhs_row["relative_l2_median"])
            al_median = float(al_row["relative_l2_median"])
            replicate_rows.append(
                {
                    "replicate": int(replicate),
                    "cycle": int(cycle),
                    "al_relative_l2_mean": al_mean,
                    "al_relative_l2_median": al_median,
                    "al_relative_l2_std": al_row.get("relative_l2_std"),
                    "lhs_relative_l2_mean": lhs_mean,
                    "lhs_relative_l2_median": lhs_median,
                    "lhs_relative_l2_std": lhs_row.get("relative_l2_std"),
                    "paired_delta": float(al_median - lhs_median),
                    "paired_mean_delta": float(al_mean - lhs_mean),
                    "mean_relative_improvement": float((lhs_mean - al_mean) / lhs_mean)
                    if not math.isclose(lhs_mean, 0.0)
                    else 0.0,
                    "median_relative_improvement": float((lhs_median - al_median) / lhs_median)
                    if not math.isclose(lhs_median, 0.0)
                    else 0.0,
                    "train_seconds_total": float(al_row["train_seconds_total"]) + float(lhs_row["train_seconds_total"]),
                    "score_seconds_total": float(al_row["score_seconds_total"]) + float(lhs_row["score_seconds_total"]),
                    "al_dpd_runtime_seconds_mean": al_row.get("dpd_runtime_seconds_mean"),
                    "lhs_dpd_runtime_seconds_mean": lhs_row.get("dpd_runtime_seconds_mean"),
                    "replacement_count": int(al_row["replacement_count"]) + int(lhs_row["replacement_count"]),
                    "quarantine_count": int(al_row["quarantine_count"]) + int(lhs_row["quarantine_count"]),
                    "finite_test_count": int(al_row["finite_test_count"])
                    if al_row.get("finite_test_count") is not None
                    else (
                        int(lhs_row["finite_test_count"]) if lhs_row.get("finite_test_count") is not None else None
                    ),
                }
            )

        if not replicate_rows:
            continue

        cycle0_rows = [row for row in replicate_rows if int(row["cycle"]) == 0]
        if cycle0_rows:
            for cycle0_row in cycle0_rows:
                if abs(float(cycle0_row["paired_delta"])) > 0.0:
                    raise ValueError(
                        "cycle_0_paired_delta_must_be_zero: "
                        f"replicate={replicate} delta={float(cycle0_row['paired_delta'])}"
                    )
        else:
            initial_candidates: list[float] = []
            initial_train_seconds: list[float] = []
            initial_score_seconds: list[float] = []
            finite_test_count: int | None = None
            for cycle_rows in rows_by_cycle.values():
                for branch in ("al", "lhs"):
                    branch_row = cycle_rows.get(branch)
                    if branch_row is None:
                        continue
                    if branch_row.get("initial_relative_l2_mean") is not None:
                        initial_candidates.append(float(branch_row["initial_relative_l2_mean"]))
                    if branch_row.get("initial_train_seconds_mean") is not None:
                        initial_train_seconds.append(float(branch_row["initial_train_seconds_mean"]))
                    if branch_row.get("initial_score_seconds_mean") is not None:
                        initial_score_seconds.append(float(branch_row["initial_score_seconds_mean"]))
                    if branch_row.get("finite_test_count") is not None and finite_test_count is None:
                        finite_test_count = int(branch_row["finite_test_count"])

            if initial_candidates:
                first = initial_candidates[0]
                if any(abs(value - first) > 0.0 for value in initial_candidates[1:]):
                    raise ValueError(f"cycle_0_initial_relative_l2_mismatch: replicate={replicate}")
                replicate_rows.append(
                    {
                        "replicate": int(replicate),
                        "cycle": 0,
                        "al_relative_l2_mean": float(first),
                        "al_relative_l2_median": float(first),
                        "lhs_relative_l2_mean": float(first),
                        "lhs_relative_l2_median": float(first),
                        "paired_delta": 0.0,
                        "paired_mean_delta": 0.0,
                        "mean_relative_improvement": 0.0,
                        "median_relative_improvement": 0.0,
                        "train_seconds_total": float(statistics.mean(initial_train_seconds))
                        if initial_train_seconds
                        else 0.0,
                        "score_seconds_total": float(statistics.mean(initial_score_seconds))
                        if initial_score_seconds
                        else 0.0,
                        "replacement_count": 0,
                        "quarantine_count": 0,
                        "finite_test_count": finite_test_count,
                        "synthetic_cycle_0": True,
                    }
                )
            else:
                raise ValueError(f"missing_cycle_0_baseline: replicate={replicate}")

        paired_rows.extend(sorted(replicate_rows, key=lambda item: int(item["cycle"])))
    return tuple(sorted(paired_rows, key=lambda item: (int(item["replicate"]), int(item["cycle"]))))


def _summarize_cycles(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(int(row["cycle"]), []).append(dict(row))

    cycle_rows: list[dict[str, Any]] = []
    for cycle, rows in sorted(grouped.items()):
        al_values = [float(item["al_relative_l2_median"]) for item in rows]
        lhs_values = [float(item["lhs_relative_l2_median"]) for item in rows]
        al_mean_values = [float(item["al_relative_l2_mean"]) for item in rows]
        lhs_mean_values = [float(item["lhs_relative_l2_mean"]) for item in rows]
        deltas = [float(item["paired_delta"]) for item in rows]
        mean_deltas = [float(item.get("paired_mean_delta", item["paired_delta"])) for item in rows]
        relative_improvements = [float(item.get("mean_relative_improvement", 0.0)) for item in rows]
        finite_test_counts = [int(item["finite_test_count"]) for item in rows if item.get("finite_test_count") is not None]
        al_ci = {
            "lower": _percentile(al_values, 0.025) if len(al_values) > 1 else float(al_values[0]),
            "upper": _percentile(al_values, 0.975) if len(al_values) > 1 else float(al_values[0]),
        }
        lhs_ci = {
            "lower": _percentile(lhs_values, 0.025) if len(lhs_values) > 1 else float(lhs_values[0]),
            "upper": _percentile(lhs_values, 0.975) if len(lhs_values) > 1 else float(lhs_values[0]),
        }
        delta_ci = _bootstrap_ci(
            deltas,
            n_resamples=int(bootstrap_resamples),
            confidence=float(confidence),
            random_seed=int(bootstrap_seed),
        )
        mean_delta_ci = _bootstrap_ci(
            mean_deltas,
            n_resamples=int(bootstrap_resamples),
            confidence=float(confidence),
            random_seed=int(bootstrap_seed),
        )
        improvement_ci = _bootstrap_ci(
            relative_improvements,
            n_resamples=int(bootstrap_resamples),
            confidence=float(confidence),
            random_seed=int(bootstrap_seed),
        )
        al_mean = _mean(al_mean_values)
        lhs_mean = _mean(lhs_mean_values)
        cycle_rows.append(
            {
                "cycle": int(cycle),
                "replicate_count": int(len(rows)),
                "active_replicate_count": int(len(rows)),
                "al_relative_l2_mean": al_mean,
                "al_relative_l2_median": _median(al_values),
                "al_relative_l2_ci_lower": float(al_ci["lower"]),
                "al_relative_l2_ci_upper": float(al_ci["upper"]),
                "lhs_relative_l2_mean": lhs_mean,
                "lhs_relative_l2_median": _median(lhs_values),
                "lhs_relative_l2_ci_lower": float(lhs_ci["lower"]),
                "lhs_relative_l2_ci_upper": float(lhs_ci["upper"]),
                "paired_delta": float(delta_ci["point_estimate"]),
                "paired_delta_ci_lower": float(delta_ci["ci_lower"]),
                "paired_delta_ci_upper": float(delta_ci["ci_upper"]),
                "paired_mean_delta": float(mean_delta_ci["point_estimate"]),
                "paired_mean_delta_ci_lower": float(mean_delta_ci["ci_lower"]),
                "paired_mean_delta_ci_upper": float(mean_delta_ci["ci_upper"]),
                "mean_relative_improvement": float((lhs_mean - al_mean) / lhs_mean)
                if not math.isclose(lhs_mean, 0.0)
                else 0.0,
                "mean_relative_improvement_ci_lower": float(improvement_ci["ci_lower"]),
                "mean_relative_improvement_ci_upper": float(improvement_ci["ci_upper"]),
                "median_relative_improvement": float((_median(lhs_values) - _median(al_values)) / _median(lhs_values))
                if not math.isclose(_median(lhs_values), 0.0)
                else 0.0,
                "paired_delta_sample_size": int(delta_ci["sample_size"]),
                "train_seconds_total": float(sum(float(item["train_seconds_total"]) for item in rows)),
                "score_seconds_total": float(sum(float(item["score_seconds_total"]) for item in rows)),
                "replacement_count": int(sum(int(item["replacement_count"]) for item in rows)),
                "quarantine_count": int(sum(int(item["quarantine_count"]) for item in rows)),
                "finite_test_count": int(max(finite_test_counts)) if finite_test_counts else None,
            }
        )
    return tuple(cycle_rows)


def _summarize_replicates(paired_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(int(row["replicate"]), []).append(dict(row))

    replicate_rows: list[dict[str, Any]] = []
    for replicate, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["cycle"]))
        cycles = [int(item["cycle"]) for item in ordered]
        al_curve = [float(item["al_relative_l2_median"]) for item in ordered]
        lhs_curve = [float(item["lhs_relative_l2_median"]) for item in ordered]
        improvement_curve = [float(item.get("mean_relative_improvement", 0.0)) for item in ordered]
        al_auc = _trapz_normalized(cycles, al_curve)
        lhs_auc = _trapz_normalized(cycles, lhs_curve)
        auc_delta = float(al_auc - lhs_auc)
        finite_test_counts = [int(item["finite_test_count"]) for item in ordered if item.get("finite_test_count") is not None]
        replicate_rows.append(
            {
                "replicate": int(replicate),
                "cycles": cycles,
                "al_relative_l2_curve": al_curve,
                "lhs_relative_l2_curve": lhs_curve,
                "mean_relative_improvement_curve": improvement_curve,
                "al_auc": float(al_auc),
                "lhs_auc": float(lhs_auc),
                "auc_delta": auc_delta,
                "area_over_cycles_delta": auc_delta,
                "final_cycle": int(ordered[-1]["cycle"]) if ordered else None,
                "final_cycle_delta": float(ordered[-1]["paired_delta"]) if ordered else None,
                "train_seconds_total": float(sum(float(item["train_seconds_total"]) for item in ordered)),
                "score_seconds_total": float(sum(float(item["score_seconds_total"]) for item in ordered)),
                "replacement_count": int(sum(int(item["replacement_count"]) for item in ordered)),
                "quarantine_count": int(sum(int(item["quarantine_count"]) for item in ordered)),
                "finite_test_count": int(max(finite_test_counts)) if finite_test_counts else None,
            }
        )
    return tuple(replicate_rows)


def _collect_added_selection_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        replicate = int(row["replicate"])
        branch = str(row["branch"])
        cycle = int(row["cycle"])
        candidate_ids = list(row.get("added_candidate_ids", ()))
        ka_values = list(row.get("added_ka", ()))
        kb_values = list(row.get("added_kb", ()))
        radp_values = list(row.get("added_radp", ()))
        shell_th_values = list(row.get("added_shell_th", ()))
        sources = list(row.get("added_selection_sources", ()))
        pool_ids = list(row.get("added_candidate_pool_ids", ()))
        statuses = list(row.get("added_selection_status", ()))
        payloads = list(row.get("added_selection_payloads", ()))
        count = max(
            len(candidate_ids),
            len(ka_values),
            len(kb_values),
            len(radp_values),
            len(shell_th_values),
            len(sources),
            len(pool_ids),
            len(statuses),
            len(payloads),
        )
        if count == 0:
            continue
        for index in range(count):
            point: dict[str, Any] = {
                "replicate": replicate,
                "branch": branch,
                "cycle": cycle,
            }
            if index < len(candidate_ids):
                point["candidate_id"] = candidate_ids[index]
            if index < len(ka_values):
                value = float(ka_values[index])
                point["added_ka"] = value
                point["ka"] = value
            if index < len(kb_values):
                value = float(kb_values[index])
                point["added_kb"] = value
                point["kb"] = value
            if index < len(radp_values):
                value = float(radp_values[index])
                point["added_radp"] = value
                point["radp"] = value
            if index < len(shell_th_values):
                value = float(shell_th_values[index])
                point["added_shell_th"] = value
                point["shell_th"] = value
            if index < len(sources):
                point["selection_source"] = sources[index]
            if index < len(pool_ids):
                point["candidate_pool_id"] = pool_ids[index]
            if index < len(statuses):
                point["selection_status"] = statuses[index]
            if index < len(payloads):
                payload = payloads[index]
                if isinstance(payload, Mapping):
                    point["selection_payload"] = dict(payload)
            if len(point) > 3:
                points.append(point)
    return points


def _parameter_coverage_projection_modes(added_points: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    projection_modes: list[dict[str, Any]] = []
    for mode, x_key, y_key in _PARAMETER_COVERAGE_PROJECTION_SPECS:
        projected_count = 0
        for point in added_points:
            if point.get(x_key) is None or point.get(y_key) is None:
                continue
            projected_count += 1
        projection_modes.append(
            {
                "mode": mode,
                "x_key": x_key,
                "y_key": y_key,
                "x_label": f"log10({x_key})",
                "y_label": f"log10({y_key})",
                "point_count": int(projected_count),
            }
        )
    return tuple(projection_modes)


def _summarize_added_selection(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts_by_source: dict[str, int] = {}
    counts_by_status: dict[str, int] = {}
    counts_by_branch: dict[str, int] = {}
    for point in points:
        branch = str(point.get("branch", ""))
        source = str(point.get("selection_source", "")) or "unknown"
        status = str(point.get("selection_status", "")) or "unknown"
        counts_by_source[source] = counts_by_source.get(source, 0) + 1
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
        counts_by_branch[branch] = counts_by_branch.get(branch, 0) + 1
    return {
        "count": len(points),
        "unique_candidates": len({point.get("candidate_id") for point in points if point.get("candidate_id") is not None}),
        "source_counts": counts_by_source,
        "status_counts": counts_by_status,
        "branch_counts": counts_by_branch,
    }


def _summarize_runtime_replacement(
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_branch: dict[str, dict[str, Any]] = {}
    overall = {
        "row_count": 0,
        "train_seconds_total": 0.0,
        "score_seconds_total": 0.0,
        "replacement_count": 0,
        "quarantine_count": 0,
    }
    for row in summary_rows:
        branch = str(row["branch"])
        bucket = by_branch.setdefault(
            branch,
            {
                "row_count": 0,
                "train_seconds_total": 0.0,
                "score_seconds_total": 0.0,
                "replacement_count": 0,
                "quarantine_count": 0,
            },
        )
        bucket["row_count"] += int(row["row_count"])
        bucket["train_seconds_total"] += float(row["train_seconds_total"] or 0.0)
        bucket["score_seconds_total"] += float(row["score_seconds_total"] or 0.0)
        bucket["replacement_count"] += int(row["replacement_count"])
        bucket["quarantine_count"] += int(row["quarantine_count"])
        overall["row_count"] += int(row["row_count"])
        overall["train_seconds_total"] += float(row["train_seconds_total"] or 0.0)
        overall["score_seconds_total"] += float(row["score_seconds_total"] or 0.0)
        overall["replacement_count"] += int(row["replacement_count"])
        overall["quarantine_count"] += int(row["quarantine_count"])

    def _finish(bucket: Mapping[str, Any]) -> dict[str, Any]:
        row_count = int(bucket["row_count"])
        train_total = float(bucket["train_seconds_total"])
        score_total = float(bucket["score_seconds_total"])
        replacement_count = int(bucket["replacement_count"])
        quarantine_count = int(bucket["quarantine_count"])
        return {
            "row_count": row_count,
            "train_seconds_total": train_total,
            "train_seconds_mean": (train_total / row_count) if row_count else None,
            "score_seconds_total": score_total,
            "score_seconds_mean": (score_total / row_count) if row_count else None,
            "replacement_count": replacement_count,
            "replacement_rate": (replacement_count / row_count) if row_count else None,
            "quarantine_count": quarantine_count,
            "quarantine_rate": (quarantine_count / row_count) if row_count else None,
        }

    return {
        "overall": _finish(overall),
        "by_branch": {branch: _finish(bucket) for branch, bucket in sorted(by_branch.items())},
    }


def _plot_two_curve_learning(
    path: Path,
    cycle_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not cycle_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(cycle_rows, key=lambda row: int(row["cycle"]))
    cycles = [int(item["cycle"]) for item in ordered]
    al_median = [float(item["al_relative_l2_median"]) for item in ordered]
    lhs_median = [float(item["lhs_relative_l2_median"]) for item in ordered]
    al_lower = [float(item["al_relative_l2_ci_lower"]) for item in ordered]
    al_upper = [float(item["al_relative_l2_ci_upper"]) for item in ordered]
    lhs_lower = [float(item["lhs_relative_l2_ci_lower"]) for item in ordered]
    lhs_upper = [float(item["lhs_relative_l2_ci_upper"]) for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(9.0, 4.5))
    axis.fill_between(cycles, al_lower, al_upper, color="#1f77b4", alpha=0.15)
    axis.fill_between(cycles, lhs_lower, lhs_upper, color="#d62728", alpha=0.15)
    axis.plot(cycles, al_median, marker="o", linewidth=2.0, color="#1f77b4", label="AL")
    axis.plot(cycles, lhs_median, marker="s", linewidth=2.0, color="#d62728", label="LHS")
    axis.set_title("AL vs LHS learning curves")
    axis.set_xticks(cycles)
    axis.set_xticklabels([f"{cycle}\n{EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + EMB_34UM_DNN_CAUSAL_STEP_SIZE * cycle}" for cycle in cycles])
    axis.set_xlabel("cycle index / successful training curves")
    axis.set_ylabel("relative L2")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_per_replicate_curves(
    path: Path,
    replicate_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not replicate_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    fig, axis = plt.subplots(1, 1, figsize=(10.0, 4.8))
    for index, row in enumerate(sorted(replicate_rows, key=lambda item: int(item["replicate"]))):
        cycles = [int(value) for value in row["cycles"]]
        al_curve = [float(value) for value in row["al_relative_l2_curve"]]
        lhs_curve = [float(value) for value in row["lhs_relative_l2_curve"]]
        alpha = 0.2 + min(index, 4) * 0.08
        axis.plot(cycles, al_curve, color="#1f77b4", alpha=alpha, linewidth=1.2)
        axis.plot(cycles, lhs_curve, color="#d62728", alpha=alpha, linewidth=1.2)
    axis.set_title("Per-replicate AL and LHS curves")
    axis.set_xlabel("cycle")
    axis.set_ylabel("relative L2")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_step_delta_ci(
    path: Path,
    cycle_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not cycle_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(cycle_rows, key=lambda row: int(row["cycle"]))
    cycles = [int(item["cycle"]) for item in ordered]
    deltas = [float(item["paired_delta"]) for item in ordered]
    lower = [float(item["paired_delta_ci_lower"]) for item in ordered]
    upper = [float(item["paired_delta_ci_upper"]) for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(9.0, 4.5))
    axis.plot(cycles, deltas, marker="o", color="#2c7a7b", linewidth=2.0, label="paired delta")
    axis.fill_between(cycles, lower, upper, color="#2c7a7b", alpha=0.18, label="95% bootstrap CI")
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axis.set_title("Per-cycle paired delta CI")
    axis.set_xlabel("cycle")
    axis.set_ylabel("AL - LHS")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_relative_improvement(
    path: Path,
    cycle_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not cycle_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(cycle_rows, key=lambda row: int(row["cycle"]))
    cycles = [int(item["cycle"]) for item in ordered]
    improvements = [100.0 * float(item.get("mean_relative_improvement", 0.0)) for item in ordered]
    lower = [100.0 * float(item.get("mean_relative_improvement_ci_lower", 0.0)) for item in ordered]
    upper = [100.0 * float(item.get("mean_relative_improvement_ci_upper", 0.0)) for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(9.0, 4.5))
    axis.plot(cycles, improvements, marker="o", color="#2ca02c", linewidth=2.0, label="mean relative improvement")
    axis.fill_between(cycles, lower, upper, color="#2ca02c", alpha=0.18, label="95% bootstrap CI")
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axis.axhline(100.0 * _MIN_FINAL_RELATIVE_IMPROVEMENT, color="#7f7f7f", linestyle=":", linewidth=1.2)
    axis.set_xticks(cycles)
    axis.set_xticklabels([f"{cycle}\n{EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + EMB_34UM_DNN_CAUSAL_STEP_SIZE * cycle}" for cycle in cycles])
    axis.set_title("AL relative improvement over LHS")
    axis.set_xlabel("cycle index / successful training curves")
    axis.set_ylabel("(LHS mean - AL mean) / LHS mean (%)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_runtime_replacement_diagnostics(
    path: Path,
    cycle_rows: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not cycle_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(cycle_rows, key=lambda row: int(row["cycle"]))
    cycles = [int(item["cycle"]) for item in ordered]
    train_seconds = [float(item["train_seconds_total"]) for item in ordered]
    score_seconds = [float(item["score_seconds_total"]) for item in ordered]
    replacement_count = [int(item["replacement_count"]) for item in ordered]
    quarantine_count = [int(item["quarantine_count"]) for item in ordered]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    axes[0].plot(cycles, train_seconds, marker="o", color="#3a6ea5", label="train seconds")
    axes[0].plot(cycles, score_seconds, marker="s", color="#8c564b", label="score seconds")
    axes[0].set_ylabel("seconds")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].bar(cycles, replacement_count, width=0.35, color="#c44e52", label="replacement")
    axes[1].bar([cycle + 0.35 for cycle in cycles], quarantine_count, width=0.35, color="#8172b2", label="quarantine")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("count")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    axes[0].set_title("Runtime and replacement diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_parameter_coverage(
    path: Path,
    added_points: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not added_points:
        path.write_bytes(_fallback_png())
        return
    if not any("ka" in point and "kb" in point for point in added_points):
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    if not any(point.get("radp") is not None and point.get("shell_th") is not None for point in added_points):
        path.write_bytes(_fallback_png())
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.4))
    branch_styles = {
        "al": {"marker": "o", "color": "#1f77b4"},
        "lhs": {"marker": "s", "color": "#d62728"},
    }
    for axis, (mode, x_key, y_key) in zip(axes.flat, _PARAMETER_COVERAGE_PROJECTION_SPECS):
        points_by_branch: dict[str, list[Mapping[str, Any]]] = {}
        for point in added_points:
            if point.get(x_key) is None or point.get(y_key) is None:
                continue
            points_by_branch.setdefault(str(point.get("branch", "unknown")), []).append(point)
        if not points_by_branch:
            axis.set_visible(False)
            continue
        for branch, points in sorted(points_by_branch.items()):
            style = branch_styles.get(branch, {"marker": "^", "color": "#2ca02c"})
            x_values = [float(point[x_key]) for point in points if point.get(x_key) is not None and point.get(y_key) is not None]
            y_values = [float(point[y_key]) for point in points if point.get(x_key) is not None and point.get(y_key) is not None]
            cycles = [int(point["cycle"]) for point in points if point.get(x_key) is not None and point.get(y_key) is not None]
            if not x_values:
                continue
            axis.scatter(
                x_values,
                y_values,
                marker=style["marker"],
                color=style["color"],
                alpha=0.85,
                label=f"{branch} cycle points",
            )
            for index in range(min(4, len(x_values))):
                axis.text(
                    x_values[index],
                    y_values[index],
                    str(cycles[index]),
                    fontsize=7,
                    alpha=0.75,
                    ha="left",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"log10({x_key})")
        axis.set_ylabel(f"log10({y_key})")
        axis.set_title(mode.replace("-vs-", " vs "))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    if metadata.get("parameter_names"):
        names = list(metadata.get("parameter_names"))
        axes.flat[0].set_title(f"{names[0] if names else 'ka'} vs {names[1] if len(names) > 1 else 'kb'}")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _plot_selection_summary(
    path: Path,
    added_points: Sequence[Mapping[str, Any]],
    *,
    include_plot: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not added_points:
        path.write_bytes(_fallback_png())
        return

    source_counts: dict[str, int] = {}
    for point in added_points:
        source = str(point.get("selection_source", "")) or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    if not source_counts:
        path.write_bytes(_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        path.write_bytes(_fallback_png())
        return

    ordered = sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))
    sources = [item[0] for item in ordered]
    counts = [item[1] for item in ordered]

    fig, axis = plt.subplots(1, 1, figsize=(10.0, 4.8))
    axis.barh(sources, counts, color="#4c78a8", alpha=0.9)
    axis.set_xlabel("candidate selection count")
    axis.set_ylabel("selection_source")
    axis.set_title("Selection source summary")
    axis.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=_PLOT_DPI)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "cycle",
        "replicate_count",
        "active_replicate_count",
        "al_relative_l2_mean",
        "al_relative_l2_median",
        "al_relative_l2_ci_lower",
        "al_relative_l2_ci_upper",
        "lhs_relative_l2_mean",
        "lhs_relative_l2_median",
        "lhs_relative_l2_ci_lower",
        "lhs_relative_l2_ci_upper",
        "paired_delta",
        "paired_delta_ci_lower",
        "paired_delta_ci_upper",
        "paired_mean_delta",
        "paired_mean_delta_ci_lower",
        "paired_mean_delta_ci_upper",
        "mean_relative_improvement",
        "mean_relative_improvement_ci_lower",
        "mean_relative_improvement_ci_upper",
        "median_relative_improvement",
        "paired_delta_sample_size",
        "train_seconds_total",
        "score_seconds_total",
        "replacement_count",
        "quarantine_count",
        "finite_test_count",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _emit_plot_sidecar(
    path: Path,
    plot_path: Path,
    *,
    source_inputs: Sequence[Mapping[str, Any]],
    generation_command: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    metadata: Mapping[str, Any],
    extra_payload: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "plot_path": str(plot_path),
        "source_inputs": list(source_inputs),
        "generation_command": generation_command,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_confidence": float(bootstrap_confidence),
        "metadata": dict(metadata),
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    _write_json(path, payload)


def _artifact_sidecar_payload(
    *,
    report: Mapping[str, Any],
    plot_paths: Mapping[str, str],
    plot_sidecar_paths: Mapping[str, str],
) -> dict[str, Any]:
    runtime = dict(report.get("runtime_replacement_summary", {}))
    final_cycle = dict(report.get("final_cycle") or {})
    area_delta = dict(report.get("area_over_cycles_delta") or {})
    metadata = dict(report.get("metadata", {}))
    selection_summary = dict(report.get("selection_summary", {}))
    added_selection_points = list(report.get("added_selection_points", ()))
    selection_projection_modes = list(report.get("selection_projection_modes", ()))
    return {
        "plot_paths": dict(plot_paths),
        "plot_sidecar_paths": dict(plot_sidecar_paths),
        "parameter_names": list(metadata.get("parameter_names", ()) or ()),
        "test_set_source": metadata.get("test_set_source"),
        "replicate_count": int(report.get("replicate_count", 0) or 0),
        "cycle_count": int(report.get("cycle_count", 0) or 0),
        "paired_delta_interpretation": _PAIRED_DELTA_INTERPRETATION,
        "active_replicate_count": int(report.get("replicate_count", 0) or 0),
        "sample_counts": {
            "source_row_count": int(report.get("source_row_count", 0) or 0),
            "usable_pair_row_count": int(report.get("usable_pair_row_count", 0) or 0),
            "finite_test_count": final_cycle.get("finite_test_count"),
        },
        "selection_summary": {
            "point_count": int(selection_summary.get("count", 0) or 0),
            "unique_candidates": int(selection_summary.get("unique_candidates", 0) or 0),
            "source_counts": dict(selection_summary.get("source_counts", {})),
            "status_counts": dict(selection_summary.get("status_counts", {})),
            "branch_counts": dict(selection_summary.get("branch_counts", {})),
            "projection_count": int(len(selection_projection_modes)),
        },
        "selection_projection_modes": selection_projection_modes,
        "added_selection_points": added_selection_points,
        "ci_method": {
            "type": "bootstrap_mean",
            "resamples": int(report.get("decision", {}).get("criteria", {}).get("bootstrap_resamples", 0) or 0),
            "seed": int(report.get("decision", {}).get("criteria", {}).get("bootstrap_seed", 0) or 0),
            "confidence": float(report.get("decision", {}).get("criteria", {}).get("bootstrap_confidence", 0.0) or 0.0),
        },
        "final_cycle_delta": {
            "cycle": final_cycle.get("cycle"),
            "point_estimate": final_cycle.get("paired_delta"),
            "ci_lower": final_cycle.get("paired_delta_ci_lower"),
            "ci_upper": final_cycle.get("paired_delta_ci_upper"),
        },
        "final_cycle_mean_relative_improvement": {
            "cycle": final_cycle.get("cycle"),
            "point_estimate": final_cycle.get("mean_relative_improvement"),
            "ci_lower": final_cycle.get("mean_relative_improvement_ci_lower"),
            "ci_upper": final_cycle.get("mean_relative_improvement_ci_upper"),
            "minimum_required": _MIN_FINAL_RELATIVE_IMPROVEMENT,
        },
        "area_over_cycles_delta": area_delta,
        "replacement_quarantine_by_sampler": runtime.get("by_branch", {}),
        "runtime_split_by_sampler": runtime.get("by_branch", {}),
    }


def _write_artifact_sidecar_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "plot_group",
        "plot_path",
        "plot_sidecar_path",
        "parameter_names",
        "test_set_source",
        "replicate_count",
        "cycle_count",
        "active_replicate_count",
        "source_row_count",
        "usable_pair_row_count",
        "finite_test_count",
        "selection_point_count",
        "selection_unique_candidates",
        "paired_delta_interpretation",
        "ci_type",
        "ci_resamples",
        "ci_seed",
        "ci_confidence",
        "final_cycle",
        "final_cycle_delta",
        "final_cycle_delta_ci_lower",
        "final_cycle_delta_ci_upper",
        "final_cycle_mean_relative_improvement",
        "final_cycle_mean_relative_improvement_ci_lower",
        "final_cycle_mean_relative_improvement_ci_upper",
        "area_over_cycles_delta",
        "area_over_cycles_delta_ci_lower",
        "area_over_cycles_delta_ci_upper",
        "al_replacement_count",
        "lhs_replacement_count",
        "al_quarantine_count",
        "lhs_quarantine_count",
        "al_train_seconds_total",
        "lhs_train_seconds_total",
        "al_score_seconds_total",
        "lhs_score_seconds_total",
    )
    plot_paths = dict(payload.get("plot_paths", {}))
    plot_sidecar_paths = dict(payload.get("plot_sidecar_paths", {}))
    counts = dict(payload.get("sample_counts", {}))
    ci_method = dict(payload.get("ci_method", {}))
    final_delta = dict(payload.get("final_cycle_delta", {}))
    final_improvement = dict(payload.get("final_cycle_mean_relative_improvement", {}))
    area_delta = dict(payload.get("area_over_cycles_delta", {}))
    by_branch = dict(payload.get("runtime_split_by_sampler", {}))
    selection_summary = dict(payload.get("selection_summary", {}))
    al_branch = dict(by_branch.get("al", {}))
    lhs_branch = dict(by_branch.get("lhs", {}))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(plot_paths):
            writer.writerow(
                {
                    "plot_group": key,
                    "plot_path": plot_paths.get(key),
                    "plot_sidecar_path": plot_sidecar_paths.get(key),
                    "parameter_names": "|".join(str(name) for name in payload.get("parameter_names", ())),
                    "test_set_source": payload.get("test_set_source"),
                    "replicate_count": payload.get("replicate_count"),
                    "cycle_count": payload.get("cycle_count"),
                    "active_replicate_count": payload.get("active_replicate_count"),
                    "source_row_count": counts.get("source_row_count"),
                    "usable_pair_row_count": counts.get("usable_pair_row_count"),
                    "finite_test_count": counts.get("finite_test_count"),
                    "selection_point_count": int(selection_summary.get("point_count", 0) or 0),
                    "selection_unique_candidates": int(selection_summary.get("unique_candidates", 0) or 0),
                    "paired_delta_interpretation": payload.get("paired_delta_interpretation"),
                    "ci_type": ci_method.get("type"),
                    "ci_resamples": ci_method.get("resamples"),
                    "ci_seed": ci_method.get("seed"),
                    "ci_confidence": ci_method.get("confidence"),
                    "final_cycle": final_delta.get("cycle"),
                    "final_cycle_delta": final_delta.get("point_estimate"),
                    "final_cycle_delta_ci_lower": final_delta.get("ci_lower"),
                    "final_cycle_delta_ci_upper": final_delta.get("ci_upper"),
                    "final_cycle_mean_relative_improvement": final_improvement.get("point_estimate"),
                    "final_cycle_mean_relative_improvement_ci_lower": final_improvement.get("ci_lower"),
                    "final_cycle_mean_relative_improvement_ci_upper": final_improvement.get("ci_upper"),
                    "area_over_cycles_delta": area_delta.get("point_estimate"),
                    "area_over_cycles_delta_ci_lower": area_delta.get("ci_lower"),
                    "area_over_cycles_delta_ci_upper": area_delta.get("ci_upper"),
                    "al_replacement_count": al_branch.get("replacement_count"),
                    "lhs_replacement_count": lhs_branch.get("replacement_count"),
                    "al_quarantine_count": al_branch.get("quarantine_count"),
                    "lhs_quarantine_count": lhs_branch.get("quarantine_count"),
                    "al_train_seconds_total": al_branch.get("train_seconds_total"),
                    "lhs_train_seconds_total": lhs_branch.get("train_seconds_total"),
                    "al_score_seconds_total": al_branch.get("score_seconds_total"),
                    "lhs_score_seconds_total": lhs_branch.get("score_seconds_total"),
                }
            )


def _coerce_plot_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: object,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    if isinstance(source, (str, Path)):
        inputs.append({"name": "rows", "type": "file", "path": str(source)})
    elif isinstance(source, Mapping):
        inputs.append({"name": "rows", "type": "mapping", "keys": sorted(str(key) for key in source.keys())})
    else:
        inputs.append({"name": "rows", "type": "sequence", "count": len(rows)})
    inputs.append({"name": "metadata", "type": "mapping", "keys": sorted(str(key) for key in metadata.keys())})
    return inputs


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


def _assess_protocol_completeness_from_metric_rows(
    *,
    parsed_rows: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    required_cycle_count = validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MAX_CYCLES)
    required_replicate_count = int(EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT)
    required_replicates = tuple(range(1, required_replicate_count + 1))
    required_cycles = tuple(range(1, required_cycle_count + 1))
    blockers: list[str] = []

    cycle_rows_by_cycle = {int(row["cycle"]): dict(row) for row in cycle_rows}
    final_cycle_row = cycle_rows_by_cycle.get(required_cycle_count)
    if final_cycle_row is None:
        blockers.append(
            f"final paired cycle {required_cycle_count} is missing from report rows."
        )
    else:
        final_replicates = int(final_cycle_row.get("replicate_count", 0) or 0)
        if final_replicates != required_replicate_count:
            blockers.append(
                f"final paired replicate count mismatch: expected {required_replicate_count}, observed {final_replicates}."
            )
        final_finite_test_count = final_cycle_row.get("finite_test_count")
        if int(final_finite_test_count or 0) != EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE:
            blockers.append(
                "finite_test_count mismatch at final cycle: "
                f"expected {EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE}, observed {int(final_finite_test_count or 0)}."
            )

    observed_cycle_set = sorted(cycle for cycle in cycle_rows_by_cycle if cycle > 0)
    if observed_cycle_set != list(required_cycles):
        blockers.append(
            f"paired cycle set mismatch: expected {list(required_cycles)}, observed {observed_cycle_set}."
        )

    for branch in ("al", "lhs"):
        branch_rows = [row for row in parsed_rows if str(row["branch"]) == branch and int(row["cycle"]) > 0]
        observed_replicates = sorted({int(row["replicate"]) for row in branch_rows})
        if observed_replicates != list(required_replicates):
            blockers.append(
                f"{branch} replicate set mismatch: expected {list(required_replicates)}, observed {observed_replicates}."
            )
        for replicate in required_replicates:
            cumulative_added = 0
            shared_estimates: list[int] = []
            for cycle in required_cycles:
                rows_for_cycle = [
                    row for row in branch_rows
                    if int(row["replicate"]) == replicate and int(row["cycle"]) == cycle
                ]
                if not rows_for_cycle:
                    blockers.append(f"{branch} missing replicate={replicate}, cycle={cycle}.")
                    continue
                added_count = int(sum(int(row.get("added_candidate_count", 0) or 0) for row in rows_for_cycle))
                if added_count != EMB_34UM_DNN_CAUSAL_STEP_SIZE:
                    blockers.append(
                        f"{branch} added count mismatch for replicate={replicate}, cycle={cycle}: "
                        f"expected {EMB_34UM_DNN_CAUSAL_STEP_SIZE}, observed {added_count}."
                    )
                cumulative_added += added_count
                train_curve_counts = [
                    int(row["train_curve_count"])
                    for row in rows_for_cycle
                    if row.get("train_curve_count") is not None
                ]
                if not train_curve_counts:
                    blockers.append(
                        f"{branch} missing train_curve_count for replicate={replicate}, cycle={cycle}; "
                        "cannot verify shared_initial completeness."
                    )
                    continue
                shared_estimates.append(max(train_curve_counts) - cumulative_added)
            if shared_estimates:
                if any(value != shared_estimates[0] for value in shared_estimates[1:]):
                    blockers.append(
                        f"{branch} shared_initial estimate is inconsistent for replicate={replicate}: {shared_estimates}."
                    )
                if shared_estimates[0] != EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE:
                    blockers.append(
                        f"{branch} shared_initial count mismatch for replicate={replicate}: "
                        f"expected {EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE}, observed {shared_estimates[0]}."
                    )

    metadata_completeness = metadata.get("protocol_completeness")
    if isinstance(metadata_completeness, Mapping):
        if not bool(metadata_completeness.get("passed", False)):
            blockers.append("metrics protocol completeness is not passed in metadata.protocol_completeness.")

    protocol_completeness = {
        "required": {
            "paired_cycle_count": required_cycle_count,
            "required_cycles": list(required_cycles),
            "final_paired_replicate_count": required_replicate_count,
            "unseen_test_count": EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
            "shared_initial_per_replicate": EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
            "step_size": EMB_34UM_DNN_CAUSAL_STEP_SIZE,
        },
        "observed": {
            "paired_cycles": observed_cycle_set,
            "final_cycle_replicate_count": int(
                cycle_rows_by_cycle.get(required_cycle_count, {}).get("replicate_count", 0) or 0
            ),
            "final_cycle_finite_test_count": int(
                cycle_rows_by_cycle.get(required_cycle_count, {}).get("finite_test_count", 0) or 0
            ),
        },
        "passed": not blockers,
    }
    return protocol_completeness, blockers


def build_emb_34um_dnn_causal_validation_report(
    *,
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = _DEFAULT_CI_SEED,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows_payload = _rows_payload_mapping(rows)
    metadata_payload = _read_metadata(metadata, rows_payload)
    source_rows = _coerce_rows(rows, record_keys=("rows", "records", "source_rows"))
    parsed_rows = [_coerce_record(row, source_index=index) for index, row in enumerate(source_rows, start=1)]
    added_selection_points = _collect_added_selection_points(parsed_rows)
    selection_summary = _summarize_added_selection(added_selection_points)
    selection_projection_modes = _parameter_coverage_projection_modes(added_selection_points)

    if not parsed_rows:
        report = {
            "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
            "status": "blocked",
            "interpretation": {"paired_delta": _PAIRED_DELTA_INTERPRETATION},
            "decision": {"status": "blocked", "passed": False, "reason": "no_rows"},
            "source_row_count": 0,
            "usable_pair_row_count": 0,
            "replicate_count": 0,
            "cycle_count": 0,
            "summary_rows": tuple(),
            "paired_cycle_rows": tuple(),
            "cycle_rows": tuple(),
            "replicate_rows": tuple(),
            "final_cycle": None,
            "auc_comparison": None,
            "runtime_replacement_summary": {"overall": {}, "by_branch": {}},
            "blocked_reasons": {"no_rows": 1},
            "selection_summary": dict(selection_summary),
            "added_selection_points": tuple(),
            "selection_projection_modes": tuple(),
            "metadata": metadata_payload,
        }
        return report, tuple()

    summary_rows = _summarize_by_branch_cycle(parsed_rows)
    paired_cycle_rows = _pair_replicate_cycle_rows(summary_rows)
    if not paired_cycle_rows:
        report = {
            "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
            "status": "blocked",
            "interpretation": {"paired_delta": _PAIRED_DELTA_INTERPRETATION},
            "decision": {"status": "blocked", "passed": False, "reason": "no_paired_cycles"},
            "source_row_count": len(parsed_rows),
            "usable_pair_row_count": 0,
            "replicate_count": len({row["replicate"] for row in parsed_rows}),
            "cycle_count": len({row["cycle"] for row in parsed_rows}),
            "summary_rows": tuple(dict(item) for item in summary_rows),
            "paired_cycle_rows": tuple(),
            "cycle_rows": tuple(),
            "replicate_rows": tuple(),
            "final_cycle": None,
            "auc_comparison": None,
            "runtime_replacement_summary": _summarize_runtime_replacement(summary_rows),
            "blocked_reasons": {"no_paired_cycles": 1},
            "selection_summary": dict(selection_summary),
            "added_selection_points": tuple(),
            "selection_projection_modes": tuple(),
            "metadata": metadata_payload,
        }
        return report, tuple()

    cycle_rows = _summarize_cycles(
        paired_cycle_rows,
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
        confidence=float(confidence),
    )
    replicate_rows = _summarize_replicates(paired_cycle_rows)
    final_cycle = dict(cycle_rows[-1])

    final_delta_ci = {
        "point_estimate": float(final_cycle["paired_delta"]),
        "lower": float(final_cycle["paired_delta_ci_lower"]),
        "upper": float(final_cycle["paired_delta_ci_upper"]),
        "sample_size": int(final_cycle["paired_delta_sample_size"]),
        "n_resamples": int(bootstrap_resamples),
        "confidence": float(confidence),
        "seed": int(bootstrap_seed),
    }

    auc_deltas = [float(row["auc_delta"]) for row in replicate_rows]
    auc_ci = _bootstrap_ci(
        auc_deltas,
        n_resamples=bootstrap_resamples,
        confidence=confidence,
        random_seed=bootstrap_seed,
    )
    area_over_cycles_delta = {
        "point_estimate": float(auc_ci["point_estimate"]),
        "ci_lower": float(auc_ci["ci_lower"]),
        "ci_upper": float(auc_ci["ci_upper"]),
        "sample_size": int(auc_ci["sample_size"]),
        "n_resamples": int(auc_ci["n_resamples"]),
        "confidence": float(auc_ci["confidence"]),
        "seed": int(auc_ci["seed"]),
    }

    protocol_completeness, protocol_blockers = _assess_protocol_completeness_from_metric_rows(
        parsed_rows=parsed_rows,
        cycle_rows=cycle_rows,
        metadata=metadata_payload,
    )

    final_mean_improvement = float(final_cycle.get("mean_relative_improvement", 0.0))
    final_improvement_ci_lower = float(final_cycle.get("mean_relative_improvement_ci_lower", 0.0))
    final_improvement_ci_upper = float(final_cycle.get("mean_relative_improvement_ci_upper", 0.0))
    final_delta_ci_upper = float(final_delta_ci["upper"])
    improvement_blockers: list[str] = []
    if final_mean_improvement < _MIN_FINAL_RELATIVE_IMPROVEMENT:
        improvement_blockers.append(
            "final mean relative improvement below 10%: "
            f"observed {final_mean_improvement:.6g}."
        )
    if final_improvement_ci_lower <= 0.0:
        improvement_blockers.append(
            "final relative-improvement confidence interval does not support positive AL improvement."
        )
    if final_delta_ci_upper >= 0.0:
        improvement_blockers.append(
            "final paired AL-minus-LHS confidence interval includes zero or favors LHS."
        )

    decision_blockers = list(protocol_blockers) + improvement_blockers

    if protocol_blockers:
        status = "blocked"
        passed = False
    else:
        status = "passed" if not improvement_blockers else "failed"
        passed = status == "passed"

    runtime_summary = _summarize_runtime_replacement(summary_rows)
    blocked_reasons = {reason: decision_blockers.count(reason) for reason in sorted(set(decision_blockers))}
    report = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": status,
        "interpretation": {"paired_delta": _PAIRED_DELTA_INTERPRETATION},
        "decision": {
            "status": status,
            "passed": bool(passed),
            "blockers": list(decision_blockers),
            "final_cycle": int(final_cycle["cycle"]),
            "final_cycle_delta_ci": {
                "point_estimate": float(final_delta_ci["point_estimate"]),
                "lower": float(final_delta_ci["lower"]),
                "upper": float(final_delta_ci["upper"]),
            },
            "final_cycle_mean_relative_improvement_ci": {
                "point_estimate": final_mean_improvement,
                "lower": final_improvement_ci_lower,
                "upper": final_improvement_ci_upper,
            },
            "auc_delta_ci": {
                "point_estimate": float(auc_ci["point_estimate"]),
                "lower": float(auc_ci["ci_lower"]),
                "upper": float(auc_ci["ci_upper"]),
            },
            "area_over_cycles_delta_ci": {
                "point_estimate": float(auc_ci["point_estimate"]),
                "lower": float(auc_ci["ci_lower"]),
                "upper": float(auc_ci["ci_upper"]),
            },
            "criteria": {
                "bootstrap_resamples": int(bootstrap_resamples),
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_confidence": float(confidence),
                "ensemble_size": 10,
                "ci_method": "bootstrap_mean",
                "minimum_final_mean_relative_improvement": _MIN_FINAL_RELATIVE_IMPROVEMENT,
                "requires_positive_relative_improvement_ci": True,
                "requires_negative_al_minus_lhs_delta_ci": True,
                "protocol_completeness_required": protocol_completeness["required"],
            },
        },
        "source_row_count": len(parsed_rows),
        "usable_pair_row_count": len(paired_cycle_rows),
        "replicate_count": len(replicate_rows),
        "active_replicate_count": len(replicate_rows),
        "cycle_count": len(cycle_rows),
        "finite_test_count": final_cycle.get("finite_test_count"),
        "summary_rows": tuple(dict(item) for item in summary_rows),
        "paired_cycle_rows": tuple(dict(item) for item in paired_cycle_rows),
        "cycle_rows": tuple(dict(item) for item in cycle_rows),
        "replicate_rows": tuple(dict(item) for item in replicate_rows),
        "final_cycle": final_cycle,
        "auc_comparison": {
            "point_estimate": float(auc_ci["point_estimate"]),
            "ci_lower": float(auc_ci["ci_lower"]),
            "ci_upper": float(auc_ci["ci_upper"]),
            "sample_size": int(auc_ci["sample_size"]),
            "n_resamples": int(auc_ci["n_resamples"]),
            "confidence": float(auc_ci["confidence"]),
            "seed": int(auc_ci["seed"]),
        },
        "area_over_cycles_delta": area_over_cycles_delta,
        "final_cycle_delta": {
            "cycle": int(final_cycle["cycle"]),
            "point_estimate": float(final_cycle["paired_delta"]),
            "ci_lower": float(final_cycle["paired_delta_ci_lower"]),
            "ci_upper": float(final_cycle["paired_delta_ci_upper"]),
            "sample_size": int(final_cycle["paired_delta_sample_size"]),
            "n_resamples": int(bootstrap_resamples),
            "confidence": float(confidence),
            "seed": int(bootstrap_seed),
        },
        "runtime_replacement_summary": runtime_summary,
        "runtime_split_by_sampler": runtime_summary.get("by_branch", {}),
        "replacement_quarantine_by_sampler": runtime_summary.get("by_branch", {}),
        "selection_summary": dict(selection_summary),
        "added_selection_points": tuple(dict(item) for item in added_selection_points),
        "selection_projection_modes": tuple(dict(item) for item in selection_projection_modes),
        "protocol_completeness": protocol_completeness,
        "blocked_reasons": blocked_reasons,
        "metadata": metadata_payload,
    }
    return report, tuple(dict(item) for item in cycle_rows)


def write_emb_34um_dnn_causal_validation_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    output_root: str | Path,
    metadata: Mapping[str, Any] | None = None,
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = _DEFAULT_CI_SEED,
    confidence: float = _DEFAULT_CONFIDENCE,
    include_plot: bool = True,
    generation_command: str | None = None,
) -> Emb34umDnnCausalValidationArtifacts:
    artifact_dir = Path(output_root)
    report_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME
    al_vs_lhs_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_FILENAME
    replicate_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_FILENAME
    step_delta_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME
    relative_improvement_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_FILENAME
    runtime_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME
    parameter_coverage_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_FILENAME
    selection_summary_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_FILENAME
    artifact_sidecar_json = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_JSON_FILENAME
    artifact_sidecar_csv = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_CSV_FILENAME

    al_vs_lhs_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME
    replicate_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME
    step_delta_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME
    relative_improvement_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_SIDECAR_FILENAME
    runtime_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME
    parameter_coverage_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_SIDECAR_FILENAME
    selection_summary_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_SIDECAR_FILENAME
    source_rows = _coerce_rows(rows, record_keys=("rows", "records", "source_rows"))

    report, cycle_rows = build_emb_34um_dnn_causal_validation_report(
        rows=rows,
        metadata=metadata,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )

    _write_csv(summary_csv_path, report.get("cycle_rows", ()))
    _plot_two_curve_learning(al_vs_lhs_plot, report.get("cycle_rows", ()), include_plot=include_plot)
    _plot_per_replicate_curves(replicate_plot, report.get("replicate_rows", ()), include_plot=include_plot)
    _plot_step_delta_ci(step_delta_plot, report.get("cycle_rows", ()), include_plot=include_plot)
    _plot_relative_improvement(relative_improvement_plot, report.get("cycle_rows", ()), include_plot=include_plot)
    _plot_runtime_replacement_diagnostics(runtime_plot, report.get("cycle_rows", ()), include_plot=include_plot)
    source_rows_for_plots = source_rows
    parsed_rows_for_plots = [
        _coerce_record(row, source_index=index) for index, row in enumerate(source_rows_for_plots, start=1)
    ]
    added_selection_points = _collect_added_selection_points(parsed_rows_for_plots)
    _plot_parameter_coverage(
        parameter_coverage_plot,
        added_selection_points,
        report.get("metadata", {}),
        include_plot=include_plot,
    )
    _plot_selection_summary(selection_summary_plot, added_selection_points, include_plot=include_plot)

    source_inputs = _coerce_plot_inputs(source_rows, source=rows, metadata=report["metadata"])
    generation = generation_command or f"write_emb_34um_dnn_causal_validation_artifacts(output_root={artifact_dir!s})"

    _emit_plot_sidecar(
        al_vs_lhs_sidecar,
        al_vs_lhs_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
    )
    _emit_plot_sidecar(
        replicate_sidecar,
        replicate_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
    )
    _emit_plot_sidecar(
        parameter_coverage_sidecar,
        parameter_coverage_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
        extra_payload={
            "projection_count": len(report.get("selection_projection_modes", ())),
            "projection_modes": list(report.get("selection_projection_modes", ())),
            "selection_summary": dict(report.get("selection_summary", {})),
            "selection_projection_modes": list(report.get("selection_projection_modes", ())),
            "added_selection_points": list(report.get("added_selection_points", ())),
        },
    )
    _emit_plot_sidecar(
        selection_summary_sidecar,
        selection_summary_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
        extra_payload={
            "projection_count": len(report.get("selection_projection_modes", ())),
            "projection_modes": list(report.get("selection_projection_modes", ())),
            "selection_summary": dict(report.get("selection_summary", {})),
            "selection_projection_modes": list(report.get("selection_projection_modes", ())),
            "added_selection_points": list(report.get("added_selection_points", ())),
        },
    )
    _emit_plot_sidecar(
        step_delta_sidecar,
        step_delta_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
    )
    _emit_plot_sidecar(
        relative_improvement_sidecar,
        relative_improvement_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
        extra_payload={
            "minimum_final_mean_relative_improvement": _MIN_FINAL_RELATIVE_IMPROVEMENT,
            "final_cycle_mean_relative_improvement": report.get("decision", {}).get(
                "final_cycle_mean_relative_improvement_ci", {}
            ),
        },
    )
    _emit_plot_sidecar(
        runtime_sidecar,
        runtime_plot,
        source_inputs=source_inputs,
        generation_command=generation,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_confidence=float(confidence),
        metadata=report["metadata"],
    )

    plot_paths = {
        "al_vs_lhs_curves": str(al_vs_lhs_plot),
        "per_replicate_curves": str(replicate_plot),
        "step_delta_ci": str(step_delta_plot),
        "relative_improvement": str(relative_improvement_plot),
        "runtime_replacement_diagnostics": str(runtime_plot),
        "parameter_coverage": str(parameter_coverage_plot),
        "selection_summary": str(selection_summary_plot),
    }
    plot_sidecar_paths = {
        "al_vs_lhs_curves": str(al_vs_lhs_sidecar),
        "per_replicate_curves": str(replicate_sidecar),
        "step_delta_ci": str(step_delta_sidecar),
        "relative_improvement": str(relative_improvement_sidecar),
        "runtime_replacement_diagnostics": str(runtime_sidecar),
        "parameter_coverage": str(parameter_coverage_sidecar),
        "selection_summary": str(selection_summary_sidecar),
    }
    artifact_sidecar_payload = _artifact_sidecar_payload(
        report=report,
        plot_paths=plot_paths,
        plot_sidecar_paths=plot_sidecar_paths,
    )
    _write_json(artifact_sidecar_json, artifact_sidecar_payload)
    _write_artifact_sidecar_csv(artifact_sidecar_csv, artifact_sidecar_payload)

    report.update(
        {
            "artifact_dir": str(artifact_dir),
            "summary_csv_path": str(summary_csv_path),
            "plot_paths": plot_paths,
            "plot_sidecar_paths": plot_sidecar_paths,
            "artifact_sidecar_json_path": str(artifact_sidecar_json),
            "artifact_sidecar_csv_path": str(artifact_sidecar_csv),
        }
    )
    _write_json(report_path, report)

    return Emb34umDnnCausalValidationArtifacts(
        artifact_dir=artifact_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        plot_paths=plot_paths,
        plot_sidecar_paths=plot_sidecar_paths,
        artifact_sidecar_json_path=artifact_sidecar_json,
        artifact_sidecar_csv_path=artifact_sidecar_csv,
        report=report,
        summary_rows=tuple(dict(item) for item in report.get("summary_rows", ())),
        paired_cycle_rows=tuple(dict(item) for item in report.get("paired_cycle_rows", ())),
    )


__all__ = [
    "EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_CSV_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_ARTIFACT_SIDECAR_JSON_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RELATIVE_IMPROVEMENT_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_PARAMETER_COVERAGE_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_SUMMARY_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME",
    "Emb34umDnnCausalValidationArtifacts",
    "build_emb_34um_dnn_causal_validation_report",
    "write_emb_34um_dnn_causal_validation_artifacts",
]
