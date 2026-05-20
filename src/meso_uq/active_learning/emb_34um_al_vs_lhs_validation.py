from __future__ import annotations

"""EMB 3.4um AL-vs-LHS validation artifacts and plots."""

import csv
import json
import math
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_al_vs_lhs_validation.v1"
EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME = "emb_34um_al_vs_lhs_validation_manifest.json"
EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME = "emb_34um_al_vs_lhs_validation_summary.csv"
EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation.png"
EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME = "emb_34um_al_vs_lhs_validation.png.json"
EMB_34UM_AL_VS_LHS_VALIDATION_SAMPLES_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_samples_ka_kb.png"
EMB_34UM_AL_VS_LHS_VALIDATION_ROUND1_SAMPLES_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_round1_samples_ka_kb.png"
EMB_34UM_AL_VS_LHS_VALIDATION_ROUND_ADDITIONS_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_round_additions.png"
EMB_34UM_AL_VS_LHS_VALIDATION_SOURCE_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_source_mix.png"
EMB_34UM_AL_VS_LHS_VALIDATION_DISAGREEMENT_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_disagreement_acquisition_ka_kb.png"
EMB_34UM_AL_VS_LHS_VALIDATION_FORCE_OVERLAY_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_force_overlays.png"
EMB_34UM_AL_VS_LHS_VALIDATION_FAILURE_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_failures.png"
EMB_34UM_AL_VS_LHS_VALIDATION_RUNTIME_PLOT_FILENAME = "emb_34um_al_vs_lhs_validation_runtime_per_curve.png"

_CURVE_METRIC_ALIASES = (
    "curve_rel_l2_pct",
    "relative_l2_pct",
    "curve_error",
    "curve_rel_l2",
    "force_curve_rel_l2_pct",
    "median_curve_rel_l2_pct",
    "rel_l2_pct",
)
_PREDICTED_ALIASES = (
    "predicted_curve",
    "prediction_curve",
    "predicted_outputs",
    "prediction_outputs",
    "outputs_pred",
    "y_pred",
)
_REFERENCE_ALIASES = (
    "reference_curve",
    "truth_curve",
    "reference_outputs",
    "target_outputs",
    "outputs_ref",
    "y_true",
    "truth",
)
_POINT_PREDICTED_ALIASES = ("predicted", "prediction", "pred", "y_pred", "y", "force_prediction")
_POINT_REFERENCE_ALIASES = ("reference", "target", "truth", "y_true", "true", "force_reference")
_POINT_AXIS_ALIASES = ("force", "axis", "force_axis", "axis_point", "force_position")
_KA_ALIASES = ("ka", "Yt", "yt", "YT")
_KB_ALIASES = ("kb",)
_RUN_TIME_ALIASES = (
    "runtime_seconds",
    "wall_time",
    "runtime",
    "seconds",
    "elapsed_seconds",
    "elapsed_time",
    "duration",
)
_FORCE_AXIS_ALIASES = ("force", "axis", "force_axis", "force_grid", "forces")
_SAMPLE_SOURCE_ALIASES = ("sample_source", "source", "acquisition_source", "selection_source", "selection_reason")
_ACQUISITION_SCORE_ALIASES = ("acquisition_score", "acq_score", "selected_candidate_score")
_DISAGREEMENT_ALIASES = ("ensemble_disagreement", "disagreement", "acq_uncertainty", "uncertainty")
_ROUND_SOURCE_ALIASES = ("round", "round_index", "iteration", "al_round")
_ORDER_SOURCE_ALIASES = ("order", "index", "candidate_index", "curve_index", "row_index", "f_delta_row_index")
_SELECTED_ALIASES = (
    "selected",
    "selected_for_training",
    "selected_for_validation",
    "is_selected",
    "active_learning_selected",
)
_QUARANTINED_ALIASES = ("quarantined", "is_quarantined", "quarantine", "quarantined_flag")
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}

_FAILED_STATUSES = {"failed", "error", "rejected", "invalid", "crashed"}
_REPLACEMENT_STATUS = {"replaced", "retry", "requeued", "superseded"}
_DEFAULT_PREFIX_COUNTS = (30, 60, 90)


@dataclass(frozen=True)
class Emb34umAlVsLhsValidationArtifacts:
    artifact_dir: Path
    manifest_path: Path
    summary_csv_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    plot_paths: dict[str, str]
    manifest: dict[str, Any]
    summary_rows: tuple[dict[str, Any], ...]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_rows(
    rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    *,
    record_keys: tuple[str, ...] = ("curve_rows", "rows", "records"),
) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, Path)):
        path = Path(rows)
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
        if rows:
            return tuple(dict(item) for item in rows.values() if isinstance(item, Mapping))
        raise ValueError(f"row payload mapping must contain one of: {', '.join(record_keys)}.")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("rows must be a sequence, mapping, JSON path, or CSV path.")
    return tuple(dict(item) for item in rows if isinstance(item, Mapping))


def _strategy(row: Mapping[str, Any]) -> str | None:
    raw = (
        str(row.get("strategy") or row.get("cohort") or row.get("source") or row.get("gate") or row.get("kind") or "")
        .strip()
        .lower()
    )
    if raw in {"al", "active_learning", "active-learning", "full", "full_gate"}:
        return "al"
    if raw in {"lhs", "lhs_gate", "lhs_comparator", "comparator", "baseline"}:
        return "lhs"
    candidate_id = str(row.get("candidate_id") or row.get("curve_id") or "").lower()
    if "-lhs-" in candidate_id or candidate_id.startswith("lhs"):
        return "lhs"
    if "-full-" in candidate_id or "-r01-" in candidate_id or "-r02-" in candidate_id or "-r03-" in candidate_id:
        return "al"
    return None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object, *, label: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{label} must be a boolean.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{label} must be a boolean.")


def _first_bool(row: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> bool | None:
    for alias in aliases:
        if alias in row:
            return _optional_bool(row.get(alias), label=label)
    return None


def _finite_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _parse_number_sequence(value: object) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                if not text.endswith("]"):
                    raise ValueError("bracketed curve sequence must be closed.") from exc
                tokens = [token for token in text[1:-1].replace(";", " ").replace(",", " ").split() if token]
                return tuple(_finite_float(token, label="curve value") for token in tokens)
            if not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes, bytearray)):
                raise ValueError("bracketed curve sequence must decode to a sequence.")
            return tuple(_finite_float(item, label="curve value") for item in loaded)
        tokens = [token for token in text.replace(";", " ").replace(",", " ").split() if token]
        return tuple(_finite_float(token, label="curve value") for token in tokens)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_finite_float(item, label="curve value") for item in value)
    return None


def _first_sequence(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[float, ...] | None:
    for alias in aliases:
        if alias in row:
            parsed = _parse_number_sequence(row.get(alias))
            if parsed is not None:
                return parsed
    return None


def _first_scalar(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            return _finite_float(row.get(alias), label=alias)
    return None


def _first_text(row: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = row.get(alias)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _curve_rel_l2_pct(predicted: Sequence[float], reference: Sequence[float]) -> float:
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("predicted and reference curves must have the same non-zero length.")
    residual_sq = sum((float(a) - float(b)) ** 2 for a, b in zip(predicted, reference))
    reference_sq = sum(float(item) ** 2 for item in reference)
    if reference_sq <= 0.0:
        raise ValueError("reference curve L2 norm must be positive.")
    return float(100.0 * math.sqrt(residual_sq) / math.sqrt(reference_sq))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute median of an empty sequence.")
    return float(statistics.median(float(item) for item in values))


def _curve_id(row: Mapping[str, Any], *, index: int) -> str:
    for key in ("curve_id", "candidate_id", "sample_id", "sample"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row_{index:06d}"


def _row_order(row: Mapping[str, Any], *, fallback_index: int) -> int:
    for key in _ORDER_SOURCE_ALIASES:
        value = _optional_int(row.get(key))
        if value is not None:
            return value
    return fallback_index


def _has_explicit_order(row: Mapping[str, Any]) -> bool:
    return any(_optional_int(row.get(key)) is not None for key in _ORDER_SOURCE_ALIASES)


def _row_round(row: Mapping[str, Any]) -> int | None:
    for key in _ROUND_SOURCE_ALIASES:
        value = _optional_int(row.get(key))
        if value is not None:
            return value
    return None


def _coerce_sample_source(row: Mapping[str, Any]) -> str:
    raw = _first_text(row, aliases=_SAMPLE_SOURCE_ALIASES)
    if raw:
        lowered = raw.strip().lower()
        if "acquis" in lowered:
            return "acquisition"
        if "explor" in lowered:
            return "exploration"
        if "selected" in lowered:
            return "acquisition"
        if lowered in {"lhs", "baseline"}:
            return lowered
        return lowered
    status = _first_text(row, aliases=("status", "result"))
    if status:
        lowered = status.lower()
        if "acquis" in lowered:
            return "acquisition"
        if "explor" in lowered:
            return "exploration"
    return "candidate"


def _coerce_failure_flags(row: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    status = (_first_text(row, aliases=("status", "result", "state")) or "").strip().lower()
    reason = _first_text(row, aliases=("reason", "reason_code", "reason_codes", "failure_reason"))
    reason_text = str(reason or "").lower()
    failed = bool(status in _FAILED_STATUSES or "fail" in status or "error" in reason_text)
    quarantine_flag = _first_bool(row, _QUARANTINED_ALIASES, label="quarantined")
    quarantined = "quarantine" in status or "quarantine" in reason_text or bool(quarantine_flag)
    replaced = status in _REPLACEMENT_STATUS or "replac" in status or "retry" in reason_text
    return failed, quarantined, replaced


def _coerce_selected_flag(row: Mapping[str, Any]) -> bool:
    return bool(_first_bool(row, _SELECTED_ALIASES, label="selected"))


def _public_curve_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _coerce_float_optional(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            try:
                return _finite_float(row.get(alias), label=alias)
            except ValueError:
                return None
    return None


def _coerce_float_optional_strict(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            return _finite_float(row.get(alias), label=alias)
    return None


def _coerce_scalar_int(row: Mapping[str, Any], aliases: Sequence[str], *, default: int | None = None) -> int | None:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            try:
                return int(row.get(alias))
            except (TypeError, ValueError):
                return default
    return default


def _coerce_force_axis(row: Mapping[str, Any]) -> tuple[float, ...] | None:
    for alias in _FORCE_AXIS_ALIASES:
        if alias not in row:
            continue
        try:
            parsed = _first_sequence(row, (alias,))
        except ValueError:
            return None
        if parsed is not None:
            return tuple(float(item) for item in parsed)
    return None


def _coerce_ka_kb(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    ka = _coerce_float_optional(row, aliases=_KA_ALIASES)
    kb = _coerce_float_optional(row, aliases=_KB_ALIASES)
    return ka, kb


def _extract_curve_records(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    direct_records: list[dict[str, Any]] = []
    grouped_points: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for index, row in enumerate(rows, start=1):
        strategy = _strategy(row)
        if strategy is None:
            skip("missing_strategy")
            continue
        curve_id = _curve_id(row, index=index)
        round_index = _row_round(row)
        order = _row_order(row, fallback_index=index)
        has_explicit_order = _has_explicit_order(row)
        metric_value = None
        invalid_curve_metric = False
        for alias in _CURVE_METRIC_ALIASES:
            if alias in row and row.get(alias) not in (None, ""):
                try:
                    metric_value = _finite_float(row.get(alias), label=alias)
                except ValueError:
                    skip("invalid_curve_metric")
                    invalid_curve_metric = True
                break
        if invalid_curve_metric:
            continue

        source = _coerce_sample_source(row)
        try:
            acquisition_score = _first_scalar(row, _ACQUISITION_SCORE_ALIASES)
            disagreement = _first_scalar(row, _DISAGREEMENT_ALIASES)
        except ValueError:
            skip("invalid_acquisition_or_disagreement")
            continue
        try:
            runtime_seconds = _coerce_float_optional_strict(row, _RUN_TIME_ALIASES)
        except ValueError:
            skip("invalid_runtime_seconds")
            continue
        try:
            failed, quarantined, replaced = _coerce_failure_flags(row)
            selected = _coerce_selected_flag(row)
        except ValueError as exc:
            if "quarantined" in str(exc):
                skip("invalid_quarantined_flag")
            else:
                skip("invalid_selected_flag")
            continue
        if quarantined:
            skip("quarantined")
            continue
        ka, kb = _coerce_ka_kb(row)
        force_axis = _coerce_force_axis(row)
        failure_reason = _first_text(row, aliases=("reason", "reason_code", "failure_reason"))

        if metric_value is not None:
            direct_records.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "curve_id": curve_id,
                    "order": order,
                    "candidate_id": _first_text(row, aliases=("candidate_id",)) or curve_id,
                    "curve_rel_l2_pct": metric_value,
                    "ka": ka,
                    "kb": kb,
                    "sample_source": source,
                    "selected": bool(selected),
                    "acquisition_score": acquisition_score,
                    "ensemble_disagreement": disagreement,
                    "runtime_seconds": runtime_seconds,
                    "failed": bool(failed),
                    "quarantined": bool(quarantined),
                    "replacement": bool(replaced),
                    "failure_reason": failure_reason,
                    "force_axis": force_axis,
                    "predicted_curve": None,
                    "reference_curve": None,
                    "_input_index": index,
                    "_has_explicit_order": has_explicit_order,
                }
            )
            continue

        try:
            predicted = _first_sequence(row, _PREDICTED_ALIASES)
            reference = _first_sequence(row, _REFERENCE_ALIASES)
        except ValueError:
            skip("invalid_curve_arrays")
            continue
        if predicted is not None or reference is not None:
            if predicted is None or reference is None:
                skip("incomplete_curve_arrays")
                continue
            try:
                metric_value = _curve_rel_l2_pct(predicted, reference)
            except ValueError:
                skip("invalid_curve_arrays")
                continue
            direct_records.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "curve_id": curve_id,
                    "order": order,
                    "candidate_id": _first_text(row, aliases=("candidate_id",)) or curve_id,
                    "curve_rel_l2_pct": metric_value,
                    "ka": ka,
                    "kb": kb,
                    "sample_source": source,
                    "selected": bool(selected),
                    "acquisition_score": acquisition_score,
                    "ensemble_disagreement": disagreement,
                    "runtime_seconds": runtime_seconds,
                    "failed": bool(failed),
                    "quarantined": bool(quarantined),
                    "replacement": bool(replaced),
                    "failure_reason": failure_reason,
                    "force_axis": force_axis,
                    "predicted_curve": tuple(float(item) for item in predicted),
                    "reference_curve": tuple(float(item) for item in reference),
                    "_input_index": index,
                    "_has_explicit_order": has_explicit_order,
                }
            )
            continue

        try:
            predicted_point = _first_scalar(row, _POINT_PREDICTED_ALIASES)
            reference_point = _first_scalar(row, _POINT_REFERENCE_ALIASES)
            point_axis_alias = next((alias for alias in _POINT_AXIS_ALIASES if alias in row), None)
            point_axis = _finite_float(row[point_axis_alias], label=point_axis_alias) if point_axis_alias else None
        except ValueError:
            if any(alias in row for alias in _POINT_AXIS_ALIASES):
                skip("invalid_point_axis")
            else:
                skip("invalid_point_curve")
            continue
        if predicted_point is None or reference_point is None:
            skip("missing_curve_metric_inputs")
            continue
        if point_axis is None:
            point_axis = 0.0
        key = (strategy, str(round_index), curve_id)
        group = grouped_points.setdefault(
            key,
            {
                "strategy": strategy,
                "round": round_index,
                "curve_id": curve_id,
                "order": order,
                "candidate_id": _first_text(row, aliases=("candidate_id",)) or curve_id,
                "points": [],
                "ka": ka,
                "kb": kb,
                "sample_source": source,
                "selected": bool(selected),
                "acquisition_score": acquisition_score,
                "ensemble_disagreement": disagreement,
                "runtime_seconds": runtime_seconds,
                "failed": bool(failed),
                "quarantined": bool(quarantined),
                "replacement": bool(replaced),
                "failure_reason": failure_reason,
                "force_axis": force_axis,
                "_input_index": index,
                "_has_explicit_order": has_explicit_order,
            },
        )
        group["points"].append((point_axis, predicted_point, reference_point))

    for group in grouped_points.values():
        try:
            points = sorted(group["points"], key=lambda item: float(item[0]))
        except ValueError:
            skip("invalid_point_axis")
            continue
        predicted = [float(item[1]) for item in points]
        reference = [float(item[2]) for item in points]
        try:
            metric_value = _curve_rel_l2_pct(predicted, reference)
        except ValueError:
            skip("invalid_point_curve")
            continue
        direct_records.append(
            {
                "strategy": group["strategy"],
                "round": group["round"],
                "curve_id": group["curve_id"],
                "order": group["order"],
                "candidate_id": group["candidate_id"],
                "curve_rel_l2_pct": metric_value,
                "ka": group["ka"],
                "kb": group["kb"],
                "sample_source": group["sample_source"],
                "selected": bool(group["selected"]),
                "acquisition_score": group["acquisition_score"],
                "ensemble_disagreement": group["ensemble_disagreement"],
                "runtime_seconds": group["runtime_seconds"],
                "failed": bool(group["failed"]),
                "quarantined": bool(group["quarantined"]),
                "replacement": bool(group["replacement"]),
                "failure_reason": group["failure_reason"],
                "force_axis": tuple(float(item[0]) for item in points),
                "predicted_curve": tuple(float(item[1]) for item in points),
                "reference_curve": tuple(float(item[2]) for item in points),
                "_input_index": group["_input_index"],
                "_has_explicit_order": group["_has_explicit_order"],
            }
        )

    if direct_records and all(bool(item.get("_has_explicit_order")) for item in direct_records):
        ordered_records = sorted(
            direct_records,
            key=lambda item: (
                item["strategy"],
                item["round"] if item["round"] is not None else 999999,
                int(item["order"]),
                item["curve_id"],
            ),
        )
    else:
        ordered_records = sorted(direct_records, key=lambda item: int(item["_input_index"]))
    records = tuple(_public_curve_record(item) for item in ordered_records)
    return records, skipped


def _derive_prefix_counts(
    al_records: Sequence[Mapping[str, Any]],
    lhs_records: Sequence[Mapping[str, Any]],
    explicit_prefix_curve_counts: Sequence[int] | None,
) -> tuple[int, ...]:
    available = min(len(al_records), len(lhs_records))
    if explicit_prefix_curve_counts is not None:
        counts = tuple(int(item) for item in explicit_prefix_curve_counts if int(item) > 0)
        return tuple(count for count in counts if 0 < count <= available)

    target_counts = tuple(count for count in _DEFAULT_PREFIX_COUNTS if count <= available)
    if target_counts:
        return target_counts

    if not al_records or not lhs_records:
        return tuple()

    rounds = sorted({int(item["round"]) for item in al_records if item.get("round") is not None})
    if not rounds:
        return (available,)
    return tuple(
        count
        for count in (
            len([item for item in al_records if item.get("round") is not None and int(item["round"]) <= round_index])
            for round_index in rounds
        )
        if 0 < count <= available
    )


def _build_summary_rows(
    *,
    curve_records: Sequence[Mapping[str, Any]],
    prefix_curve_counts: Sequence[int] | None,
) -> tuple[dict[str, Any], ...]:
    al_records = [item for item in curve_records if item["strategy"] == "al"]
    lhs_records = [item for item in curve_records if item["strategy"] == "lhs"]
    counts = _derive_prefix_counts(al_records, lhs_records, explicit_prefix_curve_counts=prefix_curve_counts)
    rows: list[dict[str, Any]] = []
    for count in counts:
        if count > len(al_records) or count > len(lhs_records):
            continue
        al_prefix = al_records[:count]
        lhs_prefix = lhs_records[:count]
        if not al_prefix or not lhs_prefix:
            continue
        al_median = _median([float(item["curve_rel_l2_pct"]) for item in al_prefix])
        lhs_median = _median([float(item["curve_rel_l2_pct"]) for item in lhs_prefix])
        max_al_round = max((int(item["round"]) for item in al_prefix if item.get("round") is not None), default=1)
        rows.append(
            {
                "prefix": int(count),
                "al_round_prefix": max_al_round,
                "prefix_curve_count": count,
                "al_curve_count": len(al_prefix),
                "lhs_curve_count": len(lhs_prefix),
                "al_median_curve_rel_l2_pct": al_median,
                "lhs_median_curve_rel_l2_pct": lhs_median,
                "median_delta": float(al_median - lhs_median),
                "median_improved": bool(al_median < lhs_median),
            }
        )
    return tuple(rows)


def _round_index_values(items: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(sorted({int(item["round"]) for item in items if item.get("round") is not None}))


def _build_round_evidence(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rounds = _round_index_values(records)
    rows: list[dict[str, Any]] = []
    for round_index in rounds:
        round_records = [item for item in records if item.get("round") == round_index]
        al_records = [item for item in round_records if item["strategy"] == "al"]
        lhs_records = [item for item in round_records if item["strategy"] == "lhs"]
        source_counts = {"candidate": 0, "exploration": 0, "acquisition": 0, "selected": 0}
        for item in al_records:
            source = str(item.get("sample_source") or "candidate").lower()
            if "explor" in source:
                source_counts["exploration"] += 1
            elif "acquis" in source:
                source_counts["acquisition"] += 1
            else:
                source_counts["candidate"] += 1
            if bool(item.get("selected")):
                source_counts["selected"] += 1
        run_times = [float(item["runtime_seconds"]) for item in al_records if item.get("runtime_seconds") is not None]
        rows.append(
            {
                "round": round_index,
                "al_curve_count": len(al_records),
                "lhs_curve_count": len(lhs_records),
                "al_selected_count": source_counts["selected"],
                "al_exploration_count": source_counts["exploration"],
                "al_acquisition_count": source_counts["acquisition"],
                "al_candidate_count": source_counts["candidate"],
                "al_failed_count": sum(1 for item in al_records if bool(item.get("failed"))),
                "al_quarantine_count": sum(1 for item in al_records if bool(item.get("quarantined"))),
                "al_replacement_count": sum(1 for item in al_records if bool(item.get("replacement"))),
                "runtime_curves": len(run_times),
                "runtime_seconds_median": statistics.median(run_times) if run_times else float("nan"),
            }
        )
    return tuple(rows)


def _attach_runtime_rows(
    records: Sequence[dict[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    if not runtime_rows:
        return tuple(dict(item) for item in records), {}

    by_curve_id: dict[str, float] = {}
    by_candidate_id: dict[str, float] = {}
    by_round_curve: dict[tuple[int, str], float] = {}
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in runtime_rows:
        try:
            runtime = _coerce_float_optional_strict(
                row,
                aliases=("runtime", "runtime_seconds", "wall_time", "seconds", "elapsed_seconds"),
            )
        except ValueError:
            skip("invalid_runtime_seconds")
            continue
        if runtime is None:
            continue
        curve_id = _first_text(row, aliases=("curve_id", "sample_id", "id"))
        candidate_id = _first_text(row, aliases=("candidate_id", "sample_id", "id"))
        round_index = _optional_int(row.get("round") or row.get("iteration") or row.get("fence_round"))
        if curve_id:
            by_curve_id[curve_id] = runtime
            if round_index is not None:
                by_round_curve[(round_index, curve_id)] = runtime
        if candidate_id:
            by_candidate_id[candidate_id] = runtime

    attached: list[dict[str, Any]] = []
    for row in records:
        updated = dict(row)
        candidate_id = str(row.get("candidate_id") or "")
        curve_id = str(row.get("curve_id") or "")
        round_index = _optional_int(row.get("round"))
        runtime = None
        if round_index is not None and curve_id:
            runtime = by_round_curve.get((round_index, curve_id))
        if runtime is None and curve_id and curve_id in by_curve_id:
            runtime = by_curve_id[curve_id]
        if runtime is None and candidate_id and candidate_id in by_candidate_id:
            runtime = by_candidate_id[candidate_id]
        if runtime is not None:
            updated["runtime_seconds"] = runtime
        attached.append(updated)
    return tuple(attached), skipped


def _coerce_bounds(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    values = _first_sequence({"_tmp": value}, aliases=("_tmp",))
    if values is None or len(values) != 2:
        return default
    a, b = float(values[0]), float(values[1])
    if a <= 0.0 or b <= 0.0 or a >= b:
        return default
    return (a, b)


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
            row.extend((245, 245, 245, 255) if (x + y) % 2 else (70, 115, 160, 255))
        rows.append(bytes(row))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + _png_chunk(b"IEND", b"")


def _write_plot(path: Path, summary_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not summary_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    x = [int(row["prefix"]) for row in summary_rows]
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(x, [float(row["al_median_curve_rel_l2_pct"]) for row in summary_rows], marker="o", label="AL prefix")
    axis.plot(x, [float(row["lhs_median_curve_rel_l2_pct"]) for row in summary_rows], marker="s", label="LHS prefix")
    axis.set_title("EMB 3.4um AL-vs-LHS validation: force-curve relative L2")
    axis.set_xlabel("Curve-prefix count")
    axis.set_ylabel("Median relative curve L2 %")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sample_scatter(path: Path, records: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not records:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    ka_kb = [
        (float(item["ka"]), float(item["kb"]))
        for item in records
        if item.get("ka") is not None and item.get("kb") is not None
    ]
    if not ka_kb:
        path.write_bytes(_fallback_png())
        return
    ka_values = [item[0] for item in ka_kb]
    kb_values = [item[1] for item in ka_kb]
    selected_kb = [
        (float(item["ka"]), float(item["kb"]))
        for item in records
        if item.get("ka") is not None and item.get("kb") is not None and bool(item.get("selected"))
    ]
    candidate_records = [
        (float(item["ka"]), float(item["kb"]))
        for item in records
        if item.get("ka") is not None and item.get("kb") is not None and not bool(item.get("selected"))
    ]
    al_records = [
        (float(item["ka"]), float(item["kb"]))
        for item in records
        if item.get("ka") is not None and item.get("kb") is not None and item.get("strategy") == "al"
    ]
    lhs_records = [
        (float(item["ka"]), float(item["kb"]))
        for item in records
        if item.get("ka") is not None and item.get("kb") is not None and item.get("strategy") == "lhs"
    ]

    fig, axis = plt.subplots(1, 1, figsize=(7, 5))
    if candidate_records:
        axis.scatter(
            [item[0] for item in candidate_records],
            [item[1] for item in candidate_records],
            alpha=0.65,
            label="Candidates",
            marker="o",
            color="#1f77b4",
        )
    if selected_kb:
        axis.scatter(
            [item[0] for item in selected_kb],
            [item[1] for item in selected_kb],
            marker="*",
            s=64,
            alpha=0.85,
            label="Selected",
            color="black",
        )
    if al_records:
        axis.scatter(
            [item[0] for item in al_records],
            [item[1] for item in al_records],
            marker="o",
            alpha=0.35,
            label="AL samples",
            facecolors="none",
            edgecolors="#1f77b4",
        )
    if lhs_records:
        axis.scatter(
            [item[0] for item in lhs_records],
            [item[1] for item in lhs_records],
            marker="o",
            alpha=0.35,
            label="LHS samples",
            facecolors="none",
            edgecolors="#ff7f0e",
        )
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    axis.set_title("Candidate and selected samples over ka, kb")
    axis.set_xlim(min(ka_values), max(ka_values))
    axis.set_ylim(min(kb_values), max(kb_values))
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_round1_sample_scatter(path: Path, records: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    round_one_records = [
        item
        for item in records
        if item.get("strategy") == "al" and _optional_int(item.get("round")) == 1
    ]
    if not round_one_records:
        path.write_bytes(_fallback_png())
        return

    ka_kb = [
        (float(item["ka"]), float(item["kb"]))
        for item in round_one_records
        if item.get("ka") is not None and item.get("kb") is not None
    ]
    if not ka_kb:
        path.write_bytes(_fallback_png())
        return
    ka_values = [item[0] for item in ka_kb]
    kb_values = [item[1] for item in ka_kb]
    selected_kb = [
        (float(item["ka"]), float(item["kb"]))
        for item in round_one_records
        if item.get("ka") is not None and item.get("kb") is not None and bool(item.get("selected"))
    ]
    candidate_records = [
        (float(item["ka"]), float(item["kb"]))
        for item in round_one_records
        if item.get("ka") is not None and item.get("kb") is not None and not bool(item.get("selected"))
    ]

    fig, axis = plt.subplots(1, 1, figsize=(7, 5))
    if candidate_records:
        axis.scatter(
            [item[0] for item in candidate_records],
            [item[1] for item in candidate_records],
            alpha=0.65,
            label="Initial round-1 candidates",
            marker="o",
            color="#1f77b4",
        )
    if selected_kb:
        axis.scatter(
            [item[0] for item in selected_kb],
            [item[1] for item in selected_kb],
            marker="*",
            s=64,
            alpha=0.85,
            label="Initial round-1 selected",
            color="black",
        )
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    axis.set_title("Round-1 initial samples over ka, kb")
    axis.set_xlim(min(ka_values), max(ka_values))
    axis.set_ylim(min(kb_values), max(kb_values))
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_round_additions(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not round_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    x = [int(row["round"]) for row in round_rows]
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(x, [int(row["al_curve_count"]) for row in round_rows], marker="o", label="AL additions")
    axis.plot(x, [int(row["lhs_curve_count"]) for row in round_rows], marker="s", label="LHS additions")
    axis.set_title("Per-round additions")
    axis.set_xlabel("Round")
    axis.set_ylabel("Curves added")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_exploration_vs_acquisition(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not round_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    x = [int(row["round"]) for row in round_rows]
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(x, [int(row["al_exploration_count"]) for row in round_rows], marker="o", label="Exploration")
    axis.plot(x, [int(row["al_acquisition_count"]) for row in round_rows], marker="s", label="Acquisition")
    axis.set_title("Exploration vs acquisition-driven AL samples")
    axis.set_xlabel("Round")
    axis.set_ylabel("AL samples")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_disagreement_map(path: Path, records: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not records:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    al_records = [item for item in records if item["strategy"] == "al"]
    selected_records = [item for item in al_records if bool(item.get("selected"))]
    source_records = [item for item in al_records if "acquis" in str(item.get("sample_source", "")).lower()]
    candidate_records = selected_records if selected_records else source_records or al_records
    valid = [
        item
        for item in candidate_records
        if item.get("ka") is not None and item.get("kb") is not None and item.get("ensemble_disagreement") is not None
    ]
    if not valid:
        if selected_records and candidate_records is not source_records:
            candidate_records = source_records or al_records
            valid = [
                item
                for item in candidate_records
                if item.get("ka") is not None and item.get("kb") is not None and item.get("ensemble_disagreement") is not None
            ]
    if not valid:
        path.write_bytes(_fallback_png())
        return
    ka = [float(item["ka"]) for item in valid]
    kb = [float(item["kb"]) for item in valid]
    disagreement = [float(item["ensemble_disagreement"]) for item in valid]
    scores = [item.get("acquisition_score") for item in valid]
    acq = [float(item) if item is not None else 0.0 for item in scores]

    fig, axis = plt.subplots(1, 1, figsize=(8, 5))
    max_acq = max(acq)
    if not math.isfinite(max_acq) or max_acq <= 0.0:
        max_acq = 1.0
    scale = [5.0 + 22.0 * max(0.0, item) / max_acq for item in acq]
    scatter = axis.scatter(ka, kb, c=disagreement, s=scale, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=axis)
    cbar.set_label("ensemble disagreement")
    axis.set_title("AL acquisition/disagreement map")
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_force_overlays(path: Path, records: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not records:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    overlay_records = [
        item
        for item in records
        if isinstance(item.get("predicted_curve"), tuple | list)
        and isinstance(item.get("reference_curve"), tuple | list)
        and item.get("predicted_curve")
    ]
    if not overlay_records:
        path.write_bytes(_fallback_png())
        return

    fig, axis = plt.subplots(1, 1, figsize=(8, 5))
    for index, item in enumerate(overlay_records[:12]):
        ref = item["reference_curve"]
        pred = item["predicted_curve"]
        if not len(ref) or not len(pred):
            continue
        x = list(range(1, max(len(ref), len(pred)) + 1))
        axis.plot(x[: len(ref)], ref, linewidth=0.8, alpha=0.35, color="#1f77b4", label="reference" if index == 0 else None)
        axis.plot(x[: len(pred)], pred, linewidth=0.8, alpha=0.35, color="#ff7f0e", label="predicted" if index == 0 else None)
    axis.set_title("Force curve overlays")
    axis.set_xlabel("force point")
    axis.set_ylabel("force")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_runtime_benchmark(path: Path, runtime_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not runtime_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    values = [float(item["runtime_seconds"]) for item in runtime_rows if item.get("runtime_seconds") is not None]
    if not values:
        path.write_bytes(_fallback_png())
        return
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(range(1, len(values) + 1), values, marker="o")
    axis.set_title("Runtime per curve")
    axis.set_xlabel("Curve index")
    axis.set_ylabel("runtime [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_failure_diagnostics(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not round_rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    rounds = [int(item["round"]) for item in round_rows]
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(rounds, [int(item["al_failed_count"]) for item in round_rows], marker="x", label="failures")
    axis.plot(rounds, [int(item["al_quarantine_count"]) for item in round_rows], marker="s", label="quarantine")
    axis.plot(rounds, [int(item["al_replacement_count"]) for item in round_rows], marker="d", label="replacements")
    axis.set_title("Failure and quarantine diagnostics")
    axis.set_xlabel("Round")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "prefix",
        "al_round_prefix",
        "prefix_curve_count",
        "al_curve_count",
        "lhs_curve_count",
        "al_median_curve_rel_l2_pct",
        "lhs_median_curve_rel_l2_pct",
        "median_delta",
        "median_improved",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _coerce_plot_records(value: object) -> tuple[dict[str, Any], ...]:
    if value is None:
        return tuple()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    raise TypeError("plot payload must be a sequence.")


def build_emb_34um_al_vs_lhs_validation_report(
    *,
    curve_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    prefix_curve_counts: Sequence[int] | None = None,
    adaptive_acquisition_available: bool = False,
    acquisition_engine_name: str | None = None,
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows = _coerce_rows(curve_rows, record_keys=("curve_rows", "rows", "records"))
    curve_records, skipped = _extract_curve_records(rows)
    if runtime_rows is not None:
        parsed_runtime_rows = _coerce_rows(
            runtime_rows,
            record_keys=("runtime_rows", "rows", "records"),
        )
        curve_records, runtime_skipped = _attach_runtime_rows(curve_records, parsed_runtime_rows)
        for reason, count in runtime_skipped.items():
            skipped[reason] = skipped.get(reason, 0) + count
    summary_rows = _build_summary_rows(curve_records=curve_records, prefix_curve_counts=prefix_curve_counts)
    round_rows = _build_round_evidence(curve_records)
    al_count = sum(1 for item in curve_records if item["strategy"] == "al")
    lhs_count = sum(1 for item in curve_records if item["strategy"] == "lhs")
    runtime_rows_payload = _coerce_plot_records(
        [
            item
            for item in curve_records
            if item.get("runtime_seconds") is not None
        ]
    )
    runtime_values = [float(item["runtime_seconds"]) for item in runtime_rows_payload]

    blockers: list[str] = []
    limitations: list[str] = []
    if not curve_records:
        blockers.append("No usable curve-level validation metrics were available.")
    if al_count == 0:
        blockers.append("No AL curve rows with curve metrics were available.")
    if lhs_count == 0:
        blockers.append("No LHS comparator curve rows with curve metrics were available.")
    if not summary_rows:
        blockers.append("No paired AL/LHS prefix summary could be computed.")
    if not adaptive_acquisition_available:
        limitations.append(
            "Adaptive acquisition engine was not supplied; AL prefixes are evaluated from provided rows only."
        )

    ka_bounds = _coerce_bounds(metadata.get("ka_bounds") if isinstance(metadata, Mapping) else None, default=(1.0e2, 6.0e5))
    kb_bounds = _coerce_bounds(metadata.get("kb_bounds") if isinstance(metadata, Mapping) else None, default=(400.0, 70000.0))

    manifest = {
        "schema_version": EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION,
        "experiment": "indentation",
        "diameter_um": "3.4",
        "primary_metric": "median_curve_rel_l2_pct",
        "prefix_targets": list(_DEFAULT_PREFIX_COUNTS),
        "source_row_count": len(rows),
        "usable_curve_count": len(curve_records),
        "al_curve_count": al_count,
        "lhs_curve_count": lhs_count,
        "runtime_curve_count": len(runtime_rows_payload),
        "runtime_seconds_count": len(runtime_values),
        "runtime_seconds_total": float(sum(runtime_values)) if runtime_values else 0.0,
        "runtime_seconds_median": statistics.median(runtime_values) if runtime_values else float("nan"),
        "skipped_row_reasons": skipped,
        "prefix_curve_counts": [int(item["prefix_curve_count"]) for item in summary_rows],
        "adaptive_acquisition": {
            "available": bool(adaptive_acquisition_available),
            "engine": str(acquisition_engine_name or ""),
            "status": "available" if adaptive_acquisition_available else "not_available",
            "note": (
                "A real acquisition engine can provide selected AL rows before this validation step."
                if adaptive_acquisition_available
                else "No adaptive acquisition engine was supplied; this module only validates provided AL and LHS rows."
            ),
        },
        "summary_rows": list(summary_rows),
        "curve_records": list(curve_records),
        "round_evidence": list(round_rows),
        "ka_bounds": [float(ka_bounds[0]), float(ka_bounds[1])],
        "kb_bounds": [float(kb_bounds[0]), float(kb_bounds[1])],
        "runtime_rows": list(runtime_rows_payload),
        "exploration_samples": [
            item for item in curve_records if str(item.get("sample_source", "")).lower().startswith("explor")
        ],
        "acquisition_samples": [item for item in curve_records if "acquis" in str(item.get("sample_source", "")).lower()],
        "selected_samples": [item for item in curve_records if bool(item.get("selected"))],
        "blockers": blockers,
        "limitations": limitations,
        "status": "ready" if summary_rows else "blocked",
        "metadata": dict(metadata or {}),
    }
    return manifest, summary_rows


def write_emb_34um_al_vs_lhs_validation_artifacts(
    *,
    curve_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any],
    output_root: str | Path,
    prefix_curve_counts: Sequence[int] | None = None,
    adaptive_acquisition_available: bool = False,
    acquisition_engine_name: str | None = None,
    runtime_rows: Sequence[Mapping[str, Any]] | str | Path | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_plot: bool = True,
) -> Emb34umAlVsLhsValidationArtifacts:
    artifact_dir = Path(output_root)
    manifest_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME
    plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME

    sample_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_SAMPLES_PLOT_FILENAME
    round1_sample_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_ROUND1_SAMPLES_PLOT_FILENAME
    round_additions_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_ROUND_ADDITIONS_PLOT_FILENAME
    source_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_SOURCE_PLOT_FILENAME
    disagreement_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_DISAGREEMENT_PLOT_FILENAME
    overlay_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_FORCE_OVERLAY_PLOT_FILENAME
    failure_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_FAILURE_PLOT_FILENAME
    runtime_plot_path = artifact_dir / EMB_34UM_AL_VS_LHS_VALIDATION_RUNTIME_PLOT_FILENAME

    manifest, summary_rows = build_emb_34um_al_vs_lhs_validation_report(
        curve_rows=curve_rows,
        prefix_curve_counts=prefix_curve_counts,
        adaptive_acquisition_available=bool(adaptive_acquisition_available),
        acquisition_engine_name=acquisition_engine_name,
        runtime_rows=runtime_rows,
        metadata=metadata,
    )
    curve_records = _coerce_plot_records(manifest["curve_records"])
    round_rows = _coerce_plot_records(manifest["round_evidence"])
    runtime_records = _coerce_plot_records(manifest["runtime_rows"])

    _write_summary_csv(summary_csv_path, summary_rows)
    _write_plot(plot_path, summary_rows, include_plot=include_plot)
    _plot_sample_scatter(sample_plot_path, curve_records, include_plot=include_plot)
    _plot_round1_sample_scatter(round1_sample_plot_path, curve_records, include_plot=include_plot)
    _plot_round_additions(round_additions_plot_path, round_rows, include_plot=include_plot)
    _plot_exploration_vs_acquisition(source_plot_path, round_rows, include_plot=include_plot)
    _plot_disagreement_map(disagreement_plot_path, curve_records, include_plot=include_plot)
    _plot_force_overlays(overlay_plot_path, curve_records, include_plot=include_plot)
    _plot_failure_diagnostics(failure_plot_path, round_rows, include_plot=include_plot)
    _plot_runtime_benchmark(runtime_plot_path, runtime_records, include_plot=include_plot)

    plot_paths = {
        "al_vs_lhs_l2": str(plot_path),
        "al_vs_lhs_validation": str(plot_path),
        "al_vs_lhs_relative_l2": str(plot_path),
        "samples_ka_kb": str(sample_plot_path),
        "initial_round1_samples": str(round1_sample_plot_path),
        "per_round_additions": str(round_additions_plot_path),
        "exploration_vs_acquisition": str(source_plot_path),
        "disagreement_acquisition_map": str(disagreement_plot_path),
        "force_curve_overlays": str(overlay_plot_path),
        "failure_quarantine_replacement": str(failure_plot_path),
        "runtime_per_curve": str(runtime_plot_path),
    }
    manifest["plot_paths"] = plot_paths
    _write_json(manifest_path, manifest)
    _write_json(plot_sidecar_path, {"plot_path": str(plot_path), **manifest})

    return Emb34umAlVsLhsValidationArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_csv_path=summary_csv_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        plot_paths=plot_paths,
        manifest=manifest,
        summary_rows=summary_rows,
    )


__all__ = [
    "EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SCHEMA_VERSION",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SAMPLES_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_ROUND1_SAMPLES_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_ROUND_ADDITIONS_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_SOURCE_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_DISAGREEMENT_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_FORCE_OVERLAY_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_FAILURE_PLOT_FILENAME",
    "EMB_34UM_AL_VS_LHS_VALIDATION_RUNTIME_PLOT_FILENAME",
    "Emb34umAlVsLhsValidationArtifacts",
    "build_emb_34um_al_vs_lhs_validation_report",
    "write_emb_34um_al_vs_lhs_validation_artifacts",
]
