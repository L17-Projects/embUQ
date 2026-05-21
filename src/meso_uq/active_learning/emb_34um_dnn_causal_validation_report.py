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


EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_report.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME = "emb_34um_dnn_causal_validation_report.json"
EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME = "emb_34um_dnn_causal_validation_summary.csv"
EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_FILENAME = "emb_34um_dnn_causal_validation_al_vs_lhs_curves.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_FILENAME = "emb_34um_dnn_causal_validation_replicate_curves.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME = "emb_34um_dnn_causal_validation_step_delta_ci.png"
EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME = "emb_34um_dnn_causal_validation_runtime_replacement_diagnostics.png"

EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_al_vs_lhs_curves.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_replicate_curves.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_step_delta_ci.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_runtime_replacement_diagnostics.png.json"
)

_DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
_DEFAULT_CI_SEED = 2_026_202_405
_DEFAULT_CONFIDENCE = 0.95
_PLOT_DPI = 120

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_BRANCH_ALIASES = ("branch", "strategy", "method")
_REPLICATE_ALIASES = ("replicate", "seed", "replica")
_CYCLE_ALIASES = ("cycle", "step", "round", "iteration")
_METRIC_ALIASES = ("relative_l2", "curve_rel_l2", "median_curve_rel_l2", "rel_l2")
_TRAIN_SECONDS_ALIASES = ("train_seconds", "train_time_seconds", "train_time", "train_runtime_seconds")
_SCORE_SECONDS_ALIASES = ("score_seconds", "score_time_seconds", "score_time", "score_runtime_seconds")
_REPLACEMENT_ALIASES = ("replacement", "replaced", "is_replaced", "retry", "requeued", "superseded")
_QUARANTINE_ALIASES = ("quarantine", "quarantined", "is_quarantined", "quarantined_flag")

_AL_BRANCH_TOKENS = {"al", "active", "active_learning"}
_LHS_BRANCH_TOKENS = {"lhs", "baseline"}


@dataclass(frozen=True)
class Emb34umDnnCausalValidationArtifacts:
    artifact_dir: Path
    report_path: Path
    summary_csv_path: Path
    plot_paths: dict[str, str]
    plot_sidecar_paths: dict[str, str]
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


def _first_text(row: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = row.get(alias)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
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
    replicate = _optional_int(_first_text(row, aliases=_REPLICATE_ALIASES))
    if replicate is None:
        raise ValueError("missing_replicate")
    branch = _normalize_branch(_first_text(row, aliases=_BRANCH_ALIASES))
    cycle = _optional_int(_first_text(row, aliases=_CYCLE_ALIASES))
    if cycle is None:
        raise ValueError("missing_cycle")
    metric = _first_float(row, _METRIC_ALIASES, label="relative_l2")
    train_seconds = _coerce_float(_first_text(row, aliases=_TRAIN_SECONDS_ALIASES), label="train_seconds", required=False)
    score_seconds = _coerce_float(_first_text(row, aliases=_SCORE_SECONDS_ALIASES), label="score_seconds", required=False)
    replacement = _optional_bool(_first_text(row, aliases=_REPLACEMENT_ALIASES))
    quarantine = _optional_bool(_first_text(row, aliases=_QUARANTINE_ALIASES))

    return {
        "replicate": int(replicate),
        "branch": branch,
        "cycle": int(cycle),
        "relative_l2": float(metric),
        "train_seconds": float(train_seconds) if train_seconds is not None else None,
        "score_seconds": float(score_seconds) if score_seconds is not None else None,
        "replacement": bool(replacement) if replacement is not None else False,
        "quarantine": bool(quarantine) if quarantine is not None else False,
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
        relative_l2_values = [float(item["relative_l2"]) for item in items]
        train_seconds = [float(item["train_seconds"]) for item in items if item.get("train_seconds") is not None]
        score_seconds = [float(item["score_seconds"]) for item in items if item.get("score_seconds") is not None]
        summary_rows.append(
            {
                "replicate": int(replicate),
                "branch": str(branch),
                "cycle": int(cycle),
                "relative_l2_mean": _mean(relative_l2_values),
                "relative_l2_median": _median(relative_l2_values),
                "relative_l2_min": float(min(relative_l2_values)),
                "relative_l2_max": float(max(relative_l2_values)),
                "row_count": int(len(items)),
                "train_seconds_total": float(sum(train_seconds)) if train_seconds else 0.0,
                "train_seconds_mean": _mean(train_seconds) if train_seconds else None,
                "score_seconds_total": float(sum(score_seconds)) if score_seconds else 0.0,
                "score_seconds_mean": _mean(score_seconds) if score_seconds else None,
                "replacement_count": int(sum(1 for item in items if bool(item.get("replacement")))),
                "quarantine_count": int(sum(1 for item in items if bool(item.get("quarantine")))),
            }
        )
    return tuple(summary_rows)


def _pair_replicate_cycle_rows(summary_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for row in summary_rows:
        key = (int(row["replicate"]), int(row["cycle"]))
        grouped.setdefault(key, {})[str(row["branch"])] = dict(row)

    paired_rows: list[dict[str, Any]] = []
    for (replicate, cycle), branches in sorted(grouped.items()):
        al_row = branches.get("al")
        lhs_row = branches.get("lhs")
        if al_row is None or lhs_row is None:
            continue
        paired_rows.append(
            {
                "replicate": int(replicate),
                "cycle": int(cycle),
                "al_relative_l2_mean": float(al_row["relative_l2_mean"]),
                "al_relative_l2_median": float(al_row["relative_l2_median"]),
                "lhs_relative_l2_mean": float(lhs_row["relative_l2_mean"]),
                "lhs_relative_l2_median": float(lhs_row["relative_l2_median"]),
                "paired_delta": float(al_row["relative_l2_median"] - lhs_row["relative_l2_median"]),
                "train_seconds_total": float(al_row["train_seconds_total"]) + float(lhs_row["train_seconds_total"]),
                "score_seconds_total": float(al_row["score_seconds_total"]) + float(lhs_row["score_seconds_total"]),
                "replacement_count": int(al_row["replacement_count"]) + int(lhs_row["replacement_count"]),
                "quarantine_count": int(al_row["quarantine_count"]) + int(lhs_row["quarantine_count"]),
            }
        )
    return tuple(paired_rows)


def _summarize_cycles(paired_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(int(row["cycle"]), []).append(dict(row))

    cycle_rows: list[dict[str, Any]] = []
    for cycle, rows in sorted(grouped.items()):
        al_values = [float(item["al_relative_l2_median"]) for item in rows]
        lhs_values = [float(item["lhs_relative_l2_median"]) for item in rows]
        deltas = [float(item["paired_delta"]) for item in rows]
        al_ci = {
            "lower": _percentile(al_values, 0.025) if len(al_values) > 1 else float(al_values[0]),
            "upper": _percentile(al_values, 0.975) if len(al_values) > 1 else float(al_values[0]),
        }
        lhs_ci = {
            "lower": _percentile(lhs_values, 0.025) if len(lhs_values) > 1 else float(lhs_values[0]),
            "upper": _percentile(lhs_values, 0.975) if len(lhs_values) > 1 else float(lhs_values[0]),
        }
        delta_ci = _bootstrap_ci(deltas)
        cycle_rows.append(
            {
                "cycle": int(cycle),
                "replicate_count": int(len(rows)),
                "al_relative_l2_mean": _mean(al_values),
                "al_relative_l2_median": _median(al_values),
                "al_relative_l2_ci_lower": float(al_ci["lower"]),
                "al_relative_l2_ci_upper": float(al_ci["upper"]),
                "lhs_relative_l2_mean": _mean(lhs_values),
                "lhs_relative_l2_median": _median(lhs_values),
                "lhs_relative_l2_ci_lower": float(lhs_ci["lower"]),
                "lhs_relative_l2_ci_upper": float(lhs_ci["upper"]),
                "paired_delta": float(delta_ci["point_estimate"]),
                "paired_delta_ci_lower": float(delta_ci["ci_lower"]),
                "paired_delta_ci_upper": float(delta_ci["ci_upper"]),
                "paired_delta_sample_size": int(delta_ci["sample_size"]),
                "train_seconds_total": float(sum(float(item["train_seconds_total"]) for item in rows)),
                "score_seconds_total": float(sum(float(item["score_seconds_total"]) for item in rows)),
                "replacement_count": int(sum(int(item["replacement_count"]) for item in rows)),
                "quarantine_count": int(sum(int(item["quarantine_count"]) for item in rows)),
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
        al_auc = _trapz_normalized(cycles, al_curve)
        lhs_auc = _trapz_normalized(cycles, lhs_curve)
        replicate_rows.append(
            {
                "replicate": int(replicate),
                "cycles": cycles,
                "al_relative_l2_curve": al_curve,
                "lhs_relative_l2_curve": lhs_curve,
                "al_auc": float(al_auc),
                "lhs_auc": float(lhs_auc),
                "auc_delta": float(al_auc - lhs_auc),
                "final_cycle": int(ordered[-1]["cycle"]) if ordered else None,
                "final_cycle_delta": float(ordered[-1]["paired_delta"]) if ordered else None,
                "train_seconds_total": float(sum(float(item["train_seconds_total"]) for item in ordered)),
                "score_seconds_total": float(sum(float(item["score_seconds_total"]) for item in ordered)),
                "replacement_count": int(sum(int(item["replacement_count"]) for item in ordered)),
                "quarantine_count": int(sum(int(item["quarantine_count"]) for item in ordered)),
            }
        )
    return tuple(replicate_rows)


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
    axis.set_xlabel("cycle")
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "cycle",
        "replicate_count",
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
        "paired_delta_sample_size",
        "train_seconds_total",
        "score_seconds_total",
        "replacement_count",
        "quarantine_count",
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
            "metadata": dict(metadata),
        },
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

    if not parsed_rows:
        report = {
            "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
            "status": "blocked",
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
            "metadata": metadata_payload,
        }
        return report, tuple()

    summary_rows = _summarize_by_branch_cycle(parsed_rows)
    paired_cycle_rows = _pair_replicate_cycle_rows(summary_rows)
    if not paired_cycle_rows:
        report = {
            "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
            "status": "blocked",
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
            "metadata": metadata_payload,
        }
        return report, tuple()

    cycle_rows = _summarize_cycles(paired_cycle_rows)
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

    status = "passed" if final_delta_ci["upper"] < 0.0 else "failed"
    passed = status == "passed"

    runtime_summary = _summarize_runtime_replacement(summary_rows)
    report = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": status,
        "decision": {
            "status": status,
            "passed": bool(passed),
            "final_cycle": int(final_cycle["cycle"]),
            "final_cycle_delta_ci": {
                "point_estimate": float(final_delta_ci["point_estimate"]),
                "lower": float(final_delta_ci["lower"]),
                "upper": float(final_delta_ci["upper"]),
            },
            "auc_delta_ci": {
                "point_estimate": float(auc_ci["point_estimate"]),
                "lower": float(auc_ci["ci_lower"]),
                "upper": float(auc_ci["ci_upper"]),
            },
            "criteria": {
                "bootstrap_resamples": int(bootstrap_resamples),
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_confidence": float(confidence),
                "ensemble_size": 10,
            },
        },
        "source_row_count": len(parsed_rows),
        "usable_pair_row_count": len(paired_cycle_rows),
        "replicate_count": len(replicate_rows),
        "cycle_count": len(cycle_rows),
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
        "runtime_replacement_summary": runtime_summary,
        "blocked_reasons": {},
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
    runtime_plot = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME

    al_vs_lhs_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME
    replicate_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME
    step_delta_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME
    runtime_sidecar = artifact_dir / EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME

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
    _plot_runtime_replacement_diagnostics(runtime_plot, report.get("cycle_rows", ()), include_plot=include_plot)

    source_rows = _coerce_rows(rows, record_keys=("rows", "records", "source_rows"))
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
        "runtime_replacement_diagnostics": str(runtime_plot),
    }
    plot_sidecar_paths = {
        "al_vs_lhs_curves": str(al_vs_lhs_sidecar),
        "per_replicate_curves": str(replicate_sidecar),
        "step_delta_ci": str(step_delta_sidecar),
        "runtime_replacement_diagnostics": str(runtime_sidecar),
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

    return Emb34umDnnCausalValidationArtifacts(
        artifact_dir=artifact_dir,
        report_path=report_path,
        summary_csv_path=summary_csv_path,
        plot_paths=plot_paths,
        plot_sidecar_paths=plot_sidecar_paths,
        report=report,
        summary_rows=tuple(dict(item) for item in report.get("summary_rows", ())),
        paired_cycle_rows=tuple(dict(item) for item in report.get("paired_cycle_rows", ())),
    )


__all__ = [
    "EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_AL_VS_LHS_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPORT_SCHEMA_VERSION",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_REPLICATE_CURVES_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_RUNTIME_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_STEP_DELTA_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_SUMMARY_CSV_FILENAME",
    "Emb34umDnnCausalValidationArtifacts",
    "build_emb_34um_dnn_causal_validation_report",
    "write_emb_34um_dnn_causal_validation_artifacts",
]
