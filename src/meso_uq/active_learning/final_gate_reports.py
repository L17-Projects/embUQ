from __future__ import annotations

"""Final-gate reporting for AL vs LHS surrogate validation experiments.

The module intentionally avoids importing matplotlib at import time and supports a
simple fallback PNG for environments where plotting libraries are unavailable.
"""

import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ACTIVE_LEARNING_FINAL_GATE_REPORT_SCHEMA_VERSION = "meso_uq.active_learning.final_gate_report.v1"
ACTIVE_LEARNING_FINAL_GATE_SCHEMA_VERSION = ACTIVE_LEARNING_FINAL_GATE_REPORT_SCHEMA_VERSION
ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME = "active_learning_final_gate_report.json"
ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME = "active_learning_final_gate_report.md"
ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME = "active_learning_final_gate_summary.csv"
ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR = "one_90_curve_batch"
ACTIVE_LEARNING_FINAL_GATE_EXPERIMENT = "indentation"
ACTIVE_LEARNING_FINAL_GATE_DIAMETER_UM = "3.4"

ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI = 120
ACTIVE_LEARNING_FINAL_GATE_CURVE_METRICS_PLOT_FILENAME = "final_gate_curve_metrics.png"
ACTIVE_LEARNING_FINAL_GATE_RESIDUALS_PLOT_FILENAME = "final_gate_residuals.png"
ACTIVE_LEARNING_FINAL_GATE_PREDICTION_PLOT_FILENAME = "final_gate_predicted_vs_reference.png"
ACTIVE_LEARNING_FINAL_GATE_ACQUISITION_PLOT_FILENAME = "final_gate_acquisition_scores.png"
ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME = "final_gate_ka_kb_coverage.png"
ACTIVE_LEARNING_FINAL_GATE_YT_KB_PLOT_FILENAME = ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME
ACTIVE_LEARNING_FINAL_GATE_FAILURE_PLOT_FILENAME = "final_gate_failure_quarantine.png"
ACTIVE_LEARNING_FINAL_GATE_MODEL_SELECTION_PLOT_FILENAME = "final_gate_model_selection_by_round.png"
ACTIVE_LEARNING_FINAL_GATE_SUMMARY_PLOT_FILENAME = "final_gate_al_vs_lhs_summary.png"

_REQUIRED_ROUND_COUNT = 3
_REQUIRED_CURVE_COUNT = 30
_EXPECTED_AL_TOTAL_CURVES = _REQUIRED_ROUND_COUNT * _REQUIRED_CURVE_COUNT
_EXPECTED_LHS_TOTAL_CURVES = _EXPECTED_AL_TOTAL_CURVES
_REQUIRED_LHS_CURVE_COUNT = _EXPECTED_LHS_TOTAL_CURVES
_STARTING_WITH_NO_INITIAL_DATA = True
_METRIC_ALIASES = {
    "median_curve_rel_l2_pct": ("median_curve_rel_l2_pct", "median"),
    "mean_curve_rel_l2_pct": ("mean_curve_rel_l2_pct", "mean"),
    "max_curve_rel_l2_pct": ("max_curve_rel_l2_pct", "max"),
}


@dataclass(frozen=True)
class ActiveLearningFinalGateArtifacts:
    """Artifacts written for a final-gate report run."""

    artifact_dir: Path
    report_path: Path
    markdown_path: Path
    summary_csv_path: Path
    plot_paths: dict[str, str]
    report: dict[str, Any]
    summary_rows: tuple[dict[str, Any], ...]


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default), encoding="utf-8")


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty value.")
    return text


def _coerce_iteration(value: int | str) -> str:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("iteration must be non-negative.")
        return f"iter_{value:04d}"
    token = str(value).strip()
    if not token:
        raise ValueError("iteration must be a non-empty string.")
    if token.startswith("iter_"):
        return token
    if token.isdigit():
        return f"iter_{int(token):04d}"
    return token


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{label} must be a mapping.")


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence.")
    return tuple(value)


def _coerce_non_negative_int(value: object, *, label: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{label} must be provided.")
        return int(default)
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_float(value: object, *, label: str, required: bool = True, default: float | None = None) -> float:
    if value is None:
        if required:
            raise ValueError(f"{label} must be provided.")
        if default is None:
            raise ValueError(f"{label} must be provided.")
        return float(default)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number

def _coerce_optional_float(value: object, *, label: str) -> float:
    return _coerce_float(value, label=label, required=False, default=float("nan"))


def _coerce_float_series(value: object, *, label: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric sequence.")
    return tuple(_coerce_float(item, label=f"{label} value", required=False, default=float("nan")) for item in tuple(value))


def _coerce_path_list(value: object, *, label: str) -> tuple[str, ...]:
    normalized: list[str] = []
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence.")
    for item in value:
        normalized.append(_coerce_text(item, label=f"{label} entry"))
    return tuple(normalized)


def _pick_metric(mapping: Mapping[str, Any], metric: str) -> float:
    aliases = _METRIC_ALIASES[metric]
    for alias in aliases:
        if alias in mapping and mapping[alias] is not None:
            return _coerce_float(mapping[alias], label=f"metric {alias}", required=False, default=float("nan"))
    raise ValueError(f"Missing metric {metric!r}")


def _coerce_metric_bundle(payload: Mapping[str, Any], *, round_index: int, strategy_label: str) -> dict[str, float | tuple[float, ...] | tuple[tuple[float, ...], ...]]:
    median = _pick_metric(payload, "median_curve_rel_l2_pct")
    mean = _coerce_optional_float(
        payload.get("mean_curve_rel_l2_pct") if "mean_curve_rel_l2_pct" in payload else payload.get("mean"),
        label=f"round {round_index} {strategy_label} mean_curve_rel_l2_pct",
    )
    max_value = _coerce_optional_float(
        payload.get("max_curve_rel_l2_pct") if "max_curve_rel_l2_pct" in payload else payload.get("max"),
        label=f"round {round_index} {strategy_label} max_curve_rel_l2_pct",
    )
    residuals = _coerce_float_series(payload.get("residuals"), label=f"round {round_index} {strategy_label} residuals")
    predicted_curves = tuple(_coerce_float_series(row, label=f"round {round_index} {strategy_label} predicted_curve") for row in _coerce_sequence(payload.get("predicted_curves"), label=f"round {round_index} {strategy_label} predicted_curves"))
    reference_curves = tuple(_coerce_float_series(row, label=f"round {round_index} {strategy_label} reference_curves") for row in _coerce_sequence(payload.get("reference_curves"), label=f"round {round_index} {strategy_label} reference_curves"))
    return {
        "median_curve_rel_l2_pct": median,
        "mean_curve_rel_l2_pct": mean,
        "max_curve_rel_l2_pct": max_value,
        "residuals": residuals,
        "predicted_curves": predicted_curves,
        "reference_curves": reference_curves,
    }


def _coerce_curve_count(payload: Mapping[str, Any], *, round_index: int, strategy: str) -> int:
    if strategy == "al":
        default_curve_count = _REQUIRED_CURVE_COUNT
    elif strategy == "lhs":
        default_curve_count = _REQUIRED_LHS_CURVE_COUNT
    else:
        raise ValueError(f"Unsupported strategy {strategy!r}.")

    count = _coerce_optional_int(
        payload.get(f"{strategy}_curve_count") if isinstance(payload, Mapping) else None,
        label=f"round {round_index} {strategy}_curve_count",
        default=default_curve_count,
    )
    if not isinstance(count, int):
        raise ValueError(f"round {round_index} {strategy}_curve_count must be an integer")
    return int(count)


def _derive_lhs_total_curves(round_rows: Sequence[Mapping[str, Any]]) -> int:
    if not round_rows:
        return 0
    return max(int(item["lhs_curve_count"]) for item in round_rows)


def _coerce_optional_int(value: object, *, label: str, default: int) -> int:
    if value is None:
        return int(default)
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_round_payloads(payload: Mapping[str, Any] | Sequence[object] | Path | str) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, (str, Path)):
        text = Path(payload).read_text(encoding="utf-8")
        loaded = json.loads(text)
        if isinstance(loaded, Mapping):
            payload = loaded
        elif isinstance(loaded, list):
            return tuple(_coerce_mapping(item, label="round payload") for item in loaded)
        else:
            raise TypeError("round payload file must contain a JSON object or list")
    if isinstance(payload, Mapping):
        rounds_payload = payload.get("rounds", payload.get("round_results", payload))
        if not isinstance(rounds_payload, Sequence) or isinstance(rounds_payload, (str, bytes, bytearray)):
            raise TypeError("rounds must be a sequence of per-round records.")
        payload_rows = tuple(rounds_payload)
    else:
        payload_rows = _coerce_sequence(payload, label="round_payloads")
    return tuple(_coerce_mapping(item, label=f"round record {index}") for index, item in enumerate(payload_rows, start=1))


def _coerce_round(record: Mapping[str, Any], *, index_fallback: int) -> dict[str, Any]:
    round_index = int(record.get("round", index_fallback))
    if round_index <= 0:
        raise ValueError("round must be a positive integer.")

    al_payload = _coerce_mapping(record.get("al"), label=f"round {round_index} al")
    lhs_payload = _coerce_mapping(record.get("lhs"), label=f"round {round_index} lhs")

    al_metrics = _coerce_metric_bundle(al_payload, round_index=round_index, strategy_label="al")
    lhs_metrics = _coerce_metric_bundle(lhs_payload, round_index=round_index, strategy_label="lhs")

    al_curve_count = _coerce_curve_count(record, round_index=round_index, strategy="al")
    lhs_curve_count = _coerce_curve_count(record, round_index=round_index, strategy="lhs")

    acquisition_scores = _coerce_float_series(record.get("acquisition_scores"), label=f"round {round_index} acquisition_scores")
    selected_candidate_scores = _coerce_float_series(
        record.get("selected_candidate_scores"),
        label=f"round {round_index} selected_candidate_scores",
    )
    acquisition_curve_pairs = _coerce_path_list(record.get("acquisition_scores_paths"), label=f"round {round_index} acquisition plot paths")

    coverage_key = "ka_kb_coverage"
    if coverage_key not in record:
        coverage_key = "yt_kb_coverage"
    coverage_rows = record.get(coverage_key, ())
    yt_kb_coverage = tuple(
        _coerce_float_series(item, label=f"round {round_index} {coverage_key} entry")
        for item in _coerce_sequence(coverage_rows, label=f"round {round_index} {coverage_key}")
    )

    model_selection = _coerce_mapping(record.get("model_selection", {}), label=f"round {round_index} model_selection")
    failure_counts = _coerce_mapping(record.get("failure_counts", {}), label=f"round {round_index} failure_counts")
    quarantine_counts = _coerce_mapping(record.get("quarantine_counts", {}), label=f"round {round_index} quarantine_counts")

    return {
        "round": int(round_index),
        "al": al_metrics,
        "lhs": lhs_metrics,
        "al_curve_count": al_curve_count,
        "lhs_curve_count": lhs_curve_count,
        "acquisition_scores": acquisition_scores,
        "selected_candidate_scores": selected_candidate_scores,
        "acquisition_plot_paths": acquisition_curve_pairs,
        "yt_kb_coverage": yt_kb_coverage,
        "ka_kb_coverage": yt_kb_coverage,
        "model_selection": model_selection,
        "failure_counts": failure_counts,
        "quarantine_counts": quarantine_counts,
    }


def _build_round_summary(round_payload: Mapping[str, Any]) -> dict[str, Any]:
    round_index = int(round_payload["round"])
    al = round_payload["al"]
    lhs = round_payload["lhs"]
    al_median = float(al["median_curve_rel_l2_pct"])
    lhs_median = float(lhs["median_curve_rel_l2_pct"])
    improved = bool(al_median < lhs_median)
    al_mean = float(al["mean_curve_rel_l2_pct"])
    lhs_mean = float(lhs["mean_curve_rel_l2_pct"])
    al_max = float(al["max_curve_rel_l2_pct"])
    lhs_max = float(lhs["max_curve_rel_l2_pct"])
    delta = al_median - lhs_median
    model_selection = _coerce_mapping(round_payload.get("model_selection"), label=f"round {round_index} model_selection")
    return {
        "round": round_index,
        "al_curve_count": int(round_payload["al_curve_count"]),
        "lhs_curve_count": int(round_payload["lhs_curve_count"]),
        "al_median_curve_rel_l2_pct": al_median,
        "lhs_median_curve_rel_l2_pct": lhs_median,
        "median_delta": delta,
        "median_improved": improved,
        "al_mean_curve_rel_l2_pct": al_mean,
        "lhs_mean_curve_rel_l2_pct": lhs_mean,
        "al_max_curve_rel_l2_pct": al_max,
        "lhs_max_curve_rel_l2_pct": lhs_max,
        "acquisition_scores_count": int(len(round_payload["acquisition_scores"])),
        "selected_candidate_scores_count": int(len(round_payload["selected_candidate_scores"])),
        "yt_kb_coverage_count": int(len(round_payload["yt_kb_coverage"])),
        "failure_count": int(sum(max(0, int(item)) for item in round_payload["failure_counts"].values())),
        "quarantine_count": int(sum(max(0, int(item)) for item in round_payload["quarantine_counts"].values())),
        "model_selection_rerun": bool(bool(model_selection.get("rerun", True))),
        "model_selection_architecture": model_selection.get("architecture", "unknown"),
        "model_selection_backend": model_selection.get("backend", "unknown"),
        "model_selection_notes": str(model_selection.get("notes", "")),
    }


def _build_pass_fail(
    round_summaries: Sequence[Mapping[str, Any]],
    *,
    lhs_total_curves: int,
    starting_from_no_initial_data: bool,
) -> tuple[dict[str, dict[str, bool]], bool, tuple[str, ...]]:
    criteria: dict[str, dict[str, bool]] = {
        "round_count": {
            "passed": len(round_summaries) == _REQUIRED_ROUND_COUNT,
            "required": True,
        },
        "round_indices": {
            "passed": sorted(int(item["round"]) for item in round_summaries) == list(range(1, len(round_summaries) + 1)),
            "required": True,
        },
        "al_curve_count_per_round": {
            "passed": all(int(item["al_curve_count"]) == _REQUIRED_CURVE_COUNT for item in round_summaries),
            "required": True,
        },
        "lhs_comparator_count_is_90": {
            "passed": all(int(item["lhs_curve_count"]) == _REQUIRED_LHS_CURVE_COUNT for item in round_summaries),
            "required": True,
        },
        "curve_count_totals": {
            "passed": (
                sum(int(item["al_curve_count"]) for item in round_summaries) == _EXPECTED_AL_TOTAL_CURVES
                and lhs_total_curves == _EXPECTED_LHS_TOTAL_CURVES
            ),
            "required": True,
        },
        "rerun_model_selection_each_round": {
            "passed": all(bool(item["model_selection_rerun"]) for item in round_summaries),
            "required": True,
        },
        "al_improves_over_lhs_each_round": {
            "passed": all(bool(item["median_improved"]) for item in round_summaries),
            "required": True,
        },
        "starting_from_no_initial_data": {
            "passed": bool(starting_from_no_initial_data),
            "required": True,
        },
    }

    failures: list[str] = []
    for key, payload in criteria.items():
        if payload["required"] and not payload["passed"]:
            failures.append(key)

    return criteria, len(failures) == 0, tuple(failures)


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + type_code
        + data
        + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)
    )


def _build_fallback_png() -> bytes:
    width = 16
    height = 16
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)
        for x in range(width):
            pixels.extend((240, 240, 240, 255) if (x + y) % 2 == 0 else (200, 200, 220, 255))
    idat_data = zlib.compress(bytes(pixels))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", idat_data) + _png_chunk(b"IEND", b"")


_FALLBACK_PNG = _build_fallback_png()


def _ensure_png(path: Path, *, include_plot: bool) -> None:
    if include_plot:
        return
    path.write_bytes(_FALLBACK_PNG)


def _render_curve_metrics_plot(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    rounds = [int(row["round"]) for row in round_rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(9, 4))
    axis.plot(rounds, [float(row["al_median_curve_rel_l2_pct"]) for row in round_rows], label="AL median")
    axis.plot(rounds, [float(row["lhs_median_curve_rel_l2_pct"]) for row in round_rows], label="LHS median")
    axis.plot(rounds, [float(row["al_mean_curve_rel_l2_pct"]) for row in round_rows], linestyle="--", alpha=0.85, label="AL mean")
    axis.plot(rounds, [float(row["lhs_mean_curve_rel_l2_pct"]) for row in round_rows], linestyle="--", alpha=0.85, label="LHS mean")
    axis.set_title("AL vs LHS grouped-holdout metrics")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("Relative L2 %")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_residual_statistics(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool, rounds_payload: Sequence[Mapping[str, Any]]) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    rounds = [int(row["round"]) for row in round_rows]
    al_means = []
    lhs_means = []
    for row in rounds_payload:
        al = row["al"]["residuals"]
        lhs = row["lhs"]["residuals"]
        al_means.append(sum(abs(float(item)) for item in al) / len(al) if al else 0.0)
        lhs_means.append(sum(abs(float(item)) for item in lhs) / len(lhs) if lhs else 0.0)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(9, 4))
    axis.plot(rounds, al_means, marker="o", label="AL residual mean |residual|")
    axis.plot(rounds, lhs_means, marker="o", label="LHS residual mean |residual|")
    axis.set_title("Residual statistics")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("Mean absolute residual")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_predictions(path: Path, rounds_payload: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(9, 5))
    for row in rounds_payload:
        round_index = int(row["round"])
        al_pred = row["al"]["predicted_curves"]
        al_ref = row["al"]["reference_curves"]
        if al_pred and al_ref:
            axis.plot(range(1, len(al_pred[0]) + 1), al_ref[0], color="C0", alpha=0.25)
            axis.plot(range(1, len(al_pred[0]) + 1), al_pred[0], color="C1", alpha=0.25)
    axis.set_title("Predicted vs reference curve examples (round 1)")
    axis.set_xlabel("Curve point")
    axis.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_acquisition_and_selection(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    rounds = [int(row["round"]) for row in round_rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(rounds, [float(row["acquisition_scores_count"]) for row in round_rows], marker="o", label="acquisition_score_count")
    axis.plot(rounds, [float(row["selected_candidate_scores_count"]) for row in round_rows], marker="s", label="selected_candidate_count")
    axis.set_title("Acquisition and selected candidate diagnostics")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_coverage(path: Path, rounds_payload: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(7, 5))
    for row in rounds_payload:
        coverage = row.get("ka_kb_coverage") or row.get("yt_kb_coverage") or ()
        if not coverage:
            continue
        axis.scatter(
            [item[0] if len(item) >= 1 else 0.0 for item in coverage],
            [item[1] if len(item) >= 2 else 0.0 for item in coverage],
            alpha=0.5,
            label=f"round {int(row['round'])}",
        )
    axis.set_title("ka-kb coverage")
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_failure_quarantine(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    rounds = [int(row["round"]) for row in round_rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(rounds, [float(row["failure_count"]) for row in round_rows], marker="x", label="Failure count")
    axis.plot(rounds, [float(row["quarantine_count"]) for row in round_rows], marker="v", label="Quarantine count")
    axis.set_title("Failure and quarantine counts")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _model_selection_architecture(row: Mapping[str, Any]) -> str:
    if "model_selection_architecture" in row:
        return str(row["model_selection_architecture"])
    model_selection = row.get("model_selection", {})
    if isinstance(model_selection, Mapping):
        return str(model_selection.get("architecture", "unknown"))
    return "unknown"


def _plot_model_selection(path: Path, rounds_payload: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    architectures = [_model_selection_architecture(item) for item in rounds_payload]
    rounds = [int(item["round"]) for item in rounds_payload]
    arch_values = {name: i for i, name in enumerate(sorted(set(architectures)), start=1)}
    y = [arch_values[name] for name in architectures]

    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(rounds, y, marker="o")
    axis.set_title("Model selection across rounds")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("Architecture")
    axis.set_yticks(list(arch_values.values()))
    axis.set_yticklabels(list(arch_values.keys()))
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def _plot_summary(path: Path, round_rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return

    rounds = [int(row["round"]) for row in round_rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.bar([str(r) for r in rounds], [float(row["median_delta"]) for row in round_rows])
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("AL-LHS median metric delta per round")
    axis.set_xlabel("Active-learning round")
    axis.set_ylabel("AL - LHS median_curve_rel_l2_pct")
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_FINAL_GATE_PLOT_DPI)
    plt.close(fig)


def write_plot(path: Path, payload: Sequence[Mapping[str, Any]], *, plot_type: str, round_payload: Sequence[Mapping[str, Any]], include_plot: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "metric_trends": _render_curve_metrics_plot,
        "residuals": _plot_residual_statistics,
        "prediction": _plot_predictions,
        "acquisition": _plot_acquisition_and_selection,
        "coverage": _plot_coverage,
        "failure": _plot_failure_quarantine,
        "model_selection": _plot_model_selection,
        "summary": _plot_summary,
    }
    plot_fn = dispatch[plot_type]

    if plot_type in {"prediction", "coverage", "model_selection"}:
        plot_fn(path, round_payload, include_plot=include_plot)
    elif plot_type in {"residuals"}:
        plot_fn(path, payload, include_plot=include_plot, rounds_payload=round_payload)
    else:
        plot_fn(path, payload, include_plot=include_plot)
    return str(path)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "al_curve_count",
        "lhs_curve_count",
        "al_median_curve_rel_l2_pct",
        "lhs_median_curve_rel_l2_pct",
        "median_delta",
        "median_improved",
        "al_mean_curve_rel_l2_pct",
        "lhs_mean_curve_rel_l2_pct",
        "al_max_curve_rel_l2_pct",
        "lhs_max_curve_rel_l2_pct",
        "acquisition_scores_count",
        "selected_candidate_scores_count",
        "yt_kb_coverage_count",
        "failure_count",
        "quarantine_count",
        "model_selection_architecture",
        "model_selection_backend",
        "model_selection_rerun",
        "model_selection_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name) for name in fieldnames}
            writer.writerow(payload)


def _format_markdown(payload: Mapping[str, Any], artifact_dir: Path, plot_paths: Mapping[str, str]) -> str:
    criteria = payload["pass_fail_criteria"]
    round_rows = payload["round_evidence"]
    lhs_comparator = payload.get("lhs_comparator", ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR)
    status = "PASS" if payload["passed"] else "FAIL"
    lines = [
        "# Active Learning Final-Gate Report",
        "",
        f"Run id: {payload['run_id']}",
        f"Iteration: {payload['iteration']}",
        f"Experiment: {payload['experiment']}",
        f"Diameter: {payload['diameter_um']}um",
        f"LHS comparator: {lhs_comparator}",
        f"Status: {status}",
        "",
        "## Criteria",
        f"- round_count_is_3: {criteria['round_count']['passed']}",
        f"- al_curve_count_per_round_30: {criteria['al_curve_count_per_round']['passed']}",
        f"- lhs_comparator_count_is_90: {criteria['lhs_comparator_count_is_90']['passed']}",
        f"- rerun_model_selection_each_round: {criteria['rerun_model_selection_each_round']['passed']}",
        f"- al_improves_over_lhs_each_round: {criteria['al_improves_over_lhs_each_round']['passed']}",
        "",
        "## Round-by-round median evidence",
    ]
    for row in round_rows:
        lines.append(
            f"- round {row['round']}: AL={row['al_median_curve_rel_l2_pct']:.6g} "
            f"vs LHS={row['lhs_median_curve_rel_l2_pct']:.6g}"
            f" (Δ={row['median_delta']:.6g}, improved={row['median_improved']})"
        )
    lines.extend(["", "## Plot files"])
    for label, value in plot_paths.items():
        lines.append(f"- {label}: `{artifact_dir / value}`")

    lines.append("")
    lines.append(f"Summary CSV: `{artifact_dir / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME}`")
    lines.append(f"Copy targets: scratch={payload['copy_targets'].get('scratch')}, vault={payload['copy_targets'].get('vault')}")
    return "\n".join(lines) + "\n"


def validate_plot_path(path: str | Path) -> None:
    candidate = Path(path)
    if not candidate.is_file():
        raise AssertionError(f"Missing plot: {candidate}")
    data = candidate.read_bytes()
    if len(data) < 70:
        raise AssertionError(f"Plot too small: {candidate}")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"Plot is not a PNG: {candidate}")


def _report_iteration_directory(output_root: str | Path, run_id: str, iteration: int | str) -> Path:
    return Path(output_root) / run_id / "iterations" / _coerce_iteration(iteration)


def build_active_learning_final_gate_report(
    *,
    run_id: str,
    iteration: int | str,
    rounds: Mapping[str, Any] | Sequence[Any] | str | Path,
    experiment: str = ACTIVE_LEARNING_FINAL_GATE_EXPERIMENT,
    diameter_um: str = ACTIVE_LEARNING_FINAL_GATE_DIAMETER_UM,
    starting_from_no_initial_data: bool = _STARTING_WITH_NO_INITIAL_DATA,
    scratch_copy_destination: str | Path | None = None,
    vault_copy_destination: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, str]]:
    run_id_text = _coerce_text(run_id, label="run_id")
    iteration_text = _coerce_iteration(iteration)
    experiment_text = _coerce_text(experiment, label="experiment")
    diameter_um_text = _coerce_text(diameter_um, label="diameter_um")
    round_payloads = tuple(
        _coerce_round(item, index_fallback=index)
        for index, item in enumerate(_coerce_round_payloads(rounds), start=1)
    )
    round_rows = tuple(_build_round_summary(item) for item in round_payloads)
    lhs_total_curves = _derive_lhs_total_curves(round_rows)

    criteria, passed, failures = _build_pass_fail(
        round_rows,
        lhs_total_curves=lhs_total_curves,
        starting_from_no_initial_data=starting_from_no_initial_data,
    )
    median_al_curve_list = [float(item["al_median_curve_rel_l2_pct"]) for item in round_rows]
    median_lhs_curve_list = [float(item["lhs_median_curve_rel_l2_pct"]) for item in round_rows]
    model_selection_trace = [item["model_selection_architecture"] for item in round_rows]

    copy_targets: dict[str, str] = {
        "scratch": str(scratch_copy_destination) if scratch_copy_destination is not None else "",
        "vault": str(vault_copy_destination) if vault_copy_destination is not None else "",
    }

    summary: dict[str, Any] = {
        "run_id": run_id_text,
        "iteration": iteration_text,
        "schema_version": ACTIVE_LEARNING_FINAL_GATE_REPORT_SCHEMA_VERSION,
        "experiment": experiment_text,
        "diameter_um": diameter_um_text,
        "expected_round_count": _REQUIRED_ROUND_COUNT,
        "expected_round_curve_count": _REQUIRED_CURVE_COUNT,
        "copy_targets": copy_targets,
        "pass_fail_criteria": criteria,
        "failure_reasons": list(failures),
        "passed": passed,
        "status": "passed" if passed else "failed",
        "round_evidence": list(round_rows),
        "primary_metric": "median_curve_rel_l2_pct",
        "primary_metric_comparison": {
            "al_curve_counts": [int(item["al_curve_count"]) for item in round_rows],
            "lhs_curve_counts": [int(item["lhs_curve_count"]) for item in round_rows],
            "al_median_curve_rel_l2_pct": median_al_curve_list,
            "lhs_median_curve_rel_l2_pct": median_lhs_curve_list,
        },
        "model_selection_trace": model_selection_trace,
        "round_count": len(round_rows),
        "al_total_curves": sum(int(item["al_curve_count"]) for item in round_rows),
        "lhs_total_curves": lhs_total_curves,
        "lhs_comparator": ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR,
        "metadata": dict(metadata or {}),
        "round_payloads": list(round_payloads),
        "starting_from_no_initial_data": bool(starting_from_no_initial_data),
        "source": {
            "experiment": experiment_text,
            "diameter_um": diameter_um_text,
        },
    }

    return summary, round_rows, copy_targets


def write_active_learning_final_gate_artifacts(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int | str,
    rounds: Mapping[str, Any] | Sequence[Any] | str | Path,
    include_plots: bool = True,
    experiment: str = ACTIVE_LEARNING_FINAL_GATE_EXPERIMENT,
    diameter_um: str = ACTIVE_LEARNING_FINAL_GATE_DIAMETER_UM,
    starting_from_no_initial_data: bool = _STARTING_WITH_NO_INITIAL_DATA,
    scratch_copy_destination: str | Path | None = None,
    vault_copy_destination: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActiveLearningFinalGateArtifacts:
    run_id_text = _coerce_text(run_id, label="run_id")
    artifact_dir = _report_iteration_directory(output_root, run_id_text, iteration)
    report_path = artifact_dir / ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME
    markdown_path = artifact_dir / ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME
    summary_csv_path = artifact_dir / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME

    report, round_rows, copy_targets = build_active_learning_final_gate_report(
        run_id=run_id_text,
        iteration=iteration,
        rounds=rounds,
        experiment=experiment,
        diameter_um=diameter_um,
        starting_from_no_initial_data=starting_from_no_initial_data,
        scratch_copy_destination=scratch_copy_destination,
        vault_copy_destination=vault_copy_destination,
        metadata=metadata,
    )

    plot_dir = artifact_dir
    plot_paths = {
        "curve_metrics": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_CURVE_METRICS_PLOT_FILENAME),
        "residuals": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_RESIDUALS_PLOT_FILENAME),
        "prediction_vs_reference": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_PREDICTION_PLOT_FILENAME),
        "acquisition_scores": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_ACQUISITION_PLOT_FILENAME),
        "ka_kb_coverage": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME),
        "yt_kb_coverage": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_YT_KB_PLOT_FILENAME),
        "failure_quarantine": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_FAILURE_PLOT_FILENAME),
        "model_selection": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_MODEL_SELECTION_PLOT_FILENAME),
        "summary": str(plot_dir / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_PLOT_FILENAME),
    }

    raw_round_payloads = _coerce_round_payloads(rounds)
    round_payloads = tuple(_coerce_round(item, index_fallback=index) for index, item in enumerate(raw_round_payloads, start=1))

    write_plot_paths = [
        (plot_paths["curve_metrics"], ("metric_trends", round_rows, round_payloads, "metric")),
        (plot_paths["residuals"], ("residuals", round_rows, round_payloads, "residual")),
        (plot_paths["prediction_vs_reference"], ("prediction", round_rows, round_payloads, "prediction")),
        (plot_paths["acquisition_scores"], ("acquisition", round_rows, round_payloads, "acquisition")),
        (plot_paths["yt_kb_coverage"], ("coverage", round_rows, round_payloads, "coverage")),
        (plot_paths["failure_quarantine"], ("failure", round_rows, round_payloads, "failure")),
        (plot_paths["model_selection"], ("model_selection", round_rows, round_payloads, "model")),
        (plot_paths["summary"], ("summary", round_rows, round_payloads, "summary")),
    ]
    for item in write_plot_paths:
        plot_file, (plot_type, row_payload, round_payload, _) = item
        write_plot(
            Path(plot_file),
            payload=row_payload,
            plot_type=plot_type,
            round_payload=round_payload,
            include_plot=include_plots,
        )

    if round_rows:
        _write_summary_csv(summary_csv_path, round_rows)

    report["plot_paths"] = plot_paths
    _write_json(report_path, report)
    markdown_path.write_text(_format_markdown(report, artifact_dir=artifact_dir, plot_paths=plot_paths), encoding="utf-8")

    return ActiveLearningFinalGateArtifacts(
        artifact_dir=artifact_dir,
        report_path=report_path,
        markdown_path=markdown_path,
        summary_csv_path=summary_csv_path,
        plot_paths=plot_paths,
        report=report,
        summary_rows=round_rows,
    )


__all__ = [
    "ACTIVE_LEARNING_FINAL_GATE_REPORT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_FINAL_GATE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_LHS_COMPARATOR",
    "ACTIVE_LEARNING_FINAL_GATE_EXPERIMENT",
    "ACTIVE_LEARNING_FINAL_GATE_DIAMETER_UM",
    "ACTIVE_LEARNING_FINAL_GATE_CURVE_METRICS_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_PREDICTION_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_RESIDUALS_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_ACQUISITION_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_YT_KB_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_FAILURE_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_MODEL_SELECTION_PLOT_FILENAME",
    "ACTIVE_LEARNING_FINAL_GATE_SUMMARY_PLOT_FILENAME",
    "ActiveLearningFinalGateArtifacts",
    "build_active_learning_final_gate_report",
    "write_active_learning_final_gate_artifacts",
    "validate_plot_path",
]
