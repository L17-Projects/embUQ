from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import math
import statistics
import numpy as np

from meso_uq.active_learning.emb_34um_dnn_surrogate import (
    EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
    DnnSurrogateLongRow,
    Emb34umDnnSurrogateFit,
    load_ensemble_member_predictions,
)


EMB_34UM_DNN_SURROGATE_SELECTION_REPORT_FILENAME = "emb_34um_dnn_causal_selection_report.json"
EMB_34UM_DNN_SURROGATE_SELECTION_MANIFEST_FILENAME = "emb_34um_dnn_causal_selection_manifest.json"
EMB_34UM_DNN_SURROGATE_HEATMAP_PLOT_FILENAME = "emb_34um_dnn_causal_candidate_score_heatmap.png"
EMB_34UM_DNN_SURROGATE_DIVERSITY_HIST_PLOT_FILENAME = "emb_34um_dnn_causal_disagreement_distribution.png"
EMB_34UM_DNN_SURROGATE_LOSS_PLOT_FILENAME = "emb_34um_dnn_causal_member_losses.png"
EMB_34UM_DNN_SURROGATE_TIMING_CANARY_PLOT_FILENAME = "emb_34um_dnn_causal_timing_canary.png"

EMB_34UM_DNN_SURROGATE_TIMING_CANARY_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_timing_canary.v1"
EMB_34UM_DNN_SURROGATE_SELECTION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_selection.v1"


def _coerce_sequence(values: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(values)


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _coerce_positive_float(value: object, *, label: str) -> float:
    number = _coerce_float(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_positive_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _coerce_candidate_point(item: Mapping[str, Any], *, index: int) -> tuple[float, float]:
    ka = _coerce_positive_float(item.get("ka", item.get("Yt", 1.0)), label=f"candidate[{index}].ka")
    kb = _coerce_positive_float(item.get("kb", item.get("kb_scale", item.get("mu", 1.0))), label=f"candidate[{index}].kb")
    return math.log10(ka), math.log10(kb)


def _coerce_log10_point(item: object, *, label: str) -> tuple[float, float]:
    if isinstance(item, Mapping):
        return _coerce_candidate_point(item, index=0)
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
        ka = _coerce_positive_float(item[0], label=f"{label}[0]")
        kb = _coerce_positive_float(item[1], label=f"{label}[1]")
        return math.log10(ka), math.log10(kb)
    raise ValueError(f"{label} must be a mapping with ka/kb or a 2-item numeric sequence.")


def _point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.dist(left, right)


def _coerce_force_grid(values: object) -> tuple[float, ...]:
    raw = _coerce_sequence(values, label="force_grid")
    parsed = [_coerce_float(item, label="force_grid") for item in raw]
    if not parsed:
        raise ValueError("force_grid must not be empty.")
    return tuple(float(v) for v in parsed)


def _predict_candidate_disagreement(
    fit: Emb34umDnnSurrogateFit,
    candidate: Mapping[str, Any],
    force_grid: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    ka = _coerce_positive_float(candidate.get("ka", candidate.get("Yt", 1.0)), label="candidate.ka")
    kb = _coerce_positive_float(candidate.get("kb", candidate.get("kb_scale", 1.0)), label="candidate.kb")
    mean_curve, member_curves = load_ensemble_member_predictions(fit, force_grid, ka=ka, kb=kb)
    if not member_curves:
        return (), 0.0
    stacked = np.array(member_curves, dtype=float)
    disagreement = float(np.mean(np.std(stacked, axis=0)))
    return mean_curve, disagreement


def _coerce_count(value: object, *, label: str) -> int:
    number = _coerce_positive_int(value, label=label)
    return number


def score_candidate_pool(
    fit: Emb34umDnnSurrogateFit,
    candidate_records: Sequence[Mapping[str, Any]],
    force_grid: Sequence[float],
    *,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    """Score candidates by ensemble disagreement plus optional nearest-point diversity term.

    Scoring fields:
    - ensemble_disagreement: mean std. across members
    - diversity_term: min distance to existing points in log10 space
    - acquisition_score: disagreement + diversity_weight * diversity_term
    """

    rows = _coerce_sequence(candidate_records, label="candidate_records")
    if not rows:
        return ()
    force_axis = _coerce_force_grid(force_grid)
    if not fit.ensemble_checkpoints:
        raise ValueError("fit must include ensemble_checkpoints.")
    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative.")

    existing = []
    for index, item in enumerate(_coerce_sequence(existing_points or (), label="existing_points"), start=1):
        existing.append(_coerce_log10_point(item, label=f"existing_points[{index}]"))

    scored: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows, start=1):
        row = _coerce_mapping(raw_row, label=f"candidate_records[{index}]")
        mean_curve, disagreement = _predict_candidate_disagreement(fit, row, force_axis)
        point = _coerce_candidate_point(row, index=index)

        if existing:
            distance = min(_point_distance(point, candidate_point) for candidate_point in existing)
        else:
            distance = 0.0

        payload = dict(row)
        payload["candidate_index"] = index
        payload["candidate_log10_point"] = point
        payload["predicted_curve"] = tuple(float(item) for item in mean_curve)
        payload["ensemble_disagreement"] = float(disagreement)
        payload["diversity_term"] = float(distance)
        payload["acquisition_score"] = float(disagreement) + float(diversity_weight) * float(distance)
        scored.append(payload)

    scored.sort(
        key=lambda item: (
            -float(item["acquisition_score"]),
            -float(item["ensemble_disagreement"]),
            float(item["candidate_log10_point"][0]),
            float(item["candidate_log10_point"][1]),
        )
    )
    return tuple(scored)


def select_greedy_diversity(
    scored_candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    """Greedy diversity selection over scored candidates."""

    rows = _coerce_sequence(scored_candidates, label="scored_candidates")
    target = _coerce_count(count, label="count")
    if target <= 0 or not rows:
        return ()

    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative.")

    remaining: list[dict[str, Any]] = [dict(item) for item in rows]
    selected_points: list[tuple[float, float]] = [
        _coerce_log10_point(point, label=f"existing_points[{index}]")
        for index, point in enumerate(existing_points or (), start=1)
    ]

    selected: list[dict[str, Any]] = []
    for _ in range(target):
        if not remaining:
            break

        best_index = -1
        best_value = (-math.inf, -math.inf, -math.inf, math.inf, math.inf)
        for position, row in enumerate(remaining):
            point = row.get("candidate_log10_point")
            if not isinstance(point, tuple):
                point = tuple(row.get("candidate_log10_point", ()))  # pragma: no cover - defensive
            point_tuple = (float(point[0]), float(point[1])) if len(point) == 2 else (0.0, 0.0)

            if selected_points:
                nearest = min(_point_distance(point_tuple, pick) for pick in selected_points)
            else:
                nearest = 0.0
            disagreement = float(row["ensemble_disagreement"])
            score = disagreement + float(diversity_weight) * float(nearest)

            candidate_sort = (
                score,
                disagreement,
                nearest,
                -point_tuple[0],
                -point_tuple[1],
            )
            if candidate_sort > best_value:
                best_index = position
                best_value = candidate_sort
                row["diversity_term"] = nearest
                row["acquisition_score"] = score

        if best_index < 0:
            break

        selected_row = remaining.pop(best_index)
        selected_point = selected_row.get("candidate_log10_point")
        if isinstance(selected_point, tuple) and len(selected_point) == 2:
            selected_points.append((float(selected_point[0]), float(selected_point[1])))
        selected.append(selected_row)

    return tuple(selected)


def select_candidates(
    scored_candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    existing_points: Sequence[tuple[float, float]] | None = None,
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
) -> tuple[Mapping[str, Any], ...]:
    return select_greedy_diversity(
        scored_candidates,
        count=count,
        existing_points=existing_points,
        diversity_weight=diversity_weight,
    )


def write_candidate_selection_artifacts(
    output_root: Path,
    fit: Emb34umDnnSurrogateFit,
    scored_candidates: Sequence[Mapping[str, Any]],
    selected_candidates: Sequence[Mapping[str, Any]],
    *,
    force_grid: Sequence[float],
    include_plot: bool = True,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_candidate_selection_report(
        fit=fit,
        scored_candidates=scored_candidates,
        selected_candidates=selected_candidates,
        force_grid=_coerce_force_grid(force_grid),
    )
    path = output_root / EMB_34UM_DNN_SURROGATE_SELECTION_REPORT_FILENAME
    payload_text = __import__("json").dumps(dict(payload), indent=2, sort_keys=True)
    path.write_text(payload_text, encoding="utf-8")
    return path


def build_candidate_selection_report(
    fit: Emb34umDnnSurrogateFit,
    scored_candidates: Sequence[Mapping[str, Any]],
    selected_candidates: Sequence[Mapping[str, Any]],
    *,
    force_grid: Sequence[float],
) -> dict[str, Any]:
    if not scored_candidates:
        score_values: tuple[float, ...] = ()
    else:
        score_values = tuple(float(item.get("acquisition_score", 0.0)) for item in scored_candidates)
    return {
        "schema_version": EMB_34UM_DNN_SURROGATE_SELECTION_SCHEMA_VERSION,
        "backend": fit.backend,
        "architecture": fit.architecture,
        "ensemble_size": fit.ensemble_member_count,
        "ensemble_seeds": list(fit.ensemble_seeds),
        "candidate_count": len(scored_candidates),
        "selected_count": len(selected_candidates),
        "acquisition_score_count": len(score_values),
        "acquisition_score": {
            "min": min(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
            "mean": statistics.mean(score_values) if score_values else 0.0,
        },
        "force_grid": _coerce_force_grid(force_grid),
        "candidate_scores": [dict(item) for item in scored_candidates],
        "selected_candidates": [dict(item) for item in selected_candidates],
    }


def compute_disagreement_distribution(selected: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    if not selected:
        return ()
    return tuple(float(item.get("ensemble_disagreement", 0.0)) for item in selected)


__all__ = [
    "EMB_34UM_DNN_SURROGATE_SELECTION_REPORT_FILENAME",
    "EMB_34UM_DNN_SURROGATE_SELECTION_MANIFEST_FILENAME",
    "EMB_34UM_DNN_SURROGATE_HEATMAP_PLOT_FILENAME",
    "EMB_34UM_DNN_SURROGATE_DIVERSITY_HIST_PLOT_FILENAME",
    "EMB_34UM_DNN_SURROGATE_LOSS_PLOT_FILENAME",
    "EMB_34UM_DNN_SURROGATE_TIMING_CANARY_PLOT_FILENAME",
    "EMB_34UM_DNN_SURROGATE_TIMING_CANARY_SCHEMA_VERSION",
    "score_candidate_pool",
    "select_greedy_diversity",
    "select_candidates",
    "build_candidate_selection_report",
    "write_candidate_selection_artifacts",
    "compute_disagreement_distribution",
]
