from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import math
import statistics
import numpy as np

from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION,
    is_dnn_causal_low_corner_excluded,
)
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
EMB_34UM_DNN_D4_ACTIVE_BOUNDS = dict(EMB_34UM_DNN_CAUSAL_BOUNDS)

EMB_34UM_DNN_SURROGATE_TIMING_CANARY_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_timing_canary.v1"
EMB_34UM_DNN_SURROGATE_SELECTION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_selection.v1"
EMB_34UM_DNN_D4_ACTIVE_DIMENSION_NAMES = ("ka", "kb", "radp", "shell_th")
EMB_34UM_DNN_D4_CANDIDATE_POOL_GENERATORS = ("auto", "random", "sobol", "stratified")
EMB_34UM_DNN_D4_CANDIDATE_POOL_MAX_ATTEMPTS_MULTIPLIER = 20
EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE = 5000
EMB_34UM_DNN_ACQUISITION_SCORE_MODES = ("disagreement", "curve_error")


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return number


def _coerce_candidate_pool_generator(value: object) -> str:
    text = str(value).strip().lower()
    if text not in EMB_34UM_DNN_D4_CANDIDATE_POOL_GENERATORS:
        raise ValueError(
            "candidate_pool_generator must be one of "
            + ", ".join(repr(item) for item in EMB_34UM_DNN_D4_CANDIDATE_POOL_GENERATORS)
            + "."
        )
    return text


def _coerce_bounds(
    value: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, tuple[float, float]]:
    source = dict(value) if value is not None else dict(EMB_34UM_DNN_D4_ACTIVE_BOUNDS)
    normalized: dict[str, tuple[float, float]] = {}
    for dimension_name in EMB_34UM_DNN_D4_ACTIVE_DIMENSION_NAMES:
        raw = source.get(dimension_name)
        if raw is None:
            raise ValueError(f"{label} is missing required dimension {dimension_name!r}.")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 2:
            raise ValueError(f"{label}[{dimension_name!r}] must be a two-item sequence.")
        lower = float(raw[0])
        upper = float(raw[1])
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
            raise ValueError(f"{label}[{dimension_name!r}] must define finite values with low < high.")
        normalized[dimension_name] = (lower, upper)
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    return normalized


def _to_log_space(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if lower <= 0.0 or upper <= 0.0 or lower >= upper:
        raise ValueError(f"Invalid bounds {bounds!r}.")
    clipped = min(1.0, max(0.0, float(value)))
    if clipped == 0.0:
        return float(lower)
    if clipped == 1.0:
        return float(upper)
    return 10.0 ** (math.log10(lower) + clipped * (math.log10(upper) - math.log10(lower)))


def _to_linear_space(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    width = float(upper) - float(lower)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"Invalid bounds {bounds!r}.")
    clipped = min(1.0, max(0.0, float(value)))
    if clipped == 0.0:
        return float(lower)
    if clipped == 1.0:
        return float(upper)
    return float(lower) + clipped * width


def _build_sobol_unit_points(
    *,
    count: int,
    dimensions: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        from scipy.stats import qmc
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ModuleNotFoundError("SciPy qmc is required for Sobol generation.") from exc
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    power = int(math.ceil(math.log2(max(1, count))))
    rows = sampler.random_base2(power)[:count]
    return tuple(tuple(float(value) for value in row) for row in rows)


def _build_stratified_unit_points(
    *,
    count: int,
    dimensions: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed)
    samples: list[list[float]] = [[0.0 for _ in range(dimensions)] for _ in range(count)]
    for dim_index in range(dimensions):
        order = list(range(count))
        rng.shuffle(order)
        for row_index, sample_index in enumerate(order):
            samples[sample_index][dim_index] = (row_index + rng.random()) / count
    return tuple(tuple(values) for values in samples)


def _build_random_unit_points(
    *,
    count: int,
    dimensions: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.random() for _ in range(dimensions)) for _ in range(count))


def _build_candidate_pool_metadata(
    *,
    requested_size: int,
    actual_size: int,
    seed: int,
    generator_type: str,
    bounds: Mapping[str, tuple[float, float]],
    dimension_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "generator_type": str(generator_type),
        "requested_size": int(requested_size),
        "actual_size": int(actual_size),
        "seed": int(seed),
        "bounds": {
            str(name): [float(values[0]), float(values[1])] for name, values in bounds.items()
        },
        "dimension_names": list(str(name) for name in dimension_names),
    }


def build_d4_candidate_pool(
    *,
    requested_size: int,
    seed: int,
    generator: str = "auto",
    existing_points: Sequence[tuple[float, ...]] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    max_attempts_multiplier: int | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Build a deterministic candidate pool for D4 AL selection.

    Returns:
        (candidate_pool, candidate_pool_metadata)

    The pool generator can be:
      - "sobol": use SciPy qmc Sobol (if available),
      - "stratified": deterministic stratified (fallback) sequence,
      - "random": unconstrained RNG sequence.
      - "auto": prefer Sobol when SciPy qmc exists, else stratified.
    """
    requested_size = _coerce_positive_int(requested_size, label="requested_size")
    seed_value = _coerce_positive_int(seed, label="seed")
    generator = _coerce_candidate_pool_generator(generator)
    bounds = _coerce_bounds(bounds, label="bounds")
    attempt_multiplier = _coerce_nonnegative_int(
        max_attempts_multiplier if max_attempts_multiplier is not None else EMB_34UM_DNN_D4_CANDIDATE_POOL_MAX_ATTEMPTS_MULTIPLIER,
        label="max_attempts_multiplier",
    )
    if attempt_multiplier < 1:
        attempt_multiplier = 1

    desired_raw_count = requested_size * attempt_multiplier
    if generator == "auto":
        try:
            unit_points = _build_sobol_unit_points(
                count=desired_raw_count,
                dimensions=4,
                seed=seed_value,
            )
            resolved_generator = "sobol"
        except ModuleNotFoundError:
            unit_points = _build_stratified_unit_points(
                count=desired_raw_count,
                dimensions=4,
                seed=seed_value,
            )
            resolved_generator = "stratified"
    elif generator == "sobol":
        try:
            unit_points = _build_sobol_unit_points(
                count=desired_raw_count,
                dimensions=4,
                seed=seed_value,
            )
            resolved_generator = "sobol"
        except ModuleNotFoundError:
            unit_points = _build_stratified_unit_points(
                count=desired_raw_count,
                dimensions=4,
                seed=seed_value,
            )
            resolved_generator = "stratified_fallback"
    elif generator == "stratified":
        unit_points = _build_stratified_unit_points(
            count=desired_raw_count,
            dimensions=4,
            seed=seed_value,
        )
        resolved_generator = "stratified"
    else:
        unit_points = _build_random_unit_points(
            count=desired_raw_count,
            dimensions=4,
            seed=seed_value,
        )
        resolved_generator = "random"

    existing_d4: set[tuple[float, ...]] = set()
    existing_d2: set[tuple[float, ...]] = set()
    if existing_points:
        for point in existing_points:
            if len(point) >= 4:
                existing_d4.add(
                    (
                        round(float(point[0]), 12),
                        round(float(point[1]), 12),
                        round(float(point[2]), 12),
                        round(float(point[3]), 12),
                    )
                )
            elif len(point) >= 2:
                existing_d2.add((round(float(point[0]), 12), round(float(point[1]), 12)))

    pool: list[dict[str, Any]] = []
    for unit_point in unit_points:
        unit_ka, unit_kb, unit_rp, unit_st = unit_point
        ka = _to_log_space(unit_ka, bounds["ka"])
        kb = _to_log_space(unit_kb, bounds["kb"])

        if not (bounds["ka"][0] <= ka <= bounds["ka"][1]) or not (bounds["kb"][0] <= kb <= bounds["kb"][1]):
            continue

        radp = _to_linear_space(unit_rp, bounds["radp"])
        shell_th = _to_linear_space(unit_st, bounds["shell_th"])
        if not (math.isfinite(radp) and math.isfinite(shell_th)):
            continue
        if is_dnn_causal_low_corner_excluded(ka, kb, radp, shell_th):
            continue

        d4_key = (round(ka, 12), round(kb, 12), round(radp, 12), round(shell_th, 12))
        d2_key = d4_key[:2]
        if d4_key in existing_d4 or d2_key in existing_d2:
            continue

        pool.append(
            {
                "candidate_id": f"pool-{len(pool) + 1:06d}",
                "ka": float(ka),
                "kb": float(kb),
                "radp": float(radp),
                "shell_th": float(shell_th),
                "b1": float(EMB_34UM_DNN_CAUSAL_B1_VALUE),
                "b2": float(EMB_34UM_DNN_CAUSAL_B2_VALUE),
                "a3": float(EMB_34UM_DNN_CAUSAL_A3_VALUE),
                "a4": float(EMB_34UM_DNN_CAUSAL_A4_VALUE),
                "runtime_fingerprint": {
                    "radp": float(radp),
                    "shell_th": float(shell_th),
                    "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                },
                "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            }
        )
        existing_d4.add(d4_key)
        if len(pool) >= requested_size:
            break

    if len(pool) < requested_size:
        raise ValueError(
            f"Unable to generate requested D4 candidate pool size of {requested_size}; generated {len(pool)}."
        )

    metadata = _build_candidate_pool_metadata(
        requested_size=requested_size,
        actual_size=len(pool),
        seed=seed_value,
        generator_type=resolved_generator,
        bounds=bounds,
        dimension_names=EMB_34UM_DNN_D4_ACTIVE_DIMENSION_NAMES,
    )
    return tuple(pool), metadata


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


def _coerce_candidate_space(value: str) -> str:
    if value not in {"d2", "d4"}:
        raise ValueError("candidate_space must be 'd2' or 'd4'.")
    return value


def _coerce_acquisition_score_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in EMB_34UM_DNN_ACQUISITION_SCORE_MODES:
        raise ValueError(
            "acquisition_score_mode must be one of "
            + ", ".join(repr(item) for item in EMB_34UM_DNN_ACQUISITION_SCORE_MODES)
            + "."
        )
    return normalized


def _coerce_parameter_value(
    item: Mapping[str, Any],
    key: str,
    *,
    index: int,
    label: str,
) -> float:
    value = None
    parameters = item.get("parameters")
    if isinstance(parameters, Mapping) and key in parameters:
        value = parameters.get(key)
    if value is None:
        value = item.get(key)
    if value is None:
        raise ValueError(f"candidate_records[{index}] is missing {label!r}.")
    return _coerce_positive_float(value, label=f"candidate_records[{index}].{key}")


def _coerce_bounded_positive(
    value: object,
    *,
    low: float,
    high: float,
    label: str,
) -> float:
    number = _coerce_positive_float(value, label=label)
    if not (low <= number <= high):
        raise ValueError(f"{label} must be within [{low}, {high}].")
    return number


def _normalize_scalar(value: float, *, low: float, high: float) -> float:
    width = high - low
    if math.isclose(width, 0.0):
        return 0.0
    return (value - low) / width


def _raw_candidate_point(item: Mapping[str, Any], *, index: int, candidate_space: str) -> tuple[float, ...]:
    ka = _coerce_parameter_value(item, "ka", index=index, label="candidate")
    kb = _coerce_parameter_value(item, "kb", index=index, label="candidate")
    if candidate_space == "d4":
        radp = _coerce_bounded_positive(
            _coerce_parameter_value(item, "radp", index=index, label="candidate"),
            low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][0],
            high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][1],
            label=f"candidate_records[{index}].radp",
        )
        shell_th = _coerce_bounded_positive(
            _coerce_parameter_value(item, "shell_th", index=index, label="candidate"),
            low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][0],
            high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][1],
            label=f"candidate_records[{index}].shell_th",
        )
        return (ka, kb, radp, shell_th)
    return (ka, kb)


def _normalize_candidate_point(
    point: tuple[float, ...],
    *,
    candidate_space: str,
) -> tuple[float, ...]:
    ka = float(point[0])
    kb = float(point[1])
    normalized: list[float] = [math.log10(ka), math.log10(kb)]
    if candidate_space == "d4":
        normalized.extend(
            (
                _normalize_scalar(
                    point[2],
                    low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][0],
                    high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][1],
                ),
                _normalize_scalar(
                    point[3],
                    low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][0],
                    high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][1],
                ),
            )
        )
    return tuple(normalized)


def _coerce_candidate_point(
    item: Mapping[str, Any],
    *,
    index: int,
    candidate_space: str = "d2",
) -> tuple[float, ...]:
    return _normalize_candidate_point(
        _raw_candidate_point(item, index=index, candidate_space=candidate_space),
        candidate_space=candidate_space,
    )


def _normalize_unit(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = float(min(values))
    maximum = float(max(values))
    width = maximum - minimum
    if math.isclose(width, 0.0):
        return [0.0 for _ in values]
    return [float((value - minimum) / width) for value in values]


def _normalize_log10_distance(distance: float, *, dimensions: int) -> float:
    if dimensions <= 0:
        return 0.0
    return float(distance / max(1.0, math.sqrt(float(dimensions))))


def _coerce_log10_point(item: object, *, label: str, candidate_space: str = "d2") -> tuple[float, ...]:
    candidate_space = _coerce_candidate_space(candidate_space)
    if isinstance(item, Mapping):
        return _coerce_candidate_point(
            item,
            index=0,
            candidate_space=candidate_space,
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        if candidate_space == "d2" and len(item) == 2:
            ka = _coerce_positive_float(item[0], label=f"{label}[0]")
            kb = _coerce_positive_float(item[1], label=f"{label}[1]")
            return (math.log10(ka), math.log10(kb))
        if candidate_space == "d4" and len(item) == 4:
            ka = _coerce_positive_float(item[0], label=f"{label}[0]")
            kb = _coerce_positive_float(item[1], label=f"{label}[1]")
            radp = _coerce_bounded_positive(
                item[2],
                low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][0],
                high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][1],
                label=f"{label}[2]",
            )
            shell_th = _coerce_bounded_positive(
                item[3],
                low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][0],
                high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][1],
                label=f"{label}[3]",
            )
            return _normalize_candidate_point((ka, kb, radp, shell_th), candidate_space="d4")
    raise ValueError(
        f"{label} must be a mapping with ka/kb"
        + ("" if candidate_space == "d2" else " and radp/shell_th")
        + (", or a 2-item numeric sequence" if candidate_space == "d2" else ", or a 4-item numeric sequence")
        + "."
    )


def _point_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.dist(left, right)


def _coerce_force_grid(values: object) -> tuple[float, ...]:
    raw = _coerce_sequence(values, label="force_grid")
    parsed = [_coerce_float(item, label="force_grid") for item in raw]
    if not parsed:
        raise ValueError("force_grid must not be empty.")
    return tuple(float(v) for v in parsed)


def _predict_candidate_uncertainty(
    fit: Emb34umDnnSurrogateFit,
    candidate: Mapping[str, Any],
    force_grid: Sequence[float],
    *,
    candidate_space: str,
) -> tuple[tuple[float, ...], float, float, float, float]:
    ka = _coerce_positive_float(candidate.get("ka", candidate.get("Yt", 1.0)), label="candidate.ka")
    kb = _coerce_positive_float(candidate.get("kb", candidate.get("kb_scale", 1.0)), label="candidate.kb")
    radp = None
    shell_th = None
    if candidate_space == "d4":
        radp = _coerce_bounded_positive(
            _coerce_parameter_value(candidate, "radp", index=0, label="candidate"),
            low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][0],
            high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["radp"][1],
            label="candidate_records[0].radp",
        )
        shell_th = _coerce_bounded_positive(
            _coerce_parameter_value(candidate, "shell_th", index=0, label="candidate"),
            low=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][0],
            high=EMB_34UM_DNN_D4_ACTIVE_BOUNDS["shell_th"][1],
            label="candidate_records[0].shell_th",
        )
    mean_curve, member_curves = load_ensemble_member_predictions(
        fit,
        force_grid,
        ka=ka,
        kb=kb,
        radp=radp,
        shell_th=shell_th,
    )
    if not member_curves:
        return (), 0.0, 0.0, 0.0, 0.0
    stacked = np.array(member_curves, dtype=float)
    per_force_std = np.std(stacked, axis=0)
    disagreement = float(np.mean(per_force_std))
    curve_uncertainty_l2 = float(np.linalg.norm(per_force_std) / math.sqrt(float(len(per_force_std))))
    mean_curve_array = np.asarray(mean_curve, dtype=float)
    mean_curve_l2 = float(np.linalg.norm(mean_curve_array) / math.sqrt(float(len(mean_curve_array))))
    predicted_curve_relative_l2_error = float(curve_uncertainty_l2 / max(mean_curve_l2, 1.0e-12))
    return (
        mean_curve,
        disagreement,
        predicted_curve_relative_l2_error,
        curve_uncertainty_l2,
        mean_curve_l2,
    )


def _coerce_count(value: object, *, label: str) -> int:
    number = _coerce_positive_int(value, label=label)
    return number


def score_candidate_pool(
    fit: Emb34umDnnSurrogateFit,
    candidate_records: Sequence[Mapping[str, Any]],
    force_grid: Sequence[float],
    *,
    existing_points: Sequence[tuple[float, ...]] | None = None,
    candidate_space: str = "d2",
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
    acquisition_score_mode: str = "disagreement",
) -> tuple[Mapping[str, Any], ...]:
    """Score candidates by model uncertainty plus optional nearest-point diversity term.

    Scoring fields:
    - ensemble_disagreement: mean std. across members
    - predicted_curve_relative_l2_error: full-curve relative uncertainty proxy
    - diversity_term: min distance to existing points in log10 space
    - acquisition_score: chosen base score + diversity_weight * diversity_term
    """

    rows = _coerce_sequence(candidate_records, label="candidate_records")
    if not rows:
        return ()
    force_axis = _coerce_force_grid(force_grid)
    if not fit.ensemble_checkpoints:
        raise ValueError("fit must include ensemble_checkpoints.")
    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative.")
    candidate_space = _coerce_candidate_space(candidate_space)
    acquisition_score_mode = _coerce_acquisition_score_mode(acquisition_score_mode)

    existing = [
        _coerce_log10_point(
            item,
            label=f"existing_points[{index}]",
            candidate_space=candidate_space,
        )
        for index, item in enumerate(_coerce_sequence(existing_points or (), label="existing_points"), start=1)
    ]

    scored: list[dict[str, Any]] = []
    disagreements: list[float] = []
    curve_error_risks: list[float] = []
    raw_diversities: list[float] = []
    for index, raw_row in enumerate(rows, start=1):
        row = _coerce_mapping(raw_row, label=f"candidate_records[{index}]")
        (
            mean_curve,
            disagreement,
            predicted_curve_relative_l2_error,
            curve_uncertainty_l2,
            mean_curve_l2,
        ) = _predict_candidate_uncertainty(
            fit,
            row,
            force_axis,
            candidate_space=candidate_space,
        )
        point = _coerce_log10_point(
            row,
            label=f"candidate_records[{index}]",
            candidate_space=candidate_space,
        )

        if existing:
            distance = min(_point_distance(point, candidate_point) for candidate_point in existing)
        else:
            distance = 0.0

        payload = dict(row)
        payload["candidate_index"] = index
        payload["candidate_log10_point"] = point
        payload["predicted_curve"] = tuple(float(item) for item in mean_curve)
        payload["ensemble_disagreement"] = float(disagreement)
        payload["predicted_curve_relative_l2_error"] = float(predicted_curve_relative_l2_error)
        payload["curve_uncertainty_l2"] = float(curve_uncertainty_l2)
        payload["predicted_curve_l2"] = float(mean_curve_l2)
        payload["diversity_term"] = float(distance)
        scored.append(payload)
        disagreements.append(float(disagreement))
        curve_error_risks.append(float(predicted_curve_relative_l2_error))
        raw_diversities.append(float(distance))

    disagreement_norm = _normalize_unit(disagreements)
    curve_error_risk_norm = _normalize_unit(curve_error_risks)
    diversity_norm = _normalize_unit(raw_diversities)
    for payload, disagreement_value, normalized_disagreement, curve_error_value, normalized_curve_error, raw_diversity, normalized_diversity in zip(
        scored,
        disagreements,
        disagreement_norm,
        curve_error_risks,
        curve_error_risk_norm,
        raw_diversities,
        diversity_norm,
    ):
        if candidate_space == "d4":
            payload["ensemble_disagreement_norm"] = float(normalized_disagreement)
            payload["predicted_curve_relative_l2_error_norm"] = float(normalized_curve_error)
            diversity_value = float(normalized_diversity)
        else:
            payload["ensemble_disagreement_norm"] = float(disagreement_value)
            payload["predicted_curve_relative_l2_error_norm"] = float(normalized_curve_error)
            diversity_value = float(raw_diversity)
        payload["diversity_term"] = diversity_value
        if acquisition_score_mode == "curve_error":
            base_score = (
                float(normalized_curve_error)
                if candidate_space == "d4"
                else float(curve_error_value)
            )
            base_field = "predicted_curve_relative_l2_error"
        else:
            base_score = (
                float(normalized_disagreement)
                if candidate_space == "d4"
                else float(disagreement_value)
            )
            base_field = "ensemble_disagreement"
        payload["acquisition_score_mode"] = acquisition_score_mode
        payload["acquisition_base_field"] = base_field
        payload["acquisition_base_score"] = float(base_score)
        payload["acquisition_score"] = float(base_score + float(diversity_weight) * diversity_value)

    scored.sort(
        key=lambda item: (
            -float(item["acquisition_score"]),
            -float(item.get("acquisition_base_score", 0.0)),
            float(item["candidate_log10_point"][0]),
            float(item["candidate_log10_point"][1]),
        )
    )
    return tuple(scored)


def select_greedy_diversity(
    scored_candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    existing_points: Sequence[tuple[float, ...]] | None = None,
    candidate_space: str = "d2",
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
    acquisition_score_mode: str = "disagreement",
) -> tuple[Mapping[str, Any], ...]:
    """Greedy diversity selection over scored candidates."""

    rows = _coerce_sequence(scored_candidates, label="scored_candidates")
    target = _coerce_count(count, label="count")
    if target <= 0 or not rows:
        return ()

    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative.")

    candidate_space = _coerce_candidate_space(candidate_space)
    acquisition_score_mode = _coerce_acquisition_score_mode(acquisition_score_mode)
    remaining: list[dict[str, Any]] = [dict(item) for item in rows]
    selected_points: list[tuple[float, ...]] = [
        _coerce_log10_point(
            point,
            label=f"existing_points[{index}]",
            candidate_space=candidate_space,
        )
        for index, point in enumerate(existing_points or (), start=1)
    ]
    selected_point_dimensions = 2 if candidate_space == "d2" else 4

    selected: list[dict[str, Any]] = []
    for _ in range(target):
        if not remaining:
            break

        best_index = -1
        best_value = (-math.inf, -math.inf, -math.inf, math.inf, math.inf)
        for position, row in enumerate(remaining):
            raw_point = row.get("candidate_log10_point")
            if isinstance(raw_point, Sequence) and not isinstance(raw_point, (str, bytes, bytearray)):
                point = tuple(float(value) for value in raw_point)
            else:
                point = ()  # pragma: no cover - defensive
            if len(point) != selected_point_dimensions:
                point = tuple(0.0 for _ in range(selected_point_dimensions))

            if selected_points:
                comparable_picks = [pick for pick in selected_points if len(pick) == len(point)]
                if candidate_space == "d4":
                    nearest = min(
                        _normalize_log10_distance(
                            _point_distance(point, pick),
                            dimensions=selected_point_dimensions,
                        )
                        for pick in comparable_picks
                    )
                elif comparable_picks:
                    nearest = min(_point_distance(point, pick) for pick in comparable_picks)
                else:
                    nearest = 0.0
                nearest = float(nearest) if comparable_picks else 0.0
            else:
                nearest = 0.0

            if acquisition_score_mode == "curve_error":
                disagreement = float(
                    row.get(
                        "predicted_curve_relative_l2_error_norm",
                        row.get("predicted_curve_relative_l2_error", 0.0),
                    )
                )
                base_field = "predicted_curve_relative_l2_error"
            elif candidate_space == "d4":
                disagreement = float(row.get("ensemble_disagreement_norm", row.get("ensemble_disagreement", 0.0)))
                base_field = "ensemble_disagreement"
            else:
                disagreement = float(row.get("ensemble_disagreement", 0.0))
                base_field = "ensemble_disagreement"

            score = disagreement + float(diversity_weight) * float(nearest)

            candidate_sort = (
                score,
                disagreement,
                nearest,
                -point[0],
                -point[1],
            )
            if candidate_sort > best_value:
                best_index = position
                best_value = candidate_sort
                row["diversity_term"] = nearest
                row["acquisition_score_mode"] = acquisition_score_mode
                row["acquisition_base_field"] = base_field
                row["acquisition_base_score"] = disagreement
                row["acquisition_score"] = score

        if best_index < 0:
            break

        selected_row = remaining.pop(best_index)
        selected_point = selected_row.get("candidate_log10_point")
        if isinstance(selected_point, Sequence) and not isinstance(selected_point, (str, bytes, bytearray)):
            selected_points.append(tuple(float(item) for item in selected_point))
        selected.append(selected_row)

    return tuple(selected)


def select_candidates(
    scored_candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    existing_points: Sequence[tuple[float, ...]] | None = None,
    candidate_space: str = "d2",
    diversity_weight: float = EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
    acquisition_score_mode: str = "disagreement",
) -> tuple[Mapping[str, Any], ...]:
    return select_greedy_diversity(
        scored_candidates,
        count=count,
        existing_points=existing_points,
        candidate_space=candidate_space,
        diversity_weight=diversity_weight,
        acquisition_score_mode=acquisition_score_mode,
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
        curve_error_values: tuple[float, ...] = ()
    else:
        score_values = tuple(float(item.get("acquisition_score", 0.0)) for item in scored_candidates)
        curve_error_values = tuple(
            float(item.get("predicted_curve_relative_l2_error", 0.0))
            for item in scored_candidates
        )
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
        "predicted_curve_relative_l2_error": {
            "min": min(curve_error_values) if curve_error_values else 0.0,
            "max": max(curve_error_values) if curve_error_values else 0.0,
            "mean": statistics.mean(curve_error_values) if curve_error_values else 0.0,
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
    "EMB_34UM_DNN_ACQUISITION_SCORE_MODES",
    "build_d4_candidate_pool",
    "score_candidate_pool",
    "select_greedy_diversity",
    "select_candidates",
    "build_candidate_selection_report",
    "write_candidate_selection_artifacts",
    "compute_disagreement_distribution",
]
