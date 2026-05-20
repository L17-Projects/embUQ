from __future__ import annotations

"""Standalone EMB 3.4um AL candidate design and batch selection logic."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import random

import numpy as np

from meso_uq.active_learning.contracts import Candidate

EMB_34UM_FINAL_GATE_DESIGN_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_final_gate_design.v1"

EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES = ("ka", "kb")
EMB_34UM_FINAL_GATE_BOUNDS = {
    "ka": (1e2, 6e5),
    "kb": (400.0, 70000.0),
}
EMB_34UM_FINAL_GATE_LOG_SPACE = True
EMB_34UM_FINAL_GATE_FAMILY = "emb"
EMB_34UM_FINAL_GATE_EXPERIMENT = "indentation"

EMB_34UM_FINAL_GATE_POOL_SIZE = 100
EMB_34UM_FINAL_GATE_BATCH_SIZE = 30
EMB_34UM_FINAL_GATE_EXPLORATION_COUNT = 6
EMB_34UM_FINAL_GATE_ACQUISITION_COUNT = 24

EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES = (30, 60, 90)
EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET = 10_000

EMB_34UM_FINAL_GATE_SOURCE_INITIAL = "initial_sobol_maximin"
EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION = "exploration"
EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION = "ensemble_disagreement_diversity"
EMB_34UM_FINAL_GATE_SOURCE_VALIDATION = "validation_sobol_maximin"


def _coerce_seed(value: int, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _coerce_count(value: int, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _coerce_positive_int(value: int, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _coerce_existing_points(
    raw_points: Sequence[Any] | Mapping[str, Any] | None,
) -> tuple[tuple[float, float], ...]:
    if raw_points is None:
        return ()
    if isinstance(raw_points, Mapping):
        raw_points = tuple(raw_points.values())
    normalized: list[tuple[float, float]] = []
    for item in raw_points:
        if isinstance(item, Candidate):
            params = item.parameters
            normalized.append((float(params["ka"]), float(params["kb"])))
            continue
        if isinstance(item, Mapping):
            if "ka" not in item or "kb" not in item:
                raise ValueError("existing_points mapping entries must include 'ka' and 'kb'.")
            normalized.append((float(item["ka"]), float(item["kb"])))
            continue
        if isinstance(item, Sequence) and len(item) == 2:
            normalized.append((float(item[0]), float(item[1])))
            continue
        raise ValueError("existing_points entries must be Candidate, mapping, or (ka, kb) sequence.")
    return tuple(normalized)


def _coerce_and_validate_points(raw_points: Sequence[Any] | None) -> tuple[tuple[float, float], ...]:
    points = _coerce_existing_points(raw_points)
    validated: list[tuple[float, float]] = []
    ka_min, ka_max = EMB_34UM_FINAL_GATE_BOUNDS["ka"]
    kb_min, kb_max = EMB_34UM_FINAL_GATE_BOUNDS["kb"]
    for ka, kb in points:
        if not math.isfinite(ka) or not math.isfinite(kb):
            raise ValueError("existing_points must contain finite ka/kb values.")
        if not (ka_min <= ka <= ka_max):
            raise ValueError(f"existing_points ka={ka!r} outside bounds {ka_min}-{ka_max}")
        if not (kb_min <= kb <= kb_max):
            raise ValueError(f"existing_points kb={kb!r} outside bounds {kb_min}-{kb_max}")
        validated.append((ka, kb))
    return tuple(validated)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1
    current = int(index)
    while current:
        current, remainder = divmod(current, base)
        denominator *= base
        value += remainder / denominator
    return value


def _build_sobol_like_sequence(count: int, seed: int) -> tuple[tuple[float, float], ...]:
    if count <= 0:
        return ()
    rng = random.Random(seed)
    shifts = (rng.random(), rng.random())
    primes = (2, 3)
    points: list[tuple[float, float]] = []
    for index in range(count):
        sample_x = (_van_der_corput(index + 1, primes[0]) + shifts[0]) % 1.0
        sample_y = (_van_der_corput(index + 1, primes[1]) + shifts[1]) % 1.0
        points.append((sample_x, sample_y))
    return tuple(points)


def _to_log_space(point: tuple[float, float]) -> tuple[float, float]:
    return (math.log10(point[0]), math.log10(point[1]))


def _from_unit_bounds(unit: float, *, bounds: tuple[float, float], use_log_space: bool) -> float:
    lower, upper = bounds
    if use_log_space:
        return 10.0 ** (math.log10(lower) + unit * (math.log10(upper) - math.log10(lower)))
    return lower + unit * (upper - lower)


def _scale_unit_points(
    unit_points: Sequence[tuple[float, float]],
    *,
    use_log_space: bool,
) -> tuple[tuple[float, float], ...]:
    ka_bounds = EMB_34UM_FINAL_GATE_BOUNDS["ka"]
    kb_bounds = EMB_34UM_FINAL_GATE_BOUNDS["kb"]
    return tuple(
        (
            _from_unit_bounds(point[0], bounds=ka_bounds, use_log_space=use_log_space),
            _from_unit_bounds(point[1], bounds=kb_bounds, use_log_space=use_log_space),
        )
        for point in unit_points
    )


def _distance(lhs: tuple[float, float], rhs: tuple[float, float]) -> float:
    return math.dist(lhs, rhs)


def _select_maximin(
    *,
    unit_candidates: Sequence[tuple[float, float]],
    count: int,
    reference: Sequence[tuple[float, float]] | None,
    seed: int,
) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return ()
    if count > len(unit_candidates):
        raise ValueError("candidate_pool_size must be at least batch_size.")

    remaining = set(range(len(unit_candidates)))
    selected: list[int] = []
    selected_vectors = list(reference or ())

    if not selected_vectors:
        center = (0.5, 0.5)
        first = min(remaining, key=lambda index: (_distance(unit_candidates[index], center), index))
        selected.append(first)
        selected_vectors.append(unit_candidates[first])
        remaining.remove(first)

    while len(selected) < count and remaining:
        best_index = -1
        best_score = -1.0
        for index in sorted(remaining):
            candidate = unit_candidates[index]
            nearest = min(_distance(candidate, selected_point) for selected_point in selected_vectors)
            if nearest > best_score:
                best_score = nearest
                best_index = index
            elif nearest == best_score:
                # deterministic tie-breaker keeps results stable under identical ties
                if index < best_index or best_index < 0:
                    best_index = index
        selected.append(best_index)
        selected_vectors.append(unit_candidates[best_index])
        remaining.remove(best_index)

    return tuple(selected)


def _agreement_from_ensemble_predictions(predictions: Sequence[Any], pool_size: int) -> tuple[float, ...]:
    if predictions is None:
        return tuple(0.0 for _ in range(pool_size))
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 3:
        raise ValueError("ensemble_disagreement must be a 3D array-like: [n_points, n_ensembles, n_curve_points].")
    if values.shape[0] != pool_size:
        raise ValueError(f"ensemble_disagreement row count must equal candidate pool size {pool_size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ensemble_disagreement must contain finite values.")
    if values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("ensemble_disagreement must include at least one ensemble and one curve point.")
    # mean std across ensemble, then mean across curve points
    return tuple(float(np.mean(np.std(values, axis=1), axis=1)))


def _select_acquisition(
    *,
    unit_candidates: Sequence[tuple[float, float]],
    disagreement: Sequence[float],
    count: int,
    reference: Sequence[tuple[float, float]],
    seed: int,
    diversity_weight: float = 0.1,
) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return ()
    if len(unit_candidates) < count:
        raise ValueError("candidate_pool_size too small for acquisition selection.")
    if len(disagreement) != len(unit_candidates):
        raise ValueError("disagreement length must match candidate pool size.")

    remaining = set(range(len(unit_candidates)))
    selected: list[int] = []
    selected_vectors = list(reference)

    while len(selected) < count and remaining:
        best_index = -1
        best_score = -math.inf
        for index in sorted(remaining):
            candidate = unit_candidates[index]
            nearest = min(_distance(candidate, item) for item in selected_vectors)
            score = float(disagreement[index]) + diversity_weight * nearest
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        selected.append(best_index)
        selected_vectors.append(unit_candidates[best_index])
        remaining.remove(best_index)

    return tuple(selected)


def _build_candidate(*, run_id: str, round_index: int, order: int, source: str, ka: float, kb: float, prefix: str) -> Candidate:
    return Candidate(
        candidate_id=f"{run_id}-{prefix}-r{round_index:02d}-c{order:03d}",
        parameters={
            "family": EMB_34UM_FINAL_GATE_FAMILY,
            "experiment": EMB_34UM_FINAL_GATE_EXPERIMENT,
            "ka": ka,
            "kb": kb,
        },
        metadata={
            "source": source,
            "round": round_index,
            "order": order,
            "selection_source": source,
            "selection_policy": "sobol_maximin" if source == EMB_34UM_FINAL_GATE_SOURCE_INITIAL else "ensemble_disagreement_diversity",
        },
    )


def build_emb_34um_final_gate_design_round(
    *,
    run_id: str,
    round_index: int,
    seed: int,
    existing_points: Sequence[Any] | Mapping[str, Any] | None = None,
    ensemble_disagreement: Sequence[Any] | None = None,
    candidate_pool_size: int = EMB_34UM_FINAL_GATE_POOL_SIZE,
    batch_size: int = EMB_34UM_FINAL_GATE_BATCH_SIZE,
    exploration_count: int = EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
    acquisition_count: int = EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
    candidate_prefix: str = "emb-34um-final-gate",
    use_log_space: bool = EMB_34UM_FINAL_GATE_LOG_SPACE,
) -> "Emb34umFinalGateDesignResult":
    round_index = _coerce_positive_int(round_index, label="round_index")
    seed = _coerce_seed(seed, label="seed")
    candidate_pool_size = _coerce_count(candidate_pool_size, "candidate_pool_size")
    batch_size = _coerce_count(batch_size, "batch_size")
    exploration_count = _coerce_count(exploration_count, "exploration_count")
    acquisition_count = _coerce_count(acquisition_count, "acquisition_count")

    if candidate_pool_size < batch_size or batch_size == 0:
        raise ValueError("candidate_pool_size must be positive and at least batch_size.")

    if batch_size != EMB_34UM_FINAL_GATE_BATCH_SIZE:
        raise ValueError(f"batch_size must be {EMB_34UM_FINAL_GATE_BATCH_SIZE} for this design policy.")

    previous = _coerce_and_validate_points(existing_points)

    unit_pool = _build_sobol_like_sequence(candidate_pool_size, seed)
    physical_pool = _scale_unit_points(unit_pool, use_log_space=use_log_space)
    if use_log_space:
        selection_pool = tuple(_to_log_space(point) for point in physical_pool)
        reference = tuple(_to_log_space(point) for point in previous)
    else:
        selection_pool = physical_pool
        reference = previous

    if round_index == 1:
        selected_indices = _select_maximin(
            unit_candidates=selection_pool,
            count=batch_size,
            reference=None,
            seed=seed,
        )
        selected_sources = [EMB_34UM_FINAL_GATE_SOURCE_INITIAL] * batch_size
    else:
        if exploration_count + acquisition_count != batch_size:
            raise ValueError("exploration_count + acquisition_count must equal batch_size.")
        if not previous:
            previous = ()

        exploration_indices = _select_maximin(
            unit_candidates=selection_pool,
            count=exploration_count,
            reference=reference,
            seed=seed,
        )
        exploration_set = set(exploration_indices)

        acquisition_pool_indices = [index for index in range(len(selection_pool)) if index not in exploration_set]
        if len(acquisition_pool_indices) < acquisition_count:
            raise ValueError("candidate_pool_size is too small after exploration selection.")

        disagreement = _agreement_from_ensemble_predictions(
            predictions=ensemble_disagreement,
            pool_size=len(selection_pool),
        ) if ensemble_disagreement is not None else tuple(0.0 for _ in range(len(selection_pool)))

        acquisition_unit_candidates = tuple(selection_pool[index] for index in acquisition_pool_indices)
        acquisition_disagreement = tuple(disagreement[index] for index in acquisition_pool_indices)

        selected_reference: list[tuple[float, float]] = [
            point for point in reference
        ]
        for index in exploration_indices:
            selected_reference.append(selection_pool[index])

        acquisition_pick_local = _select_acquisition(
            unit_candidates=acquisition_unit_candidates,
            disagreement=acquisition_disagreement,
            count=acquisition_count,
            reference=selected_reference,
            seed=seed,
        )
        acquisition_indices = tuple(
            acquisition_pool_indices[local_index] for local_index in acquisition_pick_local
        )
        selected_indices = tuple(exploration_indices + acquisition_indices)
        selected_sources = ([EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION] * len(exploration_indices)) + ([
            EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION
        ] * len(acquisition_indices))

    if round_index != 1:
        if len(selected_indices) != batch_size:
            raise RuntimeError("selection logic produced unexpected number of candidates")

    selected_candidates: list[Candidate] = []
    for order, index in enumerate(selected_indices, start=1):
        ka, kb = physical_pool[index]
        selected_candidates.append(
            _build_candidate(
                run_id=run_id,
                round_index=round_index,
                order=order,
                source=selected_sources[order - 1],
                ka=ka,
                kb=kb,
                prefix=candidate_prefix,
            )
        )

    source_counts: dict[str, int] = {}
    for source in selected_sources:
        source_counts[source] = source_counts.get(source, 0) + 1

    pool_candidates = tuple(
        _build_candidate(
            run_id=run_id,
            round_index=round_index,
            order=index + 1,
            source="pool",
            ka=point[0],
            kb=point[1],
            prefix=f"{candidate_prefix}-pool",
        )
        for index, point in enumerate(physical_pool)
    )

    return Emb34umFinalGateDesignResult(
        candidates=tuple(selected_candidates),
        round_pool=pool_candidates,
        manifest={
            "schema_version": EMB_34UM_FINAL_GATE_DESIGN_SCHEMA_VERSION,
            "run_id": run_id,
            "round": round_index,
            "status": "ready",
            "active_variables": list(EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES),
            "parameter_bounds": {
                name: [float(lower), float(upper)] for name, (lower, upper) in EMB_34UM_FINAL_GATE_BOUNDS.items()
            },
            "use_log_space": bool(use_log_space),
            "selected_source_distribution": source_counts,
            "candidate_pool_size": candidate_pool_size,
            "batch_size": batch_size,
            "selection_seed": seed,
            "sources": list(source_counts),
            "lhs_comparator_prefixes": list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES),
            "selected_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "source": candidate.metadata["source"],
                    "round": candidate.metadata["round"],
                    "order": candidate.metadata["order"],
                    "ka": float(candidate.parameters["ka"]),
                    "kb": float(candidate.parameters["kb"]),
                }
                for candidate in selected_candidates
            ],
        },
    )


def build_emb_34um_final_gate_validation_design(
    *,
    run_id: str,
    seed: int,
    design_size: int = 90,
    candidate_prefix: str = "emb-34um-final-gate-validation",
    use_log_space: bool = EMB_34UM_FINAL_GATE_LOG_SPACE,
    seed_offset: int = EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET,
) -> "Emb34umFinalGateValidationResult":
    seed = _coerce_seed(seed, label="seed")
    design_size = _coerce_positive_int(design_size, label="design_size")
    validation_seed = seed + seed_offset

    unit_points = _build_sobol_like_sequence(design_size, validation_seed)
    physical_points = _scale_unit_points(unit_points, use_log_space=use_log_space)
    scaled_selection = tuple(_to_log_space(point) for point in physical_points) if use_log_space else physical_points
    selected_indices = _select_maximin(
        unit_candidates=scaled_selection,
        count=design_size,
        reference=None,
        seed=validation_seed,
    )

    candidates: list[Candidate] = []
    for order, source_index in enumerate(selected_indices, start=1):
        ka, kb = physical_points[source_index]
        candidates.append(
            _build_candidate(
                run_id=run_id,
                round_index=0,
                order=order,
                source=EMB_34UM_FINAL_GATE_SOURCE_VALIDATION,
                ka=ka,
                kb=kb,
                prefix=candidate_prefix,
            )
        )

    return Emb34umFinalGateValidationResult(
        candidates=tuple(candidates),
        manifest={
            "schema_version": EMB_34UM_FINAL_GATE_DESIGN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "ready",
            "design_type": "validation",
            "active_variables": list(EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES),
            "parameter_bounds": {
                name: [float(lower), float(upper)] for name, (lower, upper) in EMB_34UM_FINAL_GATE_BOUNDS.items()
            },
            "use_log_space": bool(use_log_space),
            "source": EMB_34UM_FINAL_GATE_SOURCE_VALIDATION,
            "selection_seed": validation_seed,
            "candidate_count": design_size,
            "lhs_comparator_prefixes": list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES),
            "selected_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "source": candidate.metadata["source"],
                    "ka": float(candidate.parameters["ka"]),
                    "kb": float(candidate.parameters["kb"]),
                }
                for candidate in candidates
            ],
        },
    )


@dataclass(frozen=True)
class Emb34umFinalGateDesignResult:
    candidates: tuple[Candidate, ...]
    round_pool: tuple[Candidate, ...]
    manifest: Mapping[str, Any]

    def selected_sources(self) -> tuple[str, ...]:
        return tuple(str(candidate.metadata["source"]) for candidate in self.candidates)


@dataclass(frozen=True)
class Emb34umFinalGateValidationResult:
    candidates: tuple[Candidate, ...]
    manifest: Mapping[str, Any]


__all__ = [
    "EMB_34UM_FINAL_GATE_ACTIVE_VARIABLES",
    "EMB_34UM_FINAL_GATE_ACQUISITION_COUNT",
    "EMB_34UM_FINAL_GATE_BATCH_SIZE",
    "EMB_34UM_FINAL_GATE_BOUNDS",
    "EMB_34UM_FINAL_GATE_EXPERIMENT",
    "EMB_34UM_FINAL_GATE_DESIGN_SCHEMA_VERSION",
    "EMB_34UM_FINAL_GATE_EXPLORATION_COUNT",
    "EMB_34UM_FINAL_GATE_FAMILY",
    "EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES",
    "EMB_34UM_FINAL_GATE_LOG_SPACE",
    "EMB_34UM_FINAL_GATE_POOL_SIZE",
    "EMB_34UM_FINAL_GATE_SOURCE_INITIAL",
    "EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION",
    "EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION",
    "EMB_34UM_FINAL_GATE_SOURCE_VALIDATION",
    "EMB_34UM_FINAL_GATE_VALIDATION_SEED_OFFSET",
    "Emb34umFinalGateDesignResult",
    "Emb34umFinalGateValidationResult",
    "build_emb_34um_final_gate_design_round",
    "build_emb_34um_final_gate_validation_design",
]
