"""Lightweight, deterministic acquisition scoring and validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import struct
import zlib

from meso_uq.active_learning.contracts import (
    ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION,
    AcquisitionScore,
    Candidate,
)

ACTIVE_LEARNING_ACQUISITION_ARTIFACT_SCHEMA_VERSION = (
    "meso_uq.active_learning.acquisition_artifacts.v1"
)
ACQUISITION_MANIFEST_FILENAME = "acquisition_manifest.json"
ACQUISITION_REPORT_FILENAME = "acquisition_report.json"
ACQUISITION_PLOT_FILENAME = "acquisition_validation_plot.png"
ACQUISITION_PLOT_SIDECAR_FILENAME = "acquisition_validation_plot.png.json"

SUPPORTED_POLICIES = frozenset(
    {
        "uncertainty",
        "expected_improvement",
        "diversity",
        "weighted_sum",
        "weighted_multiobjective",
    }
)

UNCERTAINTY_FIELD_ALIASES = (
    "uncertainty",
    "pred_std",
    "prediction_std",
    "std",
    "acquisition_uncertainty",
)
EXPECTED_IMPROVEMENT_FIELD_ALIASES = (
    "expected_improvement",
    "acquisition_ei",
    "ei",
)
MEAN_FIELD_ALIASES = (
    "pred_mean",
    "prediction_mean",
    "mean",
)
STD_FIELD_ALIASES = (
    "pred_std",
    "prediction_std",
    "std",
)
BASELINE_FIELD_ALIASES = (
    "best_observed",
    "observed_best",
    "incumbent",
    "baseline",
)

_PATH_SENTINEL = object()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    iteration = int(value)
    if iteration < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return iteration


def _coerce_candidates(payload: Sequence[object]) -> tuple[Candidate, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("candidates must be a sequence.")
    normalized: list[Candidate] = []
    for item in payload:
        normalized.append(item if isinstance(item, Candidate) else Candidate.from_dict(item))
    return tuple(normalized)


def _coerce_policy(payload: object) -> str:
    policy = _coerce_nonempty_text(payload, field_name="policy").lower()
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"policy must be one of: {', '.join(sorted(SUPPORTED_POLICIES))}.")
    return policy


def _coerce_score(value: object, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite.")
    return number


def _coerce_paths(payload: Sequence[str | Sequence[object]] | None) -> tuple[tuple[str, ...], ...]:
    if payload is None:
        return ()
    normalized: list[tuple[str, ...]] = []
    for raw_path in payload:
        if isinstance(raw_path, str):
            parts = tuple(part.strip() for part in raw_path.split(".") if part.strip())
        else:
            parts = tuple(str(part).strip() for part in raw_path if str(part).strip())
        if not parts:
            raise ValueError("diversity_paths entries must contain at least one path component.")
        normalized.append(parts)
    if not normalized:
        raise ValueError("diversity_paths must contain at least one path.")
    return tuple(normalized)


def _coerce_weights(
    payload: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        items = tuple(payload.items())
    elif isinstance(payload, Sequence):
        items = tuple(payload)
    else:
        raise ValueError("weights must be a mapping or sequence of (name, weight) pairs.")
    if not items:
        raise ValueError("weights must contain at least one item.")
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in items:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("weight names must be non-empty.")
        normalized[name] = _coerce_score(raw_weight, context=f"weight for {name!r}")
    return normalized


def _coerce_reference_candidates(
    payload: Sequence[object] | None,
) -> tuple[Candidate, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("reference_candidates must be a sequence.")
    normalized = []
    for item in payload:
        normalized.append(item if isinstance(item, Candidate) else Candidate.from_dict(item))
    return tuple(normalized)


def _coerce_path(value: str | Sequence[object]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(".") if part.strip())
    else:
        parts = tuple(str(part).strip() for part in value if str(part).strip())
    if not parts:
        raise ValueError("lookup path must contain at least one component.")
    return parts


def _at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return _PATH_SENTINEL
        if key not in current:
            return _PATH_SENTINEL
        current = current[key]
    return current


def _flatten_numeric_payload(payload: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        path = prefix + (str(key),)
        if isinstance(value, Mapping):
            flat.update(_flatten_numeric_payload(value, path))
            continue
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                index_path = path + (str(index),)
                if isinstance(item, Mapping):
                    flat.update(_flatten_numeric_payload(item, index_path))
                    continue
                try:
                    flat[".".join(index_path)] = _coerce_score(
                        item,
                        context=f"value at {'.'.join(index_path)}",
                    )
                except ValueError:
                    continue
            continue
        if value is None or isinstance(value, (str, bytes, bytearray, bool)):
            continue
        try:
            flat[".".join(path)] = _coerce_score(value, context=f"value at {'.'.join(path)}")
        except ValueError:
            continue
    return flat


def _lookup_scalar(
    candidate: Candidate,
    payloads: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> tuple[bool, float | None]:
    flat_payloads = tuple(_flatten_numeric_payload(payload) for payload in payloads)
    for alias in aliases:
        path = _coerce_path(alias)
        dotted = ".".join(path)
        for payload in payloads:
            value = _at_path(payload, path)
            if value is not _PATH_SENTINEL:
                return True, _coerce_score(
                    value,
                    context=f"candidate[{candidate.candidate_id}] field {alias!r}",
                )
        for flat_payload in flat_payloads:
            if alias in flat_payload:
                return True, float(flat_payload[alias])
            if dotted in flat_payload:
                return True, float(flat_payload[dotted])
    return False, None


def _candidate_family_and_payload(candidate: Candidate) -> tuple[str, Mapping[str, Any]]:
    raw_family = candidate.parameters.get("family", candidate.parameters.get("dpd_family"))
    if raw_family is not None:
        family = str(raw_family).strip().lower() or "unknown"
    else:
        family = "unknown"

    normalized_family = family.lower()
    if normalized_family in {"gv", "emb"}:
        payload = candidate.parameters.get(f"{normalized_family}_launch")
        if isinstance(payload, Mapping):
            return normalized_family, payload

    for fallback in ("gv", "emb"):
        payload = candidate.parameters.get(f"{fallback}_launch")
        if isinstance(payload, Mapping):
            return fallback, payload

    return family, candidate.parameters


def _candidate_score_payloads(candidate: Candidate) -> tuple[Mapping[str, Any], ...]:
    family, family_payload = _candidate_family_and_payload(candidate)
    if family_payload is candidate.parameters:
        return (candidate.parameters, candidate.metadata)
    return (family_payload, candidate.parameters, candidate.metadata)


def _candidate_feature_vector(
    candidate: Candidate,
    *,
    paths: tuple[tuple[str, ...], ...],
    payloads: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if paths:
        vector: list[float] = []
        missing_paths: list[str] = []
        for path in paths:
            found = False
            for payload in payloads:
                value = _at_path(payload, path)
                if value is _PATH_SENTINEL:
                    continue
                try:
                    vector.append(_coerce_score(value, context=f"vector value for {'.'.join(path)}"))
                    found = True
                    break
                except ValueError:
                    continue
            if not found:
                missing_paths.append(".".join(path))
        return tuple(vector), tuple(missing_paths)

    flat: dict[str, float] = {}
    for payload in payloads:
        flat.update(_flatten_numeric_payload(payload))
    return tuple(flat[key] for key in sorted(flat)), ()


def _euclidean_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    max_len = max(len(lhs), len(rhs))
    if max_len == 0:
        return 0.0
    total = 0.0
    for index in range(max_len):
        left = float(lhs[index]) if index < len(lhs) else 0.0
        right = float(rhs[index]) if index < len(rhs) else 0.0
        total += (left - right) ** 2
    return math.sqrt(total)


@dataclass(frozen=True)
class AcquisitionScoringConfig:
    policy: str
    weights: Mapping[str, float] = field(default_factory=dict)
    expected_improvement_baseline: float | None = None
    diversity_paths: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    reference_candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    schema_version: str = ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _coerce_policy(self.policy))
        object.__setattr__(self, "weights", _coerce_weights(self.weights))
        if self.expected_improvement_baseline is not None:
            object.__setattr__(
                self,
                "expected_improvement_baseline",
                _coerce_score(
                    self.expected_improvement_baseline,
                    context="expected_improvement_baseline",
                ),
            )
        object.__setattr__(self, "diversity_paths", _coerce_paths(self.diversity_paths))
        object.__setattr__(
            self,
            "reference_candidates",
            _coerce_reference_candidates(self.reference_candidates),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "weights": dict(self.weights),
            "expected_improvement_baseline": self.expected_improvement_baseline,
            "diversity_paths": [".".join(path) for path in self.diversity_paths],
            "reference_candidates": len(self.reference_candidates),
            "schema_version": self.schema_version,
        }


def _score_uncertainty(candidate: Candidate, payloads: tuple[Mapping[str, Any], ...], family: str) -> tuple[bool, float, dict[str, Any] | None]:
    found, value = _lookup_scalar(candidate, payloads, UNCERTAINTY_FIELD_ALIASES)
    if not found or value is None:
        return False, 0.0, {"missing_field": "uncertainty"}
    return (
        True,
        value,
        {"policy": "uncertainty", "family": family, "candidate_hash": candidate.stable_hash()},
    )


def _score_expected_improvement(candidate: Candidate, payloads: tuple[Mapping[str, Any], ...], family: str, config: AcquisitionScoringConfig) -> tuple[bool, float, dict[str, Any] | None]:
    found, value = _lookup_scalar(candidate, payloads, EXPECTED_IMPROVEMENT_FIELD_ALIASES)
    if found and value is not None:
        return (
            True,
            value,
            {"policy": "expected_improvement", "family": family, "candidate_hash": candidate.stable_hash()},
        )

    found_mean, mean = _lookup_scalar(candidate, payloads, MEAN_FIELD_ALIASES)
    found_spread, spread = _lookup_scalar(candidate, payloads, STD_FIELD_ALIASES)
    if not found_mean or not found_spread or mean is None or spread is None:
        return False, 0.0, {"missing_field": "pred_mean or pred_std"}

    baseline = (
        config.expected_improvement_baseline
        if config.expected_improvement_baseline is not None
        else 0.0
    )
    if config.expected_improvement_baseline is None:
        found_base, base_value = _lookup_scalar(candidate, payloads, BASELINE_FIELD_ALIASES)
        if found_base and base_value is not None:
            baseline = base_value
    score = max(mean - baseline, 0.0) * spread
    return (
        True,
        score,
        {
            "policy": "expected_improvement",
            "family": family,
            "candidate_hash": candidate.stable_hash(),
            "baseline": baseline,
        },
    )


def _score_weighted(candidate: Candidate, payloads: tuple[Mapping[str, Any], ...], family: str, config: AcquisitionScoringConfig) -> tuple[bool, float, dict[str, Any] | None]:
    if not config.weights:
        raise ValueError("weighted_sum and weighted_multiobjective require non-empty weights.")

    components: dict[str, float] = {}
    total = 0.0
    for name, weight in config.weights.items():
        found, value = _lookup_scalar(candidate, payloads, (name,))
        if not found or value is None:
            return False, 0.0, {"missing_field": name}
        component = value * weight
        components[name] = component
        total += component

    return (
        True,
        total,
        {
            "policy": config.policy,
            "family": family,
            "candidate_hash": candidate.stable_hash(),
            "components": components,
        },
    )


def _score_diversity(
    candidate: Candidate,
    *,
    candidate_index: int,
    candidate_vector: tuple[float, ...],
    all_vectors: Sequence[tuple[int, str, str, tuple[float, ...]]],
) -> tuple[bool, float, dict[str, Any] | None]:
    if not all_vectors:
        return False, 0.0, {"missing_field": "diversity_vectors"}
    distances: list[float] = []
    for index, _candidate_id, _candidate_hash, other_vector in all_vectors:
        if index == candidate_index:
            continue
        distances.append(_euclidean_distance(candidate_vector, other_vector))
    if not distances:
        return True, 0.0, {
            "policy": "diversity",
            "family": _candidate_family_and_payload(candidate)[0],
            "candidate_hash": candidate.stable_hash(),
            "feature_count": len(candidate_vector),
            "reference_count": max(0, len(all_vectors) - 1),
        }
    return (
        True,
        min(distances),
        {
            "policy": "diversity",
            "family": _candidate_family_and_payload(candidate)[0],
            "candidate_hash": candidate.stable_hash(),
            "feature_count": len(candidate_vector),
            "min_distance": min(distances),
            "max_distance": max(distances),
            "reference_count": max(0, len(all_vectors) - 1),
        },
    )


def _compute_score(
    candidate: Candidate,
    payloads: tuple[Mapping[str, Any], ...],
    config: AcquisitionScoringConfig,
    family: str,
    *,
    candidate_index: int,
    candidate_vector: tuple[float, ...],
    all_vectors: Sequence[tuple[int, str, str, tuple[float, ...]]] | None = None,
) -> tuple[bool, float, Mapping[str, Any] | None]:
    if config.policy == "uncertainty":
        return _score_uncertainty(candidate, payloads, family)
    if config.policy == "expected_improvement":
        return _score_expected_improvement(candidate, payloads, family, config)
    if config.policy in {"weighted_sum", "weighted_multiobjective"}:
        return _score_weighted(candidate, payloads, family, config)
    if config.policy == "diversity":
        if all_vectors is None:
            raise ValueError("diversity policy requires feature vectors.")
        return _score_diversity(
            candidate,
            candidate_index=candidate_index,
            candidate_vector=candidate_vector,
            all_vectors=all_vectors,
        )
    raise ValueError(f"Unsupported policy: {config.policy}")


def _rank_acquisition_scores(
    candidates: tuple[tuple[Candidate, AcquisitionScore], ...],
) -> tuple[AcquisitionScore, ...]:
    decorated = []
    for index, (candidate, score) in enumerate(candidates):
        decorated.append(
            (
                -score.score,
                score.candidate_id,
                candidate.stable_hash(),
                index,
                score,
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return tuple(item[4] for item in decorated)


@dataclass(frozen=True)
class AcquisitionScoringResult:
    config: AcquisitionScoringConfig
    scores: tuple[AcquisitionScore, ...]
    ranked_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    rejection_reasons: Mapping[str, tuple[str, ...]]
    family_distribution: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION,
            "policy": self.config.policy,
            "weights": dict(self.config.weights),
            "candidate_count": sum(self.family_distribution.values()),
            "scored_count": len(self.scores),
            "rejected_count": len(self.rejected_candidate_ids),
            "family_distribution": dict(self.family_distribution),
            "scores": [score.as_dict() for score in self.scores],
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "rejection_reasons": {key: list(value) for key, value in self.rejection_reasons.items()},
        }


def score_acquisition_candidates(
    candidates: Sequence[object],
    *,
    policy: str,
    weights: Mapping[str, float] | Sequence[tuple[str, float]] | None = None,
    expected_improvement_baseline: float | None = None,
    diversity_paths: Sequence[str | Sequence[object]] | None = None,
    reference_candidates: Sequence[object] | None = None,
) -> AcquisitionScoringResult:
    normalized_candidates = _coerce_candidates(candidates)
    config = AcquisitionScoringConfig(
        policy=policy,
        weights=weights,
        expected_improvement_baseline=expected_improvement_baseline,
        diversity_paths=diversity_paths,
        reference_candidates=reference_candidates,
    )

    if not normalized_candidates:
        return AcquisitionScoringResult(
            config=config,
            scores=(),
            ranked_candidate_ids=(),
            rejected_candidate_ids=(),
            rejection_reasons={},
            family_distribution={},
        )

    candidate_payloads = []
    family_distribution: dict[str, int] = {}
    for candidate in normalized_candidates:
        family, _family_payload = _candidate_family_and_payload(candidate)
        family_distribution[family] = family_distribution.get(family, 0) + 1
        candidate_payloads.append((candidate, family, _candidate_score_payloads(candidate)))

    reference_payloads = [_candidate_score_payloads(item) for item in config.reference_candidates]
    candidate_vectors: dict[int, tuple[float, ...]] = {}
    missing_diversity_paths: dict[int, tuple[str, ...]] = {}
    valid_candidate_vectors: list[tuple[int, str, str, tuple[float, ...]]] = []
    for index, (candidate, _family, payloads) in enumerate(candidate_payloads):
        vector, missing_paths = _candidate_feature_vector(
            candidate,
            paths=config.diversity_paths,
            payloads=payloads,
        )
        candidate_vectors[index] = vector
        if missing_paths:
            missing_diversity_paths[index] = missing_paths
            continue
        valid_candidate_vectors.append(
            (index, candidate.candidate_id, candidate.stable_hash(), vector)
        )

    all_vectors: list[tuple[int, str, str, tuple[float, ...]]] = list(valid_candidate_vectors)
    reference_diversity_errors: list[tuple[str, tuple[str, ...]]] = []
    for offset, (reference_candidate, reference_payload) in enumerate(
        zip(config.reference_candidates, reference_payloads),
    ):
        vector, missing_paths = _candidate_feature_vector(
            reference_candidate,
            paths=config.diversity_paths,
            payloads=reference_payload,
        )
        if missing_paths:
            if config.policy == "diversity":
                reference_diversity_errors.append(
                    (
                        reference_candidate.candidate_id,
                        tuple(f"diversity_paths.{path}" for path in missing_paths),
                    )
                )
            continue
        all_vectors.append(
            (
                -(offset + 1),
                reference_candidate.candidate_id,
                reference_candidate.stable_hash(),
                vector,
            )
        )
    if reference_diversity_errors:
        details = "; ".join(
            f"{candidate_id}: {', '.join(paths)}"
            for candidate_id, paths in reference_diversity_errors
        )
        raise ValueError(
            "reference_candidates contain invalid or missing diversity_paths: "
            f"{details}."
        )

    scored: list[tuple[Candidate, AcquisitionScore]] = []
    rejected: dict[str, tuple[str, ...]] = {}
    for index, (candidate, family, payloads) in enumerate(candidate_payloads):
        if config.policy == "diversity" and index in missing_diversity_paths:
            rejected[candidate.candidate_id] = tuple(
                f"diversity_paths.{path}" for path in missing_diversity_paths[index]
            )
            continue
        vector = candidate_vectors.get(index, ())
        try:
            accepted, value, metadata = _compute_score(
                candidate,
                payloads,
                config,
                family,
                candidate_index=index,
                candidate_vector=vector,
                all_vectors=tuple(all_vectors),
            )
        except Exception as exc:
            rejected[candidate.candidate_id] = (f"{exc}",)
            continue
        if not accepted:
            reason = ("missing_field",)
            if isinstance(metadata, Mapping) and "missing_field" in metadata:
                value_missing = str(metadata.get("missing_field"))
                if value_missing:
                    reason = (value_missing,)
            rejected[candidate.candidate_id] = reason
            continue
        if metadata is None:
            metadata = {}
        score_payload = dict(metadata)
        score_payload["policy"] = config.policy
        scored.append(
            (
                candidate,
                AcquisitionScore(
                    candidate_id=candidate.candidate_id,
                    score=value,
                    metadata=score_payload,
                ),
            )
        )

    ranked_scores = _rank_acquisition_scores(tuple(scored))
    ranked_candidate_ids = tuple(score.candidate_id for score in ranked_scores)
    return AcquisitionScoringResult(
        config=config,
        scores=ranked_scores,
        ranked_candidate_ids=ranked_candidate_ids,
        rejected_candidate_ids=tuple(rejected.keys()),
        rejection_reasons=rejected,
        family_distribution=family_distribution,
    )


@dataclass(frozen=True)
class AcquisitionScoringArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _build_acquisition_manifest(
    result: AcquisitionScoringResult,
    run_id: str,
    iteration: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": ACTIVE_LEARNING_ACQUISITION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": iteration,
        "policy": result.config.policy,
        "candidate_count": sum(result.family_distribution.values()),
        "scored_count": len(result.scores),
        "rejected_count": len(result.rejected_candidate_ids),
        "family_distribution": dict(result.family_distribution),
        "candidate_ids": [item.candidate_id for item in result.scores],
        "rejected_candidate_ids": list(result.rejected_candidate_ids),
        "artifact_dir": str(artifact_dir),
    }


def _build_acquisition_report(result: AcquisitionScoringResult, run_id: str, iteration: int) -> dict[str, Any]:
    return {
        "schema_version": ACTIVE_LEARNING_ACQUISITION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": iteration,
        "policy": result.config.policy,
        "scores": [item.as_dict() for item in result.scores],
        "ranked_candidate_ids": list(result.ranked_candidate_ids),
        "candidate_ids": [item.candidate_id for item in result.scores],
        "rejected_candidate_ids": list(result.rejected_candidate_ids),
        "rejection_reasons": {key: list(value) for key, value in result.rejection_reasons.items()},
        "family_distribution": dict(result.family_distribution),
        "weights": dict(result.config.weights),
        "candidate_count": sum(result.family_distribution.values()),
        "scored_count": len(result.scores),
        "rejected_count": len(result.rejected_candidate_ids),
        "plot": ACQUISITION_PLOT_FILENAME,
    }


def _artifact_iteration_directory(output_root: str | Path, run_id: str, iteration: int) -> Path:
    return Path(output_root) / run_id / "iterations" / f"iter_{iteration:04d}"


def _write_plot(path: Path, report: Mapping[str, Any], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_build_fallback_png(report))
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_build_fallback_png(report))
        return

    try:
        scores = _report_scores(report)
        candidate_ids = [str(item.get("candidate_id", "")) for item in report.get("scores", ()) if isinstance(item, Mapping)]
        x_values = list(range(1, len(scores) + 1))
        fig, axis = plt.subplots()
        if scores:
            axis.plot(x_values, scores, marker="o", linewidth=1.0)
        else:
            axis.plot([1], [0], marker="o")
        axis.set_title(f"Acquisition: {report.get('policy', 'unknown')}")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Score")
        axis.grid(True, alpha=0.35)
        if candidate_ids and len(candidate_ids) == len(scores):
            axis.set_xticks(x_values)
            axis.set_xticklabels(candidate_ids, rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
    except Exception:
        path.write_bytes(_build_fallback_png(report))


def write_acquisition_artifacts(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int,
    result: AcquisitionScoringResult,
    include_plot: bool = True,
) -> AcquisitionScoringArtifacts:
    run_id_text = _coerce_nonempty_text(run_id, field_name="run_id")
    iteration_num = _coerce_non_negative_int(iteration, field_name="iteration")
    artifact_dir = _artifact_iteration_directory(output_root, run_id_text, iteration_num)
    manifest_path = artifact_dir / ACQUISITION_MANIFEST_FILENAME
    report_path = artifact_dir / ACQUISITION_REPORT_FILENAME
    plot_path = artifact_dir / ACQUISITION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / ACQUISITION_PLOT_SIDECAR_FILENAME

    manifest = _build_acquisition_manifest(result, run_id_text, iteration_num, artifact_dir)
    report = _build_acquisition_report(result, run_id_text, iteration_num)
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _write_plot(plot_path, report, include_plot=include_plot)
    _write_json(plot_sidecar_path, report)

    return AcquisitionScoringArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        report=report,
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(
        ">I",
        zlib.crc32(chunk_type + data) & 0xFFFFFFFF,
    )


def _report_scores(report: Mapping[str, Any]) -> list[float]:
    scores: list[float] = []
    for item in report.get("scores", ()):
        if not isinstance(item, Mapping):
            continue
        try:
            scores.append(_coerce_score(item["score"], context="plot score"))
        except (KeyError, ValueError):
            continue
    return scores


def _set_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, top), min(height, bottom + 1)):
        for x in range(max(0, left), min(width, right + 1)):
            _set_pixel(pixels, width, height, x, y, color)


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _encode_png_rgb(width: int, height: int, pixels: bytearray) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    stride = width * 3
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(pixels[row * stride : (row + 1) * stride])
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(bytes(raw))) + _png_chunk(
        b"IEND",
        b"",
    )


def _build_fallback_png(report: Mapping[str, Any]) -> bytes:
    width = 360
    height = 220
    left = 42
    right = 18
    top = 24
    bottom = 38
    plot_right = width - right
    plot_bottom = height - bottom
    plot_width = plot_right - left
    plot_height = plot_bottom - top
    pixels = bytearray([255, 255, 255]) * width * height
    scores = _report_scores(report)

    axis_color = (45, 45, 45)
    grid_color = (218, 224, 231)
    bar_color = (181, 207, 234)
    line_color = (31, 96, 158)
    marker_color = (15, 57, 96)
    empty_color = (165, 172, 181)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + round(plot_height * fraction)
        _draw_line(pixels, width, height, left, y, plot_right, y, grid_color)
    _draw_line(pixels, width, height, left, top, left, plot_bottom, axis_color)
    _draw_line(pixels, width, height, left, plot_bottom, plot_right, plot_bottom, axis_color)

    if not scores:
        _draw_line(
            pixels,
            width,
            height,
            left + 22,
            plot_bottom - 18,
            plot_right - 22,
            top + 18,
            empty_color,
        )
        return _encode_png_rgb(width, height, pixels)

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    points: list[tuple[int, int]] = []
    count = len(scores)
    for index, score in enumerate(scores):
        if count == 1:
            x = left + plot_width // 2
        else:
            x = left + round((plot_width * index) / (count - 1))
        normalized = 0.5 if score_range == 0.0 else (score - min_score) / score_range
        y = top + round((1.0 - normalized) * plot_height)
        points.append((x, y))
        _draw_rect(pixels, width, height, x - 3, y, x + 3, plot_bottom - 1, bar_color)

    for first, second in zip(points, points[1:]):
        _draw_line(pixels, width, height, first[0], first[1], second[0], second[1], line_color)
    for x, y in points:
        _draw_rect(pixels, width, height, x - 3, y - 3, x + 3, y + 3, marker_color)

    return _encode_png_rgb(width, height, pixels)


__all__ = [
    "ACTIVE_LEARNING_ACQUISITION_ARTIFACT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION",
    "ACQUISITION_MANIFEST_FILENAME",
    "ACQUISITION_PLOT_FILENAME",
    "ACQUISITION_PLOT_SIDECAR_FILENAME",
    "ACQUISITION_REPORT_FILENAME",
    "AcquisitionScoringArtifacts",
    "AcquisitionScoringConfig",
    "AcquisitionScoringResult",
    "SUPPORTED_POLICIES",
    "score_acquisition_candidates",
    "write_acquisition_artifacts",
]
