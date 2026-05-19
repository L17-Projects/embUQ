from __future__ import annotations

"""Dependency-light candidate generation for Active Learning."""

import copy
import json
import math
import random
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate


ACTIVE_LEARNING_CANDIDATE_DIMENSION_SCHEMA_VERSION = "meso_uq.active_learning.candidate_dimension.v1"
ACTIVE_LEARNING_CANDIDATE_GENERATION_CONFIG_SCHEMA_VERSION = (
    "meso_uq.active_learning.candidate_generation_config.v1"
)
ACTIVE_LEARNING_CANDIDATE_GENERATION_RESULT_SCHEMA_VERSION = (
    "meso_uq.active_learning.candidate_generation_result.v1"
)
ACTIVE_LEARNING_CANDIDATE_GENERATION_ARTIFACT_SCHEMA_VERSION = (
    "meso_uq.active_learning.candidate_generation_artifacts.v1"
)
CANDIDATE_GENERATION_MANIFEST_FILENAME = "candidate_generation_manifest.json"
CANDIDATE_GENERATION_REPORT_FILENAME = "candidate_generation_report.json"
CANDIDATE_GENERATION_PLOT_FILENAME = "candidate_generation_validation_plot.png"
CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME = "candidate_generation_validation_plot.png.json"

_SUPPORTED_FAMILIES = frozenset({"gv", "emb"})
_SUPPORTED_STRATEGIES = frozenset(
    {
        "posterior_region",
        "posterior_boundary",
        "prior_exploration",
        "validity_boundary",
    }
)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_finite_float(value: object, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    return number


def _coerce_positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _coerce_path(value: str | Sequence[object]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(".") if part.strip())
    elif isinstance(value, Sequence):
        parts = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise ValueError("dimension path must be a dot string or sequence.")
    if not parts:
        raise ValueError("dimension path must contain at least one component.")
    return parts


def _set_nested_value(payload: dict[str, Any], path: Sequence[str], value: float) -> None:
    current: dict[str, Any] = payload
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set generated value below non-mapping path component {part!r}.")
        current = existing
    current[path[-1]] = value


def _scope_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "scope"


@dataclass(frozen=True)
class CandidateParameterDimension:
    name: str
    path: tuple[str, ...] | str | Sequence[object]
    lower: float
    upper: float
    posterior_mean: float | None = None
    posterior_std: float | None = None
    units: str | None = None
    schema_version: str = ACTIVE_LEARNING_CANDIDATE_DIMENSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _coerce_nonempty_text(self.name, field_name="dimension name"))
        object.__setattr__(self, "path", _coerce_path(self.path))
        lower = _coerce_finite_float(self.lower, field_name=f"dimension {self.name} lower")
        upper = _coerce_finite_float(self.upper, field_name=f"dimension {self.name} upper")
        if lower >= upper:
            raise ValueError(f"dimension {self.name} lower must be < upper.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.posterior_mean is not None:
            mean = _coerce_finite_float(self.posterior_mean, field_name=f"dimension {self.name} posterior_mean")
            object.__setattr__(self, "posterior_mean", mean)
        if self.posterior_std is not None:
            std = _coerce_finite_float(self.posterior_std, field_name=f"dimension {self.name} posterior_std")
            if std <= 0.0:
                raise ValueError(f"dimension {self.name} posterior_std must be > 0.")
            object.__setattr__(self, "posterior_std", std)
        if self.units is not None:
            object.__setattr__(self, "units", str(self.units))

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": list(self.path),
            "lower": self.lower,
            "upper": self.upper,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "units": self.units,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateParameterDimension":
        return cls(
            name=str(payload["name"]),
            path=payload["path"],
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
            posterior_mean=payload.get("posterior_mean"),
            posterior_std=payload.get("posterior_std"),
            units=payload.get("units"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_CANDIDATE_DIMENSION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class CandidateGenerationConfig:
    family: str
    experiment: str
    dimensions: tuple[CandidateParameterDimension, ...] | Sequence[CandidateParameterDimension | Mapping[str, Any]]
    count: int
    strategy: str = "prior_exploration"
    seed: int = 0
    payload_template: Mapping[str, Any] = field(default_factory=dict)
    candidate_prefix: str = "al-candidate"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_CANDIDATE_GENERATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        family = _coerce_nonempty_text(self.family, field_name="family").lower()
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(sorted(_SUPPORTED_FAMILIES))}.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "experiment", _coerce_nonempty_text(self.experiment, field_name="experiment"))
        object.__setattr__(self, "count", _coerce_positive_integer(self.count, field_name="count"))
        strategy = _coerce_nonempty_text(self.strategy, field_name="strategy")
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(f"strategy must be one of: {', '.join(sorted(_SUPPORTED_STRATEGIES))}.")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "seed", int(self.seed))
        dimensions = tuple(
            item if isinstance(item, CandidateParameterDimension) else CandidateParameterDimension.from_dict(item)
            for item in self.dimensions
        )
        if not dimensions:
            raise ValueError("dimensions must contain at least one parameter dimension.")
        names = [item.name for item in dimensions]
        if len(set(names)) != len(names):
            raise ValueError("dimension names must be unique.")
        object.__setattr__(self, "dimensions", dimensions)
        if not isinstance(self.payload_template, Mapping):
            raise ValueError("payload_template must be a mapping.")
        object.__setattr__(self, "payload_template", copy.deepcopy(dict(self.payload_template)))
        object.__setattr__(self, "candidate_prefix", _coerce_nonempty_text(self.candidate_prefix, field_name="candidate_prefix"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def payload_key(self) -> str:
        return f"{self.family}_launch"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "experiment": self.experiment,
            "dimensions": [item.as_dict() for item in self.dimensions],
            "count": self.count,
            "strategy": self.strategy,
            "seed": self.seed,
            "payload_template": copy.deepcopy(dict(self.payload_template)),
            "candidate_prefix": self.candidate_prefix,
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateGenerationConfig":
        return cls(
            family=str(payload["family"]),
            experiment=str(payload["experiment"]),
            dimensions=tuple(payload["dimensions"]),
            count=payload["count"],
            strategy=str(payload.get("strategy", "prior_exploration")),
            seed=int(payload.get("seed", 0)),
            payload_template=payload.get("payload_template", {}),
            candidate_prefix=str(payload.get("candidate_prefix", "al-candidate")),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_CANDIDATE_GENERATION_CONFIG_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class CandidateGenerationResult:
    config: CandidateGenerationConfig
    candidates: tuple[Candidate, ...]
    generated_values: tuple[Mapping[str, float], ...]
    schema_version: str = ACTIVE_LEARNING_CANDIDATE_GENERATION_RESULT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "candidate_count": len(self.candidates),
            "family": self.config.family,
            "experiment": self.config.experiment,
            "strategy": self.config.strategy,
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "generated_values": [dict(values) for values in self.generated_values],
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class CandidateGenerationArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _sample_dimension(
    *,
    dimension: CandidateParameterDimension,
    strategy: str,
    rng: random.Random,
    sample_index: int,
    dimension_index: int,
    sample_count: int,
) -> float:
    if strategy == "prior_exploration":
        return rng.uniform(dimension.lower, dimension.upper)
    if strategy == "posterior_region":
        mean = dimension.posterior_mean if dimension.posterior_mean is not None else dimension.midpoint
        std = dimension.posterior_std if dimension.posterior_std is not None else dimension.width / 6.0
        return dimension.clamp(rng.gauss(mean, std))
    if strategy == "posterior_boundary":
        mean = dimension.posterior_mean if dimension.posterior_mean is not None else dimension.midpoint
        std = dimension.posterior_std if dimension.posterior_std is not None else dimension.width / 6.0
        sign = -1.0 if (sample_index + dimension_index) % 2 else 1.0
        scale = 1.5 + 0.5 * ((sample_index + dimension_index) % 3)
        return dimension.clamp(mean + sign * scale * std)
    if strategy == "validity_boundary":
        margin = 0.05 * dimension.width
        use_upper = (sample_index + dimension_index) % 2 == 0
        edge = dimension.upper - margin if use_upper else dimension.lower + margin
        jitter = rng.uniform(-0.01 * dimension.width, 0.01 * dimension.width)
        return dimension.clamp(edge + jitter)
    raise ValueError(f"Unsupported candidate generation strategy: {strategy}.")


def generate_candidate_batch(config: CandidateGenerationConfig | Mapping[str, Any]) -> CandidateGenerationResult:
    normalized_config = (
        config if isinstance(config, CandidateGenerationConfig) else CandidateGenerationConfig.from_dict(config)
    )
    rng = random.Random(normalized_config.seed)
    candidates: list[Candidate] = []
    generated_values: list[dict[str, float]] = []
    family_token = _scope_token(normalized_config.family)
    experiment_token = _scope_token(normalized_config.experiment)
    seed_token = _scope_token(str(normalized_config.seed))
    for sample_index in range(normalized_config.count):
        payload = copy.deepcopy(dict(normalized_config.payload_template))
        payload["experiment"] = normalized_config.experiment
        values: dict[str, float] = {}
        for dimension_index, dimension in enumerate(normalized_config.dimensions):
            value = _sample_dimension(
                dimension=dimension,
                strategy=normalized_config.strategy,
                rng=rng,
                sample_index=sample_index,
                dimension_index=dimension_index,
                sample_count=normalized_config.count,
            )
            _set_nested_value(payload, dimension.path, value)
            values[dimension.name] = value
        candidate_id = (
            f"{normalized_config.candidate_prefix}-"
            f"{family_token}-{experiment_token}-"
            f"{normalized_config.strategy}-seed-{seed_token}-{sample_index:04d}"
        )
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                parameters={
                    "family": normalized_config.family,
                    normalized_config.payload_key: payload,
                },
                metadata={
                    **dict(normalized_config.metadata),
                    "source": "active_learning_candidate_generation",
                    "strategy": normalized_config.strategy,
                    "family": normalized_config.family,
                    "experiment": normalized_config.experiment,
                    "seed": normalized_config.seed,
                    "sample_index": sample_index,
                    "generated_values": dict(values),
                    "parameter_paths": {
                        dimension.name: list(dimension.path)
                        for dimension in normalized_config.dimensions
                    },
                },
            )
        )
        generated_values.append(values)
    return CandidateGenerationResult(
        config=normalized_config,
        candidates=tuple(candidates),
        generated_values=tuple(generated_values),
    )


def _artifact_iteration_dir(output_root: str | Path, run_id: str, iteration: int) -> Path:
    return Path(output_root) / run_id / "iterations" / f"iter_{iteration:04d}"


def _plot_candidate_generation(path: Path, result: CandidateGenerationResult, *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return

    dimensions = [dimension.name for dimension in result.config.dimensions]
    fig, axes = plt.subplots(max(1, len(dimensions)), 1, figsize=(8, max(3, 2.5 * len(dimensions))))
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = [axes]
    for axis, dimension in zip(axes, result.config.dimensions, strict=False):
        values = [sample[dimension.name] for sample in result.generated_values]
        axis.plot(range(len(values)), values, marker="o", linewidth=1.0)
        axis.axhline(dimension.lower, color="red", linestyle="--", linewidth=0.8)
        axis.axhline(dimension.upper, color="red", linestyle="--", linewidth=0.8)
        axis.set_title(f"{dimension.name} ({'.'.join(dimension.path)})")
        axis.set_xlabel("Candidate")
        axis.set_ylabel(dimension.units or "value")
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"Candidate Generation: {result.config.family}/{result.config.experiment}/{result.config.strategy}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_candidate_generation_artifacts(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int,
    result: CandidateGenerationResult,
    include_plot: bool = True,
) -> CandidateGenerationArtifacts:
    run_id_text = _coerce_nonempty_text(run_id, field_name="run_id")
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    artifact_dir = _artifact_iteration_dir(output_root, run_id_text, iteration)
    manifest_path = artifact_dir / CANDIDATE_GENERATION_MANIFEST_FILENAME
    report_path = artifact_dir / CANDIDATE_GENERATION_REPORT_FILENAME
    plot_path = artifact_dir / CANDIDATE_GENERATION_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME
    manifest = result.as_dict()
    report = {
        "schema_version": ACTIVE_LEARNING_CANDIDATE_GENERATION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id_text,
        "iteration": iteration,
        "family_distribution": {result.config.family: len(result.candidates)},
        "experiment_distribution": {result.config.experiment: len(result.candidates)},
        "strategy": result.config.strategy,
        "dimension_ranges": {
            dimension.name: {
                "path": list(dimension.path),
                "lower": dimension.lower,
                "upper": dimension.upper,
            }
            for dimension in result.config.dimensions
        },
        "candidate_hashes": {
            candidate.candidate_id: candidate.stable_hash()
            for candidate in result.candidates
        },
        "generated_values": [dict(values) for values in result.generated_values],
        "plot": str(plot_path),
    }
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _plot_candidate_generation(plot_path, result, include_plot=include_plot)
    _write_json(plot_sidecar_path, report)
    return CandidateGenerationArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        report=report,
    )


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _build_fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_pixel = b"\x00\x00\x00\x00\x00"
    idat_data = zlib.compress(raw_pixel)
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", idat_data) + _png_chunk(b"IEND", b"")


_FALLBACK_PNG = _build_fallback_png()


__all__ = [
    "ACTIVE_LEARNING_CANDIDATE_DIMENSION_SCHEMA_VERSION",
    "ACTIVE_LEARNING_CANDIDATE_GENERATION_ARTIFACT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_CANDIDATE_GENERATION_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_CANDIDATE_GENERATION_RESULT_SCHEMA_VERSION",
    "CANDIDATE_GENERATION_MANIFEST_FILENAME",
    "CANDIDATE_GENERATION_PLOT_FILENAME",
    "CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME",
    "CANDIDATE_GENERATION_REPORT_FILENAME",
    "CandidateGenerationArtifacts",
    "CandidateGenerationConfig",
    "CandidateGenerationResult",
    "CandidateParameterDimension",
    "generate_candidate_batch",
    "write_candidate_generation_artifacts",
]
