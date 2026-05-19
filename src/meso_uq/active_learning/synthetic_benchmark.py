from __future__ import annotations

"""Lightweight synthetic benchmark records for Active Learning dry-runs."""

import hashlib
import json
import math
import random
import struct
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate


ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_SCHEMA_VERSION = (
    "meso_uq.active_learning.synthetic_benchmark.v1"
)
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_CONFIG_SCHEMA_VERSION = (
    "meso_uq.active_learning.synthetic_benchmark_config.v1"
)
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_RESULT_SCHEMA_VERSION = (
    "meso_uq.active_learning.synthetic_benchmark_result.v1"
)
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_ARTIFACT_SCHEMA_VERSION = (
    "meso_uq.active_learning.synthetic_benchmark_artifact.v1"
)
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_MANIFEST_FILENAME = "synthetic_benchmark_manifest.json"
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_REPORT_FILENAME = "synthetic_benchmark_report.json"
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_FILENAME = "synthetic_benchmark_validation_plot.png"
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_SIDECAR_FILENAME = (
    "synthetic_benchmark_validation_plot.png.json"
)
ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_DPI = 90
_VALIDATION_PLOT_WIDTH = 720
_VALIDATION_PLOT_HEIGHT = 540

_SUPPORTED_FAMILIES = frozenset({"gv", "emb"})
_SUPPORTED_BASELINES = ("random", "lhs", "sobol_like")
_DEFAULT_SEED = 7
_DEFAULT_COMPLETION_STEPS = 6
_DEFAULT_RESPONSE_POINTS = (0.0, 0.2, 0.45, 0.7, 1.0)
_RESPONSE_TEMPLATE_GV = (0.03, 0.12, 0.24, 0.41, 0.52)
_RESPONSE_TEMPLATE_EMB = (0.21, 0.29, 0.52, 0.77, 0.99)
_GV_PARAMS = (
    ("material_parameters", "ka", 0.85, 1.35),
    ("material_parameters", "kb", 1.05, 1.45),
    ("geometry", "radGV", 1.75, 2.25),
)
_EMB_PARAMS = (
    ("physical_parameters", "elastic_modulus", 0.10, 1.10),
    ("physical_parameters", "thickness", 0.20, 0.90),
    ("controls", "indentation_depth", 0.10, 0.90),
)
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_points(points: Sequence[float] | None) -> tuple[float, ...]:
    if points is None:
        return _DEFAULT_RESPONSE_POINTS
    coerced = tuple(float(value) for value in points)
    if len(coerced) < 2:
        raise ValueError("response_points must contain at least two values.")
    for value in coerced:
        if not math.isfinite(value):
            raise ValueError("response_points values must be finite.")
    return coerced


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + type_code
        + data
        + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)
    )


def _encode_png_rgba(*, width: int, height: int, pixels: bytes) -> bytes:
    if len(pixels) != width * height * 4:
        raise ValueError("pixels must contain one RGBA tuple per output pixel.")
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    stride = width * 4
    raw_rows = bytearray()
    for row_index in range(height):
        raw_rows.append(0)
        start = row_index * stride
        raw_rows.extend(pixels[start : start + stride])
    idat_data = zlib.compress(bytes(raw_rows))
    return (
        signature
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", idat_data)
        + _png_chunk(b"IEND", b"")
    )


def _build_fallback_png() -> bytes:
    return _encode_png_rgba(width=1, height=1, pixels=b"\x00\x00\x00\x00")


def _set_pixel(
    pixels: bytearray,
    *,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    index = (y * width + x) * 4
    pixels[index : index + 4] = bytes(color)


def _draw_rect(
    pixels: bytearray,
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int, int],
) -> None:
    clipped_left = max(0, min(width, left))
    clipped_right = max(0, min(width, right))
    clipped_top = max(0, min(height, top))
    clipped_bottom = max(0, min(height, bottom))
    for y in range(clipped_top, clipped_bottom):
        for x in range(clipped_left, clipped_right):
            _set_pixel(pixels, width=width, height=height, x=x, y=y, color=color)


def _draw_line(
    pixels: bytearray,
    *,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_pixel(pixels, width=width, height=height, x=x0, y=y0, color=color)
        if x0 == x1 and y0 == y1:
            break
        next_error = 2 * error
        if next_error >= dy:
            error += dy
            x0 += step_x
        if next_error <= dx:
            error += dx
            y0 += step_y


def _draw_axes(
    pixels: bytearray,
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    axis_color = (50, 50, 50, 255)
    grid_color = (220, 220, 220, 255)
    for step in range(1, 4):
        y = bottom - int((bottom - top) * step / 4)
        _draw_line(
            pixels,
            width=width,
            height=height,
            x0=left,
            y0=y,
            x1=right,
            y1=y,
            color=grid_color,
        )
    _draw_line(
        pixels,
        width=width,
        height=height,
        x0=left,
        y0=top,
        x1=left,
        y1=bottom,
        color=axis_color,
    )
    _draw_line(
        pixels,
        width=width,
        height=height,
        x0=left,
        y0=bottom,
        x1=right,
        y1=bottom,
        color=axis_color,
    )


def _plot_coordinate(
    *,
    value: float,
    lower: float,
    upper: float,
    start: int,
    end: int,
    invert: bool = False,
) -> int:
    if upper == lower:
        fraction = 0.5
    else:
        fraction = (value - lower) / (upper - lower)
    fraction = max(0.0, min(1.0, fraction))
    if invert:
        fraction = 1.0 - fraction
    return int(round(start + fraction * (end - start)))


def _build_validation_png(result: "SyntheticBenchmarkResult") -> bytes:
    width = _VALIDATION_PLOT_WIDTH
    height = _VALIDATION_PLOT_HEIGHT
    pixels = bytearray((255, 255, 255, 255) * width * height)
    response_box = (64, 28, width - 28, 246)
    baseline_box = (64, 302, width - 28, height - 54)
    status_colors = {
        "queued": (120, 120, 120, 255),
        "partial": (230, 126, 34, 255),
        "completed": (31, 119, 180, 255),
    }

    _draw_axes(
        pixels,
        width=width,
        height=height,
        left=response_box[0],
        top=response_box[1],
        right=response_box[2],
        bottom=response_box[3],
    )
    _draw_axes(
        pixels,
        width=width,
        height=height,
        left=baseline_box[0],
        top=baseline_box[1],
        right=baseline_box[2],
        bottom=baseline_box[3],
    )

    displayed_records = result.records[: min(4, len(result.records))]
    if displayed_records:
        x_values = [float(value) for value in displayed_records[0].response_points]
        y_values = [float(value) for record in displayed_records for value in record.response_curve]
        x_lower = min(x_values)
        x_upper = max(x_values)
        y_lower = min(y_values)
        y_upper = max(y_values)
        padding = (y_upper - y_lower) * 0.08 if y_upper != y_lower else 0.1
        y_lower -= padding
        y_upper += padding

        for index, record in enumerate(displayed_records):
            color = status_colors.get(record.completion_status, (31, 119, 180, 255))
            points = [
                (
                    _plot_coordinate(
                        value=float(x_value),
                        lower=x_lower,
                        upper=x_upper,
                        start=response_box[0],
                        end=response_box[2],
                    ),
                    _plot_coordinate(
                        value=float(y_value),
                        lower=y_lower,
                        upper=y_upper,
                        start=response_box[1],
                        end=response_box[3],
                        invert=True,
                    ),
                )
                for x_value, y_value in zip(record.response_points, record.response_curve, strict=False)
            ]
            for first, second in zip(points, points[1:], strict=False):
                _draw_line(
                    pixels,
                    width=width,
                    height=height,
                    x0=first[0],
                    y0=first[1],
                    x1=second[0],
                    y1=second[1],
                    color=color,
                )
            for x, y in points:
                _draw_rect(
                    pixels,
                    width=width,
                    height=height,
                    left=x - 3,
                    top=y - 3,
                    right=x + 4,
                    bottom=y + 4,
                    color=color,
                )
            legend_x = response_box[0] + index * 42
            _draw_rect(
                pixels,
                width=width,
                height=height,
                left=legend_x,
                top=response_box[3] + 16,
                right=legend_x + 26,
                bottom=response_box[3] + 28,
                color=color,
            )

        bars: list[tuple[int, float, tuple[int, int, int, int]]] = []
        for record_index, record in enumerate(displayed_records):
            color = status_colors.get(record.completion_status, (31, 119, 180, 255))
            for value in record.baseline_distances.values():
                bars.append((record_index, float(value), color))

        max_distance = max((value for _, value, _ in bars), default=1.0)
        if max_distance <= 0.0:
            max_distance = 1.0
        if bars:
            available_width = baseline_box[2] - baseline_box[0]
            slot_width = max(4, available_width // len(bars))
            bar_width = max(3, int(slot_width * 0.62))
            for bar_index, (_record_index, value, color) in enumerate(bars):
                center = baseline_box[0] + int((bar_index + 0.5) * available_width / len(bars))
                bar_height = int((baseline_box[3] - baseline_box[1]) * value / max_distance)
                _draw_rect(
                    pixels,
                    width=width,
                    height=height,
                    left=center - bar_width // 2,
                    top=baseline_box[3] - bar_height,
                    right=center + bar_width // 2,
                    bottom=baseline_box[3],
                    color=color,
                )

    return _encode_png_rgba(width=width, height=height, pixels=bytes(pixels))


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1
    i = index
    while i:
        i, remainder = divmod(i, base)
        denominator *= base
        value += remainder / denominator
    return value


def _scale_unit_to_range(unit_value: float, *, lower: float, upper: float) -> float:
    return lower + unit_value * (upper - lower)


def build_random_sequence(
    *,
    count: int,
    dimensions: int,
    seed: int = _DEFAULT_SEED,
) -> tuple[tuple[float, ...], ...]:
    if count < 0:
        raise ValueError("count must be a non-negative integer.")
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")
    rng = random.Random(seed)
    return tuple(tuple(rng.random() for _ in range(dimensions)) for _ in range(count))


def build_latin_hypercube_sequence(
    *,
    count: int,
    dimensions: int,
    seed: int = _DEFAULT_SEED,
) -> tuple[tuple[float, ...], ...]:
    if count < 0:
        raise ValueError("count must be a non-negative integer.")
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")
    rng = random.Random(seed)
    samples = [[0.0 for _ in range(dimensions)] for _ in range(count)]
    for dim_index in range(dimensions):
        order = list(range(count))
        rng.shuffle(order)
        for row_index, sample_index in enumerate(order):
            samples[sample_index][dim_index] = (row_index + rng.random()) / count
    return tuple(tuple(values) for values in samples)


def build_sobol_like_sequence(
    *,
    count: int,
    dimensions: int,
    seed: int = _DEFAULT_SEED,
) -> tuple[tuple[float, ...], ...]:
    if count < 0:
        raise ValueError("count must be a non-negative integer.")
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")
    rng = random.Random(seed)
    shifts = tuple(rng.random() for _ in range(dimensions))
    sequences: list[tuple[float, ...]] = []
    for sample_index in range(count):
        values = []
        for dim_index in range(dimensions):
            value = _van_der_corput(sample_index + 1, _PRIMES[dim_index % len(_PRIMES)])
            shifted = (value + shifts[dim_index]) % 1.0
            values.append(shifted)
        sequences.append(tuple(values))
    return tuple(sequences)


_BASELINE_GENERATORS = {
    "random": build_random_sequence,
    "lhs": build_latin_hypercube_sequence,
    "sobol_like": build_sobol_like_sequence,
}


def _parameter_specs(family: str) -> tuple[tuple[str, str, float, float], ...]:
    return _GV_PARAMS if family == "gv" else _EMB_PARAMS


def _parameter_payload_key(*, family: str) -> str:
    return "gv_launch" if family == "gv" else "emb_launch"


def _build_launch_payload(
    *,
    family: str,
    experiment: str,
    unit_samples: Sequence[float],
    sample_index: int,
    sequence_name: str,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {"family": family}
    payload_key = _parameter_payload_key(family=family)
    samples = {
        spec[1]: _scale_unit_to_range(unit_value, lower=spec[2], upper=spec[3])
        for spec, unit_value in zip(_parameter_specs(family), unit_samples, strict=False)
    }
    if family == "gv":
        payload[payload_key] = {
            "experiment": experiment,
            "material_parameters": {
                "ka": samples["ka"],
                "kb": samples["kb"],
            },
            "geometry": {"radGV": samples["radGV"], "height": 14.28},
            "controls": {
                "bpress": -91.0,
                "tot_force": [500.0, 750.0],
            },
            "sequence": sequence_name,
            "seed": seed,
            "sample_index": sample_index,
            "run_id": run_id,
            "sample": dict(samples),
        }
        return payload

    payload[payload_key] = {
        "experiment": experiment,
        "physical_parameters": {
            "elastic_modulus": samples["elastic_modulus"],
            "thickness": samples["thickness"],
        },
        "controls": {
            "indentation_depth": samples["indentation_depth"],
        },
        "sequence": sequence_name,
        "seed": seed,
        "sample_index": sample_index,
        "run_id": run_id,
        "sample": dict(samples),
    }
    return payload


def _interpolate_template(values: Sequence[float], x: float) -> float:
    if x <= 0.0:
        return float(values[0])
    if x >= 1.0:
        return float(values[-1])
    scaled = x * (len(values) - 1)
    lower = int(scaled)
    upper = lower + 1
    t = scaled - lower
    return float(values[lower] * (1.0 - t) + values[upper] * t)


def _response_curve(
    *,
    points: Sequence[float],
    family: str,
    unit_samples: Sequence[float],
    completion_ratio: float,
) -> tuple[float, ...]:
    if not points:
        raise ValueError("response_points must not be empty.")
    xmin = min(points)
    xmax = max(points)
    if xmax == xmin:
        raise ValueError("response_points must contain at least two distinct values.")
    template = _RESPONSE_TEMPLATE_GV if family == "gv" else _RESPONSE_TEMPLATE_EMB
    center = sum(unit_samples) / len(unit_samples)
    scale = 0.5 + 0.5 * center
    result = []
    for index, coordinate in enumerate(points):
        normalized_x = (float(coordinate) - xmin) / (xmax - xmin)
        base_value = _interpolate_template(template, normalized_x)
        delta = (
            0.08 * scale * math.sin((normalized_x + center * math.pi) * (1.0 + completion_ratio))
            + 0.02 * math.cos(index + center)
        )
        result.append(base_value + delta)
    return tuple(result)


def _reduced_observables(curve: Sequence[float]) -> tuple[tuple[str, float], ...]:
    values = [float(value) for value in curve]
    total = sum(values)
    return (
        ("peak", max(values)),
        ("final", values[-1]),
        ("mean", total / len(values)),
    )


def _completion_state(*, index: int, step_count: int) -> tuple[str, int, int, float, str | None]:
    if index % 4 == 0:
        return "queued", 0, step_count, 0.0, None
    if index % 4 == 1:
        completed = max(1, step_count // 2)
        return "partial", completed, step_count, completed / step_count, f"{index:06d}"
    return "completed", step_count, step_count, 1.0, None


def _build_dataset_id(*, family: str, experiment: str, run_id: str, candidate_id: str) -> str:
    return f"{family}:{experiment}:{run_id}:{candidate_id}"


def _build_pseudo_hdf5_ref(*, family: str, run_id: str, candidate_id: str) -> str:
    token = hashlib.sha256(
        f"{family}|{run_id}|{candidate_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"h5://synthetic/{family}/{run_id}/{candidate_id}.h5::/{token}"


def compare_to_baselines(
    *,
    candidate_vectors: Sequence[Sequence[float]],
    seed: int,
    baseline_names: Sequence[str],
) -> tuple[tuple[dict[str, float], ...], dict[str, dict[str, float]]]:
    if not candidate_vectors:
        return tuple(), {}
    vector_count = len(candidate_vectors)
    dimensions = len(candidate_vectors[0])
    if any(len(item) != dimensions for item in candidate_vectors):
        raise ValueError("candidate_vectors must contain vectors with equal dimensionality.")

    normalized_baselines = []
    for name in baseline_names:
        normalized = str(name).strip().lower()
        if normalized not in _BASELINE_GENERATORS:
            raise ValueError(f"Unknown baseline '{name}'. Supported: {', '.join(_SUPPORTED_BASELINES)}.")
        normalized_baselines.append(normalized)

    per_candidate: list[dict[str, float]] = []
    distances_by_name: dict[str, list[float]] = {name: [] for name in normalized_baselines}

    for baseline_name in normalized_baselines:
        baseline_vectors = _BASELINE_GENERATORS[baseline_name](
            count=vector_count,
            dimensions=dimensions,
            seed=seed,
        )
        for candidate_index, candidate_vector in enumerate(candidate_vectors):
            baseline_vector = baseline_vectors[candidate_index]
            entry = {
                baseline_name: math.dist(candidate_vector, baseline_vector)
            }
            if len(per_candidate) <= candidate_index:
                per_candidate.append({})
            per_candidate[candidate_index].update(entry)
            distances_by_name[baseline_name].append(entry[baseline_name])

    aggregate: dict[str, dict[str, float]] = {}
    for name, values in distances_by_name.items():
        aggregate[name] = {
            "mean_distance": sum(values) / len(values),
            "min_distance": min(values),
            "max_distance": max(values),
        }
    return tuple(per_candidate), aggregate


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    family: str
    experiment: str
    count: int
    seed: int = _DEFAULT_SEED
    sequence_name: str = "random"
    candidate_prefix: str = "al-synthetic"
    response_points: tuple[float, ...] = field(default_factory=lambda: _DEFAULT_RESPONSE_POINTS)
    completion_steps: int = _DEFAULT_COMPLETION_STEPS
    run_id: str = "synthetic-benchmark"
    baseline_names: tuple[str, ...] = ("random", "lhs", "sobol_like")
    schema_version: str = ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_CONFIG_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _coerce_nonempty_text(self.family, field_name="family").lower())
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(sorted(_SUPPORTED_FAMILIES))}.")
        object.__setattr__(self, "experiment", _coerce_nonempty_text(self.experiment, field_name="experiment"))
        if not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be a positive integer.")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "sequence_name",
            _coerce_nonempty_text(self.sequence_name, field_name="sequence_name").lower(),
        )
        if self.sequence_name not in _BASELINE_GENERATORS:
            raise ValueError(f"sequence_name must be one of: {', '.join(_SUPPORTED_BASELINES)}.")
        if not isinstance(self.completion_steps, int) or self.completion_steps < 1:
            raise ValueError("completion_steps must be a positive integer.")
        object.__setattr__(self, "response_points", _coerce_points(self.response_points))
        object.__setattr__(self, "run_id", _coerce_nonempty_text(self.run_id, field_name="run_id"))
        baselines = tuple(str(item).strip().lower() for item in self.baseline_names)
        if not baselines:
            raise ValueError("baseline_names must contain at least one value.")
        for baseline in baselines:
            if baseline not in _BASELINE_GENERATORS:
                raise ValueError(f"unknown baseline '{baseline}'.")
        object.__setattr__(self, "baseline_names", baselines)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticBenchmarkConfig":
        return cls(
            family=payload["family"],
            experiment=payload["experiment"],
            count=int(payload["count"]),
            seed=int(payload.get("seed", _DEFAULT_SEED)),
            sequence_name=payload.get("sequence_name", "random"),
            candidate_prefix=str(payload.get("candidate_prefix", "al-synthetic")),
            response_points=tuple(payload.get("response_points", _DEFAULT_RESPONSE_POINTS)),
            completion_steps=int(payload.get("completion_steps", _DEFAULT_COMPLETION_STEPS)),
            run_id=str(payload.get("run_id", "synthetic-benchmark")),
            baseline_names=tuple(payload.get("baseline_names", ("random", "lhs", "sobol_like"))),
            schema_version=str(
                payload.get(
                    "schema_version",
                    ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_CONFIG_SCHEMA_VERSION,
                )
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SyntheticBenchmarkComparison:
    per_candidate: tuple[dict[str, float], ...]
    summary: dict[str, dict[str, float]]

    @classmethod
    def from_vectors(
        cls,
        *,
        candidate_vectors: Sequence[Sequence[float]],
        seed: int,
        baseline_names: Sequence[str],
    ) -> "SyntheticBenchmarkComparison":
        per_candidate, summary = compare_to_baselines(
            candidate_vectors=candidate_vectors,
            seed=seed,
            baseline_names=baseline_names,
        )
        return cls(per_candidate=per_candidate, summary=summary)

    def as_dict(self) -> dict[str, Any]:
        return {"per_candidate": list(self.per_candidate), "summary": self.summary}


@dataclass(frozen=True)
class SyntheticBenchmarkRecord:
    candidate: Candidate
    response_points: tuple[float, ...]
    response_curve: tuple[float, ...]
    reduced_observables: tuple[tuple[str, float], ...]
    dataset_id: str
    pseudo_hdf5_ref: str
    completion_status: str
    completion_ratio: float
    completed_steps: int
    total_steps: int
    async_token: str | None
    baseline_distances: dict[str, float]
    family: str
    experiment: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_parameters": self.candidate.parameters,
            "response_points": list(self.response_points),
            "response_curve": list(self.response_curve),
            "reduced_observables": dict(self.reduced_observables),
            "dataset_id": self.dataset_id,
            "pseudo_hdf5_ref": self.pseudo_hdf5_ref,
            "completion_status": self.completion_status,
            "completion_ratio": self.completion_ratio,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "async_token": self.async_token,
            "baseline_distances": dict(self.baseline_distances),
            "family": self.family,
            "experiment": self.experiment,
        }


@dataclass(frozen=True)
class SyntheticBenchmarkResult:
    config: SyntheticBenchmarkConfig
    records: tuple[SyntheticBenchmarkRecord, ...]
    baseline_comparison: SyntheticBenchmarkComparison
    schema_version: str = ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_RESULT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": {
                "family": self.config.family,
                "experiment": self.config.experiment,
                "count": self.config.count,
                "seed": self.config.seed,
                "sequence_name": self.config.sequence_name,
                "candidate_prefix": self.config.candidate_prefix,
                "response_points": list(self.config.response_points),
                "completion_steps": self.config.completion_steps,
                "run_id": self.config.run_id,
                "baseline_names": list(self.config.baseline_names),
            },
            "candidate_count": len(self.records),
            "baseline_comparison": self.baseline_comparison.as_dict(),
            "records": [record.as_dict() for record in self.records],
        }


@dataclass(frozen=True)
class SyntheticBenchmarkArtifacts:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _artifact_directory(*, output_root: str | Path, run_id: str, iteration: int) -> Path:
    return Path(output_root) / run_id / "iterations" / f"iter_{iteration:04d}"


def generate_synthetic_benchmark(
    config: SyntheticBenchmarkConfig | Mapping[str, Any],
) -> SyntheticBenchmarkResult:
    normalized = (
        config if isinstance(config, SyntheticBenchmarkConfig) else SyntheticBenchmarkConfig.from_dict(config)
    )
    vectors = _BASELINE_GENERATORS[normalized.sequence_name](
        count=normalized.count,
        dimensions=len(_parameter_specs(normalized.family)),
        seed=normalized.seed,
    )

    records: list[SyntheticBenchmarkRecord] = []
    candidate_vectors: list[tuple[float, ...]] = []

    for sample_index, unit_samples in enumerate(vectors):
        unit_samples = tuple(float(item) for item in unit_samples)
        candidate_id = f"{normalized.candidate_prefix}-{sample_index:04d}"
        launch_payload = _build_launch_payload(
            family=normalized.family,
            experiment=normalized.experiment,
            unit_samples=unit_samples,
            sample_index=sample_index,
            sequence_name=normalized.sequence_name,
            run_id=normalized.run_id,
            seed=normalized.seed,
        )
        completion_status, completed_steps, total_steps, completion_ratio, async_token = _completion_state(
            index=sample_index,
            step_count=normalized.completion_steps,
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            parameters=launch_payload,
            metadata={
                "source": "active_learning_synthetic_benchmark",
                "candidate_id": candidate_id,
                "family": normalized.family,
                "experiment": normalized.experiment,
                "sequence": normalized.sequence_name,
                "seed": normalized.seed,
                "sample": {spec[1]: value for spec, value in zip(_parameter_specs(normalized.family), unit_samples, strict=False)},
            },
        )
        response_curve = _response_curve(
            points=normalized.response_points,
            family=normalized.family,
            unit_samples=unit_samples,
            completion_ratio=completion_ratio,
        )
        records.append(
            SyntheticBenchmarkRecord(
                candidate=candidate,
                response_points=normalized.response_points,
                response_curve=response_curve,
                reduced_observables=_reduced_observables(response_curve),
                dataset_id=_build_dataset_id(
                    family=normalized.family,
                    experiment=normalized.experiment,
                    run_id=normalized.run_id,
                    candidate_id=candidate_id,
                ),
                pseudo_hdf5_ref=_build_pseudo_hdf5_ref(
                    family=normalized.family,
                    run_id=normalized.run_id,
                    candidate_id=candidate_id,
                ),
                completion_status=completion_status,
                completion_ratio=completion_ratio,
                completed_steps=completed_steps,
                total_steps=total_steps,
                async_token=async_token,
                baseline_distances={},
                family=normalized.family,
                experiment=normalized.experiment,
            )
        )
        candidate_vectors.append(unit_samples)

    comparison = SyntheticBenchmarkComparison.from_vectors(
        candidate_vectors=candidate_vectors,
        seed=normalized.seed,
        baseline_names=normalized.baseline_names,
    )

    records = [
        SyntheticBenchmarkRecord(
            candidate=record.candidate,
            response_points=record.response_points,
            response_curve=record.response_curve,
            reduced_observables=record.reduced_observables,
            dataset_id=record.dataset_id,
            pseudo_hdf5_ref=record.pseudo_hdf5_ref,
            completion_status=record.completion_status,
            completion_ratio=record.completion_ratio,
            completed_steps=record.completed_steps,
            total_steps=record.total_steps,
            async_token=record.async_token,
            baseline_distances=comparison.per_candidate[index],
            family=record.family,
            experiment=record.experiment,
        )
        for index, record in enumerate(records)
    ]
    return SyntheticBenchmarkResult(
        config=normalized,
        records=tuple(records),
        baseline_comparison=comparison,
    )


def compare_baselines(
    *,
    candidate_vectors: Sequence[Sequence[float]],
    seed: int,
    baseline_names: Sequence[str],
) -> SyntheticBenchmarkComparison:
    return SyntheticBenchmarkComparison.from_vectors(
        candidate_vectors=candidate_vectors,
        seed=seed,
        baseline_names=baseline_names,
    )


def _plot_benchmark(path: Path, result: SyntheticBenchmarkResult, *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_build_fallback_png())
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_build_validation_png(result))
        return

    max_records = min(4, len(result.records))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axis_curves = axes[0]
    axis_baseline = axes[1]
    color_cycle = {
        "queued": "tab:gray",
        "partial": "tab:orange",
        "completed": "tab:blue",
    }

    for index, record in enumerate(result.records[:max_records]):
        color = color_cycle.get(record.completion_status, "tab:blue")
        axis_curves.plot(
            record.response_points,
            record.response_curve,
            marker="o",
            alpha=0.8,
            color=color,
            label=f"{record.candidate.candidate_id} [{record.completion_status}]",
        )
        x = [f"{index}-{name}" for name in record.baseline_distances]
        axis_baseline.bar(
            x,
            list(record.baseline_distances.values()),
            alpha=0.4,
            color=color,
        )

    axis_curves.set_title("Synthetic response curves")
    axis_curves.set_xlabel("Control fraction")
    axis_curves.set_ylabel("Response")
    axis_curves.grid(True)
    axis_curves.legend(fontsize=7)
    axis_baseline.set_title("Baseline distance (per candidate)")
    axis_baseline.set_xlabel("Candidate / baseline")
    axis_baseline.set_ylabel("L2 distance")
    axis_baseline.tick_params(axis="x", rotation=35)
    axis_baseline.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_DPI)
    plt.close(fig)


def _summary_report(result: SyntheticBenchmarkResult) -> dict[str, Any]:
    status_counter = Counter(item.completion_status for item in result.records)
    reduced_by_name: dict[str, dict[str, float]] = {}
    if result.records:
        first_observables = dict(result.records[0].reduced_observables)
        for name in first_observables:
            values = [float(dict(record.reduced_observables)[name]) for record in result.records]
            reduced_by_name[name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }

    return {
        "schema_version": ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "run_id": result.config.run_id,
        "family": result.config.family,
        "experiment": result.config.experiment,
        "record_count": len(result.records),
        "candidate_count": len(result.records),
        "sequence_name": result.config.sequence_name,
        "baseline_comparison": result.baseline_comparison.summary,
        "completion_status_counts": dict(status_counter),
        "completion_ratio_mean": sum(record.completion_ratio for record in result.records) / len(result.records),
        "mean_reduced_observables": reduced_by_name,
        "completion_status": [
            {
                "candidate_id": record.candidate.candidate_id,
                "status": record.completion_status,
                "completion_ratio": record.completion_ratio,
                "async_token": record.async_token,
            }
            for record in result.records
        ],
        "reduced_observables": [dict(record.reduced_observables) for record in result.records],
        "response_curves": [list(record.response_curve) for record in result.records],
    }


def write_synthetic_benchmark_artifacts(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int,
    result: SyntheticBenchmarkResult,
    include_plot: bool = True,
) -> SyntheticBenchmarkArtifacts:
    if not str(run_id).strip():
        raise ValueError("run_id must be a non-empty string.")
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer.")

    artifact_dir = _artifact_directory(
        output_root=output_root,
        run_id=_coerce_nonempty_text(run_id, field_name="run_id"),
        iteration=iteration,
    )
    manifest_path = artifact_dir / ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_MANIFEST_FILENAME
    report_path = artifact_dir / ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_REPORT_FILENAME
    plot_path = artifact_dir / ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_SIDECAR_FILENAME

    manifest = result.as_dict()
    report = _summary_report(result)
    report["candidate_ids"] = [record.candidate.candidate_id for record in result.records]
    report["candidate_hashes"] = [record.candidate.stable_hash() for record in result.records]
    report["family_distribution"] = {result.config.family: len(result.records)}
    report["experiment_distribution"] = {result.config.experiment: len(result.records)}
    report["plot"] = str(plot_path)

    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _plot_benchmark(plot_path, result, include_plot=include_plot)
    _write_json(plot_sidecar_path, report)

    return SyntheticBenchmarkArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        report=report,
    )


__all__ = [
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_ARTIFACT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_MANIFEST_FILENAME",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_DPI",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_FILENAME",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_PLOT_SIDECAR_FILENAME",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_REPORT_FILENAME",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_RESULT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SYNTHETIC_BENCHMARK_SCHEMA_VERSION",
    "SyntheticBenchmarkArtifacts",
    "SyntheticBenchmarkComparison",
    "SyntheticBenchmarkConfig",
    "SyntheticBenchmarkRecord",
    "SyntheticBenchmarkResult",
    "build_latin_hypercube_sequence",
    "build_random_sequence",
    "build_sobol_like_sequence",
    "compare_baselines",
    "compare_to_baselines",
    "generate_synthetic_benchmark",
    "write_synthetic_benchmark_artifacts",
]
