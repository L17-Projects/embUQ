"""NativeCuda Phase 2 profiling and threshold helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SETUP_BUCKETS = (
    "context_seconds",
    "compile_seconds",
    "module_load_seconds",
    "alloc_seconds",
    "h2d_seconds",
    "total_seconds",
)
BATCH_BUCKETS = (
    "log_prior_seconds",
    "flatten_seconds",
    "context_seconds",
    "alloc_seconds",
    "h2d_seconds",
    "launch_seconds",
    "compute_seconds",
    "d2h_seconds",
    "free_seconds",
    "host_reduce_seconds",
    "total_seconds",
)


@dataclass(frozen=True)
class NativeCudaPerformanceThresholds:
    """Fixed thresholds for one NativeCuda profile-harness run."""

    min_setup_count: int = 1
    min_batch_count: int = 1
    max_setup_total_seconds: float = 120.0
    max_setup_compile_seconds: float = 120.0
    max_batch_total_seconds: float = 10.0
    max_batch_alloc_seconds: float = 1.0
    max_batch_h2d_seconds: float = 1.0
    max_batch_compute_seconds: float = 5.0
    max_batch_d2h_seconds: float = 1.0
    max_batch_host_reduce_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.min_setup_count < 0:
            raise ValueError("min_setup_count must be non-negative.")
        if self.min_batch_count < 0:
            raise ValueError("min_batch_count must be non-negative.")
        for name in (
            "max_setup_total_seconds",
            "max_setup_compile_seconds",
            "max_batch_total_seconds",
            "max_batch_alloc_seconds",
            "max_batch_h2d_seconds",
            "max_batch_compute_seconds",
            "max_batch_d2h_seconds",
            "max_batch_host_reduce_seconds",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value.")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "min_setup_count": self.min_setup_count,
            "min_batch_count": self.min_batch_count,
            "max_setup_total_seconds": self.max_setup_total_seconds,
            "max_setup_compile_seconds": self.max_setup_compile_seconds,
            "max_batch_total_seconds": self.max_batch_total_seconds,
            "max_batch_alloc_seconds": self.max_batch_alloc_seconds,
            "max_batch_h2d_seconds": self.max_batch_h2d_seconds,
            "max_batch_compute_seconds": self.max_batch_compute_seconds,
            "max_batch_d2h_seconds": self.max_batch_d2h_seconds,
            "max_batch_host_reduce_seconds": self.max_batch_host_reduce_seconds,
        }


DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS = NativeCudaPerformanceThresholds()


@dataclass(frozen=True)
class NativeCudaProfileSummary:
    source: str | None
    setup_count: int
    batch_count: int
    setup_max_seconds: Mapping[str, float]
    setup_mean_seconds: Mapping[str, float]
    batch_max_seconds: Mapping[str, float]
    batch_mean_seconds: Mapping[str, float]
    batch_size_max: int | None
    parameter_count_max: int | None
    sub_problem_count_max: int | None
    dynamic_prior_count_max: int | None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "setup_count": self.setup_count,
            "batch_count": self.batch_count,
            "setup_max_seconds": dict(self.setup_max_seconds),
            "setup_mean_seconds": dict(self.setup_mean_seconds),
            "batch_max_seconds": dict(self.batch_max_seconds),
            "batch_mean_seconds": dict(self.batch_mean_seconds),
            "batch_size_max": self.batch_size_max,
            "parameter_count_max": self.parameter_count_max,
            "sub_problem_count_max": self.sub_problem_count_max,
            "dynamic_prior_count_max": self.dynamic_prior_count_max,
        }


@dataclass(frozen=True)
class NativeCudaPerformanceReport:
    passed: bool
    mismatches: tuple[str, ...]
    thresholds: NativeCudaPerformanceThresholds
    summary: NativeCudaProfileSummary

    def to_manifest(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "thresholds": self.thresholds.to_manifest(),
            "summary": self.summary.to_manifest(),
        }


def load_native_cuda_profile_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    profile_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        lines = profile_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read NativeCuda profile JSONL {profile_path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{profile_path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{profile_path}:{line_number}: profile record must be a JSON object.")
        _validate_profile_record(record, source=f"{profile_path}:{line_number}")
        records.append(record)
    return tuple(records)


def summarize_native_cuda_profile(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str | Path | None = None,
) -> NativeCudaProfileSummary:
    setup_records = [record for record in records if record.get("kind") == "native_cuda_setup"]
    batch_records = [record for record in records if record.get("kind") == "native_cuda_batch"]
    for index, record in enumerate(records):
        _validate_profile_record(record, source=f"record[{index}]")

    return NativeCudaProfileSummary(
        source=str(source) if source is not None else None,
        setup_count=len(setup_records),
        batch_count=len(batch_records),
        setup_max_seconds=_bucket_max(setup_records, SETUP_BUCKETS),
        setup_mean_seconds=_bucket_mean(setup_records, SETUP_BUCKETS),
        batch_max_seconds=_bucket_max(batch_records, BATCH_BUCKETS),
        batch_mean_seconds=_bucket_mean(batch_records, BATCH_BUCKETS),
        batch_size_max=_int_max(batch_records, "batch_size"),
        parameter_count_max=_int_max(batch_records, "parameter_count"),
        sub_problem_count_max=_int_max((*setup_records, *batch_records), "sub_problem_count"),
        dynamic_prior_count_max=_int_max((*setup_records, *batch_records), "dynamic_prior_count"),
    )


def check_native_cuda_profile(
    path: str | Path,
    *,
    thresholds: NativeCudaPerformanceThresholds = DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS,
) -> NativeCudaPerformanceReport:
    profile_path = Path(path)
    summary = summarize_native_cuda_profile(
        load_native_cuda_profile_records(profile_path),
        source=profile_path,
    )
    return compare_native_cuda_profile(summary, thresholds=thresholds)


def compare_native_cuda_profile(
    summary: NativeCudaProfileSummary,
    *,
    thresholds: NativeCudaPerformanceThresholds = DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS,
) -> NativeCudaPerformanceReport:
    mismatches: list[str] = []
    if summary.setup_count < thresholds.min_setup_count:
        mismatches.append(
            f"setup_count={summary.setup_count} below min_setup_count={thresholds.min_setup_count}"
        )
    if summary.batch_count < thresholds.min_batch_count:
        mismatches.append(
            f"batch_count={summary.batch_count} below min_batch_count={thresholds.min_batch_count}"
        )

    _check_bucket_threshold(
        mismatches,
        label="setup.total_seconds",
        actual=summary.setup_max_seconds.get("total_seconds", 0.0),
        threshold_name="max_setup_total_seconds",
        threshold=thresholds.max_setup_total_seconds,
    )
    _check_bucket_threshold(
        mismatches,
        label="setup.compile_seconds",
        actual=summary.setup_max_seconds.get("compile_seconds", 0.0),
        threshold_name="max_setup_compile_seconds",
        threshold=thresholds.max_setup_compile_seconds,
    )
    for bucket, threshold_name, threshold in (
        ("total_seconds", "max_batch_total_seconds", thresholds.max_batch_total_seconds),
        ("alloc_seconds", "max_batch_alloc_seconds", thresholds.max_batch_alloc_seconds),
        ("h2d_seconds", "max_batch_h2d_seconds", thresholds.max_batch_h2d_seconds),
        ("compute_seconds", "max_batch_compute_seconds", thresholds.max_batch_compute_seconds),
        ("d2h_seconds", "max_batch_d2h_seconds", thresholds.max_batch_d2h_seconds),
        (
            "host_reduce_seconds",
            "max_batch_host_reduce_seconds",
            thresholds.max_batch_host_reduce_seconds,
        ),
    ):
        _check_bucket_threshold(
            mismatches,
            label=f"batch.{bucket}",
            actual=summary.batch_max_seconds.get(bucket, 0.0),
            threshold_name=threshold_name,
            threshold=threshold,
        )

    return NativeCudaPerformanceReport(
        passed=not mismatches,
        mismatches=tuple(mismatches),
        thresholds=thresholds,
        summary=summary,
    )


def _validate_profile_record(record: Mapping[str, Any], *, source: str) -> None:
    kind = record.get("kind")
    if kind == "native_cuda_setup":
        buckets = SETUP_BUCKETS
    elif kind == "native_cuda_batch":
        buckets = BATCH_BUCKETS
    else:
        raise ValueError(f"{source}: unsupported NativeCuda profile record kind {kind!r}.")
    for bucket in buckets:
        _numeric_value(record, bucket, source=source)


def _numeric_value(record: Mapping[str, Any], name: str, *, source: str) -> float:
    if name not in record:
        raise ValueError(f"{source}: missing NativeCuda profile field {name!r}.")
    try:
        value = float(record[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: NativeCuda profile field {name!r} must be numeric.") from exc
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{source}: NativeCuda profile field {name!r} must be finite and non-negative.")
    return value


def _bucket_max(records: Sequence[Mapping[str, Any]], buckets: Iterable[str]) -> dict[str, float]:
    return {
        bucket: max((_numeric_value(record, bucket, source="profile") for record in records), default=0.0)
        for bucket in buckets
    }


def _bucket_mean(records: Sequence[Mapping[str, Any]], buckets: Iterable[str]) -> dict[str, float]:
    if not records:
        return {bucket: 0.0 for bucket in buckets}
    return {
        bucket: sum(_numeric_value(record, bucket, source="profile") for record in records) / len(records)
        for bucket in buckets
    }


def _int_max(records: Sequence[Mapping[str, Any]], field: str) -> int | None:
    values: list[int] = []
    for record in records:
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, bool):
            raise ValueError(f"NativeCuda profile field {field!r} must be an integer.")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NativeCuda profile field {field!r} must be an integer.") from exc
        if number < 0:
            raise ValueError(f"NativeCuda profile field {field!r} must be non-negative.")
        values.append(number)
    return max(values) if values else None


def _check_bucket_threshold(
    mismatches: list[str],
    *,
    label: str,
    actual: float,
    threshold_name: str,
    threshold: float,
) -> None:
    if actual > threshold:
        mismatches.append(f"{label} max {actual:.6g}s exceeds {threshold_name}={threshold:.6g}s")


__all__ = [
    "BATCH_BUCKETS",
    "DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS",
    "NativeCudaPerformanceReport",
    "NativeCudaPerformanceThresholds",
    "NativeCudaProfileSummary",
    "SETUP_BUCKETS",
    "check_native_cuda_profile",
    "compare_native_cuda_profile",
    "load_native_cuda_profile_records",
    "summarize_native_cuda_profile",
]
