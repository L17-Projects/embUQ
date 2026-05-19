"""Artifact helpers for Active Learning iterations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import struct
import zlib
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import ActiveLearningIterationLineage


ACTIVE_LEARNING_ITERATION_MANIFEST_SCHEMA_VERSION = "meso_uq.active_learning.iteration_manifest.v1"
ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME = "iteration_manifest.json"
ACTIVE_LEARNING_STAGE_REPORT_FILENAME = "stage_validation.json"
ACTIVE_LEARNING_PLOT_FILENAME = "validation_plot.png"
ACTIVE_LEARNING_PLOT_DPI = 90


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _json_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default)


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _artifact_iteration_directory(
    output_root: str | Path,
    run_id: str,
    iteration: int,
) -> Path:
    return Path(output_root) / run_id / "iterations" / f"iter_{iteration:04d}"


def _to_float(value: Any, *, context: str) -> float:
    try:
        value_as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric.") from exc
    if not value_as_float == value_as_float:
        raise ValueError(f"{context} must be finite.")
    return value_as_float


def _coerce_failures(failures: Sequence[object], iteration: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for failure in failures:
        if not hasattr(failure, "as_dict"):
            raise TypeError(f"Failure payload for iteration {iteration} must expose as_dict().")
        items.append(failure.as_dict())  # type: ignore[attr-defined]
    return items


@dataclass(frozen=True)
class IterationArtifacts:
    iteration_dir: Path
    manifest_path: Path
    sidecar_path: Path
    plot_path: Path
    manifest: dict[str, Any]
    sidecar: dict[str, Any]


def _write_plot(path: Path, payload: Mapping[str, Any], *, include_plot: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot:
        path.write_bytes(_FALLBACK_PNG)
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_FALLBACK_PNG)
        return
    selected_scores = list(payload.get("selected_scores", ()))
    y_values = [_to_float(value, context="validation score") for value in selected_scores]
    if not y_values:
        y_values = [0.0]
    x_values = list(range(1, len(y_values) + 1))
    fig, axis = plt.subplots()
    axis.plot(x_values, y_values)
    axis.set_title("Active-learning iteration validation")
    axis.set_xlabel("Selection rank")
    axis.set_ylabel("Validation score")
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=ACTIVE_LEARNING_PLOT_DPI)
    plt.close(fig)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_json_payload(_normalize_payload(payload)), encoding="utf-8")


def write_iteration_artifacts(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int,
    lineage: ActiveLearningIterationLineage,
    stage_failures: Sequence[object],
    stage_scores: Mapping[str, float],
    selected_candidate_ids: Sequence[str],
    include_plot: bool = True,
) -> IterationArtifacts:
    if not run_id:
        raise ValueError("run_id must be a non-empty string.")
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")

    iteration_dir = _artifact_iteration_directory(output_root, run_id, iteration)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "schema_version": ACTIVE_LEARNING_ITERATION_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": iteration,
        "iteration_id": lineage.iteration_id,
        "parent_iteration_id": lineage.parent_iteration_id,
        "candidate_count": len(lineage.selected_candidate_hashes),
        "request_count": lineage.request_count,
        "success_count": lineage.success_count,
        "failure_count": lineage.failure_count,
        "completion_reason": lineage.completion_reason,
    }

    sidecar_payload = {
        "schema_version": ACTIVE_LEARNING_ITERATION_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": iteration,
        "iteration_id": lineage.iteration_id,
        "selected_candidates": list(selected_candidate_ids),
        "stage_scores": dict(stage_scores),
        "selected_candidate_hashes": list(lineage.selected_candidate_hashes),
        "selected_scores": [stage_scores.get(candidate_id, 0.0) for candidate_id in selected_candidate_ids],
        "iteration_failures": _coerce_failures(stage_failures, iteration=iteration),
    }

    manifest_path = iteration_dir / ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME
    sidecar_path = iteration_dir / ACTIVE_LEARNING_STAGE_REPORT_FILENAME
    plot_path = iteration_dir / ACTIVE_LEARNING_PLOT_FILENAME

    _write_json(manifest_path, manifest_payload)
    _write_json(sidecar_path, sidecar_payload)
    _write_plot(plot_path, sidecar_payload, include_plot=include_plot)

    return IterationArtifacts(
        iteration_dir=iteration_dir,
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
        plot_path=plot_path,
        manifest=manifest_payload,
        sidecar=sidecar_payload,
    )


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _build_fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_pixel = b"\x00\x00\x00\x00\x00"
    idat_data = zlib.compress(raw_pixel)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", idat_data)
        + _png_chunk(b"IEND", b"")
    )


_FALLBACK_PNG = _build_fallback_png()

__all__ = [
    "ACTIVE_LEARNING_ITERATION_MANIFEST_FILENAME",
    "ACTIVE_LEARNING_ITERATION_MANIFEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_PLOT_FILENAME",
    "ACTIVE_LEARNING_STAGE_REPORT_FILENAME",
    "IterationArtifacts",
    "write_iteration_artifacts",
]
