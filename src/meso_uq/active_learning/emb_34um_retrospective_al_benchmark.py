from __future__ import annotations

"""Retrospective EMB 3.4um active-design benchmark against LHS baselines."""

import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_retrospective_al_benchmark.v1"
)
EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_MANIFEST_FILENAME = (
    "emb_34um_retrospective_al_vs_lhs_manifest.json"
)
EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SUMMARY_CSV_FILENAME = (
    "emb_34um_retrospective_al_vs_lhs_summary.csv"
)
EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SELECTED_CSV_FILENAME = (
    "emb_34um_retrospective_al_selected_candidates.csv"
)
EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_FILENAME = "emb_34um_retrospective_al_vs_lhs.png"
EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_SIDECAR_FILENAME = (
    "emb_34um_retrospective_al_vs_lhs.png.json"
)


@dataclass(frozen=True)
class Emb34umRetrospectiveAlBenchmarkArtifacts:
    artifact_dir: Path
    manifest_path: Path
    summary_csv_path: Path
    selected_candidates_csv_path: Path
    plot_path: Path
    plot_sidecar_path: Path
    manifest: dict[str, Any]
    summary_rows: tuple[dict[str, Any], ...]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_samples_all(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError(f"Expected a 2D samples_all.dat table with at least 12 columns at {path!s}.")
    remaining = data.shape[1] - 8
    if remaining <= 0 or remaining % 2 != 0:
        raise ValueError(f"Expected row layout parameters[8] outputs[N] forces[N] at {path!s}.")
    force_count = remaining // 2
    parameters = data[:, :8]
    outputs = data[:, 8 : 8 + force_count]
    forces = data[:, 8 + force_count : 8 + 2 * force_count]
    if not np.allclose(forces, forces[0]):
        raise ValueError("Force grid must be fixed across all EMB 3.4um rows.")
    if np.any(parameters[:, 0] <= 0.0) or np.any(parameters[:, 2] <= 0.0):
        raise ValueError("Yt and kb must be positive for log-space benchmarking.")
    diameter = 2.0 * parameters[:, 7:8]
    displacements = diameter - outputs
    features = np.column_stack([np.log10(parameters[:, 0]), np.log10(parameters[:, 2])])
    return parameters, features, displacements, forces[0]


def _fit_feature_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    span = np.maximum(max_values - min_values, 1.0e-12)
    return min_values, span


def _apply_feature_scaler(features: np.ndarray, min_values: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (features - min_values) / span


def _predict_knn(
    *,
    train_features: np.ndarray,
    train_curves: np.ndarray,
    validation_features: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    if len(train_features) == 0:
        raise ValueError("At least one training curve is required.")
    k = min(max(1, int(neighbors)), len(train_features))
    distances = np.linalg.norm(validation_features[:, None, :] - train_features[None, :, :], axis=2)
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    nearest = np.take_along_axis(distances, indices, axis=1)
    weights = 1.0 / (nearest + 1.0e-6)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, train_curves[indices])


def _median_curve_rel_l2_pct(predicted: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.maximum(np.linalg.norm(reference, axis=1), 1.0e-12)
    values = 100.0 * np.linalg.norm(predicted - reference, axis=1) / denominator
    return float(np.median(values))


def _maximin_pick(
    *,
    pool_features: np.ndarray,
    remaining: set[int],
    selected: list[int],
    count: int,
) -> list[int]:
    picked: list[int] = []
    if not selected and remaining:
        candidates = np.array(sorted(remaining), dtype=int)
        distances = np.linalg.norm(pool_features[candidates] - 0.5, axis=1)
        first = int(candidates[int(np.argmax(distances))])
        picked.append(first)
        selected.append(first)
        remaining.remove(first)

    while len(picked) < count and remaining:
        candidates = np.array(sorted(remaining), dtype=int)
        selected_features = pool_features[np.array(selected, dtype=int)]
        distances = np.linalg.norm(pool_features[candidates, None, :] - selected_features[None, :, :], axis=2)
        nearest = np.min(distances, axis=1)
        candidate = int(candidates[int(np.argmax(nearest))])
        picked.append(candidate)
        selected.append(candidate)
        remaining.remove(candidate)
    return picked


def _lhs_nearest_candidates(*, pool_features: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    design = np.empty((count, pool_features.shape[1]), dtype=float)
    for dimension in range(pool_features.shape[1]):
        design[:, dimension] = (rng.permutation(count) + rng.random(count)) / count

    selected: list[int] = []
    remaining = set(range(len(pool_features)))
    for point in design:
        candidates = np.array(sorted(remaining), dtype=int)
        distances = np.linalg.norm(pool_features[candidates] - point, axis=1)
        candidate = int(candidates[int(np.argmin(distances))])
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)


def _fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 16
    height = 16
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((246, 246, 246, 255) if (x + y) % 2 else (0, 107, 95, 255))
        rows.append(bytes(row))
    return signature + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + _png_chunk(b"IEND", b"")


def _write_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, include_plot: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not include_plot or not rows:
        path.write_bytes(_fallback_png())
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.write_bytes(_fallback_png())
        return

    x = np.array([int(row["curve_count"]) for row in rows], dtype=int)
    al_values = np.array([float(row["al_median_curve_rel_l2_pct"]) for row in rows], dtype=float)
    lhs_values = np.array([float(row["lhs_median_curve_rel_l2_pct"]) for row in rows], dtype=float)
    lhs_p25 = np.array([float(row["lhs_p25_curve_rel_l2_pct"]) for row in rows], dtype=float)
    lhs_p75 = np.array([float(row["lhs_p75_curve_rel_l2_pct"]) for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.fill_between(x, lhs_p25, lhs_p75, color="#c8c8c8", alpha=0.6, label="LHS interquartile range")
    axis.plot(x, lhs_values, marker="s", color="#595959", label="LHS median")
    axis.plot(x, al_values, marker="o", color="#006b5f", linewidth=2.2, label="Active maximin")
    axis.axvline(90, color="#b04a00", linestyle="--", linewidth=1.0, label="90-curve gate")
    axis.set_xlabel("Executed curves")
    axis.set_ylabel("median_curve_rel_l2_pct")
    axis.set_title("EMB 3.4um retrospective active learning vs LHS")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_emb_34um_retrospective_al_benchmark(
    *,
    samples_all_path: str | Path,
    candidate_pool_size: int = 500,
    max_executed_curves: int = 300,
    round_size: int = 30,
    validation_count: int = 3000,
    lhs_replicates: int = 50,
    seed: int = 1,
    neighbors: int = 3,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if candidate_pool_size <= 0 or candidate_pool_size > 500:
        raise ValueError("candidate_pool_size must be in [1, 500].")
    if max_executed_curves <= 0 or max_executed_curves > candidate_pool_size:
        raise ValueError("max_executed_curves must be in [1, candidate_pool_size].")
    if round_size <= 0:
        raise ValueError("round_size must be positive.")
    if lhs_replicates <= 0:
        raise ValueError("lhs_replicates must be positive.")

    samples_path = Path(samples_all_path)
    parameters, features, curves, force_grid = _load_samples_all(samples_path)
    if validation_count <= 0:
        raise ValueError("validation_count must be positive.")
    if validation_count + candidate_pool_size > len(features):
        raise ValueError("validation_count + candidate_pool_size exceeds available rows.")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(features))
    validation_indices = permutation[:validation_count]
    pool_indices = permutation[validation_count : validation_count + candidate_pool_size]
    pool_raw_features = features[pool_indices]
    validation_raw_features = features[validation_indices]
    feature_min, feature_span = _fit_feature_scaler(pool_raw_features)
    pool_features = _apply_feature_scaler(pool_raw_features, feature_min, feature_span)
    validation_features = _apply_feature_scaler(validation_raw_features, feature_min, feature_span)
    validation_curves = curves[validation_indices]
    pool_curves = curves[pool_indices]

    def score(selection: Sequence[int]) -> float:
        predicted = _predict_knn(
            train_features=pool_features[np.array(selection, dtype=int)],
            train_curves=pool_curves[np.array(selection, dtype=int)],
            validation_features=validation_features,
            neighbors=neighbors,
        )
        return _median_curve_rel_l2_pct(predicted, validation_curves)

    selected: list[int] = []
    remaining = set(range(candidate_pool_size))
    counts: list[int] = []
    al_metrics: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    while len(selected) < max_executed_curves:
        picked = _maximin_pick(
            pool_features=pool_features,
            remaining=remaining,
            selected=selected,
            count=min(round_size, max_executed_curves - len(selected)),
        )
        round_index = int(math.ceil(len(selected) / round_size))
        for pool_position in picked:
            source_row_index = int(pool_indices[pool_position])
            selected_rows.append(
                {
                    "strategy": "active_maximin",
                    "round": round_index,
                    "order": len(selected_rows) + 1,
                    "pool_position": int(pool_position),
                    "source_row_index": source_row_index,
                    "Yt": float(parameters[source_row_index, 0]),
                    "kb": float(parameters[source_row_index, 2]),
                }
            )
        counts.append(len(selected))
        al_metrics.append(score(selected))

    lhs_metrics = []
    for replicate in range(lhs_replicates):
        lhs_selection = _lhs_nearest_candidates(
            pool_features=pool_features,
            count=max_executed_curves,
            seed=seed + 1000 + replicate,
        )
        lhs_metrics.append([score(lhs_selection[:count]) for count in counts])
    lhs_array = np.asarray(lhs_metrics, dtype=float)
    lhs_median = np.median(lhs_array, axis=0)
    lhs_p25 = np.percentile(lhs_array, 25, axis=0)
    lhs_p75 = np.percentile(lhs_array, 75, axis=0)

    summary_rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        summary_rows.append(
            {
                "curve_count": int(count),
                "al_median_curve_rel_l2_pct": float(al_metrics[index]),
                "lhs_median_curve_rel_l2_pct": float(lhs_median[index]),
                "lhs_p25_curve_rel_l2_pct": float(lhs_p25[index]),
                "lhs_p75_curve_rel_l2_pct": float(lhs_p75[index]),
                "al_minus_lhs_median": float(al_metrics[index] - lhs_median[index]),
                "al_improved_vs_lhs_median": bool(al_metrics[index] < lhs_median[index]),
            }
        )

    manifest = {
        "schema_version": EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SCHEMA_VERSION,
        "status": "ready",
        "benchmark_type": "retrospective_existing_dpd_oracle_table",
        "samples_all_path": str(samples_path),
        "candidate_pool_size": int(candidate_pool_size),
        "max_executed_curves": int(max_executed_curves),
        "round_size": int(round_size),
        "validation_count": int(validation_count),
        "lhs_replicates": int(lhs_replicates),
        "seed": int(seed),
        "neighbors": int(neighbors),
        "force_grid": [float(item) for item in force_grid],
        "primary_metric": "median_curve_rel_l2_pct",
        "active_design_policy": "maximin_farthest_point_in_log10_Yt_log10_kb",
        "lhs_policy": "latin_hypercube_nearest_available_candidate",
        "surrogate": "deterministic_distance_weighted_knn",
        "feature_scaling": {
            "fit_scope": "candidate_pool",
            "feature_columns": ["log10_Yt", "log10_kb"],
            "min": [float(item) for item in feature_min],
            "span": [float(item) for item in feature_span],
        },
        "notes": [
            "Uses existing dense EMB 3.4um DPD rows as a retrospective oracle.",
            "This benchmark validates the active-design policy before fresh production DPD execution.",
            "It is not a substitute for the fresh DPD production array and ingestion gate.",
        ],
        "summary_rows": summary_rows,
        "final_result": summary_rows[-1],
    }
    return manifest, tuple(summary_rows), tuple(selected_rows)


def write_emb_34um_retrospective_al_benchmark_artifacts(
    *,
    samples_all_path: str | Path,
    output_root: str | Path,
    candidate_pool_size: int = 500,
    max_executed_curves: int = 300,
    round_size: int = 30,
    validation_count: int = 3000,
    lhs_replicates: int = 50,
    seed: int = 1,
    neighbors: int = 3,
    include_plot: bool = True,
) -> Emb34umRetrospectiveAlBenchmarkArtifacts:
    artifact_dir = Path(output_root)
    manifest_path = artifact_dir / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_MANIFEST_FILENAME
    summary_csv_path = artifact_dir / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SUMMARY_CSV_FILENAME
    selected_csv_path = artifact_dir / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SELECTED_CSV_FILENAME
    plot_path = artifact_dir / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_FILENAME
    plot_sidecar_path = artifact_dir / EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_SIDECAR_FILENAME

    manifest, summary_rows, selected_rows = build_emb_34um_retrospective_al_benchmark(
        samples_all_path=samples_all_path,
        candidate_pool_size=candidate_pool_size,
        max_executed_curves=max_executed_curves,
        round_size=round_size,
        validation_count=validation_count,
        lhs_replicates=lhs_replicates,
        seed=seed,
        neighbors=neighbors,
    )
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(selected_csv_path, selected_rows)
    _write_plot(plot_path, summary_rows, include_plot=include_plot)
    manifest.update(
        {
            "artifact_root": str(artifact_dir),
            "plot_path": str(plot_path),
            "plot_sidecar_path": str(plot_sidecar_path),
            "summary_csv": str(summary_csv_path),
            "selected_candidates_csv": str(selected_csv_path),
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(plot_sidecar_path, {"plot_path": str(plot_path), **manifest})
    return Emb34umRetrospectiveAlBenchmarkArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_csv_path=summary_csv_path,
        selected_candidates_csv_path=selected_csv_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        manifest=manifest,
        summary_rows=summary_rows,
    )


__all__ = [
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_MANIFEST_FILENAME",
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_FILENAME",
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SCHEMA_VERSION",
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SELECTED_CSV_FILENAME",
    "EMB_34UM_RETROSPECTIVE_AL_BENCHMARK_SUMMARY_CSV_FILENAME",
    "Emb34umRetrospectiveAlBenchmarkArtifacts",
    "build_emb_34um_retrospective_al_benchmark",
    "write_emb_34um_retrospective_al_benchmark_artifacts",
]
