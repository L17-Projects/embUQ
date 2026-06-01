#!/usr/bin/env python3
from __future__ import annotations

"""Train, score, and select EMB 3.4um causal DNN candidates."""

import argparse
import json
import math
import os
import struct
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to locate repository root.")


def _bootstrap_paths() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_paths()

import numpy as np

from meso_uq.active_learning.emb_34um_dnn_acquisition import (
    EMB_34UM_DNN_ACQUISITION_SCORE_MODES,
    compute_disagreement_distribution,
    score_candidate_pool,
    select_greedy_diversity,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
    is_dnn_causal_low_corner_excluded,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
    validate_dnn_causal_timing_canary,
)
from meso_uq.active_learning.emb_34um_dnn_surrogate import (
    EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT,
    EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE,
    DnnSurrogateLongRow,
    Emb34umDnnSurrogateFit,
    convert_completed_curves_to_long_rows,
    train_emb_34um_dnn_surrogate_ensemble,
)

EMB_34UM_DNN_TRAIN_SCORE_SELECT_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_train_score_select.v1"
)
EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME = "emb_34um_dnn_train_score_select_manifest.json"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_REPORT_FILENAME = "emb_34um_dnn_train_score_select_report.json"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_REPORT_FILENAME = "emb_34um_dnn_train_score_select_timing_canary.json"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_SELECTION_PLOT_FILENAME = "emb_34um_dnn_train_score_select_candidate_heatmap.png"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_DISTRIBUTION_PLOT_FILENAME = "emb_34um_dnn_train_score_select_disagreement_hist.png"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_LOSS_PLOT_FILENAME = "emb_34um_dnn_train_score_select_member_losses.png"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_PLOT_FILENAME = "emb_34um_dnn_train_score_select_timing_canary.png"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_KA_RADP_PLOT_FILENAME = "emb_34um_dnn_train_score_select_ka_radp.png"
EMB_34UM_DNN_TRAIN_SCORE_SELECT_KB_SHELL_TH_PLOT_FILENAME = (
    "emb_34um_dnn_train_score_select_kb_shell_th.png"
)
EMB_34UM_DNN_TRAIN_SCORE_SELECT_RADP_SHELL_TH_PLOT_FILENAME = (
    "emb_34um_dnn_train_score_select_radp_shell_th.png"
)
EMB_34UM_DNN_TRAIN_SCORE_SELECT_CURVE_ERROR_PLOT_FILENAME = (
    "emb_34um_dnn_train_score_select_curve_error_risk.png"
)


def _load_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be a JSON object.")
        normalized.append(item)
    return normalized


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


def _coerce_bounded_positive(
    value: object,
    *,
    key: str,
    source: str,
) -> float:
    low, high = EMB_34UM_DNN_CAUSAL_BOUNDS[key]
    number = _coerce_positive_float(value, label=f"{source}.{key}")
    if not (low <= number <= high):
        raise ValueError(f"{source}.{key} must be within [{low}, {high}].")
    return number


def _coerce_candidate_space(value: str | None) -> str:
    normalized = "d4" if value is None else str(value).strip().lower()
    if normalized not in {"d2", "d4"}:
        raise ValueError("--candidate-space must be either d2 or d4.")
    return normalized


def _candidate_param(row: Mapping[str, Any], key: str, *, index: int, source: str, default: object = None) -> float:
    value = None
    parameters = row.get("parameters")
    if isinstance(parameters, Mapping) and key in parameters:
        value = parameters.get(key)
    if value is None and default is not None:
        value = default
    if value is None and key in row:
        value = row.get(key)
    if value is None:
        raise ValueError(f"{source} is missing {key!r}.")
    return _coerce_float(value, label=f"{source}.{key}") if key in ("ka", "kb", "radp", "shell_th") else float(value)


def _coerce_raw_candidate_point(
    row: Mapping[str, Any],
    *,
    index: int,
    candidate_space: str,
) -> tuple[float, ...]:
    ka = _coerce_positive_float(_candidate_param(row, "ka", index=index, source="candidate", default=row.get("Yt", None)), label=f"candidate_records[{index}].ka")
    kb = _coerce_positive_float(_candidate_param(row, "kb", index=index, source="candidate", default=row.get("kb_scale", row.get("mu", None))), label=f"candidate_records[{index}].kb")
    if candidate_space == "d2":
        return (ka, kb)
    radp = _coerce_bounded_positive(
        _candidate_param(row, "radp", index=index, source="candidate"),
        key="radp",
        source=f"candidate_records[{index}]",
    )
    shell_th = _coerce_bounded_positive(
        _candidate_param(row, "shell_th", index=index, source="candidate"),
        key="shell_th",
        source=f"candidate_records[{index}]",
    )
    return (ka, kb, radp, shell_th)


def _normalize_candidate_point_values(
    point: tuple[float, ...],
    *,
    candidate_space: str,
) -> tuple[float, ...]:
    ka = math.log10(point[0])
    kb = math.log10(point[1])
    if candidate_space == "d2":
        return (ka, kb)
    return (
        ka,
        kb,
        (point[2] - EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0])
        / (EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][1] - EMB_34UM_DNN_CAUSAL_BOUNDS["radp"][0]),
        (point[3] - EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0])
        / (EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][1] - EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"][0]),
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


def _coerce_existing_points(
    raw: str | None,
    *,
    candidate_space: str,
) -> tuple[tuple[float, ...], ...]:
    if raw is None:
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("--existing-points must decode to a JSON list.")
    points: list[tuple[float, ...]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"existing_points[{index}] must be a list or tuple.")
        if candidate_space == "d2":
            if len(item) != 2:
                raise ValueError(f"existing_points[{index}] must be a [ka, kb] pair.")
            ka = _coerce_positive_float(item[0], label=f"existing_points[{index}][0]")
            kb = _coerce_positive_float(item[1], label=f"existing_points[{index}][1]")
            points.append((ka, kb))
            continue
        if len(item) != 4:
            raise ValueError(f"existing_points[{index}] must be a [ka, kb, radp, shell_th] tuple in d4.")
        ka = _coerce_positive_float(item[0], label=f"existing_points[{index}][0]")
        kb = _coerce_positive_float(item[1], label=f"existing_points[{index}][1]")
        radp = _coerce_bounded_positive(
            item[2],
            key="radp",
            source=f"existing_points[{index}]",
        )
        shell_th = _coerce_bounded_positive(
            item[3],
            key="shell_th",
            source=f"existing_points[{index}]",
        )
        points.append((ka, kb, radp, shell_th))
    return tuple(points)


def _existing_points_from_completed_rows(
    completed_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_space: str,
) -> tuple[tuple[float, ...], ...]:
    points: list[tuple[float, ...]] = []
    for index, row in enumerate(completed_rows, start=1):
        ka, kb, *rest = _coerce_raw_candidate_point(row, index=index, candidate_space=candidate_space)
        point: tuple[float, ...]
        if candidate_space == "d2":
            point = (ka, kb)
        else:
            if not rest:
                raise ValueError(f"completed_rows[{index}].parameters must contain radp and shell_th for d4.")
            point = (ka, kb, rest[0], rest[1])
        points.append(point)
    return tuple(points)


def _point_key(point: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in point)


def _exclude_existing_points_from_candidate_pool(
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    existing_points: Sequence[tuple[float, ...]],
    candidate_space: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    existing_keys = {_point_key(point) for point in existing_points}
    if not existing_keys:
        return tuple(dict(item) for item in candidate_pool), ()

    filtered: list[dict[str, Any]] = []
    skipped_ids: list[str] = []
    for index, raw_row in enumerate(candidate_pool, start=1):
        row = dict(raw_row)
        point = _coerce_raw_candidate_point(row, index=index, candidate_space=candidate_space)
        key = _point_key(point)
        if key in existing_keys:
            skipped_ids.append(str(row.get("candidate_id", f"candidate-{index:04d}")))
            continue
        filtered.append(row)
    return tuple(filtered), tuple(skipped_ids)


def _validate_candidate_pool_exclusion(candidate_pool: Sequence[Mapping[str, Any]]) -> None:
    blocked = [
        str(row.get("candidate_id", index))
        for index, row in enumerate(candidate_pool, start=1)
        if is_dnn_causal_low_corner_excluded(
            row.get("ka", row.get("parameters", {}).get("ka")),
            row.get("kb", row.get("parameters", {}).get("kb")),
            row.get("radp", row.get("parameters", {}).get("radp")),
            row.get("shell_th", row.get("parameters", {}).get("shell_th")),
        )
    ]
    if blocked:
        raise ValueError(
            "candidate_pool contains EMB 3.4um runtime-risk timeout candidates: "
            + ", ".join(blocked[:10])
        )


def _coerce_force_grid_arg(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return validate_dnn_causal_force_grid(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    return validate_dnn_causal_force_grid(json.loads(raw))


def _write_placeholder_png(path: Path, *, width: int = 8, height: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + bytes((231, 236, 239) * width)
    raw = row * height

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw)),
            _chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _plot_candidate_heatmap(
    path: Path,
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scored:
        _write_placeholder_png(path)
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    ka = [10.0 ** float(item["candidate_log10_point"][0]) for item in scored]
    kb = [10.0 ** float(item["candidate_log10_point"][1]) for item in scored]
    scores = [float(item.get("acquisition_score", 0.0)) for item in scored]

    figure, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(ka, kb, c=scores, s=28, cmap="viridis")
    if selected:
        selected_ka = [10.0 ** float(item["candidate_log10_point"][0]) for item in selected]
        selected_kb = [10.0 ** float(item["candidate_log10_point"][1]) for item in selected]
        axis.scatter(
            selected_ka,
            selected_kb,
            edgecolors="#d62728",
            facecolors="none",
            s=90,
            linewidth=1.2,
        )
    figure.colorbar(scatter, ax=axis, label="acquisition score")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    axis.set_title("DNN candidate score heatmap")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_curve_error_risk(
    path: Path,
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scored:
        _write_placeholder_png(path)
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    ka = [10.0 ** float(item["candidate_log10_point"][0]) for item in scored]
    kb = [10.0 ** float(item["candidate_log10_point"][1]) for item in scored]
    risk = [float(item.get("predicted_curve_relative_l2_error", 0.0)) for item in scored]

    figure, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(ka, kb, c=risk, s=28, cmap="magma")
    if selected:
        selected_ka = [10.0 ** float(item["candidate_log10_point"][0]) for item in selected]
        selected_kb = [10.0 ** float(item["candidate_log10_point"][1]) for item in selected]
        axis.scatter(
            selected_ka,
            selected_kb,
            edgecolors="#2ca02c",
            facecolors="none",
            s=90,
            linewidth=1.2,
        )
    figure.colorbar(scatter, ax=axis, label="predicted curve relative L2 error")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("ka")
    axis.set_ylabel("kb")
    axis.set_title("DNN predicted curve-error risk")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _extract_candidate_parameter_value(
    row: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> float:
    parameters = row.get("parameters")
    if isinstance(parameters, Mapping) and key in parameters:
        return _coerce_float(parameters[key], label=label)
    if key in row:
        return _coerce_float(row[key], label=label)
    raise ValueError(f"{label} is missing.")


def _plot_candidate_pair_scatter(
    path: Path,
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    x_transform=lambda value: value,
    y_transform=lambda value: value,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scored:
        _write_placeholder_png(path)
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    xs = [x_transform(_extract_candidate_parameter_value(item, x_key, label=f"{x_key}")) for item in scored]
    ys = [y_transform(_extract_candidate_parameter_value(item, y_key, label=f"{y_key}")) for item in scored]
    scores = [float(item.get("acquisition_score", 0.0)) for item in scored]

    figure, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(xs, ys, c=scores, s=28, cmap="viridis")
    if selected:
        selected_x = [x_transform(_extract_candidate_parameter_value(item, x_key, label=f"{x_key}")) for item in selected]
        selected_y = [y_transform(_extract_candidate_parameter_value(item, y_key, label=f"{y_key}")) for item in selected]
        axis.scatter(
            selected_x,
            selected_y,
            edgecolors="#d62728",
            facecolors="none",
            s=90,
            linewidth=1.2,
        )
    figure.colorbar(scatter, ax=axis, label="acquisition score")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title("DNN candidate score pair")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_distribution(path: Path, selected: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = compute_disagreement_distribution(selected)
    if not values:
        _write_placeholder_png(path)
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(values, bins=min(15, max(3, len(values))), color="#1f77b4", alpha=0.85)
    axis.set_xlabel("ensemble disagreement")
    axis.set_ylabel("count")
    axis.set_title("Selected disagreement distribution")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_losses(path: Path, fit: Emb34umDnnSurrogateFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fit.training_history:
        _write_placeholder_png(path)
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    figure, axis = plt.subplots(figsize=(7, 4))
    plotted = False
    for member in fit.training_history:
        train_history = member.get("train_loss_history", ())
        validation_history = member.get("validation_loss_history", ())
        if train_history:
            axis.plot(train_history, linewidth=0.9, alpha=0.75)
            plotted = True
        if validation_history:
            axis.plot(validation_history, linewidth=0.9, alpha=0.75, linestyle="--")
            plotted = True
    if not plotted:
        plt.close(figure)
        _write_placeholder_png(path)
        return
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Per-member train and validation loss")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_timing_canary(path: Path, *, train_seconds: float, score_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(["train", "score"], [float(train_seconds), float(score_seconds)], color=["#4c78a8", "#f58518"])
    axis.set_ylabel("seconds")
    axis.set_title("DNN timing canary")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-rows", required=True, help="Path to JSON list of completed DPD rows.")
    parser.add_argument("--candidate-pool", required=True, help="Path to JSON list of candidate ka/kb rows.")
    parser.add_argument("--output-root", required=True, help="Directory for train/score/select artifacts.")
    parser.add_argument("--ensemble-size", type=int, default=EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE)
    parser.add_argument("--ensemble-seed", action="append", default=[], type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--architecture", default=EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--diversity-weight", type=float, default=EMB_34UM_DNN_SURROGATE_DIVERSITY_WEIGHT)
    parser.add_argument(
        "--acquisition-score-mode",
        default="disagreement",
        choices=EMB_34UM_DNN_ACQUISITION_SCORE_MODES,
        help="Candidate base score: legacy ensemble disagreement or full-curve relative-L2 risk.",
    )
    parser.add_argument("--candidate-space", default="d4", choices=("d2", "d4"))
    parser.add_argument("--force-grid", default=None, help="Optional JSON list override for the force grid.")
    parser.add_argument("--timing-canary", action="store_true")
    parser.add_argument("--timing-canary-point-count", type=int, default=EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT)
    parser.add_argument("--timing-canary-override-reason", default="")
    parser.add_argument("--existing-points", default=None, help="JSON list of existing D2 [ka, kb] or D4 [ka, kb, radp, shell_th] points.")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Skip Torch training and use deterministic synthetic scoring.")
    return parser


def _build_synthetic_fit(
    *,
    output_root: Path,
    force_grid: tuple[float, ...],
    train_rows: tuple[DnnSurrogateLongRow, ...],
    ensemble_size: int,
    ensemble_seeds: tuple[int, ...],
    architecture: str,
    device: str,
) -> Emb34umDnnSurrogateFit:
    has_d4_rows = any(row.radp is not None or row.shell_th is not None for row in train_rows)
    if has_d4_rows:
        if any(row.radp is None or row.shell_th is None for row in train_rows):
            raise ValueError("synthetic DNN fit cannot mix D2 and D4 training rows.")
        features = np.asarray(
            [(row.ka_log10, row.kb_log10, float(row.radp), float(row.shell_th), row.force_norm) for row in train_rows],
            dtype=float,
        )
    else:
        features = np.asarray([(row.ka_log10, row.kb_log10, row.force_norm) for row in train_rows], dtype=float)
    targets = np.asarray([row.observable for row in train_rows], dtype=float)
    feature_mean = tuple(float(item) for item in features.mean(axis=0))
    feature_scale_arr = features.std(axis=0)
    feature_scale_arr[feature_scale_arr == 0.0] = 1.0
    feature_scale = tuple(float(item) for item in feature_scale_arr)
    target_mean = float(targets.mean())
    target_scale = float(targets.std()) or 1.0

    members_dir = output_root / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []
    training_history: list[dict[str, Any]] = []
    for index, seed in enumerate(ensemble_seeds, start=1):
        checkpoint = members_dir / f"synthetic_member_{index:02d}_seed_{seed}.pt"
        checkpoint.write_text("synthetic-checkpoint\n", encoding="utf-8")
        checkpoints.append(checkpoint)
        training_history.append(
            {
                "member_index": index,
                "seed": int(seed),
                "effective_seed": int(seed),
                "checkpoint_path": str(checkpoint),
                "train_runtime_seconds": 0.0,
                "train_loss_history": [1.0 / (index + step) for step in (1, 2, 3)],
                "validation_loss_history": [1.25 / (index + step) for step in (1, 2, 3)],
            }
        )

    return Emb34umDnnSurrogateFit(
        architecture=architecture,
        backend="fixed_architecture_dnn",
        ensemble_seeds=ensemble_seeds,
        ensemble_checkpoints=tuple(checkpoints),
        ensemble_member_count=ensemble_size,
        train_runtime_seconds=0.0,
        score_runtime_seconds=0.0,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        force_min=float(force_grid[0]),
        force_max=float(force_grid[-1]),
        force_grid=force_grid,
        device=device,
        train_row_count=len(train_rows),
        validation_row_count=max(1, int(round(len(train_rows) * 0.1))),
        model_path=output_root / "synthetic_model.pt",
        training_history=tuple(training_history),
    )


def _synthetic_score_candidates(
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    existing_points: Sequence[tuple[float, ...]],
    candidate_space: str,
    diversity_weight: float,
    force_grid: Sequence[float],
    acquisition_score_mode: str,
) -> tuple[dict[str, Any], ...]:
    candidate_space = _coerce_candidate_space(candidate_space)
    if not candidate_pool:
        return ()

    scored: list[dict[str, Any]] = []
    normalized_points: list[tuple[float, ...]] = []
    for index, raw_row in enumerate(candidate_pool, start=1):
        row = dict(raw_row)
        row["candidate_index"] = index
        point = _coerce_raw_candidate_point(row, index=index, candidate_space=candidate_space)
        normalized_points.append(_normalize_candidate_point_values(point, candidate_space=candidate_space))

    if normalized_points:
        centroid = tuple(sum(values) / len(values) for values in zip(*normalized_points))
    else:
        centroid = ()
    raw_disagreements = [math.dist(point, centroid) for point in normalized_points]

    normalized_existing = tuple(
        _normalize_candidate_point_values(point, candidate_space=candidate_space)
        for point in existing_points
    )
    raw_diversity = [
        (
            min(math.dist(point, existing_point) for existing_point in normalized_existing)
            if normalized_existing
            else 0.0
        )
        for point in normalized_points
    ]

    if candidate_space == "d4":
        disagreement_norm = _normalize_unit(raw_disagreements)
        diversity_norm = _normalize_unit(raw_diversity)
    else:
        disagreement_norm = raw_disagreements
        diversity_norm = raw_diversity
    raw_curve_risk = [
        float(raw_disagreement / max(1.0e-12, 0.1 + abs(point[0]) + abs(point[1])))
        for point, raw_disagreement in zip(normalized_points, raw_disagreements)
    ]
    curve_risk_norm = _normalize_unit(raw_curve_risk)

    for row, normalized_disagreement, normalized_diversity, normalized_point, raw_disagreement, raw_curve_error_risk in zip(
        candidate_pool,
        disagreement_norm,
        diversity_norm,
        normalized_points,
        raw_disagreements,
        raw_curve_risk,
    ):
        row = dict(row)
        row["candidate_log10_point"] = normalized_point
        row["ensemble_disagreement"] = float(raw_disagreement)
        row["ensemble_disagreement_norm"] = float(normalized_disagreement)
        row["predicted_curve_relative_l2_error"] = float(raw_curve_error_risk)
        row["predicted_curve_relative_l2_error_norm"] = float(
            curve_risk_norm[len(scored)] if curve_risk_norm else 0.0
        )
        row["curve_uncertainty_l2"] = float(raw_disagreement)
        row["predicted_curve_l2"] = float(max(1.0e-12, 0.1 + abs(normalized_point[0]) + abs(normalized_point[1])))
        row["diversity_term"] = float(normalized_diversity)
        if acquisition_score_mode == "curve_error":
            base_score = float(row["predicted_curve_relative_l2_error_norm"])
            base_field = "predicted_curve_relative_l2_error"
        else:
            base_score = float(normalized_disagreement)
            base_field = "ensemble_disagreement"
        row["acquisition_score_mode"] = acquisition_score_mode
        row["acquisition_base_field"] = base_field
        row["acquisition_base_score"] = float(base_score)
        row["acquisition_score"] = float(base_score + float(diversity_weight) * normalized_diversity)
        ka_log10, kb_log10 = float(normalized_point[0]), float(normalized_point[1])
        row["predicted_curve"] = tuple(
            float(0.5 * base_score + 0.05 * step + 0.01 * ka_log10 - 0.01 * kb_log10)
            for step, _ in enumerate(force_grid)
        )
        scored.append(row)

    # Preserve deterministic ranking for legacy and new space behavior.
    scored.sort(
        key=lambda item: (
            -float(item["acquisition_score"]),
            -float(item["ensemble_disagreement_norm"] if candidate_space == "d4" else item["ensemble_disagreement"]),
            float(item["candidate_log10_point"][0]),
            float(item["candidate_log10_point"][1]),
        )
    )
    for index, item in enumerate(scored, start=1):
        item["candidate_index"] = index
    return tuple(scored)


def _timing_payload(
    *,
    enabled: bool,
    point_count: int,
    train_seconds: float,
    score_seconds: float,
    override_reason: str = "",
) -> dict[str, Any]:
    payload = {
        "enabled": bool(enabled),
        "point_count": int(point_count),
        "train_runtime_seconds": float(train_seconds),
        "score_runtime_seconds": float(score_seconds),
    }
    if override_reason:
        payload["override_reason"] = override_reason
    return payload


def _jsonable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        point = payload.get("candidate_log10_point")
        curve = payload.get("predicted_curve")
        if isinstance(point, tuple):
            payload["candidate_log10_point"] = list(point)
        if isinstance(curve, tuple):
            payload["predicted_curve"] = list(curve)
        normalized.append(payload)
    return normalized


def _fit_payload(fit: Emb34umDnnSurrogateFit, *, score_runtime_seconds: float) -> dict[str, Any]:
    return {
        "architecture": fit.architecture,
        "backend": fit.backend,
        "ensemble_size": fit.ensemble_member_count,
        "ensemble_seeds": list(fit.ensemble_seeds),
        "ensemble_checkpoints": [str(path) for path in fit.ensemble_checkpoints],
        "train_runtime_seconds": float(fit.train_runtime_seconds),
        "score_runtime_seconds": float(score_runtime_seconds),
        "feature_mean": list(fit.feature_mean),
        "feature_scale": list(fit.feature_scale),
        "target_mean": float(fit.target_mean),
        "target_scale": float(fit.target_scale),
        "force_grid": list(fit.force_grid),
        "device": fit.device,
        "train_row_count": fit.train_row_count,
        "validation_row_count": fit.validation_row_count,
        "training_history": [dict(item) for item in fit.training_history],
    }


def _selection_summary(scored: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disagreement_values = [float(item["ensemble_disagreement"]) for item in scored]
    curve_error_values = [float(item.get("predicted_curve_relative_l2_error", 0.0)) for item in scored]
    acquisition_values = [float(item["acquisition_score"]) for item in scored]
    diversity_values = [float(item["diversity_term"]) for item in selected]
    return {
        "candidate_pool_count": len(scored),
        "selected_count": len(selected),
        "ensemble_disagreement": {
            "min": min(disagreement_values) if disagreement_values else 0.0,
            "max": max(disagreement_values) if disagreement_values else 0.0,
            "mean": float(np.mean(disagreement_values)) if disagreement_values else 0.0,
        },
        "predicted_curve_relative_l2_error": {
            "min": min(curve_error_values) if curve_error_values else 0.0,
            "max": max(curve_error_values) if curve_error_values else 0.0,
            "mean": float(np.mean(curve_error_values)) if curve_error_values else 0.0,
        },
        "acquisition_score": {
            "min": min(acquisition_values) if acquisition_values else 0.0,
            "max": max(acquisition_values) if acquisition_values else 0.0,
            "mean": float(np.mean(acquisition_values)) if acquisition_values else 0.0,
        },
        "selected_diversity_term": {
            "min": min(diversity_values) if diversity_values else 0.0,
            "max": max(diversity_values) if diversity_values else 0.0,
            "mean": float(np.mean(diversity_values)) if diversity_values else 0.0,
        },
    }


def _build_manifest(
    *,
    args: argparse.Namespace,
    fit: Emb34umDnnSurrogateFit,
    completed_rows: Sequence[Mapping[str, Any]],
    existing_points: Sequence[tuple[float, ...]],
    existing_points_source: str,
    candidate_pool: Sequence[Mapping[str, Any]],
    skipped_existing_candidate_ids: Sequence[str],
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    train_runtime_seconds: float,
    score_runtime_seconds: float,
    timing_payload: dict[str, Any],
    plot_paths: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": EMB_34UM_DNN_TRAIN_SCORE_SELECT_SCHEMA_VERSION,
        "selector_backend": fit.backend,
        "candidate_space": args.candidate_space,
        "fit": _fit_payload(fit, score_runtime_seconds=score_runtime_seconds),
        "selection": {
            **_selection_summary(scored, selected),
            "acquisition_score_mode": args.acquisition_score_mode,
            "skipped_existing_candidate_count": len(skipped_existing_candidate_ids),
            "skipped_existing_candidate_ids": list(skipped_existing_candidate_ids),
        },
        "candidate_pool": [dict(item) for item in candidate_pool],
        "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
        "candidate_scores": _jsonable_rows(scored),
        "selected_points": _jsonable_rows(selected),
        "completed_curve_count": len(completed_rows),
        "existing_points": {
            "source": existing_points_source,
            "count": len(existing_points),
            "used_for_diversity": True,
            "diversity_weight": float(args.diversity_weight),
            "points": [
                [float(item) for item in point]
                for point in existing_points
            ],
        },
        "train_runtime_seconds": float(train_runtime_seconds),
        "score_runtime_seconds": float(score_runtime_seconds),
        "timing_canary": timing_payload,
        "resources": {
            "device": fit.device,
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        },
        "rng_seeds": {
            "ensemble_seeds": list(fit.ensemble_seeds),
            "seed_offset": int(args.seed_offset),
            "dry_run": bool(args.dry_run),
        },
        "plots": dict(plot_paths),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    validate_dnn_causal_ensemble_size(args.ensemble_size)
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")
    if args.diversity_weight < 0.0:
        raise ValueError("--diversity-weight must be non-negative.")
    candidate_space = _coerce_candidate_space(args.candidate_space)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed_rows = _load_json_list(Path(args.completed_rows), label="completed_rows")
    candidate_pool = _load_json_list(Path(args.candidate_pool), label="candidate_pool")
    if not candidate_pool:
        raise ValueError("candidate_pool must not be empty.")
    _validate_candidate_pool_exclusion(candidate_pool)

    force_grid = _coerce_force_grid_arg(args.force_grid)
    if args.existing_points is None:
        existing_points = _existing_points_from_completed_rows(
            completed_rows,
            candidate_space=candidate_space,
        )
        existing_points_source = "completed_rows"
    else:
        existing_points = _coerce_existing_points(args.existing_points, candidate_space=candidate_space)
        existing_points_source = "cli_existing_points"
    candidate_pool, skipped_existing_candidate_ids = _exclude_existing_points_from_candidate_pool(
        candidate_pool,
        existing_points=existing_points,
        candidate_space=candidate_space,
    )
    if not candidate_pool:
        raise ValueError("candidate_pool has no novel points after excluding already completed points.")
    ensemble_seeds = tuple(args.ensemble_seed) if args.ensemble_seed else None

    timing_canary_config = None
    if args.timing_canary:
        timing_canary_config = validate_dnn_causal_timing_canary(
            {
                "required": True,
                "point_count": int(args.timing_canary_point_count),
                **(
                    {"override_reason": args.timing_canary_override_reason}
                    if args.timing_canary_override_reason
                    else {}
                ),
            }
        )

    train_rows = convert_completed_curves_to_long_rows(completed_rows, force_grid=force_grid)

    train_start = perf_counter()
    if args.dry_run:
        resolved_seeds = ensemble_seeds or tuple(range(1, args.ensemble_size + 1))
        fit = _build_synthetic_fit(
            output_root=output_root,
            force_grid=force_grid,
            train_rows=train_rows,
            ensemble_size=args.ensemble_size,
            ensemble_seeds=resolved_seeds,
            architecture=args.architecture,
            device=args.device,
        )
        fit = replace(fit, train_runtime_seconds=perf_counter() - train_start)
    else:
        fit, _ = train_emb_34um_dnn_surrogate_ensemble(
            train_rows,
            architecture=args.architecture,
            ensemble_size=args.ensemble_size,
            ensemble_seeds=ensemble_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            force_grid=force_grid,
            output_root=output_root,
            device=args.device,
            seed_offset=args.seed_offset,
            timing_canary=bool(timing_canary_config),
        )
        fit = replace(fit, train_runtime_seconds=perf_counter() - train_start)
    train_runtime_seconds = float(fit.train_runtime_seconds)

    score_start = perf_counter()
    if args.dry_run:
        scored = _synthetic_score_candidates(
            candidate_pool,
            existing_points=existing_points,
            candidate_space=candidate_space,
            diversity_weight=float(args.diversity_weight),
            force_grid=force_grid,
            acquisition_score_mode=args.acquisition_score_mode,
        )
    else:
        scored = tuple(
            dict(item)
            for item in score_candidate_pool(
                fit,
                candidate_records=candidate_pool,
                force_grid=force_grid,
                existing_points=existing_points,
                candidate_space=candidate_space,
                diversity_weight=float(args.diversity_weight),
                acquisition_score_mode=args.acquisition_score_mode,
            )
        )
    score_runtime_seconds = perf_counter() - score_start
    fit = replace(fit, score_runtime_seconds=float(score_runtime_seconds))

    selected = select_greedy_diversity(
        scored,
        count=min(args.top_n, len(scored)),
        existing_points=existing_points,
        candidate_space=candidate_space,
        diversity_weight=float(args.diversity_weight),
        acquisition_score_mode=args.acquisition_score_mode,
    )

    heatmap_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_SELECTION_PLOT_FILENAME
    curve_error_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_CURVE_ERROR_PLOT_FILENAME
    distribution_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_DISTRIBUTION_PLOT_FILENAME
    loss_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_LOSS_PLOT_FILENAME
    _plot_candidate_heatmap(heatmap_path, scored, selected)
    _plot_curve_error_risk(curve_error_path, scored, selected)
    _plot_distribution(distribution_path, selected)
    _plot_losses(loss_path, fit)

    timing_plot_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_PLOT_FILENAME
    timing_report_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_TIMING_REPORT_FILENAME
    if args.timing_canary:
        _plot_timing_canary(
            timing_plot_path,
            train_seconds=train_runtime_seconds,
            score_seconds=score_runtime_seconds,
        )
        timing_payload = _timing_payload(
            enabled=True,
            point_count=timing_canary_config["point_count"],
            train_seconds=train_runtime_seconds,
            score_seconds=score_runtime_seconds,
            override_reason=timing_canary_config.get("override_reason", ""),
        )
        timing_report_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        timing_payload = _timing_payload(
            enabled=False,
            point_count=EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
            train_seconds=train_runtime_seconds,
            score_seconds=score_runtime_seconds,
        )

    plot_paths = {
        "candidate_score_heatmap": str(heatmap_path),
        "curve_error_risk": str(curve_error_path),
        "selected_overlay": str(heatmap_path),
        "disagreement_distribution": str(distribution_path),
        "member_losses": str(loss_path),
    }
    if args.timing_canary:
        plot_paths["timing_canary"] = str(timing_plot_path)
    if candidate_space == "d4":
        ka_radp_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_KA_RADP_PLOT_FILENAME
        kb_shell_th_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_KB_SHELL_TH_PLOT_FILENAME
        radp_shell_th_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_RADP_SHELL_TH_PLOT_FILENAME
        _plot_candidate_pair_scatter(
            ka_radp_path,
            scored,
            selected,
            x_key="ka",
            y_key="radp",
            x_label="ka",
            y_label="radp",
            x_transform=math.log10,
        )
        _plot_candidate_pair_scatter(
            kb_shell_th_path,
            scored,
            selected,
            x_key="kb",
            y_key="shell_th",
            x_label="kb",
            y_label="shell_th",
            x_transform=math.log10,
        )
        _plot_candidate_pair_scatter(
            radp_shell_th_path,
            scored,
            selected,
            x_key="radp",
            y_key="shell_th",
            x_label="radp",
            y_label="shell_th",
        )
        plot_paths["ka_radp"] = str(ka_radp_path)
        plot_paths["kb_shell_th"] = str(kb_shell_th_path)
        plot_paths["radp_shell_th"] = str(radp_shell_th_path)

    manifest = _build_manifest(
        args=args,
        fit=fit,
        completed_rows=completed_rows,
        existing_points=existing_points,
        existing_points_source=existing_points_source,
        candidate_pool=candidate_pool,
        skipped_existing_candidate_ids=skipped_existing_candidate_ids,
        scored=scored,
        selected=selected,
        train_runtime_seconds=train_runtime_seconds,
        score_runtime_seconds=score_runtime_seconds,
        timing_payload=timing_payload,
        plot_paths=plot_paths,
    )
    report = {
        "schema_version": EMB_34UM_DNN_TRAIN_SCORE_SELECT_SCHEMA_VERSION,
        "status": "dry-run" if args.dry_run else "completed",
        "fit": manifest["fit"],
        "selection": manifest["selection"],
        "timing_canary": manifest["timing_canary"],
        "plots": manifest["plots"],
    }

    manifest_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_MANIFEST_FILENAME
    report_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_REPORT_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    if args.timing_canary:
        print(f"timing_report={timing_report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
