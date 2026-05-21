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
    compute_disagreement_distribution,
    score_candidate_pool,
    select_greedy_diversity,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
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


def _coerce_existing_points(raw: str | None) -> tuple[tuple[float, float], ...]:
    if raw is None:
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("--existing-points must decode to a JSON list.")
    points: list[tuple[float, float]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"existing_points[{index}] must be a [ka, kb] pair.")
        ka = float(item[0])
        kb = float(item[1])
        if ka <= 0.0 or kb <= 0.0:
            raise ValueError("existing point coordinates must be positive.")
        points.append((ka, kb))
    return tuple(points)


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


def _plot_candidate_heatmap(path: Path, scored: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> None:
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
    parser.add_argument("--force-grid", default=None, help="Optional JSON list override for the force grid.")
    parser.add_argument("--timing-canary", action="store_true")
    parser.add_argument("--timing-canary-point-count", type=int, default=EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT)
    parser.add_argument("--timing-canary-override-reason", default="")
    parser.add_argument("--existing-points", default=None, help="JSON list of existing [ka, kb] points.")
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
    existing_points: Sequence[tuple[float, float]],
    diversity_weight: float,
    force_grid: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    existing_log10 = [(math.log10(ka), math.log10(kb)) for ka, kb in existing_points]
    scored: list[dict[str, Any]] = []
    ka_values = [math.log10(float(row["ka"])) for row in candidate_pool]
    kb_values = [math.log10(float(row["kb"])) for row in candidate_pool]
    ka_center = float(sum(ka_values) / len(ka_values))
    kb_center = float(sum(kb_values) / len(kb_values))

    for index, raw_row in enumerate(candidate_pool, start=1):
        row = dict(raw_row)
        ka_log10 = math.log10(float(row["ka"]))
        kb_log10 = math.log10(float(row["kb"]))
        disagreement = abs(ka_log10 - ka_center) + abs(kb_log10 - kb_center)
        disagreement /= max(1.0, max(abs(v - ka_center) for v in ka_values) + max(abs(v - kb_center) for v in kb_values))
        diversity_term = (
            min(math.dist((ka_log10, kb_log10), point) for point in existing_log10)
            if existing_log10
            else 0.0
        )
        row["candidate_index"] = index
        row["candidate_log10_point"] = (ka_log10, kb_log10)
        row["ensemble_disagreement"] = float(disagreement)
        row["diversity_term"] = float(diversity_term)
        row["acquisition_score"] = float(disagreement + diversity_weight * diversity_term)
        row["predicted_curve"] = tuple(
            float(0.5 * disagreement + 0.05 * step + 0.01 * ka_log10 - 0.01 * kb_log10)
            for step, _ in enumerate(force_grid)
        )
        scored.append(row)

    scored.sort(
        key=lambda item: (
            -float(item["acquisition_score"]),
            -float(item["ensemble_disagreement"]),
            float(item["candidate_log10_point"][0]),
            float(item["candidate_log10_point"][1]),
        )
    )
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
    candidate_pool: Sequence[Mapping[str, Any]],
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
        "fit": _fit_payload(fit, score_runtime_seconds=score_runtime_seconds),
        "selection": _selection_summary(scored, selected),
        "candidate_pool": [dict(item) for item in candidate_pool],
        "candidate_scores": _jsonable_rows(scored),
        "selected_points": _jsonable_rows(selected),
        "completed_curve_count": len(completed_rows),
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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed_rows = _load_json_list(Path(args.completed_rows), label="completed_rows")
    candidate_pool = _load_json_list(Path(args.candidate_pool), label="candidate_pool")
    if not candidate_pool:
        raise ValueError("candidate_pool must not be empty.")

    force_grid = _coerce_force_grid_arg(args.force_grid)
    existing_points = _coerce_existing_points(args.existing_points)
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
            diversity_weight=float(args.diversity_weight),
            force_grid=force_grid,
        )
    else:
        scored = tuple(
            dict(item)
            for item in score_candidate_pool(
                fit,
                candidate_records=candidate_pool,
                force_grid=force_grid,
                existing_points=existing_points,
                diversity_weight=float(args.diversity_weight),
            )
        )
    score_runtime_seconds = perf_counter() - score_start
    fit = replace(fit, score_runtime_seconds=float(score_runtime_seconds))

    selected = select_greedy_diversity(
        scored,
        count=min(args.top_n, len(scored)),
        existing_points=existing_points,
        diversity_weight=float(args.diversity_weight),
    )

    heatmap_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_SELECTION_PLOT_FILENAME
    distribution_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_DISTRIBUTION_PLOT_FILENAME
    loss_path = output_root / EMB_34UM_DNN_TRAIN_SCORE_SELECT_LOSS_PLOT_FILENAME
    _plot_candidate_heatmap(heatmap_path, scored, selected)
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
        "selected_overlay": str(heatmap_path),
        "disagreement_distribution": str(distribution_path),
        "member_losses": str(loss_path),
    }
    if args.timing_canary:
        plot_paths["timing_canary"] = str(timing_plot_path)

    manifest = _build_manifest(
        args=args,
        fit=fit,
        completed_rows=completed_rows,
        candidate_pool=candidate_pool,
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
