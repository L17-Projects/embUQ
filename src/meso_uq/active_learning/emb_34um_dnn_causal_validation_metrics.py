from __future__ import annotations

"""Generate per-replicate DNN AL-vs-LHS validation metric rows for production analysis."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (
    normalize_emb_34um_dnn_causal_validation_records,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    validate_dnn_causal_cycle_count,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
)
from meso_uq.active_learning.emb_34um_dnn_surrogate import (
    EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE,
    EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME,
    EMB_34UM_DNN_SURROGATE_REPORT_FILENAME,
    load_ensemble_member_predictions,
    train_emb_34um_dnn_surrogate_ensemble,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_metric_rows.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME = "emb_34um_dnn_causal_validation_rows.json"
EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME = "emb_34um_dnn_causal_validation_metric_report.json"
_PRODUCTION_BRANCHES = frozenset({"unseen_test", "shared_initial", "al", "lhs"})


@dataclass(frozen=True)
class Emb34umDnnCausalValidationMetricArtifacts:
    artifact_dir: Path
    rows_path: Path
    report_path: Path
    rows_payload: dict[str, Any]
    report_payload: dict[str, Any]


def _coerce_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(value)


def _coerce_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _coerce_int(value: object, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _coerce_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _coerce_candidate_space(value: object, *, label: str, default: str = "d4") -> str:
    text = str(default if value in (None, "") else value).strip().lower()
    if text not in {"d2", "d4"}:
        raise ValueError(f"{label} must be 'd2' or 'd4'.")
    return text


def _optional_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _canonical_completed_row(row: Mapping[str, Any], *, source_index: int) -> dict[str, Any]:
    payload = _coerce_mapping(row, label=f"completed_rows[{source_index}]")
    branch = str(payload.get("branch", "")).strip().lower()
    if branch not in {"unseen_test", "shared_initial", "al", "lhs"}:
        raise ValueError(f"completed_rows[{source_index}].branch is unsupported: {branch!r}")
    candidate_space = _coerce_candidate_space(
        payload.get("candidate_space"),
        label=f"completed_rows[{source_index}].candidate_space",
    )

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    ka = payload.get("ka", parameters.get("ka"))
    kb = payload.get("kb", parameters.get("kb"))
    radp = payload.get("radp", parameters.get("radp"))
    shell_th = payload.get("shell_th", parameters.get("shell_th"))
    if ka is None or kb is None:
        raise ValueError(f"completed_rows[{source_index}] is missing ka/kb.")
    radp_float: float | None = _coerce_float(radp, label="radp") if radp is not None else None
    shell_th_float: float | None = _coerce_float(shell_th, label="shell_th") if shell_th is not None else None
    if (radp_float is None) != (shell_th_float is None):
        raise ValueError(f"completed_rows[{source_index}] must provide both radp and shell_th together.")
    if candidate_space == "d4" and (radp_float is None or shell_th_float is None):
        raise ValueError(
            f"completed_rows[{source_index}] must include radp and shell_th for D4 causal validation. "
            "Use candidate_space='d2' only for explicit legacy rows."
        )

    force_grid = payload.get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID)
    curve = payload.get("reference_curve", payload.get("force_curve", payload.get("target_curve")))
    if curve is None:
        raise ValueError(f"completed_rows[{source_index}] is missing curve values.")

    resolved_force_grid = validate_dnn_causal_force_grid(_coerce_sequence(force_grid, label="force_grid"))
    resolved_curve = tuple(
        _coerce_float(item, label=f"completed_rows[{source_index}].curve[{idx}]")
        for idx, item in enumerate(_coerce_sequence(curve, label="curve"), start=1)
    )
    if len(resolved_force_grid) != len(resolved_curve):
        raise ValueError(f"completed_rows[{source_index}] force_grid and curve lengths must match.")

    return {
        "candidate_id": str(payload.get("candidate_id", payload.get("curve_id", f"curve-{source_index:06d}"))),
        "branch": branch,
        "replicate": _coerce_int(payload.get("replicate", 0), label="replicate"),
        "cycle": _coerce_int(payload.get("cycle", 0), label="cycle"),
        "candidate_space": candidate_space,
        "ka": _coerce_float(ka, label="ka"),
        "kb": _coerce_float(kb, label="kb"),
        "radp": radp_float,
        "shell_th": shell_th_float,
        "force_grid": resolved_force_grid,
        "reference_curve": resolved_curve,
        "replacement": _as_bool(payload.get("replacement", False)),
        "quarantine": _as_bool(payload.get("quarantine", payload.get("quarantined", False))),
        "selection_source": _optional_text(payload.get("selection_source"))
        or _optional_text(payload.get("sample_source"))
        or _optional_text(payload.get("source"))
        or _optional_text(payload.get("selection_mode")),
        "candidate_pool_id": _optional_text(payload.get("candidate_pool_id"))
        or _optional_text(payload.get("selection_candidate_id")),
        "selection_payload": _coerce_mapping(payload.get("selection_payload", {}), label="selection_payload")
        if isinstance(payload.get("selection_payload", {}), Mapping)
        else {},
        "selection_status": _optional_text(payload.get("selection_status")),
    }


def _normalize_completed_rows(source: object) -> list[dict[str, Any]]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        raw_rows = [dict(item) for item in source if isinstance(item, Mapping)]
    else:
        raw_rows = normalize_emb_34um_dnn_causal_validation_records(source)
    return [_canonical_completed_row(item, source_index=index) for index, item in enumerate(raw_rows, start=1)]


def _relative_l2(predicted: Sequence[float], reference: Sequence[float]) -> float:
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference curves must have equal length.")
    diff_norm_sq = 0.0
    ref_norm_sq = 0.0
    for left, right in zip(predicted, reference):
        delta = float(left) - float(right)
        diff_norm_sq += delta * delta
        ref_norm_sq += float(right) * float(right)
    denom = math.sqrt(ref_norm_sq)
    if denom <= 1.0e-12:
        denom = 1.0
    return float(math.sqrt(diff_norm_sq) / denom)


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(float(item) for item in values))


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    parsed = [float(item) for item in values]
    return {
        "min": float(min(parsed)),
        "max": float(max(parsed)),
        "mean": float(statistics.mean(parsed)),
        "median": float(statistics.median(parsed)),
        "std": _safe_std(parsed),
    }


def _per_force_error_summary(
    *,
    force_grid: Sequence[float],
    predicted_curves: Sequence[Sequence[float]],
    reference_curves: Sequence[Sequence[float]],
) -> tuple[dict[str, float | int], ...]:
    if not predicted_curves or not reference_curves:
        return tuple(
            {
                "force_index": index,
                "force": float(force),
                "mean_signed_error": 0.0,
                "mean_absolute_error": 0.0,
                "median_absolute_error": 0.0,
                "rmse": 0.0,
            }
            for index, force in enumerate(force_grid)
        )

    rows: list[dict[str, float | int]] = []
    for force_index, force in enumerate(force_grid):
        signed_errors: list[float] = []
        absolute_errors: list[float] = []
        squared_errors: list[float] = []
        for predicted, reference in zip(predicted_curves, reference_curves):
            delta = float(predicted[force_index]) - float(reference[force_index])
            signed_errors.append(delta)
            absolute_errors.append(abs(delta))
            squared_errors.append(delta * delta)
        rows.append(
            {
                "force_index": int(force_index),
                "force": float(force),
                "mean_signed_error": float(statistics.mean(signed_errors)),
                "mean_absolute_error": float(statistics.mean(absolute_errors)),
                "median_absolute_error": float(statistics.median(absolute_errors)),
                "rmse": float(math.sqrt(statistics.mean(squared_errors))),
            }
        )
    return tuple(rows)


def _runtime_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = [
        float(row["runtime_seconds"])
        for row in rows
        if row.get("runtime_seconds") not in (None, "")
    ]
    stats = _summary(values)
    return {
        "count": int(len(values)),
        "total": float(sum(values)),
        **stats,
    }


def _checkpoint_index_payload(
    *,
    output_root: Path | None,
    architecture: str,
    ensemble_size: int,
    ensemble_seeds: Sequence[int],
    checkpoint_paths: Sequence[Path],
    train_kwargs: Mapping[str, Any] | None,
    training_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    options = dict(train_kwargs or {})
    return {
        "architecture": str(architecture),
        "ensemble_size": int(ensemble_size),
        "ensemble_seeds": [int(seed) for seed in ensemble_seeds],
        "ensemble_checkpoints": [str(path) for path in checkpoint_paths],
        "training_manifest_path": str(output_root / EMB_34UM_DNN_SURROGATE_MANIFEST_FILENAME) if output_root else "",
        "training_report_path": str(output_root / EMB_34UM_DNN_SURROGATE_REPORT_FILENAME) if output_root else "",
        "optimizer_settings": {
            "epochs": int(options.get("epochs", 200)),
            "batch_size": int(options.get("batch_size", 128)),
            "learning_rate": float(options.get("learning_rate", 1.0e-3)),
            "validation_fraction": float(options.get("validation_fraction", 0.1)),
        },
        "training_history": [dict(item) for item in training_history or ()],
    }


def _training_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    joined = "\n".join(sorted(str(item.get("candidate_id", "")) for item in rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _canonical_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def _add_optional_parameters(item: Mapping[str, Any]) -> dict[str, float]:
        payload = {
            "ka": float(item["ka"]),
            "kb": float(item["kb"]),
        }
        if item.get("radp") is not None:
            payload["radp"] = float(item["radp"])
        if item.get("shell_th") is not None:
            payload["shell_th"] = float(item["shell_th"])
        return payload

    return [
        {
            "candidate_id": str(item["candidate_id"]),
            "parameters": _add_optional_parameters(item),
            "force_grid": list(item["force_grid"]),
            "reference_curve": list(item["reference_curve"]),
        }
        for item in rows
    ]


def _dry_run_metrics(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    unseen_rows: Sequence[Mapping[str, Any]],
    branch: str,
    cycle: int,
    train_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ka_center = statistics.mean(math.log10(float(item["ka"])) for item in train_rows)
    kb_center = statistics.mean(math.log10(float(item["kb"])) for item in train_rows)
    relative_l2_values: list[float] = []
    predicted_curves: list[list[float]] = []
    reference_curves: list[list[float]] = []
    spread_values: list[float] = []
    branch_bias = 0.003 if branch == "lhs" and int(cycle) > 0 else 0.0
    for row in unseen_rows:
        ka_delta = math.log10(float(row["ka"])) - ka_center
        kb_delta = math.log10(float(row["kb"])) - kb_center
        scale = 0.02 + 0.005 * math.hypot(ka_delta, kb_delta) + 0.001 * float(cycle) + branch_bias
        predicted = [float(value) * (1.0 + scale) for value in row["reference_curve"]]
        predicted_curves.append(predicted)
        reference_curves.append([float(value) for value in row["reference_curve"]])
        spread_values.append(float(0.2 * abs(scale)))
        relative_l2_values.append(_relative_l2(predicted, row["reference_curve"]))

    train_seconds = 1.0e-4 * len(train_rows) * EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    score_seconds = 5.0e-5 * len(unseen_rows) * EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    force_grid = validate_dnn_causal_force_grid(unseen_rows[0]["force_grid"])
    return {
        "relative_l2_mean": float(statistics.mean(relative_l2_values)),
        "relative_l2_median": float(statistics.median(relative_l2_values)),
        "relative_l2_std": _safe_std(relative_l2_values),
        "train_seconds": float(train_seconds),
        "score_seconds": float(score_seconds),
        "per_force_point_error_summary": list(
            _per_force_error_summary(
                force_grid=force_grid,
                predicted_curves=predicted_curves,
                reference_curves=reference_curves,
            )
        ),
        "ensemble_mean_prediction_error": {
            "relative_l2": _summary(relative_l2_values),
            "mean_absolute_error": _summary(
                [
                    abs(float(predicted) - float(reference))
                    for curve, ref_curve in zip(predicted_curves, reference_curves)
                    for predicted, reference in zip(curve, ref_curve)
                ]
            ),
        },
        "ensemble_spread_summary": {
            **_summary(spread_values),
            "member_count": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "source": "dry_run_synthetic",
        },
        "model_checkpoint_index": _checkpoint_index_payload(
            output_root=None,
            architecture=str((train_kwargs or {}).get("architecture", EMB_34UM_DNN_SURROGATE_FIXED_ARCHITECTURE)),
            ensemble_size=EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            ensemble_seeds=tuple(range(1, EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE + 1)),
            checkpoint_paths=(),
            train_kwargs=train_kwargs,
        ),
    }


def _real_metrics(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    unseen_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    ensemble_size: int,
    device: str | None,
    force_grid: tuple[float, ...],
    train_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options = dict(train_kwargs or {})
    train_start = perf_counter()
    fit, _ = train_emb_34um_dnn_surrogate_ensemble(
        _canonical_training_rows(train_rows),
        ensemble_size=ensemble_size,
        force_grid=force_grid,
        output_root=output_root,
        device=device,
        **options,
    )
    train_seconds = perf_counter() - train_start

    score_start = perf_counter()
    relative_l2_values: list[float] = []
    predicted_curves: list[tuple[float, ...]] = []
    reference_curves: list[tuple[float, ...]] = []
    spread_values: list[float] = []
    relative_spread_values: list[float] = []
    for unseen in unseen_rows:
        radp = unseen.get("radp")
        shell_th = unseen.get("shell_th")
        predicted_curve, member_curves = load_ensemble_member_predictions(
            fit,
            force_grid,
            ka=float(unseen["ka"]),
            kb=float(unseen["kb"]),
            radp=radp if radp is None else float(radp),
            shell_th=(
                shell_th if shell_th is None else float(shell_th)
            ),
        )
        reference_curve = tuple(float(value) for value in unseen["reference_curve"])
        predicted_curves.append(tuple(float(value) for value in predicted_curve))
        reference_curves.append(reference_curve)
        relative_l2_values.append(_relative_l2(predicted_curve, reference_curve))
        if member_curves:
            try:
                import numpy as np

                stacked = np.asarray(member_curves, dtype=float)
                per_force_std = stacked.std(axis=0)
                spread_values.append(float(per_force_std.mean()))
                mean_curve_norm = float(np.linalg.norm(np.asarray(predicted_curve, dtype=float)))
                spread_norm = float(np.linalg.norm(per_force_std))
                relative_spread_values.append(spread_norm / max(mean_curve_norm, 1.0e-12))
            except Exception:
                spread_values.append(0.0)
                relative_spread_values.append(0.0)
        else:
            spread_values.append(0.0)
            relative_spread_values.append(0.0)
    score_seconds = perf_counter() - score_start
    return {
        "relative_l2_mean": float(statistics.mean(relative_l2_values)),
        "relative_l2_median": float(statistics.median(relative_l2_values)),
        "relative_l2_std": _safe_std(relative_l2_values),
        "train_seconds": float(train_seconds),
        "score_seconds": float(score_seconds),
        "per_force_point_error_summary": list(
            _per_force_error_summary(
                force_grid=force_grid,
                predicted_curves=predicted_curves,
                reference_curves=reference_curves,
            )
        ),
        "ensemble_mean_prediction_error": {
            "relative_l2": _summary(relative_l2_values),
            "mean_absolute_error": _summary(
                [
                    abs(float(predicted) - float(reference))
                    for curve, ref_curve in zip(predicted_curves, reference_curves)
                    for predicted, reference in zip(curve, ref_curve)
                ]
            ),
        },
        "ensemble_spread_summary": {
            **_summary(spread_values),
            "relative_spread": _summary(relative_spread_values),
            "member_count": int(fit.ensemble_member_count),
            "source": "dnn_ensemble_member_predictions",
        },
        "model_checkpoint_index": _checkpoint_index_payload(
            output_root=output_root,
            architecture=fit.architecture,
            ensemble_size=fit.ensemble_member_count,
            ensemble_seeds=fit.ensemble_seeds,
            checkpoint_paths=fit.ensemble_checkpoints,
            train_kwargs=train_kwargs,
            training_history=fit.training_history,
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_relative_l2_metrics(
    *,
    dry_run: bool,
    train_rows: Sequence[Mapping[str, Any]],
    unseen_rows: Sequence[Mapping[str, Any]],
    branch: str,
    cycle: int,
    output_root: Path | None,
    ensemble_size: int,
    device: str | None,
    force_grid: tuple[float, ...],
    train_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if dry_run:
        return _dry_run_metrics(
            train_rows=train_rows,
            unseen_rows=unseen_rows,
            branch=branch,
            cycle=cycle,
            train_kwargs=train_kwargs,
        )

    if output_root is None:
        raise ValueError("output_root is required when dry_run is disabled.")
    return _real_metrics(
        train_rows=train_rows,
        unseen_rows=unseen_rows,
        output_root=output_root,
        ensemble_size=ensemble_size,
        device=device,
        force_grid=force_grid,
        train_kwargs=train_kwargs,
    )


def _infer_parameter_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    for row in rows:
        parameters = row.get("parameters")
        if isinstance(parameters, Mapping):
            active = [key for key in EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES if key in parameters]
            if active:
                return tuple(str(key) for key in active)
        active = [key for key in EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES if row.get(key) is not None]
        if active:
            return tuple(str(key) for key in active)
    return ("ka", "kb")


def _infer_test_set_source(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        if str(row.get("branch", "")).strip() != "unseen_test":
            continue
        source = _optional_text(row.get("selection_source"))
        if source:
            return source
        source = _optional_text(row.get("selection_mode"))
        if source:
            return source
        source = _optional_text(row.get("source"))
        if source:
            return source
    return EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE


def _source_branch_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"shared_initial": 0, "unseen_test": 0}
    for row in rows:
        branch = str(row.get("branch", "")).strip()
        if branch in counts:
            counts[branch] += 1
    return counts


def _assess_protocol_completeness(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    required_cycle_count = validate_dnn_causal_cycle_count(EMB_34UM_DNN_CAUSAL_MAX_CYCLES)
    required_replicate_count = int(EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT)
    required_replicates = tuple(range(1, required_replicate_count + 1))
    required_cycles = tuple(range(1, required_cycle_count + 1))
    blockers: list[str] = []

    production_rows = [row for row in rows if str(row["branch"]) in _PRODUCTION_BRANCHES]
    non_d4_rows = [
        row for row in production_rows
        if str(row.get("candidate_space", "")).strip().lower() != "d4"
    ]
    if non_d4_rows:
        blockers.append("Production completed rows must all use candidate_space='d4'.")

    unseen_rows = [row for row in rows if str(row["branch"]) == "unseen_test"]
    if len(unseen_rows) != EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE:
        blockers.append(
            "unseen_test count mismatch: "
            f"expected {EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE}, observed {len(unseen_rows)}."
        )

    shared_by_replicate: dict[int, int] = {}
    for row in rows:
        if str(row["branch"]) != "shared_initial":
            continue
        replicate = int(row["replicate"])
        shared_by_replicate[replicate] = shared_by_replicate.get(replicate, 0) + 1

    observed_shared_replicates = sorted(shared_by_replicate)
    if tuple(observed_shared_replicates) != required_replicates:
        blockers.append(
            "shared_initial replicate set mismatch: "
            f"expected {list(required_replicates)}, observed {observed_shared_replicates}."
        )
    for replicate in required_replicates:
        observed = int(shared_by_replicate.get(replicate, 0))
        if observed != EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE:
            blockers.append(
                f"shared_initial count mismatch for replicate={replicate}: "
                f"expected {EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE}, observed {observed}."
            )

    for branch in ("al", "lhs"):
        counts: dict[int, dict[int, int]] = {}
        for row in rows:
            if str(row["branch"]) != branch:
                continue
            replicate = int(row["replicate"])
            cycle = int(row["cycle"])
            counts.setdefault(replicate, {})
            counts[replicate][cycle] = counts[replicate].get(cycle, 0) + 1

        observed_replicates = sorted(counts)
        if tuple(observed_replicates) != required_replicates:
            blockers.append(
                f"{branch} replicate set mismatch: expected {list(required_replicates)}, observed {observed_replicates}."
            )
        for replicate in required_replicates:
            cycle_counts = counts.get(replicate, {})
            unexpected_cycles = sorted(cycle for cycle in cycle_counts if cycle not in required_cycles)
            if unexpected_cycles:
                blockers.append(
                    f"{branch} includes unexpected cycles for replicate={replicate}: {unexpected_cycles}."
                )
            for cycle in required_cycles:
                observed = int(cycle_counts.get(cycle, 0))
                if observed != EMB_34UM_DNN_CAUSAL_STEP_SIZE:
                    blockers.append(
                        f"{branch} count mismatch for replicate={replicate}, cycle={cycle}: "
                        f"expected {EMB_34UM_DNN_CAUSAL_STEP_SIZE}, observed {observed}."
                    )

    protocol_completeness = {
        "required": {
            "unseen_test_count": EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
            "replicate_count": required_replicate_count,
            "shared_initial_per_replicate": EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
            "step_size": EMB_34UM_DNN_CAUSAL_STEP_SIZE,
            "paired_cycle_count": required_cycle_count,
            "required_cycles": list(required_cycles),
        },
        "observed": {
            "unseen_test_count": len(unseen_rows),
            "shared_initial_by_replicate": {
                str(replicate): int(shared_by_replicate.get(replicate, 0))
                for replicate in required_replicates
            },
            "al_by_replicate_cycle": {
                str(replicate): {
                    str(cycle): int(
                        sum(
                            1 for row in rows
                            if str(row["branch"]) == "al"
                            and int(row["replicate"]) == replicate
                            and int(row["cycle"]) == cycle
                        )
                    )
                    for cycle in required_cycles
                }
                for replicate in required_replicates
            },
            "lhs_by_replicate_cycle": {
                str(replicate): {
                    str(cycle): int(
                        sum(
                            1 for row in rows
                            if str(row["branch"]) == "lhs"
                            and int(row["replicate"]) == replicate
                            and int(row["cycle"]) == cycle
                        )
                    )
                    for cycle in required_cycles
                }
                for replicate in required_replicates
            },
            "non_d4_count": len(non_d4_rows),
        },
        "passed": not blockers,
    }
    return protocol_completeness, blockers


def build_emb_34um_dnn_causal_validation_metric_rows(
    *,
    completed_rows_source: object,
    dry_run: bool,
    output_root: Path | str | None = None,
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    device: str | None = None,
    train_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensemble_size = validate_dnn_causal_ensemble_size(ensemble_size)
    rows = _normalize_completed_rows(completed_rows_source)
    source_branch_counts = _source_branch_counts(rows)
    source_parameter_names = _infer_parameter_names(rows)
    blockers: list[str] = []
    if not rows:
        blockers.append("No completed rows were provided.")
        return [], {
            "status": "blocked",
            "passed": False,
            "blockers": blockers,
            "source_branch_counts": source_branch_counts,
            "source_parameter_names": list(source_parameter_names),
        }

    unseen_rows = [row for row in rows if row["branch"] == "unseen_test"]
    if not unseen_rows:
        blockers.append("No unseen_test rows are available for validation scoring.")
        return [], {
            "status": "blocked",
            "passed": False,
            "blockers": blockers,
            "source_branch_counts": source_branch_counts,
            "source_parameter_names": list(source_parameter_names),
        }
    protocol_completeness, completeness_blockers = _assess_protocol_completeness(rows)
    blockers.extend(completeness_blockers)

    force_grid = validate_dnn_causal_force_grid(unseen_rows[0]["force_grid"])
    per_replicate_shared: dict[int, list[dict[str, Any]]] = {}
    per_replicate_al: dict[int, dict[int, list[dict[str, Any]]]] = {}
    per_replicate_lhs: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        branch = str(row["branch"])
        replicate = int(row["replicate"])
        cycle = int(row["cycle"])
        if branch == "shared_initial":
            per_replicate_shared.setdefault(replicate, []).append(row)
        elif branch == "al":
            per_replicate_al.setdefault(replicate, {}).setdefault(cycle, []).append(row)
        elif branch == "lhs":
            per_replicate_lhs.setdefault(replicate, {}).setdefault(cycle, []).append(row)

    resolved_output_root = Path(output_root) if output_root is not None else None
    metric_rows: list[dict[str, Any]] = []
    for replicate in sorted(set(per_replicate_shared) & set(per_replicate_al) & set(per_replicate_lhs)):
        shared_rows = sorted(per_replicate_shared.get(replicate, []), key=lambda item: item["candidate_id"])
        al_cycles = set(per_replicate_al[replicate])
        lhs_cycles = set(per_replicate_lhs[replicate])
        paired_cycles = sorted(cycle for cycle in al_cycles & lhs_cycles if cycle > 0)
        if not shared_rows:
            blockers.append(f"replicate={replicate} has no shared_initial rows.")
            continue
        if not paired_cycles:
            blockers.append(f"replicate={replicate} has no paired AL/LHS cycles.")
            continue

        if dry_run:
            initial_metrics = _evaluate_relative_l2_metrics(
                dry_run=True,
                train_rows=shared_rows,
                unseen_rows=unseen_rows,
                branch="shared_initial",
                cycle=0,
                output_root=None,
                ensemble_size=ensemble_size,
                device=device,
                force_grid=force_grid,
                train_kwargs=train_kwargs,
            )
        else:
            if resolved_output_root is None:
                raise ValueError("output_root is required when dry_run is disabled.")
            initial_metrics = _evaluate_relative_l2_metrics(
                dry_run=False,
                train_rows=shared_rows,
                unseen_rows=unseen_rows,
                branch="shared_initial",
                cycle=0,
                output_root=resolved_output_root / "models" / f"replica-{replicate:03d}" / "cycle-00" / "shared_initial",
                ensemble_size=ensemble_size,
                device=device,
                force_grid=force_grid,
                train_kwargs=train_kwargs,
            )

        for cycle in paired_cycles:
            al_train = list(shared_rows)
            lhs_train = list(shared_rows)
            for prior_cycle in sorted(value for value in al_cycles if value <= cycle):
                al_train.extend(per_replicate_al[replicate][prior_cycle])
            for prior_cycle in sorted(value for value in lhs_cycles if value <= cycle):
                lhs_train.extend(per_replicate_lhs[replicate][prior_cycle])

            al_added_rows = list(per_replicate_al[replicate].get(cycle, []))
            lhs_added_rows = list(per_replicate_lhs[replicate].get(cycle, []))

            for branch, train_rows in (("al", al_train), ("lhs", lhs_train)):
                added_rows = al_added_rows if branch == "al" else lhs_added_rows
                batch_runtime_summary = _runtime_summary(added_rows)
                branch_output_root = (
                    None
                    if resolved_output_root is None
                    else resolved_output_root
                    / "models"
                    / f"replica-{replicate:03d}"
                    / f"cycle-{cycle:02d}"
                    / branch
                )
                metrics = _evaluate_relative_l2_metrics(
                    dry_run=dry_run,
                    train_rows=train_rows,
                    unseen_rows=unseen_rows,
                    branch=branch,
                    cycle=cycle,
                    output_root=branch_output_root,
                    ensemble_size=ensemble_size,
                    device=device,
                    force_grid=force_grid,
                    train_kwargs=train_kwargs,
                )

                replacement_count = int(sum(1 for item in train_rows if _as_bool(item.get("replacement", False))))
                quarantine_count = int(sum(1 for item in train_rows if _as_bool(item.get("quarantine", False))))
                metric_rows.append(
                    {
                        "replicate": int(replicate),
                        "branch": branch,
                        "cycle": int(cycle),
                        "relative_l2": float(metrics["relative_l2_median"]),
                        "relative_l2_mean": float(metrics["relative_l2_mean"]),
                        "relative_l2_median": float(metrics["relative_l2_median"]),
                        "relative_l2_std": float(metrics.get("relative_l2_std", 0.0)),
                        "initial_relative_l2": float(initial_metrics["relative_l2_median"]),
                        "initial_relative_l2_mean": float(initial_metrics["relative_l2_mean"]),
                        "initial_relative_l2_median": float(initial_metrics["relative_l2_median"]),
                        "initial_relative_l2_std": float(initial_metrics.get("relative_l2_std", 0.0)),
                        "train_seconds": float(metrics["train_seconds"]),
                        "score_seconds": float(metrics["score_seconds"]),
                        "initial_train_seconds": float(initial_metrics["train_seconds"]),
                        "initial_score_seconds": float(initial_metrics["score_seconds"]),
                        "per_force_point_error_summary": metrics.get("per_force_point_error_summary", []),
                        "ensemble_mean_prediction_error": dict(metrics.get("ensemble_mean_prediction_error", {})),
                        "ensemble_spread_summary": dict(metrics.get("ensemble_spread_summary", {})),
                        "model_checkpoint_index": dict(metrics.get("model_checkpoint_index", {})),
                        "initial_model_checkpoint_index": dict(initial_metrics.get("model_checkpoint_index", {})),
                        "dnn_training_manifest_path": str(
                            dict(metrics.get("model_checkpoint_index", {})).get("training_manifest_path", "")
                        ),
                        "dnn_training_report_path": str(
                            dict(metrics.get("model_checkpoint_index", {})).get("training_report_path", "")
                        ),
                        "model_checkpoint_paths": list(
                            dict(metrics.get("model_checkpoint_index", {})).get("ensemble_checkpoints", [])
                        ),
                        "replacement": replacement_count > 0,
                        "replacement_count": replacement_count,
                        "quarantine": quarantine_count > 0,
                        "quarantine_count": quarantine_count,
                        "dpd_runtime_summary": batch_runtime_summary,
                        "dpd_runtime_seconds_mean": float(batch_runtime_summary.get("mean", 0.0)),
                        "dpd_runtime_seconds_median": float(batch_runtime_summary.get("median", 0.0)),
                        "dpd_runtime_seconds_std": float(batch_runtime_summary.get("std", 0.0)),
                        "added_candidate_count": len(added_rows),
                        "added_candidate_ids": [str(item.get("candidate_id", "")) for item in added_rows],
                        "added_ka": [float(item["ka"]) for item in added_rows],
                        "added_kb": [float(item["kb"]) for item in added_rows],
                        "added_radp": [item["radp"] for item in added_rows],
                        "added_shell_th": [item["shell_th"] for item in added_rows],
                        "added_selection_sources": [
                            _optional_text(item.get("selection_source", "")) or _optional_text(item.get("source", ""))
                            or _optional_text(item.get("sample_source", ""))
                            for item in added_rows
                        ],
                        "added_candidate_pool_ids": [
                            _optional_text(item.get("candidate_pool_id", "")) or _optional_text(item.get("selection_candidate_id", ""))
                            for item in added_rows
                        ],
                        "added_selection_status": [
                            _optional_text(item.get("selection_status", "")) for item in added_rows
                        ],
                        "added_selection_payloads": [
                            _coerce_mapping(item.get("selection_payload", {}), label="selection_payload")
                            if isinstance(item.get("selection_payload", {}), Mapping)
                            else {}
                            for item in added_rows
                        ],
                        "train_curve_count": len(train_rows),
                        "unseen_curve_count": len(unseen_rows),
                        "finite_test_count": len(unseen_rows),
                        "train_set_signature": _training_signature(train_rows),
                        "ensemble_size": ensemble_size,
                        "dry_run": bool(dry_run),
                    }
                )

    metric_rows.sort(key=lambda item: (int(item["replicate"]), int(item["cycle"]), str(item["branch"])))
    if not metric_rows:
        blockers.append("No AL/LHS metric rows were generated from completed rows.")

    report = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION,
        "status": "blocked" if blockers else "passed",
        "passed": not blockers,
        "blockers": blockers,
        "protocol_completeness": protocol_completeness,
        "row_count": len(metric_rows),
        "replicate_count": len({int(item["replicate"]) for item in metric_rows}),
        "cycle_count": len({int(item["cycle"]) for item in metric_rows}),
        "branch_counts": {
            "al": int(sum(1 for item in metric_rows if str(item["branch"]) == "al")),
            "lhs": int(sum(1 for item in metric_rows if str(item["branch"]) == "lhs")),
        },
        "source_branch_counts": source_branch_counts,
        "source_parameter_names": list(source_parameter_names),
        "dry_run": bool(dry_run),
        "ensemble_size": ensemble_size,
    }
    return metric_rows, report


def write_emb_34um_dnn_causal_validation_metric_artifacts(
    *,
    completed_rows_source: object,
    output_root: str | Path,
    rows_filename: str,
    report_filename: str,
    dry_run: bool,
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    device: str | None = None,
    train_kwargs: Mapping[str, Any] | None = None,
) -> Emb34umDnnCausalValidationMetricArtifacts:
    artifact_dir = Path(output_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows, report = build_emb_34um_dnn_causal_validation_metric_rows(
        completed_rows_source=completed_rows_source,
        dry_run=dry_run,
        output_root=artifact_dir,
        ensemble_size=ensemble_size,
        device=device,
        train_kwargs=train_kwargs,
    )
    source_branch_counts = report.get("source_branch_counts", {})
    if not isinstance(source_branch_counts, Mapping):
        source_branch_counts = {}
    source_parameter_names = report.get("source_parameter_names")
    if not isinstance(source_parameter_names, Sequence) or isinstance(source_parameter_names, (str, bytes, bytearray)):
        source_parameter_names = _infer_parameter_names(rows)

    rows_payload = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION,
        "row_count": len(rows),
        "metadata": {
            "ensemble_size": validate_dnn_causal_ensemble_size(ensemble_size),
            "force_grid": list(validate_dnn_causal_force_grid(EMB_34UM_DNN_CAUSAL_FORCE_GRID)),
            "shared_initial_count": int(source_branch_counts.get("shared_initial", 0)),
            "unseen_test_count": int(source_branch_counts.get("unseen_test", 0)),
            "protocol_completeness": report.get("protocol_completeness", {}),
            "parameter_names": [str(item) for item in source_parameter_names],
            "test_set_source": _infer_test_set_source(rows),
            "dry_run": bool(dry_run),
            "metric_definition": "median force-curve relative L2 over unseen_test curves",
            "model_checkpoint_index": [
                {
                    "replicate": int(row["replicate"]),
                    "branch": str(row["branch"]),
                    "cycle": int(row["cycle"]),
                    **dict(row.get("model_checkpoint_index", {})),
                }
                for row in rows
            ],
        },
        "rows": rows,
    }
    report_payload = {
        **report,
        "rows_path": str(artifact_dir / rows_filename),
    }

    rows_path = artifact_dir / rows_filename
    report_path = artifact_dir / report_filename
    _write_json(rows_path, rows_payload)
    _write_json(report_path, report_payload)
    return Emb34umDnnCausalValidationMetricArtifacts(
        artifact_dir=artifact_dir,
        rows_path=rows_path,
        report_path=report_path,
        rows_payload=rows_payload,
        report_payload=report_payload,
    )


__all__ = [
    "EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME",
    "EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME",
    "Emb34umDnnCausalValidationMetricArtifacts",
    "build_emb_34um_dnn_causal_validation_metric_rows",
    "write_emb_34um_dnn_causal_validation_metric_artifacts",
]
