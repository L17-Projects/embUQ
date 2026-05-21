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
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    validate_dnn_causal_ensemble_size,
    validate_dnn_causal_force_grid,
)
from meso_uq.active_learning.emb_34um_dnn_surrogate import (
    load_ensemble_member_predictions,
    train_emb_34um_dnn_surrogate_ensemble,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_metric_rows.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME = "emb_34um_dnn_causal_validation_rows.json"
EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME = "emb_34um_dnn_causal_validation_metric_report.json"


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

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    ka = payload.get("ka", parameters.get("ka"))
    kb = payload.get("kb", parameters.get("kb"))
    if ka is None or kb is None:
        raise ValueError(f"completed_rows[{source_index}] is missing ka/kb.")

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
        "ka": _coerce_float(ka, label="ka"),
        "kb": _coerce_float(kb, label="kb"),
        "force_grid": resolved_force_grid,
        "reference_curve": resolved_curve,
        "replacement": _as_bool(payload.get("replacement", False)),
        "quarantine": _as_bool(payload.get("quarantine", payload.get("quarantined", False))),
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


def _training_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    joined = "\n".join(sorted(str(item.get("candidate_id", "")) for item in rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _canonical_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(item["candidate_id"]),
            "parameters": {"ka": float(item["ka"]), "kb": float(item["kb"])},
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
) -> dict[str, float]:
    ka_center = statistics.mean(math.log10(float(item["ka"])) for item in train_rows)
    kb_center = statistics.mean(math.log10(float(item["kb"])) for item in train_rows)
    metrics: list[float] = []
    branch_bias = 0.003 if branch == "lhs" else 0.0
    for row in unseen_rows:
        ka_delta = math.log10(float(row["ka"])) - ka_center
        kb_delta = math.log10(float(row["kb"])) - kb_center
        scale = 0.02 + 0.005 * math.hypot(ka_delta, kb_delta) + 0.001 * float(cycle) + branch_bias
        predicted = [float(value) * (1.0 + scale) for value in row["reference_curve"]]
        metrics.append(_relative_l2(predicted, row["reference_curve"]))

    train_seconds = 1.0e-4 * len(train_rows) * EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    score_seconds = 5.0e-5 * len(unseen_rows) * EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    return {
        "relative_l2_mean": float(statistics.mean(metrics)),
        "relative_l2_median": float(statistics.median(metrics)),
        "train_seconds": float(train_seconds),
        "score_seconds": float(score_seconds),
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
) -> dict[str, float]:
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
    metrics: list[float] = []
    for unseen in unseen_rows:
        predicted_curve, _ = load_ensemble_member_predictions(
            fit,
            force_grid,
            ka=float(unseen["ka"]),
            kb=float(unseen["kb"]),
        )
        metrics.append(_relative_l2(predicted_curve, unseen["reference_curve"]))
    score_seconds = perf_counter() - score_start
    return {
        "relative_l2_mean": float(statistics.mean(metrics)),
        "relative_l2_median": float(statistics.median(metrics)),
        "train_seconds": float(train_seconds),
        "score_seconds": float(score_seconds),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    blockers: list[str] = []
    if not rows:
        blockers.append("No completed rows were provided.")
        return [], {"status": "blocked", "passed": False, "blockers": blockers}

    unseen_rows = [row for row in rows if row["branch"] == "unseen_test"]
    if not unseen_rows:
        blockers.append("No unseen_test rows are available for validation scoring.")
        return [], {"status": "blocked", "passed": False, "blockers": blockers}

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

        for cycle in paired_cycles:
            al_train = list(shared_rows)
            lhs_train = list(shared_rows)
            for prior_cycle in sorted(value for value in al_cycles if value <= cycle):
                al_train.extend(per_replicate_al[replicate][prior_cycle])
            for prior_cycle in sorted(value for value in lhs_cycles if value <= cycle):
                lhs_train.extend(per_replicate_lhs[replicate][prior_cycle])

            for branch, train_rows in (("al", al_train), ("lhs", lhs_train)):
                if dry_run:
                    metrics = _dry_run_metrics(
                        train_rows=train_rows,
                        unseen_rows=unseen_rows,
                        branch=branch,
                        cycle=cycle,
                    )
                else:
                    if resolved_output_root is None:
                        raise ValueError("output_root is required when dry_run is disabled.")
                    branch_output_root = (
                        resolved_output_root
                        / "models"
                        / f"replica-{replicate:03d}"
                        / f"cycle-{cycle:02d}"
                        / branch
                    )
                    metrics = _real_metrics(
                        train_rows=train_rows,
                        unseen_rows=unseen_rows,
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
                        "train_seconds": float(metrics["train_seconds"]),
                        "score_seconds": float(metrics["score_seconds"]),
                        "replacement": replacement_count > 0,
                        "replacement_count": replacement_count,
                        "quarantine": quarantine_count > 0,
                        "quarantine_count": quarantine_count,
                        "train_curve_count": len(train_rows),
                        "unseen_curve_count": len(unseen_rows),
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
        "row_count": len(metric_rows),
        "replicate_count": len({int(item["replicate"]) for item in metric_rows}),
        "cycle_count": len({int(item["cycle"]) for item in metric_rows}),
        "branch_counts": {
            "al": int(sum(1 for item in metric_rows if str(item["branch"]) == "al")),
            "lhs": int(sum(1 for item in metric_rows if str(item["branch"]) == "lhs")),
        },
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

    rows_payload = {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_SCHEMA_VERSION,
        "row_count": len(rows),
        "metadata": {
            "ensemble_size": validate_dnn_causal_ensemble_size(ensemble_size),
            "force_grid": list(validate_dnn_causal_force_grid(EMB_34UM_DNN_CAUSAL_FORCE_GRID)),
            "dry_run": bool(dry_run),
            "metric_definition": "median force-curve relative L2 over unseen_test curves",
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
