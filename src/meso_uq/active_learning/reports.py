"""Concise Active Learning report helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json


ACTIVE_LEARNING_CONCISE_REPORT_SCHEMA_VERSION = "meso_uq.active_learning.concise_report.v1"
ACTIVE_LEARNING_REPORT_FILENAME = "active_learning_concise_report.json"
ACTIVE_LEARNING_REPORT_MARKDOWN_FILENAME = "active_learning_concise_report.md"


def _coerce_report_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.name:
        raise ValueError(f"{label} must be a non-empty path.")
    return path


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string.")
    return text


def _coerce_mapping(payload: object, *, label: str) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    raise TypeError(f"{label} must be a mapping.")


def _coerce_sequence(payload: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence.")
    return tuple(payload)


def _coerce_iteration(value: int | str) -> str:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("iteration must be non-negative.")
        return f"iter_{value:04d}"
    token = str(value).strip()
    if not token:
        raise ValueError("iteration must be a non-empty string.")
    if token.startswith("iter_"):
        return token
    if token.isdigit():
        return f"iter_{int(token):04d}"
    return token.replace("-", "_")


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    return str(value)


def _load_payload(payload_or_path: Mapping[str, Any] | str | Path | None, *, label: str) -> dict[str, Any] | None:
    if payload_or_path is None:
        return None
    if isinstance(payload_or_path, Mapping):
        return _coerce_mapping(payload_or_path, label=f"{label} artifact")
    if isinstance(payload_or_path, (str, Path)):
        path = _coerce_report_path(payload_or_path, label=f"{label} artifact")
        return _coerce_mapping(json.loads(path.read_text(encoding="utf-8")), label=f"{label} artifact")
    raise TypeError(f"{label} artifact must be a mapping or path.")


def _append_candidate(
    candidate_id: str,
    candidate_hash: str | None,
    *,
    candidate_ids: list[str],
    candidate_hashes: list[str],
    seen: set[str],
) -> None:
    if candidate_id in seen:
        return
    seen.add(candidate_id)
    candidate_ids.append(candidate_id)
    if candidate_hash:
        candidate_hashes.append(candidate_hash)


def _increment_counter(name: str | None, counter: Counter[str]) -> None:
    if not name:
        return
    counter[_coerce_text(name, label="name")] += 1


def _parse_candidate_generation(
    report: Mapping[str, Any] | None,
    *,
    candidate_ids: list[str],
    candidate_hashes: list[str],
    family_distribution: Counter[str],
    experiment_distribution: Counter[str],
) -> None:
    if report is None:
        return
    seen = set(candidate_ids)

    candidate_hash_payload = report.get("candidate_hashes")
    if isinstance(candidate_hash_payload, Mapping):
        for candidate_id, candidate_hash in candidate_hash_payload.items():
            _append_candidate(
                str(candidate_id),
                str(candidate_hash),
                candidate_ids=candidate_ids,
                candidate_hashes=candidate_hashes,
                seen=seen,
            )
    elif isinstance(candidate_hash_payload, Sequence) and not isinstance(candidate_hash_payload, (str, bytes, bytearray)):
        for item in candidate_hash_payload:
            if isinstance(item, Mapping):
                candidate_id = item.get("candidate_id")
                if candidate_id is None:
                    continue
                candidate_hash = item.get("candidate_hash") or item.get("hash")
                _append_candidate(
                    str(candidate_id),
                    str(candidate_hash) if candidate_hash is not None else None,
                    candidate_ids=candidate_ids,
                    candidate_hashes=candidate_hashes,
                    seen=seen,
                )
            else:
                _append_candidate(
                    str(item),
                    None,
                    candidate_ids=candidate_ids,
                    candidate_hashes=candidate_hashes,
                    seen=seen,
                )

    for candidate_id in report.get("candidate_ids", ()):
        _append_candidate(
            str(candidate_id),
            None,
            candidate_ids=candidate_ids,
            candidate_hashes=candidate_hashes,
            seen=seen,
        )

    for family, count in _coerce_mapping(report.get("family_distribution", {}), label="family_distribution").items():
        if isinstance(count, int):
            family_distribution.update({_coerce_text(family, label="family").lower(): count})
    for experiment, count in _coerce_mapping(report.get("experiment_distribution", {}), label="experiment_distribution").items():
        if isinstance(count, int):
            experiment_distribution.update({_coerce_text(experiment, label="experiment"): count})


def _parse_acquisition_report(
    report: Mapping[str, Any] | None,
    *,
    acquisition_terms: dict[str, Any],
    family_distribution: Counter[str],
    experiment_distribution: Counter[str],
    candidate_ids: list[str],
    candidate_hashes: list[str],
) -> None:
    if report is None:
        return
    seen = set(candidate_ids)
    weights: dict[str, float] = {}
    for key, value in _coerce_mapping(report.get("weights", {}), label="acquisition weights").items():
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    if "policy" in report:
        acquisition_terms["policy"] = report.get("policy")
    if weights:
        acquisition_terms["weights"] = weights
    for key in ("candidate_count", "scored_count"):
        value = report.get(key)
        if isinstance(value, int):
            acquisition_terms[key] = value

    for candidate_id in report.get("candidate_ids", ()):
        _append_candidate(
            str(candidate_id),
            None,
            candidate_ids=candidate_ids,
            candidate_hashes=candidate_hashes,
            seen=seen,
        )

    for family, count in report.get("family_distribution", {}).items():
        if isinstance(count, int):
            family_distribution.update({_coerce_text(family, label="family").lower(): count})
    for experiment, count in report.get("experiment_distribution", {}).items():
        if isinstance(count, int):
            experiment_distribution.update({_coerce_text(experiment, label="experiment"): count})


def _parse_selected_batch(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    selected_candidates = tuple(report.get("selected_candidates", ()))
    selected_scores = tuple(report.get("selected_scores", ()))
    selected_candidate_hashes = tuple(report.get("selected_candidate_hashes", ()))
    return {
        "selected_count": len(selected_candidates),
        "selected_candidates": [str(candidate_id) for candidate_id in selected_candidates],
        "selected_scores": [float(score) for score in selected_scores if isinstance(score, (int, float))],
        "selected_candidate_hashes": [str(item) for item in selected_candidate_hashes],
    }


def _parse_constraint_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    reason_code_counts = _coerce_mapping(report.get("reason_code_counts", {}), label="reason code counts")
    parsed: dict[str, Any] = {
        "reason_code_counts": {str(key): int(value) for key, value in reason_code_counts.items()},
        "per_candidate_reasons": {},
        "rejected_candidate_ids": [str(candidate_id) for candidate_id in _coerce_sequence(report.get("rejected_candidate_ids", ()), label="rejected candidate ids")],
    }

    per_candidate = _coerce_mapping(report.get("per_candidate_reasons", {}), label="per-candidate reasons")
    if not isinstance(per_candidate, dict):
        return parsed

    for candidate_id, reasons in per_candidate.items():
        normalized = reasons
        if isinstance(normalized, Mapping):
            normalized = (normalized,)
        elif not isinstance(normalized, Sequence) or isinstance(normalized, (str, bytes, bytearray)):
            normalized = (normalized,)
        parsed["per_candidate_reasons"][str(candidate_id)] = []
        for reason in _coerce_sequence(normalized, label=f"candidate {candidate_id} reasons"):
            if isinstance(reason, Mapping):
                parsed["per_candidate_reasons"][str(candidate_id)].append(
                    {"code": str(reason.get("code", "")), "message": str(reason.get("message", ""))}
                )
            else:
                parsed["per_candidate_reasons"][str(candidate_id)].append(
                    {"code": str(reason), "message": ""}
                )
    return parsed


def _parse_simulation_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = {
        "record_count": int(report.get("record_count", 0)) if isinstance(report.get("record_count"), int) else 0,
        "family_distribution": {},
        "experiment_distribution": {},
        "run_status_distribution": {},
    }
    records = report.get("records", ())
    if not isinstance(records, (list, tuple)):
        return summary

    family_counter: Counter[str] = Counter()
    experiment_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()

    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        _increment_counter(raw_record.get("family"), family_counter)
        _increment_counter(raw_record.get("experiment"), experiment_counter)
        _increment_counter(raw_record.get("run_status", raw_record.get("status", "unknown")), status_counter)

    summary["record_count"] = summary["record_count"] or len(records)
    summary["family_distribution"] = dict(family_counter)
    summary["experiment_distribution"] = dict(experiment_counter)
    summary["run_status_distribution"] = dict(status_counter)
    return summary


def _parse_result_ingestion_status(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    status_counts = _coerce_mapping(report.get("status_counts", {}), label="status_counts")
    family_counts = _coerce_mapping(report.get("family_counts", {}), label="family_counts")
    return {
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "family_counts": {str(key): int(value) for key, value in family_counts.items()},
        "record_count": int(report.get("record_count", 0)),
        "ingestion_hash": report.get("ingestion_hash"),
        "missing_refs": [dict(entry) for entry in _coerce_sequence(report.get("missing_refs", ()), label="missing_refs") if isinstance(entry, Mapping)],
    }


def _parse_constraint_candidates(report: Mapping[str, Any] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if report is None:
        return (), ()
    valid_candidates = [
        str(entry["candidate_id"])
        for entry in _coerce_sequence(report.get("valid_candidates", ()), label="valid candidates")
        if isinstance(entry, Mapping) and entry.get("candidate_id") is not None
    ]
    rejected_candidates = [
        str(candidate_id) for candidate_id in _coerce_sequence(report.get("rejected_candidate_ids", ()), label="rejected_candidate_ids")
    ]
    return tuple(valid_candidates), tuple(rejected_candidates)


def _collect_candidates_from_records(
    report: Mapping[str, Any] | None,
    *,
    family_distribution: Counter[str],
    experiment_distribution: Counter[str],
    candidate_ids: list[str],
    candidate_hashes: list[str],
    seen_candidate_ids: set[str],
    include_candidate_hash: bool = False,
) -> None:
    if report is None:
        return
    records = report.get("records", ())
    if not isinstance(records, (list, tuple)):
        return
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        candidate_id = raw_record.get("candidate_id")
        if candidate_id is not None:
            candidate_id_text = str(candidate_id)
            is_new_candidate = candidate_id_text not in seen_candidate_ids
            _append_candidate(
                candidate_id_text,
                str(raw_record.get("candidate_hash", "")) if include_candidate_hash else None,
                candidate_ids=candidate_ids,
                candidate_hashes=candidate_hashes,
                seen=seen_candidate_ids,
            )
            if is_new_candidate:
                seen_candidate_ids.add(candidate_id_text)
                _increment_counter(raw_record.get("family"), family_distribution)
                _increment_counter(raw_record.get("experiment"), experiment_distribution)


def _collect_plot_paths(paths: Sequence[str | Path | Mapping[str, Any] | None]) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if raw is None:
            continue
        if isinstance(raw, (str, Path)):
            value = str(raw)
            if value not in seen:
                seen.add(value)
                collected.append(value)
            continue
        if isinstance(raw, Mapping):
            mapping = _coerce_mapping(raw, label="plot artifact")
            candidates: tuple[Any, ...] = ()
            for key in ("plot", "plot_path", "path", "artifact"):
                if key in mapping:
                    value = mapping[key]
                    if isinstance(value, Mapping):
                        nested_values: list[Any] = []
                        for nested_key in ("path", "plot", "uri"):
                            nested_value = value.get(nested_key)
                            if isinstance(nested_value, (str, Path)):
                                nested_values.append(nested_value)
                        candidates = tuple(nested_values)
                        break
                    candidates = (value,)
                    break
            for value in candidates:
                if isinstance(value, (str, Path)):
                    text = str(value)
                    if text not in seen:
                        seen.add(text)
                        collected.append(text)
    return collected


def _report_iteration_directory(output_root: str | Path, run_id: str, iteration: int | str) -> Path:
    return Path(output_root) / run_id / "iterations" / _coerce_iteration(iteration)


def build_active_learning_concise_report(
    *,
    run_id: str,
    iteration: int | str,
    candidate_generation_report: Mapping[str, Any] | str | Path | None = None,
    acquisition_report: Mapping[str, Any] | str | Path | None = None,
    constraints_report: Mapping[str, Any] | str | Path | None = None,
    selected_batch_report: Mapping[str, Any] | str | Path | None = None,
    simulation_manifest: Mapping[str, Any] | str | Path | None = None,
    quarantine_summary: Mapping[str, Any] | str | Path | None = None,
    result_ingestion_report: Mapping[str, Any] | str | Path | None = None,
    validation_plot_paths: Sequence[str | Path | Mapping[str, Any] | None] | None = None,
    reproducibility_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_generation = _load_payload(candidate_generation_report, label="candidate generation")
    acquisition = _load_payload(acquisition_report, label="acquisition")
    constraints = _load_payload(constraints_report, label="constraints")
    selected_batch = _load_payload(selected_batch_report, label="selected batch")
    simulation = _load_payload(simulation_manifest, label="simulation manifest")
    quarantine = _load_payload(quarantine_summary, label="quarantine")
    ingestion = _load_payload(result_ingestion_report, label="result ingestion")

    candidate_ids: list[str] = []
    candidate_hashes: list[str] = []
    family_distribution: Counter[str] = Counter()
    experiment_distribution: Counter[str] = Counter()
    seen_candidate_ids: set[str] = set()

    acquisition_terms: dict[str, Any] = {}

    _parse_candidate_generation(
        candidate_generation,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
    )
    _parse_acquisition_report(
        acquisition,
        acquisition_terms=acquisition_terms,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
    )
    seen_candidate_ids.update(candidate_ids)
    _collect_candidates_from_records(
        selected_batch,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
        seen_candidate_ids=seen_candidate_ids,
        include_candidate_hash=True,
    )
    _collect_candidates_from_records(
        acquisition,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
        seen_candidate_ids=seen_candidate_ids,
    )
    _collect_candidates_from_records(
        simulation,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
        seen_candidate_ids=seen_candidate_ids,
    )
    _collect_candidates_from_records(
        ingestion,
        family_distribution=family_distribution,
        experiment_distribution=experiment_distribution,
        candidate_ids=candidate_ids,
        candidate_hashes=candidate_hashes,
        include_candidate_hash=True,
        seen_candidate_ids=seen_candidate_ids,
    )

    valid_constraint, rejected_constraint = _parse_constraint_candidates(constraints)
    seen = set(candidate_ids)
    for candidate_id in valid_constraint:
        _append_candidate(
            candidate_id,
            None,
            candidate_ids=candidate_ids,
            candidate_hashes=candidate_hashes,
            seen=seen,
        )
    for candidate_id in rejected_constraint:
        _append_candidate(
            candidate_id,
            None,
            candidate_ids=candidate_ids,
            candidate_hashes=candidate_hashes,
            seen=seen,
        )

    return {
        "schema_version": ACTIVE_LEARNING_CONCISE_REPORT_SCHEMA_VERSION,
        "run_id": _coerce_text(run_id, label="run_id"),
        "iteration": _coerce_iteration(iteration),
        "candidate_ids": candidate_ids,
        "candidate_hashes": list(dict.fromkeys(candidate_hashes)),
        "family_distribution": dict(family_distribution),
        "experiment_distribution": dict(experiment_distribution),
        "constraint_reasons": _parse_constraint_report(constraints),
        "acquisition_terms": acquisition_terms,
        "selected_batch_summary": _parse_selected_batch(selected_batch),
        "simulation_summary": _parse_simulation_summary(simulation),
        "quarantine_summary": quarantine or {},
        "result_ingestion_status": _parse_result_ingestion_status(ingestion),
        "validation_plot_paths": _collect_plot_paths(
            validation_plot_paths if validation_plot_paths is not None else (),
        ),
        "reproducibility_metadata": _normalize_json(dict(reproducibility_metadata or {})),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_json(dict(payload)), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Active-Learning Concise Report",
        "",
        f"run_id: {payload.get('run_id', '')}",
        f"iteration: {payload.get('iteration', '')}",
        "",
        "## Candidates",
        f"- count: {len(payload.get('candidate_ids', ())) if isinstance(payload.get('candidate_ids'), list) else 0}",
        f"- hashes: {len(payload.get('candidate_hashes', ()))}",
        "",
        "## Family Distribution",
    ]
    family_lines = [f"- {family}: {count}" for family, count in sorted(payload.get("family_distribution", {}).items())]
    lines.extend(family_lines if family_lines else ["- none"])
    lines.extend(
        [
            "",
            "## Experiment Distribution",
        ]
    )
    experiment_lines = [
        f"- {experiment}: {count}" for experiment, count in sorted(payload.get("experiment_distribution", {}).items())
    ]
    lines.extend(experiment_lines if experiment_lines else ["- none"])
    lines.extend(
        [
            "",
            "## Constraint Reasons",
            f"- rejected_candidates: {len(payload.get('constraint_reasons', {}).get('rejected_candidate_ids', []))}",
            f"- reason_code_counts: {payload.get('constraint_reasons', {}).get('reason_code_counts', {})}",
            "",
            "## Acquisition Terms",
            f"- policy: {payload.get('acquisition_terms', {}).get('policy', 'unknown')}",
            f"- weights: {payload.get('acquisition_terms', {}).get('weights', {})}",
            "",
            "## Selected Batch",
            f"- selected_count: {payload.get('selected_batch_summary', {}).get('selected_count', 0)}",
            f"- selected_candidates: {payload.get('selected_batch_summary', {}).get('selected_candidates', [])}",
            "",
            "## Simulation Summary",
            f"- record_count: {payload.get('simulation_summary', {}).get('record_count', 0)}",
            f"- run_status_distribution: {payload.get('simulation_summary', {}).get('run_status_distribution', {})}",
            "",
            "## Quarantine Summary",
            f"- failed_count: {payload.get('quarantine_summary', {}).get('failed_count', 0)}",
            "",
            "## Result Ingestion",
            f"- status_counts: {payload.get('result_ingestion_status', {}).get('status_counts', {})}",
            f"- record_count: {payload.get('result_ingestion_status', {}).get('record_count', 0)}",
            "",
            "## Validation Plot Paths",
        ]
    )
    plot_lines = [f"- {path}" for path in payload.get("validation_plot_paths", ())]
    lines.extend(plot_lines if plot_lines else ["- none"])
    lines.extend(["", "## Reproducibility", f"- metadata: {payload.get('reproducibility_metadata', {})}"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ActiveLearningConciseReportArtifacts:
    artifact_dir: Path
    json_path: Path
    markdown_path: Path
    report: dict[str, Any]


def write_active_learning_concise_report(
    *,
    output_root: str | Path,
    run_id: str,
    iteration: int | str,
    candidate_generation_report: Mapping[str, Any] | str | Path | None = None,
    acquisition_report: Mapping[str, Any] | str | Path | None = None,
    constraints_report: Mapping[str, Any] | str | Path | None = None,
    selected_batch_report: Mapping[str, Any] | str | Path | None = None,
    simulation_manifest: Mapping[str, Any] | str | Path | None = None,
    quarantine_summary: Mapping[str, Any] | str | Path | None = None,
    result_ingestion_report: Mapping[str, Any] | str | Path | None = None,
    validation_plot_paths: Sequence[str | Path | Mapping[str, Any] | None] | None = None,
    reproducibility_metadata: Mapping[str, Any] | None = None,
) -> ActiveLearningConciseReportArtifacts:
    run_id_text = _coerce_text(run_id, label="run_id")
    artifact_dir = _report_iteration_directory(output_root, run_id_text, iteration)
    json_path = artifact_dir / ACTIVE_LEARNING_REPORT_FILENAME
    markdown_path = artifact_dir / ACTIVE_LEARNING_REPORT_MARKDOWN_FILENAME

    report = build_active_learning_concise_report(
        run_id=run_id_text,
        iteration=iteration,
        candidate_generation_report=candidate_generation_report,
        acquisition_report=acquisition_report,
        constraints_report=constraints_report,
        selected_batch_report=selected_batch_report,
        simulation_manifest=simulation_manifest,
        quarantine_summary=quarantine_summary,
        result_ingestion_report=result_ingestion_report,
        validation_plot_paths=validation_plot_paths,
        reproducibility_metadata=reproducibility_metadata,
    )
    _write_json(json_path, report)
    _write_markdown(markdown_path, report)
    return ActiveLearningConciseReportArtifacts(
        artifact_dir=artifact_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )


def _build_concise_report(
    run_id: str,
    iteration: int | str,
    candidate_generation_report: Mapping[str, Any] | str | Path | None = None,
    acquisition_report: Mapping[str, Any] | str | Path | None = None,
    constraints_report: Mapping[str, Any] | str | Path | None = None,
    selected_batch_report: Mapping[str, Any] | str | Path | None = None,
    simulation_manifest: Mapping[str, Any] | str | Path | None = None,
    quarantine_summary: Mapping[str, Any] | str | Path | None = None,
    result_ingestion_report: Mapping[str, Any] | str | Path | None = None,
    validation_plot_paths: Sequence[str | Path | Mapping[str, Any] | None] | None = None,
    reproducibility_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_active_learning_concise_report(
        run_id=run_id,
        iteration=iteration,
        candidate_generation_report=candidate_generation_report,
        acquisition_report=acquisition_report,
        constraints_report=constraints_report,
        selected_batch_report=selected_batch_report,
        simulation_manifest=simulation_manifest,
        quarantine_summary=quarantine_summary,
        result_ingestion_report=result_ingestion_report,
        validation_plot_paths=validation_plot_paths,
        reproducibility_metadata=reproducibility_metadata,
    )


__all__ = [
    "ACTIVE_LEARNING_CONCISE_REPORT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_REPORT_FILENAME",
    "ACTIVE_LEARNING_REPORT_MARKDOWN_FILENAME",
    "ActiveLearningConciseReportArtifacts",
    "_build_concise_report",
    "build_active_learning_concise_report",
    "write_active_learning_concise_report",
]
