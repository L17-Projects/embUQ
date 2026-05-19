from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from meso_uq.active_learning.candidate_generation import (
    CANDIDATE_GENERATION_MANIFEST_FILENAME,
    CANDIDATE_GENERATION_PLOT_FILENAME,
    CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME,
    CANDIDATE_GENERATION_REPORT_FILENAME,
    CandidateGenerationConfig,
    generate_candidate_batch,
    write_candidate_generation_artifacts,
)
from meso_uq.active_learning.acquisition import (
    ACQUISITION_MANIFEST_FILENAME,
    ACQUISITION_PLOT_FILENAME,
    ACQUISITION_PLOT_SIDECAR_FILENAME,
    ACQUISITION_REPORT_FILENAME,
    score_acquisition_candidates,
    write_acquisition_artifacts,
)
from meso_uq.active_learning.constraints import validate_active_learning_candidates


CANDIDATE_INPUT_CONFIG_KIND = "candidate"
GENERATION_CONFIG_KIND = "generation"
_VALIDATION_CANDIDATE_REPORT_FILENAME = "candidate_validation_report.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")
    return path


def _load_payload(path: str | Path) -> Any:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}")
    text = raw_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input file {raw_path} is empty.")

    suffix = raw_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        try:
            payload = json.loads(text)
        except Exception:
            payload = yaml.safe_load(text)
    if payload is None:
        raise ValueError(f"Input file {raw_path} does not contain a supported JSON/YAML payload.")
    return payload


def _coerce_sequence(payload: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    if not payload:
        raise ValueError(f"{label} must contain at least one item.")
    return tuple(payload)


def _extract_candidate_payload(payload: object) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, Mapping):
        if "candidates" in payload:
            payloads = payload["candidates"]
        elif "selected_candidates" in payload:
            payloads = payload["selected_candidates"]
        elif "results" in payload:
            payloads = tuple(
                result["candidate"] if isinstance(result, Mapping) and "candidate" in result else result
                for result in _coerce_sequence(payload["results"], label="result payload")
            )
        elif "records" in payload:
            payloads = tuple(
                result["candidate"] if isinstance(result, Mapping) and "candidate" in result else result
                for result in _coerce_sequence(payload["records"], label="result payload")
            )
        elif "candidate_id" in payload and "parameters" in payload:
            payloads = (payload,)
        else:
            candidate_payload = payload.get("candidate") if isinstance(payload, Mapping) else None
            if isinstance(candidate_payload, Mapping):
                payloads = (candidate_payload,)
            elif isinstance(candidate_payload, Sequence) and not isinstance(candidate_payload, (str, bytes, bytearray)):
                payloads = tuple(candidate_payload)
            else:
                raise ValueError(
                    "Candidate payload must expose one of: a top-level sequence,"
                    " 'candidates', 'selected_candidates', 'results', or 'records'."
                )
        return _coerce_sequence(payloads, label="candidate payload")
    return _coerce_sequence(payload, label="candidate payload")


def _normalize_optional_weights(payload: Sequence[str] | None) -> Mapping[str, float] | None:
    if not payload:
        return None
    normalized: dict[str, float] = {}
    for item in payload:
        if "=" not in item:
            raise ValueError(f"Invalid weight '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        value = float(raw_value)
        if not name:
            raise ValueError(f"Invalid weight '{item}'. Name must be non-empty.")
        normalized[name] = value
    return normalized


def _coerce_iteration(value: int | str) -> int:
    iteration = int(value)
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    return iteration


def validate_active_learning_config(
    *,
    input_path: str | Path,
    kind: str = "auto",
) -> tuple[str, dict[str, Any]]:
    payload = _load_payload(input_path)

    normalized_kind = str(kind).lower().strip()
    if normalized_kind == "auto":
        if isinstance(payload, Mapping) and all(
            key in payload
            for key in ("family", "experiment", "dimensions", "count")
        ):
            normalized_kind = GENERATION_CONFIG_KIND
        else:
            normalized_kind = CANDIDATE_INPUT_CONFIG_KIND

    if normalized_kind == GENERATION_CONFIG_KIND:
        config = CandidateGenerationConfig.from_dict(payload)
        return GENERATION_CONFIG_KIND, config.as_dict()

    if normalized_kind != CANDIDATE_INPUT_CONFIG_KIND:
        raise ValueError(f"Unsupported validation kind: {kind!r}.")

    candidates = [item for item in _extract_candidate_payload(payload)]
    report = validate_active_learning_candidates(candidates)
    return CANDIDATE_INPUT_CONFIG_KIND, {
        "valid_candidate_count": len(report.valid_candidates),
        "rejected_candidate_ids": list(report.rejected_candidate_ids),
        "reason_code_counts": dict(report.reason_code_counts),
        "per_candidate_reasons": {
            candidate_id: [{"code": item.code, "message": item.message} for item in reasons]
            for candidate_id, reasons in report.per_candidate_reasons.items()
        },
        "summary": "ok" if not report.rejected_candidate_ids else "invalid",
    }


def run_generate_command(
    *,
    config_path: str | Path,
    output_root: str | Path,
    run_id: str,
    iteration: int | str,
    include_plot: bool = True,
) -> dict[str, Any]:
    config_payload = _load_payload(config_path)
    config = CandidateGenerationConfig.from_dict(config_payload)
    generation_result = generate_candidate_batch(config)
    validation_report = validate_active_learning_candidates(generation_result.candidates)

    artifacts = write_candidate_generation_artifacts(
        output_root=output_root,
        run_id=run_id,
        iteration=_coerce_iteration(iteration),
        result=generation_result,
        include_plot=include_plot,
    )

    validation_report_path = _write_json(
        artifacts.artifact_dir / _VALIDATION_CANDIDATE_REPORT_FILENAME,
        {
            "schema_version": "meso_uq.active_learning.validation_report.v1",
            "run_id": run_id,
            "iteration": _coerce_iteration(iteration),
            "candidate_count": len(generation_result.candidates),
            "valid_candidate_count": len(validation_report.valid_candidates),
            "rejected_candidate_ids": list(validation_report.rejected_candidate_ids),
            "reason_code_counts": dict(validation_report.reason_code_counts),
            "manifest": artifacts.manifest_path.name,
            "report": artifacts.report_path.name,
            "plot": artifacts.plot_path.name,
            "plot_sidecar": artifacts.plot_sidecar_path.name,
        },
    )

    return {
        "command": "generate",
        "run_id": run_id,
        "iteration": _coerce_iteration(iteration),
        "candidate_generation_manifest": artifacts.manifest_path.name,
        "candidate_generation_report": artifacts.report_path.name,
        "candidate_generation_plot": artifacts.plot_path.name,
        "candidate_validation_report": validation_report_path.name,
        "candidate_count": len(generation_result.candidates),
        "valid_candidate_count": len(validation_report.valid_candidates),
        "rejected_candidate_count": len(validation_report.rejected_candidate_ids),
        "invalid": bool(validation_report.rejected_candidate_ids),
        "artifacts": {
            "manifest_path": str(artifacts.manifest_path),
            "report_path": str(artifacts.report_path),
            "plot_path": str(artifacts.plot_path),
            "plot_sidecar_path": str(artifacts.plot_sidecar_path),
            "validation_report_path": str(validation_report_path),
        },
    }


def run_acquire_command(
    *,
    candidates_path: str | Path,
    policy: str,
    output_root: str | Path,
    run_id: str,
    iteration: int | str,
    weights: Sequence[str] | None = None,
    expected_improvement_baseline: float | None = None,
    diversity_paths: Sequence[str] | None = None,
    reference_candidates_path: str | Path | None = None,
    include_plot: bool = True,
) -> dict[str, Any]:
    candidates_payload = _load_payload(candidates_path)
    candidates = _extract_candidate_payload(candidates_payload)

    reference_candidates = None
    if reference_candidates_path is not None:
        raw_reference_payload = _load_payload(reference_candidates_path)
        reference_candidates = _extract_candidate_payload(raw_reference_payload)

    result = score_acquisition_candidates(
        candidates,
        policy=policy,
        weights=_normalize_optional_weights(weights),
        expected_improvement_baseline=expected_improvement_baseline,
        diversity_paths=tuple(diversity_paths) if diversity_paths else None,
        reference_candidates=reference_candidates,
    )

    artifacts = write_acquisition_artifacts(
        output_root=output_root,
        run_id=run_id,
        iteration=_coerce_iteration(iteration),
        result=result,
        include_plot=include_plot,
    )

    return {
        "command": "acquire",
        "run_id": run_id,
        "iteration": _coerce_iteration(iteration),
        "policy": policy,
        "candidate_count": len(candidates),
        "scored_count": len(result.scores),
        "rejected_count": len(result.rejected_candidate_ids),
        "artifacts": {
            "manifest_path": str(artifacts.manifest_path),
            "report_path": str(artifacts.report_path),
            "plot_path": str(artifacts.plot_path),
            "plot_sidecar_path": str(artifacts.plot_sidecar_path),
        },
        "acquisition_manifest": artifacts.manifest_path.name,
        "acquisition_report": artifacts.report_path.name,
        "acquisition_plot": artifacts.plot_path.name,
        "acquisition_plot_sidecar": artifacts.plot_sidecar_path.name,
    }


__all__ = [
    "_extract_candidate_payload",
    "_load_payload",
    "run_acquire_command",
    "run_generate_command",
    "validate_active_learning_config",
    "CANDIDATE_INPUT_CONFIG_KIND",
    "GENERATION_CONFIG_KIND",
    "ACQUISITION_MANIFEST_FILENAME",
    "ACQUISITION_PLOT_FILENAME",
    "ACQUISITION_PLOT_SIDECAR_FILENAME",
    "ACQUISITION_REPORT_FILENAME",
    "CANDIDATE_GENERATION_MANIFEST_FILENAME",
    "CANDIDATE_GENERATION_PLOT_FILENAME",
    "CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME",
    "CANDIDATE_GENERATION_REPORT_FILENAME",
    "_VALIDATION_CANDIDATE_REPORT_FILENAME",
]
