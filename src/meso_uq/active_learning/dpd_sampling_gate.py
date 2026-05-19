"""Dry-run gate for Active Learning-selected DPD candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, as_candidate, candidate_hash
from meso_uq.dpd_sampling.boundary import build_and_render_dpd_sampling_batch
from meso_uq.dpd_sampling.contracts import DPDDataRef, DPDSamplingRenderResult

ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION = "meso_uq.active_learning.dpd_dry_run_gate.v1"
ACTIVE_LEARNING_DPD_DRY_RUN_GATE_MANIFEST_FILENAME = "dpd_sampling_gate_manifest.json"
ACTIVE_LEARNING_DPD_DRY_RUN_GATE_REPORT_FILENAME = "dpd_sampling_gate_report.json"


def _coerce_nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty value.")
    return text


def _coerce_path_token(value: object, *, field_name: str) -> str:
    token = _coerce_nonempty_text(value, field_name=field_name).replace("/", "_")
    token = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token).strip("._")
    if not token:
        raise ValueError(f"{field_name} resolved to an empty path token.")
    return token


def _coerce_candidates(value: Sequence[object]) -> tuple[Candidate, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError("selected_candidates must be a non-empty sequence.")
    normalized = tuple(as_candidate(item) for item in value)
    if not normalized:
        raise ValueError("selected_candidates must contain at least one candidate.")
    return normalized


def _coerce_paths(value: Sequence[str | Path] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    paths: list[str] = []
    for item in value:
        path = _coerce_nonempty_text(item, field_name="validation_plot_path")
        paths.append(path)
    return tuple(paths)


def _normalize_candidate_by_token(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    seen: set[str] = set()
    normalized: dict[str, Candidate] = {}
    for candidate in candidates:
        token = _coerce_path_token(candidate.candidate_id, field_name="candidate_id")
        if token in seen:
            raise ValueError(f"Duplicate selected candidate id after normalization: {token!r}.")
        seen.add(token)
        normalized[token] = candidate
    return normalized


def _candidate_records(
    *,
    request_candidates: tuple[Candidate, ...],
    manifest_candidates: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    by_token = _normalize_candidate_by_token(request_candidates)

    records: list[dict[str, Any]] = []
    for manifest in manifest_candidates:
        candidate_id = manifest.candidate_id
        candidate = by_token.get(candidate_id)
        if candidate is None:
            candidate = by_token.get(_coerce_path_token(candidate_id, field_name="candidate_id"))
        if candidate is None:
            raise ValueError(f"Missing selected candidate payload for manifest candidate {candidate_id!r}.")

        experiment = manifest.normalized_payload.get("experiment")
        experiment_text = _coerce_nonempty_text(
            experiment,
            field_name=f"candidate {candidate_id!r} experiment",
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash(candidate),
                "family": manifest.family,
                "experiment": experiment_text,
                "platform": manifest.platform,
                "output_root": str(manifest.output_root),
                "expected_hdf5_contract_fields": [
                    item.as_manifest() for item in manifest.expected_hdf5_datasets
                ],
            }
        )
    return tuple(records)


def _candidate_experiments(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_coerce_nonempty_text(record["experiment"], field_name="experiment") for record in records))


def _expected_hdf5_refs(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        for item in record.get("expected_hdf5_contract_fields", ()):
            payload = dict(item)
            family = str(payload.get("family", "")).strip()
            dataset_id = str(payload.get("dataset_id", "")).strip()
            hdf5_path = str(payload.get("hdf5_path", "")).strip()
            if not family or not dataset_id or not hdf5_path:
                raise ValueError("Expected HDF5 contract items must include family, dataset_id, and hdf5_path.")
            key = (family, dataset_id, hdf5_path)
            if key in seen:
                continue
            seen.add(key)
            manifest_path = payload.get("manifest_path")
            refs.append(
                DPDDataRef(
                    family=family,
                    dataset_id=dataset_id,
                    hdf5_path=hdf5_path,
                    manifest_path=str(manifest_path),
                ).as_manifest()
            )
    return tuple(refs)


def _coerce_plot_paths(paths: Sequence[Path]) -> tuple[str, ...]:
    return tuple(str(item) for item in paths if item is not None)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            default=lambda value: str(value) if isinstance(value, Path) else value,
        ),
        encoding="utf-8",
    )
    return path


def _build_pass_fail_criteria(
    *,
    candidate_count: int,
    candidate_experiments: Sequence[str],
    render_result: DPDSamplingRenderResult,
) -> tuple[dict[str, bool], bool, tuple[str, ...]]:
    criteria = {
        "candidate_count_positive": candidate_count > 0,
        "single_experiment": len(set(candidate_experiments)) <= 1,
        "single_family": len({render_result.batch_request.family}) == 1,
        "render_only_mode": not bool(render_result.validation_report.submission.get("submitted")),
        "no_submission_commands": not bool(render_result.validation_report.submission.get("submission_commands")),
    }
    failures: list[str] = []
    if not criteria["candidate_count_positive"]:
        failures.append("no_candidates")
    if not criteria["single_experiment"]:
        failures.append("multiple_experiments")
    if not criteria["render_only_mode"]:
        failures.append("submission_detected")
    if not criteria["no_submission_commands"]:
        failures.append("submission_commands_present")
    return criteria, (len(failures) == 0), tuple(failures)


def _derive_render_platforms(request_platform: str, platforms: Sequence[str] | None) -> tuple[str, ...]:
    if platforms:
        ordered = []
        for platform in platforms:
            name = _coerce_nonempty_text(platform, field_name="render_platform")
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)
    return (request_platform,)


@dataclass(frozen=True)
class ActiveLearningDPDSamplingGateArtifacts:
    """Artifacts produced by the AL DPD dry-run gate."""

    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    plot_path: Path | None
    plot_sidecar_path: Path | None
    batch_request: Any
    render_result: DPDSamplingRenderResult
    manifest: dict[str, Any]
    report: dict[str, Any]


def build_and_render_active_learning_dpd_sampling_gate(
    selected_candidates: Sequence[object],
    *,
    run_id: object,
    iteration: object,
    platform: object,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object] | None = None,
    defaults: Mapping[str, object] | None = None,
    campaign_root: str | Path | None = None,
    batch_id: str | None = None,
    platforms: Iterable[object] | None = None,
    overwrite: bool = False,
    upstream_validation_plot_paths: Sequence[str | Path] | None = None,
) -> ActiveLearningDPDSamplingGateArtifacts:
    """Render AL-selected GV/EMB candidates into scheduler-ready DPD artifacts only."""

    candidates = _coerce_candidates(selected_candidates)
    resolved_platform = str(platform).strip().lower()
    if not resolved_platform:
        raise ValueError("platform must be a non-empty string.")
    run_id_text = _coerce_nonempty_text(run_id, field_name="run_id")
    iteration_text = _coerce_nonempty_text(iteration, field_name="iteration")
    upstream_plots = _coerce_paths(upstream_validation_plot_paths)

    render_result = build_and_render_dpd_sampling_batch(
        candidates,
        run_id=run_id_text,
        iteration=iteration_text,
        platform=resolved_platform,
        walltime=walltime,
        gpu_count=gpu_count,
        provenance_tags=provenance_tags,
        defaults=defaults,
        campaign_root=campaign_root,
        batch_id=batch_id,
        platforms=platforms,
        overwrite=overwrite,
    )

    candidate_records = _candidate_records(
        request_candidates=candidates,
        manifest_candidates=render_result.batch_request.candidate_manifests,
    )
    candidate_count = len(candidate_records)
    candidate_experiments = _candidate_experiments(candidate_records)

    scheduler_boundary = {
        "mode": "render_only",
        "platform": resolved_platform,
        "submission": render_result.validation_report.submission,
    }
    scheduler_boundary["render_platforms"] = list(
        _derive_render_platforms(
            request_platform=render_result.batch_request.platform,
            platforms=tuple(_coerce_nonempty_text(item, field_name="platform") for item in (platforms or ())),
        )
    )

    expected_refs = _expected_hdf5_refs(candidate_records)
    criteria, passed, failure_reasons = _build_pass_fail_criteria(
        candidate_count=candidate_count,
        candidate_experiments=candidate_experiments,
        render_result=render_result,
    )

    render_plot_paths = _coerce_plot_paths(render_result.plot_paths) if render_result.plot_paths else ()
    render_plot_sidecar_paths = _coerce_plot_paths(render_result.plot_sidecar_paths) if render_result.plot_sidecar_paths else ()

    manifest = {
        "schema_version": ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION,
        "run_id": run_id_text,
        "iteration": iteration_text,
        "batch_id": render_result.batch_request.batch_id,
        "family": render_result.batch_request.family,
        "experiment": candidate_experiments[0] if candidate_experiments else "",
        "platform": render_result.batch_request.platform,
        "candidate_count": candidate_count,
        "candidate_records": candidate_records,
        "expected_hdf5_refs": list(expected_refs),
        "output_roots": [str(manifest.output_root) for manifest in render_result.batch_request.candidate_manifests],
        "scheduler_boundary": scheduler_boundary,
        "batch_request_path": str(render_result.batch_request.manifest_path),
        "batch_validation_report_path": str(render_result.batch_request.validation_report_path),
        "rendered_manifest_paths": [str(item) for item in render_result.rendered_manifest_paths],
        "validation_artifacts": {
            "dpd_plot_paths": render_plot_paths,
            "dpd_plot_sidecar_paths": render_plot_sidecar_paths,
            "upstream_validation_plot_paths": list(upstream_plots),
        },
        "expected_hdf5_ref_count": len(expected_refs),
        "pass_fail_criteria": criteria,
        "pass": passed,
        "failure_reasons": list(failure_reasons),
    }

    report = {
        "schema_version": ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION,
        "run_id": run_id_text,
        "iteration": iteration_text,
        "batch_id": render_result.batch_request.batch_id,
        "family": render_result.batch_request.family,
        "experiment": candidate_experiments[0] if candidate_experiments else "",
        "platform": render_result.batch_request.platform,
        "candidate_count": candidate_count,
        "candidate_hashes": [item["candidate_hash"] for item in candidate_records],
        "candidate_ids": [item["candidate_id"] for item in candidate_records],
        "criteria": criteria,
        "passed": passed,
        "failure_reasons": list(failure_reasons),
        "validation_artifacts": {
            "dpd_plot_paths": render_plot_paths,
            "dpd_plot_sidecar_paths": render_plot_sidecar_paths,
            "upstream_validation_plot_paths": list(upstream_plots),
        },
    }

    if not passed:
        manifest["status"] = "failed"
        report["status"] = "failed"
    else:
        manifest["status"] = "passed"
        report["status"] = "passed"

    artifact_dir = render_result.campaign_root
    manifest_path = artifact_dir / ACTIVE_LEARNING_DPD_DRY_RUN_GATE_MANIFEST_FILENAME
    report_path = artifact_dir / ACTIVE_LEARNING_DPD_DRY_RUN_GATE_REPORT_FILENAME
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)

    plot_path = render_result.plot_paths[0] if render_result.plot_paths else None
    plot_sidecar_path = render_result.plot_sidecar_paths[0] if render_result.plot_sidecar_paths else None

    return ActiveLearningDPDSamplingGateArtifacts(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        plot_path=plot_path,
        plot_sidecar_path=plot_sidecar_path,
        batch_request=render_result.batch_request,
        render_result=render_result,
        manifest=manifest,
        report=report,
    )


__all__ = [
    "ACTIVE_LEARNING_DPD_DRY_RUN_GATE_MANIFEST_FILENAME",
    "ACTIVE_LEARNING_DPD_DRY_RUN_GATE_REPORT_FILENAME",
    "ACTIVE_LEARNING_DPD_DRY_RUN_GATE_SCHEMA_VERSION",
    "ActiveLearningDPDSamplingGateArtifacts",
    "build_and_render_active_learning_dpd_sampling_gate",
]
