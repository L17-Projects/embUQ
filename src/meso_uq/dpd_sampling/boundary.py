"""DPD sampling boundary entrypoints for active-learning style adapters."""

from __future__ import annotations

import json
import os
import shlex
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, as_candidate
from meso_uq.core import Platform, coerce_platform
from meso_uq.platforms import validate_platform_path_policy
from meso_uq.scheduler_routing import parse_slurm_time_limit
from meso_uq.structures.gv.active_learning_handoff import (
    GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION,
    GVActiveLearningLaunchHandoff,
    build_gv_active_learning_launch_handoff,
    render_gv_active_learning_launch_handoff,
)
from meso_uq.structures.gv.launch import build_gv_launch_campaign_manifest

from .contracts import (
    DPD_SAMPLING_BATCH_SCHEMA_VERSION,
    DPD_SAMPLING_RENDER_SCHEMA_VERSION,
    DPD_SAMPLING_VALIDATION_SCHEMA_VERSION,
    DPDDataRef,
    DPDCandidateManifest,
    DPDSamplingBatchRequest,
    DPDSamplingRenderResult,
    DPDValidationReport,
)


_SUPPORTED_PLATFORMS = {Platform.KAROLINA, Platform.VEGA}
_GV_FAMILY = "gv"
_EMB_FAMILY = "emb"
_GV_PAYLOAD_KEYS = ("gv_launch",)
_EMB_PAYLOAD_KEYS = ("emb_launch",)
_ALLOWED_FAMILIES = {_GV_FAMILY, _EMB_FAMILY}
_FAMILY_MARKER_FIELDS = frozenset({"family", "dpd_family"})
_GV_FORBIDDEN_OUTPUT_DIR_NAMES = frozenset({"gv", "gv_simulation_files", "scripts", "src", "tests"})
_BATCH_DEFAULT_PREFIX = "dpd-sampling"
_GV_CANONICAL_RENDER_SCHEMA_VERSION = "meso_uq.dpd_sampling.gv_render.v1"
_EMB_CANONICAL_RENDER_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_render.v1"
_VALIDATION_PLOT_SCHEMA_VERSION = "meso_uq.dpd_sampling.validation_plot.v1"
_ADAPTER_OWNED_FIELDS = frozenset(
    {
        "platform",
        "output_root",
        "walltime",
        "gpu_count",
        "provenance_tags",
        "submission",
        "submission_commands",
    }
)


def _coerce_nonempty_str(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _coerce_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): val for key, val in value.items()}


def _coerce_mapping_optional(value: Mapping[str, object] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _coerce_mapping(value, field_name=field_name)


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _coerce_path_token(value: object, *, field_name: str) -> str:
    token = _coerce_nonempty_str(value, field_name=field_name).replace("/", "_")
    token = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in token).strip("._")
    if not token:
        raise ValueError(f"{field_name} must be path-safe non-empty text.")
    return token


def _normalize_run_id(value: object) -> str:
    return _coerce_path_token(value, field_name="run_id")


def _normalize_iteration(value: object) -> str:
    raw = _coerce_nonempty_str(value, field_name="iteration").strip()
    lowered = raw.lower()
    if lowered.startswith("iter_"):
        payload = raw[5:]
    elif lowered.startswith("iter-"):
        payload = raw[5:]
    else:
        payload = raw
    if payload.isdigit():
        return f"iter_{int(payload):04d}"
    normalized_payload = _coerce_path_token(payload, field_name="iteration")
    return f"iter_{normalized_payload}"


def _default_campaign_root(run_id: object, iteration: object) -> Path:
    normalized_run = _normalize_run_id(run_id)
    normalized_iteration = _normalize_iteration(iteration)
    return Path("_runs") / "active_learning" / normalized_run / "iterations" / normalized_iteration


def _coerce_platform(platform: Platform | str) -> Platform:
    normalized = coerce_platform(platform)
    if normalized not in _SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform {platform!r} is not supported for DPD sampling. "
            f"Allowed: {sorted(item.value for item in _SUPPORTED_PLATFORMS)}"
        )
    return normalized


def _coerce_gv_batch_id(run_id: str, iteration: str, batch_id: str | None) -> str:
    return _coerce_path_token(batch_id or f"{_BATCH_DEFAULT_PREFIX}:{run_id}:{iteration}", field_name="batch_id")


def _coerce_platforms(
    platforms: Platform | str | Iterable[Platform | str],
) -> tuple[Platform, ...]:
    if isinstance(platforms, (str, Platform)):
        return (_coerce_platform(platforms),)
    if isinstance(platforms, Iterable):
        normalized = tuple(_coerce_platform(item) for item in platforms)
        if not normalized:
            raise ValueError("platforms must include at least one value.")
        return normalized
    return (_coerce_platform(platforms),)


def _strip_family_marker_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _FAMILY_MARKER_FIELDS}


def _coerce_family(value: object, *, candidate_id: str) -> str:
    family = _coerce_nonempty_str(value, field_name=f"candidate {candidate_id} family").lower()
    if family not in _ALLOWED_FAMILIES:
        raise ValueError(f"Candidate {candidate_id!r} has unknown family {family!r}.")
    return family


def _reject_scheduler_owned_fields(payload: Mapping[str, Any], *, context: str) -> list[str]:
    rejected = sorted(_ADAPTER_OWNED_FIELDS.intersection(payload.keys()))
    if rejected:
        names = ", ".join(rejected)
        raise ValueError(
            f"{context} contains scheduler-owned fields: {names}. "
            "Move platform/gpu_count/walltime/provenance settings to boundary arguments."
        )
    return rejected


def _extract_candidate_payload(candidate: Candidate) -> tuple[str, dict[str, Any]]:
    payload = _coerce_mapping(candidate.parameters, field_name=f"candidate {candidate.candidate_id} parameters")

    explicit_family = payload.get("family")
    if explicit_family is None:
        explicit_family = payload.get("dpd_family")
    if explicit_family is not None:
        family = _coerce_family(explicit_family, candidate_id=candidate.candidate_id)
        if family == _GV_FAMILY:
            for key in _GV_PAYLOAD_KEYS:
                if key in payload:
                    return family, _coerce_mapping(payload[key], field_name=f"candidate {candidate.candidate_id} {key}")
            if {"experiment", "controls"}.issubset(payload) or "material_parameters" in payload:
                return family, _strip_family_marker_fields(payload)
            raise ValueError(
                f"Candidate {candidate.candidate_id!r} declares family='gv' but lacks "
                "gv payload fields."
            )
        if family == _EMB_FAMILY:
            for key in _EMB_PAYLOAD_KEYS:
                if key in payload:
                    return family, _coerce_mapping(payload[key], field_name=f"candidate {candidate.candidate_id} {key}")
            if "experiment" in payload:
                return family, payload
            raise ValueError(
                f"Candidate {candidate.candidate_id!r} declares family='emb' but lacks "
                "emb payload fields."
            )

    for key in _GV_PAYLOAD_KEYS:
        if key in payload:
            return _GV_FAMILY, _coerce_mapping(payload[key], field_name=f"candidate {candidate.candidate_id} {key}")

    for key in _EMB_PAYLOAD_KEYS:
        if key in payload:
            return _EMB_FAMILY, _coerce_mapping(payload[key], field_name=f"candidate {candidate.candidate_id} {key}")

    # GV legacy-compatible fallback.
    if {"experiment", "controls"}.issubset(payload) or "material_parameters" in payload:
        return _GV_FAMILY, payload

    raise ValueError(
        f"Candidate {candidate.candidate_id!r} does not declare a supported DPD family payload."
    )


def _extract_family_payloads(
    selected_candidates: Sequence[object],
) -> tuple[str, list[tuple[Candidate, str, dict[str, Any]]]]:
    candidates = tuple(as_candidate(item) for item in selected_candidates)
    if not candidates:
        raise ValueError("selected_candidates must contain at least one candidate.")

    seen = set[str]()
    first_family: str | None = None
    payloads: list[tuple[Candidate, str, dict[str, Any]]] = []
    mixed_candidate_ids: list[str] = []

    for candidate in candidates:
        normalized_id = _coerce_path_token(candidate.candidate_id, field_name=f"candidate_id {candidate.candidate_id!r}")
        if normalized_id in seen:
            raise ValueError(f"Duplicate selected candidate id {candidate.candidate_id!r}.")
        seen.add(normalized_id)

        candidate_family, payload = _extract_candidate_payload(candidate)
        if first_family is None:
            first_family = candidate_family
        elif candidate_family != first_family:
            mixed_candidate_ids.append(candidate.candidate_id)
            continue
        root_payload = _coerce_mapping(
            candidate.parameters,
            field_name=f"candidate {candidate.candidate_id} parameters",
        )
        _reject_scheduler_owned_fields(root_payload, context=f"Candidate {candidate.candidate_id!r} payload")
        _reject_scheduler_owned_fields(payload, context=f"Candidate {candidate.candidate_id!r} family payload")
        normalized_candidate = Candidate(candidate_id=normalized_id, parameters=payload)
        payloads.append((normalized_candidate, candidate_family, payload))

    if mixed_candidate_ids:
        raise ValueError(
            "Mixed candidate families are not allowed in one DPD sampling batch; "
            f"mixed candidates: {', '.join(sorted(mixed_candidate_ids))}."
        )

    if first_family is None:
        raise ValueError("Unable to infer DPD candidate family.")
    if not payloads:
        raise ValueError("All candidates were rejected due to mixed-family constraints.")

    return first_family, payloads


def _coerce_plot_paths(
    request: DPDSamplingBatchRequest,
    report: DPDValidationReport,
) -> tuple[Path, Path]:
    pyplot = _import_matplotlib_pyplot()
    plot_path = request.campaign_root / "dpd_sampling_validation_plot.png"
    sidecar_path = request.campaign_root / "dpd_sampling_validation_plot.png.json"

    families = sorted(report.family_distribution)
    family_counts = [report.family_distribution[name] for name in families]
    platforms = sorted(report.platform_distribution)
    platform_counts = [report.platform_distribution[name] for name in platforms]
    runtimes = sorted(report.runtime_distribution)
    runtime_counts = [report.runtime_distribution[name] for name in runtimes]

    fig, axes = pyplot.subplots(2, 2, figsize=(13, 9))
    try:
        axes[0, 0].bar(families, family_counts, color="#4c72b0")
        axes[0, 0].set_title("Candidate Family Distribution")
        axes[0, 0].set_xlabel("Family")
        axes[0, 0].set_ylabel("Count")

        axes[0, 1].bar(platforms, platform_counts, color="#c44e52")
        axes[0, 1].set_title("Platform Distribution")
        axes[0, 1].set_xlabel("Platform")
        axes[0, 1].set_ylabel("Count")

        axes[1, 0].bar(runtimes, runtime_counts, color="#8172b2")
        axes[1, 0].set_title("Runtime Distribution")
        axes[1, 0].set_xlabel("Runtime")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

        axes[1, 1].set_title("Submission and Rejections")
        axes[1, 1].axis("off")
        lines = [
            f"submission.submitted: {report.submission.get('submitted')}",
            f"submission_commands: {len(report.submission.get('submission_commands', []))}",
            f"scheduler_boundary: {sorted(item.value for item in _SUPPORTED_PLATFORMS)}",
            f"mixed_family_rejections: {', '.join(report.mixed_family_rejections) or 'none'}",
            f"expected_hdf5_refs: {len(report.expected_hdf5_refs)}",
        ]
        y = 0.95
        for line in lines:
            axes[1, 1].text(0.03, y, line, transform=axes[1, 1].transAxes, ha="left", va="top")
            y -= 0.14

        if report.expected_hdf5_refs:
            preview_refs = [
                f"{item.dataset_id} -> {item.hdf5_path}"
                for item in report.expected_hdf5_refs[:3]
            ]
            for line in preview_refs:
                axes[1, 1].text(0.03, y, line, transform=axes[1, 1].transAxes, ha="left", va="top", fontsize=6)
                y -= 0.1

        fig.suptitle(f"DPD Sampling Validation: {request.batch_id}")
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.12,
            top=0.90,
            wspace=0.35,
            hspace=0.45,
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
    finally:
        pyplot.close(fig)

    sidecar_payload = {
        "schema_version": _VALIDATION_PLOT_SCHEMA_VERSION,
        "plot": str(plot_path),
        "batch_id": request.batch_id,
        "submission": report.submission,
        "family_distribution": dict(report.family_distribution),
        "platform_distribution": dict(report.platform_distribution),
        "runtime_distribution": dict(report.runtime_distribution),
        "mixed_family_rejections": list(report.mixed_family_rejections),
        "scheduler_boundary": {
            "platform_whitelist": [item.value for item in sorted(_SUPPORTED_PLATFORMS, key=lambda p: p.value)],
            "submission_state": report.submission,
        },
        "expected_hdf5_refs": [item.as_manifest() for item in report.expected_hdf5_refs],
    }
    _write_json_payload(sidecar_payload, sidecar_path)
    return plot_path, sidecar_path


def _render_validation_plot(
    request: DPDSamplingBatchRequest,
    report: DPDValidationReport,
) -> tuple[Path, Path]:
    return _coerce_plot_paths(request, report)


def _write_json_payload(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_coerce_mapping(payload, field_name="payload"), sort_keys=True, indent=2), encoding="utf-8")
    return path


def _write_candidate_manifest(
    request: DPDSamplingBatchRequest,
    manifest: DPDCandidateManifest,
    rendered_payload: Mapping[str, Any],
) -> Path:
    candidate_root = manifest.output_root
    candidate_root.mkdir(parents=True, exist_ok=True)
    payload = manifest.as_manifest()
    payload["rendered_payload"] = _coerce_mapping(rendered_payload, field_name="rendered_payload")
    payload["submission"] = {
        "submitted": False,
        "submission_commands": [],
    }
    target = candidate_root / "dpd_sampling_candidate_manifest.json"
    return _write_json_payload(payload, target)


def _build_submission_state(platform: str) -> dict[str, Any]:
    return {
        "submitted": False,
        "submission_commands": [],
        "scheduler_boundary": {
            "platform": platform,
            "platform_whitelist": [item.value for item in sorted(_SUPPORTED_PLATFORMS, key=lambda item: item.value)],
        },
    }


def _resolve_gv_launch_artifact_path(path: Path, *, output_root_base: Path | None) -> Path:
    if path.is_absolute() or output_root_base is None:
        return path
    return output_root_base / path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[3]


def _validate_gv_final_output_root(*, output_root: Path, platform: Platform) -> None:
    resolved_output_root = output_root.resolve()
    repo_root = _repo_root()
    unsafe_roots = tuple(repo_root / name for name in _GV_FORBIDDEN_OUTPUT_DIR_NAMES)
    for unsafe_root in unsafe_roots:
        if resolved_output_root == unsafe_root or unsafe_root in resolved_output_root.parents:
            raise ValueError(
                "GV DPD sampling output_root must not be inside repository source roots: "
                f"{unsafe_root}."
            )
    if output_root.name in _GV_FORBIDDEN_OUTPUT_DIR_NAMES:
        raise ValueError(
            "GV DPD sampling output_root must not be a repository source directory."
        )

    policy_errors = [
        error
        for error in validate_platform_path_policy(
            platform,
            {"output_root": output_root.as_posix()},
            label="dpd_sampling_gv_final_output_root",
        )
        if "absolute paths are forbidden" not in error
    ]
    if policy_errors:
        raise ValueError(policy_errors[0])


def _rebase_path_string(value: str, *, old_root: Path, new_root: Path) -> str:
    old_text = old_root.as_posix()
    new_text = new_root.as_posix()
    if value == old_text:
        return new_text
    old_prefix = f"{old_text}/"
    if value.startswith(old_prefix):
        return f"{new_text}/{value.removeprefix(old_prefix)}"
    return value


def _rebase_manifest_path_strings(payload: Any, *, old_root: Path, new_root: Path) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): _rebase_manifest_path_strings(value, old_root=old_root, new_root=new_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rebase_manifest_path_strings(value, old_root=old_root, new_root=new_root)
            for value in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            _rebase_manifest_path_strings(value, old_root=old_root, new_root=new_root)
            for value in payload
        )
    if isinstance(payload, str):
        return _rebase_path_string(payload, old_root=old_root, new_root=new_root)
    return payload


def _rebase_gv_scheduler_script(
    script_path: Path,
    *,
    old_root: Path,
    new_root: Path,
) -> None:
    old_text = old_root.as_posix()
    new_text = new_root.as_posix()
    lines: list[str] = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("CAMPAIGN_DIR="):
            lines.append(f"CAMPAIGN_DIR={shlex.quote(new_text)}")
        elif line.startswith(("#SBATCH --output=", "#SBATCH --error=")):
            lines.append(line.replace(old_text, new_text, 1))
        elif line.startswith(("CAMPAIGN_HDF5_PATH=", "RUN_MANIFEST_PATHS=", "RUN_HDF5_PATHS=")):
            lines.append(line.replace(old_text, new_text))
        else:
            lines.append(line)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rebase_gv_rendered_absolute_root(
    rendered: Any,
    *,
    output_root_base: Path,
) -> dict[str, Any]:
    relative_root = Path(rendered.request.output_root)
    absolute_root = output_root_base / relative_root

    for script in rendered.scheduler_scripts:
        script_path = _resolve_gv_launch_artifact_path(
            Path(script.script_path),
            output_root_base=output_root_base,
        )
        _rebase_gv_scheduler_script(
            script_path,
            old_root=relative_root,
            new_root=absolute_root,
        )

    render_manifest_path = _resolve_gv_launch_artifact_path(
        Path(rendered.manifest_path),
        output_root_base=output_root_base,
    )
    manifest_payload = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    manifest_payload = _rebase_manifest_path_strings(
        manifest_payload,
        old_root=relative_root,
        new_root=absolute_root,
    )
    _write_json_payload(manifest_payload, render_manifest_path)

    return _rebase_manifest_path_strings(
        rendered.to_manifest(),
        old_root=relative_root,
        new_root=absolute_root,
    )


def _manifest_from_gv_launch(
    launch_handoff: GVActiveLearningLaunchHandoff,
    *,
    platform: str,
    campaign_root: Path,
    output_root_base: Path | None = None,
) -> tuple[DPDCandidateManifest, ...]:
    manifests: list[DPDCandidateManifest] = []
    for item in launch_handoff.candidates:
        request = item.launch_request
        campaign_manifest = build_gv_launch_campaign_manifest(request)
        output_root = Path(request.output_root)
        if output_root_base is not None and not output_root.is_absolute():
            output_root = output_root_base / output_root
        normalized_payload = request.to_manifest()
        if output_root_base is not None:
            normalized_payload = _rebase_manifest_path_strings(
                normalized_payload,
                old_root=Path(request.output_root),
                new_root=output_root,
            )
        refs = [
            DPDDataRef(
                family=_GV_FAMILY,
                dataset_id=campaign_manifest.dataset_id,
                hdf5_path=_resolve_gv_launch_artifact_path(
                    campaign_manifest.hdf5_path,
                    output_root_base=output_root_base,
                ),
                manifest_path=_resolve_gv_launch_artifact_path(
                    campaign_manifest.manifest_path,
                    output_root_base=output_root_base,
                ),
            ),
        ]
        refs.extend(
            DPDDataRef(
                family=_GV_FAMILY,
                dataset_id=run.dataset_id,
                hdf5_path=_resolve_gv_launch_artifact_path(
                    run.hdf5_path,
                    output_root_base=output_root_base,
                ),
                manifest_path=_resolve_gv_launch_artifact_path(
                    run.manifest_path,
                    output_root_base=output_root_base,
                ),
            )
            for run in campaign_manifest.runs
        )
        manifests.append(
            DPDCandidateManifest(
                candidate_id=item.candidate_id,
                family=_GV_FAMILY,
                platform=platform,
                output_root=output_root,
                campaign_root=campaign_root,
                normalized_payload=normalized_payload,
                expected_hdf5_datasets=tuple(refs),
            )
        )
    return tuple(manifests)


def _build_gv_candidate_manifests(
    selected_candidates: Sequence[tuple[Candidate, str, dict[str, Any]]],
    *,
    run_id: str,
    iteration: str,
    platform: Platform,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object],
    defaults: Mapping[str, object] | None,
    campaign_root: Path,
    batch_id: str | None,
) -> tuple[GVActiveLearningLaunchHandoff, tuple[DPDCandidateManifest, ...]]:
    normalized_defaults = _coerce_mapping_optional(defaults, field_name="defaults")
    normalized_provenance = _coerce_mapping_optional(provenance_tags, field_name="provenance_tags")
    normalized_batch_id = _coerce_gv_batch_id(run_id, iteration, batch_id)
    if campaign_root.is_absolute():
        for candidate, _family, _payload in selected_candidates:
            _validate_gv_final_output_root(
                output_root=campaign_root / candidate.candidate_id,
                platform=platform,
            )

    gv_candidates: list[Candidate] = [
        Candidate(candidate_id=candidate.candidate_id, parameters={_GV_PAYLOAD_KEYS[0]: payload})
        for candidate, _, payload in selected_candidates
    ]
    handoff_campaign_root = Path(".") if campaign_root.is_absolute() else campaign_root

    handoff = build_gv_active_learning_launch_handoff(
        gv_candidates,
        campaign_root=handoff_campaign_root,
        platform=platform,
        walltime=walltime,
        gpu_count=gpu_count,
        provenance_tags=normalized_provenance,
        defaults=normalized_defaults,
        batch_id=normalized_batch_id,
    )
    candidate_manifests = _manifest_from_gv_launch(
        handoff,
        platform=platform.value,
        campaign_root=campaign_root,
        output_root_base=None if not campaign_root.is_absolute() else campaign_root,
    )
    return handoff, candidate_manifests


def _build_emb_candidate_manifests(
    selected_candidates: Sequence[tuple[Candidate, str, dict[str, Any]]],
    *,
    run_id: str,
    iteration: str,
    platform: str,
    campaign_root: Path,
) -> tuple[DPDCandidateManifest, ...]:
    manifests: list[DPDCandidateManifest] = []
    for candidate, _, payload in selected_candidates:
        normalized_payload = dict(payload)
        normalized_payload["placeholder"] = True
        token = _coerce_path_token(candidate.candidate_id, field_name="candidate_id")
        normalized_payload.setdefault("experiment", payload.get("experiment", token))
        output_root = campaign_root / "emb" / token
        manifest_dataset_id = f"emb_{_coerce_path_token(normalized_payload['experiment'], field_name='experiment')}_{token}"
        hdf5_path = output_root / f"{manifest_dataset_id}.h5"
        manifests.append(
            DPDCandidateManifest(
                candidate_id=token,
                family=_EMB_FAMILY,
                platform=platform,
                output_root=output_root,
                campaign_root=campaign_root,
                normalized_payload=normalized_payload,
                expected_hdf5_datasets=(
                    DPDDataRef(
                        family=_EMB_FAMILY,
                        dataset_id=manifest_dataset_id,
                        hdf5_path=hdf5_path,
                        manifest_path=output_root / "emb_placeholder_manifest.json",
                    ),
                ),
            )
        )
    return tuple(manifests)


def build_dpd_sampling_batch_request(
    selected_candidates: Sequence[object],
    *,
    run_id: object,
    iteration: object,
    platform: Platform | str,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object] | None = None,
    defaults: Mapping[str, object] | None = None,
    campaign_root: str | Path | None = None,
    batch_id: str | None = None,
) -> DPDSamplingBatchRequest:
    if not isinstance(selected_candidates, Sequence) or isinstance(selected_candidates, (bytes, bytearray, str)):
        raise ValueError("selected_candidates must be a non-empty sequence.")
    if not selected_candidates:
        raise ValueError("selected_candidates must contain at least one candidate.")

    normalized_platform = _coerce_platform(platform)
    normalized_run = _normalize_run_id(run_id)
    normalized_iteration = _normalize_iteration(iteration)
    normalized_walltime = _coerce_nonempty_str(walltime, field_name="walltime")
    normalized_walltime_seconds = parse_slurm_time_limit(normalized_walltime)
    if not isinstance(gpu_count, int) or gpu_count <= 0:
        raise ValueError("gpu_count must be a positive integer.")

    root = Path(campaign_root) if campaign_root is not None else _default_campaign_root(normalized_run, normalized_iteration)

    family, payloads = _extract_family_payloads(selected_candidates)
    normalized_provenance = _coerce_mapping_optional(provenance_tags, field_name="provenance_tags")
    normalized_defaults = _coerce_mapping_optional(defaults, field_name="defaults")

    handoff: GVActiveLearningLaunchHandoff | None = None
    candidate_manifests: tuple[DPDCandidateManifest, ...]

    if family == _GV_FAMILY:
        handoff, candidate_manifests = _build_gv_candidate_manifests(
            payloads,
            run_id=normalized_run,
            iteration=normalized_iteration,
            platform=normalized_platform,
            walltime=normalized_walltime,
            gpu_count=gpu_count,
            provenance_tags=normalized_provenance,
            defaults=normalized_defaults,
            campaign_root=root,
            batch_id=batch_id,
        )
    else:
        candidate_manifests = _build_emb_candidate_manifests(
            payloads,
            run_id=normalized_run,
            iteration=normalized_iteration,
            platform=normalized_platform.value,
            campaign_root=root,
        )

    resolved_batch_id = _coerce_gv_batch_id(normalized_run, normalized_iteration, batch_id)

    return DPDSamplingBatchRequest(
        batch_id=resolved_batch_id,
        run_id=normalized_run,
        iteration=normalized_iteration,
        family=family,
        platform=normalized_platform.value,
        walltime=normalized_walltime,
        walltime_seconds=normalized_walltime_seconds,
        gpu_count=gpu_count,
        provenance_tags=normalized_provenance,
        campaign_root=root,
        candidate_manifests=candidate_manifests,
        defaults=normalized_defaults,
        _gv_handoff=handoff,
    )


def build_dpd_sampling_batch(
    selected_candidates: Sequence[object],
    *,
    run_id: object,
    iteration: object,
    platform: Platform | str,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object] | None = None,
    defaults: Mapping[str, object] | None = None,
    campaign_root: str | Path | None = None,
    batch_id: str | None = None,
) -> DPDSamplingBatchRequest:
    return build_dpd_sampling_batch_request(
        selected_candidates,
        run_id=run_id,
        iteration=iteration,
        platform=platform,
        walltime=walltime,
        gpu_count=gpu_count,
        provenance_tags=provenance_tags,
        defaults=defaults,
        campaign_root=campaign_root,
        batch_id=batch_id,
    )


def _render_emb_manifests(request: DPDSamplingBatchRequest) -> tuple[Path, ...]:
    paths: list[Path] = []
    for manifest in request.candidate_manifests:
        rendered_payload = {
            "schema_version": _EMB_CANONICAL_RENDER_SCHEMA_VERSION,
            "family": request.family,
            "platform": request.platform,
            "run_id": request.run_id,
            "iteration": request.iteration,
            "candidate_id": manifest.candidate_id,
            "submission": _build_submission_state(request.platform),
            "expected_hdf5_datasets": [item.as_manifest() for item in manifest.expected_hdf5_datasets],
            "notes": "Emb placeholder rendering: no production Mirheo parity claim.",
        }
        paths.append(_write_candidate_manifest(request, manifest, rendered_payload))
    return tuple(paths)


def _render_gv_manifests(
    request: DPDSamplingBatchRequest,
    handoff: GVActiveLearningLaunchHandoff,
    platforms: Platform | str | Iterable[Platform | str] | None,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    if handoff.campaign_root == Path("."):
        with _temporary_cwd(request.campaign_root):
            rendered_campaigns = render_gv_active_learning_launch_handoff(
                handoff,
                platforms=platforms or request.platform,
                overwrite=overwrite,
            )
    else:
        rendered_campaigns = render_gv_active_learning_launch_handoff(
            handoff,
            platforms=platforms or request.platform,
            overwrite=overwrite,
        )
    handoff_candidate_map = {item.candidate_id: item for item in handoff.candidates}
    request_candidate_map = {item.candidate_id: item for item in request.candidate_manifests}
    paths: list[Path] = []
    for rendered in rendered_campaigns:
        if request.campaign_root.is_absolute() and handoff.campaign_root == Path("."):
            rendered_payload = _rebase_gv_rendered_absolute_root(
                rendered,
                output_root_base=request.campaign_root,
            )
        else:
            rendered_payload = rendered.to_manifest()
        rendered_payload["submission"] = _build_submission_state(request.platform)
        rendered_payload["submission"]["submission_commands"] = []
        rendered_payload["notes"] = "GV rendering delegated to existing active-learning handoff."
        manifest = handoff_candidate_map.get(rendered.request.campaign_id)
        if manifest is None:
            raise ValueError(f"GV rendering produced unknown campaign id '{rendered.request.campaign_id}'.")
        candidate_manifest = request_candidate_map.get(manifest.candidate_id)
        if candidate_manifest is None:
            raise ValueError(f"Missing DPD candidate manifest for {manifest.candidate_id!r}.")
        rendered_payload["candidate_id"] = manifest.candidate_id
        rendered_payload["normalized_payload"] = candidate_manifest.normalized_payload
        target = _write_candidate_manifest(request, candidate_manifest, rendered_payload)
        paths.append(target)
    return tuple(paths)


def _build_validation_report(
    request: DPDSamplingBatchRequest,
    *,
    mixed_family_rejections: Sequence[str] | None = None,
) -> DPDValidationReport:
    family_counter = Counter(manifest.family for manifest in request.candidate_manifests)
    platform_counter = Counter(manifest.platform for manifest in request.candidate_manifests)
    runtime_key = f"{request.walltime}:{request.platform}:{request.gpu_count}"
    runtime_counter = Counter(runtime_key for _manifest in request.candidate_manifests)

    expected_refs = tuple(
        ref for manifest in request.candidate_manifests for ref in manifest.expected_hdf5_datasets
    )
    return DPDValidationReport(
        batch_id=request.batch_id,
        run_id=request.run_id,
        iteration=request.iteration,
        family=request.family,
        candidates=len(request.candidate_manifests),
        family_distribution=dict(family_counter),
        platform_distribution=dict(platform_counter),
        runtime_distribution=dict(runtime_counter),
        mixed_family_rejections=tuple(mixed_family_rejections or ()),
        scheduler_owned_field_rejections=tuple(),
        rejected_candidates=tuple(),
        expected_hdf5_refs=expected_refs,
        submission=_build_submission_state(request.platform),
    )


def _import_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def render_dpd_sampling_batch(
    request: DPDSamplingBatchRequest,
    *,
    platforms: Platform | str | Iterable[Platform | str] | None = None,
    overwrite: bool = False,
) -> DPDSamplingRenderResult:
    if not isinstance(request, DPDSamplingBatchRequest):
        raise ValueError("request must be a DPDSamplingBatchRequest.")

    normalized_platforms = (
        _coerce_platforms(platforms) if platforms is not None else None
    )

    root = request.campaign_root
    root.mkdir(parents=True, exist_ok=True)
    rendered_manifest_paths: list[Path] = []
    if request.family == _GV_FAMILY:
        if request._gv_handoff is None:
            raise ValueError("GV DPD batch request is missing launch handoff adapter payload.")
        rendered_manifest_paths.extend(
            _render_gv_manifests(
                request,
                request._gv_handoff,
                platforms=normalized_platforms or request.platform,
                overwrite=overwrite,
            )
        )
    elif request.family == _EMB_FAMILY:
        rendered_manifest_paths.extend(_render_emb_manifests(request))
    else:
        raise ValueError("Unsupported DPD family.")

    batch_manifest = request.as_manifest()
    manifest_path = _write_json_payload(batch_manifest, request.manifest_path)
    rendered_manifest_paths.append(manifest_path)

    report = _build_validation_report(request, mixed_family_rejections=())
    validation_path = _write_json_payload(report.as_manifest(), request.validation_report_path)
    rendered_manifest_paths.append(validation_path)

    plot_path, plot_sidecar_path = _render_validation_plot(request, report)
    result = DPDSamplingRenderResult(
        batch_request=request,
        campaign_root=root,
        rendered_manifest_paths=tuple(rendered_manifest_paths),
        plot_paths=(plot_path,),
        validation_report=report,
        plot_sidecar_paths=(plot_sidecar_path,),
    )
    return result


def build_and_render_dpd_sampling_batch(
    selected_candidates: Sequence[object],
    *,
    run_id: object,
    iteration: object,
    platform: Platform | str,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object] | None = None,
    defaults: Mapping[str, object] | None = None,
    campaign_root: str | Path | None = None,
    batch_id: str | None = None,
    platforms: Platform | str | Iterable[Platform | str] | None = None,
    overwrite: bool = False,
) -> DPDSamplingRenderResult:
    request = build_dpd_sampling_batch_request(
        selected_candidates,
        run_id=run_id,
        iteration=iteration,
        platform=platform,
        walltime=walltime,
        gpu_count=gpu_count,
        provenance_tags=provenance_tags,
        defaults=defaults,
        campaign_root=campaign_root,
        batch_id=batch_id,
    )
    return render_dpd_sampling_batch(
        request,
        platforms=request.platform if platforms is None else platforms,
        overwrite=overwrite,
    )


def validate_dpd_sampling_batch(selected_candidates: Sequence[object]) -> bool:
    try:
        _ = build_dpd_sampling_batch_request(
            selected_candidates,
            run_id="validation",
            iteration="0000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
        )
    except Exception as exc:
        raise ValueError(f"DPD batch validation failed: {exc}") from exc
    return True


__all__ = [
    "build_dpd_sampling_batch",
    "build_dpd_sampling_batch_request",
    "build_and_render_dpd_sampling_batch",
    "render_dpd_sampling_batch",
    "validate_dpd_sampling_batch",
    "GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION",
    "DPD_SAMPLING_BATCH_SCHEMA_VERSION",
    "DPD_SAMPLING_RENDER_SCHEMA_VERSION",
    "DPD_SAMPLING_VALIDATION_SCHEMA_VERSION",
]
