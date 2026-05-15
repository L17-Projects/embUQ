from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, as_candidate
from meso_uq.core import Platform

from .launch import (
    GVLaunchRenderedCampaign,
    GVLaunchRequest,
    build_gv_launch_campaign_manifest,
    render_gv_launch_campaigns,
    validate_gv_launch_request,
)


GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION = "meso_uq.gv.active_learning_handoff.v1"
GV_ACTIVE_LEARNING_LAUNCH_PAYLOAD_KEY = "gv_launch"

_CANDIDATE_OWNED_LAUNCH_KEYS = frozenset(
    {
        "experiment",
        "material_parameters",
        "geometry",
        "controls",
        "radGV",
        "height",
    }
)
_ADAPTER_OWNED_LAUNCH_KEYS = frozenset(
    {
        "platform",
        "output_root",
        "walltime",
        "gpu_count",
        "provenance_tags",
    }
)


@dataclass(frozen=True)
class GVActiveLearningLaunchCandidate:
    """One selected active-learning candidate mapped to one GV launch request."""

    candidate: Candidate
    launch_request: GVLaunchRequest

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def to_manifest(self) -> dict[str, Any]:
        campaign_manifest = build_gv_launch_campaign_manifest(self.launch_request)
        return {
            "candidate_id": self.candidate_id,
            "candidate": self.candidate.as_dict(),
            "launch_request": self.launch_request.to_manifest(),
            "expected_hdf5_datasets": {
                "campaign": {
                    "dataset_id": campaign_manifest.dataset_id,
                    "hdf5_path": str(campaign_manifest.hdf5_path),
                    "manifest_path": str(campaign_manifest.manifest_path),
                },
                "runs": [
                    {
                        "dataset_id": run.dataset_id,
                        "hdf5_path": str(run.hdf5_path),
                        "manifest_path": str(run.manifest_path),
                    }
                    for run in campaign_manifest.runs
                ],
            },
        }


@dataclass(frozen=True)
class GVActiveLearningLaunchHandoff:
    """Render-free GV boundary object for selected active-learning candidates."""

    campaign_root: Path
    candidates: tuple[GVActiveLearningLaunchCandidate, ...]

    @property
    def launch_requests(self) -> tuple[GVLaunchRequest, ...]:
        return tuple(item.launch_request for item in self.candidates)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION,
            "source": "active_learning_selected_candidates",
            "campaign_root": str(self.campaign_root),
            "candidate_count": len(self.candidates),
            "launch_request_count": len(self.launch_requests),
            "candidates": [item.to_manifest() for item in self.candidates],
            "submission": {
                "submitted": False,
                "submission_commands": [],
            },
        }


def build_gv_active_learning_launch_handoff(
    selected_candidates: Sequence[object],
    *,
    campaign_root: str | Path,
    platform: Platform | str,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object],
    defaults: Mapping[str, object] | None = None,
    batch_id: str | None = None,
    payload_key: str = GV_ACTIVE_LEARNING_LAUNCH_PAYLOAD_KEY,
) -> GVActiveLearningLaunchHandoff:
    """Convert selected active-learning candidates into validated GV launch requests.

    The selected candidates own only scientific GV launch fields:
    experiment, material parameters, geometry, and controls. Scheduler-facing fields
    are provided once to this adapter and are rejected inside candidate payloads.
    """

    candidates = _normalize_candidates(selected_candidates)
    campaign_root_path = _normalize_campaign_root(campaign_root)
    default_launch_fields = _normalize_launch_payload(defaults or {}, context="defaults")
    normalized_batch_id = str(batch_id).strip() if batch_id is not None else campaign_root_path.name
    if not normalized_batch_id:
        raise ValueError("GV active-learning batch_id must be a non-empty string.")

    used_campaign_dirs: set[Path] = set()
    launch_candidates: list[GVActiveLearningLaunchCandidate] = []
    for candidate in candidates:
        candidate_payload = _candidate_launch_payload(candidate, payload_key=payload_key)
        launch_fields = _merge_launch_fields(default_launch_fields, candidate_payload)
        _require_launch_field(launch_fields, "experiment", candidate.candidate_id)
        _require_launch_field(launch_fields, "material_parameters", candidate.candidate_id)
        _require_geometry_fields(launch_fields, candidate.candidate_id)
        _require_launch_field(launch_fields, "controls", candidate.candidate_id)

        candidate_campaign_dir = campaign_root_path / _safe_path_token(candidate.candidate_id)
        if candidate_campaign_dir in used_campaign_dirs:
            raise ValueError(
                "GV active-learning selected candidates produce duplicate campaign directories: "
                f"{candidate_campaign_dir}."
            )
        used_campaign_dirs.add(candidate_campaign_dir)

        launch_request = validate_gv_launch_request(
            experiment=launch_fields["experiment"],
            material_parameters=_expect_mapping(
                launch_fields["material_parameters"],
                context=f"candidate {candidate.candidate_id!r} material_parameters",
            ),
            geometry=launch_fields.get("geometry"),
            controls=_expect_mapping(
                launch_fields["controls"],
                context=f"candidate {candidate.candidate_id!r} controls",
            ),
            radGV=launch_fields.get("radGV"),  # type: ignore[arg-type]
            height=launch_fields.get("height"),  # type: ignore[arg-type]
            platform=platform,
            output_root=candidate_campaign_dir,
            walltime=walltime,
            gpu_count=gpu_count,
            provenance_tags=_candidate_provenance_tags(
                provenance_tags,
                batch_id=normalized_batch_id,
                candidate_id=candidate.candidate_id,
            ),
        )
        launch_candidates.append(
            GVActiveLearningLaunchCandidate(
                candidate=candidate,
                launch_request=launch_request,
            )
        )

    return GVActiveLearningLaunchHandoff(
        campaign_root=campaign_root_path,
        candidates=tuple(launch_candidates),
    )


def render_gv_active_learning_launch_handoff(
    handoff: GVActiveLearningLaunchHandoff,
    *,
    platforms: Platform | str | Iterable[Platform | str] | None = None,
    overwrite: bool = False,
) -> tuple[GVLaunchRenderedCampaign, ...]:
    """Render scheduler-ready GV launch artifacts for a handoff without submitting jobs."""

    if not isinstance(handoff, GVActiveLearningLaunchHandoff):
        raise ValueError("handoff must be a GVActiveLearningLaunchHandoff.")
    return render_gv_launch_campaigns(
        handoff.launch_requests,
        platforms=platforms,
        overwrite=overwrite,
    )


def _normalize_candidates(selected_candidates: Sequence[object]) -> tuple[Candidate, ...]:
    if not isinstance(selected_candidates, Sequence) or isinstance(
        selected_candidates,
        (str, bytes, bytearray),
    ):
        raise ValueError("selected_candidates must be a non-empty sequence.")
    if not selected_candidates:
        raise ValueError("selected_candidates must contain at least one candidate.")

    candidates = tuple(as_candidate(item) for item in selected_candidates)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError(f"Duplicate selected candidate id: {candidate.candidate_id!r}.")
        seen.add(candidate.candidate_id)
    return candidates


def _normalize_campaign_root(campaign_root: str | Path) -> Path:
    text = str(campaign_root).strip()
    if not text:
        raise ValueError("GV active-learning campaign_root must be a non-empty path.")
    root = Path(text)
    if any(part == ".." for part in root.parts):
        raise ValueError("GV active-learning campaign_root must not contain path traversal segments.")
    return root


def _candidate_launch_payload(candidate: Candidate, *, payload_key: str) -> dict[str, object]:
    key = str(payload_key).strip()
    if not key:
        raise ValueError("GV active-learning payload_key must be a non-empty string.")
    raw_payload = candidate.parameters.get(key, candidate.parameters)
    if not isinstance(raw_payload, Mapping):
        raise ValueError(
            f"GV active-learning candidate {candidate.candidate_id!r} payload must be a mapping."
        )
    return _normalize_launch_payload(raw_payload, context=f"candidate {candidate.candidate_id!r}")


def _normalize_launch_payload(payload: Mapping[str, object], *, context: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"GV active-learning {context} must be a mapping.")
    normalized = {str(key): value for key, value in payload.items()}
    adapter_owned = sorted(set(normalized) & _ADAPTER_OWNED_LAUNCH_KEYS)
    if adapter_owned:
        names = ", ".join(adapter_owned)
        raise ValueError(
            f"GV active-learning {context} contains adapter-owned launch fields: {names}. "
            "Pass scheduler and provenance fields to build_gv_active_learning_launch_handoff instead."
        )
    unknown = sorted(set(normalized) - _CANDIDATE_OWNED_LAUNCH_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"GV active-learning {context} contains unsupported launch fields: {names}.")
    return normalized


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"GV active-learning {context} must be a mapping.")
    return value


def _merge_launch_fields(
    default_launch_fields: Mapping[str, object],
    candidate_payload: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(default_launch_fields)
    if "controls" in default_launch_fields and "controls" in candidate_payload:
        default_controls = _expect_mapping(
            default_launch_fields["controls"],
            context="defaults controls",
        )
        candidate_controls = _expect_mapping(
            candidate_payload["controls"],
            context="candidate controls",
        )
        merged["controls"] = {**default_controls, **candidate_controls}
    merged.update({key: value for key, value in candidate_payload.items() if key != "controls"})
    return merged


def _require_launch_field(payload: Mapping[str, object], field_name: str, candidate_id: str) -> None:
    if field_name not in payload:
        raise ValueError(
            f"GV active-learning candidate {candidate_id!r} is missing required launch field "
            f"{field_name!r}."
        )


def _require_geometry_fields(payload: Mapping[str, object], candidate_id: str) -> None:
    has_geometry = "geometry" in payload
    has_explicit_geometry_field = "radGV" in payload or "height" in payload
    if has_geometry and has_explicit_geometry_field:
        raise ValueError(
            f"GV active-learning candidate {candidate_id!r} mixes 'geometry' with explicit "
            "'radGV'/'height' fields; provide exactly one geometry form."
        )
    has_rad_height = "radGV" in payload and "height" in payload
    if not has_geometry and not has_rad_height:
        raise ValueError(
            f"GV active-learning candidate {candidate_id!r} is missing required launch geometry; "
            "provide 'geometry' or both 'radGV' and 'height'."
        )


def _candidate_provenance_tags(
    provenance_tags: Mapping[str, object],
    *,
    batch_id: str,
    candidate_id: str,
) -> dict[str, object]:
    if not isinstance(provenance_tags, Mapping):
        raise ValueError("GV active-learning provenance_tags must be a mapping.")
    tags = dict(provenance_tags)
    tags["source"] = "active_learning_selected_candidates"
    tags["active_learning_batch_id"] = batch_id
    tags["active_learning_candidate_id"] = candidate_id
    return tags


def _safe_path_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    token = token.strip("._")
    if not token:
        raise ValueError(f"GV active-learning candidate id {value!r} cannot form a path-safe token.")
    return token


__all__ = [
    "GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION",
    "GV_ACTIVE_LEARNING_LAUNCH_PAYLOAD_KEY",
    "GVActiveLearningLaunchCandidate",
    "GVActiveLearningLaunchHandoff",
    "build_gv_active_learning_launch_handoff",
    "render_gv_active_learning_launch_handoff",
]
