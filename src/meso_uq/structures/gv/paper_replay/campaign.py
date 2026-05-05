from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from meso_uq.structures import get_structure

MANIFEST_SCHEMA_VERSION = 1
CAMPAIGN_RUN_ROOT = Path("_runs") / "gv" / "figure_replay"
CAMPAIGN_MANIFEST_FILENAME = "campaign_manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = (completed.stdout or "").strip()
    return value or None


def _normalize_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in normalized or "\\" in normalized or ".." in normalized:
        raise ValueError(f"{field_name} must not contain path traversal components.")
    return normalized


def _ensure_finite(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _ensure_under_root(path: Path, root: Path, *, field_name: str) -> Path:
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"{field_name} must live under {base}.")
    return resolved


def _manifest_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_manifest") and callable(value.to_manifest):
        return _manifest_safe(value.to_manifest())
    if isinstance(value, dict):
        return {str(key): _manifest_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_safe(item) for item in value]
    return value


def _validate_source_pdf_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    validated: list[Path] = []
    for index, path in enumerate(paths):
        resolved = _resolve_path(path)
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(f"source_pdfs[{index}] must point to a PDF file.")
        validated.append(resolved)
    return tuple(validated)


def _validate_number_map(payload: dict[str, float], *, field_name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in payload.items():
        normalized[str(name)] = _ensure_finite(value, field_name=f"{field_name}[{name!r}]")
    return normalized


def _validate_control_map(payload: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in payload.items():
        if isinstance(value, (list, tuple)):
            values = [
                _ensure_finite(item, field_name=f"{field_name}[{name!r}]")
                for item in value
            ]
            if not values:
                raise ValueError(f"{field_name}[{name!r}] must contain at least one sweep value.")
            normalized[str(name)] = values
        else:
            normalized[str(name)] = _ensure_finite(value, field_name=f"{field_name}[{name!r}]")
    return normalized


def _load_control_value(value: Any) -> Any:
    if isinstance(value, list):
        return [float(item) for item in value]
    return float(value)


@dataclass(frozen=True)
class GVPaperReplayGitHead:
    commit: str | None
    branch: str | None
    dirty_worktree: bool | None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "dirty_worktree": self.dirty_worktree,
        }


@dataclass(frozen=True)
class GVPaperReplayFiniteCheck:
    name: str
    passed: bool
    finite_ratio: float
    nonfinite_count: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _normalize_identifier(self.name, field_name="finite_check.name")
        ratio = _ensure_finite(self.finite_ratio, field_name="finite_check.finite_ratio")
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("finite_check.finite_ratio must be between 0 and 1.")
        if int(self.nonfinite_count) < 0:
            raise ValueError("finite_check.nonfinite_count must be >= 0.")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "finite_ratio": float(self.finite_ratio),
            "nonfinite_count": int(self.nonfinite_count),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class GVPaperReplayDataRange:
    name: str
    minimum: float
    maximum: float
    units: str | None = None

    def __post_init__(self) -> None:
        _normalize_identifier(self.name, field_name="data_range.name")
        minimum = _ensure_finite(self.minimum, field_name=f"data_range[{self.name}].minimum")
        maximum = _ensure_finite(self.maximum, field_name=f"data_range[{self.name}].maximum")
        if minimum > maximum:
            raise ValueError(f"data_range[{self.name}] minimum must be <= maximum.")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "units": self.units,
        }


@dataclass(frozen=True)
class GVPaperReplayComparisonPacket:
    lane: str
    status: str
    reference_paths: tuple[Path, ...]
    replay_paths: tuple[Path, ...]
    metric_placeholders: tuple[str, ...] = ("l2_error", "max_abs_error", "status_note")
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _normalize_identifier(self.lane, field_name="comparison_packet.lane")
        _normalize_identifier(self.status, field_name="comparison_packet.status")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "reference_paths": [str(path) for path in self.reference_paths],
            "replay_paths": [str(path) for path in self.replay_paths],
            "metrics": {name: None for name in self.metric_placeholders},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class GVPaperReplayLaneRecord:
    lane: str
    experiment: str
    mode: str
    source_pdfs: tuple[Path, ...]
    runtime_commands: tuple[tuple[str, ...], ...]
    material_parameters: dict[str, float]
    geometry: dict[str, float]
    controls: dict[str, Any]
    output_paths: dict[str, Path]
    plot_paths: tuple[Path, ...]
    validation_status: str
    data_ranges: tuple[GVPaperReplayDataRange, ...]
    finite_checks: tuple[GVPaperReplayFiniteCheck, ...]
    slurm_job_ids: tuple[str, ...] = ()
    runtime_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lane_name = _normalize_identifier(self.lane, field_name="lane")
        if self.experiment != lane_name:
            raise ValueError("lane must match experiment for GV paper replay lanes.")
        structure = get_structure("gv")
        try:
            structure.get_experiment(self.experiment, include_experimental=True)
        except KeyError as exc:
            raise ValueError(f"Unknown GV experiment {self.experiment!r}.") from exc
        _normalize_identifier(self.mode, field_name="mode")
        _normalize_identifier(self.validation_status, field_name="validation_status")
        if not self.runtime_commands:
            raise ValueError("runtime_commands must contain at least one command.")
        for index, command in enumerate(self.runtime_commands):
            if not command:
                raise ValueError(f"runtime_commands[{index}] must not be empty.")
        object.__setattr__(self, "source_pdfs", _validate_source_pdf_paths(self.source_pdfs))
        object.__setattr__(
            self,
            "material_parameters",
            _validate_number_map(self.material_parameters, field_name="material_parameters"),
        )
        object.__setattr__(
            self,
            "geometry",
            _validate_number_map(self.geometry, field_name="geometry"),
        )
        object.__setattr__(
            self,
            "controls",
            _validate_control_map(self.controls, field_name="controls"),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "experiment": self.experiment,
            "mode": self.mode,
            "source_pdfs": [str(path) for path in self.source_pdfs],
            "runtime_commands": [list(command) for command in self.runtime_commands],
            "material_parameters": dict(self.material_parameters),
            "geometry": dict(self.geometry),
            "controls": dict(self.controls),
            "output_paths": {name: str(path) for name, path in self.output_paths.items()},
            "plot_paths": [str(path) for path in self.plot_paths],
            "validation_status": self.validation_status,
            "data_ranges": [item.to_manifest() for item in self.data_ranges],
            "finite_checks": [item.to_manifest() for item in self.finite_checks],
            "slurm_job_ids": list(self.slurm_job_ids),
            "runtime_ids": list(self.runtime_ids),
            "notes": list(self.notes),
        }


class GVPaperReplayLaneRunner(Protocol):
    """Minimal lane-module contract for operational replay integration."""

    def run_lane(
        self,
        *,
        campaign_id: str,
        campaign_root: Path,
        lane: str,
        source_pdfs: tuple[Path, ...],
        dry_run: bool,
        fixture_mode: bool,
    ) -> GVPaperReplayLaneRecord:
        ...


@dataclass(frozen=True)
class GVPaperReplayCampaignManifest:
    campaign_id: str
    campaign_root: Path
    generated_at_utc: str
    schema_version: int
    git_head: GVPaperReplayGitHead
    dry_run: bool
    fixture_mode: bool
    source_pdfs: tuple[Path, ...]
    lanes: tuple[GVPaperReplayLaneRecord, ...]
    comparison_packets: tuple[GVPaperReplayComparisonPacket, ...]
    manifest_path: Path | None = None
    slurm_job_ids: tuple[str, ...] = ()
    runtime_ids: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        validation = validate_campaign_manifest(self, write_safe=True)
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "campaign_root": str(self.campaign_root),
            "generated_at_utc": self.generated_at_utc,
            "git_head": self.git_head.to_manifest(),
            "dry_run": self.dry_run,
            "fixture_mode": self.fixture_mode,
            "source_pdfs": [str(path) for path in self.source_pdfs],
            "slurm_job_ids": list(self.slurm_job_ids),
            "runtime_ids": list(self.runtime_ids),
            "lanes": [lane.to_manifest() for lane in self.lanes],
            "comparison_packets": [packet.to_manifest() for packet in self.comparison_packets],
            "validation": validation,
        }


def collect_git_head(repo_root: Path) -> GVPaperReplayGitHead:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _run_git(repo_root, "status", "--porcelain")
    dirty = bool(dirty_output) if dirty_output is not None else None
    return GVPaperReplayGitHead(commit=commit, branch=branch, dirty_worktree=dirty)


def resolve_campaign_root(
    *,
    repo_root: Path,
    campaign_id: str,
    output_root: str | Path | None = None,
) -> Path:
    normalized_id = _normalize_identifier(campaign_id, field_name="campaign_id")
    expected_root = repo_root.resolve() / CAMPAIGN_RUN_ROOT / normalized_id
    if output_root is None:
        return expected_root
    resolved = _resolve_path(output_root)
    allowed_parent = (repo_root.resolve() / CAMPAIGN_RUN_ROOT).resolve()
    _ensure_under_root(resolved, allowed_parent, field_name="output_root")
    if resolved.name != normalized_id:
        raise ValueError("output_root leaf directory must match campaign_id.")
    return resolved


def build_comparison_packet_skeletons(
    lanes: tuple[GVPaperReplayLaneRecord, ...],
) -> tuple[GVPaperReplayComparisonPacket, ...]:
    packets: list[GVPaperReplayComparisonPacket] = []
    for lane in lanes:
        replay_paths = tuple(path for path in lane.plot_paths)
        packets.append(
            GVPaperReplayComparisonPacket(
                lane=lane.lane,
                status="ready_for_qualitative_review",
                reference_paths=lane.source_pdfs,
                replay_paths=replay_paths,
                notes=(
                    "DPD replay plot is ready for qualitative comparison against the mapped paper/SI figure.",
                ),
            )
        )
    return tuple(packets)


def validate_campaign_manifest(
    manifest: GVPaperReplayCampaignManifest,
    *,
    write_safe: bool = False,
) -> dict[str, Any]:
    repo_root = manifest.campaign_root.resolve().parents[3]
    expected_parent = (repo_root / CAMPAIGN_RUN_ROOT).resolve()
    campaign_root = _ensure_under_root(manifest.campaign_root, expected_parent, field_name="campaign_root")
    if campaign_root.name != manifest.campaign_id:
        raise ValueError("campaign_root leaf directory must match campaign_id.")
    source_pdfs = _validate_source_pdf_paths(manifest.source_pdfs)
    lane_ids: set[str] = set()
    finite_failures = 0
    for lane in manifest.lanes:
        if lane.lane in lane_ids:
            raise ValueError(f"Duplicate lane record {lane.lane!r} in campaign manifest.")
        lane_ids.add(lane.lane)
        for path in lane.output_paths.values():
            _ensure_under_root(path, repo_root / "_runs", field_name=f"lane[{lane.lane}].output_path")
        for path in lane.plot_paths:
            _ensure_under_root(path, repo_root / "_runs", field_name=f"lane[{lane.lane}].plot_path")
        for check in lane.finite_checks:
            if not check.passed:
                finite_failures += 1
    missing_source_pdfs = [str(path) for path in source_pdfs if not path.exists()]
    validation_status = "passed" if finite_failures == 0 and not missing_source_pdfs else "failed"
    return {
        "status": validation_status,
        "campaign_root_ok": True,
        "lane_count": len(manifest.lanes),
        "comparison_packet_count": len(manifest.comparison_packets),
        "finite_failure_count": finite_failures,
        "missing_source_pdfs": missing_source_pdfs,
        "write_safe": write_safe,
    }


def write_campaign_manifest(
    *,
    manifest: GVPaperReplayCampaignManifest,
    manifest_path: Path | None = None,
) -> Path:
    target = manifest_path or manifest.manifest_path or (manifest.campaign_root / CAMPAIGN_MANIFEST_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_manifest()
    target.write_text(json.dumps(_manifest_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_lane_record_fixture(path: str | Path) -> GVPaperReplayLaneRecord:
    resolved = _resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lane fixture {resolved} must contain a JSON object.")

    data_ranges = tuple(
        GVPaperReplayDataRange(
            name=str(item["name"]),
            minimum=float(item["minimum"]),
            maximum=float(item["maximum"]),
            units=item.get("units"),
        )
        for item in payload.get("data_ranges", [])
    )
    finite_checks = tuple(
        GVPaperReplayFiniteCheck(
            name=str(item["name"]),
            passed=bool(item["passed"]),
            finite_ratio=float(item["finite_ratio"]),
            nonfinite_count=int(item["nonfinite_count"]),
            notes=tuple(str(note) for note in item.get("notes", [])),
        )
        for item in payload.get("finite_checks", [])
    )
    return GVPaperReplayLaneRecord(
        lane=str(payload["lane"]),
        experiment=str(payload.get("experiment", payload["lane"])),
        mode=str(payload.get("mode", "fixture")),
        source_pdfs=tuple(_resolve_path(item) for item in payload.get("source_pdfs", [])),
        runtime_commands=tuple(tuple(str(part) for part in command) for command in payload["runtime_commands"]),
        material_parameters={str(name): float(value) for name, value in payload.get("material_parameters", {}).items()},
        geometry={str(name): float(value) for name, value in payload.get("geometry", {}).items()},
        controls={str(name): _load_control_value(value) for name, value in payload.get("controls", {}).items()},
        output_paths={str(name): _resolve_path(value) for name, value in payload.get("output_paths", {}).items()},
        plot_paths=tuple(_resolve_path(item) for item in payload.get("plot_paths", [])),
        validation_status=str(payload.get("validation_status", "passed")),
        data_ranges=data_ranges,
        finite_checks=finite_checks,
        slurm_job_ids=tuple(str(item) for item in payload.get("slurm_job_ids", [])),
        runtime_ids=tuple(str(item) for item in payload.get("runtime_ids", [])),
        notes=tuple(str(note) for note in payload.get("notes", [])),
    )


def build_fixture_lane_record(
    *,
    repo_root: Path,
    campaign_root: Path,
    lane: str,
    source_pdfs: tuple[Path, ...],
    fixture_mode: bool,
) -> GVPaperReplayLaneRecord:
    lane_name = _normalize_identifier(lane, field_name="lane")
    experiment_spec = get_structure("gv").get_experiment(lane_name, include_experimental=True)
    lane_root = campaign_root / "lanes" / lane_name
    plots_root = lane_root / "plots"
    outputs_root = lane_root / "outputs"
    plots_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    plot_path = plots_root / f"{lane_name}_preview.txt"
    summary_path = outputs_root / f"{lane_name}_summary.json"
    plot_path.write_text("fixture plot placeholder\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "lane": lane_name,
                "mode": "fixture" if fixture_mode else "dry-run",
                "created_at_utc": _utc_now_iso(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runtime_command = (
        "python",
        str((repo_root / "scripts" / "workflows" / "gv" / "run_gv_runtime.py").resolve()),
        "--experiment",
        lane_name,
        "--output-root",
        str(lane_root),
    )
    controls = {
        control.name: (1.0 if index == 0 else 0.0)
        for index, control in enumerate(experiment_spec.controls)
    }
    return GVPaperReplayLaneRecord(
        lane=lane_name,
        experiment=lane_name,
        mode="fixture" if fixture_mode else "dry-run",
        source_pdfs=source_pdfs,
        runtime_commands=(runtime_command,),
        material_parameters={
            "ka": 1.0,
            "kb": 0.2,
            "mu": 0.5,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 0.4,
            "c": 0.0,
        },
        geometry={"radius": 2.0, "height": 14.28},
        controls=controls,
        output_paths={"summary": summary_path, "lane_root": lane_root},
        plot_paths=(plot_path,),
        validation_status="passed",
        data_ranges=(
            GVPaperReplayDataRange(name="response_x", minimum=0.0, maximum=1.0, units="arb"),
            GVPaperReplayDataRange(name="response_y", minimum=0.0, maximum=1.0, units="arb"),
        ),
        finite_checks=(
            GVPaperReplayFiniteCheck(
                name="response_channels",
                passed=True,
                finite_ratio=1.0,
                nonfinite_count=0,
                notes=("Fixture replay emits only finite placeholder values.",),
            ),
        ),
        runtime_ids=(f"fixture:{lane_name}",),
        notes=("Synthetic fixture lane for CI-safe paper replay manifest coverage.",),
    )
