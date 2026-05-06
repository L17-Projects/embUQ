#!/usr/bin/env python3
"""Assemble a reusable GV paper figure replay campaign provenance packet."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.structures import get_structure  # noqa: E402
from meso_uq.structures.gv.paper_replay import (  # noqa: E402
    CAMPAIGN_MANIFEST_FILENAME,
    CAMPAIGN_RUN_ROOT,
    MANIFEST_SCHEMA_VERSION,
    GVPaperReplayCampaignManifest,
    GVPaperReplayDataRange,
    GVPaperReplayFiniteCheck,
    GVPaperReplayLaneRecord,
    build_comparison_packet_skeletons,
    build_fixture_lane_record,
    collect_git_head,
    load_lane_record_fixture,
    resolve_campaign_root,
    write_campaign_manifest,
)
from meso_uq.structures.gv.paper_replay_constants import (  # noqa: E402
    load_gv_paper_replay_profile,
    validate_gv_paper_replay_profile,
)
from meso_uq.structures.gv.paper_replay_lanes import (  # noqa: E402
    plan_buckling_paper_replay_lane,
    plan_eigenmodes_paper_replay_lane,
    plan_stretching_paper_replay_lane,
    plan_torsion_paper_replay_lane,
    plot_buckling_paper_replay,
    plot_eigenmodes_paper_replay,
    run_buckling_paper_replay_lane,
    run_eigenmodes_paper_replay_lane,
    run_stretching_paper_replay_lane,
    run_torsion_paper_replay_lane,
)


SUPPORTED_REPLAY_LANES = ("stretching", "buckling", "torsion", "eigenmodes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        help="Repeatable GV experiment lane to include. Defaults to all non-experimental GV experiments.",
    )
    parser.add_argument(
        "--source-pdf",
        action="append",
        default=[],
        help="Repeatable source PDF path captured in campaign provenance.",
    )
    parser.add_argument(
        "--fixture-lane",
        action="append",
        default=[],
        help="Optional lane fixture JSON file to ingest instead of synthesizing a fixture lane.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--fixture-mode", action="store_true", default=False)
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        default=False,
        help="Use the paper-faithful sweep sizes and controls for operational Mirheo replay lanes.",
    )
    parser.add_argument(
        "--stretching-point-start",
        type=int,
        default=None,
        help="Optional inclusive start index for paper-exact stretching control points.",
    )
    parser.add_argument(
        "--stretching-point-stop",
        type=int,
        default=None,
        help="Optional exclusive stop index for paper-exact stretching control points.",
    )
    return parser


def _default_lanes() -> list[str]:
    return list(SUPPORTED_REPLAY_LANES)


def _resolve_lanes(args: argparse.Namespace) -> list[str]:
    lanes = list(args.lane or [])
    resolved = lanes or _default_lanes()
    unsupported = sorted(set(resolved) - set(SUPPORTED_REPLAY_LANES))
    if unsupported:
        raise ValueError(
            "Unsupported GV paper replay lane(s): "
            + ", ".join(unsupported)
            + f". Supported lanes are: {', '.join(SUPPORTED_REPLAY_LANES)}."
        )
    return resolved


def _resolve_source_pdfs(values: list[str], *, default_values: tuple[str, ...] = ()) -> tuple[Path, ...]:
    selected = tuple(values) if values else default_values
    return tuple(Path(item).expanduser().resolve() for item in selected)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.number, np.bool_)):
        return value.item()
    return value


def _channel_arrays(channels: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, values in channels.items():
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"Operational lane channel {name!r} is empty.")
        arrays[str(name)] = array
    return arrays


def _data_ranges(channels: Mapping[str, np.ndarray]) -> tuple[GVPaperReplayDataRange, ...]:
    ranges: list[GVPaperReplayDataRange] = []
    for name, array in sorted(channels.items()):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError(f"Operational lane channel {name!r} has no finite values.")
        ranges.append(
            GVPaperReplayDataRange(
                name=name,
                minimum=float(np.min(finite)),
                maximum=float(np.max(finite)),
            )
        )
    return tuple(ranges)


def _finite_checks(channels: Mapping[str, np.ndarray]) -> tuple[GVPaperReplayFiniteCheck, ...]:
    checks: list[GVPaperReplayFiniteCheck] = []
    for name, array in sorted(channels.items()):
        finite_mask = np.isfinite(array)
        finite_count = int(np.count_nonzero(finite_mask))
        total = int(array.size)
        nonfinite_count = total - finite_count
        checks.append(
            GVPaperReplayFiniteCheck(
                name=name,
                passed=nonfinite_count == 0,
                finite_ratio=float(finite_count / total),
                nonfinite_count=nonfinite_count,
            )
        )
    return tuple(checks)


def _lane_controls(result: object) -> dict[str, Any]:
    plan = getattr(result, "plan", None)
    if plan is not None and hasattr(plan, "controls"):
        return dict(plan.controls)
    provenance = getattr(result, "provenance", {})
    if isinstance(provenance, Mapping):
        lane_plan = provenance.get("lane_plan")
        if isinstance(lane_plan, Mapping) and isinstance(lane_plan.get("controls"), Mapping):
            return dict(lane_plan["controls"])
    if hasattr(result, "controls"):
        return dict(getattr(result, "controls"))
    raise ValueError("Operational lane result does not expose controls.")


def _lane_geometry(result: object) -> dict[str, float]:
    plan = getattr(result, "plan", None)
    if plan is not None and hasattr(plan, "geometry_radius") and hasattr(plan, "geometry_height"):
        return {"radGV": float(plan.geometry_radius), "height": float(plan.geometry_height)}
    if hasattr(result, "geometry"):
        return {str(name): float(value) for name, value in dict(getattr(result, "geometry")).items()}
    raise ValueError("Operational lane result does not expose geometry.")


def _lane_material_parameters(result: object) -> dict[str, float]:
    plan = getattr(result, "plan", None)
    if plan is not None and hasattr(plan, "material_parameters"):
        return {str(name): float(value) for name, value in dict(plan.material_parameters).items()}
    if hasattr(result, "material_parameters"):
        return {str(name): float(value) for name, value in dict(getattr(result, "material_parameters")).items()}
    raise ValueError("Operational lane result does not expose material_parameters.")


def _lane_work_dirs(result: object) -> tuple[str, ...]:
    raw = getattr(result, "raw_sample_result", None)
    if isinstance(raw, Mapping):
        work_dirs = raw.get("work_dirs", ())
        if isinstance(work_dirs, tuple | list):
            return tuple(str(path) for path in work_dirs)
    provenance = getattr(result, "provenance", {})
    if isinstance(provenance, Mapping):
        work_dirs = provenance.get("work_dirs", ())
        if isinstance(work_dirs, tuple | list):
            return tuple(str(path) for path in work_dirs)
    return ()


def _lane_runtime_ids(result: object) -> tuple[str, ...]:
    manifests: list[Any] = []
    raw = getattr(result, "raw_sample_result", None)
    if isinstance(raw, Mapping):
        runtime_manifests = raw.get("runtime_manifests", ())
        if isinstance(runtime_manifests, tuple | list):
            manifests.extend(runtime_manifests)
    provenance = getattr(result, "provenance", {})
    if isinstance(provenance, Mapping):
        runtime_manifests = provenance.get("runtime_manifests", ())
        if isinstance(runtime_manifests, tuple | list):
            manifests.extend(runtime_manifests)

    identifiers: list[str] = []
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            continue
        value = manifest.get("dataset_id") or manifest.get("control_id")
        if value is not None:
            identifiers.append(str(value))
    return tuple(dict.fromkeys(identifiers))


def _slurm_job_ids() -> tuple[str, ...]:
    ids = [
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_JOB_ID"),
    ]
    return tuple(dict.fromkeys(str(item) for item in ids if item))


def _active_slurm_partition() -> str | None:
    value = os.environ.get("SLURM_JOB_PARTITION")
    if value is None:
        return None
    text = value.strip()
    return text or None


def _validate_stretching_partition_policy(
    *,
    plan: object,
    paper_exact: bool,
    lane: str,
) -> None:
    if lane != "stretching" or not paper_exact:
        return
    partition = _active_slurm_partition()
    if partition != "dev":
        return
    controls = getattr(plan, "controls", {})
    if not isinstance(controls, Mapping):
        return
    tot_force = controls.get("tot_force")
    if not isinstance(tot_force, tuple | list):
        return
    if len(tot_force) <= 15:
        return
    raise ValueError(
        "Paper-exact stretching sweeps with more than 15 control points are not allowed on the dev partition. "
        "Use --stretching-point-start/--stretching-point-stop to keep each shard at 15 points or fewer, "
        "or rerun on the gpu partition."
    )


def _manifest_without_full_channels(result: object) -> dict[str, Any]:
    if hasattr(result, "to_manifest"):
        payload = dict(result.to_manifest())
    elif hasattr(result, "manifest"):
        payload = dict(getattr(result, "manifest"))
    else:
        payload = {}
    channels = getattr(result, "channels", {})
    if isinstance(channels, Mapping):
        payload["channels"] = {
            str(name): {"shape": list(np.asarray(values).shape)}
            for name, values in channels.items()
        }
    return payload


def _write_lane_summary(
    *,
    lane: str,
    result: object,
    lane_root: Path,
    plot_paths: tuple[Path, ...],
) -> Path:
    output_root = lane_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"{lane}_summary.json"
    channels = _channel_arrays(getattr(result, "channels"))
    summary_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "lane": lane,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "manifest": _manifest_without_full_channels(result),
                    "summary": getattr(result, "summary", {}),
                    "data_ranges": [item.to_manifest() for item in _data_ranges(channels)],
                    "finite_checks": [item.to_manifest() for item in _finite_checks(channels)],
                    "work_dirs": _lane_work_dirs(result),
                    "runtime_ids": _lane_runtime_ids(result),
                    "slurm_job_ids": _slurm_job_ids(),
                    "plot_paths": [str(path) for path in plot_paths],
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary_path


def _build_operational_lane_record(
    *,
    lane: str,
    result: object,
    source_pdfs: tuple[Path, ...],
    lane_root: Path,
    plot_paths: tuple[Path, ...],
    runtime_command: tuple[str, ...],
) -> GVPaperReplayLaneRecord:
    channels = _channel_arrays(getattr(result, "channels"))
    summary_path = _write_lane_summary(
        lane=lane,
        result=result,
        lane_root=lane_root,
        plot_paths=plot_paths,
    )
    work_dirs = _lane_work_dirs(result)
    runtime_ids = _lane_runtime_ids(result)
    output_paths: dict[str, Path] = {"summary": summary_path, "lane_root": lane_root}
    for index, work_dir in enumerate(work_dirs):
        output_paths[f"work_dir_{index:03d}"] = Path(work_dir)
    return GVPaperReplayLaneRecord(
        lane=lane,
        experiment=lane,
        mode="operational",
        source_pdfs=source_pdfs,
        runtime_commands=(runtime_command,),
        material_parameters=_lane_material_parameters(result),
        geometry=_lane_geometry(result),
        controls=_lane_controls(result),
        output_paths=output_paths,
        plot_paths=plot_paths,
        validation_status="passed",
        data_ranges=_data_ranges(channels),
        finite_checks=_finite_checks(channels),
        slurm_job_ids=_slurm_job_ids(),
        runtime_ids=runtime_ids,
        notes=("Operational GV paper replay lane generated through the public GV sampling API.",),
    )


def _run_operational_lane(
    *,
    lane: str,
    campaign_id: str,
    campaign_root: Path,
    source_pdfs: tuple[Path, ...],
    runtime_command: tuple[str, ...],
    paper_exact: bool = False,
    stretching_point_start: int | None = None,
    stretching_point_stop: int | None = None,
) -> GVPaperReplayLaneRecord:
    try:
        previous_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        profile = load_gv_paper_replay_profile()
        validate_gv_paper_replay_profile(profile, require_positive_material_parameters=False)
        lane_material_values = getattr(profile, "material_values_for_lane", None)
        material_parameters = (
            lane_material_values(lane)
            if paper_exact and callable(lane_material_values)
            else profile.material_values()
        )
        geometry = profile.geometry.values()
        lane_root_relative = CAMPAIGN_RUN_ROOT / campaign_id / "lanes" / lane
        lane_root = campaign_root / "lanes" / lane
        raw_provenance = profile.provenance.to_dict()

        if lane == "stretching":
            plan = plan_stretching_paper_replay_lane(
                campaign_id=campaign_id,
                geometry_radius=geometry["radGV"],
                geometry_height=geometry["height"],
                material_parameters=material_parameters,
                paper_exact=paper_exact,
                output_root=lane_root_relative,
                raw_provenance=raw_provenance,
                point_start=stretching_point_start,
                point_stop=stretching_point_stop,
            )
            _validate_stretching_partition_policy(
                plan=plan,
                paper_exact=paper_exact,
                lane=lane,
            )
            result = run_stretching_paper_replay_lane(plan)
            plot_paths = (result.plot_path.resolve(),) if result.plot_path is not None else ()
        elif lane == "torsion":
            plan = plan_torsion_paper_replay_lane(
                campaign_id=campaign_id,
                geometry_radius=geometry["radGV"],
                geometry_height=geometry["height"],
                material_parameters=material_parameters,
                paper_exact=paper_exact,
                output_root=lane_root_relative,
                raw_provenance=raw_provenance,
            )
            result = run_torsion_paper_replay_lane(plan)
            plot_paths = (result.plot_path.resolve(),) if result.plot_path is not None else ()
        elif lane == "buckling":
            plan = plan_buckling_paper_replay_lane(
                material_parameters=material_parameters,
                radGV=geometry["radGV"],
                height=geometry["height"],
                paper_exact=paper_exact,
                output_root=lane_root_relative,
            )
            result = run_buckling_paper_replay_lane(plan)
            plot_path = plot_buckling_paper_replay(
                result,
                output_path=lane_root_relative / "plots" / "buckling_relative_volume.png",
            )
            plot_paths = (plot_path.resolve(),)
        elif lane == "eigenmodes":
            plan = plan_eigenmodes_paper_replay_lane(
                material_parameters=material_parameters,
                radGV=geometry["radGV"],
                height=geometry["height"],
                bpress=-91.0,
                mode_count=30,
                paper_exact=paper_exact,
                output_root=lane_root_relative,
            )
            result = run_eigenmodes_paper_replay_lane(plan)
            plot_path = plot_eigenmodes_paper_replay(
                result,
                output_path=lane_root_relative / "plots" / "eigenmode_spectrum.png",
            )
            plot_paths = (plot_path.resolve(),)
        else:
            raise ValueError(f"Unsupported GV paper replay lane {lane!r}.")
    finally:
        os.chdir(previous_cwd)

    return _build_operational_lane_record(
        lane=lane,
        result=result,
        source_pdfs=source_pdfs,
        lane_root=lane_root,
        plot_paths=plot_paths,
        runtime_command=runtime_command,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        campaign_root = resolve_campaign_root(
            repo_root=REPO_ROOT,
            campaign_id=args.campaign_id,
            output_root=args.output_root,
        )
    except ValueError as exc:
        parser.error(str(exc))

    profile = load_gv_paper_replay_profile()
    try:
        lanes = _resolve_lanes(args)
    except ValueError as exc:
        parser.error(str(exc))
    source_pdfs = _resolve_source_pdfs(
        args.source_pdf,
        default_values=(profile.provenance.paper_pdf_path, profile.provenance.si_pdf_path),
    )
    campaign_root.mkdir(parents=True, exist_ok=True)

    lane_records = []
    for fixture_path in args.fixture_lane:
        lane_records.append(load_lane_record_fixture(fixture_path))
    if not lane_records:
        runtime_command = (sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:]))
        if args.fixture_mode or args.dry_run:
            for lane in lanes:
                lane_records.append(
                    build_fixture_lane_record(
                        repo_root=REPO_ROOT,
                        campaign_root=campaign_root,
                        lane=lane,
                        source_pdfs=source_pdfs,
                        fixture_mode=bool(args.fixture_mode),
                    )
                )
        else:
            for lane in lanes:
                lane_records.append(
                    _run_operational_lane(
                        lane=lane,
                        campaign_id=str(args.campaign_id),
                        campaign_root=campaign_root,
                        source_pdfs=source_pdfs,
                        runtime_command=runtime_command,
                        paper_exact=bool(args.paper_exact),
                        stretching_point_start=args.stretching_point_start,
                        stretching_point_stop=args.stretching_point_stop,
                    )
                )

    aggregate_source_pdfs = tuple(
        sorted(
            set(source_pdfs).union(*(set(record.source_pdfs) for record in lane_records)),
            key=str,
        )
    )

    manifest = GVPaperReplayCampaignManifest(
        campaign_id=str(args.campaign_id),
        campaign_root=campaign_root,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        schema_version=MANIFEST_SCHEMA_VERSION,
        git_head=collect_git_head(REPO_ROOT),
        dry_run=bool(args.dry_run),
        fixture_mode=bool(args.fixture_mode),
        source_pdfs=aggregate_source_pdfs,
        lanes=tuple(lane_records),
        comparison_packets=build_comparison_packet_skeletons(tuple(lane_records)),
        manifest_path=campaign_root / CAMPAIGN_MANIFEST_FILENAME,
        slurm_job_ids=tuple(sorted({item for lane in lane_records for item in lane.slurm_job_ids})),
        runtime_ids=tuple(sorted({item for lane in lane_records for item in lane.runtime_ids})),
    )
    manifest_path = write_campaign_manifest(manifest=manifest)
    print(f"GV paper replay manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
