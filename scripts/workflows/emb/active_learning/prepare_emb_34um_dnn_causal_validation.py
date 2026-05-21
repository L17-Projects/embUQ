#!/usr/bin/env python3
"""Prepare and render EMB 3.4um DNN causal-validation campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Mapping


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _script_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from meso_uq.active_learning import Candidate
from meso_uq.active_learning.contracts import candidate_hash
from meso_uq.active_learning.dpd_sampling_gate import (
    build_and_render_active_learning_dpd_sampling_gate,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (
    EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    build_emb_34um_dnn_causal_validation_campaign_manifest,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
)
from meso_uq.active_learning.emb_34um_dpd_adapter import EMB_34UM_DPD_SCHEMA_VERSION


DEFAULT_SCRATCH_ROOT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_dnn_causal_validation"
)
DEFAULT_VAULT_ROOT = Path(
    "/home/it4i-bbenvegnen/knowledge/vault/07 Sessions/UQ_DPD/assets/active_learning_emb_34um_causal_validation"
)


def _load_force_grid(path: Path) -> tuple[float, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Force-grid file does not exist: {path}")

    values: list[float] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        tokens = text.split()
        if len(tokens) < 10:
            raise ValueError(f"Unexpected force-grid row in {path}: {len(tokens)} tokens.")
        n_after_header = len(tokens) - 8
        if n_after_header % 2:
            raise ValueError(f"Malformed force-grid row in {path}.")
        point_count = n_after_header // 2
        point_tokens = tokens[8 + point_count : 8 + 2 * point_count]
        row = [float(token) for token in point_tokens]
        if not row:
            raise ValueError(f"No force points found in {path}")
        if values is None:
            values = row
        elif values != row:
            raise ValueError(f"Inconsistent force-grid rows in {path}")

    if values is None:
        raise ValueError(f"No force-grid data found in {path}")
    return tuple(float(item) for item in values)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _fallback_png() -> bytes:
    width = 16
    height = 16
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((31, 89, 96) if (x + y) % 2 == 0 else (210, 230, 238))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return header + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _coerce_force_grid(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("force_grid must be non-empty.")
    return tuple(float(item) for item in values)


def _coerce_stage_records(records: Any, *, stage: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(records, (list, tuple)):
        raise ValueError(f"{stage} candidate_records must be a list or tuple.")
    return tuple(dict(item) for item in records)


def _build_candidates_for_batch(
    records: tuple[Mapping[str, Any], ...],
    *,
    force_grid: tuple[float, ...],
) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = []
    for record in records:
        candidate_id = str(record["candidate_id"])
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                parameters={
                    "family": str(record["family"]),
                    "experiment": str(record["experiment"]),
                    "ka": float(record["ka"]),
                    "kb": float(record["kb"]),
                    "force_grid": list(force_grid),
                },
                metadata={
                    k: v
                    for k, v in {
                        "selection_seed": int(record.get("selection_seed", 0)),
                        "selection_mode": str(record.get("selection_mode", "dnn_causal_fresh_only")),
                        "selection_source": str(record.get("selection_source", "fresh_dpd")),
                        "selection_status": str(record.get("selection_status", "rendered")),
                        "selection_payload": record.get("selection_payload", {}),
                    }.items()
                },
            )
        )
    return tuple(candidates)


def _ensure_request_payload_schema_version(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered_payload = payload.get("rendered_payload")
    if isinstance(rendered_payload, Mapping):
        request = rendered_payload.get("request_payload")
    else:
        request = payload.get("request_payload")

    if not isinstance(request, Mapping):
        normalized = payload.get("normalized_payload")
        request = normalized if isinstance(normalized, Mapping) else None

    if not isinstance(request, Mapping):
        raise ValueError(f"{path} is missing a rendered request payload.")
    schema_version = request.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError(f"{path} is missing request payload schema_version.")
    return schema_version


def _write_placeholder_batch_summary(
    *,
    batch_root: Path,
    mode: str,
    candidate_count: int,
    output_roots: list[str],
) -> dict[str, Any]:
    payload = {
        "schema_version": "meso_uq.active_learning.emb_34um_dnn_causal_validation_batch_summary.v1",
        "status": "adaptive_placeholder_not_rendered",
        "campaign_root": str(batch_root.parent.parent),
        "stage": mode,
        "mode": mode,
        "candidate_count": int(candidate_count),
        "expected_output_roots": output_roots,
        "selection_required": "runtime_selection_required",
        "render_only": True,
        "submission_expected": {
            "render_only": True,
            "submission_commands_empty": True,
            "submitted": False,
        },
        "rendered_candidate_manifests": [],
    }
    summary_path = batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
    _write_json(summary_path, payload)
    return payload


def _render_batch(
    *,
    candidates: tuple[Candidate, ...],
    batch_root: Path,
    run_id: str,
    batch_id: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Each batch must contain at least one candidate.")

    report = build_and_render_active_learning_dpd_sampling_gate(
        candidates,
        run_id=run_id,
        iteration="0",
        platform="karolina",
        walltime=walltime,
        gpu_count=1,
        campaign_root=batch_root,
        batch_id=batch_id,
        overwrite=True,
        provenance_tags={"source_issue": "MESOUQ-AL-34UM-DNN-CAUSAL"},
    )
    rendered_candidate_manifests = tuple(
        manifest_path
        for manifest_path in report.render_result.rendered_manifest_paths
        if manifest_path.name == "dpd_sampling_candidate_manifest.json"
    )
    if len(rendered_candidate_manifests) < len(candidates):
        raise ValueError(
            f"Expected at least {len(candidates)} rendered candidate manifests, got {len(rendered_candidate_manifests)}."
        )

    request_versions = [
        _ensure_request_payload_schema_version(Path(manifest_path))
        for manifest_path in rendered_candidate_manifests[: len(candidates)]
    ]
    if any(item != EMB_34UM_DPD_SCHEMA_VERSION for item in request_versions):
        raise ValueError(
            f"DPD request schema mismatch in batch {batch_id}: {sorted(set(request_versions))}"
        )

    payload = {
        "batch_root": str(batch_root),
        "batch_id": batch_id,
        "run_id": run_id,
        "manifest_path": str(report.manifest_path),
        "report_path": str(report.report_path),
        "candidate_count": len(candidates),
        "expected_output_roots": [
            str(record["output_root"])
            for record in report.manifest.get("candidate_records", [])
            if isinstance(record, Mapping) and "output_root" in record
        ],
        "expected_request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
        "request_payload_schema_versions": request_versions,
        "rendered_candidate_manifests": [str(item) for item in rendered_candidate_manifests[: len(candidates)]],
        "submission_expected": {
            "render_only": report.report["criteria"]["render_only_mode"],
            "submission_commands_empty": not bool(
                report.render_result.validation_report.submission.get("submission_commands", [])
            ),
            "submitted": bool(report.render_result.validation_report.submission.get("submitted")),
        },
    }
    batch_root.mkdir(parents=True, exist_ok=True)
    _write_json(batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME, payload)
    return payload


def _iter_batch_stages(manifest: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    stages: list[tuple[str, Mapping[str, Any]]]=[]
    stages.append(("unseen_test", manifest["unseen_test"]))

    for seed_payload in manifest.get("seeds", ()):
        seed_payload = dict(seed_payload)
        replica = int(seed_payload["replica"])
        stages.append((f"replica-{replica:03d}/shared_initial", seed_payload["shared_initial"]))
        for step, al_payload in enumerate(seed_payload["al_steps"], start=1):
            stages.append((f"replica-{replica:03d}/al-step-{step:02d}", al_payload))
        for step, lhs_payload in enumerate(seed_payload["lhs_steps"], start=1):
            stages.append((f"replica-{replica:03d}/lhs-step-{step:02d}", lhs_payload))
    return tuple(stages)


def _stage_output_root(manifest_root: Path, stage_key: str) -> Path:
    if stage_key == "unseen_test":
        return manifest_root / stage_key
    mode = Path(stage_key)
    return manifest_root / mode.parent / mode.name


def _gather_points(manifest: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    points: dict[str, list[tuple[float, float]]] = {
        "unseen_test": [],
        "shared_initial": [],
        "lhs": [],
        "al_placeholder": [],
    }
    unseen_records = manifest.get("unseen_test", {}).get("candidate_records", ())
    for raw in unseen_records:
        if isinstance(raw, Mapping):
            points["unseen_test"].append((float(raw["ka"]), float(raw["kb"])))

    for seed_payload in manifest.get("seeds", ()):
        if not isinstance(seed_payload, Mapping):
            continue
        shared_records = seed_payload.get("shared_initial", {}).get("candidate_records", ())
        for raw in shared_records:
            if isinstance(raw, Mapping):
                points["shared_initial"].append((float(raw["ka"]), float(raw["kb"])))

        lhs_steps = seed_payload.get("lhs_steps", ())
        for lhs in lhs_steps:
            if not isinstance(lhs, Mapping):
                continue
            for raw in lhs.get("candidate_records", ()):
                if isinstance(raw, Mapping):
                    points["lhs"].append((float(raw["ka"]), float(raw["kb"])))

        al_steps = seed_payload.get("al_steps", ())
        for step_payload in al_steps:
            if not isinstance(step_payload, Mapping):
                continue
            placeholders = int(step_payload.get("candidate_count", 0))
            if placeholders:
                points["al_placeholder"].append((0.0, 0.0))
    return points


def _write_coverage_plot_requirements_only(
    campaign_root: Path,
    *,
    manifest: Mapping[str, Any],
    force_grid_size: int,
) -> tuple[Path, Path]:
    points_by_stage = _gather_points(manifest)
    plot_path = campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME
    sidecar_path = campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME

    state = {"status": "rendered"}
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axis = plt.subplots(1, 1, figsize=(7.0, 5.0))
        for stage, points in points_by_stage.items():
            if not points:
                continue
            if stage == "al_placeholder":
                continue
            axis.scatter(
                [math.log10(p[0]) for p in points],
                [math.log10(p[1]) for p in points],
                s=8,
                alpha=0.8,
                label=stage,
            )
        if points_by_stage["al_placeholder"]:
            axis.text(
                0.02,
                0.98,
                f"AL placeholders: {len(points_by_stage['al_placeholder'])} cycles pending",
                transform=axis.transAxes,
                va="top",
            )
        axis.set_xlabel("log10(ka)")
        axis.set_ylabel("log10(kb)")
        axis.set_title("EMB 3.4um DNN causal validation coverage")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception:
        plot_path.write_bytes(_fallback_png())
        state["status"] = "fallback_png"

    state.update(
        {
            "required": True,
            "mode": "log10(ka)-vs-log10(kb)",
            "force_grid_size": force_grid_size,
            "point_counts": {
                "unseen_test": len(points_by_stage["unseen_test"]),
                "shared_initial": len(points_by_stage["shared_initial"]),
                "lhs": len(points_by_stage["lhs"]),
                "al_placeholders": len(points_by_stage["al_placeholder"]),
            },
            "plot_path": str(plot_path),
        }
    )
    sidecar_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return plot_path, sidecar_path


def prepare_emb_34um_dnn_causal_validation(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path | None,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    cycle_count: int,
    include_plot_requirements: bool,
) -> dict[str, Any]:
    if force_grid_path is None:
        force_grid = tuple(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
        force_grid_source = "protocol_default"
    else:
        force_grid = _load_force_grid(force_grid_path)
        force_grid_source = str(force_grid_path)
    campaign_root = scratch_root / timestamp
    vault_root_timestamp = vault_root / timestamp

    manifest = build_emb_34um_dnn_causal_validation_campaign_manifest(
        timestamp=timestamp,
        run_id_prefix=run_id_prefix,
        campaign_root=campaign_root,
        scratch_root=scratch_root,
        vault_root_timestamp=vault_root_timestamp,
        force_grid=force_grid,
        cycle_count=cycle_count,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        include_coverage_plot_requirements=include_plot_requirements,
    )
    command_inventory_path = campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME
    manifest_path = campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest["command_inventory_path"] = str(command_inventory_path)
    manifest["campaign_root"] = str(campaign_root)
    manifest["vault_root"] = str(vault_root)
    manifest["vault_root_timestamp"] = str(vault_root_timestamp)
    manifest["manifest_filename"] = EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest["force_grid_source"] = force_grid_source

    runtime_batches: list[dict[str, Any]] = []
    for stage_key, stage_payload in _iter_batch_stages(manifest):
        records = _coerce_stage_records(stage_payload.get("candidate_records", []), stage=stage_key)
        if not records:
            runtime_batches.append(
                _write_placeholder_batch_summary(
                    batch_root=_stage_output_root(campaign_root, stage_key),
                    mode=str(stage_key).replace("/", "-"),
                    candidate_count=int(stage_payload.get("candidate_count", 0)),
                    output_roots=[str(_stage_output_root(campaign_root, stage_key))],
                )
            )
            continue

        candidates = _build_candidates_for_batch(records, force_grid=_coerce_force_grid(force_grid))
        output_root = _stage_output_root(campaign_root, stage_key)
        runtime_batches.append(
            _render_batch(
                candidates=candidates,
                batch_root=output_root,
                run_id=f"{run_id_prefix}-{stage_key.replace('/', '-')}",
                batch_id=stage_key.replace("/", "-"),
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            )
        )

    manifest["runtime_batches"] = runtime_batches

    if include_plot_requirements:
        plot_path, sidecar_path = _write_coverage_plot_requirements_only(
            campaign_root,
            manifest=manifest,
            force_grid_size=len(force_grid),
        )
        manifest["validation_plot"]["coverage_plot"] = str(plot_path)
        manifest["validation_plot"]["coverage_plot_sidecar"] = str(sidecar_path)

    _write_json(manifest_path, manifest)
    _write_json(command_inventory_path, manifest["command_inventory"])
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "command_inventory_path": command_inventory_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT))
    parser.add_argument("--vault-root", default=str(DEFAULT_VAULT_ROOT))
    parser.add_argument(
        "--force-grid",
        default=None,
        help="Optional path to an exact 8-point force-grid file. Defaults to the protocol force grid.",
    )
    parser.add_argument("--walltime", default="01:00:00")
    parser.add_argument("--concurrent-jobs", type=int, default=30)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--run-id-prefix", default="emb-34um-dnn-causal-validation")
    parser.add_argument("--cycle-count", type=int, default=5)
    parser.add_argument("--include-coverage-plot-requirements", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = build_parser().parse_args(argv)

    prepare_emb_34um_dnn_causal_validation(
        timestamp=args.timestamp,
        scratch_root=Path(args.scratch_root),
        vault_root=Path(args.vault_root),
        force_grid_path=Path(args.force_grid) if args.force_grid else None,
        walltime=args.walltime,
        concurrent_jobs=args.concurrent_jobs,
        retry_limit=args.retry_limit,
        run_id_prefix=args.run_id_prefix,
        cycle_count=args.cycle_count,
        include_plot_requirements=args.include_coverage_plot_requirements,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
