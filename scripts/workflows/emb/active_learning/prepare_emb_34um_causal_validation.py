#!/usr/bin/env python3
"""Prepare causal EMB 3.4um AL-vs-LHS validation manifests."""

from __future__ import annotations

import argparse
import json
import hashlib
import struct
from collections.abc import Sequence
import sys
from pathlib import Path
from typing import Any, Mapping
import zlib


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _script_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


from meso_uq.active_learning import Candidate
from meso_uq.active_learning.contracts import candidate_hash
from meso_uq.active_learning.dpd_sampling_gate import build_and_render_active_learning_dpd_sampling_gate
from meso_uq.active_learning.emb_34um_causal_validation_design import (
    EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
    EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS,
    EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION,
    build_emb_34um_causal_validation_campaign_manifest,
)
from meso_uq.active_learning.emb_34um_dpd_adapter import (
    EMB_34UM_RUNTIME_FINGERPRINT,
    EMB_34UM_DPD_SCHEMA_VERSION,
)


def _stage_batch_root(campaign_root: Path, seed: int, mode: str) -> Path:
    return campaign_root / f"seed-{seed:03d}" / str(mode)

DEFAULT_SCRATCH_ROOT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_causal_validation"
)
DEFAULT_VAULT_ROOT = Path(
    "/home/it4i-bbenvegnen/knowledge/vault/07 Sessions/UQ_DPD/assets/active_learning_emb_34um_causal_validation"
)
DEFAULT_FORCE_GRID_DATA = Path("emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat")
DEFAULT_RENDER_COVERAGE_PLOT = False


def _load_force_grid(path: Path) -> tuple[float, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Force grid file does not exist: {path}")
    force_grid: tuple[float, ...] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        tokens = text.split()
        if len(tokens) < 10:
            raise ValueError(f"Unexpected force-grid row in {path!s}: {len(tokens)} tokens.")
        n_after_header = len(tokens) - 8
        if n_after_header % 2 != 0:
            raise ValueError(f"Unexpected force-column layout in {path!s}.")
        point_count = n_after_header // 2
        row_tokens = tokens[8 + point_count : 8 + 2 * point_count]
        row = tuple(float(item) for item in row_tokens)
        if not row:
            raise ValueError(f"No force points in {path!s}.")
        if force_grid is None:
            force_grid = row
        elif force_grid != row:
            raise ValueError(f"Mismatched force-grid rows in {path!s}.")
    if force_grid is None:
        raise ValueError(f"No force points found in {path!s}.")
    return force_grid


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + type_code
        + data
        + struct.pack(">I", zlib.crc32(type_code + data) & 0xFFFFFFFF)
    )


def _fallback_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 16
    height = 16
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows: list[bytes] = []
    for row_index in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((210, 230, 238) if (row_index + x) % 2 else (31, 89, 96))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _static_manifest_points(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    points: list[dict[str, Any]] = []
    for seed_payload in manifest.get("seeds", ()):
        if not isinstance(seed_payload, Mapping):
            continue
        seed = int(seed_payload.get("seed", 0))
        for stage_name in ("shared_initial", "validation"):
            batch = seed_payload.get(stage_name)
            if not isinstance(batch, Mapping):
                continue
            for record in batch.get("candidate_records", ()):
                if not isinstance(record, Mapping):
                    continue
                points.append(
                    {
                        "seed": seed,
                        "stage": stage_name,
                        "ka": float(record["ka"]),
                        "kb": float(record["kb"]),
                    }
                )
        for step_index, batch in enumerate(seed_payload.get("lhs_steps", ()), start=1):
            if not isinstance(batch, Mapping):
                continue
            for record in batch.get("candidate_records", ()):
                if not isinstance(record, Mapping):
                    continue
                points.append(
                    {
                        "seed": seed,
                        "stage": f"lhs-step-{step_index:02d}",
                        "ka": float(record["ka"]),
                        "kb": float(record["kb"]),
                    }
                )
    return tuple(points)


def _write_coverage_plot_requirements_only(
    campaign_root: Path,
    *,
    manifest: Mapping[str, Any],
    force_grid_size: int,
) -> tuple[Path, Path]:
    plot_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME
    sidecar_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME
    points = _static_manifest_points(manifest)
    plot_status = "rendered"
    try:
        import matplotlib.pyplot as plt  # type: ignore

        stage_colors = {
            "shared_initial": "#264653",
            "validation": "#2a9d8f",
        }
        fig, axis = plt.subplots(1, 1, figsize=(7.0, 5.0))
        for stage_name in ("shared_initial", "validation"):
            stage_points = [item for item in points if item["stage"] == stage_name]
            if not stage_points:
                continue
            axis.scatter(
                [item["ka"] for item in stage_points],
                [item["kb"] for item in stage_points],
                s=12,
                alpha=0.75,
                color=stage_colors[stage_name],
                label=stage_name,
            )
        lhs_points = [item for item in points if str(item["stage"]).startswith("lhs-step-")]
        if lhs_points:
            axis.scatter(
                [item["ka"] for item in lhs_points],
                [item["kb"] for item in lhs_points],
                s=8,
                alpha=0.35,
                color="#e76f51",
                label="lhs planned",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("ka")
        axis.set_ylabel("kb")
        axis.set_title("EMB 3.4um causal validation static coverage")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception:
        plot_status = "fallback_png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(_fallback_png())

    plot_root = {
        "status": plot_status,
        "required": True,
        "required_force_grid_size": force_grid_size,
        "plot_path": str(plot_path),
        "static_point_count": len(points),
        "static_policy_counts": {
            "shared_initial": sum(1 for item in points if item["stage"] == "shared_initial"),
            "validation": sum(1 for item in points if item["stage"] == "validation"),
            "lhs": sum(1 for item in points if str(item["stage"]).startswith("lhs-step-")),
        },
        "adaptive_al_note": "AL points are selected and plotted after each select-render stage.",
    }
    sidecar_path.write_text(json.dumps(plot_root, indent=2, sort_keys=True), encoding="utf-8")
    return plot_path, sidecar_path


def _coerce_force_grid(values: Sequence[object], *, field_name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must be a non-empty sequence.")
    normalized: list[float] = []
    for value in values:
        normalized.append(float(value))
    return tuple(normalized)


def _coerce_stage_records(value: object, *, stage: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{stage} candidate_records must be a list.")
    return tuple(dict(item) for item in value)


def _build_candidates_for_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    force_grid: tuple[float, ...],
) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = []
    for record in records:
        ka = float(record["ka"])
        kb = float(record["kb"])
        candidate_id = str(record["candidate_id"])
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": ka,
                    "kb": kb,
                    "force_grid": list(force_grid),
                    "runtime_fingerprint": {
                        **dict(EMB_34UM_RUNTIME_FINGERPRINT),
                        **dict(record.get("runtime_fingerprint", {})),
                    },
                },
                metadata={
                    "seed": int(record["seed"]),
                    "step": int(record["step"]),
                    "selection_seed": int(record["selection_seed"]),
                    "selection_mode": str(record.get("selection_mode", "causal_fresh_only")),
                    "method": str(record["method"]),
                    "force_grid_signature": str(record.get("force_grid_signature", "")),
                    "causal_validation_mode": True,
                    "fresh_only": True,
                    "request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
                    "output_root": str(record["output_root"]),
                    "vault_output_root": str(record["vault_output_root"]),
                    "schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
                },
            )
        )
    if not candidates:
        raise ValueError("Each batch must contain at least one candidate.")
    return tuple(candidates)


def _lineage_payload(candidates: tuple[Candidate, ...]) -> dict[str, Any]:
    entries = [
        {
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate_hash(candidate),
            "family": "emb",
        }
        for candidate in candidates
    ]
    lineage_fingerprint = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"candidate_lineage": entries, "lineage_fingerprint": lineage_fingerprint}


def _ensure_request_payload_schema_version(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered_payload = payload.get("rendered_payload")
    if not isinstance(rendered_payload, dict):
        raise ValueError(f"{path} has a non-dict rendered_payload object.")
    request_payload = rendered_payload.get("request_payload")
    if not isinstance(request_payload, dict):
        raise ValueError(f"{path} has a non-dict request_payload object.")
    schema_version = request_payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError(f"{path} is missing request_payload.schema_version.")
    return str(schema_version)


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
    try:
        result = build_and_render_active_learning_dpd_sampling_gate(
            candidates,
            run_id=run_id,
            iteration="0",
            platform="karolina",
            walltime=walltime,
            gpu_count=1,
            campaign_root=batch_root,
            batch_id=batch_id,
            overwrite=True,
            provenance_tags={"source_issue": "MESOUQ-AL-34UM-CAUSAL"},
        )
    except ModuleNotFoundError as exc:
        if "matplotlib" not in str(exc):
            raise
        from meso_uq.dpd_sampling import boundary as _dpd_boundary

        original_render_validation_plot = _dpd_boundary._render_validation_plot

        def _fallback_render_validation_plot(request, report):
            plot_path = request.campaign_root / "dpd_sampling_validation_plot.png"
            sidecar_path = request.campaign_root / "dpd_sampling_validation_plot.png.json"
            plot_path.write_text("", encoding="utf-8")
            sidecar_path.write_text("{}", encoding="utf-8")
            return plot_path, sidecar_path

        _dpd_boundary._render_validation_plot = _fallback_render_validation_plot
        try:
            result = build_and_render_active_learning_dpd_sampling_gate(
                candidates,
                run_id=run_id,
                iteration="0",
                platform="karolina",
                walltime=walltime,
                gpu_count=1,
                campaign_root=batch_root,
                batch_id=batch_id,
                overwrite=True,
                provenance_tags={"source_issue": "MESOUQ-AL-34UM-CAUSAL"},
            )
        finally:
            _dpd_boundary._render_validation_plot = original_render_validation_plot

    rendered_candidate_manifests = tuple(
        manifest_path
        for manifest_path in result.render_result.rendered_manifest_paths
        if manifest_path.name == "dpd_sampling_candidate_manifest.json"
    )
    if len(rendered_candidate_manifests) < len(candidates):
        raise ValueError(
            f"Expected at least {len(candidates)} rendered candidate manifests, got {len(rendered_candidate_manifests)}."
        )

    payload = {
        "batch_root": str(batch_root),
        "batch_id": batch_id,
        "run_id": run_id,
        "manifest_path": str(result.manifest_path),
        "report_path": str(result.report_path),
        "candidate_count": len(candidates),
        "expected_request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
        "request_payload_schema_versions": [
            _ensure_request_payload_schema_version(manifest_path)
            for manifest_path in rendered_candidate_manifests[: len(candidates)]
        ],
        "rendered_candidate_manifests": [str(item) for item in rendered_candidate_manifests[: len(candidates)]],
        "expected_output_roots": [
            record["output_root"] for record in result.manifest.get("candidate_records", []) if isinstance(record, Mapping)
        ],
        "scheduler_boundary": result.manifest["scheduler_boundary"],
        "candidate_lineage": _lineage_payload(candidates)["candidate_lineage"],
        "lineage_fingerprint": _lineage_payload(candidates)["lineage_fingerprint"],
        "submission_expected": {
            "platform": "karolina",
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
            "render_only": result.report["criteria"]["render_only_mode"],
            "submission_commands_empty": not bool(
                result.render_result.validation_report.submission.get("submission_commands", [])
            ),
            "submitted": bool(result.render_result.validation_report.submission.get("submitted")),
        },
    }
    payload["submission_expected"]["submission_commands"] = (
        result.render_result.validation_report.submission.get("submission_commands", [])
    )
    payload["submission_expected"]["candidate_count_positive"] = result.report["criteria"]["candidate_count_positive"]
    payload["submission_expected"]["single_experiment"] = result.report["criteria"]["single_experiment"]
    payload["submission_expected"]["single_family"] = result.report["criteria"]["single_family"]

    if any(item != EMB_34UM_DPD_SCHEMA_VERSION for item in payload["request_payload_schema_versions"]):
        raise ValueError(
            f"Rendered EMB request payload schema mismatch in batch {batch_id}: "
            f"{sorted(set(payload['request_payload_schema_versions']))}"
        )

    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "emb_34um_batch_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _iter_batch_stages(seed_payload: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    return (
        ("shared_initial", seed_payload["shared_initial"]),
        ("validation", seed_payload["validation"]),
        *tuple(
            (f"lhs-step-{step:02d}", batch) for step, batch in enumerate(seed_payload["lhs_steps"], start=1)
        ),
    )


def prepare_emb_34um_causal_validation(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    step_count: int,
    seed_count: int,
    include_plot_requirements: bool,
) -> dict[str, Any]:
    if step_count < EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS:
        raise ValueError(
            f"step_count must be at least {EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS}."
        )
    if seed_count < len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS):
        raise ValueError(
            f"seed_count must include all primary seeds {EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS}."
        )
    if seed_count > EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS:
        raise ValueError(
            f"seed_count must be in 1..{EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS}, got {seed_count}."
        )

    force_grid = _load_force_grid(force_grid_path)
    campaign_root = scratch_root / timestamp
    vault_root_timestamp = vault_root / timestamp

    manifest = build_emb_34um_causal_validation_campaign_manifest(
        timestamp=timestamp,
        run_id_prefix=run_id_prefix,
        campaign_root=campaign_root,
        scratch_root=scratch_root,
        vault_root_timestamp=vault_root_timestamp,
        force_grid=force_grid,
        step_count=step_count,
        seed_count=seed_count,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        include_coverage_plot_requirements=include_plot_requirements,
    )
    manifest["schema_version"] = EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION
    manifest["command_inventory_path"] = str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME)
    manifest["manifest_filename"] = EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME
    manifest["campaign_root"] = str(campaign_root)
    manifest["vault_root"] = str(vault_root)

    command_inventory_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME
    manifest_path = campaign_root / EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME

    runtime_batches: list[dict[str, Any]] = []
    for seed_payload in manifest["seeds"]:
        seed = int(seed_payload["seed"])
        for stage_name, stage_payload in _iter_batch_stages(seed_payload):
            batch_records = _coerce_stage_records(stage_payload.get("candidate_records"), stage=stage_name)
            batch_root = _stage_batch_root(campaign_root, seed, stage_name)
            if not batch_records:
                runtime_batches.append(
                    {
                        "batch_root": str(batch_root),
                        "batch_id": stage_name,
                        "run_id": f"{run_id_prefix}-seed{seed:03d}-{stage_name}",
                        "candidate_count": 0,
                        "status": "adaptive_placeholder_not_rendered",
                    }
                )
                continue
            candidates = _build_candidates_for_batch(
                batch_records,
                force_grid=force_grid,
            )
            runtime_batches.append(
                _render_batch(
                    candidates=candidates,
                    batch_root=batch_root,
                    run_id=f"{run_id_prefix}-seed{seed:03d}-{stage_name}",
                    batch_id=stage_name,
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
        "manifest_path": manifest_path,
        "command_inventory_path": command_inventory_path,
        "manifest": manifest,
        "campaign_root": campaign_root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True, help="Campaign timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument(
        "--scratch-root",
        default=str(DEFAULT_SCRATCH_ROOT),
        help="Base path for scratch campaign output.",
    )
    parser.add_argument(
        "--vault-root",
        default=str(DEFAULT_VAULT_ROOT),
        help="Base path for vault artifacts.",
    )
    parser.add_argument(
        "--force-grid",
        default=str(DEFAULT_FORCE_GRID_DATA),
        help="3.4um force-grid source file (samples_all.dat).",
    )
    parser.add_argument("--walltime", default="01:00:00")
    parser.add_argument("--concurrent-jobs", type=int, default=30)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--run-id-prefix", default="emb-34um-causal-validation")
    parser.add_argument("--step-count", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS))
    parser.add_argument(
        "--include-coverage-plot-requirements",
        action="store_true",
        default=DEFAULT_RENDER_COVERAGE_PLOT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _script_root_path = _script_root()
    repo_src_root = _script_root_path / "src"
    if str(repo_src_root) not in sys.path:
        sys.path.insert(0, str(repo_src_root))
    if str(_script_root_path) not in sys.path:
        sys.path.insert(0, str(_script_root_path))

    result = prepare_emb_34um_causal_validation(
        timestamp=args.timestamp,
        scratch_root=Path(args.scratch_root),
        vault_root=Path(args.vault_root),
        force_grid_path=Path(args.force_grid),
        walltime=args.walltime,
        concurrent_jobs=args.concurrent_jobs,
        retry_limit=args.retry_limit,
        run_id_prefix=args.run_id_prefix,
        step_count=args.step_count,
        seed_count=args.seed_count,
        include_plot_requirements=args.include_coverage_plot_requirements,
    )
    print(f"manifest_path={result['manifest_path']}")
    print(f"command_inventory_path={result['command_inventory_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
