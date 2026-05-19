#!/usr/bin/env python3
"""Prepare EMB 3.4um final-gate AL manifests and submission commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_SCRATCH_ROOT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_final_gate"
)
DEFAULT_VAULT_ROOT = Path(
    "/home/it4i-bbenvegnen/knowledge/vault/07 Sessions/UQ_DPD/assets/active_learning_emb_3p4_final_gate"
)
DEFAULT_FORCE_GRID_DATA = Path("emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat")
FULL_GATE_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_final_gate.v1"
EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_34um_request.v1"

YT_MIN = 1e5
YT_MAX = 1e9
KB_MIN = 400.0
KB_MAX = 70000.0

FULL_ROUNDS = 3
PER_ROUND = 30
CANARY_FORCE_COUNT = 3
TOTAL_FULL = FULL_ROUNDS * PER_ROUND
FULL_SEED = 2026
CANARY_YT = 3.0e7
CANARY_KB = 4.5e3


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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meso_uq.active_learning import Candidate  # noqa: E402
from meso_uq.active_learning.contracts import candidate_hash  # noqa: E402
from meso_uq.active_learning.dpd_sampling_gate import build_and_render_active_learning_dpd_sampling_gate  # noqa: E402


def _load_force_grid(path: Path) -> tuple[float, ...]:
    force_points: tuple[float, ...] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        tokens = value.split()
        if len(tokens) < 10:
            raise ValueError(f"Unexpected row width in {path!r}: {len(tokens)} columns.")
        n_after_header = len(tokens) - 8
        if n_after_header % 2 != 0:
            raise ValueError(f"Unexpected curve column layout in {path!r}: {len(tokens)} columns.")
        m = n_after_header // 2
        row_points = tuple(float(item) for item in tokens[8 + m : 8 + 2 * m])
        if not row_points:
            raise ValueError(f"No force points parsed from {path!r}.")
        if force_points is None:
            force_points = row_points
        elif force_points != row_points:
            raise ValueError(f"Force-grid columns differ between rows in {path!r}.")
    if force_points is None:
        raise ValueError(f"No force points found in {path!r}.")
    return force_points


def _lhs_1d(count: int, lower: float, upper: float, seed: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    bins = list(range(count))
    rng.shuffle(bins)
    width = upper - lower
    return tuple(lower + (index + rng.random()) * width / count for index in bins)


def _canonical_canary_force_points(force_grid: tuple[float, ...], count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("canary force count must be positive")
    if len(force_grid) <= count:
        return tuple(force_grid)
    if count == 3:
        return (force_grid[0], force_grid[(len(force_grid) - 1) // 2], force_grid[-1])
    return tuple(force_grid[round((len(force_grid) - 1) * index / (count - 1))] for index in range(count))


def _build_full_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    ys = _lhs_1d(TOTAL_FULL, YT_MIN, YT_MAX, FULL_SEED)
    ks = _lhs_1d(TOTAL_FULL, KB_MIN, KB_MAX, FULL_SEED + 1)
    force_grid_payload = [float(item) for item in force_grid]
    candidates: list[Candidate] = []

    for index in range(TOTAL_FULL):
        batch_round = index // PER_ROUND + 1
        step = index % PER_ROUND + 1
        candidates.append(
            Candidate(
                candidate_id=f"{run_id}-full-r{batch_round:02d}-c{step:03d}",
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "Yt": float(f"{ys[index]:.6g}"),
                    "kb": float(f"{ks[index]:.6g}"),
                    "force_grid": list(force_grid_payload),
                },
                metadata={
                    "campaign": "full",
                    "round": batch_round,
                    "position": step,
                    "lhs_rounds": FULL_ROUNDS,
                    "lhs_per_round": PER_ROUND,
                },
            )
        )
    return tuple(candidates)


def _build_canary_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    force_grid_payload = [float(item) for item in force_grid]
    return (
        Candidate(
            candidate_id=f"{run_id}-canary-001",
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "Yt": CANARY_YT,
                "kb": CANARY_KB,
                "canary": True,
                "force_grid": list(force_grid_payload),
            },
            metadata={"campaign": "canary", "purpose": "smoke"},
        ),
    )


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


def _submission_command(
    *,
    mode: str,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    campaign_root: Path,
    run_id_prefix: str,
    concurrent_jobs: int,
    retry_limit: int,
) -> dict[str, Any]:
    sbatch_script = (
        _script_root()
        / "scripts"
        / "platforms"
        / "karolina"
        / "sbatch"
        / "emb_34um_active_learning_array.sbatch"
    )
    candidate_count = TOTAL_FULL if mode == "full" else 1
    array_spec = f"0-{candidate_count - 1}"
    if mode == "full":
        array_spec = f"{array_spec}%{concurrent_jobs}"
    return {
        "script": str(sbatch_script),
        "command": (
            "sbatch --parsable "
            f"--array={array_spec} "
            "--export="
            f"TIMESTAMP={shlex.quote(timestamp)},"
            f"SCRATCH_ROOT={shlex.quote(str(scratch_root))},"
            f"VAULT_ROOT={shlex.quote(str(vault_root))},"
            f"RUN_ID_PREFIX={shlex.quote(run_id_prefix)},"
            f"CAMPAIGN_ROOT={shlex.quote(str(campaign_root))},"
            f"MODE={shlex.quote(mode)},"
            "EXECUTION_MODE=execute,"
            f"CONCURRENT_JOBS={concurrent_jobs},"
            f"RETRY_LIMIT={retry_limit} "
            f"{shlex.quote(str(sbatch_script))}"
        ),
        "mode": mode,
        "array": array_spec,
        "execution_mode": "execute",
    }


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
    if schema_version != EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has unexpected request_payload.schema_version={schema_version!r}; "
            f"expected {EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION!r}."
        )
    return EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION


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
        provenance_tags={"source_issue": "MESOUQ-AL-34UM"},
    )

    lineage = _lineage_payload(candidates)
    rendered_candidate_manifests = tuple(
        manifest_path
        for manifest_path in result.render_result.rendered_manifest_paths
        if manifest_path.name == "dpd_sampling_candidate_manifest.json"
    )
    if len(rendered_candidate_manifests) < len(candidates):
        raise ValueError(
            f"Expected at least {len(candidates)} rendered candidate manifests, "
            f"got {len(rendered_candidate_manifests)}."
        )

    payload = {
        "batch_root": str(batch_root),
        "batch_id": batch_id,
        "run_id": run_id,
        "manifest_path": str(result.manifest_path),
        "report_path": str(result.report_path),
        "candidate_count": len(candidates),
        "expected_request_payload_schema_version": EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION,
        "request_payload_schema_versions": [
            _ensure_request_payload_schema_version(manifest_path)
            for manifest_path in rendered_candidate_manifests[: len(candidates)]
        ],
        "rendered_candidate_manifests": [str(item) for item in rendered_candidate_manifests[: len(candidates)]],
        "scheduler_boundary": result.manifest["scheduler_boundary"],
        "candidate_lineage": lineage["candidate_lineage"],
        "lineage_fingerprint": lineage["lineage_fingerprint"],
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
        "status": result.manifest["status"],
        "pass": result.manifest["pass"],
        "dpd_payload_summary": {
            k: v for k, v in result.manifest.items() if k in {"candidate_records", "expected_hdf5_refs"}
        },
    }
    payload["submission_expected"]["submission_commands"] = (
        result.render_result.validation_report.submission.get("submission_commands", [])
    )
    payload["submission_expected"]["candidate_count_positive"] = result.report["criteria"]["candidate_count_positive"]
    payload["submission_expected"]["single_experiment"] = result.report["criteria"]["single_experiment"]
    payload["submission_expected"]["single_family"] = result.report["criteria"]["single_family"]
    payload["submission_expected"]["no_submission_commands"] = result.report["criteria"][
        "no_submission_commands"
    ]

    if any(item != EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION for item in payload["request_payload_schema_versions"]):
        raise ValueError(
            f"Rendered EMB request payload schema mismatch in batch {batch_id}: "
            f"{sorted(set(payload['request_payload_schema_versions']))}"
        )

    # Keep one compact sidecar with the important values at batch level.
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "emb_34um_batch_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def prepare_emb_34um_final_gate(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    canary_force_count: int,
    skip_vault_copy: bool,
) -> dict[str, Any]:
    if canary_force_count <= 0:
        raise ValueError("canary_force_count must be positive.")
    if canary_force_count != CANARY_FORCE_COUNT:
        raise ValueError(
            f"canary_force_count is fixed at {CANARY_FORCE_COUNT} for the EMB 3.4um final gate."
        )
    if concurrent_jobs < 1:
        raise ValueError("concurrent_jobs must be positive.")
    if retry_limit < 0:
        raise ValueError("retry_limit must be zero or positive.")

    campaign_root = scratch_root / timestamp
    force_grid = _load_force_grid(force_grid_path)
    if len(force_grid) < canary_force_count:
        raise ValueError("Force grid does not contain enough points for canary sweep.")

    canary_forces = _canonical_canary_force_points(force_grid, canary_force_count)
    full_force_points = force_grid

    full_candidates = _build_full_candidates(run_id=run_id_prefix, force_grid=full_force_points)
    canary_candidates = _build_canary_candidates(run_id=run_id_prefix, force_grid=canary_forces)

    if len(full_candidates) != TOTAL_FULL:
        raise RuntimeError("Unexpected full-gate candidate count")

    full_root = campaign_root / "full_gate"
    canary_root = campaign_root / "canary"

    full_payload = _render_batch(
        candidates=full_candidates,
        batch_root=full_root,
        run_id=f"{run_id_prefix}-full",
        batch_id="full-3x30",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )
    canary_payload = _render_batch(
        candidates=canary_candidates,
        batch_root=canary_root,
        run_id=f"{run_id_prefix}-canary",
        batch_id="canary-001",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )

    manifest_path = campaign_root / "emb_34um_final_gate_campaign_manifest.json"
    campaign_root.mkdir(parents=True, exist_ok=True)

    acceptance = {
        "full_count_is_90": len(full_candidates) == TOTAL_FULL,
        "canary_count_is_1": len(canary_candidates) == 1,
        "canary_points_is_3": len(canary_forces) == canary_force_count == CANARY_FORCE_COUNT,
        "full_render_only": full_payload["submission_expected"]["submission_commands"] == [],
        "canary_render_only": canary_payload["submission_expected"]["submission_commands"] == [],
        "full_submitted": full_payload["submission_expected"]["submitted"] is False,
        "canary_submitted": canary_payload["submission_expected"]["submitted"] is False,
    }

    manifest = {
        "schema_version": FULL_GATE_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id_prefix": run_id_prefix,
        "campaign_root": str(campaign_root),
        "scratch_root": str(scratch_root),
        "vault_root": str(vault_root),
        "vault_root_timestamp": str(vault_root / timestamp),
        "platform": "karolina",
        "expected_resources": {
            "platform": "karolina",
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
            "gpu_count": 1,
        },
        "canary": {
            "force_point_count": len(canary_forces),
            "force_points": list(canary_forces),
        },
        "full_force_points": len(full_force_points),
        "full_gate": full_payload,
        "canary_gate": canary_payload,
        "acceptance": {
            "pass": all(acceptance.values()),
            "criteria": acceptance,
        },
        "commands": {
            "full": _submission_command(
                mode="full",
                timestamp=timestamp,
                scratch_root=scratch_root,
                vault_root=vault_root,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            ),
            "canary": _submission_command(
                mode="canary",
                timestamp=timestamp,
                scratch_root=scratch_root,
                vault_root=vault_root,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            ),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not skip_vault_copy:
        target = vault_root / timestamp
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(campaign_root, target)

    return {
        "manifest_path": manifest_path,
        "campaign_root": campaign_root,
        "vault_root_timestamp": vault_root / timestamp,
        "manifest": manifest,
        "full_gate": full_payload,
        "canary_gate": canary_payload,
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
        help="Base path for vault evidence copy.",
    )
    parser.add_argument(
        "--force-grid",
        default=str(DEFAULT_FORCE_GRID_DATA),
        help="3.4um force grid source file path (samples_all.dat).",
    )
    parser.add_argument("--walltime", default="00:30:00")
    parser.add_argument("--concurrent-jobs", type=int, default=30)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--run-id-prefix", default="emb-34um-final-gate")
    parser.add_argument("--canary-force-count", type=int, default=CANARY_FORCE_COUNT)
    parser.add_argument("--skip-vault-copy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = prepare_emb_34um_final_gate(
        timestamp=args.timestamp,
        scratch_root=Path(args.scratch_root),
        vault_root=Path(args.vault_root),
        force_grid_path=Path(args.force_grid),
        walltime=args.walltime,
        concurrent_jobs=args.concurrent_jobs,
        retry_limit=args.retry_limit,
        run_id_prefix=args.run_id_prefix,
        canary_force_count=args.canary_force_count,
        skip_vault_copy=args.skip_vault_copy,
    )
    print(f"manifest_path={result['manifest_path']}")
    print(
        f"full_gate_manifest={result['full_gate']['manifest_path']} "
        f"canary_gate_manifest={result['canary_gate']['manifest_path']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
