#!/usr/bin/env python3
"""Prepare EMB 3.4um final-gate AL manifests and submission commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SCRATCH_ROOT = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_final_gate"
)
DEFAULT_VAULT_ROOT = Path(
    "/home/it4i-bbenvegnen/knowledge/vault/07 Sessions/UQ_DPD/assets/active_learning_emb_3p4_final_gate"
)
DEFAULT_FORCE_GRID_DATA = Path("emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat")
FULL_GATE_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_final_gate.v1"
EXPECTED_REQUEST_PAYLOAD_SCHEMA_VERSION = "meso_uq.dpd_sampling.emb_34um_request.v1"

KA_MIN = 1e2
KA_MAX = 6e5
KB_MIN = 400.0
KB_MAX = 70000.0

FULL_ROUNDS = 3
PER_ROUND = 30
CANARY_FORCE_COUNT = 3
TOTAL_FULL = FULL_ROUNDS * PER_ROUND
LHS_GATE_NAME = "lhs"
VALIDATION_GATE_NAME = "validation"
BENCHMARK_GATE_NAME = "benchmark"
LHS_CANDIDATE_COUNT = 90
BENCHMARK_TARGET_COUNT = 3
BENCHMARK_LABELS = ("low", "center", "high")
VALIDATION_TARGET_COUNT = 30
CANARY_CANDIDATE_COUNT = 1
CANDIDATE_POOL_SIZE = 100
FULL_SEED = 2026
LHS_SEED = 3034
BENCHMARK_SEED = 4044
VALIDATION_SEED = 5055
CANARY_KB = 4.5e3
DNN_ENSEMBLE_TARGET_SIZE = 10
EXECUTION_MODE_RENDER_ONLY = "render-only"
LINEAR_TRACE_PROJECT = "Active Learning Engine"
LINEAR_TRACE_ISSUE = "MES-210"
LINEAR_TRACE_ENGINE = "Active Learning Engine"
PARAMETER_SPACE = "log10"


def _legacy_yt_to_ka(yt: float) -> float:
    """Compatibility helper for legacy Yt-based AL controls."""
    import yaml

    current = Path(__file__).resolve()
    defaults_path = None
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            defaults_path = parent / "emb" / "indentation" / "src" / "parameters-default.emb.yaml"
            break
    if defaults_path is None:
        raise RuntimeError("Unable to resolve repository root for legacy Yt->ka mapping.")
    if not defaults_path.exists():
        raise RuntimeError(f"Unable to load EMB defaults from {defaults_path!s} for legacy mapping.")
    with defaults_path.open("r", encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    if not isinstance(defaults, dict):
        raise RuntimeError(f"Unexpected EMB defaults payload at {defaults_path!s}.")

    required = ("ul", "kbol", "t0", "shell_th", "nu", "fscale")
    for key in required:
        if key not in defaults:
            raise RuntimeError(f"Missing EMB default key {key!r} needed for legacy Yt->ka mapping.")

    ul = float(defaults["ul"])
    kbol = float(defaults["kbol"])
    t0 = float(defaults["t0"])
    shell_th = float(defaults["shell_th"])
    nu = float(defaults["nu"])
    fscale = float(defaults["fscale"])
    if any(not math.isfinite(value) or math.isclose(value, 0.0) for value in (ul, kbol, t0, shell_th, fscale)):
        raise RuntimeError("Invalid EMB defaults for legacy Yt->ka mapping.")
    if nu == 1.0:
        raise RuntimeError("Invalid EMB defaults for legacy Yt->ka mapping.")

    ka_per_yt = fscale * shell_th / (2.0 * (1.0 - nu)) / ((kbol * t0) / ul**2)
    return float(yt) * ka_per_yt


CANARY_KA = _legacy_yt_to_ka(3.0e7)


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
from meso_uq.active_learning.emb_34um_final_gate_design import (  # noqa: E402
    EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
    EMB_34UM_FINAL_GATE_BATCH_SIZE,
    EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
    EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES,
    EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION,
    EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION,
    EMB_34UM_FINAL_GATE_SOURCE_INITIAL,
    build_emb_34um_final_gate_design_round,
    build_emb_34um_final_gate_validation_design,
)
from meso_uq.active_learning.emb_34um_dpd_adapter import EMB_34UM_RUNTIME_FINGERPRINT  # noqa: E402


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


def _lhs_1d(
    count: int,
    lower: float,
    upper: float,
    seed: int,
    *,
    log_scale: bool = False,
) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    bins = list(range(count))
    rng.shuffle(bins)
    if log_scale:
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("log-space bounds must be positive.")
        lower = math.log10(lower)
        upper = math.log10(upper)
    width = upper - lower
    return tuple(lower + (index + rng.random()) * width / count for index in bins)


def _scale_samples(
    *,
    samples: tuple[float, ...],
    log_scale: bool,
) -> tuple[float, ...]:
    if not log_scale:
        return samples
    scaled: list[float] = []
    for sample in samples:
        scaled.append(10.0 ** sample)
    return tuple(scaled)


def _lhs_2d(
    count: int,
    ka_lower: float,
    ka_upper: float,
    kb_lower: float,
    kb_upper: float,
    seed: int,
    *,
    log_scale: bool = False,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        zip(
            _scale_samples(
                samples=_lhs_1d(count, ka_lower, ka_upper, seed, log_scale=log_scale),
                log_scale=log_scale,
            ),
            _scale_samples(
                samples=_lhs_1d(count, kb_lower, kb_upper, seed + 1, log_scale=log_scale),
                log_scale=log_scale,
            ),
        )
    )


def _stiffness_points(
    *,
    ka_lower: float,
    ka_upper: float,
    kb_lower: float,
    kb_upper: float,
    log_scale: bool,
) -> tuple[tuple[str, float, float], ...]:
    if ka_lower <= 0.0 or ka_upper <= 0.0 or kb_lower <= 0.0 or kb_upper <= 0.0:
        raise ValueError("log-space bounds must be positive.")
    if log_scale:
        ka_mid = 10.0 ** ((math.log10(ka_lower) + math.log10(ka_upper)) / 2.0)
        kb_mid = 10.0 ** ((math.log10(kb_lower) + math.log10(kb_upper)) / 2.0)
    else:
        ka_mid = (ka_lower + ka_upper) / 2.0
        kb_mid = (kb_lower + kb_upper) / 2.0
    return (
        ("low", ka_lower, kb_lower),
        ("center", ka_mid, kb_mid),
        ("high", ka_upper, kb_upper),
    )


def _canonical_canary_force_points(force_grid: tuple[float, ...], count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("canary force count must be positive")
    if len(force_grid) <= count:
        return tuple(force_grid)
    if count == 3:
        return (force_grid[0], force_grid[(len(force_grid) - 1) // 2], force_grid[-1])
    return tuple(force_grid[round((len(force_grid) - 1) * index / (count - 1))] for index in range(count))


def _attach_force_grid(
    candidate: Candidate,
    *,
    force_grid: tuple[float, ...],
    campaign: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Candidate:
    force_grid_payload = [float(item) for item in force_grid]
    metadata = {
        **dict(candidate.metadata),
        "campaign": campaign,
        "parameter_space": PARAMETER_SPACE,
        "bounds": {
            "ka": [KA_MIN, KA_MAX],
            "kb": [KB_MIN, KB_MAX],
        },
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    runtime_fingerprint = {
        **dict(EMB_34UM_RUNTIME_FINGERPRINT),
        **dict(candidate.parameters.get("runtime_fingerprint", {})),
    }
    return Candidate(
        candidate_id=candidate.candidate_id,
        parameters={
            **dict(candidate.parameters),
            "force_grid": list(force_grid_payload),
            "runtime_fingerprint": runtime_fingerprint,
        },
        metadata=metadata,
    )


def _build_full_candidates(
    run_id: str,
    *,
    force_grid: tuple[float, ...],
) -> tuple[tuple[Candidate, ...], tuple[dict[str, Any], ...]]:
    candidates: list[Candidate] = []
    design_manifests: list[dict[str, Any]] = []

    for round_index in range(1, FULL_ROUNDS + 1):
        round_result = build_emb_34um_final_gate_design_round(
            run_id=run_id,
            round_index=round_index,
            seed=FULL_SEED + round_index - 1,
            existing_points=tuple(candidates),
            ensemble_disagreement=None,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
            batch_size=PER_ROUND,
            exploration_count=EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
            acquisition_count=EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
            candidate_prefix="full",
            use_log_space=PARAMETER_SPACE == "log10",
        )
        design_manifests.append(dict(round_result.manifest))
        for candidate in round_result.candidates:
            candidates.append(
                _attach_force_grid(
                    candidate,
                    force_grid=force_grid,
                    campaign="full",
                    extra_metadata={
                        "position": int(candidate.metadata["order"]),
                        "round_target": PER_ROUND,
                        "candidate_pool_size": CANDIDATE_POOL_SIZE,
                        "lhs_comparator_prefixes": list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES),
                    },
                )
            )

    return tuple(candidates), tuple(design_manifests)


def _build_canary_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    force_grid_payload = [float(item) for item in force_grid]
    return (
        Candidate(
            candidate_id=f"{run_id}-canary-001",
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": float(f"{CANARY_KA:.6g}"),
                "kb": CANARY_KB,
                "canary": True,
                "force_grid": list(force_grid_payload),
                "runtime_fingerprint": dict(EMB_34UM_RUNTIME_FINGERPRINT),
            },
            metadata={
                "campaign": "canary",
                "purpose": "smoke",
                "parameter_space": PARAMETER_SPACE,
                "bounds": {
                    "ka": [KA_MIN, KA_MAX],
                    "kb": [KB_MIN, KB_MAX],
                },
            },
        ),
    )


def _build_lhs_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    points = _lhs_2d(
        LHS_CANDIDATE_COUNT,
        KA_MIN,
        KA_MAX,
        KB_MIN,
        KB_MAX,
        LHS_SEED,
        log_scale=PARAMETER_SPACE == "log10",
    )
    force_grid_payload = [float(item) for item in force_grid]
    candidates: list[Candidate] = []
    for step, (ka, kb) in enumerate(points, start=1):
        candidates.append(
            Candidate(
                candidate_id=f"{run_id}-lhs-c{step:03d}",
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": float(f"{ka:.6g}"),
                    "kb": float(f"{kb:.6g}"),
                    "force_grid": list(force_grid_payload),
                    "runtime_fingerprint": dict(EMB_34UM_RUNTIME_FINGERPRINT),
                },
                metadata={
                    "campaign": "lhs",
                    "step": step,
                    "target_count": LHS_CANDIDATE_COUNT,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "parameter_space": PARAMETER_SPACE,
                    "bounds": {
                        "ka": [KA_MIN, KA_MAX],
                        "kb": [KB_MIN, KB_MAX],
                    },
                    "purpose": "comparator",
                },
            )
        )
    return tuple(candidates)


def _build_validation_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    validation_result = build_emb_34um_final_gate_validation_design(
        run_id=run_id,
        seed=VALIDATION_SEED,
        design_size=VALIDATION_TARGET_COUNT,
        candidate_prefix="validation",
        use_log_space=PARAMETER_SPACE == "log10",
    )
    candidates: list[Candidate] = []
    for step, candidate in enumerate(validation_result.candidates, start=1):
        candidates.append(
            _attach_force_grid(
                candidate,
                force_grid=force_grid,
                campaign="validation",
                extra_metadata={
                    "campaign": "validation",
                    "step": step,
                    "position": step,
                    "target_count": VALIDATION_TARGET_COUNT,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "purpose": "validation",
                },
            )
        )
    return tuple(candidates)


def _build_benchmark_candidates(run_id: str, *, force_grid: tuple[float, ...]) -> tuple[Candidate, ...]:
    points = _stiffness_points(
        ka_lower=KA_MIN,
        ka_upper=KA_MAX,
        kb_lower=KB_MIN,
        kb_upper=KB_MAX,
        log_scale=PARAMETER_SPACE == "log10",
    )
    force_grid_payload = [float(item) for item in force_grid]
    candidates: list[Candidate] = []
    for step, (label, ka, kb) in enumerate(points, start=1):
        candidates.append(
            Candidate(
                candidate_id=f"{run_id}-benchmark-{label}",
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "ka": float(f"{ka:.6g}"),
                    "kb": float(f"{kb:.6g}"),
                    "force_grid": list(force_grid_payload),
                    "runtime_fingerprint": dict(EMB_34UM_RUNTIME_FINGERPRINT),
                },
                metadata={
                    "campaign": "benchmark",
                    "label": label,
                    "step": step,
                    "position": step,
                    "purpose": "runtime_benchmark",
                    "parameter_space": PARAMETER_SPACE,
                    "bounds": {
                        "ka": [KA_MIN, KA_MAX],
                        "kb": [KB_MIN, KB_MAX],
                    },
                    "dnn_ensemble_target_size": DNN_ENSEMBLE_TARGET_SIZE,
                },
            )
        )
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


def _submission_command(
    *,
    mode: str,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    campaign_root: Path,
    run_id_prefix: str,
    walltime: str,
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
    if mode in {"full", LHS_GATE_NAME, VALIDATION_GATE_NAME, BENCHMARK_GATE_NAME}:
        candidate_count = {
            "full": TOTAL_FULL,
            LHS_GATE_NAME: LHS_CANDIDATE_COUNT,
            VALIDATION_GATE_NAME: VALIDATION_TARGET_COUNT,
            BENCHMARK_GATE_NAME: BENCHMARK_TARGET_COUNT,
        }[mode]
    else:
        candidate_count = CANARY_CANDIDATE_COUNT
    if candidate_count <= 0:
        raise ValueError(f"Unable to determine candidate_count for mode {mode!r}.")
    array_spec = f"0-{candidate_count - 1}"
    if mode == "full":
        array_spec = f"{array_spec}%{concurrent_jobs}"
    elif mode in {LHS_GATE_NAME, VALIDATION_GATE_NAME, BENCHMARK_GATE_NAME} and concurrent_jobs > 1:
        array_spec = f"{array_spec}%{concurrent_jobs}"
    return {
        "script": str(sbatch_script),
        "command": (
            "sbatch --parsable "
            f"--time={shlex.quote(walltime)} "
            f"--array={array_spec} "
            "--export=ALL,"
            f"TIMESTAMP={shlex.quote(timestamp)},"
            f"SCRATCH_ROOT={shlex.quote(str(scratch_root))},"
            f"VAULT_ROOT={shlex.quote(str(vault_root))},"
            f"RUN_ID_PREFIX={shlex.quote(run_id_prefix)},"
            f"CAMPAIGN_ROOT={shlex.quote(str(campaign_root))},"
            f"REPO_ROOT={shlex.quote(str(_script_root()))},"
            f"MODE={shlex.quote(mode)},"
            f"EXECUTION_MODE={shlex.quote(EXECUTION_MODE_RENDER_ONLY)},"
            f"CONCURRENT_JOBS={concurrent_jobs},"
            f"RETRY_LIMIT={retry_limit},"
            f"PYTHON_EXECUTABLE={shlex.quote(sys.executable)} "
            f"{shlex.quote(str(sbatch_script))}"
        ),
        "mode": mode,
        "array": array_spec,
        "execution_mode": EXECUTION_MODE_RENDER_ONLY,
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
            provenance_tags={"source_issue": "MESOUQ-AL-34UM"},
        )
    except ModuleNotFoundError as exc:
        if "matplotlib" not in str(exc):
            raise
        from meso_uq.dpd_sampling import boundary as _dpd_boundary

        original_render_validation_plot = _dpd_boundary._render_validation_plot

        def _fallback_render_validation_plot(request, report) -> tuple[Path, Path]:
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
                provenance_tags={"source_issue": "MESOUQ-AL-34UM"},
            )
        finally:
            _dpd_boundary._render_validation_plot = original_render_validation_plot

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
        "parameter_space": PARAMETER_SPACE,
        "bounds": {
            "ka": [KA_MIN, KA_MAX],
            "kb": [KB_MIN, KB_MAX],
        },
        "request_payload_schema_versions": [
            _ensure_request_payload_schema_version(manifest_path)
            for manifest_path in rendered_candidate_manifests[: len(candidates)]
        ],
        "rendered_candidate_manifests": [str(item) for item in rendered_candidate_manifests[: len(candidates)]],
        "expected_output_roots": [
            record["output_root"] for record in result.manifest.get("candidate_records", []) if isinstance(record, dict)
        ],
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
    lhs_candidate_count: int,
    validation_target_count: int,
    dnn_ensemble_target_size: int,
    candidate_pool_size: int,
    skip_vault_copy: bool,
) -> dict[str, Any]:
    if canary_force_count <= 0:
        raise ValueError("canary_force_count must be positive.")
    if canary_force_count != CANARY_FORCE_COUNT:
        raise ValueError(
            f"canary_force_count is fixed at {CANARY_FORCE_COUNT} for the EMB 3.4um final gate."
        )
    if lhs_candidate_count <= 0:
        raise ValueError("lhs_candidate_count must be positive.")
    if validation_target_count <= 0:
        raise ValueError("validation_target_count must be positive.")
    if dnn_ensemble_target_size <= 0:
        raise ValueError("dnn_ensemble_target_size must be positive.")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive.")
    if lhs_candidate_count != LHS_CANDIDATE_COUNT:
        raise ValueError(
            f"lhs_candidate_count is fixed at {LHS_CANDIDATE_COUNT} for the EMB 3.4um final gate."
        )
    if validation_target_count != VALIDATION_TARGET_COUNT:
        raise ValueError(
            f"validation_target_count is fixed at {VALIDATION_TARGET_COUNT} for the EMB 3.4um final gate."
        )
    if dnn_ensemble_target_size != DNN_ENSEMBLE_TARGET_SIZE:
        raise ValueError(
            f"dnn_ensemble_target_size is fixed at {DNN_ENSEMBLE_TARGET_SIZE} for the EMB 3.4um final gate."
        )
    if candidate_pool_size != CANDIDATE_POOL_SIZE:
        raise ValueError(
            f"candidate_pool_size is fixed at {CANDIDATE_POOL_SIZE} for the EMB 3.4um final gate."
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

    full_candidates, full_design_manifests = _build_full_candidates(run_id=run_id_prefix, force_grid=full_force_points)
    canary_candidates = _build_canary_candidates(run_id=run_id_prefix, force_grid=canary_forces)
    lhs_candidates = _build_lhs_candidates(run_id=run_id_prefix, force_grid=full_force_points)
    benchmark_candidates = _build_benchmark_candidates(run_id=run_id_prefix, force_grid=full_force_points)
    validation_candidates = _build_validation_candidates(run_id=run_id_prefix, force_grid=full_force_points)

    if len(full_candidates) != TOTAL_FULL:
        raise RuntimeError("Unexpected full-gate candidate count")
    if len(lhs_candidates) != LHS_CANDIDATE_COUNT:
        raise RuntimeError("Unexpected lhs candidate count.")
    if len(benchmark_candidates) != BENCHMARK_TARGET_COUNT:
        raise RuntimeError("Unexpected benchmark candidate count.")
    if len(validation_candidates) != VALIDATION_TARGET_COUNT:
        raise RuntimeError("Unexpected validation candidate count.")

    full_root = campaign_root / "full_gate"
    canary_root = campaign_root / "canary"
    lhs_root = campaign_root / "lhs_gate"
    benchmark_root = campaign_root / "benchmark"
    validation_root = campaign_root / "validation"

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
    lhs_payload = _render_batch(
        candidates=lhs_candidates,
        batch_root=lhs_root,
        run_id=f"{run_id_prefix}-lhs",
        batch_id="lhs-90",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )
    benchmark_payload = _render_batch(
        candidates=benchmark_candidates,
        batch_root=benchmark_root,
        run_id=f"{run_id_prefix}-benchmark",
        batch_id="benchmark-3",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )
    validation_payload = _render_batch(
        candidates=validation_candidates,
        batch_root=validation_root,
        run_id=f"{run_id_prefix}-validation",
        batch_id="validation-30",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )

    manifest_path = campaign_root / "emb_34um_final_gate_campaign_manifest.json"
    campaign_root.mkdir(parents=True, exist_ok=True)

    acceptance = {
        "full_count_is_90": len(full_candidates) == TOTAL_FULL,
        "canary_count_is_1": len(canary_candidates) == CANARY_CANDIDATE_COUNT,
        "lhs_count_is_90": len(lhs_candidates) == LHS_CANDIDATE_COUNT,
        "benchmark_count_is_3": len(benchmark_candidates) == BENCHMARK_TARGET_COUNT,
        "validation_count_is_30": len(validation_candidates) == VALIDATION_TARGET_COUNT,
        "canary_points_is_3": len(canary_forces) == canary_force_count == CANARY_FORCE_COUNT,
        "full_render_only": full_payload["submission_expected"]["submission_commands"] == [],
        "canary_render_only": canary_payload["submission_expected"]["submission_commands"] == [],
        "lhs_render_only": lhs_payload["submission_expected"]["submission_commands"] == [],
        "benchmark_render_only": benchmark_payload["submission_expected"]["submission_commands"] == [],
        "validation_render_only": validation_payload["submission_expected"]["submission_commands"] == [],
        "full_submitted": full_payload["submission_expected"]["submitted"] is False,
        "canary_submitted": canary_payload["submission_expected"]["submitted"] is False,
        "lhs_submitted": lhs_payload["submission_expected"]["submitted"] is False,
        "benchmark_submitted": benchmark_payload["submission_expected"]["submitted"] is False,
        "validation_submitted": validation_payload["submission_expected"]["submitted"] is False,
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
            "walltime": walltime,
            "gpu_count": 1,
            "python_executable": sys.executable,
        },
        "linear_traceability": {
            "project": LINEAR_TRACE_PROJECT,
            "issue": LINEAR_TRACE_ISSUE,
            "engine": LINEAR_TRACE_ENGINE,
        },
        "policy": {
            "candidate_pool_size": candidate_pool_size,
            "adaptive_selection": {
                "engine": "dnn_ensemble_disagreement_diversity",
                "round_1_source": EMB_34UM_FINAL_GATE_SOURCE_INITIAL,
                "later_round_exploration_source": EMB_34UM_FINAL_GATE_SOURCE_EXPLORATION,
                "later_round_acquisition_source": EMB_34UM_FINAL_GATE_SOURCE_ACQUISITION,
                "exploration_count": EMB_34UM_FINAL_GATE_EXPLORATION_COUNT,
                "acquisition_count": EMB_34UM_FINAL_GATE_ACQUISITION_COUNT,
                "batch_size": EMB_34UM_FINAL_GATE_BATCH_SIZE,
                "lhs_comparator_prefixes": list(EMB_34UM_FINAL_GATE_LHS_COMPARATOR_PREFIXES),
                "posterior_aware": False,
            },
            "accepted_curves": {
                "rounds": FULL_ROUNDS,
                "per_round": PER_ROUND,
                "total": TOTAL_FULL,
            },
            "lhs_comparator_curves": LHS_CANDIDATE_COUNT,
            "validation_target_curves": VALIDATION_TARGET_COUNT,
            "benchmark_curves": BENCHMARK_TARGET_COUNT,
            "dnn_ensemble_target_size": dnn_ensemble_target_size,
            "failure_policy": {
                "retry_limit": retry_limit,
                "retries_before_quarantine": 3,
                "replacement_mode": "quarantine_then_gate_shortfall",
            },
            "sampling": {
                "parameter_space": PARAMETER_SPACE,
                "bounds": {
                    "ka": [KA_MIN, KA_MAX],
                    "kb": [KB_MIN, KB_MAX],
                },
            },
            "seeds": {
                "full": FULL_SEED,
                "lhs": LHS_SEED,
                "benchmark": BENCHMARK_SEED,
                "validation": VALIDATION_SEED,
            },
        },
        "runtime_artifacts": {
            "full": [str(Path(root) / "emb_34um_runtime_status.json") for root in full_payload.get("expected_output_roots", [])],
            "canary": [str(Path(root) / "emb_34um_runtime_status.json") for root in canary_payload.get("expected_output_roots", [])],
            "lhs": [str(Path(root) / "emb_34um_runtime_status.json") for root in lhs_payload.get("expected_output_roots", [])],
            "benchmark": [str(Path(root) / "emb_34um_runtime_status.json") for root in benchmark_payload.get("expected_output_roots", [])],
            "validation": [str(Path(root) / "emb_34um_runtime_status.json") for root in validation_payload.get("expected_output_roots", [])],
        },
        "canary": {
            "force_point_count": len(canary_forces),
            "force_points": list(canary_forces),
        },
        "full_force_points": len(full_force_points),
        "full_design": {
            "round_manifests": list(full_design_manifests),
            "selection_note": (
                "Round 1 uses Sobol/maximin from no prior data; later rounds reserve 6 exploratory "
                "curves and 24 acquisition-driven curves using ensemble disagreement plus diversity."
            ),
        },
        "full_gate": full_payload,
        "canary_gate": canary_payload,
        "lhs_gate": lhs_payload,
        "benchmark_gate": benchmark_payload,
        "validation_gate": validation_payload,
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
                walltime=walltime,
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
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            ),
            "lhs": _submission_command(
                mode=LHS_GATE_NAME,
                timestamp=timestamp,
                scratch_root=scratch_root,
                vault_root=vault_root,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            ),
            "benchmark": _submission_command(
                mode=BENCHMARK_GATE_NAME,
                timestamp=timestamp,
                scratch_root=scratch_root,
                vault_root=vault_root,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
            ),
            "validation": _submission_command(
                mode=VALIDATION_GATE_NAME,
                timestamp=timestamp,
                scratch_root=scratch_root,
                vault_root=vault_root,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                walltime=walltime,
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
        "lhs_gate": lhs_payload,
        "benchmark_gate": benchmark_payload,
        "validation_gate": validation_payload,
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
    parser.add_argument("--candidate-pool-size", type=int, default=CANDIDATE_POOL_SIZE)
    parser.add_argument("--lhs-candidate-count", type=int, default=LHS_CANDIDATE_COUNT)
    parser.add_argument("--validation-target-count", type=int, default=VALIDATION_TARGET_COUNT)
    parser.add_argument("--dnn-target-size", type=int, default=DNN_ENSEMBLE_TARGET_SIZE)
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
        lhs_candidate_count=args.lhs_candidate_count,
        validation_target_count=args.validation_target_count,
        dnn_ensemble_target_size=args.dnn_target_size,
        candidate_pool_size=args.candidate_pool_size,
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
