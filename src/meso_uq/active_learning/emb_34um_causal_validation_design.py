from __future__ import annotations

"""Build deterministic render-only manifests for EMB 3.4um causal AL-vs-LHS validation."""

import hashlib
import json
import math
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, candidate_hash
from meso_uq.active_learning.emb_34um_dpd_adapter import (
    EMB_34UM_RUNTIME_FINGERPRINT,
    EMB_34UM_DPD_SCHEMA_VERSION,
)

EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_causal_validation.v1"
EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME = "emb_34um_causal_validation_manifest.json"
EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME = "emb_34um_causal_validation_command_inventory.json"
EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME = "emb_34um_causal_validation_manifest_coverage.png"
EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME = (
    "emb_34um_causal_validation_manifest_coverage.png.json"
)
EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME = "emb_34um_batch_summary.json"
EMB_34UM_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME = "emb_34um_causal_validation_selection_manifest.json"
EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME = (
    "emb_34um_causal_validation_selection_batch_summary.json"
)
EMB_34UM_CAUSAL_VALIDATION_FAMILY = "emb"
EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT = "indentation"
EMB_34UM_CAUSAL_VALIDATION_ACTIVE_VARIABLES = ("ka", "kb")
EMB_34UM_CAUSAL_VALIDATION_BOUNDS = {"ka": (1e2, 6e5), "kb": (400.0, 70000.0)}
EMB_34UM_CAUSAL_VALIDATION_PARAMETER_SPACE = "log10"
EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS = (1, 2, 3)
EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS = 5
EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS = 5
EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE = 100
EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE = 100
EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE = 100
EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE = "render-only"
EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_POOL_SIZE = 500
EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_ACQUISITION_COUNT = 80
EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_EXPLORATION_COUNT = 20
EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED = (
    EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE
    + EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS * EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE * 2
    + EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE
)

_GLOBAL_SEED_OFFSET = 10_000
_GLOBAL_MODULUS = 1_000_000_007
_GLOBAL_MULT_X = 37_156_633
_GLOBAL_MULT_Y = 65_498_713


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root for causal manifest builder.")


def _coerce_nonempty_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty text value.")
    return text


def _coerce_positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _coerce_seed_count(seed_count: int) -> int:
    seed_count = _coerce_positive_int(seed_count, label="seed_count")
    if seed_count > EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS:
        raise ValueError(
            f"seed_count must be in [1, {EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS}], got {seed_count}."
        )
    if seed_count < len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS):
        raise ValueError(
            "seed_count must include all primary seeds 1, 2, 3. Minimum is 3."
        )
    return seed_count


def _coerce_step_count(step_count: int) -> int:
    step_count = _coerce_positive_int(step_count, label="step_count")
    if step_count < EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS:
        raise ValueError(
            f"step_count must be at least {EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS}, got {step_count}."
        )
    return step_count


def _coerce_size(value: int, *, label: str, fixed: int | None = None) -> int:
    value = _coerce_positive_int(value, label=label)
    if fixed is not None and value != fixed:
        raise ValueError(f"{label} is fixed at {fixed}.")
    return value


def _coerce_force_grid(force_grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(item) for item in force_grid)
    if not values:
        raise ValueError("force_grid must contain at least one value.")
    for value in values:
        if not math.isfinite(value):
            raise ValueError("force_grid must contain finite values.")
    return values


def _force_grid_signature(force_grid: Sequence[float]) -> str:
    payload = tuple(float(item) for item in force_grid)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _index_to_unit_point(index: int) -> tuple[float, float]:
    if index < 0:
        raise ValueError("index must be non-negative.")
    return (
        ((index * _GLOBAL_MULT_X) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
        ((index * _GLOBAL_MULT_Y) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
    )


def _from_unit_to_log_space(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if lower <= 0 or upper <= 0 or lower >= upper:
        raise ValueError(f"Invalid log space bounds: {bounds!r}.")
    return 10.0 ** (math.log10(lower) + value * (math.log10(upper) - math.log10(lower)))


def _sample_points(*, start_index: int, count: int) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            _from_unit_to_log_space(_index_to_unit_point(start_index + offset)[0], EMB_34UM_CAUSAL_VALIDATION_BOUNDS["ka"]),
            _from_unit_to_log_space(_index_to_unit_point(start_index + offset)[1], EMB_34UM_CAUSAL_VALIDATION_BOUNDS["kb"]),
        )
        for offset in range(count)
    )


def _selection_seed(seed: int, method: str, step: int | None) -> int:
    step_value = 0 if step is None else step
    if method == "shared":
        method_offset = 11_000
    elif method == "al":
        method_offset = 22_000
    elif method == "lhs":
        method_offset = 33_000
    elif method == "validation":
        method_offset = 44_000
    else:
        raise ValueError(f"Unknown method {method!r}.")
    return _GLOBAL_SEED_OFFSET * seed + method_offset + step_value


@dataclass(frozen=True)
class _Batch:
    method: str
    seed: int
    step: int | None
    selection_seed: int
    candidate_count: int
    candidate_records: tuple[dict[str, Any], ...]
    selection_payload: Mapping[str, Any] = field(default_factory=dict)
    output_roots: tuple[str, ...] = field(default_factory=tuple)
    vault_output_roots: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "step": self.step,
            "selection_seed": self.selection_seed,
            "candidate_count": int(self.candidate_count),
            "candidate_records": list(self.candidate_records),
            **dict(self.selection_payload),
            "output_roots": list(self.output_roots),
            "vault_output_roots": list(self.vault_output_roots),
        }


@dataclass(frozen=True)
class _SeedPayload:
    seed: int
    shared_initial: Mapping[str, Any]
    al_steps: tuple[Mapping[str, Any], ...]
    lhs_steps: tuple[Mapping[str, Any], ...]
    validation: Mapping[str, Any]
    replacement_queues: Mapping[str, Any]
    candidate_reserve: Mapping[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        al_expected_scratch = [
            root
            for batch in self.al_steps
            for root in batch.get("output_roots", [])
        ]
        al_expected_vault = [
            root
            for batch in self.al_steps
            for root in batch.get("vault_output_roots", [])
        ]
        return {
            "seed": self.seed,
            "required_seed": self.seed in EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS,
            "shared_initial": dict(self.shared_initial),
            "al_steps": [dict(item) for item in self.al_steps],
            "lhs_steps": [dict(item) for item in self.lhs_steps],
            "validation": dict(self.validation),
            "replacement_queues": dict(self.replacement_queues),
            "candidate_reserve": dict(self.candidate_reserve),
            "expected_output_roots": {
                "scratch": [record["output_root"] for record in self.shared_initial["candidate_records"]]
                + al_expected_scratch
                + [record["output_root"] for batch in self.lhs_steps for record in batch["candidate_records"]]
                + [record["output_root"] for record in self.validation["candidate_records"]],
                "vault": [record["vault_output_root"] for record in self.shared_initial["candidate_records"]]
                + al_expected_vault
                + [record["vault_output_root"] for batch in self.lhs_steps for record in batch["candidate_records"]]
                + [record["vault_output_root"] for record in self.validation["candidate_records"]],
            },
        }


def _build_records_for_batch(
    *,
    seed: int,
    method: str,
    step: int | None,
    start_index: int,
    count: int,
    run_id_prefix: str,
    campaign_root: Path,
    vault_root_timestamp: Path,
    force_grid_signature: str,
    force_grid_values: tuple[float, ...] | None = None,
    candidate_count: int | None = None,
    selection_payload: Mapping[str, Any] | None = None,
) -> _Batch:
    selection_seed_value = _selection_seed(seed, method=method, step=step)
    resolved_candidate_count = _coerce_size(count if candidate_count is None else candidate_count, label="count")
    points = _sample_points(start_index=start_index, count=resolved_candidate_count)

    candidate_records: list[dict[str, Any]] = []
    output_roots: list[str] = []
    vault_output_roots: list[str] = []

    if step is None:
        stage_tag = "shared_initial" if method == "shared" else f"{method}"
    else:
        stage_tag = f"{method}-step-{step:02d}"
    candidate_metadata_method = stage_tag

    for order, (ka, kb) in enumerate(points, start=1):
        if step is None:
            candidate_id = f"{run_id_prefix}-seed{seed:03d}-{method}-c{order:03d}"
            local_step = 0
        else:
            candidate_id = f"{run_id_prefix}-seed{seed:03d}-{method}-step{step:02d}-c{order:03d}"
            local_step = step

        output_root = campaign_root / f"seed-{seed:03d}" / stage_tag / "emb" / candidate_id
        vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)
        candidate = Candidate(
            candidate_id=candidate_id,
            parameters={
                "family": EMB_34UM_CAUSAL_VALIDATION_FAMILY,
                "experiment": EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT,
                "ka": float(ka),
                "kb": float(kb),
                **({"force_grid": list(force_grid_values)} if force_grid_values is not None else {}),
            },
            metadata={
                "seed": seed,
                "method": method,
                "step": local_step,
                "selection_seed": selection_seed_value,
                "fresh_only": True,
                "selection_mode": "causal_fresh_only",
                "force_grid_signature": force_grid_signature,
                "runtime_fingerprint": dict(EMB_34UM_RUNTIME_FINGERPRINT),
                "causal_validation_mode": True,
                "request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
            },
        )
        record = {
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash(candidate),
            "family": EMB_34UM_CAUSAL_VALIDATION_FAMILY,
            "experiment": EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT,
            "ka": float(ka),
            "kb": float(kb),
            "force_grid": list(force_grid_values) if force_grid_values is not None else None,
            "selection_seed": selection_seed_value,
            "selection_mode": "causal_fresh_only",
            "step": local_step,
            "method": candidate_metadata_method,
            "seed": seed,
            "output_root": str(output_root),
            "vault_output_root": str(vault_output_root),
        }
        candidate_records.append(record)
        output_roots.append(str(output_root))
        vault_output_roots.append(str(vault_output_root))

    return _Batch(
        method=method,
        seed=seed,
        step=step,
        selection_seed=selection_seed_value,
        candidate_count=resolved_candidate_count,
        candidate_records=tuple(candidate_records),
        selection_payload=selection_payload or {},
        output_roots=tuple(output_roots),
        vault_output_roots=tuple(vault_output_roots),
    )


def _build_adaptive_al_batch(
    *,
    seed: int,
    step: int,
    step_size: int,
    campaign_root: Path,
    vault_root_timestamp: Path,
    required_cumulative_precedents: list[str],
    run_id_prefix: str,
) -> _Batch:
    step_str = f"{step:02d}"
    stage_name = f"al-step-{step_str}"
    stage_root = campaign_root / f"seed-{seed:03d}" / stage_name
    selection_manifest_path = stage_root / EMB_34UM_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME
    selection_batch_summary_path = stage_root / EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME
    sbatch_summary_path = stage_root / EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
    payload = {
        "selection_required_count": step_size,
        "candidate_pool_size": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_POOL_SIZE,
        "acquisition_count": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_ACQUISITION_COUNT,
        "exploration_count": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_EXPLORATION_COUNT,
        "selection_prerequisites": required_cumulative_precedents,
        "selection_seed": _selection_seed(seed, method="al", step=step),
        "selection_manifest_path": str(selection_manifest_path),
        "selection_batch_summary_path": str(selection_batch_summary_path),
        "expected_batch_summary_path": str(sbatch_summary_path),
        "select_render_action": f"al-step-{step_str}-select-render",
        "selection_status": "placeholder_runtime_selection_required",
        "output_root": str(stage_root),
    }
    return _Batch(
        method="al",
        seed=seed,
        step=step,
        selection_seed=_selection_seed(seed, method="al", step=step),
        candidate_count=step_size,
        candidate_records=(),
        selection_payload=payload,
        output_roots=(str(stage_root),),
        vault_output_roots=(str(vault_root_timestamp / stage_root.relative_to(campaign_root)),),
    )


def _build_seed_payload(
    *,
    seed: int,
    run_id_prefix: str,
    campaign_root: Path,
    vault_root_timestamp: Path,
    step_count: int,
    step_size: int,
    shared_size: int,
    validation_size: int,
    force_grid_signature: str,
    force_grid_values: tuple[float, ...],
) -> _SeedPayload:
    per_seed_span = shared_size + (2 * step_count * step_size) + validation_size
    seed_base = (seed - 1) * per_seed_span
    shared = _build_records_for_batch(
        seed=seed,
        method="shared",
        step=None,
        start_index=seed_base,
        count=shared_size,
        run_id_prefix=run_id_prefix,
        campaign_root=campaign_root,
        vault_root_timestamp=vault_root_timestamp,
        force_grid_signature=force_grid_signature,
        force_grid_values=force_grid_values,
    ).as_manifest()

    al_steps: list[Mapping[str, Any]] = []
    shared_summary = (
        campaign_root / f"seed-{seed:03d}" / "shared_initial" / EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME
    )
    for step in range(1, step_count + 1):
        prerequisites = [str(shared_summary)]
        if step > 1:
            for previous in range(1, step):
                previous_root = campaign_root / f"seed-{seed:03d}" / f"al-step-{previous:02d}"
                prerequisites.append(
                    str(previous_root / EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME)
                )
                prerequisites.append(
                    str(previous_root / EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME)
                )
        al_steps.append(
            _build_adaptive_al_batch(
                seed=seed,
                step=step,
                step_size=step_size,
                campaign_root=campaign_root,
                vault_root_timestamp=vault_root_timestamp,
                required_cumulative_precedents=prerequisites,
                run_id_prefix=run_id_prefix,
            ).as_manifest()
        )

    lhs_steps: list[Mapping[str, Any]] = []
    lhs_base = seed_base + shared_size + step_count * step_size
    for step in range(1, step_count + 1):
        start_index = lhs_base + (step - 1) * step_size
        lhs_steps.append(
            _build_records_for_batch(
                seed=seed,
                method="lhs",
                step=step,
                start_index=start_index,
                count=step_size,
                run_id_prefix=run_id_prefix,
                campaign_root=campaign_root,
                vault_root_timestamp=vault_root_timestamp,
                force_grid_signature=force_grid_signature,
                force_grid_values=force_grid_values,
            ).as_manifest()
        )

    validation_start = seed_base + shared_size + step_count * step_size * 2
    validation = _build_records_for_batch(
        seed=seed,
        method="validation",
        step=None,
        start_index=validation_start,
        count=validation_size,
        run_id_prefix=run_id_prefix,
        campaign_root=campaign_root,
        vault_root_timestamp=vault_root_timestamp,
        force_grid_signature=force_grid_signature,
        force_grid_values=force_grid_values,
    ).as_manifest()

    replacement_queues = {
        "al": {
            "enabled": True,
            "mode": "metadata_only",
            "max_replacements": 0,
            "note": (
                "No replacement execution is performed in render-only mode; the queue records are metadata only."
            ),
        },
        "lhs": {
            "enabled": True,
            "mode": "metadata_only",
            "max_replacements": 0,
            "note": (
                "No replacement execution is performed in render-only mode; the queue records are metadata only."
            ),
        },
    }
    candidate_reserve = {
        "al": {"required": 0, "status": "empty", "metadata_only": True},
        "lhs": {"required": 0, "status": "empty", "metadata_only": True},
    }
    return _SeedPayload(
        seed=seed,
        shared_initial=shared,
        al_steps=tuple(al_steps),
        lhs_steps=tuple(lhs_steps),
        validation=validation,
        replacement_queues=replacement_queues,
        candidate_reserve=candidate_reserve,
    )


def _build_command_payload(
    *,
    mode: str,
    seed: int,
    step: int,
    candidate_count: int,
    campaign_root: Path,
    run_id_prefix: str,
    timestamp: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    scratch_root: Path,
    vault_root_timestamp: Path,
) -> dict[str, Any]:
    sbatch_script = (
        _script_root()
        / "scripts"
        / "platforms"
        / "karolina"
        / "sbatch"
        / "emb_34um_active_learning_array.sbatch"
    )
    array = f"0-{candidate_count - 1}" if candidate_count > 0 else "0-0"
    array = f"{array}%{concurrent_jobs}"

    batch_root = (
        campaign_root
        / f"seed-{seed:03d}"
        / f"{mode}"
    )
    stage_path = str(batch_root)

    return {
        "mode": mode,
        "seed": seed,
        "step": step,
        "script": str(sbatch_script),
        "campaign_root": str(campaign_root),
        "batch_root": stage_path,
        "candidate_count": candidate_count,
        "selection_mode": "causal_fresh_only",
        "execution_mode": EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE,
        "array": array,
        "command": (
            "sbatch --parsable "
            f"--time={shlex.quote(walltime)} "
            f"--array={shlex.quote(array)} "
            "--export="
            f"TIMESTAMP={shlex.quote(timestamp)},"
            f"SCRATCH_ROOT={shlex.quote(str(scratch_root))},"
            f"VAULT_ROOT={shlex.quote(str(vault_root_timestamp))},"
            f"RUN_ID_PREFIX={shlex.quote(run_id_prefix)},"
            f"CAMPAIGN_ROOT={shlex.quote(str(campaign_root))},"
            f"REPO_ROOT={shlex.quote(str(_script_root()))},"
            f"MODE={shlex.quote(mode)},"
            f"BATCH_DIR_OVERRIDE={shlex.quote(stage_path)},"
            f"SEED={seed},"
            f"STEP={step},"
            f"EXECUTION_MODE={shlex.quote(EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE)},"
            f"CONCURRENT_JOBS={concurrent_jobs},"
            f"RETRY_LIMIT={retry_limit},"
            f"PYTHON_EXECUTABLE={shlex.quote(sys.executable)} "
            f"{shlex.quote(str(sbatch_script))}"
        ),
    }


def _seed_batches_for_command_inventory(
    *,
    seed_payload: Mapping[str, Any],
    campaign_root: Path,
    run_id_prefix: str,
    timestamp: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    scratch_root: Path,
    vault_root_timestamp: Path,
) -> tuple[dict[str, Any], ...]:
    seed = int(seed_payload["seed"])
    shared = seed_payload["shared_initial"]
    validation = seed_payload["validation"]

    entries: list[dict[str, Any]] = []
    entries.append(
        _build_command_payload(
            mode="shared_initial",
            seed=seed,
            step=0,
            candidate_count=len(shared["candidate_records"]),
            campaign_root=campaign_root,
            run_id_prefix=run_id_prefix,
            timestamp=timestamp,
            walltime=walltime,
            concurrent_jobs=concurrent_jobs,
            retry_limit=retry_limit,
            scratch_root=scratch_root,
            vault_root_timestamp=vault_root_timestamp,
        )
    )

    for step in range(1, len(seed_payload["al_steps"]) + 1):
        al_payload = seed_payload["al_steps"][step - 1]
        entries.append(
            _build_command_payload(
                mode=f"al-step-{step:02d}",
                seed=seed,
                step=step,
                candidate_count=int(al_payload["candidate_count"]),
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                timestamp=timestamp,
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
                scratch_root=scratch_root,
                vault_root_timestamp=vault_root_timestamp,
            )
        )
        lhs_payload = seed_payload["lhs_steps"][step - 1]
        entries.append(
            _build_command_payload(
                mode=f"lhs-step-{step:02d}",
                seed=seed,
                step=step,
                candidate_count=len(lhs_payload["candidate_records"]),
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                timestamp=timestamp,
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
                scratch_root=scratch_root,
                vault_root_timestamp=vault_root_timestamp,
            )
        )
    entries.append(
        _build_command_payload(
            mode="validation",
            seed=seed,
            step=0,
            candidate_count=len(validation["candidate_records"]),
            campaign_root=campaign_root,
            run_id_prefix=run_id_prefix,
            timestamp=timestamp,
            walltime=walltime,
            concurrent_jobs=concurrent_jobs,
            retry_limit=retry_limit,
            scratch_root=scratch_root,
            vault_root_timestamp=vault_root_timestamp,
        )
    )
    return tuple(entries)


def build_emb_34um_causal_validation_campaign_manifest(
    *,
    timestamp: str,
    run_id_prefix: str,
    campaign_root: Path,
    scratch_root: Path,
    vault_root_timestamp: Path,
    force_grid: Sequence[float],
    seed_count: int = len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS),
    step_count: int = EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS,
    step_size: int = EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE,
    shared_size: int = EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE,
    validation_size: int = EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
    walltime: str = "00:30:00",
    concurrent_jobs: int = 30,
    retry_limit: int = 3,
    include_coverage_plot_requirements: bool = True,
) -> dict[str, Any]:
    timestamp = _coerce_nonempty_text(timestamp, label="timestamp")
    run_id_prefix = _coerce_nonempty_text(run_id_prefix, label="run_id_prefix")
    campaign_root = Path(campaign_root)
    scratch_root = Path(scratch_root)
    vault_root_timestamp = Path(vault_root_timestamp)
    walltime = _coerce_nonempty_text(walltime, label="walltime")

    if " " in walltime:
        raise ValueError("walltime must be in HH:MM:SS format.")
    if not walltime[0].isdigit():
        raise ValueError("walltime must be in HH:MM:SS format.")
    if walltime.count(":") != 2:
        raise ValueError("walltime must be in HH:MM:SS format.")

    concurrent_jobs = _coerce_positive_int(concurrent_jobs, label="concurrent_jobs")
    retry_limit = _coerce_nonnegative_int(retry_limit, label="retry_limit")
    seed_count = _coerce_seed_count(seed_count)
    step_count = _coerce_step_count(step_count)
    step_size = _coerce_size(step_size, label="step_size", fixed=EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE)
    shared_size = _coerce_size(shared_size, label="shared_size", fixed=EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE)
    validation_size = _coerce_size(
        validation_size,
        label="validation_size",
        fixed=EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE,
    )

    force_grid_values = _coerce_force_grid(force_grid)
    force_grid_sig = _force_grid_signature(force_grid_values)

    seeds_payloads: list[dict[str, Any]] = []
    for seed in range(1, seed_count + 1):
        payload = _build_seed_payload(
            seed=seed,
            run_id_prefix=run_id_prefix,
            campaign_root=campaign_root,
            vault_root_timestamp=vault_root_timestamp,
            step_count=step_count,
            step_size=step_size,
            shared_size=shared_size,
            validation_size=validation_size,
            force_grid_signature=force_grid_sig,
            force_grid_values=force_grid_values,
        ).as_manifest()
        seeds_payloads.append(payload)

    expected_scratch_roots: list[str] = []
    expected_vault_roots: list[str] = []
    for payload in seeds_payloads:
        expected_scratch_roots.extend(payload["expected_output_roots"]["scratch"])
        expected_vault_roots.extend(payload["expected_output_roots"]["vault"])
        for batch_name in ("shared_initial", "validation", "al_steps", "lhs_steps"):
            if batch_name in ("shared_initial", "validation"):
                batch = payload[batch_name]
                batch_records = batch["candidate_records"]
                for record in batch_records:
                    if record["family"] != EMB_34UM_CAUSAL_VALIDATION_FAMILY:
                        raise ValueError("All records must use the EMB family.")
                    if record["experiment"] != EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT:
                        raise ValueError("All records must use the indentation experiment.")
                    if not Path(record["output_root"]).is_relative_to(campaign_root):
                        raise ValueError("Expected output root must be under campaign root.")
                    if not Path(record["vault_output_root"]).is_relative_to(vault_root_timestamp):
                        raise ValueError("Expected vault root must be under vault root timestamp.")
            else:
                for batch in payload[batch_name]:
                    for record in batch["candidate_records"]:
                        if record["family"] != EMB_34UM_CAUSAL_VALIDATION_FAMILY:
                            raise ValueError("All records must use the EMB family.")
                        if record["experiment"] != EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT:
                            raise ValueError("All records must use the indentation experiment.")
                        if not Path(record["output_root"]).is_relative_to(campaign_root):
                            raise ValueError("Expected output root must be under campaign root.")
                        if not Path(record["vault_output_root"]).is_relative_to(vault_root_timestamp):
                            raise ValueError("Expected vault root must be under vault root timestamp.")

    command_inventory: list[dict[str, Any]] = []
    for seed_payload in seeds_payloads:
        command_inventory.extend(
            _seed_batches_for_command_inventory(
                seed_payload=seed_payload,
                campaign_root=campaign_root,
                run_id_prefix=run_id_prefix,
                timestamp=timestamp,
                walltime=walltime,
                concurrent_jobs=concurrent_jobs,
                retry_limit=retry_limit,
                scratch_root=scratch_root,
                vault_root_timestamp=vault_root_timestamp,
            )
        )

    required_per_seed = (
        shared_size
        + step_count * step_size * 2
        + validation_size
    )
    seed_required_primary = len(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS)
    required_primary_curves = seed_required_primary * required_per_seed

    manifest: dict[str, Any] = {
        "schema_version": EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id_prefix": run_id_prefix,
        "campaign_root": str(campaign_root),
        "scratch_root": str(scratch_root),
        "vault_root_timestamp": str(vault_root_timestamp),
        "platform": "karolina",
        "execution_mode": EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE,
        "family": EMB_34UM_CAUSAL_VALIDATION_FAMILY,
        "experiment": EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT,
        "active_variables": list(EMB_34UM_CAUSAL_VALIDATION_ACTIVE_VARIABLES),
        "sampling": {
            "parameter_space": EMB_34UM_CAUSAL_VALIDATION_PARAMETER_SPACE,
            "bounds": {
                "ka": list(EMB_34UM_CAUSAL_VALIDATION_BOUNDS["ka"]),
                "kb": list(EMB_34UM_CAUSAL_VALIDATION_BOUNDS["kb"]),
            },
            "reduced_parameters_only": True,
        },
        "policy": {
            "required_seed_count": seed_required_primary,
            "required_primary_seed_count": seed_required_primary,
            "max_seed_count": EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
            "requested_seed_count": seed_count,
            "step_count": step_count,
            "step_size": step_size,
            "shared_size": shared_size,
            "validation_size": validation_size,
            "fresh_only": True,
            "adaptive_selection": {
                "enabled": True,
                "selection_required_per_step": step_size,
                "candidate_pool_size": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_POOL_SIZE,
                "acquisition_count": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_ACQUISITION_COUNT,
                "exploration_count": EMB_34UM_CAUSAL_VALIDATION_ADAPTIVE_SELECTION_EXPLORATION_COUNT,
            },
        },
        "statistical_gate": {
            "required_curves_per_seed": required_per_seed,
            "required_primary_seed_curves": required_primary_curves,
            "required_requested_seed_curves": seed_count * required_per_seed,
            "required_shared_initial": shared_size,
            "required_al_steps": step_count,
            "required_lhs_steps": step_count,
            "required_step_size": step_size,
            "required_validation": validation_size,
            "replacement_mode": "metadata_only",
        },
        "provenance": {
            "selection_mode": "causal_fresh_only",
            "fresh_only": True,
            "force_grid_signature": force_grid_sig,
            "force_grid_size": len(force_grid_values),
            "required_primary_seeds": list(EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS),
            "max_seed_count": EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS,
        },
        "command_inventory": {
            "entries": command_inventory,
            "count": len(command_inventory),
            "execution_mode": EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE,
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
        },
        "expected_output_roots": {
            "scratch": expected_scratch_roots,
            "vault": expected_vault_roots,
        },
        "static_preplanned": {
            "shared_initial": {
                "candidate_count": shared_size,
                "selection_mode": "precomputed_manifest_records",
            },
            "validation": {
                "candidate_count": validation_size,
                "selection_mode": "precomputed_manifest_records",
            },
            "lhs_steps": [
                {
                    "step": step,
                    "stage": f"lhs-step-{step:02d}",
                    "candidate_count": step_size,
                    "selection_mode": "precomputed_manifest_records",
                }
                for step in range(1, step_count + 1)
            ],
        },
        "adaptive_al_placeholders": {
            "selection_mode": "runtime_select_render",
            "al_steps": [
                {
                    "step": step,
                    "stage": f"al-step-{step:02d}",
                    "candidate_count": step_size,
                    "candidate_records": [],
                    "select_render_action": f"al-step-{step:02d}-select-render",
                }
                for step in range(1, step_count + 1)
            ],
        },
        "validation_plot": {
            "required": bool(include_coverage_plot_requirements),
            "coverage_plot": str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME),
            "coverage_plot_sidecar": str(campaign_root / EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME),
        },
        "seeds": seeds_payloads,
    }
    return manifest


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


__all__ = [
    "EMB_34UM_CAUSAL_VALIDATION_ACTIVE_VARIABLES",
    "EMB_34UM_CAUSAL_VALIDATION_BOUNDS",
    "EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_EXECUTION_MODE",
    "EMB_34UM_CAUSAL_VALIDATION_EXPERIMENT",
    "EMB_34UM_CAUSAL_VALIDATION_FAMILY",
    "EMB_34UM_CAUSAL_VALIDATION_MAX_SEEDS",
    "EMB_34UM_CAUSAL_VALIDATION_MANIFEST_FILENAME",
    "EMB_34UM_CAUSAL_VALIDATION_MIN_STEPS",
    "EMB_34UM_CAUSAL_VALIDATION_PARAMETER_SPACE",
    "EMB_34UM_CAUSAL_VALIDATION_PRIMARY_SEEDS",
    "EMB_34UM_CAUSAL_VALIDATION_REQUIRED_CURVES_PER_SEED",
    "EMB_34UM_CAUSAL_VALIDATION_SCHEMA_VERSION",
    "EMB_34UM_CAUSAL_VALIDATION_SHARED_SIZE",
    "EMB_34UM_CAUSAL_VALIDATION_STEP_SIZE",
    "EMB_34UM_CAUSAL_VALIDATION_VALIDATION_SIZE",
    "build_emb_34um_causal_validation_campaign_manifest",
]
