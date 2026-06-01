from __future__ import annotations

"""Build deterministic render/manifests for EMB 3.4um DNN causal validation campaigns."""

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.active_learning.contracts import Candidate, candidate_hash
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES,
    EMB_34UM_DNN_CAUSAL_A4_VALUE,
    EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
    EMB_34UM_DNN_CAUSAL_B1_VALUE,
    EMB_34UM_DNN_CAUSAL_B2_VALUE,
    EMB_34UM_DNN_CAUSAL_A3_VALUE,
    EMB_34UM_DNN_CAUSAL_BOUNDS,
    EMB_34UM_DNN_CAUSAL_EXPERIMENT,
    EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
    EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
    EMB_34UM_DNN_CAUSAL_FAMILY,
    EMB_34UM_DNN_CAUSAL_FORCE_COUNT,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID,
    EMB_34UM_DNN_CAUSAL_FORCE_MAX,
    EMB_34UM_DNN_CAUSAL_FORCE_MIN,
    EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY,
    EMB_34UM_DNN_CAUSAL_FRESH_ONLY,
    EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
    EMB_34UM_DNN_CAUSAL_OPERATIONAL_DOMAIN,
    EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
    EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
    EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION,
    EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
    EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
    EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
    EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
    EMB_34UM_DNN_CAUSAL_REPLICATES,
    EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    dnn_causal_total_dpd_curve_count,
    dnn_causal_total_dpd_curve_count_with_pilot,
    EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
    EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
    is_dnn_causal_low_corner_excluded,
    validate_dnn_causal_cycle_count,
    validate_dnn_causal_force_grid,
    validate_dnn_causal_replicate_count,
)
from meso_uq.active_learning.emb_34um_dnn_acquisition import (
    EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE,
    build_d4_candidate_pool,
)
from meso_uq.active_learning.emb_34um_dpd_adapter import (
    EMB_34UM_DPD_SCHEMA_VERSION,
    EMB_34UM_RUNTIME_FINGERPRINT,
)


EMB_34UM_DNN_CAUSAL_VALIDATION_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation.v1"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME = (
    "emb_34um_dnn_causal_validation_manifest.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME = (
    "emb_34um_dnn_causal_validation_command_inventory.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME = (
    "emb_34um_dnn_causal_validation_manifest_coverage.png"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME = (
    "emb_34um_dnn_causal_validation_manifest_coverage.png.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME = "emb_34um_batch_summary.json"
EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME = (
    "emb_34um_dnn_causal_validation_selection_manifest.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME = (
    "emb_34um_dnn_causal_validation_selection_batch_summary.json"
)
EMB_34UM_DNN_CAUSAL_VALIDATION_EXECUTION_MODE = "render-only"

_GLOBAL_SEED_OFFSET = 10_000
_GLOBAL_MODULUS = 1_000_000_007
_GLOBAL_MULT_X = 37_156_633
_GLOBAL_MULT_Y = 65_498_713
_GLOBAL_MULT_Z = 89_000_231
_GLOBAL_MULT_W = 73_001_103
_UNSEEN_TEST_START_INDEX = 900_000


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root for DNN causal design builder.")


def _coerce_nonempty_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string.")
    return text


def _coerce_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _coerce_walltime(value: str) -> str:
    text = _coerce_nonempty_text(value, label="walltime")
    if text.count(":") != 2:
        raise ValueError("walltime must be HH:MM:SS")
    return text


def _coerce_force_grid(values: Sequence[float] | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return validate_dnn_causal_force_grid(values)


def _coerce_replica_count(value: int) -> int:
    return validate_dnn_causal_replicate_count(value)


def _coerce_max_replicate_count(value: int) -> int:
    max_count = _coerce_positive_int(value, label="max_replicate_count")
    if max_count != EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT:
        raise ValueError(
            f"max_replicate_count is fixed at {EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT}, got {max_count}."
        )
    return max_count


def _coerce_cycle_count(value: int) -> int:
    return validate_dnn_causal_cycle_count(value)


def _coerce_step_size(value: int) -> int:
    step_size = _coerce_positive_int(value, label="step_size")
    if step_size != EMB_34UM_DNN_CAUSAL_STEP_SIZE:
        raise ValueError(f"step_size is fixed at {EMB_34UM_DNN_CAUSAL_STEP_SIZE}, got {step_size}.")
    return step_size


def _coerce_shared_initial_size(value: int) -> int:
    size = _coerce_positive_int(value, label="shared_initial_size")
    if size != EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE:
        raise ValueError(f"shared_initial_size is fixed at {EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE}, got {size}.")
    return size


def _coerce_unseen_test_size(value: int) -> int:
    size = _coerce_positive_int(value, label="unseen_test_size")
    if size != EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE:
        raise ValueError(f"unseen_test_size is fixed at {EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE}, got {size}.")
    return size


def _force_grid_signature(force_grid: Sequence[float]) -> str:
    return hashlib.sha256(json.dumps(tuple(float(v) for v in force_grid), sort_keys=True).encode("utf-8")).hexdigest()


def _selection_seed(replica: int, method: str, cycle: int | None) -> int:
    if cycle is None:
        cycle_value = 0
    else:
        cycle_value = int(cycle)

    if method == "shared":
        offset = 11_000
    elif method == "unseen":
        offset = 12_000
    elif method == "lhs":
        offset = 22_000
    elif method == "al":
        offset = 33_000
    elif method == "pilot":
        offset = 44_000
    else:
        raise ValueError(f"Unknown selection method {method!r}.")
    return _GLOBAL_SEED_OFFSET * int(replica) + offset + cycle_value


def _index_to_unit_point(index: int) -> tuple[float, float]:
    if index < 0:
        raise ValueError("index must be non-negative.")
    return (
        ((index * _GLOBAL_MULT_X) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
        ((index * _GLOBAL_MULT_Y) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
    )


def _index_to_unit_point_4d(index: int) -> tuple[float, float, float, float]:
    x, y = _index_to_unit_point(index)
    return (
        x,
        y,
        ((index * _GLOBAL_MULT_Z) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
        ((index * _GLOBAL_MULT_W) % _GLOBAL_MODULUS) / _GLOBAL_MODULUS,
    )


def _from_unit_to_log_space(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if lower <= 0.0 or upper <= 0.0 or lower >= upper:
        raise ValueError(f"Invalid bounds {bounds!r}.")
    unit = min(1.0, max(0.0, float(value)))
    if unit == 0.0:
        return float(lower)
    if unit == 1.0:
        return float(upper)
    return 10.0 ** (math.log10(lower) + unit * (math.log10(upper) - math.log10(lower)))


def _from_unit_to_linear_space(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if not (upper > lower):
        raise ValueError(f"Invalid bounds {bounds!r}.")
    unit = min(1.0, max(0.0, float(value)))
    if unit == 0.0:
        return float(lower)
    if unit == 1.0:
        return float(upper)
    return lower + unit * (upper - lower)


def _sample_points_with_source_indices(*, start_index: int, count: int) -> tuple[tuple[int, float, float], ...]:
    count = _coerce_positive_int(count, label="count")
    values: list[tuple[int, float, float]] = []
    source_index = start_index
    attempts = 0
    max_attempts = count * 10_000
    while len(values) < count and attempts < max_attempts:
        attempts += 1
        x, y = _index_to_unit_point(source_index)
        ka = _from_unit_to_log_space(x, EMB_34UM_DNN_CAUSAL_BOUNDS["ka"])
        kb = _from_unit_to_log_space(y, EMB_34UM_DNN_CAUSAL_BOUNDS["kb"])
        if not is_dnn_causal_low_corner_excluded(ka, kb):
            values.append((source_index, ka, kb))
        source_index += 1
    if len(values) < count:
        raise ValueError(
            f"Unable to sample {count} allowed EMB 3.4um points from start_index={start_index}; "
            f"generated {len(values)} before the attempt limit."
        )
    return tuple(values)


def _sample_points_with_source_indices_4d(*, start_index: int, count: int) -> tuple[tuple[int, float, float, float, float], ...]:
    count = _coerce_positive_int(count, label="count")
    values: list[tuple[int, float, float, float, float]] = []
    source_index = start_index
    attempts = 0
    max_attempts = count * 10_000
    while len(values) < count and attempts < max_attempts:
        attempts += 1
        x, y, u, v = _index_to_unit_point_4d(source_index)
        ka = _from_unit_to_log_space(x, EMB_34UM_DNN_CAUSAL_BOUNDS["ka"])
        kb = _from_unit_to_log_space(y, EMB_34UM_DNN_CAUSAL_BOUNDS["kb"])

        radp = _from_unit_to_linear_space(u, EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
        shell_th = _from_unit_to_linear_space(v, EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])
        if is_dnn_causal_low_corner_excluded(ka, kb, radp, shell_th):
            source_index += 1
            continue

        values.append((source_index, ka, kb, radp, shell_th))
        source_index += 1

    if len(values) < count:
        raise ValueError(
            f"Unable to sample {count} allowed EMB 3.4um points from start_index={start_index}; "
            f"generated {len(values)} before the attempt limit."
        )
    return tuple(values)


def _sample_points(*, start_index: int, count: int) -> tuple[tuple[float, float], ...]:
    return tuple((ka, kb) for _, ka, kb in _sample_points_with_source_indices(start_index=start_index, count=count))


def _batch_root(
    campaign_root: Path,
    *,
    replica: int | None,
    mode: str,
    cycle: int | None = None,
) -> Path:
    if mode == "unseen_test":
        return campaign_root / "unseen_test"
    if mode == "pilot":
        return campaign_root / "pilot"
    if replica is None:
        raise ValueError("Replica is required for non-shared stages.")
    if mode.startswith(("al-step-", "lhs-step-")):
        return campaign_root / f"replica-{replica:03d}" / mode
    if cycle is None:
        return campaign_root / f"replica-{replica:03d}" / mode
    return campaign_root / f"replica-{replica:03d}" / f"{mode}-{cycle:02d}"


def _candidate_records(
    *,
    campaign_root: Path,
    vault_root_timestamp: Path,
    replica: int,
    method: str,
    cycle: int | None,
    start_index: int,
    count: int,
    run_id_prefix: str,
    force_grid: tuple[float, ...],
    force_grid_signature: str,
    method_label: str,
    selection_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_label = f"{method}"
    if cycle is not None:
        stage_label = f"{method}-cycle-{cycle:02d}"

    batch_root = _batch_root(campaign_root, replica=replica, mode=method, cycle=cycle)
    seed_value = _selection_seed(replica=replica, method=method, cycle=cycle)
    points = _sample_points_with_source_indices_4d(start_index=start_index, count=count)

    records: list[dict[str, Any]] = []
    output_roots: list[str] = []
    vault_output_roots: list[str] = []

    for order, (source_index, ka, kb, radp, shell_th) in enumerate(points, start=1):
        candidate_id = (
            f"{run_id_prefix}-rep{replica:03d}-{method_label}-{order:03d}"
            if cycle is None
            else f"{run_id_prefix}-rep{replica:03d}-{method_label}-c{cycle:02d}-{order:03d}"
        )
        output_root = batch_root / "emb" / candidate_id
        vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)

        candidate = Candidate(
            candidate_id=candidate_id,
            parameters={
                "family": EMB_34UM_DNN_CAUSAL_FAMILY,
                "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
                EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[0]: float(ka),
                EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[1]: float(kb),
                EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[2]: float(radp),
                EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[3]: float(shell_th),
                "force_grid": list(force_grid),
            },
            metadata={
                "replica": int(replica),
                "cycle": int(cycle or 0),
                "stage": method,
                "source": "fresh_dpd",
                "selection_seed": seed_value,
            },
        )

        runtime_fingerprint = {
            **dict(EMB_34UM_RUNTIME_FINGERPRINT),
            "radp": float(radp),
            "shell_th": float(shell_th),
            "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
            "direct_stiffness_override": True,
        }

        record = {
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash(candidate),
            "family": EMB_34UM_DNN_CAUSAL_FAMILY,
            "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
            "ka": float(ka),
            "kb": float(kb),
            "radp": float(radp),
            "shell_th": float(shell_th),
            "force_grid": list(force_grid),
            "replica": int(replica),
            "cycle": int(cycle or 0),
            "method": method,
            "selection_seed": seed_value,
            "source_index": int(source_index),
            "selection_mode": "dnn_causal_fresh_only",
            "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            "selection_source": (
                EMB_34UM_DNN_CAUSAL_LHS_SOURCE
                if method == "lhs"
                else EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE
                if method == "unseen"
                else "fresh_dpd"
            ),
            "output_root": str(output_root),
            "vault_output_root": str(vault_output_root),
            "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
            "runtime_fingerprint": runtime_fingerprint,
            "request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
            "request_payload": {
                "schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
                "family": EMB_34UM_DNN_CAUSAL_FAMILY,
                "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
                "replica": int(replica),
                "cycle": int(cycle or 0),
                "method": method,
                "force_grid": list(force_grid),
                "force_grid_signature": force_grid_signature,
                "runtime_fingerprint": runtime_fingerprint,
                "parameters": {
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[0]: float(ka),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[1]: float(kb),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[2]: float(radp),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[3]: float(shell_th),
                    "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
                    "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
                    "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
                    "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
                    "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                },
            },
            "selection_mode_metadata": {
                "stage": stage_label,
                "method_label": method_label,
                "force_grid_signature": force_grid_signature,
                "source_index": int(source_index),
                "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
                "parameter_space": EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
                "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
            },
        }

        if selection_payload:
            record.update(dict(selection_payload))

        records.append(record)
        output_roots.append(str(output_root))
        vault_output_roots.append(str(vault_output_root))

    return {
        "method": method,
        "replica": int(replica),
        "cycle": int(cycle or 0),
        "selection_seed": seed_value,
        "candidate_count": count,
        "candidate_records": tuple(records),
        "selection_payload": dict(selection_payload or {}),
        "output_roots": tuple(output_roots),
        "vault_output_roots": tuple(vault_output_roots),
    }


def _pilot_candidate_records(
    *,
    campaign_root: Path,
    vault_root_timestamp: Path,
    run_id_prefix: str,
    force_grid: tuple[float, ...],
    force_grid_signature: str,
) -> dict[str, Any]:
    batch_root = _batch_root(campaign_root, replica=None, mode="pilot")
    seed_value = _selection_seed(replica=0, method="pilot", cycle=None)

    ka_kb_profiles: tuple[tuple[str, float, float], ...] = (
        ("edge_low_ka_high_kb", 0.0, 0.85),
        ("edge_high_ka_low_kb", 1.0, 0.15),
        ("edge_low_ka_max_kb", 0.0, 1.0),
        ("edge_high_ka_max_kb", 1.0, 1.0),
        ("interior_mid", 0.50, 0.50),
        ("interior_low_high", 0.25, 0.75),
        ("interior_high_low", 0.75, 0.25),
        ("interior_high", 0.75, 0.75),
    )
    geometry_profiles: tuple[tuple[str, float, float], ...] = (
        ("geometry_low", 0.0, 0.0),
        ("geometry_low_high", 0.0, 1.0),
        ("geometry_high_low", 1.0, 0.0),
        ("geometry_high", 1.0, 1.0),
    )

    records: list[dict[str, Any]] = []
    output_roots: list[str] = []
    vault_output_roots: list[str] = []

    for pair_index, (pair_label, ka_unit, kb_unit) in enumerate(ka_kb_profiles):
        ka = _from_unit_to_log_space(ka_unit, EMB_34UM_DNN_CAUSAL_BOUNDS["ka"])
        kb = _from_unit_to_log_space(kb_unit, EMB_34UM_DNN_CAUSAL_BOUNDS["kb"])

        for geometry_index, (geometry_label, radp_unit, shell_th_unit) in enumerate(geometry_profiles):
            radp = _from_unit_to_linear_space(radp_unit, EMB_34UM_DNN_CAUSAL_BOUNDS["radp"])
            shell_th = _from_unit_to_linear_space(shell_th_unit, EMB_34UM_DNN_CAUSAL_BOUNDS["shell_th"])
            if is_dnn_causal_low_corner_excluded(ka, kb, radp, shell_th):
                raise ValueError(
                    f"Pilot profile {pair_label!r}/{geometry_label!r} unexpectedly falls into the runtime-risk exclusion region."
                )
            order = pair_index * len(geometry_profiles) + geometry_index + 1
            candidate_id = f"{run_id_prefix}-pilot-{order:03d}"
            output_root = batch_root / "emb" / candidate_id
            vault_output_root = vault_root_timestamp / output_root.relative_to(campaign_root)
            pilot_profile = f"{pair_label}:{geometry_label}"

            candidate = Candidate(
                candidate_id=candidate_id,
                parameters={
                    "family": EMB_34UM_DNN_CAUSAL_FAMILY,
                    "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[0]: float(ka),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[1]: float(kb),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[2]: float(radp),
                    EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[3]: float(shell_th),
                    "force_grid": list(force_grid),
                },
                metadata={
                    "stage": "pilot",
                    "source": "fresh_dpd_pilot",
                    "selection_seed": seed_value,
                    "pilot_profile": pilot_profile,
                },
            )

            runtime_fingerprint = {
                **dict(EMB_34UM_RUNTIME_FINGERPRINT),
                "radp": float(radp),
                "shell_th": float(shell_th),
                "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                "direct_stiffness_override": True,
            }

            record = {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash(candidate),
                "family": EMB_34UM_DNN_CAUSAL_FAMILY,
                "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
                "ka": float(ka),
                "kb": float(kb),
                "radp": float(radp),
                "shell_th": float(shell_th),
                "force_grid": list(force_grid),
                "replica": 0,
                "cycle": 0,
                "method": "pilot",
                "selection_seed": seed_value,
                "source_index": order - 1,
                "selection_mode": "dnn_causal_fresh_only",
                "selection_source": "fresh_dpd_pilot",
                "selection_status": "rendered",
                "fresh_only": True,
                "stage": "pilot",
                "pilot_profile": pilot_profile,
                "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
                "output_root": str(output_root),
                "vault_output_root": str(vault_output_root),
                "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                "runtime_fingerprint": runtime_fingerprint,
                "request_payload_schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
                "request_payload": {
                    "schema_version": EMB_34UM_DPD_SCHEMA_VERSION,
                    "family": EMB_34UM_DNN_CAUSAL_FAMILY,
                    "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
                    "replica": 0,
                    "cycle": 0,
                    "method": "pilot",
                    "force_grid": list(force_grid),
                    "force_grid_signature": force_grid_signature,
                    "runtime_fingerprint": runtime_fingerprint,
                    "parameters": {
                        EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[0]: float(ka),
                        EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[1]: float(kb),
                        EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[2]: float(radp),
                        EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES[3]: float(shell_th),
                        "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
                        "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
                        "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
                        "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
                        "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
                    },
                },
                "selection_mode_metadata": {
                    "stage": "pilot",
                    "method_label": "fresh_dpd_pilot",
                    "selection_source": "fresh_dpd_pilot",
                    "pilot_profile": pilot_profile,
                    "source_index": order - 1,
                    "selection_seed": seed_value,
                    "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
                    "parameter_space": EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
                    "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
                },
            }

            records.append(record)
            output_roots.append(str(output_root))
            vault_output_roots.append(str(vault_output_root))

    return {
        "method": "pilot",
        "replica": 0,
        "cycle": 0,
        "selection_seed": seed_value,
        "candidate_count": len(records),
        "candidate_records": tuple(records),
        "selection_payload": {
            "selection_mode": "dnn_causal_fresh_only",
            "selection_source": "fresh_dpd_pilot",
            "fresh_only": True,
            "stage": "pilot",
        },
        "output_roots": tuple(output_roots),
        "vault_output_roots": tuple(vault_output_roots),
    }


def _al_placeholder_batch(
    *,
    replica: int,
    cycle: int,
    step_size: int,
    campaign_root: Path,
    vault_root_timestamp: Path,
) -> dict[str, Any]:
    batch_root = _batch_root(campaign_root, replica=replica, mode="al-step", cycle=cycle)
    vault_root = vault_root_timestamp / batch_root.relative_to(campaign_root)
    selection_seed = _selection_seed(replica=replica, method="al", cycle=cycle)
    _, candidate_pool_metadata = build_d4_candidate_pool(
        requested_size=EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE,
        seed=selection_seed,
        generator="auto",
    )

    return {
        "method": "al",
        "replica": int(replica),
        "cycle": int(cycle),
        "selection_seed": selection_seed,
        "candidate_count": int(step_size),
        "candidate_records": tuple(),
        "selection_payload": {
            "selection_required_count": int(step_size),
            "candidate_pool_size": EMB_34UM_DNN_D4_CANDIDATE_POOL_DEFAULT_REQUESTED_SIZE,
            "acquisition_count": 80,
            "exploration_count": 20,
            "candidate_pool_generation": {
                "candidate_pool_generator": "auto",
                **candidate_pool_metadata,
            },
            "selection_seed": selection_seed,
            "selection_status": "placeholder_runtime_selection_required",
            "selection_prerequisites": [
                str(campaign_root / f"replica-{replica:03d}" / "shared_initial" / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME)
            ],
            "selection_manifest_path": str(batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_MANIFEST_FILENAME),
            "selection_batch_summary_path": str(
                batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_SELECTION_BATCH_SUMMARY_FILENAME
            ),
            "expected_batch_summary_path": str(batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME),
            "select_render_action": f"al-step-{cycle:02d}-select-render",
            "output_root": str(batch_root),
            "selection_mode": "runtime_selection_required",
            "fresh_only": True,
            "stage": f"al-step-{cycle:02d}",
        },
        "output_roots": (str(batch_root),),
        "vault_output_roots": (str(vault_root),),
    }


def _build_seed_payload(
    *,
    replica: int,
    run_id_prefix: str,
    campaign_root: Path,
    vault_root_timestamp: Path,
    cycle_count: int,
    step_size: int,
    shared_initial_size: int,
    force_grid: tuple[float, ...],
    force_grid_signature: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    shared_start = (replica - 1) * 20_000
    shared_initial = _candidate_records(
        campaign_root=campaign_root,
        vault_root_timestamp=vault_root_timestamp,
        replica=replica,
        method="shared",
        cycle=None,
        start_index=shared_start,
        count=shared_initial_size,
        run_id_prefix=run_id_prefix,
        force_grid=force_grid,
        force_grid_signature=force_grid_signature,
        method_label="shared-initial",
        selection_payload={
            "selection_mode": "fresh_initial",
            "selection_source": "fresh_dpd",
            "fresh_only": True,
        },
    )

    al_steps: list[dict[str, Any]] = []
    lhs_steps: list[dict[str, Any]] = []

    lhs_start = shared_start + shared_initial_size
    for cycle in range(1, cycle_count + 1):
        al_steps.append(
            _al_placeholder_batch(
                replica=replica,
                cycle=cycle,
                step_size=step_size,
                campaign_root=campaign_root,
                vault_root_timestamp=vault_root_timestamp,
            )
        )
        lhs_steps.append(
            _candidate_records(
                campaign_root=campaign_root,
                vault_root_timestamp=vault_root_timestamp,
                replica=replica,
                method="lhs",
                cycle=cycle,
                start_index=lhs_start + (cycle - 1) * step_size,
                count=step_size,
                run_id_prefix=run_id_prefix,
                force_grid=force_grid,
                force_grid_signature=force_grid_signature,
                method_label=f"lhs-step{cycle:02d}",
                selection_payload={
                    "selection_mode": "fresh_dpd",
                    "selection_seed": _selection_seed(replica=replica, method="lhs", cycle=cycle),
                    "selection_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
                    "selection_status": "fresh_dpd",
                    "fresh_only": True,
                    "stage": f"lhs-step-{cycle:02d}",
                },
            )
        )

    return (
        {
            "replica": int(replica),
            "shared_initial": {
                **shared_initial,
                "stage": "shared_initial",
                "batch_root": str(campaign_root / f"replica-{replica:03d}" / "shared_initial"),
            },
            "al_steps": tuple(al_steps),
            "lhs_steps": tuple(lhs_steps),
            "replacement_queues": {
                "al": {
                    "enabled": True,
                    "mode": "metadata_only",
                    "max_replacements": 0,
                    "note": "DNN replacement is metadata-only in render manifest mode.",
                },
                "lhs": {
                    "enabled": True,
                    "mode": "metadata_only",
                    "max_replacements": 0,
                    "note": "LHS fresh batches are fully rendered by design; replacements are metadata-only.",
                },
            },
            "candidate_reserve": {
                "al": {"required": 0, "status": "empty", "metadata_only": True},
                "lhs": {"required": 0, "status": "empty", "metadata_only": True},
            },
        },
        al_steps,
        lhs_steps,
    )


def _build_command_payload(
    *,
    mode: str,
    replica: int,
    cycle: int | None,
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
    array = f"0-{candidate_count - 1}" if candidate_count > 0 else "0-0"
    array = f"{array}%{concurrent_jobs}"
    batch_root = _batch_root(campaign_root, replica=replica if replica else None, mode=mode, cycle=cycle)

    script_path = shlex.quote(
        str(
            _script_root()
            / "scripts"
            / "platforms"
            / "karolina"
            / "sbatch"
            / "emb_34um_dnn_causal_validation_array.sbatch"
        )
    )
    return {
        "mode": mode,
        "branch": "dnn_causal",
        "replica": int(replica),
        "cycle": int(cycle or 0),
        "candidate_count": int(candidate_count),
        "selection_mode": "dnn_causal_fresh_only",
        "batch_root": str(batch_root),
        "execution_mode": EMB_34UM_DNN_CAUSAL_VALIDATION_EXECUTION_MODE,
        "batch_summary_path": str(batch_root / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME),
        "array": array,
        "command": (
            "sbatch --parsable "
            f"--time={shlex.quote(walltime)} "
            f"--array={shlex.quote(array)} "
            "--export=ALL,"
            f"TIMESTAMP={shlex.quote(timestamp)},"
            f"SCRATCH_ROOT={shlex.quote(str(scratch_root))},"
            f"VAULT_ROOT={shlex.quote(str(vault_root_timestamp))},"
            f"RUN_ID_PREFIX={shlex.quote(run_id_prefix)},"
            f"CAMPAIGN_ROOT={shlex.quote(str(campaign_root))},"
            f"REPO_ROOT={shlex.quote(str(_script_root()))},"
            f"MODE={shlex.quote(mode)},"
            f"BATCH_DIR_OVERRIDE={shlex.quote(str(batch_root))},"
            f"REPLICA={int(replica)},"
            f"CYCLE={int(cycle or 0)},"
            f"EXECUTION_MODE={shlex.quote(EMB_34UM_DNN_CAUSAL_VALIDATION_EXECUTION_MODE)},"
            f"CONCURRENT_JOBS={int(concurrent_jobs)},"
            f"RETRY_LIMIT={int(retry_limit)},"
            f"PYTHON_EXECUTABLE={shlex.quote('python')} "
            f"{script_path}"
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
    replica = int(seed_payload["replica"])
    entries: list[dict[str, Any]] = []

    shared_initial = seed_payload["shared_initial"]
    entries.append(
        _build_command_payload(
            mode="shared_initial",
            replica=replica,
            cycle=None,
            candidate_count=len(shared_initial["candidate_records"]),
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
        entries.append(
            _build_command_payload(
                mode=f"al-step-{step:02d}",
                replica=replica,
                cycle=step,
                candidate_count=int(seed_payload["al_steps"][step - 1]["candidate_count"]),
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
        lhs = seed_payload["lhs_steps"][step - 1]
        entries.append(
            _build_command_payload(
                mode=f"lhs-step-{step:02d}",
                replica=replica,
                cycle=step,
                candidate_count=len(lhs["candidate_records"]),
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


def build_emb_34um_dnn_causal_validation_campaign_manifest(
    *,
    timestamp: str,
    run_id_prefix: str,
    campaign_root: Path,
    scratch_root: Path,
    vault_root_timestamp: Path,
    force_grid: Sequence[float],
    cycle_count: int = EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
    active_replicate_count: int = EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
    max_replicate_count: int = EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
    replica_count: int | None = None,
    shared_initial_size: int = EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
    unseen_test_size: int = EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
    step_size: int = EMB_34UM_DNN_CAUSAL_STEP_SIZE,
    walltime: str = EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
    concurrent_jobs: int = 30,
    retry_limit: int = EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
    include_coverage_plot_requirements: bool = True,
) -> dict[str, Any]:
    timestamp = _coerce_nonempty_text(timestamp, label="timestamp")
    run_id_prefix = _coerce_nonempty_text(run_id_prefix, label="run_id_prefix")
    campaign_root = Path(campaign_root)
    scratch_root = Path(scratch_root)
    vault_root_timestamp = Path(vault_root_timestamp)

    cycle_count = _coerce_cycle_count(cycle_count)
    if replica_count is not None:
        if active_replicate_count != EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT:
            raise ValueError("Provide either active_replicate_count or replica_count, not both.")
        active_replicate_count = int(replica_count)
    active_replicate_count = _coerce_replica_count(active_replicate_count)
    max_replicate_count = _coerce_max_replicate_count(max_replicate_count)
    if active_replicate_count > max_replicate_count:
        raise ValueError("active_replicate_count must not exceed max_replicate_count.")
    shared_initial_size = _coerce_shared_initial_size(shared_initial_size)
    unseen_test_size = _coerce_unseen_test_size(unseen_test_size)
    step_size = _coerce_step_size(step_size)
    walltime = _coerce_walltime(walltime)
    force_grid = _coerce_force_grid(force_grid)
    concurrent_jobs = _coerce_positive_int(concurrent_jobs, label="concurrent_jobs")
    retry_limit = _coerce_nonnegative_int(retry_limit, label="retry_limit")

    force_grid_signature = _force_grid_signature(force_grid)

    unseen_test = _candidate_records(
        campaign_root=campaign_root,
        vault_root_timestamp=vault_root_timestamp,
        replica=1,
        method="unseen",
        cycle=0,
        start_index=_UNSEEN_TEST_START_INDEX,
        count=unseen_test_size,
        run_id_prefix=run_id_prefix,
        force_grid=force_grid,
        force_grid_signature=force_grid_signature,
        method_label="unseen-test",
        selection_payload={
            "selection_mode": "fresh_test_set",
            "selection_source": EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
            "selection_seed": _selection_seed(replica=1, method="unseen", cycle=0),
            "fresh_only": True,
            "stage": "unseen_test",
            "batch_root": str(_batch_root(campaign_root, replica=1, mode="unseen_test")),
        },
    )
    pilot = _pilot_candidate_records(
        campaign_root=campaign_root,
        vault_root_timestamp=vault_root_timestamp,
        run_id_prefix=run_id_prefix,
        force_grid=force_grid,
        force_grid_signature=force_grid_signature,
    )

    seeds_payload: list[dict[str, Any]] = []
    adaptive_al_placeholders: list[dict[str, Any]] = []
    expected_scratch_roots: list[str] = []
    expected_vault_roots: list[str] = []

    for record in unseen_test["candidate_records"]:
        expected_scratch_roots.append(record["output_root"])
        expected_vault_roots.append(record["vault_output_root"])
    for record in pilot["candidate_records"]:
        expected_scratch_roots.append(record["output_root"])
        expected_vault_roots.append(record["vault_output_root"])

    for replica in EMB_34UM_DNN_CAUSAL_REPLICATES[:active_replicate_count]:
        seed_payload, al_steps, lhs_steps = _build_seed_payload(
            replica=replica,
            run_id_prefix=run_id_prefix,
            campaign_root=campaign_root,
            vault_root_timestamp=vault_root_timestamp,
            cycle_count=cycle_count,
            step_size=step_size,
            shared_initial_size=shared_initial_size,
            force_grid=force_grid,
            force_grid_signature=force_grid_signature,
        )

        for step_payload in al_steps:
            adaptive_al_placeholders.append(
                {
                    "replica": replica,
                    "cycle": int(step_payload["cycle"]),
                    "candidate_count": int(step_payload["candidate_count"]),
                    "selection_seed": int(step_payload["selection_seed"]),
                    "output_root": str(step_payload["output_roots"][0]),
                    "vault_output_root": str(step_payload["vault_output_roots"][0]),
                    "selection_status": "placeholder_runtime_selection_required",
                }
            )

        for stage in (seed_payload["shared_initial"], *lhs_steps):
            for output_root in stage.get("output_roots", ()):  # type: ignore[arg-type]
                expected_scratch_roots.append(str(output_root))
            for output_root in stage.get("vault_output_roots", ()):  # type: ignore[arg-type]
                expected_vault_roots.append(str(output_root))
        for step_payload in al_steps:
            for output_root in step_payload.get("output_roots", ()):  # type: ignore[arg-type]
                expected_scratch_roots.append(str(output_root))
            for output_root in step_payload.get("vault_output_roots", ()):  # type: ignore[arg-type]
                expected_vault_roots.append(str(output_root))

        seeds_payload.append(seed_payload)

    command_inventory_entries: list[dict[str, Any]] = []
    for seed_payload in seeds_payload:
        command_inventory_entries.extend(
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

    command_inventory_entries.append(
        _build_command_payload(
            mode="unseen_test",
            replica=0,
            cycle=0,
            candidate_count=len(unseen_test["candidate_records"]),
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
    command_inventory_entries.append(
        _build_command_payload(
            mode="pilot",
            replica=0,
            cycle=0,
            candidate_count=len(pilot["candidate_records"]),
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

    return {
        "schema_version": EMB_34UM_DNN_CAUSAL_VALIDATION_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id_prefix": run_id_prefix,
        "campaign_root": str(campaign_root),
        "scratch_root": str(scratch_root),
        "vault_root_timestamp": str(vault_root_timestamp),
        "policy": {
            "branch": "dnn_causal",
            "primary_replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
            "max_replicate_count": max_replicate_count,
            "active_replicate_count": active_replicate_count,
            "replicate_count": active_replicate_count,
            "replicates": list(range(1, active_replicate_count + 1)),
            "cycle_count": cycle_count,
            "min_cycles": EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
            "max_cycles": EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
            "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
            "bounds": {name: list(bounds) for name, bounds in EMB_34UM_DNN_CAUSAL_BOUNDS.items()},
            "parameter_space": EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
            "operational_domain": dict(EMB_34UM_DNN_CAUSAL_OPERATIONAL_DOMAIN),
            "fixed_coefficients": {
                "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
                "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
                "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
                "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
            },
            "fixed_controls": {
                "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
            },
            "pilot_size": EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
            "shared_initial_size": shared_initial_size,
            "unseen_test_size": unseen_test_size,
            "unseen_test_start_index": _UNSEEN_TEST_START_INDEX,
            "step_size": step_size,
            "force_grid": list(force_grid),
            "force_count": EMB_34UM_DNN_CAUSAL_FORCE_COUNT,
            "force_min": EMB_34UM_DNN_CAUSAL_FORCE_MIN,
            "force_max": EMB_34UM_DNN_CAUSAL_FORCE_MAX,
            "force_grid_signature": force_grid_signature,
            "force_grid_policy": EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY,
            "fresh_only": EMB_34UM_DNN_CAUSAL_FRESH_ONLY,
            "primary_statistic": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
            "primary_statistic_target": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
            "final_ci_upper_bound_threshold": EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
            "min_final_relative_improvement": EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
            "lhs_source": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
            "test_set_source": EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
            "uncertainty_source": EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "total_dpd_curve_count": dnn_causal_total_dpd_curve_count(
                cycle_count=cycle_count,
                active_replicate_count=active_replicate_count,
            ),
            "total_dpd_curve_count_with_pilot": dnn_causal_total_dpd_curve_count_with_pilot(
                cycle_count=cycle_count,
                active_replicate_count=active_replicate_count,
            ),
            "dpd_walltime_target": walltime,
            "dpd_walltime_target_default": EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
            "retry_limit_default": EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
            "family": EMB_34UM_DNN_CAUSAL_FAMILY,
            "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
            "selection_mode": "fresh_only",
            "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            "low_corner_exclusion": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
        },
        "runtime_fingerprint": dict(EMB_34UM_RUNTIME_FINGERPRINT),
        "unseen_test": {
            "stage": "unseen_test",
            "selection_seed": unseen_test["selection_seed"],
            "candidate_records": list(unseen_test["candidate_records"]),
            "batch_root": str(_batch_root(campaign_root, replica=1, mode="unseen_test")),
        },
        "pilot": {
            "stage": "pilot",
            "selection_seed": pilot["selection_seed"],
            "candidate_count": pilot["candidate_count"],
            "candidate_records": list(pilot["candidate_records"]),
            "selection_mode": "dnn_causal_fresh_only",
            "selection_source": "fresh_dpd_pilot",
            "fresh_only": True,
            "batch_root": str(_batch_root(campaign_root, replica=None, mode="pilot")),
            "batch_summary_path": str(campaign_root / "pilot" / EMB_34UM_DNN_CAUSAL_VALIDATION_BATCH_SUMMARY_FILENAME),
        },
        "seeds": seeds_payload,
        "adaptive_al_placeholders": {
            "al_steps": adaptive_al_placeholders,
            "selection_mode": "runtime_selection_required",
            "fresh_only": True,
        },
        "output_roots": expected_scratch_roots,
        "vault_roots": expected_vault_roots,
        "expected_output_roots": {
            "scratch": expected_scratch_roots,
            "vault": expected_vault_roots,
        },
        "provenance": {
            "branch": "dnn_causal",
            "ensemble_size": EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
            "dnn_selection": "fixed_architecture_dnn",
            "lhs_policy": EMB_34UM_DNN_CAUSAL_LHS_SOURCE,
            "unseen_test_start_index": _UNSEEN_TEST_START_INDEX,
            "fresh_only": EMB_34UM_DNN_CAUSAL_FRESH_ONLY,
            "force_grid_signature": force_grid_signature,
            "total_dpd_curve_count": dnn_causal_total_dpd_curve_count(
                cycle_count=cycle_count,
                active_replicate_count=active_replicate_count,
            ),
            "total_dpd_curve_count_with_pilot": dnn_causal_total_dpd_curve_count_with_pilot(
                cycle_count=cycle_count,
                active_replicate_count=active_replicate_count,
            ),
            "primary_replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
            "max_replicate_count": max_replicate_count,
            "active_replicate_count": active_replicate_count,
            "dpd_walltime_target": walltime,
            "dpd_walltime_target_default": EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
            "low_corner_exclusion": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
            "bounds": {name: list(bounds) for name, bounds in EMB_34UM_DNN_CAUSAL_BOUNDS.items()},
            "operational_domain": dict(EMB_34UM_DNN_CAUSAL_OPERATIONAL_DOMAIN),
            "fixed_coefficients": {
                "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
                "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
                "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
                "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
            },
            "fixed_controls": {
                "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
            },
            "pilot_size": EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
            "primary_statistic": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
            "primary_statistic_target": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
            "final_ci_upper_bound_threshold": EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
            "min_final_relative_improvement": EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
        },
        "command_inventory": {
            "count": len(command_inventory_entries),
            "walltime": walltime,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
            "retry_policy": "single_attempt_quarantine_then_replace" if retry_limit == 0 else "bounded_retry_then_replace",
            "entries": command_inventory_entries,
        },
        "scheduler_resources": {
            "platform": "karolina",
            "walltime": walltime,
            "dpd_walltime_target_default": EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
            "concurrent_jobs": concurrent_jobs,
            "retry_limit": retry_limit,
            "submit_script": str(
                _script_root()
                / "scripts"
                / "platforms"
                / "karolina"
                / "sbatch"
                / "emb_34um_dnn_causal_validation_array.sbatch"
            ),
        },
        "validation_plot": {
            "required": bool(include_coverage_plot_requirements),
            "mode": "log10(ka)-vs-log10(kb)",
            "force_grid_size": len(force_grid),
            "unseen_test_count": unseen_test_size,
            "shared_initial_per_replica": shared_initial_size,
            "lhs_per_cycle_per_replica": step_size,
            "primary_replicate_count": EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
            "max_replicate_count": max_replicate_count,
            "active_replicate_count": active_replicate_count,
            "replicate_count": active_replicate_count,
            "cycle_count": cycle_count,
            "coverage_plot_filename": EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_FILENAME,
            "coverage_plot_sidecar_filename": EMB_34UM_DNN_CAUSAL_VALIDATION_COVERAGE_PLOT_SIDECAR_FILENAME,
        },
    }
