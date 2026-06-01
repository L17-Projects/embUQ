from __future__ import annotations

"""Protocol contract for the EMB 3.4um DNN causal AL-vs-LHS protocol."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION = (
    "meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol.v1"
)
# Backward-compatible alias used by existing callers.
EMB_34UM_DNN_CAUSAL_VALIDATION_SCHEMA_VERSION = EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION

EMB_34UM_DNN_CAUSAL_EXPERIMENT = "indentation"
EMB_34UM_DNN_CAUSAL_FAMILY = "emb"
EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES = ("ka", "kb", "radp", "shell_th")
EMB_34UM_DNN_CAUSAL_BOUNDS = {
    "ka": (1e3, 1e5),
    "kb": (2e3, 2e4),
    "radp": (6.45, 6.60),
    "shell_th": (3.75e-9, 4.0e-9),
}
EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE = "log10"
EMB_34UM_DNN_CAUSAL_OPERATIONAL_DOMAIN = {
    "name": "emb_34um_d4_safe_box_failfast_v3",
    "status": "pilot_required_before_production",
    "purpose": "AL-vs-LHS causal comparison in an operationally stable D4 subdomain.",
    "basis": (
        "Narrowed on 2026-05-26 after Karolina pilot timeouts at thin-shell and "
        "extreme ka/kb/radp combinations, then narrowed again after a completed "
        "pilot at radp=6.70 and shell_th=3.5e-9 produced a non-finite final "
        "force-curve value, then expanded the low-ka/low-kb exclusion after "
        "a completed V2 pilot timed out at ka=3162.2776601683795 and "
        "kb=3556.558820077846. The causal comparison only requires a stable "
        "AL/LHS sampling domain, not the widest physical prior."
    ),
    "quarantine_policy": "failed_or_timed_out_candidates_are_replaced_by_next_successful_candidate",
    "metrics_policy": "final_metrics_use_successful_curves_and_successful_replacements_only",
}
EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION = {
    "name": "emb_34um_d4_runtime_risk_exclusion_v4",
    "enabled": True,
    "ka_upper_exclusive": 5000.0,
    "kb_upper_exclusive": 5000.0,
    "confirmed_timeout_region": {"ka_max": 3162.2776601683795, "kb_max": 3556.558820077846},
    "observed_slow_or_timeout_points": [
        {"ka": 189.530, "kb": 453.441, "date": "2026-05-22"},
        {"ka": 136.956, "kb": 736.554, "date": "2026-05-22"},
        {"ka": 3162.2776601683795, "kb": 3556.558820077846, "date": "2026-05-26"},
        {
            "ka": 1272.145949837055,
            "kb": 9011.372292672091,
            "radp": 6.489854540732141,
            "shell_th": 3.9170402720459855e-9,
            "date": "2026-05-27",
            "runtime_seconds": 1223,
            "slurm_state": "TIMEOUT",
        },
        {
            "ka": 22432.81994559314,
            "kb": 2436.648543412221,
            "radp": 6.584751551356739,
            "shell_th": 3.844815260086293e-9,
            "date": "2026-05-27",
            "runtime_seconds": 721,
            "slurm_state": "TIMEOUT",
        },
    ],
    "additional_4d_timeout_regions": [
        {
            "name": "random500_curve_error_c030_timeout_neighborhood",
            "ka_min_inclusive": 1200.0,
            "ka_max_exclusive": 1325.0,
            "kb_min_inclusive": 8500.0,
            "kb_max_exclusive": 9300.0,
            "radp_min_inclusive": 6.47,
            "radp_max_exclusive": 6.505,
            "shell_th_min_inclusive": 3.88e-9,
            "shell_th_max_exclusive": 3.94e-9,
            "confirmed_timeout_point": {
                "candidate_id": "emb-34um-acquisition-real-dpd-random500_curve_error-c030",
                "ka": 1272.145949837055,
                "kb": 9011.372292672091,
                "radp": 6.489854540732141,
                "shell_th": 3.9170402720459855e-9,
                "slurm_job": "4390713_29",
                "elapsed": "00:20:23",
            },
            "nearby_successful_replacement": {
                "candidate_id": "emb-34um-acquisition-real-dpd-random500_curve_error-c031",
                "ka": 1358.7312744799053,
                "kb": 9561.356860131997,
                "radp": 6.506801087377002,
                "shell_th": 3.95994097350211e-9,
                "slurm_job": "4390837_0",
                "elapsed": "00:02:52",
            },
        },
        {
            "name": "final_gate_unseen_c014_timeout_neighborhood",
            "ka_min_inclusive": 22000.0,
            "ka_max_exclusive": 23000.0,
            "kb_min_inclusive": 2300.0,
            "kb_max_exclusive": 2600.0,
            "radp_min_inclusive": 6.58,
            "radp_max_exclusive": 6.59,
            "shell_th_min_inclusive": 3.84e-9,
            "shell_th_max_exclusive": 3.85e-9,
            "confirmed_timeout_point": {
                "candidate_id": "emb-34um-final-al-vs-lhs-causal-gate-rep001-unseen-test-c00-014",
                "ka": 22432.81994559314,
                "kb": 2436.648543412221,
                "radp": 6.584751551356739,
                "shell_th": 3.844815260086293e-9,
                "slurm_job": "4391573_13",
                "elapsed": "00:12:01",
            },
            "nearby_successful_neighbors": [
                {
                    "candidate_id": "emb-34um-final-al-vs-lhs-causal-gate-rep001-unseen-test-c00-013",
                    "ka": 18904.735,
                    "kb": 2095.531,
                    "radp": 6.571402,
                    "shell_th": 3.82656498446e-9,
                },
                {
                    "candidate_id": "emb-34um-final-al-vs-lhs-causal-gate-rep001-unseen-test-c00-015",
                    "ka": 26619.331,
                    "kb": 2833.294,
                    "radp": 6.598102,
                    "shell_th": 3.86306553571e-9,
                },
            ],
        },
    ],
    "basis": (
        "Karolina EMB 3.4um indentation DPD production timeout diagnostics from 2026-05-21, "
        "2026-05-22, 2026-05-26, and 2026-05-27. The policy keeps the original low-ka/low-kb "
        "2D exclusion and adds narrow 4D timeout neighborhoods for the 2026-05-27 c030 "
        "and final-gate unseen c014 timeouts while preserving nearby successful points."
    ),
}

EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT = 3
EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT = 5
# Backward-compatible alias for callers that still import replicate_count.
EMB_34UM_DNN_CAUSAL_REPLICATE_COUNT = EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
EMB_34UM_DNN_CAUSAL_REPLICATES = tuple(
    range(1, EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT + 1)
)
EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE = 200
EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE = 300
EMB_34UM_DNN_CAUSAL_STEP_SIZE = 100
EMB_34UM_DNN_CAUSAL_AL_REPLACEMENT_RESERVE_SIZE = 50
EMB_34UM_DNN_CAUSAL_MIN_CYCLES = 5
EMB_34UM_DNN_CAUSAL_MAX_CYCLES = 5
EMB_34UM_DNN_CAUSAL_PILOT_SIZE = 32
EMB_34UM_DNN_CAUSAL_FRESH_ONLY = True
EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC = "AL_error_minus_LHS_error"
EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET = "relative_L2"
EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD = 0.0
EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT = 0.10

# Fixed pressure control used by EMB indentation campaign defaults.
EMB_34UM_DNN_CAUSAL_BPRESS_VALUE = -91.0

EMB_34UM_DNN_CAUSAL_FORCE_GRID = tuple(5000.0 * index / 7.0 for index in range(8))
EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY = "linspace_0_5000_8_including_endpoints"
EMB_34UM_DNN_CAUSAL_FORCE_COUNT = len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)
EMB_34UM_DNN_CAUSAL_FORCE_MIN = EMB_34UM_DNN_CAUSAL_FORCE_GRID[0]
EMB_34UM_DNN_CAUSAL_FORCE_MAX = EMB_34UM_DNN_CAUSAL_FORCE_GRID[-1]

EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET = "00:12:00"
EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT = EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET
EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT = 0

EMB_34UM_DNN_CAUSAL_LHS_SOURCE = "fresh_dpd"
EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE = "fresh_dpd_shared_unseen"

EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND = "fixed_architecture_dnn"
EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE = "same_architecture_dnn_ensemble"

# Geometry coefficients fixed for this protocol variant.
EMB_34UM_DNN_CAUSAL_B1_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_B2_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_A3_VALUE = 0.0
EMB_34UM_DNN_CAUSAL_A4_VALUE = 0.0

EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES = frozenset(
    {
        "samples_all",
        "samples_all.dat",
        "historical",
        "preexisting",
        "replay",
        "archive",
    }
)

EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS = frozenset(
    {
        "emb_34um_final_gate_surrogate",
        "lightweight",
        "numpy",
        "poly",
        "random-feature",
        "random_feature",
        "ridge",
    }
)

# Production DNN ensemble contract.
EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE = 10

# Timing canary contract:
#  - required before production by default.
#  - can be downgraded to 5 points only with explicit override reason.
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_REQUIRED = True
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT = 8
EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK = 5


def _coerce_positive_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return number


def _coerce_nonnegative_int(value: object, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _coerce_float(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _coerce_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean.")


def _coerce_sequence(values: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(values)


def _coerce_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _coerce_float_sequence(values: object, *, label: str) -> tuple[float, ...]:
    values = _coerce_sequence(values, label=label)
    parsed = tuple(_coerce_float(value, label=label) for value in values)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return parsed


def validate_dnn_causal_force_grid(values: Sequence[object]) -> tuple[float, ...]:
    """Validate the production 8-point force grid for the DNN causal gate."""

    normalized = _coerce_float_sequence(values, label="force_grid")
    if len(normalized) != len(EMB_34UM_DNN_CAUSAL_FORCE_GRID):
        raise ValueError(
            f"DNN causal force grid must contain {len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)} points, "
            f"got {len(normalized)}."
        )
    for actual, expected in zip(normalized, EMB_34UM_DNN_CAUSAL_FORCE_GRID):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "DNN causal force grid must be exactly 8 evenly spaced points from 0 to 5000 including endpoints."
            )
    return normalized


def validate_dnn_causal_cycle_count(cycle_count: object) -> int:
    cycles = _coerce_positive_int(cycle_count, label="cycle_count")
    if cycles < EMB_34UM_DNN_CAUSAL_MIN_CYCLES:
        raise ValueError(
            f"cycle_count must be at least {EMB_34UM_DNN_CAUSAL_MIN_CYCLES}."
        )
    if cycles > EMB_34UM_DNN_CAUSAL_MAX_CYCLES:
        raise ValueError(
            f"cycle_count must be at most {EMB_34UM_DNN_CAUSAL_MAX_CYCLES}."
        )
    return cycles


def validate_dnn_causal_replicate_count(replicate_count: object) -> int:
    value = _coerce_positive_int(replicate_count, label="replicate_count")
    if value > EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT:
        raise ValueError(
            f"replicate_count must be within [1, {EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT}]."
        )
    return value


def validate_dnn_causal_ensemble_size(ensemble_size: object) -> int:
    value = _coerce_positive_int(ensemble_size, label="ensemble_size")
    if value != EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE:
        raise ValueError(f"ensemble_size is fixed at {EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE}.")
    return value


def validate_dnn_causal_lhs_source(source: object) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        raise ValueError("lhs_source must be non-empty.")
    if normalized in EMB_34UM_DNN_CAUSAL_FORBIDDEN_FINAL_LHS_SOURCES:
        raise ValueError(
            "Final DNN causal LHS must use fresh DPD curves, not samples_all.dat, replay, archive, or historical data."
        )
    if normalized != EMB_34UM_DNN_CAUSAL_LHS_SOURCE:
        raise ValueError(f"lhs_source must be {EMB_34UM_DNN_CAUSAL_LHS_SOURCE!r}.")
    return normalized


def validate_dnn_causal_selector_backend(backend: object) -> str:
    normalized = str(backend or "").strip().lower()
    if not normalized:
        raise ValueError("selector_backend must be non-empty.")
    for token in EMB_34UM_DNN_CAUSAL_FORBIDDEN_SELECTOR_TOKENS:
        if token in normalized:
            raise ValueError(
                "DNN causal selection must not use the lightweight NumPy/ridge/random-feature selector."
            )
    if normalized != EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND:
        raise ValueError(
            f"selector_backend must be the fixed value {EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND!r}."
        )
    return normalized


def validate_dnn_causal_timing_canary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        raise ValueError("timing_canary is required.")
    required = _coerce_bool(payload.get("required", None), label="timing_canary.required")
    if not required:
        raise ValueError("timing_canary must be required before production.")

    point_count = _coerce_positive_int(
        payload.get("point_count", EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT),
        label="timing_canary.point_count",
    )
    if point_count == EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT:
        return {
            "required": True,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
        }
    if point_count == EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK:
        reason = _coerce_text(
            payload.get("override_reason", ""),
            label="timing_canary.override_reason",
        )
        return {
            "required": True,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK,
            "override_reason": reason,
        }

    raise ValueError(
        f"timing_canary.point_count must be {EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT} "
        f"or downgraded explicitly to {EMB_34UM_DNN_CAUSAL_TIMING_CANARY_DOWNTICK}."
    )


def is_dnn_causal_low_corner_excluded(
    ka: object,
    kb: object,
    radp: object | None = None,
    shell_th: object | None = None,
) -> bool:
    """Return True for EMB 3.4um indentation runtime-risk points."""

    if not EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION["enabled"]:
        return False
    ka_value = _coerce_float(ka, label="ka")
    kb_value = _coerce_float(kb, label="kb")
    if (
        ka_value < float(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION["ka_upper_exclusive"])
        and kb_value < float(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION["kb_upper_exclusive"])
    ):
        return True

    if radp is None or shell_th is None:
        return False

    radp_value = _coerce_float(radp, label="radp")
    shell_th_value = _coerce_float(shell_th, label="shell_th")
    for region in EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION.get("additional_4d_timeout_regions", ()):
        if (
            float(region["ka_min_inclusive"]) <= ka_value < float(region["ka_max_exclusive"])
            and float(region["kb_min_inclusive"]) <= kb_value < float(region["kb_max_exclusive"])
            and float(region["radp_min_inclusive"]) <= radp_value < float(region["radp_max_exclusive"])
            and float(region["shell_th_min_inclusive"]) <= shell_th_value < float(region["shell_th_max_exclusive"])
        ):
            return True
    return False


def validate_dnn_causal_walltime_target(value: object) -> str:
    walltime = _coerce_text(value, label="dpd_walltime_target")
    if walltime.count(":") != 2:
        raise ValueError("dpd_walltime_target must be HH:MM:SS.")
    return walltime


def dnn_causal_training_curve_count(*, cycle_count: int) -> int:
    cycles = validate_dnn_causal_cycle_count(cycle_count)
    return dnn_causal_branch_curve_count(cycle_count=cycles)


def dnn_causal_curves_per_replicate(*, cycle_count: int) -> int:
    cycles = validate_dnn_causal_cycle_count(cycle_count)
    return EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE + 2 * cycles * EMB_34UM_DNN_CAUSAL_STEP_SIZE


def dnn_causal_branch_curve_count(*, cycle_count: int) -> int:
    return dnn_causal_curves_per_replicate(cycle_count=cycle_count)


def dnn_causal_total_dpd_curve_count(
    *,
    cycle_count: int,
    active_replicate_count: int | None = None,
    replicate_count: int | None = None,
) -> int:
    cycles = validate_dnn_causal_cycle_count(cycle_count)
    if active_replicate_count is None:
        active_replicates = (
            validate_dnn_causal_replicate_count(replicate_count)
            if replicate_count is not None
            else EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
        )
    else:
        active_replicates = validate_dnn_causal_replicate_count(active_replicate_count)
        if replicate_count is not None:
            compatibility_count = validate_dnn_causal_replicate_count(replicate_count)
            if compatibility_count != active_replicates:
                raise ValueError("active_replicate_count and replicate_count must match when both are provided.")
    per_replicate = dnn_causal_curves_per_replicate(cycle_count=cycles)
    return EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE + active_replicates * per_replicate


def dnn_causal_total_dpd_curve_count_with_pilot(
    *,
    cycle_count: int,
    active_replicate_count: int | None = None,
    replicate_count: int | None = None,
) -> int:
    return (
        dnn_causal_total_dpd_curve_count(
            cycle_count=cycle_count,
            active_replicate_count=active_replicate_count,
            replicate_count=replicate_count,
        )
        + EMB_34UM_DNN_CAUSAL_PILOT_SIZE
    )


def dnn_causal_training_row_count(*, curve_count: int) -> int:
    curves = _coerce_positive_int(curve_count, label="curve_count")
    return curves * len(EMB_34UM_DNN_CAUSAL_FORCE_GRID)


def validate_dnn_causal_protocol_payload(payload: Mapping[str, Any]) -> "Emb34umDnnCausalValidationProtocol":
    if not isinstance(payload, Mapping):
        raise ValueError("protocol payload must be a mapping.")

    if "ensemble_size" not in payload:
        raise ValueError("protocol payload must include ensemble_size.")
    if "timing_canary" not in payload:
        raise ValueError("protocol payload must include timing_canary.")

    return Emb34umDnnCausalValidationProtocol(
        cycle_count=payload.get("cycle_count", EMB_34UM_DNN_CAUSAL_MIN_CYCLES),
        primary_replicate_count=payload.get(
            "primary_replicate_count",
            EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT,
        ),
        max_replicate_count=payload.get(
            "max_replicate_count",
            EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT,
        ),
        active_replicate_count=payload.get(
            "active_replicate_count",
            payload.get("replicate_count", EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT),
        ),
        force_grid=payload.get("force_grid", EMB_34UM_DNN_CAUSAL_FORCE_GRID),
        lhs_source=payload.get("lhs_source", EMB_34UM_DNN_CAUSAL_LHS_SOURCE),
        selector_backend=payload.get("selector_backend", EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND),
        uncertainty_source=payload.get(
            "uncertainty_source",
            EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE,
        ),
        ensemble_size=payload["ensemble_size"],
        timing_canary=payload["timing_canary"],
        dpd_walltime_target=payload.get(
            "dpd_walltime_target",
            EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
        ),
    )


@dataclass(frozen=True)
class Emb34umDnnCausalValidationProtocol:
    cycle_count: int = EMB_34UM_DNN_CAUSAL_MIN_CYCLES
    primary_replicate_count: int = EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
    max_replicate_count: int = EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT
    active_replicate_count: int = EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT
    force_grid: tuple[float, ...] = EMB_34UM_DNN_CAUSAL_FORCE_GRID
    lhs_source: str = EMB_34UM_DNN_CAUSAL_LHS_SOURCE
    selector_backend: str = EMB_34UM_DNN_CAUSAL_SELECTOR_BACKEND
    uncertainty_source: str = EMB_34UM_DNN_CAUSAL_RECOMMENDED_UNCERTAINTY_SOURCE
    ensemble_size: int = EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE
    dpd_walltime_target: str = EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT
    timing_canary: Mapping[str, Any] = field(
        default_factory=lambda: {
            "required": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_REQUIRED,
            "point_count": EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_count", validate_dnn_causal_cycle_count(self.cycle_count))
        object.__setattr__(
            self,
            "primary_replicate_count",
            _coerce_positive_int(
                self.primary_replicate_count,
                label="primary_replicate_count",
            ),
        )
        object.__setattr__(
            self,
            "max_replicate_count",
            _coerce_positive_int(
                self.max_replicate_count,
                label="max_replicate_count",
            ),
        )
        if self.primary_replicate_count != EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT:
            raise ValueError(
                f"primary_replicate_count is fixed at {EMB_34UM_DNN_CAUSAL_PRIMARY_REPLICATE_COUNT}."
            )
        if self.max_replicate_count != EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT:
            raise ValueError(
                f"max_replicate_count is fixed at {EMB_34UM_DNN_CAUSAL_MAX_REPLICATE_COUNT}."
            )
        object.__setattr__(
            self,
            "active_replicate_count",
            validate_dnn_causal_replicate_count(self.active_replicate_count),
        )
        if self.active_replicate_count > self.max_replicate_count:
            raise ValueError("active_replicate_count must not exceed max_replicate_count.")
        object.__setattr__(
            self,
            "force_grid",
            validate_dnn_causal_force_grid(self.force_grid),
        )
        object.__setattr__(self, "lhs_source", validate_dnn_causal_lhs_source(self.lhs_source))
        object.__setattr__(
            self,
            "selector_backend",
            validate_dnn_causal_selector_backend(self.selector_backend),
        )
        object.__setattr__(
            self,
            "uncertainty_source",
            _coerce_text(self.uncertainty_source, label="uncertainty_source"),
        )
        object.__setattr__(
            self,
            "ensemble_size",
            validate_dnn_causal_ensemble_size(self.ensemble_size),
        )
        object.__setattr__(
            self,
            "timing_canary",
            validate_dnn_causal_timing_canary(dict(self.timing_canary)),
        )
        object.__setattr__(
            self,
            "dpd_walltime_target",
            validate_dnn_causal_walltime_target(self.dpd_walltime_target),
        )

    @property
    def rows_per_curve(self) -> int:
        return len(self.force_grid)

    @property
    def total_dpd_curve_count(self) -> int:
        return dnn_causal_total_dpd_curve_count(
            cycle_count=self.cycle_count,
            active_replicate_count=self.active_replicate_count,
        )

    @property
    def replicate_count(self) -> int:
        """Backward-compatible alias for active_replicate_count."""
        return self.active_replicate_count

    @property
    def fixed_coefficients(self) -> dict[str, float]:
        return {
            "b1": EMB_34UM_DNN_CAUSAL_B1_VALUE,
            "b2": EMB_34UM_DNN_CAUSAL_B2_VALUE,
            "a3": EMB_34UM_DNN_CAUSAL_A3_VALUE,
            "a4": EMB_34UM_DNN_CAUSAL_A4_VALUE,
        }

    @property
    def fixed_controls(self) -> dict[str, float]:
        return {
            "bpress": EMB_34UM_DNN_CAUSAL_BPRESS_VALUE,
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EMB_34UM_DNN_CAUSAL_SCHEMA_VERSION,
            "family": EMB_34UM_DNN_CAUSAL_FAMILY,
            "experiment": EMB_34UM_DNN_CAUSAL_EXPERIMENT,
            "active_variables": list(EMB_34UM_DNN_CAUSAL_ACTIVE_VARIABLES),
            "bounds": {name: list(bounds) for name, bounds in EMB_34UM_DNN_CAUSAL_BOUNDS.items()},
            "parameter_space": EMB_34UM_DNN_CAUSAL_PARAMETER_SPACE,
            "operational_domain": dict(EMB_34UM_DNN_CAUSAL_OPERATIONAL_DOMAIN),
            "exclusion_policy": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            "low_corner_exclusion": dict(EMB_34UM_DNN_CAUSAL_LOW_CORNER_EXCLUSION),
            "fixed_coefficients": self.fixed_coefficients,
            "fixed_controls": self.fixed_controls,
            "primary_replicate_count": self.primary_replicate_count,
            "max_replicate_count": self.max_replicate_count,
            "active_replicate_count": self.active_replicate_count,
            "replicate_count": self.active_replicate_count,
            "replicates": list(range(1, self.active_replicate_count + 1)),
            "shared_initial_size": EMB_34UM_DNN_CAUSAL_SHARED_INITIAL_SIZE,
            "unseen_test_size": EMB_34UM_DNN_CAUSAL_UNSEEN_TEST_SIZE,
            "step_size": EMB_34UM_DNN_CAUSAL_STEP_SIZE,
            "min_cycles": EMB_34UM_DNN_CAUSAL_MIN_CYCLES,
            "max_cycles": EMB_34UM_DNN_CAUSAL_MAX_CYCLES,
            "pilot_size": EMB_34UM_DNN_CAUSAL_PILOT_SIZE,
            "fresh_only": EMB_34UM_DNN_CAUSAL_FRESH_ONLY,
            "primary_statistic": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC,
            "primary_statistic_target": EMB_34UM_DNN_CAUSAL_PRIMARY_STATISTIC_TARGET,
            "final_ci_upper_bound_threshold": EMB_34UM_DNN_CAUSAL_FINAL_CI_UPPER_BOUND_THRESHOLD,
            "min_final_relative_improvement": EMB_34UM_DNN_CAUSAL_MIN_FINAL_RELATIVE_IMPROVEMENT,
            "cycle_count": self.cycle_count,
            "force_grid": list(self.force_grid),
            "force_count": EMB_34UM_DNN_CAUSAL_FORCE_COUNT,
            "force_min": EMB_34UM_DNN_CAUSAL_FORCE_MIN,
            "force_max": EMB_34UM_DNN_CAUSAL_FORCE_MAX,
            "force_grid_policy": EMB_34UM_DNN_CAUSAL_FORCE_GRID_POLICY,
            "lhs_source": self.lhs_source,
            "test_set_source": EMB_34UM_DNN_CAUSAL_TEST_SET_SOURCE,
            "selector_backend": self.selector_backend,
            "uncertainty_source": self.uncertainty_source,
            "ensemble_size": self.ensemble_size,
            "dpd_walltime_target": self.dpd_walltime_target,
            "dpd_walltime_target_default": EMB_34UM_DNN_CAUSAL_DPD_WALLTIME_TARGET_DEFAULT,
            "retry_limit_default": EMB_34UM_DNN_CAUSAL_RETRY_LIMIT_DEFAULT,
            "timing_canary": dict(self.timing_canary),
            "rows_per_curve": self.rows_per_curve,
            "branch_training_curve_count_at_final_cycle": dnn_causal_branch_curve_count(
                cycle_count=self.cycle_count,
            ),
            "total_dpd_curve_count": self.total_dpd_curve_count,
        }


def validate_dnn_causal_protocol_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = validate_dnn_causal_protocol_payload(payload)
    return protocol.as_manifest()
